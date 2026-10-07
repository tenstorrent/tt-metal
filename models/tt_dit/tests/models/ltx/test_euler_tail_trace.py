# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Opt-in native Euler trace contract, without a model/checkpoint.

C10_RESULTS=<fresh.pt> pytest '<this-file>::test_euler_tail_trace[galaxy]' -s
Use the broker and normal build/native provenance checks. Small SP-sharded inputs
exercise all 10 shipped steps, FP32/BF16 producer outputs, A/B/A, padding, and
coexisting producer/tail traces. This is not full-model quality or speed evidence.
"""

import ast
import hashlib
import os
import subprocess
import time
from pathlib import Path

import pytest
import torch


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _bits(value):
    assert value.dtype == torch.bfloat16
    return value.contiguous().view(torch.int16)


def _schedules():
    tree = ast.parse(Path("models/tt_dit/pipelines/ltx/pipeline_ltx_distilled.py").read_text())
    return {
        name: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
        for name in (target.id,)
        if name in ("_DEFAULT_S1_SIGMAS", "_DEFAULT_S2_SIGMAS")
    }


def pytest_generate_tests(metafunc):
    from models.tt_dit.utils.test import line_params_req_exact_devices, ring_params_8k_req_exact_devices

    common = {"l1_small_size": 8192, "trace_region_size": 100_000_000}
    metafunc.parametrize(
        "mesh_device,device_params",
        [
            pytest.param((4, 8), {**ring_params_8k_req_exact_devices, **common}, id="galaxy"),
            pytest.param((2, 4), {**line_params_req_exact_devices, **common}, id="f07"),
        ],
        indirect=["mesh_device", "device_params"],
    )


@pytest.mark.skip_post_commit
@pytest.mark.skipif(not os.environ.get("C10_RESULTS"), reason="opt-in Euler trace contract")
def test_euler_tail_trace(mesh_device, device_params):
    import ttnn
    from models.tt_dit.utils.ltx_euler import EulerTail, euler_tail
    from models.tt_dit.utils.tensor import typed_tensor
    from models.tt_dit.utils.tracing import Tracer, set_kernel_prewarm_capturing

    result = Path(os.environ["C10_RESULTS"])
    assert not result.exists(), "use a fresh result path"
    assert os.environ.get("TT_METAL_DEVICE_PROFILER", "0") in ("", "0")
    capture_only = bool(os.environ.get("TT_METAL_KERNEL_CAPTURE_ONLY"))
    mesh_shape = tuple(mesh_device.shape)
    sp = mesh_shape[1]
    stages, rows = [], []
    sources = [
        __file__,
        "models/tt_dit/utils/ltx_euler.py",
        "models/tt_dit/utils/tracing.py",
        "models/tt_dit/pipelines/ltx/pipeline_ltx.py",
        "models/tt_dit/pipelines/ltx/pipeline_ltx_distilled.py",
    ]
    record = dict(
        schema=1,
        status="INCOMPLETE",
        mesh=list(mesh_shape),
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_sha256={p: _sha(p) for p in sources},
        native_sha256=_sha("ttnn/ttnn/_ttnn.so"),
        cgroup=Path("/proc/self/cgroup").read_text(),
        rows=rows,
        timing_scope="five within-process tail-only observations; uploads/producer/readback excluded; no speed or AV acceptance",
    )

    def save():
        if not capture_only:
            result.parent.mkdir(parents=True, exist_ok=True)
            torch.save(record, result)

    def tensor(value, dtype):
        return typed_tensor(value, dtype, device=mesh_device, mesh_axis=1, shard_dim=2)

    def upload(value, device):
        host = typed_tensor(value, device.dtype, device=mesh_device, mesh_axis=1, shard_dim=2, on_host=True)
        ttnn.copy_host_to_device_tensor(host, device)

    def compare(stage, label):
        row = dict(stage=stage["name"], label=label, replicas={})
        good = True
        for axis in ("video", "audio"):
            baseline, candidate = stage["baseline"][axis], stage["candidate"][axis]
            coords = [tuple(int(c) for c in coord) for coord in candidate.tensor_topology().mesh_coords()]
            assert coords == [tuple(int(c) for c in coord) for coord in baseline.tensor_topology().mesh_coords()]
            assert len(coords) == len(set(coords)) == mesh_shape[0] * mesh_shape[1]
            row["replicas"][axis] = []
            for coord, left, right in zip(
                coords, ttnn.get_device_tensors(baseline), ttnn.get_device_tensors(candidate), strict=True
            ):
                expected, actual = ttnn.to_torch(left).contiguous(), ttnn.to_torch(right).contiguous()
                n = actual.shape[2]
                pad_start = max(0, min(n, stage["real"][axis] - coord[1] * n))
                exact = torch.equal(_bits(expected), _bits(actual))
                finite = bool(torch.isfinite(actual).all())
                pad_zero = bool((actual[:, :, pad_start:, :] == 0).all())
                item = dict(
                    coord=coord,
                    shape=tuple(actual.shape),
                    exact=exact,
                    finite=finite,
                    pad_zero=pad_zero,
                    sha256=hashlib.sha256(_bits(actual).numpy().tobytes()).hexdigest(),
                )
                if not (exact and finite and pad_zero):
                    item.update(baseline_values=expected, actual_values=actual)
                    good = False
                row["replicas"][axis].append(item)
        rows.append(row)
        save()
        assert good, f"Euler native mismatch: {stage['name']}/{label}; raw failure saved"
        return row

    def update(stage, index, seed):
        torch.manual_seed(seed + index)
        for axis in ("video", "audio"):
            value = torch.randn(stage["initial"][axis].shape).to(stage["dtype"])
            upload(value, stage["velocity"][axis])
            upload(value, stage["baseline_velocity"][axis])
        return stage["producer"](stage["velocity"]["video"], stage["velocity"]["audio"])

    def advance(stage, velocity, dt):
        # Independent baseline velocity buffers also predate all captures.
        baseline_velocity = tuple(stage["baseline_velocity"][axis] for axis in ("video", "audio"))
        args = (
            stage["candidate"]["video"],
            stage["candidate"]["audio"],
            *velocity,
            stage["mask"]["video"],
            stage["mask"]["audio"],
        )
        stage["tail"](*args, dt)
        euler_tail(
            stage["baseline"]["video"],
            stage["baseline"]["audio"],
            *baseline_velocity,
            stage["mask"]["video"],
            stage["mask"]["audio"],
            dt,
        )

    try:
        # Every input/mask/control latent for both stages predates ALL captures.
        for dtype, native_dtype in ((torch.float32, ttnn.float32), (torch.bfloat16, ttnn.bfloat16)):
            for index, (name, schedule) in enumerate(_schedules().items()):
                shapes = dict(video=(1, 1, (32 if index == 0 else 64) * sp, 128), audio=(1, 1, 32 * sp, 128))
                stage = dict(
                    name=f"{name}/{dtype}",
                    dtype=dtype,
                    real=dict(video=shapes["video"][2] - 17, audio=shapes["audio"][2] - 3),
                    baseline={},
                    candidate={},
                    velocity={},
                    baseline_velocity={},
                    mask={},
                    initial={},
                    schedule=torch.tensor(schedule, dtype=torch.float32).tolist(),
                    tail=EulerTail(mesh_device),
                    producer=Tracer(
                        lambda v, a: (ttnn.clone(v), ttnn.clone(a)),
                        device=mesh_device,
                        prep_run=True,
                        clone_prep_inputs=False,
                    ),
                )
                for axis, shape in shapes.items():
                    torch.manual_seed(10 + index)
                    stage["initial"][axis] = torch.randn(shape).bfloat16()
                    stage["baseline"][axis] = tensor(stage["initial"][axis], ttnn.bfloat16)
                    stage["candidate"][axis] = tensor(stage["initial"][axis], ttnn.bfloat16)
                    stage["velocity"][axis] = tensor(torch.zeros(shape), native_dtype)
                    stage["baseline_velocity"][axis] = tensor(torch.zeros(shape), native_dtype)
                    mask = torch.ones(shape[:-1] + (1,)).bfloat16()
                    mask[:, :, stage["real"][axis] :, :] = 0
                    stage["mask"][axis] = tensor(mask, ttnn.bfloat16)
                stages.append(stage)
        set_kernel_prewarm_capturing(capture_only)
        for stage in stages:
            anchor = None
            for visit, seed in enumerate((31, 59, 31)):
                for axis in ("video", "audio"):
                    upload(stage["initial"][axis], stage["baseline"][axis])
                    upload(stage["initial"][axis], stage["candidate"][axis])
                sigmas = stage["schedule"]
                for index, (sigma, next_sigma) in enumerate(zip(sigmas[:-1], sigmas[1:])):
                    velocity = update(stage, index, seed)
                    if capture_only:
                        euler_tail(
                            stage["candidate"]["video"],
                            stage["candidate"]["audio"],
                            *velocity,
                            stage["mask"]["video"],
                            stage["mask"]["audio"],
                            next_sigma - sigma,
                        )
                        continue
                    assert stage["producer"].trace_captured
                    advance(stage, velocity, next_sigma - sigma)
                    row = compare(stage, f"visit{visit}/step{index}")
                if not capture_only:
                    hashes = {axis: [r["sha256"] for r in row["replicas"][axis]] for axis in ("video", "audio")}
                    if visit == 0:
                        anchor = hashes
                        stage["anchor"] = hashes
                    elif visit == 1:
                        assert hashes != anchor, "changed velocity produced identical state"
                    else:
                        assert hashes == anchor, "restored A drifted"
            if not capture_only:
                timings = []
                for index in range(5):
                    velocity = update(stage, 0, 31)
                    baseline_velocity = tuple(stage["baseline_velocity"][axis] for axis in ("video", "audio"))
                    ttnn.synchronize_device(mesh_device)
                    start = time.perf_counter()
                    stage["tail"](
                        stage["candidate"]["video"],
                        stage["candidate"]["audio"],
                        *velocity,
                        stage["mask"]["video"],
                        stage["mask"]["audio"],
                        sigmas[1] - sigmas[0],
                    )
                    ttnn.synchronize_device(mesh_device)
                    timings.append((time.perf_counter() - start) * 1000)
                    euler_tail(
                        stage["baseline"]["video"],
                        stage["baseline"]["audio"],
                        *baseline_velocity,
                        stage["mask"]["video"],
                        stage["mask"]["audio"],
                        sigmas[1] - sigmas[0],
                    )
                    compare(stage, f"timed{index}")
                record.setdefault("tail_ms", {})[stage["name"]] = timings
        if capture_only:
            pytest.skip("kernel recipes only; no native evidence")
        # Revisit both stages after every producer/tail family is live.
        for stage in stages:
            for axis in ("video", "audio"):
                upload(stage["initial"][axis], stage["baseline"][axis])
                upload(stage["initial"][axis], stage["candidate"][axis])
            sigmas = stage["schedule"]
            for index, (sigma, next_sigma) in enumerate(zip(sigmas[:-1], sigmas[1:])):
                advance(stage, update(stage, index, 31), next_sigma - sigma)
                row = compare(stage, f"coexisting/step{index}")
            hashes = {axis: [r["sha256"] for r in row["replicas"][axis]] for axis in ("video", "audio")}
            assert hashes == stage["anchor"], "coexisting trace families changed restored A"
        record["status"] = "C10_NATIVE_CONTRACT_PASS"
    except BaseException as error:
        record["failure"] = repr(error)
        raise
    finally:
        try:
            save()
        finally:
            set_kernel_prewarm_capturing(False)
            for stage in stages:
                stage["tail"].release()
            for stage in stages:
                stage["producer"].release_trace()
