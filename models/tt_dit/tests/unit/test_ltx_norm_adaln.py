# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""C02 scalar AdaLN fusion: exact production helper, changed-input traces, timing.

Prepare CPU fixtures and verify saved results outside the device reservation:
  python -m models.tt_dit.tests.unit.test_ltx_norm_adaln --prepare /tmp/c02/fixtures
  # Under the broker, one shape/route per process:
  C02_SHAPE=tp4_v_block_s1 C02_MODE=base C02_FIXTURES=/tmp/c02/fixtures \
    C02_RESULTS=/tmp/c02/results pytest <this-file> -s
  # Repeat with C02_MODE=fused, then outside the reservation:
  python -m models.tt_dit.tests.unit.test_ltx_norm_adaln --verify /tmp/c02/results \
    --fixtures /tmp/c02/fixtures --shape tp4_v_block_s1

Collection emits C02_CORRECTNESS_PENDING, never a quality PASS. C02_PROFILE=1
replays one two-forward trace between profiler drains instead of timing batches.
TT_METAL_KERNEL_CAPTURE_ONLY=1 collects both ping-pong recipes then explicitly
skips before reading outputs; that skip is not correctness/performance evidence.
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import pytest
import torch

EPSILON = 1e-6
SHAPES = {"tp4_v_block_s1": (1216, 4096), "tp4_v_block_s2": (4864, 4096), "tp4_a_block": (32, 2048)}


def _file_hash(path):
    with path.open("rb") as handle:
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_hash(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _prepare(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for shape, (rows, dim) in SHAPES.items():
        samples = []
        for seed in (17, 31):
            generator = torch.Generator().manual_seed(seed)
            # Nonzero mean and both signs in the affine stress the reduction and
            # shift/scale mapping, not just an identity gamma or zero-mean input.
            x = (torch.randn((1, 1, rows, dim), generator=generator) * 2 + 3).bfloat16()
            scale = torch.randn((1, 1, 1, dim), generator=generator).bfloat16()
            scale_p1 = (scale.float() + 1).bfloat16()
            shift = torch.randn((1, 1, 1, dim), generator=generator).bfloat16()
            xf = x.float()
            normed = xf * (xf.square().mean(-1, keepdim=True) + EPSILON).rsqrt()
            # Use the ACTUAL bf16 scale_p1 supplied to the device, not an
            # unrounded float32 scale+1; the distinction matters for this test.
            reference = normed * scale_p1.float() + shift.float()
            samples.append({"seed": seed, "x": x, "scale_p1": scale_p1, "shift": shift, "reference": reference})
        path = directory / f"{shape}.pt"
        torch.save({"shape": shape, "epsilon": EPSILON, "samples": samples}, path)
        print(f"C02_FIXTURE {path} sha256={_file_hash(path)}")


@pytest.fixture
def device_params():
    from models.tt_dit.utils.test import ring_params

    return {**ring_params, "trace_region_size": 1_048_576}


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.skipif(
    not os.environ.get("C02_SHAPE") and not os.environ.get("C02_MODE"), reason="opt-in C02 evidence collection"
)
@pytest.mark.skip_post_commit
def test_collect_ltx_norm_adaln(mesh_device):
    import ttnn
    from models.tt_dit.layers.normalization import DistributedRMSNorm
    from models.tt_dit.models.transformers.ltx.transformer_ltx import _norm_adaln
    from models.tt_dit.parallel.manager import CCLManager
    from models.tt_dit.tests.unit.test_distributed_rmsnorm_fused import LTX, _gather, _make_cfgs
    from models.tt_dit.utils.tensor import bf16_tensor

    shape, mode = os.environ["C02_SHAPE"], os.environ["C02_MODE"]
    assert shape in SHAPES and mode in {"base", "fused"}, "invalid explicit C02 selection"
    rows, dim = SHAPES[shape]
    cfg = next(c for c in _make_cfgs(LTX, 4) if c.cid == shape)
    assert (cfg.rows, cfg.dim) == (rows, dim), "production shape table changed"
    fixture_path = Path(os.environ["C02_FIXTURES"]) / f"{shape}.pt"
    result_dir = Path(os.environ["C02_RESULTS"])
    result_path = result_dir / f"{shape}-{mode}.pt"
    assert not result_path.exists(), f"refusing to overwrite raw evidence: {result_path}"
    fixture = torch.load(fixture_path, map_location="cpu", weights_only=True)
    assert fixture["shape"] == shape and fixture["epsilon"] == EPSILON
    samples = fixture["samples"]
    inputs = {
        key: bf16_tensor(samples[0][key], device=mesh_device, mesh_axis=0, shard_dim=-1)
        for key in ("x", "scale_p1", "shift")
    }
    ccl = CCLManager(mesh_device=mesh_device, num_links=2, topology=ttnn.Topology.Ring)
    norm = DistributedRMSNorm(
        embedding_dim=dim,
        norm_eps=EPSILON,
        norm_elementwise_affine=False,
        mesh_axis=0,
        mesh_device=mesh_device,
        ccl_manager=ccl,
    )

    def run():
        return _norm_adaln(norm, inputs["x"], inputs["shift"], inputs["scale_p1"], fuse=mode == "fused")

    def read(tensor):
        return _gather(tensor, 0).bfloat16()  # one SP replica, all four TP shards

    # Paired forwards bind both semaphore/POB slots; repeating one slot can race
    # a late fabric increment. All input addresses predate capture and stay live.
    warm = [run(), run()]
    if os.environ.get("TT_METAL_KERNEL_CAPTURE_ONLY") == "1":
        pytest.skip("kernel recipe capture only; not correctness/timing evidence")
    ttnn.synchronize_device(mesh_device)
    eager_hashes = [_tensor_hash(read(output)) for output in warm]
    trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    outputs = [run(), run()]
    ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
    timing_samples = []
    replays, saved_outputs = [], []
    try:
        for sample_index in (0, 1, 0):
            for key, destination in inputs.items():
                source = bf16_tensor(
                    samples[sample_index][key], device=mesh_device, mesh_axis=0, shard_dim=-1, on_host=True
                )
                ttnn.copy_host_to_device_tensor(source, destination)
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            actual = [read(output) for output in outputs]
            replays.append({"sample_index": sample_index, "output_hashes": [_tensor_hash(t) for t in actual]})
            if len(saved_outputs) < 2:
                saved_outputs.append(actual[0])
        if os.environ.get("C02_PROFILE") == "1":
            from tracy import signpost

            ttnn.ReadDeviceProfiler(mesh_device)
            signpost("start", f"C02 {shape} {mode}: two scalar norm/AdaLN forwards")
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            ttnn.ReadDeviceProfiler(mesh_device)
            signpost("stop")
        else:
            for _ in range(5):
                ttnn.synchronize_device(mesh_device)
                start = time.perf_counter()
                for _ in range(20):
                    ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh_device)
                timing_samples.append((time.perf_counter() - start) * 1e6 / 40)
        final_hashes = [_tensor_hash(read(output)) for output in outputs]
    finally:
        ttnn.release_trace(mesh_device, trace)
    result_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "shape": shape,
        "mode": mode,
        "fixture_sha256": _file_hash(fixture_path),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {
            p: _file_hash(Path(p))
            for p in (
                "models/tt_dit/models/transformers/ltx/transformer_ltx.py",
                __file__,
            )
        },
        "mesh_shape": list(mesh_device.shape),
        "arch": str(mesh_device.arch()),
        "eager_hashes": eager_hashes,
        "replays": replays,
        "saved_outputs": saved_outputs,
        "final_hashes": final_hashes,
        "samples_us_per_norm_adaln": timing_samples,
        "profiler_env": os.environ.get("TT_METAL_DEVICE_PROFILER", "0"),
        "timing_boundary": "synchronized two-forward trace; 20 replays per batch divided by40",
    }
    torch.save(record, result_path)
    print(f"C02_CORRECTNESS_PENDING {result_path}; verify both routes off-device")
    print("C02_TIMING_SAMPLES " + json.dumps(timing_samples))


def _metrics(actual, reference):
    a, b = actual.float().flatten(), reference.float().flatten()
    assert torch.isfinite(a).all() and torch.isfinite(b).all(), "non-finite norm/AdaLN output"
    return {
        "pcc": torch.corrcoef(torch.stack((a, b)))[0, 1].item(),
        "relative_rmse": ((a - b).square().mean().sqrt() / b.std()).item(),
        "max_abs_error": (a - b).abs().max().item(),
    }


def _verify(result_dir, fixture_dir, shape):
    report_path = result_dir / f"{shape}-verified.json"
    report_path.unlink(missing_ok=True)
    fixture_path = fixture_dir / f"{shape}.pt"
    fixture = torch.load(fixture_path, map_location="cpu", weights_only=True)
    results, reports = {}, {}
    for mode in ("base", "fused"):
        path = result_dir / f"{shape}-{mode}.pt"
        result = torch.load(path, map_location="cpu", weights_only=True)
        assert result["shape"] == shape and result["mode"] == mode
        assert result["fixture_sha256"] == _file_hash(fixture_path), "fixture provenance changed"
        assert [r["sample_index"] for r in result["replays"]] == [0, 1, 0]
        expected_hashes = [_tensor_hash(out) for out in result["saved_outputs"]]
        assert (
            len(expected_hashes) == 2 and expected_hashes[0] != expected_hashes[1]
        ), "changed inputs did not change outputs"
        assert result["eager_hashes"] == result["final_hashes"] == [expected_hashes[0]] * 2, "eager/final trace drift"
        for replay in result["replays"]:
            assert (
                replay["output_hashes"] == [expected_hashes[replay["sample_index"]]] * 2
            ), "stale/nondeterministic replay"
        metrics = [
            _metrics(out, sample["reference"]) for out, sample in zip(result["saved_outputs"], fixture["samples"])
        ]
        assert all(m["pcc"] >= 0.999 and m["relative_rmse"] <= 0.02 for m in metrics), metrics
        reports[mode] = {
            "commit": result["commit"],
            "result_sha256": _file_hash(path),
            "metrics": metrics,
            "samples_us_per_norm_adaln": result["samples_us_per_norm_adaln"],
            "profiler_env": result["profiler_env"],
        }
        results[mode] = result
    # Fused fp32 affine omits a bf16 intermediate. Report the numerical delta;
    # exact cross-route parity is neither expected nor a denoising quality gate.
    pair_metrics = [
        _metrics(fused, base)
        for fused, base in zip(results["fused"]["saved_outputs"], results["base"]["saved_outputs"])
    ]
    report = {
        "shape": shape,
        "quality_pass": True,
        "fixture_sha256": _file_hash(fixture_path),
        "routes": reports,
        "fused_vs_base": pair_metrics,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print("C02_CORRECTNESS_PASS " + json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", type=Path)
    action.add_argument("--verify", type=Path)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--shape", choices=list(SHAPES))
    args = parser.parse_args()
    if args.prepare:
        _prepare(args.prepare)
    else:
        if not (args.fixtures and args.shape):
            parser.error("--verify requires --fixtures and --shape")
        _verify(args.verify, args.fixtures, args.shape)
