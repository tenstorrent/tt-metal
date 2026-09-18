# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""C01 changed-input A/B/A regression, separate from the existing timing test.

CPU: python -m models.tt_dit.tests.unit.test_ltx_qk_rope_replay --prepare DIR
Broker: C01_ABA_SHAPE=tp4_v_selfattn_qk_s1 C01_ABA_MODE=base
        C01_ABA_FIXTURES=DIR C01_ABA_RESULTS=RESULTS pytest <this-file> -s
Repeat with MODE=fused, preserve, or active; CPU: --verify RESULTS --fixtures DIR
--shape SHAPE --variant fused|preserve|active. Preserve and active require exact baseline parity.
Collection only emits C01_ABA_CORRECTNESS_PENDING. No timing is collected.
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

import pytest
import torch

from models.tt_dit.tests.unit.test_ltx_norm_adaln import _file_hash, _metrics, _tensor_hash

SHAPES = {
    "tp4_v_selfattn_qk_s1": (1216, 4096, 128),
    "tp4_v_selfattn_qk_s2": (4864, 4096, 128),
    "tp4_a_selfattn_qk": (32, 2048, 64),
}
EPSILON = 1e-6


def _prepare(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for shape, (rows, dim, head_dim) in SHAPES.items():
        weight = torch.randn((1, dim), generator=torch.Generator().manual_seed(0)).bfloat16()
        samples = []
        for seed in (17, 31):
            generator = torch.Generator().manual_seed(seed)
            x = (torch.randn((1, 1, rows, dim), generator=generator) * 2 + 3).bfloat16()
            angles = torch.randn((1, 32, rows, head_dim // 2), generator=generator)
            cos = angles.cos().repeat_interleave(2, dim=-1).bfloat16()
            sin = angles.sin().repeat_interleave(2, dim=-1).bfloat16()
            xf = x.float().reshape(rows, dim)
            normalized = xf * (xf.square().mean(-1, keepdim=True) + EPSILON).rsqrt() * weight.float()
            yh = normalized.reshape(rows, 32, head_dim)
            rotated = torch.stack((-yh[..., 1::2], yh[..., 0::2]), dim=-1).flatten(-2)
            reference = (yh * cos.float()[0].permute(1, 0, 2) + rotated * sin.float()[0].permute(1, 0, 2)).reshape(
                rows, dim
            )
            samples.append({"seed": seed, "x": x, "cos": cos, "sin": sin, "reference": reference})
        path = directory / f"{shape}.pt"
        torch.save({"shape": shape, "epsilon": EPSILON, "weight": weight, "samples": samples}, path)
        print(f"C01_ABA_FIXTURE {path} sha256={_file_hash(path)}")


@pytest.fixture
def device_params():
    from models.tt_dit.utils.test import ring_params

    return {**ring_params, "trace_region_size": 1_048_576}


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.skipif(
    not os.environ.get("C01_ABA_SHAPE") and not os.environ.get("C01_ABA_MODE"),
    reason="opt-in C01 changed-input regression",
)
@pytest.mark.skip_post_commit
def test_ltx_qk_rope_replay(mesh_device):
    import ttnn
    from models.tt_dit.layers.normalization import DistributedRMSNorm
    from models.tt_dit.parallel.manager import CCLManager
    from models.tt_dit.tests.unit.test_distributed_rmsnorm_fused import LTX, _gather, _make_cfgs
    from models.tt_dit.utils.mochi import get_rot_transformation_mat
    from models.tt_dit.utils.tensor import bf16_tensor

    shape, mode = os.environ["C01_ABA_SHAPE"], os.environ["C01_ABA_MODE"]
    assert shape in SHAPES and mode in {"base", "fused", "preserve", "active"}
    cfg = next(c for c in _make_cfgs(LTX, 4) if c.cid == shape)
    assert (cfg.rows, cfg.dim, cfg.head_dim) == SHAPES[shape], "production shape table changed"
    fixture_path = Path(os.environ["C01_ABA_FIXTURES"]) / f"{shape}.pt"
    result_dir = Path(os.environ["C01_ABA_RESULTS"])
    result_path = result_dir / f"{shape}-{mode}.pt"
    assert not result_path.exists(), f"refusing to overwrite raw evidence: {result_path}"
    fixture = torch.load(fixture_path, map_location="cpu", weights_only=True)
    assert fixture["shape"] == shape and fixture["epsilon"] == EPSILON
    samples = fixture["samples"]

    def mapped(key, tensor, *, on_host=False):
        # Features shard on their last dimension; per-head RoPE shards on heads.
        return bf16_tensor(tensor, device=mesh_device, mesh_axis=0, shard_dim=-1 if key == "x" else 1, on_host=on_host)

    inputs = {key: mapped(key, samples[0][key]) for key in ("x", "cos", "sin")}
    weight = bf16_tensor(fixture["weight"], device=mesh_device, mesh_axis=0, shard_dim=-1)
    transform = bf16_tensor(get_rot_transformation_mat(), device=mesh_device)
    ccl = CCLManager(mesh_device=mesh_device, num_links=2, topology=ttnn.Topology.Ring)
    norm = DistributedRMSNorm(
        embedding_dim=cfg.dim,
        norm_eps=EPSILON,
        norm_elementwise_affine=False,
        mesh_axis=0,
        mesh_device=mesh_device,
        ccl_manager=ccl,
    )
    rope_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def run():
        args = {"num_heads_per_device": cfg.heads, "dynamic_weight": weight}
        if mode in {"fused", "preserve"}:
            return norm(
                inputs["x"],
                **args,
                rope_cos=inputs["cos"],
                rope_sin=inputs["sin"],
                trans_mat=transform,
                preserve_rope_rounding=mode == "preserve",
            )
        normalized = norm(inputs["x"], **args)
        return ttnn.experimental.rotary_embedding_llama(
            normalized,
            inputs["cos"],
            inputs["sin"],
            transform,
            compute_kernel_config=rope_config,
            **({"active_cores_only": True} if mode == "active" else {}),
        )

    def read(output):
        return _gather(output, 0).bfloat16()

    warm = [run(), run()]
    if os.environ.get("TT_METAL_KERNEL_CAPTURE_ONLY") == "1":
        pytest.skip("kernel recipe capture only; not replay correctness evidence")
    ttnn.synchronize_device(mesh_device)
    eager_hashes = [_tensor_hash(read(output)) for output in warm]
    trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    outputs = [run(), run()]
    ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
    replays, saved_outputs = [], []
    try:
        for sample_index in (0, 1, 0):
            for key, destination in inputs.items():
                host_tensor = mapped(key, samples[sample_index][key], on_host=True)
                ttnn.copy_host_to_device_tensor(host_tensor, destination)
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            actual = [read(output) for output in outputs]
            replays.append({"sample_index": sample_index, "output_hashes": [_tensor_hash(t) for t in actual]})
            if len(saved_outputs) < 2:
                saved_outputs.append(actual[0])
    finally:
        ttnn.release_trace(mesh_device, trace)
    result_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "shape": shape,
        "mode": mode,
        "fixture_sha256": _file_hash(fixture_path),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {
            str(p): _file_hash(Path(p))
            for p in (
                __file__,
                "models/tt_dit/layers/normalization.py",
                "ttnn/cpp/ttnn/operations/experimental/ccl/dit_fused_distributed_rmsnorm/device/"
                "dit_fused_distributed_rmsnorm_program_factory.cpp",
                "ttnn/cpp/ttnn/operations/experimental/ccl/dit_fused_distributed_rmsnorm/device/"
                "kernels/compute/dit_rmsnorm_fused_compute.cpp",
            )
        },
        "mesh_shape": list(mesh_device.shape),
        "arch": str(mesh_device.arch()),
        "eager_hashes": eager_hashes,
        "replays": replays,
        "saved_outputs": saved_outputs,
    }
    torch.save(record, result_path)
    print(f"C01_ABA_CORRECTNESS_PENDING {result_path}; verify both routes off-device")


def _verify(result_dir, fixture_dir, shape, variant="fused"):
    suffix = "" if variant == "fused" else f"-{variant}"
    report_path = result_dir / f"{shape}{suffix}-verified.json"
    report_path.unlink(missing_ok=True)
    fixture_path = fixture_dir / f"{shape}.pt"
    fixture = torch.load(fixture_path, map_location="cpu", weights_only=True)
    reports, saved_outputs = {}, {}
    for mode in ("base", variant):
        path = result_dir / f"{shape}-{mode}.pt"
        result = torch.load(path, map_location="cpu", weights_only=True)
        assert result["shape"] == shape and result["mode"] == mode
        assert result["fixture_sha256"] == _file_hash(fixture_path), "fixture provenance changed"
        assert [r["sample_index"] for r in result["replays"]] == [0, 1, 0]
        expected = [_tensor_hash(out) for out in result["saved_outputs"]]
        assert len(expected) == 2 and expected[0] != expected[1], "changed inputs did not change outputs"
        assert result["eager_hashes"] == [expected[0]] * 2, "eager/trace drift"
        for replay in result["replays"]:
            assert replay["output_hashes"] == [expected[replay["sample_index"]]] * 2, "stale/nondeterministic replay"
        metrics = [
            _metrics(out, sample["reference"]) for out, sample in zip(result["saved_outputs"], fixture["samples"])
        ]
        assert all(m["pcc"] >= 0.999 and m["relative_rmse"] <= 0.02 for m in metrics), metrics
        reports[mode] = {"commit": result["commit"], "result_sha256": _file_hash(path), "metrics": metrics}
        saved_outputs[mode] = result["saved_outputs"]
    if variant in {"preserve", "active"}:
        assert all(
            torch.equal(base, preserve) for base, preserve in zip(saved_outputs["base"], saved_outputs[variant])
        ), f"{variant} route differs from baseline; diagnose before quality/performance claims"
    report = {
        "shape": shape,
        "variant": variant,
        "quality_pass": True,
        "exact_baseline_parity": variant in {"preserve", "active"},
        "fixture_sha256": _file_hash(fixture_path),
        "routes": reports,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print("C01_ABA_CORRECTNESS_PASS " + json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", type=Path)
    action.add_argument("--verify", type=Path)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--shape", choices=list(SHAPES))
    parser.add_argument("--variant", choices=("fused", "preserve", "active"), default="fused")
    args = parser.parse_args()
    if args.prepare:
        _prepare(args.prepare)
    else:
        if not (args.fixtures and args.shape):
            parser.error("--verify requires --fixtures and --shape")
        _verify(args.verify, args.fixtures, args.shape, args.variant)
