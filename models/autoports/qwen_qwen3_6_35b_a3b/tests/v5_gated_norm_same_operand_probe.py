# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic same-operand gated norm stages from actual combined layer34 captures.

No production changes; --oracle-only performs CPU reference/source checks without
TTNN import. Numerical differences are reported without relaxing decoder gates.
"""
import argparse
import hashlib
import json
import os
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "doc/fused_decoder/endurance"
REVISION = "345d467b117d1fd5eda9b879a5988de9ae5f47a7"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def compare(ref, got):
    ref, got = ref.detach().cpu().reshape(-1), got.detach().cpu().reshape(-1)
    assert ref.shape == got.shape
    x, y = ref.double(), got.double()
    finite = bool(torch.isfinite(y).all())
    same_dtype = ref.dtype == got.dtype
    codes = (
        int((ref.contiguous().view(torch.uint8) != got.contiguous().view(torch.uint8)).sum()) if same_dtype else None
    )
    diff = x - y
    xc, yc = x - x.mean(), y - y.mean()
    denom = float(xc.norm() * yc.norm())
    return dict(
        reference_dtype=str(ref.dtype),
        actual_dtype=str(got.dtype),
        finite=finite,
        different_values=int((ref != got).sum()),
        different_bytes=codes,
        exact_bytes=same_dtype and codes == 0,
        maxabs=float(diff.abs().max()),
        relative_l2=float(diff.norm() / x.norm()) if float(x.norm()) else float(diff.norm()),
        pcc=float((xc @ yc) / denom) if denom else None,
    )


def reference(core, z, gamma, eps):
    core = core.reshape(1, 32, 128).bfloat16()
    z = z.reshape(1, 32, 128).bfloat16()
    wide = core.float()
    squares = wide.pow(2)
    sums = squares.sum(-1, keepdim=True)
    variance = squares.mean(-1, keepdim=True)
    shifted = variance + eps
    inverse = torch.rsqrt(shifted)
    normalized_fp32 = wide * inverse
    normalized = normalized_fp32.bfloat16()
    weighted = gamma.reshape(1, 1, 128) * normalized
    gate = F.silu(z.float())
    gated = (weighted * gate).bfloat16()
    return dict(
        core_bf16=core,
        squares=squares,
        sums=sums,
        variance=variance,
        shifted=shifted,
        inverse=inverse,
        normalized_fp32=normalized_fp32,
        normalized_bf16=normalized,
        weighted_fp32_carrier=weighted.float(),
        weighted_bf16=weighted,
        silu_standalone_fp32=gate,
        gated_bf16=gated,
    )


def run(args):
    torch.set_num_threads(8)
    assert not args.output.exists(), "Use a fresh output path to preserve prior results"
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    key = f"model.language_model.layers.{args.layer}.linear_attn.norm.weight"
    with safe_open(args.checkpoint / index[key], framework="pt", device="cpu") as handle:
        gamma = handle.get_tensor(key)
    assert gamma.dtype == torch.bfloat16 and tuple(gamma.shape) == (128,)
    eps = json.loads((args.checkpoint / "config.json").read_text())["text_config"]["rms_norm_eps"]
    captures = []
    result = dict(
        status="cpu_oracles_ready",
        diagnostic_only=True,
        acceptance_eligible=False,
        probe_sha256=sha(__file__),
        checkpoint_revision=REVISION,
        layer=args.layer,
        source_snapshot=str(args.snapshot),
        source_snapshot_sha256=sha(args.snapshot),
        positions=args.positions,
        epsilon=eps,
        variants=["native_bf16", "native_fp32_then_bf16", "explicit_fp32"],
        gamma_sha256=hashlib.sha256(gamma.view(torch.uint8).numpy().tobytes()).hexdigest(),
        comparisons_are="Same actual TT core/Z operands versus canonical CPU HF stages; original HF is separate context",
        timing="Not measured; no incomplete-adapter performance claim",
        cases=[],
        device_closed=None,
    )
    for pos in args.positions:
        path = Path(str(args.capture_prefix) + f".pos{pos}.pt")
        d = torch.load(path, map_location="cpu", weights_only=False)
        assert d["layer"] == args.layer and d["position"] == pos
        # Snapshot-aware capture provenance may identify the snapshot separately.
        values = list(d.get("source_sha256", {}).values())
        measured = d.get("measured_decoder_source_sha256")
        assert (
            result["source_snapshot_sha256"] in values or measured == result["source_snapshot_sha256"]
        ), "Capture/source identity mismatch"
        t, h = d["tt_values"], d["hf_values"]
        cpu = reference(t["core"], t["projection_z"], gamma, eps)
        hf_cpu = reference(h["core"], h["projection_z"], gamma, eps)
        assert torch.equal(hf_cpu["gated_bf16"].reshape(-1), h["gated_core"].reshape(-1))
        if "cpu_canonical_gated_core_on_tt_inputs" in t:
            assert torch.equal(cpu["gated_bf16"].reshape(-1), t["cpu_canonical_gated_core_on_tt_inputs"].reshape(-1))
        captures.append((pos, t, cpu))
        result["cases"].append(
            dict(
                position=pos,
                capture=str(path),
                capture_sha256=sha(path),
                capture_source_sha256=d.get("source_sha256"),
                original_captured_gated_vs_same_input_hf=compare(cpu["gated_bf16"], t["gated_core"]),
                original_core_bf16_vs_hf=compare(h["core"], t["core"].bfloat16()),
                variants={},
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record

    record(args.output)
    save()
    if args.oracle_only:
        assert "ttnn" not in __import__("sys").modules
        print("CPU_ORACLES_PASS", len(captures))
        return
    for name in ("TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER"):
        assert os.environ.get(name) in (None, "", "0"), name
    import ttnn

    mesh = None
    outputs = {}
    try:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], l1_small_size=32768)
        result.update(status="running", device_closed=False)
        compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        def upload(value, dtype):
            return ttnn.from_torch(
                value.contiguous(),
                device=mesh,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        gamma_tt = upload(gamma.float().reshape(1, 1, 128), ttnn.float32)
        round_op = ttnn.UnaryWithParam(ttnn.UnaryOpType.TYPECAST, float(ttnn.float32.value), float(ttnn.bfloat16.value))
        widen_op = ttnn.UnaryWithParam(ttnn.UnaryOpType.TYPECAST, float(ttnn.bfloat16.value), float(ttnn.float32.value))
        mem = ttnn.L1_MEMORY_CONFIG
        for case, (pos, t, cpu) in zip(result["cases"], captures):
            # Start from actual FP32 recurrence output so the mandatory BF16 core boundary is exercised.
            core = upload(t["core"].reshape(1, 32, 128), ttnn.float32)
            z = upload(t["projection_z"].reshape(1, 32, 128), ttnn.bfloat16)
            outputs[str(pos)] = {"cpu_reference": cpu, "variants": {}}
            for variant in result["variants"]:
                stages = {}
                core_bf16 = ttnn.typecast(core, ttnn.bfloat16, memory_config=mem)
                stages["core_bf16"] = core_bf16
                if variant == "native_bf16":
                    normalized = ttnn.rms_norm(core_bf16, epsilon=eps, compute_kernel_config=compute, memory_config=mem)
                else:
                    wide = ttnn.typecast(core_bf16, ttnn.float32, memory_config=mem)
                    if variant == "native_fp32_then_bf16":
                        normalized_fp32 = ttnn.rms_norm(
                            wide, epsilon=eps, compute_kernel_config=compute, memory_config=mem
                        )
                    else:
                        squares = ttnn.multiply(
                            wide, wide, dtype=ttnn.float32, fast_and_approximate_mode=False, memory_config=mem
                        )
                        sums = ttnn.sum(squares, dim=-1, keepdim=True, compute_kernel_config=compute, memory_config=mem)
                        variance = ttnn.multiply(
                            sums, 1 / 128, dtype=ttnn.float32, fast_and_approximate_mode=False, memory_config=mem
                        )
                        shifted = ttnn.add(variance, eps, memory_config=mem)
                        inverse = ttnn.rsqrt(shifted, fast_and_approximate_mode=False, memory_config=mem)
                        normalized_fp32 = ttnn.multiply(
                            wide, inverse, dtype=ttnn.float32, fast_and_approximate_mode=False, memory_config=mem
                        )
                        stages.update(squares=squares, sums=sums, variance=variance, shifted=shifted, inverse=inverse)
                    stages["normalized_fp32"] = normalized_fp32
                    normalized = ttnn.typecast(normalized_fp32, ttnn.bfloat16, memory_config=mem)
                stages["normalized_bf16"] = normalized
                weighted = ttnn.multiply(
                    normalized,
                    gamma_tt,
                    dtype=ttnn.float32,
                    fast_and_approximate_mode=False,
                    activations=[round_op, widen_op],
                    memory_config=mem,
                )
                stages["weighted_fp32_carrier"] = weighted
                stages["weighted_bf16"] = ttnn.typecast(weighted, ttnn.bfloat16, memory_config=mem)
                zw = ttnn.typecast(z, ttnn.float32, memory_config=mem)
                stages["silu_standalone_fp32"] = ttnn.silu(zw, memory_config=mem)
                stages["gated_bf16"] = ttnn.multiply(
                    weighted,
                    zw,
                    dtype=ttnn.bfloat16,
                    fast_and_approximate_mode=False,
                    input_tensor_b_activations=[ttnn.UnaryOpType.SILU],
                    memory_config=mem,
                )
                # Readback is diagnostic-only, outside any timed or captured region.
                host = {k: ttnn.to_torch(v).reshape(cpu[k].shape).clone() for k, v in stages.items()}
                checks = {k: compare(cpu[k], v) for k, v in host.items()}
                assert all(v["finite"] for v in checks.values())
                checks["gated_vs_captured_tt"] = compare(t["gated_core"], host["gated_bf16"])
                case["variants"][variant] = checks
                outputs[str(pos)]["variants"][variant] = host
                save()
                del stages, host, normalized, weighted, zw, core_bf16
                if variant != "native_bf16":
                    del normalized_fp32, wide
                if variant == "explicit_fp32":
                    del squares, sums, variance, shifted, inverse
            del core, z
            print("SAME_OPERAND_GATED_NORM", pos, flush=True)
        tensors = Path(str(args.output) + ".tensors.pt")
        torch.save(outputs, tensors)
        result.update(status="completed_diagnostic", tensors=str(tensors), tensors_sha256=sha(tensors))
    except Exception:
        result.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
            result["device_closed"] = True
        save()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture-prefix", type=Path, default=E / "v5_combined_layer34_stream_probe.json")
    p.add_argument("--snapshot", type=Path, default=E / "v5_ordered_qz_postnorm_bf16_gate.py.snapshot")
    p.add_argument("--layer", type=int, default=34)
    p.add_argument("--positions", type=int, nargs="+", default=[128, 129, 130, 137, 181])
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
        / "hub/models--Qwen--Qwen3.6-35B-A3B/snapshots"
        / REVISION,
    )
    p.add_argument("--oracle-only", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
