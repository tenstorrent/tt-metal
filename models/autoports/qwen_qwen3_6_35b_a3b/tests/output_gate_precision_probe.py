# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic only: isolate output gamma rounding and SiLU operand-CB precision.

Reads archived actual core/Z tensors. Canonical uploads are explicitly diagnostic
controls, never a decoder fallback or acceptance evidence. No latency claims.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record

ROOT = Path(__file__).parents[1]
D = ROOT / "doc/fused_decoder/endurance"


def metrics(reference, actual):
    reference, actual = reference.contiguous(), actual.reshape(reference.shape).contiguous()
    a, b = reference.double().flatten(), actual.double().flatten()
    ac, bc = a - a.mean(), b - b.mean()
    return dict(
        pcc=1.0 if torch.equal(a, b) else float(ac @ bc / (ac.norm() * bc.norm())),
        relative_l2=float((a - b).norm() / a.norm()),
        max_abs=float((a - b).abs().max()),
        value_differences=int((a != b).sum()),
        bf16_code_differences=int(
            (reference.bfloat16().view(torch.int16) != actual.bfloat16().view(torch.int16)).sum()
        ),
        elements=a.numel(),
    )


def run(args):
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    record(args.output)
    torch.set_num_threads(4)
    snapshot = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / (
        "hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/345d467b117d1fd5eda9b879a5988de9ae5f47a7"
    )
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    name = "model.language_model.layers.0.linear_attn.norm.weight"
    with safe_open(snapshot / index[name], framework="pt", device="cpu") as f:
        gamma = f.get_tensor(name)
    eps = json.loads((snapshot / "config.json").read_text())["text_config"]["rms_norm_eps"]
    captured = torch.load(args.input, weights_only=False)["tt_values"]
    core = captured["core"].reshape(1, 32, 128).bfloat16()
    z = captured["projection_packed"][..., 8192:12288].reshape(1, 32, 128).bfloat16()
    normalized = (core.float() * torch.rsqrt(core.float().square().mean(-1, keepdim=True) + eps)).bfloat16()
    weighted = (normalized * gamma).bfloat16()
    silu32 = torch.nn.functional.silu(z.float())
    expected = (weighted.float() * silu32).bfloat16()
    extra_silu_round = (weighted.float() * silu32.bfloat16().float()).bfloat16()
    result = dict(
        diagnostic_only=True,
        acceptance_eligible=False,
        execution="serialized physical chip0",
        input=str(args.input),
        input_sha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        checkpoint_revision=snapshot.name,
        cpu_extra_silu_round_effect=metrics(expected, extra_silu_round),
        measurements={},
    )
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    try:
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

        def host(value):
            return ttnn.to_torch(value).reshape(1, 32, 128)

        round_op = ttnn.UnaryWithParam(ttnn.UnaryOpType.TYPECAST, float(ttnn.float32.value), float(ttnn.bfloat16.value))
        widen_op = ttnn.UnaryWithParam(ttnn.UnaryOpType.TYPECAST, float(ttnn.bfloat16.value), float(ttnn.float32.value))
        norm = ttnn.rms_norm(upload(core, ttnn.bfloat16), epsilon=eps, compute_kernel_config=compute)
        tg = upload(gamma.reshape(1, 1, 128).float(), ttnn.float32)
        raw = ttnn.multiply(norm, tg, dtype=ttnn.float32)
        separate = ttnn.typecast(ttnn.typecast(raw, ttnn.bfloat16), ttnn.float32)
        chained = ttnn.multiply(norm, tg, dtype=ttnn.float32, activations=[round_op, widen_op])
        result["normalized"] = metrics(normalized, host(norm))
        result["gamma_chain_vs_separate"] = metrics(host(separate), host(chained))
        z16 = upload(z, ttnn.bfloat16)
        z32 = ttnn.typecast(z16, ttnn.float32)
        standalone_silu32 = ttnn.silu(z32)
        for label, tw in (
            ("native_gamma_separate", separate),
            ("native_gamma_chain", chained),
            ("canonical_weighted_isolation", upload(weighted.float(), ttnn.float32)),
        ):
            host_weighted = host(tw)
            same_weighted_expected = (host_weighted.float() * silu32).bfloat16()
            same_weighted_extra_round = (host_weighted.float() * silu32.bfloat16().float()).bfloat16()
            outputs = {}
            for zlabel, tz in (("z_bf16", z16), ("z_fp32", z32)):
                out = ttnn.multiply(tw, tz, input_tensor_b_activations=[ttnn.UnaryOpType.SILU], dtype=ttnn.bfloat16)
                outputs[zlabel] = host(out)
            outputs["standalone_fp32_silu"] = host(ttnn.multiply(tw, standalone_silu32, dtype=ttnn.bfloat16))
            result["measurements"][label] = dict(
                weighted_vs_hf=metrics(weighted.float(), host_weighted),
                outputs={
                    name: dict(
                        vs_hf=metrics(expected, value),
                        vs_same_weighted_fp32_silu=metrics(same_weighted_expected, value),
                        vs_same_weighted_bf16_silu=metrics(same_weighted_extra_round, value),
                    )
                    for name, value in outputs.items()
                },
                bf16_z_vs_fp32_z=metrics(outputs["z_fp32"], outputs["z_bf16"]),
                fused_fp32_vs_standalone=metrics(outputs["standalone_fp32_silu"], outputs["z_fp32"]),
            )
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default=str(D / "routing_seed154.json.pos5991.pt"))
    p.add_argument("--output", required=True)
    run(p.parse_args())
