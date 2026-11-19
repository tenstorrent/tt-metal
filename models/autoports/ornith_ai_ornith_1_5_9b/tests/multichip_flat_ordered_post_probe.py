# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Direct TP4 postnorm probe against separate TTNN ops and a PyTorch recurrence.

Run only through the serialized device runner, after installing the CB repair.
This does not patch kernels. It uses independent persistent states for the fused
and separate paths, tests DRAM/L1 input readers on alternating steps, and retains
the production BF16 recurrence, RMSNorm, and activated-LHS pack boundaries.
Physical output padding is reported, never treated as a public logical token.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_flat_ordered as ordered
from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_flat_ordered_post as fused
from models.autoports.ornith_ai_ornith_1_5_9b.tests.run_decoder import pcc
from models.autoports.ornith_ai_ornith_1_5_9b.tests.validate_optimized import device_only


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batches", default="1,2,13,14,16,17,26,27,31,32")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--epsilon", type=float, default=1e-6)
    ap.add_argument("--wave", type=int, default=104)
    args = ap.parse_args()
    batches = list(map(int, args.batches.split(",")))
    assert args.steps >= 1 and all(1 <= b <= 32 for b in batches)
    assert args.epsilon > 0 and 1 <= args.wave <= 110
    torch.set_num_threads(8)
    torch.manual_seed(7901)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_files = [Path(__file__), Path(ordered.__file__), Path(fused.__file__)]
    source_files += sorted(ordered.KERNEL_DIR.glob("*")) + sorted(fused.KERNEL_DIR.glob("*"))
    result = {
        "command": sys.argv,
        "options": vars(args),
        "mesh": [1, 4],
        "seed": 7901,
        "watcher": os.getenv("TT_METAL_WATCHER"),
        "tracking": os.getenv("TT_METAL_TRACE_ALLOC_TRACKING", "0"),
        "skip_program_cache_tracking": os.getenv("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0"),
        "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files if p.is_file()},
        "control": "flat_ordered + production reshape/permute/BF16/rms_norm/multiply(SILU)",
        "state_control": "independent persistent TTNN state; bitwise equality required",
        "padding_contract": "Only logical B rows are consumed; physical tail diagnostics are informational",
        "records": [],
        "complete": False,
    }

    def save():
        output_path.write_text(json.dumps(result, indent=2) + "\n")

    def stats(value):
        finite = value[torch.isfinite(value)]
        return {
            "shape": list(value.shape),
            "finite": int(finite.numel()),
            "elements": value.numel(),
            "finite_maxabs": float(finite.abs().max()) if finite.numel() else None,
        }

    def check(actual, expected, label, record, *, bitwise=False):
        actual, expected = actual.float(), expected.float()
        equal = torch.equal(actual, expected)
        finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
        score = 1.0 if equal else (pcc(expected, actual) if finite else 0.0)
        record[label] = {
            "pcc": score,
            "bitwise_equal": equal,
            "maxabs_error": float((actual - expected).abs().max()) if finite else None,
            "actual": stats(actual),
            "expected": stats(expected),
        }
        # Constant/zero references need a value check: PCC alone is undefined.
        constant = expected.numel() and bool(torch.all(expected == expected.flatten()[0]))
        passed = finite and (equal if bitwise or constant else score >= 0.995)
        if not passed:
            artifact = output_path.with_suffix(
                f".B{record['batch']}.step{record['step']}.rank{record['rank']}.{label}.pt"
            )
            torch.save({"actual": actual, "expected": expected}, artifact)
            record[label]["failure_tensors"] = str(artifact)
            save()
            raise AssertionError((record["batch"], record["step"], record["rank"], label, record[label]))

    save()
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    compute = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def device(value, dtype, memory=ttnn.DRAM_MEMORY_CONFIG):
        return ttnn.from_torch(
            value.contiguous(),
            device=mesh,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    try:
        for batch in batches:
            state = torch.randn(batch * 8, 128, 128) * 0.01
            state_fused = device(state, ttnn.float32)
            state_control = device(state, ttnn.float32)
            for step in range(args.steps):
                memory = ttnn.DRAM_MEMORY_CONFIG if step % 2 == 0 else ttnn.L1_MEMORY_CONFIG
                conv = (torch.randn(1, 1, batch, 2048) * 0.2).bfloat16()
                if step % 4 == 0:
                    conv[..., :1024] = 0
                elif step % 4 == 2:
                    # Unequal row/head values, including near-zero norm inputs.
                    conv[..., :1024] *= 0.001
                g = -torch.rand(batch, 1, 8) * (0.01 if step % 2 else 3.0)
                beta = torch.rand(batch, 1, 8).bfloat16()
                gate = (torch.randn(1, 1, batch, 1024) * (0.1, 1.0, 6.0)[step % 3]).bfloat16()
                gamma = (1 + 0.35 * torch.randn(1, 1, 1, 128)).bfloat16()
                gamma[..., ::7] *= -1
                gamma[..., ::31] = 0

                flat = conv.float().reshape(batch, 2048)
                q = flat[:, :512].reshape(batch, 4, 128).repeat_interleave(2, 1).reshape(-1, 128)
                k = flat[:, 512:1024].reshape(batch, 4, 128).repeat_interleave(2, 1).reshape(-1, 128)
                v = flat[:, 1024:].reshape(-1, 128)
                q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * 128**-0.5
                k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
                decayed = state * g.reshape(-1, 1, 1).exp()
                delta = (v - torch.einsum("hk,hkv->hv", k, decayed)) * beta.float().reshape(-1, 1)
                state = decayed + k.unsqueeze(-1) * delta.unsqueeze(-2)
                raw_expected = torch.einsum("hk,hkv->hv", q, state).reshape(batch, 8, 128).bfloat16().float()
                normalized = raw_expected * torch.rsqrt(raw_expected.square().mean(-1, keepdim=True) + args.epsilon)
                normalized = (normalized * gamma.float().reshape(128)).bfloat16().reshape(1, 1, batch, 1024)
                # binary_ng packs activated LHS to BF16 before the product.
                activated_gate = torch.nn.functional.silu(gate.float()).bfloat16()
                expected = (normalized.float() * activated_gate.float()).bfloat16()
                conv_tt, g_tt = device(conv, ttnn.bfloat16, memory), device(g, ttnn.float32, memory)
                beta_tt, gate_tt = device(beta, ttnn.bfloat16, memory), device(gate, ttnn.bfloat16, memory)
                gamma_tt = device(gamma, ttnn.bfloat16, memory)
                with device_only():
                    actual_tt = fused.flat_ordered_post(
                        conv_tt, g_tt, beta_tt, state_fused, gate_tt, gamma_tt, args.epsilon, args.wave
                    )
                    raw_tt = ordered.flat_ordered(conv_tt, g_tt, beta_tt, state_control, args.wave)
                    control_tt = ttnn.permute(ttnn.reshape(raw_tt, (batch, 8, 1, 128)), (0, 2, 1, 3))
                    control_tt = ttnn.rms_norm(
                        ttnn.typecast(control_tt, ttnn.bfloat16),
                        weight=gamma_tt,
                        epsilon=args.epsilon,
                        compute_kernel_config=compute,
                    )
                    control_tt = ttnn.reshape(control_tt, (1, 1, batch, 1024))
                    control_tt = ttnn.multiply(gate_tt, control_tt, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
                    # Explicit logical+padded shapes expose existing backing;
                    # the ordinary reshape overload rejects the larger volume.
                    physical_tt = ttnn.reshape(actual_tt, ttnn.Shape(actual_tt.padded_shape), actual_tt.padded_shape)
                rank_tensors = zip(
                    *[
                        ttnn.get_device_tensors(t)
                        for t in (actual_tt, control_tt, state_fused, state_control, physical_tt)
                    ]
                )
                for rank, tensors in enumerate(rank_tensors):
                    actual, control, actual_state, control_state, physical = [ttnn.to_torch(t).float() for t in tensors]
                    record = {
                        "batch": batch,
                        "step": step,
                        "rank": rank,
                        "input_memory": "DRAM" if step % 2 == 0 else "L1",
                    }
                    result["records"].append(record)
                    check(actual_state, control_state, "state_vs_ttnn", record, bitwise=True)
                    check(actual_state, state, "state_vs_torch", record)
                    check(actual, control, "output_vs_ttnn", record)
                    check(actual, expected, "output_vs_torch", record)
                    record["physical_padding"] = stats(physical[..., batch:, :])
                save()
            print("FLAT_ORDERED_POST_BATCH_PASS", batch, flush=True)
        result["complete"] = True
    except BaseException as exc:
        result["error"] = repr(exc)
        raise
    finally:
        save()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
