# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""BF16 state implementation probe; run only via the serialized device runner.

Three independent persistent states start from the same BF16-rounded values.
The candidate stores BF16. The quantized control runs production FP32 post-loop,
then rounds its updated state BF16 -> FP32 before the NEXT token. A third FP32
control never rounds after initialization and separately measures precision loss.
Passing this implementation probe does not establish baseline quality acceptance.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_bf16_state_rne_kernel as fused
from models.autoports.ornith_ai_ornith_1_5_9b.tests.validate_optimized import device_only
from models.autoports.ornith_ai_ornith_1_5_9b.tt import flat_ordered_post_loop as control


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batches", default="1,2,6,13,14,16,17,26,27,31,32")
    ap.add_argument("--waves", default="104,7")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--epsilon", type=float, default=1e-6)
    args = ap.parse_args()
    batches = [int(x) for x in args.batches.split(",")]
    waves = [int(x) for x in args.waves.split(",")]
    assert args.steps > 0 and args.epsilon > 0
    assert all(1 <= b <= 32 for b in batches) and all(1 <= w <= 110 for w in waves)
    torch.set_num_threads(8)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), Path(fused.__file__), Path(control.__file__)]
    sources += sorted(fused.KERNEL_DIR.glob("*")) + sorted(control.KERNEL_DIR.glob("*"))
    sources += [Path("ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/device/kernels/compute/chunk_gdn_math.hpp")]
    result = {
        "command": sys.argv,
        "options": vars(args),
        "mesh": [1, 4],
        "seed_rule": "7905 + batch, reset for each wave",
        "watcher": os.getenv("TT_METAL_WATCHER"),
        "tracking": os.getenv("TT_METAL_TRACE_ALLOC_TRACKING", "0"),
        "skip_program_cache_tracking": os.getenv("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0"),
        "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources if p.is_file()},
        "control": "Production FP32 post-loop; BF16 round of updated state after output, before next step",
        "gates": "Finite output/state; global and each head PCC >= .995; quantized updated state bitwise equal",
        "nonquantized_metrics": "Separate precision-loss diagnostics, not an implementation gate or baseline-quality waiver",
        "scope": "Direct numerical/ownership probe on TP4; no decoder timing, trace, or pool dispatch claim",
        "control_memory": "DRAM for both FP32 controls and roundtrip temporaries; candidate retains DRAM/L1 coverage",
        "lifetime": "Release uploaded inputs, output/temporary tensors and rank views after each host comparison; release states after each batch",
        "records": [],
        "complete": False,
    }

    def save():
        output.write_text(json.dumps(result, indent=2) + "\n")

    def stats(value):
        finite = torch.isfinite(value)
        return {
            "shape": list(value.shape),
            "finite": int(finite.sum()),
            "elements": value.numel(),
            "finite_maxabs": float(value[finite].abs().max()) if bool(finite.any()) else None,
        }

    def score(actual, expected):
        if not bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()):
            return 0.0
        if torch.equal(actual, expected):
            return 1.0
        # Double precision prevents squared-sum underflow for tiny output heads.
        a, b = actual.double().flatten(), expected.double().flatten()
        a, b = a - a.mean(), b - b.mean()
        denom = a.norm() * b.norm()
        return float((a @ b) / denom) if float(denom) > 0 else 0.0

    def compare(actual, expected, label, record, *, gate=False, bitwise=False):
        assert actual.shape == expected.shape, (label, actual.shape, expected.shape)
        actual, expected = actual.float(), expected.float()
        heads = record["batch"] * 8
        head_scores = [score(a, b) for a, b in zip(actual.reshape(heads, -1), expected.reshape(heads, -1))]
        global_score = score(actual, expected)
        finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
        equal = torch.equal(actual, expected)
        record[label] = {
            "pcc": global_score,
            "min_head_pcc": min(head_scores),
            "head_pcc": head_scores,
            "bitwise_equal": equal,
            "maxabs_error": float((actual - expected).abs().max()) if finite else None,
            "actual": stats(actual),
            "expected": stats(expected),
            "gated": gate,
            "bitwise_required": bitwise,
        }
        passed = finite and min(global_score, *head_scores) >= 0.995 and (equal or not bitwise)
        if gate and not passed:
            artifact = output.with_suffix(
                f".B{record['batch']}.wave{record['wave']}.step{record['step']}.rank{record['rank']}.{label}.pt"
            )
            torch.save({"actual": actual, "expected": expected}, artifact)
            record[label]["failure_tensors"] = str(artifact)
            save()
            raise AssertionError((label, record))

    save()
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)

    def device(value, dtype, memory=ttnn.DRAM_MEMORY_CONFIG):
        return ttnn.from_torch(
            value.contiguous(), device=mesh, dtype=dtype, layout=ttnn.TILE_LAYOUT,
            memory_config=memory, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    try:
        for wave_index, wave in enumerate(waves):
            state_memory = ttnn.DRAM_MEMORY_CONFIG if wave_index % 2 == 0 else ttnn.L1_MEMORY_CONFIG
            for batch in batches:
                torch.manual_seed(7905 + batch)
                initial = (torch.randn(batch * 8, 128, 128) * 0.01).bfloat16()
                state_fused = device(initial, ttnn.bfloat16, state_memory)
                state_quantized = device(initial.float(), ttnn.float32, ttnn.DRAM_MEMORY_CONFIG)
                state_full = device(initial.float(), ttnn.float32, ttnn.DRAM_MEMORY_CONFIG)
                assert len({s.buffer_address() for s in (state_fused, state_quantized, state_full)}) == 3
                for step in range(args.steps):
                    memory = ttnn.DRAM_MEMORY_CONFIG if step % 2 == 0 else ttnn.L1_MEMORY_CONFIG
                    conv = (torch.randn(1, 1, batch, 2048) * 0.2).bfloat16()
                    if step % 4 == 0:
                        conv[..., :1024] = 0
                    elif step % 4 == 2:
                        conv[..., :1024] *= 0.001
                    g = -torch.rand(batch, 1, 8) * (0.01 if step % 2 else 3.0)
                    beta = torch.rand(batch, 1, 8).bfloat16()
                    if step % 4 == 3:
                        beta[..., :4] = torch.tensor([0.0, 1e-12, 1e-6, 1.0]).bfloat16()
                    gate = (torch.randn(1, 1, batch, 1024) * (0.1, 1, 6)[step % 3]).bfloat16()
                    gamma = (1 + torch.randn(1, 1, 1, 128) * 0.35).bfloat16()
                    gamma[..., ::7] *= -1
                    gamma[..., ::31] = 0
                    conv_tt, g_tt = device(conv, ttnn.bfloat16, memory), device(g, ttnn.float32, memory)
                    beta_tt, gate_tt = device(beta, ttnn.bfloat16, memory), device(gate, ttnn.bfloat16, memory)
                    gamma_tt = device(gamma, ttnn.bfloat16, memory)
                    with device_only():
                        actual_tt = fused.flat_ordered_post_loop(
                            conv_tt, g_tt, beta_tt, state_fused, gate_tt, gamma_tt, args.epsilon, wave
                        )
                        expected_tt = control.flat_ordered_post_loop(
                            conv_tt, g_tt, beta_tt, state_quantized, gate_tt, gamma_tt, args.epsilon, wave
                        )
                        full_tt = control.flat_ordered_post_loop(
                            conv_tt, g_tt, beta_tt, state_full, gate_tt, gamma_tt, args.epsilon, wave
                        )
                        # Output was formed using the old state. Round only the
                        # updated control state, then restore its persistent FP32
                        # allocation for the next token. Never copy candidate state.
                        rounded_tt = ttnn.typecast(state_quantized, ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                        widened_tt = ttnn.typecast(rounded_tt, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                        ttnn.copy(widened_tt, state_quantized)
                    tensors = [actual_tt, expected_tt, full_tt, state_fused, rounded_tt, state_full, state_quantized]
                    ranks = [ttnn.get_device_tensors(t) for t in tensors]
                    assert all(len(t) == 4 for t in ranks)
                    for rank, values in enumerate(zip(*ranks)):
                        actual, expected, full, updated, rounded, full_state, restored = [
                            ttnn.to_torch(t).float() for t in values
                        ]
                        record = {
                            "batch": batch, "wave": wave, "step": step, "rank": rank,
                            "input_memory": "DRAM" if step % 2 == 0 else "L1",
                            "state_memory": "DRAM" if wave_index % 2 == 0 else "L1",
                            "control_state_memory": "DRAM",
                            "max_heads_per_core": (batch * 8 + min(wave, batch * 8) - 1) // min(wave, batch * 8),
                        }
                        result["records"].append(record)
                        # Save precision diagnostics before evaluating strict gates.
                        compare(actual, full, "output_vs_unquantized_fp32", record)
                        compare(updated, full_state, "state_vs_unquantized_fp32", record)
                        compare(expected, full, "quantized_control_output_vs_unquantized_fp32", record)
                        compare(restored, rounded, "control_roundtrip", record, gate=True, bitwise=True)
                        compare(updated, rounded, "state_vs_quantized_control", record, gate=True, bitwise=True)
                        compare(actual, expected, "output_vs_quantized_control", record, gate=True)
                    save()
                    # Every rank has been copied to host above. Drop all device
                    # aliases before the next step/batch; do not force-deallocate
                    # a persistent buffer or retain its rank views accidentally.
                    del values, ranks, tensors
                    del actual_tt, expected_tt, full_tt, rounded_tt, widened_tt
                    del conv_tt, g_tt, beta_tt, gate_tt, gamma_tt
                print("BF16_STATE_RNE_LIFETIME_DIRECT_PASS", batch, "wave", wave, flush=True)
                del state_fused, state_quantized, state_full
        result["complete"] = True
    except BaseException as exc:
        result["error"] = repr(exc)
        raise
    finally:
        save()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
