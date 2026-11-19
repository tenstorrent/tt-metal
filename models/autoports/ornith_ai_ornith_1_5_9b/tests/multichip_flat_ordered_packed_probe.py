# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Packed gate preparation versus exact TTNN gates plus post-loop recurrence.

Run under the serialized device runner with Watcher. Independent persistent
states receive identical inputs; the control retains the BF16 sigmoid pack,
FP32 add, FP32 softplus activation pack and FP32 decay product. Narrow worker
waves exercise repeated CB reuse, including heads crossing request boundaries.
This is a numerical/isolation probe, not a decoder or performance benchmark.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_flat_ordered_packed as fused
from models.autoports.ornith_ai_ornith_1_5_9b.tests import multichip_flat_ordered_post_loop as control
from models.autoports.ornith_ai_ornith_1_5_9b.tests.run_decoder import pcc
from models.autoports.ornith_ai_ornith_1_5_9b.tests.validate_optimized import device_only


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batches", default="1,2,6,13,14,16,17,26,27,31,32")
    ap.add_argument("--waves", default="104,7")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--epsilon", type=float, default=1e-6)
    args = ap.parse_args()
    batches, waves = [int(x) for x in args.batches.split(",")], [int(x) for x in args.waves.split(",")]
    assert args.steps > 0 and args.epsilon > 0
    assert all(1 <= b <= 32 for b in batches) and all(1 <= w <= 110 for w in waves)
    torch.set_num_threads(8)
    torch.manual_seed(7902)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), Path(fused.__file__), Path(control.__file__)]
    sources += sorted(fused.KERNEL_DIR.glob("*")) + sorted(control.KERNEL_DIR.glob("*"))
    result = {
        "command": sys.argv,
        "options": vars(args),
        "mesh": [1, 4],
        "seed": 7902,
        "watcher": os.getenv("TT_METAL_WATCHER"),
        "tracking": os.getenv("TT_METAL_TRACE_ALLOC_TRACKING", "0"),
        "skip_program_cache_tracking": os.getenv("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0"),
        "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources if p.is_file()},
        "control": "Exact TTNN packed slices/sigmoid/FP32 add/softplus*decay, then flat_ordered_post_loop",
        "gates": "Finite output and state, global and per-head PCC >= .995; zero/constant requires equality",
        "records": [],
        "complete": False,
    }

    def save():
        output.write_text(json.dumps(result, indent=2) + "\n")

    def score(actual, expected):
        if torch.equal(actual, expected):
            return 1.0
        if not bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()):
            return 0.0
        if bool(torch.all(expected == expected.flatten()[0])):
            return 0.0
        return pcc(expected, actual)

    def check(actual, expected, label, record, *, heads=False):
        actual, expected = actual.float(), expected.float()
        scores = [score(actual, expected)]
        if heads:
            shape = (record["batch"] * 8, -1)
            scores += [score(a, b) for a, b in zip(actual.reshape(shape), expected.reshape(shape))]
        finite = torch.isfinite(actual)
        record[label] = {
            "pcc": scores[0],
            "min_head_pcc": min(scores),
            "bitwise_equal": torch.equal(actual, expected),
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
            "finite": int(finite.sum()),
            "elements": actual.numel(),
            "finite_maxabs": float(actual[finite].abs().max()) if bool(finite.any()) else None,
            "maxabs_error": float((actual - expected).abs().max()) if bool(finite.all()) else None,
        }
        if not all(s >= 0.995 for s in scores):
            failure = output.with_suffix(
                f".B{record['batch']}.wave{record['wave']}.step{record['step']}.rank{record['rank']}.{label}.pt"
            )
            torch.save({"actual": actual, "expected": expected}, failure)
            record[label]["failure_tensors"] = str(failure)
            save()
            raise AssertionError((label, record))

    save()
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)

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
        for wave in waves:
            for batch in batches:
                initial = torch.randn(batch * 8, 128, 128) * 0.01
                state_fused, state_control = device(initial, ttnn.float32), device(initial, ttnn.float32)
                for step in range(args.steps):
                    memory = ttnn.DRAM_MEMORY_CONFIG if step % 2 == 0 else ttnn.L1_MEMORY_CONFIG
                    conv = (torch.randn(1, 1, batch, 2048) * 0.2).bfloat16()
                    if step % 4 == 0:
                        conv[..., :1024] = 0
                    elif step % 4 == 2:
                        conv[..., :1024] *= 0.001
                    # Fill unused projection lanes with unrelated, nonzero data.
                    # Reading a neighboring head or the beta/a tile gap must fail.
                    packed = (torch.randn(1, 1, batch, 3136) * 7).bfloat16()
                    packed[..., 2048:3072] = (torch.randn(1, 1, batch, 1024) * (0.1, 1, 6)[step % 3]).bfloat16()
                    raw_beta = torch.randn(1, 1, batch, 8) * (0.1, 2, 12)[step % 3]
                    raw_a = torch.randn(1, 1, batch, 8) * (0.1, 3, 12)[step % 3]
                    # Hit sigmoid saturation and both sides of softplus threshold.
                    if step % 4 == 3:
                        raw_beta[..., :4] = torch.tensor([-30.0, -10.0, 10.0, 30.0])
                        raw_a[..., :4] = torch.tensor([-30.0, -0.001, 19.875, 20.125])
                    packed[..., 3072:3080] = raw_beta.bfloat16()
                    packed[..., 3104:3112] = raw_a.bfloat16()
                    bias = torch.randn(1, 1, 8) * 0.5
                    decay = -torch.exp(torch.randn(1, 1, 8).clamp(-3, 1))
                    if step % 2 == 0:
                        bias = bias.bfloat16().float()  # model-like and full FP32 cases
                    gamma = (1 + torch.randn(1, 1, 1, 128) * 0.35).bfloat16()
                    gamma[..., ::7] *= -1
                    gamma[..., ::31] = 0
                    conv_tt, packed_tt = device(conv, ttnn.bfloat16, memory), device(packed, ttnn.bfloat16, memory)
                    bias_tt, decay_tt = device(bias, ttnn.float32, memory), device(decay, ttnn.float32, memory)
                    gamma_tt = device(gamma, ttnn.bfloat16, memory)
                    with device_only():
                        actual_tt = fused.flat_ordered_packed(
                            conv_tt, packed_tt, state_fused, bias_tt, decay_tt, gamma_tt, args.epsilon, wave
                        )
                        beta_tt = ttnn.reshape(ttnn.sigmoid(packed_tt[:, :, :, 3072:3080]), (batch, 1, 8))
                        a_tt = ttnn.typecast(ttnn.reshape(packed_tt[:, :, :, 3104:3112], (1, batch, 8)), ttnn.float32)
                        g_tt = ttnn.reshape(
                            ttnn.multiply(
                                ttnn.add(a_tt, bias_tt),
                                decay_tt,
                                input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)],
                            ),
                            (batch, 1, 8),
                        )
                        gate_tt = packed_tt[:, :, :, 2048:3072]
                        expected_tt = control.flat_ordered_post_loop(
                            conv_tt, g_tt, beta_tt, state_control, gate_tt, gamma_tt, args.epsilon, wave
                        )
                    assert beta_tt.dtype == ttnn.bfloat16 and g_tt.dtype == ttnn.float32
                    tensors = [actual_tt, expected_tt, state_fused, state_control, beta_tt, g_tt]
                    rank_tensors = [ttnn.get_device_tensors(t) for t in tensors]
                    assert all(len(t) == 4 for t in rank_tensors)
                    expected_beta = packed[..., 3072:3080].float().sigmoid().bfloat16().reshape(batch, 1, 8)
                    expected_g = (
                        torch.nn.functional.softplus(
                            packed[..., 3104:3112].float().reshape(1, batch, 8) + bias, beta=1.0, threshold=20.0
                        )
                        * decay
                    ).reshape(batch, 1, 8)
                    for rank, values in enumerate(zip(*rank_tensors)):
                        actual, expected, updated, updated_control, beta_host, g_host = [
                            ttnn.to_torch(t).float() for t in values
                        ]
                        record = {
                            "batch": batch,
                            "wave": wave,
                            "step": step,
                            "rank": rank,
                            "input_memory": "DRAM" if step % 2 == 0 else "L1",
                            "bias_precision": "BF16 widened" if step % 2 == 0 else "FP32",
                            "max_heads_per_core": (batch * 8 + min(wave, batch * 8) - 1) // min(wave, batch * 8),
                        }
                        result["records"].append(record)
                        check(beta_host, expected_beta, "control_beta_vs_torch", record)
                        check(g_host, expected_g, "control_g_vs_torch", record)
                        check(updated, updated_control, "state_vs_ttnn", record, heads=True)
                        check(actual, expected, "output_vs_ttnn", record, heads=True)
                    save()
                print("FLAT_ORDERED_PACKED_BATCH_PASS", batch, "wave", wave, flush=True)
        result["complete"] = True
    except BaseException as exc:
        result["error"] = repr(exc)
        raise
    finally:
        save()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
