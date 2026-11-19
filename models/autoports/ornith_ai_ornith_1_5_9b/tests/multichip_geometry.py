# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Precision-locked local DRAM geometry sweep using real TP4 decoder weights."""

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests.run_decoder import config, pcc, weights
from models.autoports.ornith_ai_ornith_1_5_9b.tt.multichip_decoder import MultichipDecoder, exact_k_shards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--role")
    ap.add_argument("--repair-probe", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(17)
    sd = weights(args.layer)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=8 * 1024 * 1024)
    rows = json.loads(Path(args.output).read_text()) if args.resume and Path(args.output).exists() else []
    completed = {(r["role"], r["cores"], r["block"], r["readers"]) for r in rows}
    try:
        layer = MultichipDecoder.from_state_dict(
            sd, hf_config=config(), layer_idx=args.layer, mesh_device=mesh, residual_layout="replicated"
        )
        for name, w in layer.dram_weights.items():
            if args.role and name != args.role:
                continue
            k, n = w.shape
            inp = ttnn.from_torch(
                torch.randn(1, 1, 1, k).bfloat16() * 0.1,
                device=mesh,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            ref = layer.linear(inp, name)
            expected = [ttnn.to_torch(v) for v in ttnn.get_device_tensors(ref)]
            del ref
            for cores in [64] if args.repair_probe else [16, 32, 64]:
                input_cores, kw = exact_k_shards(k, cores)
                nw = math.ceil(n / (32 * cores))
                im = ttnn.create_sharded_memory_config(
                    (32, kw * 32),
                    ttnn.CoreGrid(x=8, y=input_cores // 8),
                    ttnn.ShardStrategy.WIDTH,
                    use_height_and_width_as_shard_shape=True,
                )
                om = ttnn.create_sharded_memory_config(
                    (32, nw * 32),
                    ttnn.CoreGrid(x=8, y=cores // 8),
                    ttnn.ShardStrategy.WIDTH,
                    use_height_and_width_as_shard_shape=True,
                )
                x = ttnn.to_memory_config(inp, im)
                for block in [v for v in range(1, kw + 1) if kw % v == 0 and (k // 32) % v == 0]:
                    for readers in [1] if args.repair_probe else [1, 2, 4, 8]:
                        if (name, cores, block, readers) in completed:
                            continue
                        print(
                            "CANDIDATE",
                            name,
                            "output_cores",
                            cores,
                            "input_cores",
                            input_cores,
                            "K",
                            k,
                            "block",
                            block,
                            "readers",
                            readers,
                            flush=True,
                        )
                        cfg = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                            in0_block_w=block,
                            per_core_M=1,
                            per_core_N=nw,
                            num_workers_per_dram_bank=readers,
                            use_multicore_pipeline=readers > 1,
                        )

                        def forward():
                            return ttnn.linear(
                                x,
                                w,
                                program_config=cfg,
                                compute_kernel_config=layer.matmul_compute(name),
                                memory_config=om,
                            )

                        out = forward()
                        ttnn.synchronize_device(mesh)
                        score = min(pcc(a, ttnn.to_torch(b)) for a, b in zip(expected, ttnn.get_device_tensors(out)))
                        assert score >= 0.995, (name, cores, block, readers, score)
                        del out
                        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                        out = forward()
                        ttnn.end_trace_capture(mesh, trace, cq_id=0)
                        times = []
                        for _ in range(3):
                            ttnn.synchronize_device(mesh)
                            start = time.perf_counter()
                            for _ in range(40):
                                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                            ttnn.synchronize_device(mesh)
                            times.append((time.perf_counter() - start) * 1000 / 40)
                        ttnn.release_trace(mesh, trace)
                        del out
                        row = {
                            "role": name,
                            "cores": cores,
                            "input_cores": input_cores,
                            "input_shard_tiles": kw,
                            "exact_k": input_cores * kw * 32 == k,
                            "block": block,
                            "readers": readers,
                            "pcc": score,
                            "ms": statistics.median(times),
                            "dtype": "BFP8",
                            "fidelity": "LoFi" if name.startswith("mlp.") else "HiFi2",
                        }
                        rows.append(row)
                        Path(args.output).write_text(json.dumps(rows, indent=2) + "\n")
            print("ROLE_DONE", name, min((r for r in rows if r["role"] == name), key=lambda r: r["ms"]), flush=True)
        print("GEOMETRY_PASS", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
