# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real TP4 weights: larger cross-shard K blocks and separate reader/pipeline controls."""

import argparse
import itertools
import json
import math
import statistics
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests.run_decoder import config, pcc, weights
from models.autoports.ornith_ai_ornith_1_5_9b.tt.multichip_decoder import MultichipDecoder


def candidate_schedule(k, n, bank_count, baseline):
    """Breadth-first geometry coverage, with the production baseline first."""

    def preferred(first, values):
        return tuple(dict.fromkeys((first, *values)))

    inputs = preferred(baseline["input_cores"], (32, 64, 48, 16, 8))
    outputs = preferred(baseline["output_cores"], (32, 48, 64, 80, 16))
    reader_counts = preferred(baseline["readers"], (2, 4, 1, 3, 8))
    pipelines = preferred(baseline["readers"] > 1, (True, False))
    families = []
    for inc, outc, readers, pipeline in itertools.product(inputs, outputs, reader_counts, pipelines):
        if k // 32 % inc or (not pipeline and readers > 3):
            continue
        shard_tiles = (
            math.ceil(n / (32 * bank_count)) if pipeline else math.ceil(n / (32 * bank_count * readers)) * readers
        )
        kw = k // 32 // inc
        family_distance = (
            inputs.index(inc),
            outputs.index(outc),
            reader_counts.index(readers),
            pipelines.index(pipeline),
        )
        for block in range(1, k // 32 + 1):
            if k // 32 % block or not (kw % block == 0 or (pipeline and block % kw == 0)):
                continue
            key = (inc, outc, readers, pipeline, block, shard_tiles)
            is_baseline = key[:5] == (
                baseline["input_cores"],
                baseline["output_cores"],
                baseline["readers"],
                baseline["readers"] > 1,
                baseline["block"],
            )
            block_priority = 0 if block == 2 else 1 if block == 1 else block
            # First cover one-knob changes across readers, input/output grids and
            # pipeline mode, instead of exhausting every block in one family.
            priority = (not is_baseline, block_priority, sum(family_distance), family_distance)
            families.append((priority, key))
    return [key for _, key in sorted(families)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--role", required=True)
    ap.add_argument("--dtype", default="bfloat8_b")
    ap.add_argument("--fidelity", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(17)
    sd = weights(args.layer)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=16 * 1024 * 1024)
    output = Path(args.output)
    records = json.loads(output.read_text()) if output.exists() else []
    done = {tuple(r["key"]) for r in records if "must be divisible by workers_per_bank" not in r.get("error", "")}
    try:
        layer = MultichipDecoder.from_state_dict(sd, hf_config=config(), layer_idx=args.layer, mesh_device=mesh)
        w = layer.dram_weights[args.role]
        k, n = w.shape
        host = (torch.randn(1, 1, args.batch, k) * 0.1).bfloat16()
        inp = ttnn.from_torch(host, device=mesh, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        # Same real weight reference with original BFP8/fidelity before changing precision.
        im, om, cfg = layer.dram_layouts[args.role]
        ref = ttnn.linear(
            ttnn.to_memory_config(inp, im),
            w,
            program_config=cfg,
            compute_kernel_config=layer.matmul_compute(args.role),
            memory_config=om,
        )
        expected = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(ref)]
        del ref
        if args.dtype != "bfloat8_b":
            w = ttnn.typecast(w, getattr(ttnn, args.dtype))
        original_weight = w
        padded_weights = {}
        bank_count = mesh.dram_grid_size().x
        bank_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(bank_count - 1, 0))})
        compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, args.fidelity),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        schedule = candidate_schedule(k, n, bank_count, layer.decode_geometry[args.role])
        for key in schedule:
            if (output.parent / "STOP").exists():
                raise SystemExit(75)
            inc, outc, readers, pipeline, block, shard_tiles = key
            if key in done:
                continue
            smaller_failure = next(
                (
                    r
                    for r in records
                    if tuple(r["key"][:4]) == key[:4]
                    and r.get("weight_shard_tiles") == shard_tiles
                    and r["key"][4] < block
                    and "beyond max L1 size" in r.get("error", "")
                ),
                None,
            )
            if smaller_failure:
                records.append(
                    dict(
                        key=key,
                        passed=False,
                        measured=False,
                        reason="Larger K block exceeds a measured L1 lower bound at identical geometry",
                        evidence_key=smaller_failure["key"],
                    )
                )
                output.write_text(json.dumps(records, indent=2) + "\n")
                continue
            # Preserve the measured shape adaptation and cache each padded
            # weight once, now lazily after resume/L1-bound checks.
            if shard_tiles not in padded_weights:
                memory = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.DRAM,
                    ttnn.ShardSpec(bank_grid, (k, shard_tiles * 32), ttnn.ShardOrientation.ROW_MAJOR),
                )
                padded_weights[shard_tiles] = ttnn.to_memory_config(
                    ttnn.to_memory_config(original_weight, ttnn.DRAM_MEMORY_CONFIG), memory
                )
            w = padded_weights[shard_tiles]
            kw = k // 32 // inc
            nw = math.ceil(n / (32 * outc))

            def mem(cores, width):
                return ttnn.create_sharded_memory_config(
                    (32, width),
                    ttnn.CoreGrid(x=8, y=cores // 8),
                    ttnn.ShardStrategy.WIDTH,
                    use_height_and_width_as_shard_shape=True,
                )

            im, om = mem(inc, kw * 32), mem(outc, nw * 32)
            cfg = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=block,
                per_core_M=1,
                per_core_N=nw,
                num_workers_per_dram_bank=readers,
                use_multicore_pipeline=pipeline,
            )
            row = dict(
                key=key,
                role=args.role,
                K=k,
                N=n,
                batch=args.batch,
                input_cores=inc,
                output_cores=outc,
                actual_output_cores=math.ceil(n / (32 * nw)),
                padded_M=32,
                in0_block_w=block,
                per_core_N=nw,
                readers=readers,
                pipeline=pipeline,
                dtype=args.dtype,
                weight_shard_tiles=shard_tiles,
                fidelity=args.fidelity,
                devices=[0, 1, 2, 3],
                regime="serial",
            )
            print("CANDIDATE", row, flush=True)
            x = ttnn.to_memory_config(inp, im)

            def forward():
                return ttnn.linear(x, w, program_config=cfg, compute_kernel_config=compute, memory_config=om)

            try:
                out = forward()
                ttnn.synchronize_device(mesh)
                row["pcc"] = min(pcc(a, ttnn.to_torch(b)) for a, b in zip(expected, ttnn.get_device_tensors(out)))
                del out
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                out = forward()
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                times = []
                for _ in range(3):
                    ttnn.synchronize_device(mesh)
                    begin = time.perf_counter()
                    for _ in range(50):
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    times.append((time.perf_counter() - begin) * 1000 / 50)
                row.update(ms=statistics.median(times), samples_ms=times, passed=row["pcc"] >= 0.995)
                ttnn.release_trace(mesh, trace)
                del out, x
            except RuntimeError as error:
                row.update(passed=False, error=str(error))
                records.append(row)
                output.write_text(json.dumps(records, indent=2) + "\n")
                raise
            records.append(row)
            output.write_text(json.dumps(records, indent=2) + "\n")
            print("RESULT", row["ms"], row["pcc"], flush=True)
        print("SWEEP_PASS", len(records), flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
