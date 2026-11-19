# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Precision-locked LM-head geometry on real reduced-stack decode activations."""

import argparse
import gc
import json
import math
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.model import OrnithModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--iterations", type=int, default=40)
    p.add_argument("--splits", type=int, default=4)
    p.add_argument("--readers", default="1,2,3,4")
    args = p.parse_args()
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
    result = {"status": "running", "policy": "BF16 activation/output, BF8 weight, HiFi2 FP32 accumulation", "rows": []}
    try:
        model = OrnithModel(mesh, layer_indices=[0, 3], decode_head="interleaved")
        cache = model.allocate_cache(batch=1, context=512)
        table = model.upload(torch.arange(16, dtype=torch.int32)[None], dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ids = model.upload(torch.arange(100, 133)[None], dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        model.prefill(ids, cache=cache, slot=0, start=0, page_table=table)
        original_logits = model.logits
        recorded = []

        def record(hidden, *, decode=False):
            if decode:
                recorded.append(hidden)
            return original_logits(hidden, decode=decode)

        model.logits = record
        token = model.upload(torch.tensor([[234]]), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        pos = model.upload(torch.tensor([33]), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        model.decode(token, cache=cache, slots=[0], positions=[pos], page_tables=[table])
        hidden = recorded[0]
        x = ttnn.rms_norm(
            hidden,
            weight=model.norm,
            epsilon=model.config.rms_norm_eps,
            program_config=model.layers[0].norm_program,
            compute_kernel_config=model.compute,
            memory_config=model.collectives.ar_mem,
        )
        n = model.vocab_size // 4
        banks = mesh.dram_grid_size().x
        grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))])
        width = n // args.splits
        assert n % args.splits == 0 and width % 32 == 0
        weights = []
        for i in range(args.splits):
            part = ttnn.slice(model.head, (0, i * width), (4096, (i + 1) * width))
            wm = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(grid, (4096, math.ceil(width / (32 * banks)) * 32), ttnn.ShardOrientation.ROW_MAJOR),
            )
            weights.append(ttnn.to_memory_config(part, wm))
        result["splits"] = args.splits
        baseline = lambda: ttnn.linear(
            ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG),
            model.head,
            compute_kernel_config=model.compute,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        reference = model.host_logits(baseline())
        result["reference_top1"] = int(reference[0, 0, 0].argmax())
        candidates = [("interleaved", baseline, {})]
        # Input stays on the selected residual grid. Larger K blocks gather
        # adjacent four-tile input shards, supported by this checkout's factory.
        for readers in map(int, args.readers.split(",")):
            for block in (2, 4, 8, 16):
                for cores in (64, 110):
                    ow = math.ceil(width / (cores * 32))
                    pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                        in0_block_w=block,
                        per_core_M=1,
                        per_core_N=ow,
                        num_workers_per_dram_bank=readers,
                        use_multicore_pipeline=readers > 1,
                    )

                    def run(pc=pc):
                        outputs = []
                        for weight in weights:
                            y = ttnn.linear(
                                x,
                                weight,
                                program_config=pc,
                                compute_kernel_config=model.compute,
                                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                                dtype=ttnn.bfloat16,
                            )
                            outputs.append(ttnn.to_memory_config(y, ttnn.DRAM_MEMORY_CONFIG))
                        return ttnn.concat(outputs, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

                    candidates.append(
                        (
                            f"dram_r{readers}_k{block}_c{cores}",
                            run,
                            {"readers": readers, "block": block, "output_cores_target": cores, "per_core_N": ow},
                        )
                    )
        for name, run, cfg in candidates:
            row = {"id": name, **cfg}
            trace = None
            y = out = None
            try:
                y = run()
                actual = model.host_logits(y)
                row["pcc"] = float(torch.corrcoef(torch.stack([reference.flatten(), actual.flatten()]))[0, 1])
                row["max_abs_error"] = float((reference - actual).abs().max())
                row["top1"] = int(actual[0, 0, 0].argmax())
                assert row["pcc"] >= 0.999 and row["top1"] == result["reference_top1"]
                ttnn.synchronize_device(mesh)
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                out = run()
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                for _ in range(4):
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                begin = time.perf_counter()
                for _ in range(args.iterations):
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                row.update(status="pass", ms=(time.perf_counter() - begin) * 1000 / args.iterations)
            except Exception as error:
                row.update(status="failed", error=str(error)[:1800])
                if not any(message in str(error) for message in ("beyond max L1 size", "clash with L1 buffers")):
                    raise
                row["status"] = "compile_l1_limit"
            finally:
                if trace is not None:
                    ttnn.release_trace(mesh, trace)
                result["rows"].append(row)
                Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
                print("HEAD_CANDIDATE", json.dumps(row), flush=True)
            del y, out
            gc.collect()
        result["status"] = "pass"
    finally:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
