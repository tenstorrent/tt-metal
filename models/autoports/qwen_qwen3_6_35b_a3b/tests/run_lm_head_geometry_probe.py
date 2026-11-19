# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""One isolated LM-head geometry candidate; launch under an external timeout.

Uses real reduced layers 0/3 and an AIME-token prefill activation. No production
policy changes. Timings include terminal norm, both casts, head and reshards.
Run each candidate in a fresh process, with profiler/watcher/tracking disabled
for ranking. Tracking may be enabled separately for correctness evidence.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tt.model import QwenModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--readers", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--cores", type=int, default=16)
    parser.add_argument("--per-core-n", type=int, default=32)
    parser.add_argument("--splits", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument(
        "--resident-l1-bytes",
        type=int,
        default=0,
        help="Approximate target persistent allocation per L1 bank; add ballast before terminal warmup",
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--prompt-len", type=int, default=128)
    args = parser.parse_args()
    if args.cores <= 0 or 64 % args.cores or args.block <= 0 or 64 % args.block:
        parser.error("cores and block must be positive divisors of tiled K=64")
    if args.per_core_n <= 0 or args.iterations < 1 or not 1 <= args.prompt_len <= 161:
        parser.error("invalid output shard, iteration count, or prompt length")
    if args.resident_l1_bytes < 0:
        parser.error("resident L1 target must be nonnegative")
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "passed": False,
        "command": sys.argv,
        "candidate": vars(args),
        "regime": "serial 1x4 Blackhole, reduced real layers0/3, terminal-only",
        "policy": "BF16 activations/output, BFP8 head, HiFi2 fp32_dest_acc_en=True",
        "environment": {
            k: os.environ.get(k)
            for k in (
                "TT_METAL_TRACE_ALLOC_TRACKING",
                "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE",
                "TT_METAL_DEVICE_PROFILER",
                "TT_METAL_WATCHER",
            )
        },
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), root / "tt/model.py", root / "tt/multichip_decoder.py")
        },
    }
    mesh = None
    traces = []
    try:
        torch.set_num_threads(8)
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000000)
        model = QwenModel(mesh, layer_indices=(0, 3), max_seq_len=512, lm_head_mode="interleaved")
        cache = model.allocate_cache(1)
        ids = json.loads((root / "doc/full_model/hf_reference/aime24_prompt_tokens.json").read_text())
        tokens = model.upload(
            torch.tensor([ids[: args.prompt_len]], dtype=torch.int32), ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        original = model.terminal
        # Retain the actual decoder output, avoiding reconstruction of activations.
        model.terminal = lambda hidden: hidden
        try:
            hidden = model.prefill(tokens, cache=cache, return_all_logits=False)
        finally:
            model.terminal = original
        k, n = tuple(model.head_weight.shape)[-2:]
        banks = mesh.dram_grid_size().x
        grid = mesh.compute_with_storage_grid_size()
        if n % (32 * args.splits):
            raise ValueError("Each logical vocabulary piece must be tile aligned")
        piece_n = n // args.splits
        output_cores = math.ceil(piece_n / (32 * args.per_core_n))
        if output_cores > grid.x * grid.y:
            raise ValueError(f"Output needs {output_cores} cores; available {grid.x * grid.y}")
        # Storage padding does not change logical N or global vocabulary offsets.
        # Use readers*bank divisibility even though the pipeline supports ragged N.
        bank_width = math.ceil(piece_n / (32 * banks * args.readers)) * 32 * args.readers
        dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
        weight_memory = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(dram_grid, (k, bank_width), ttnn.ShardOrientation.ROW_MAJOR),
        )
        weights = []
        for split in range(args.splits):
            # Slice each existing device's contiguous vocabulary shard locally.
            # Concatenation after projection restores that same device-local order.
            piece = (
                model.head_weight
                if args.splits == 1
                else ttnn.slice(model.head_weight, (0, split * piece_n), (k, (split + 1) * piece_n))
            )
            weights.append(ttnn.to_memory_config(piece, weight_memory))
            del piece

        before = ttnn.get_memory_view(mesh, ttnn.BufferType.L1)
        allocated = int(before.total_bytes_allocated_per_bank)
        ballast = None
        if args.resident_l1_bytes > allocated:
            per_core_bytes = math.ceil((args.resident_l1_bytes - allocated) / 32) * 32
            count = grid.x * grid.y
            # Row-major uint32 width shards give each worker a resident allocation.
            # This approximates capacity pressure, not the full model's exact addresses.
            ballast_mem = ttnn.create_sharded_memory_config(
                (1, per_core_bytes // 4),
                ttnn.num_cores_to_corerangeset(count, grid, row_wise=True),
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            ballast = ttnn.empty(
                (1, count * per_core_bytes // 4),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                memory_config=ballast_mem,
            )
        after = ttnn.get_memory_view(mesh, ttnn.BufferType.L1)
        record["resident_l1"] = {
            "target_bytes_per_bank": args.resident_l1_bytes,
            "before_bytes_per_bank": allocated,
            "after_bytes_per_bank": int(after.total_bytes_allocated_per_bank),
            "ballast_allocated": ballast is not None,
            "limitation": "Capacity-pressure proxy only; does not reproduce full-stack per-core allocation addresses",
        }

        def memory(count, width):
            return ttnn.create_sharded_memory_config(
                (32, width),
                ttnn.num_cores_to_corerangeset(count, grid, row_wise=True),
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

        input_memory = memory(args.cores, k // args.cores)
        output_memory = memory(output_cores, 32 * args.per_core_n)
        pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=args.block,
            per_core_M=1,
            per_core_N=args.per_core_n,
            fused_activation=None,
            num_workers_per_dram_bank=args.readers,
            use_multicore_pipeline=args.readers > 1,
        )

        def candidate():
            x = ttnn.to_memory_config(hidden, ttnn.DRAM_MEMORY_CONFIG)
            x = ttnn.rms_norm(
                ttnn.typecast(x, ttnn.float32),
                weight=model.norm_weight,
                epsilon=model.config.rms_norm_eps,
                compute_kernel_config=model.compute,
            )
            x = ttnn.to_memory_config(ttnn.typecast(x, ttnn.bfloat16), input_memory)
            outputs = []
            for weight in weights:
                y = ttnn.linear(
                    x,
                    weight,
                    compute_kernel_config=model.compute,
                    program_config=pc,
                    dtype=ttnn.bfloat16,
                    memory_config=output_memory,
                )
                outputs.append(ttnn.to_memory_config(y, ttnn.DRAM_MEMORY_CONFIG))
            return outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=-1)

        forwards = {"baseline": lambda: original(hidden), "candidate": candidate}
        destinations = {}
        host = {}
        # Complete compilation and all persistent output allocation before capture.
        for name, forward in forwards.items():
            value = forward()
            destinations[name] = ttnn.empty_like(value)
            ttnn.copy(value, destinations[name])
            host[name] = model.host_logits(value).flatten()
            del value
        pcc = torch.corrcoef(torch.stack([host["baseline"], host["candidate"]]))[0, 1].item()
        top1_equal = host["baseline"].argmax().item() == host["candidate"].argmax().item()
        record["correctness"] = {
            "full_vocab_pcc": pcc,
            "top1_equal": top1_equal,
            "baseline_top1": host["baseline"].argmax().item(),
            "candidate_top1": host["candidate"].argmax().item(),
            "max_abs_error": (host["baseline"] - host["candidate"]).abs().max().item(),
        }
        record["geometry"] = {
            "k": k,
            "local_n": n,
            "splits": args.splits,
            "piece_logical_n": piece_n,
            "piece_ranges": [[i * piece_n, (i + 1) * piece_n] for i in range(args.splits)],
            "banks": banks,
            "bank_storage_width": bank_width,
            "input_cores": args.cores,
            "output_cores": output_cores,
            "vocab_offsets_unchanged": True,
        }
        ttnn.synchronize_device(mesh)
        named_traces = {}
        for name, forward in forwards.items():
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            try:
                ttnn.copy(forward(), destinations[name])
            finally:
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                traces.append(trace)
                named_traces[name] = trace
        timings = {}
        for name, trace in named_traces.items():
            for _ in range(3):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            begin = time.perf_counter()
            for _ in range(args.iterations):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            timings[name] = (time.perf_counter() - begin) * 1000 / args.iterations
            replayed = model.host_logits(destinations[name]).flatten()
            if not torch.equal(replayed, host[name]):
                raise AssertionError(f"{name}: trace result differs from eager result")
        record["terminal_ms"] = timings
        record["timing_boundary"] = "norm, casts, head, reshards, persistent output copy; nonblocking queued replay"
        record["speedup"] = timings["baseline"] / timings["candidate"]
        record["passed"] = pcc >= 0.995 and top1_equal
        if not record["passed"]:
            raise AssertionError("candidate terminal correctness failed")
    except BaseException:
        record["error"] = traceback.format_exc()
        raise
    finally:
        output.write_text(json.dumps(record, indent=2) + "\n")
        if mesh is not None:
            for trace in traces:
                ttnn.release_trace(mesh, trace)
            ttnn.close_mesh_device(mesh)
    print(
        "LM_HEAD_GEOMETRY",
        json.dumps({k: record[k] for k in ("passed", "correctness", "terminal_ms", "speedup")}),
        flush=True,
    )


if __name__ == "__main__":
    main()
