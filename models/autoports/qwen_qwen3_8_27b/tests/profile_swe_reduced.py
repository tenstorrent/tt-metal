# SPDX-License-Identifier: Apache-2.0
"""Reduced real layers0/3 plus full terminal; never an all64-layer timing."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from tracy import signpost

import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.generator import build_generator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/data/mvasiljev/weights/Qwen3.8-27B")
    parser.add_argument("--grouped", action="store_true")
    parser.add_argument("--masked-recurrence", action="store_true")
    parser.add_argument("--flat-prefill", action="store_true")
    parser.add_argument("--compact-b1-recurrence", action="store_true")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument(
        "--fixed-profile-domain",
        action="store_true",
        help="Warm only this fixed profiler workload; not general generator acceptance",
    )
    parser.add_argument("--active-requests", type=int)
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    active_requests = args.active_requests if args.active_requests is not None else args.batch
    if not 1 <= active_requests <= args.batch:
        parser.error("active-requests must be within 1..batch")
    assert not os.getenv("TT_METAL_WATCHER"), "Watcher and profiler require separate runs"
    os.environ["QWEN38_GROUPED_RECURRENCE"] = "1" if args.grouped else "0"
    os.environ["QWEN38_MASKED_RECURRENCE"] = "1" if args.masked_recurrence else "0"
    os.environ["QWEN38_FLAT_PREFILL"] = "1" if args.flat_prefill else "0"
    os.environ["QWEN38_COMPACT_B1_RECURRENCE"] = "1" if args.compact_b1_recurrence else "0"
    torch.set_num_threads(8)
    ttnn.CONFIG.throw_exception_on_fallback = True
    root = Path(__file__).resolve().parents[1]
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024**2)
    try:
        gen = build_generator(
            root,
            mesh,
            checkpoint=args.checkpoint,
            layer_indices=[0, 3],
            batch_size=args.batch,
            context_length=args.context,
            prepare=False,
            prefill_chunk_size=args.prefill_chunk_size,
        )
        gen.prepare_decode(prefill_lengths=[args.prefill_chunk_size] if args.fixed_profile_domain else None)
        prompts = [[100] * 4096 for _ in range(active_requests)]
        gen.generate(prompts, 3, stop_on_eos=False)
        gen.reset()
        active = torch.arange(args.batch) < active_requests
        gen.refresh(active.reshape(args.batch, 1, 1).float(), gen.active_mask)
        gen.model_trace = gen.decode_traces[active_requests < args.batch]
        values = torch.zeros(args.batch, 4096, dtype=torch.int64)
        values[:active_requests] = 100
        lengths = [4096] * active_requests + [0] * (args.batch - active_requests)
        ttnn.synchronize_device(mesh)
        ttnn.ReadDeviceProfiler(mesh)
        signpost("PERF_PREFILL")
        logits = gen.prefill_forward(
            values,
            page_table=gen.page_table,
            kv_cache=gen.kv_cache,
            prompt_lens=lengths,
            return_device=True,
        )
        ttnn.copy(ttnn.reshape(logits, [1, 1, args.batch, 248320 // 4]), gen.logits)
        ttnn.execute_trace(mesh, gen.sample_trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        signpost("PERF_PREFILL_END")
        ttnn.ReadDeviceProfiler(mesh)
        gen.refresh(torch.tensor([n if n else -1 for n in lengths], dtype=torch.int32), gen.positions)
        gen.refresh(torch.tensor(lengths, dtype=torch.int32).reshape(args.batch, 1), gen.rope_indices)
        gen.refresh(torch.zeros(1, dtype=torch.int32), gen.output_index)
        ttnn.synchronize_device(mesh)
        signpost("PERF_DECODE")
        ttnn.execute_trace(mesh, gen.model_trace, cq_id=0, blocking=False)
        ttnn.execute_trace(mesh, gen.buffered_sample_trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        signpost("PERF_DECODE_END")
        ttnn.ReadDeviceProfiler(mesh)
        positions = ttnn.to_torch(ttnn.get_device_tensors(gen.positions)[0]).reshape(-1).tolist()
        assert positions == [4097] * active_requests + [-1] * (args.batch - active_requests), positions
        result = dict(
            scope="Reduced real layers0/3 plus embedding/norm/head/sampling/feedback; not all64 layers",
            grouped=args.grouped,
            masked_recurrence=args.masked_recurrence,
            flat_prefill=args.flat_prefill,
            compact_b1_recurrence=args.compact_b1_recurrence,
            checkpoint=args.checkpoint,
            batch=args.batch,
            active_requests=active_requests,
            prompt=4096,
            decode_position=4096,
            physical_prefill_chunk=args.prefill_chunk_size,
            fixture_cache_context=args.context,
            fixed_profile_domain=args.fixed_profile_domain,
            preparation_scope=(
                "Only measured physical chunk" if args.fixed_profile_domain else "Complete generator domain"
            ),
            native_context=262144,
            history="synthetic token100 prompt, then autoregressive first sampled token",
            measured_decode_replays=1,
            positions_after=positions,
            trace_ids=[str(gen.model_trace), str(gen.buffered_sample_trace)],
            precision=gen.model.precision_config,
            source_sha256={
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [
                    Path(__file__),
                    *root.glob("tt/*.py"),
                    root / "tt/selected_precision_config.json",
                    *root.glob("tt/kernels/*.cpp"),
                ]
            },
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
