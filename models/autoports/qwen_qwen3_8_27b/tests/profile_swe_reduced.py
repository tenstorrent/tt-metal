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
    parser.add_argument("--grouped", action="store_true")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not os.getenv("TT_METAL_WATCHER"), "Watcher and profiler require separate runs"
    os.environ["QWEN38_GROUPED_RECURRENCE"] = "1" if args.grouped else "0"
    torch.set_num_threads(8)
    root = Path(__file__).resolve().parents[1]
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024**2)
    try:
        gen = build_generator(
            root,
            mesh,
            checkpoint="/data/mvasiljev/weights/Qwen3.8-27B",
            layer_indices=[0, 3],
            batch_size=args.batch,
            context_length=8192,
            prepare=False,
            prefill_chunk_size=256,
        )
        gen.prepare_decode()
        prompts = [[100] * 4096 for _ in range(args.batch)]
        gen.generate(prompts, 3, stop_on_eos=False)
        gen.reset()
        ttnn.synchronize_device(mesh)
        ttnn.ReadDeviceProfiler(mesh)
        signpost("PERF_PREFILL")
        logits = gen.prefill_forward(
            torch.tensor(prompts),
            page_table=gen.page_table,
            kv_cache=gen.kv_cache,
            prompt_lens=[4096] * args.batch,
            return_device=True,
        )
        ttnn.copy(ttnn.reshape(logits, [1, 1, args.batch, 248320 // 4]), gen.logits)
        ttnn.execute_trace(mesh, gen.sample_trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        signpost("PERF_PREFILL_END")
        ttnn.ReadDeviceProfiler(mesh)
        gen.refresh(torch.full((args.batch,), 4096, dtype=torch.int32), gen.positions)
        gen.refresh(torch.full((args.batch, 1), 4096, dtype=torch.int32), gen.rope_indices)
        gen.refresh(torch.zeros(1, dtype=torch.int32), gen.output_index)
        ttnn.synchronize_device(mesh)
        signpost("PERF_DECODE")
        ttnn.execute_trace(mesh, gen.model_trace, cq_id=0, blocking=False)
        ttnn.execute_trace(mesh, gen.buffered_sample_trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        signpost("PERF_DECODE_END")
        ttnn.ReadDeviceProfiler(mesh)
        positions = ttnn.to_torch(ttnn.get_device_tensors(gen.positions)[0]).reshape(-1).tolist()
        assert positions == [4097] * args.batch, positions
        result = dict(
            scope="Reduced real layers0/3 plus embedding/norm/head/sampling/feedback; not all64 layers",
            grouped=args.grouped,
            batch=args.batch,
            prompt=4096,
            decode_position=4096,
            physical_prefill_chunk=256,
            fixture_cache_context=8192,
            native_context=262144,
            history="synthetic token100 prompt, then autoregressive first sampled token",
            measured_decode_replays=1,
            positions_after=positions,
            trace_ids=[str(gen.model_trace), str(gen.buffered_sample_trace)],
            precision=gen.model.precision_config,
            source_sha256={
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [Path(__file__), *root.glob("tt/*.py"), *root.glob("tt/kernels/*grouped*")]
            },
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
