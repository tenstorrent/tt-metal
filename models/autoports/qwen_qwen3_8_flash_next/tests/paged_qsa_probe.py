# SPDX-License-Identifier: Apache-2.0
"""Real-weight paged-vs-dense QSA control; run only under serialized device ownership.

This is a layer-level reduced inner-loop check, never final serving evidence.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.checkpoint import CheckpointState
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import MeshOps, ParallelQSA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--poison-workspace", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    checkpoint = Path(
        os.environ.get(
            "TT_QSA_CHECKPOINT",
            "/mnt/models/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/de4b8e4d43b917e7706784d8bb445c9af86a3540",
        )
    )
    cfg = Qwen4ExpConfig.from_pretrained(checkpoint, local_files_only=True).text_config
    cfg.max_position_embeddings = args.context
    state = CheckpointState(checkpoint, "model.language_model.layers.3.self_attn.")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    traces, rows = [], []
    try:
        ops = MeshOps(mesh)
        dense = ParallelQSA.build(state, cfg, ops, 3, args.batch, args.batch)
        paged = ParallelQSA.build(state, cfg, ops, 3, args.batch, args.batch, attention_pool_tokens=args.batch * 512)
        poison = None
        if args.poison_workspace:
            # Invalid pooled-key columns must be masked independently of the
            # scratch buffer's previous contents, including nonfinite values.
            poison = ttnn.from_torch(
                torch.full(tuple(paged.pooled_keys.shape), float("nan"), dtype=torch.bfloat16),
                dtype=paged.pooled_keys.dtype,
                layout=paged.pooled_keys.layout,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        def poison_workspace():
            if poison is not None:
                ttnn.copy_host_to_device_tensor(poison, paged.pooled_keys)

        table = torch.full((args.batch, args.context // 32), -1, dtype=torch.int32)
        for lane in range(args.batch):
            table[lane, :5] = torch.tensor([7, 1, 12, 0, 15]) + lane * 16
        slots = torch.zeros(args.batch, 32, dtype=torch.int32)
        slots[:, 0] = torch.arange(args.batch)
        positions = torch.zeros(args.batch, 32, dtype=torch.int32)
        page = ops.upload(table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        sl = ops.upload(slots, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        pos = ops.upload(positions, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        rng = torch.Generator().manual_seed(149)
        hidden = torch.randn(1, args.batch, 140, 2560, generator=rng).bfloat16() * 0.05

        def refresh(tensor, data, dtype):
            host = ttnn.from_torch(
                data, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
            )
            ttnn.copy_host_to_device_tensor(host, tensor)

        def snapshot(value):
            return [ttnn.to_torch(t).float().flatten() for t in ttnn.get_device_tensors(value)]

        def compare(a, b, label):
            maxima, pccs = [], []
            a = a if isinstance(a, list) else snapshot(a)
            b = b if isinstance(b, list) else snapshot(b)
            for left, right in zip(a, b):
                assert torch.isfinite(left).all() and torch.isfinite(right).all()
                maxima.append((left - right).abs().max().item())
                pccs.append(torch.corrcoef(torch.stack((left, right)))[0, 1].item())
            row = dict(
                label=label,
                pcc_min=min(pccs),
                max_abs=max(maxima),
                positions=positions[:, 0].tolist(),
                pages=table[:, :5].tolist(),
                slots=slots[:, 0].tolist(),
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
            assert min(pccs) > 0.9999, row

        start = 0
        order = torch.arange(args.batch)
        # Every pair 3/4,31/32,127/128 is reached by a completed prefix,
        # then a singleton continuation. This covers incomplete group reuse.
        for size in (3, 1, 1, 26, 1, 1, 94, 1, 1):
            positions[:, 0] = start
            refresh(pos, positions, ttnn.uint32)
            x = ops.upload(hidden[:, :, start : start + size])
            left = dense.prefill_forward(x, current_pos=pos, page_table=page, slots=sl, chunk_size=min(size, 128))
            poison_workspace()
            right = paged.prefill_forward(x, current_pos=pos, page_table=page, slots=sl, chunk_size=min(size, 128))
            compare(left, right, f"prefill_start{start}_length{size}")
            start += size
        if args.batch > 1:
            order = torch.roll(order, 1)
            table = table[order]
            slots[:, 0] = order
            refresh(page, table, ttnn.int32)
            refresh(sl, slots, ttnn.uint32)
        decode_x = ops.upload(hidden[:, order, start : start + 1])
        positions[:, 0] = start
        refresh(pos, positions, ttnn.uint32)

        def decode(model):
            if model is paged:
                poison_workspace()
            return model.decode_forward(decode_x, current_pos=pos, page_table=page, slots=sl)

        # Both models' decode programs are prepared before either trace exists.
        compare(decode(dense), decode(paged), "reordered_decode_warm")
        if args.trace:
            outputs = []
            for model in (dense, paged):
                # Host writes belong before capture, never in the device graph.
                if model is paged:
                    poison_workspace()
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                outputs.append(model.decode_forward(decode_x, current_pos=pos, page_table=page, slots=sl))
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                traces.append(trace)
            from ttnn.tools import trace_allocation_tracker

            for output in outputs:
                # Captured outputs have no persistent aliases. Snapshot each
                # immediately after its producing trace before the other trace
                # may clobber transient addresses; next replay defines it again.
                trace_allocation_tracker.acknowledge_corruptible(output)
            for step in range(1, 4):
                positions[:, 0] = start + step
                refresh(pos, positions, ttnn.uint32)
                # Changed token and position payloads reach identical traces.
                host = ttnn.from_torch(
                    hidden[:, order, start + step : start + step + 1],
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                )
                ttnn.copy_host_to_device_tensor(host, decode_x)
                snapshots = []
                for trace, output in zip(traces, outputs):
                    if trace == traces[1]:
                        poison_workspace()
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                    snapshots.append(snapshot(output))
                compare(*snapshots, f"trace_replay{step}")
        else:
            for step in range(1, 4):
                positions[:, 0] = start + step
                refresh(pos, positions, ttnn.uint32)
                decode_x = ops.upload(hidden[:, order, start + step : start + step + 1])
                compare(decode(dense), decode(paged), f"reordered_decode{step}")
        payload = dict(
            status="pass",
            batch=args.batch,
            context=args.context,
            physical_tokens=args.batch * 512,
            real_weights=str(checkpoint),
            layer=3,
            cases=rows,
            trace_ids=[str(t) for t in traces],
            trace_alloc_tracking=os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING"),
            poisoned_workspace=args.poison_workspace,
            pooled_shapes=dict(cache=list(paged.pooled_cache.shape), workspace=list(paged.pooled_keys.shape)),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))
    finally:
        for trace in traces:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
