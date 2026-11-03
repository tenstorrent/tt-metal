# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TT baseline vs TP4 at advertised capacity under full-stack memory reservation.

Independent HF K/V seeds long prefixes; compares traced final-position decode and
nonaligned continuation prefill to OptimizedDecoder. This is not full prefill.
"""

import argparse
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path

import torch
from transformers.cache_utils import DynamicCache

import ttnn

from ..tt.multichip_decoder import MultichipDecoder
from ..tt.optimized_decoder import OptimizedDecoder
from .fused_reference import ROOT, pcc, reference
from .multichip_activations import recorded_inputs
from .multichip_test_utils import device_only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--capacity", type=int, default=524288)
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--reserve-stack", action="store_true")
    ap.add_argument("--cache-dtype", default="bfloat8_b")
    ap.add_argument("--options", type=json.loads, default={})
    ap.add_argument("--artifact-dir", type=Path)
    args = ap.parse_args()
    torch.set_num_threads(16)
    torch.manual_seed(812)
    c, hf, rope, sd = reference(args.layer)
    capacity = args.capacity
    blocks = math.ceil(capacity / 32)
    pattern = recorded_inputs(args.layer, 1, 256)
    mod = importlib.import_module(type(hf).__module__)
    with torch.no_grad():
        norm = hf.input_layernorm(pattern)
        kbase = hf.self_attn.k_proj(norm).reshape(1, 256, 8, 128).transpose(1, 2)
        sample = DynamicCache(config=c)
        hf.self_attn(
            norm,
            position_embeddings=rope(pattern, torch.arange(256)[None]),
            attention_mask=None,
            past_key_values=sample,
        )
        cos, sin = rope(pattern, torch.arange(blocks * 32)[None])
        keys = kbase.repeat(1, 1, math.ceil(blocks * 32 / 256), 1)[:, :, : blocks * 32]
        keys = keys * cos[:, None] + mod.rotate_half(keys) * sin[:, None]
        values = sample.layers[args.layer].values.repeat(1, 1, math.ceil(blocks * 32 / 256), 1)[:, :, : blocks * 32]
    baseline_root = ROOT / "doc/multichip_decoder"
    root = args.artifact_dir or baseline_root
    root.mkdir(parents=True, exist_ok=True)
    path = baseline_root / f"local/capacity_baseline_l{args.layer}_{capacity}_{args.cache_dtype}.pt"
    golden = {} if args.baseline else torch.load(path, weights_only=True)
    if not args.baseline:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 1 if args.baseline else 4),
        physical_device_ids=[0] if args.baseline else [],
        trace_region_size=32 * 1024 * 1024,
    )
    tid = None
    try:
        ttnn.CONFIG.throw_exception_on_fallback = True
        sharded = not args.baseline and args.options.get("residual_layout") == "sharded"

        def tt(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, shard=None):
            return ttnn.from_torch(
                v.contiguous(),
                device=mesh,
                dtype=dtype,
                layout=layout,
                mesh_mapper=(
                    ttnn.ReplicateTensorToMesh(mesh) if shard is None else ttnn.ShardTensorToMesh(mesh, dim=shard)
                ),
            )

        def cpu(v):
            result = ttnn.to_torch(v, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=3 if sharded else 0))
            return result if args.baseline or sharded else result[:1]

        klass = OptimizedDecoder if args.baseline else MultichipDecoder
        model = klass.from_state_dict(sd, hf_config=c, layer_idx=args.layer, mesh_device=mesh, options=args.options)
        pages = torch.randperm(blocks)
        kv = []
        for source in (keys, values):
            logical = source[0].reshape(8, blocks, 32, 128).permute(1, 0, 2, 3)
            physical = torch.empty_like(logical)
            physical[pages] = logical
            kv.append(tt(physical, getattr(ttnn, args.cache_dtype), shard=None if args.baseline else 1))
        del keys, values, source, physical, logical
        table = tt(pages[None].int(), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        x = pattern[:, (capacity - 1) % 256 : (capacity - 1) % 256 + 1]
        dx = tt(x.reshape(1, 1, 1, 2560), shard=3 if sharded else None)
        dk = dict(
            cos=tt(cos[:, capacity - 1 : capacity].reshape(1, 1, 1, 128)),
            sin=tt(sin[:, capacity - 1 : capacity].reshape(1, 1, 1, 128)),
            page_table=table,
            current_pos=tt(torch.tensor([capacity - 1], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
            kv_cache=kv,
        )
        parts = []
        for length in (31, 33, 2049):
            if length > capacity:
                continue
            start = capacity - length
            inp = pattern.index_select(1, torch.arange(start, capacity) % 256)
            parts.append(
                (
                    length,
                    start,
                    tt(inp.unsqueeze(0), shard=3 if sharded else None),
                    tt(cos[:, start:capacity].unsqueeze(1)),
                    tt(sin[:, start:capacity].unsqueeze(1)),
                )
            )
        reservation = []
        if args.reserve_stack:
            assert not args.baseline
            plan = json.loads((root / "memory_capacity_plan.json").read_text())
            view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
            required = plan["per_device_total_bytes"] - view.total_bytes_allocated_per_bank * view.num_banks
            # Hold dummy allocations for other layers/embedding/head and the full
            # workspace allowance; no model computation is done on these buffers.
            while required > 0:
                size = min(required, 1024**3)
                rows = math.ceil(size / (2048 * 2 * 32)) * 32
                reservation.append(
                    ttnn.empty(
                        (1, 1, rows, 2048),
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=mesh,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                )
                required -= rows * 2048 * 2
        view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
        memory = {
            name: getattr(view, name)
            for name in (
                "num_banks",
                "total_bytes_per_bank",
                "total_bytes_allocated_per_bank",
                "total_bytes_free_per_bank",
            )
        }
        print("CAPACITY_MEMORY", memory, flush=True)
        scores = []
        # Prepare all eager variants before trace; record full output, not only final rows.
        for length, start, x, co, si in parts:
            with device_only():
                out = model.prefill_forward(x, cos=co, sin=si, page_table=table, kv_cache=kv, start_pos=start)
            actual = cpu(out).clone()
            out.deallocate(True)
            if args.baseline:
                golden[str(length)] = actual
            score = pcc(actual, golden[str(length)])
            assert score >= 0.995
            scores.append(dict(length=length, start=start, pcc=score))
            print("PREFILL", scores[-1], flush=True)
        for _ in range(3):
            with device_only():
                out = model.decode_forward(dx, **dk)
            actual = cpu(out).clone()
            out.deallocate(True)
        if args.baseline:
            golden["decode"] = actual
            torch.save(golden, path)
        score = pcc(actual, golden["decode"])
        assert score >= 0.995
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            with device_only():
                traced = model.decode_forward(dx, **dk)
        finally:
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
        replay = []
        for _ in range(5):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            now = cpu(traced)
            assert torch.equal(now, actual)
            replay.append(pcc(now, golden["decode"]))
        result = dict(
            layer=args.layer,
            capacity=capacity,
            decode_position=capacity - 1,
            baseline=args.baseline,
            prefill=scores,
            decode_pcc=score,
            trace_pcc=replay,
            stack_reservation=args.reserve_stack,
            memory=memory,
            trace_tracking=os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING", "0"),
            cache_dtype=args.cache_dtype,
            source_sha256=hashlib.sha256(Path(inspect.getfile(klass)).read_bytes()).hexdigest(),
        )
        (
            root / f'capacity_{"baseline" if args.baseline else "tp4"}_l{args.layer}_{capacity}_{args.cache_dtype}.json'
        ).write_text(json.dumps(result, indent=2) + "\n")
        print("PASS", result, flush=True)
    finally:
        if tid is not None:
            ttnn.release_trace(mesh, tid)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
