"""Nonzero request slots, unequal positions, live trace page-table remapping."""

import argparse
import json
from pathlib import Path

import torch
from transformers import DynamicCache

import ttnn

from ..tt.optimized_decoder import OptimizedDecoder
from .run_optimized import DOC, load_reference, pcc, refresh, to_device
from .runtime_audit import instrument
from .sweep_optimized import real_activations


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DOC / "positions.json"))
    parser.add_argument("--pages", type=int, default=4)
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()
    batch, slots = args.batch, args.batch + 1
    torch.set_num_threads(16)
    config, state, hf, hf_rope = load_reference()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    records = []
    try:
        layer = OptimizedDecoder.from_state_dict(state, hf_config=config, layer_idx=0, mesh_device=mesh)
        counts = instrument(layer)
        torch.manual_seed(915)
        table = torch.randperm(slots * args.pages).reshape(slots, args.pages).int()
        tt_table = to_device(table, mesh, True)
        caches = tuple(
            ttnn.zeros(
                (slots * args.pages, 8, 32, 128),
                dtype=layer.kv_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for _ in range(2)
        )
        x = real_activations(slots * 128).reshape(slots, 128, 4096)
        rope = hf_rope(x, torch.arange(128)[None].expand(slots, -1))
        ref = DynamicCache(config=config)
        hf(x, position_embeddings=rope, past_key_values=ref)
        tx = to_device(x.unsqueeze(0), mesh)
        tr = tuple(to_device(r.unsqueeze(1), mesh) for r in rope)
        out = layer.prefill_forward(
            tx, rope=tr, kv_cache=caches, page_table=tt_table, plan=layer.prepare_prefill(seq_len=128)
        )
        out.deallocate(True)
        untouched = [ttnn.to_torch(c)[table[0].long()].clone() for c in caches]
        positions = torch.tensor([31 + (34 * b) % 90 for b in range(batch)], dtype=torch.int32)
        request_rows = list(range(batch, 0, -1))
        references = []
        for row, pos in zip(request_rows, positions):
            cache = DynamicCache(config=config)
            cache.update(
                ref.layers[0].keys[row : row + 1, :, :pos, :], ref.layers[0].values[row : row + 1, :, :pos, :], 0
            )
            references.append(cache)
        decode_table = table[request_rows].contiguous()
        dt = to_device(decode_table, mesh, True)
        dx = x[:batch, 13:14].contiguous()
        dr = hf_rope(dx, positions[:, None].long())
        td = to_device(dx.transpose(0, 1).unsqueeze(0), mesh)
        rr = tuple(to_device(r.unsqueeze(0).repeat(1, 1, 32, 1), mesh) for r in dr)
        pp = to_device(positions, mesh, True)
        kwargs = dict(rope=rr, kv_cache=caches, page_table=dt, current_pos=pp)
        warm = layer.decode_forward(td, **kwargs)
        warm.deallocate(True)
        ttnn.synchronize_device(mesh)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        out = layer.decode_forward(td, **kwargs)
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        for step in range(3):
            if step:
                positions += 1
                if step == 2:
                    positions = positions.flip(0).contiguous()
                    decode_table = decode_table.flip(0).contiguous()
                    references.reverse()
                    refresh(dt, decode_table, True)
                dx = x[:batch, 13 + step : 14 + step].contiguous()
                dr = hf_rope(dx, positions[:, None].long())
                refresh(td, dx.transpose(0, 1).unsqueeze(0))
                refresh(pp, positions, True)
                for dest, source in zip(rr, dr):
                    refresh(dest, source.unsqueeze(0).repeat(1, 1, 32, 1))
            expected = torch.cat(
                [
                    hf(
                        dx[b : b + 1],
                        position_embeddings=tuple(r[b : b + 1] for r in dr),
                        past_key_values=references[b],
                    )
                    for b in range(batch)
                ]
            )
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            actual = ttnn.to_torch(out).reshape_as(expected)
            score = pcc(actual, expected)
            assert score >= 0.995, score
            records.append({"positions": positions.tolist(), "pcc": score, "page_table_changed": step == 2})
            print(records[-1], flush=True)
        ttnn.release_trace(mesh, trace)
        for before, cache in zip(untouched, caches):
            assert torch.equal(before, ttnn.to_torch(cache)[table[0].long()]), "Unowned request cache changed"
        Path(args.output).write_text(
            json.dumps(
                {
                    "records": records,
                    "page_capacity": args.pages * 32,
                    "batch": batch,
                    "unowned_pages_unchanged": True,
                    "runtime_audit_completed_calls": counts,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
