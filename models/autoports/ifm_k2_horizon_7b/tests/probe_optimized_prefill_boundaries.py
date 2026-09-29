"""Public layer checks across each prefill numerical-dispatch boundary."""

import argparse
import json
import math
from pathlib import Path

import torch
from transformers import DynamicCache

import ttnn

from ..tt.optimized_decoder import OptimizedDecoder
from .run_optimized import load_reference, pcc, to_device
from .runtime_audit import instrument
from .sweep_optimized import real_activations


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    args = p.parse_args()
    torch.set_num_threads(16)
    config, state, hf, rope_fn = load_reference()
    cases = [
        (8160, 31),
        (8160, 32),
        (8192, 32),
        (32640, 127),
        (32640, 128),
        (32768, 128),
        (65280, 255),
        (65280, 256),
        (65536, 256),
    ]
    length = 65792
    block = real_activations(4096)[None]
    kk = []
    vv = []
    for start in range(0, length, 4096):
        x = block[:, : min(4096, length - start)]
        norm = hf.input_layernorm(x)
        cos, sin = rope_fn(x, torch.arange(start, start + x.shape[1])[None])
        k = hf.self_attn.k_proj(norm).reshape(1, -1, 8, 128).transpose(1, 2)
        kk.append(k * cos[:, None] + torch.cat([-k[..., 64:], k[..., :64]], -1) * sin[:, None])
        vv.append(hf.self_attn.v_proj(norm).reshape(1, -1, 8, 128).transpose(1, 2))
    keys, values = torch.cat(kk, 2), torch.cat(vv, 2)
    del kk, vv, norm, k
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    rows = []
    try:
        layer = OptimizedDecoder.from_state_dict(state, hf_config=config, layer_idx=0, mesh_device=mesh)
        audit = instrument(layer)
        pages = length // 32
        table = torch.randperm(pages).int()[None]
        pt = to_device(table, mesh, True)
        physical = []
        for data in [keys, values]:
            h = torch.empty(pages, 8, 32, 128, dtype=torch.bfloat16)
            h[table[0].long()] = data.reshape(8, pages, 32, 128).permute(1, 0, 2, 3)
            physical.append(h)
        for start, count in cases:
            x = block[:, start % 4096 : start % 4096 + count].contiguous()
            rope = rope_fn(x, torch.arange(start, start + count)[None])
            ref = DynamicCache(config=config)
            ref.update(keys[:, :, :start], values[:, :, :start], 0)
            mask = torch.where(
                torch.arange(start + count)[None, :] <= torch.arange(start, start + count)[:, None], 0.0, float("-inf")
            ).bfloat16()[None, None]
            expected = hf(x, position_embeddings=rope, past_key_values=ref, attention_mask=mask)
            caches = tuple(
                ttnn.from_torch(t, device=mesh, dtype=layer.kv_dtype, layout=ttnn.TILE_LAYOUT) for t in physical
            )
            tx = to_device(x[None], mesh)
            tr = tuple(to_device(t[:, None], mesh) for t in rope)
            out = layer.prefill_forward(
                tx, rope=tr, kv_cache=caches, page_table=pt, plan=layer.prepare_prefill(seq_len=count, start_pos=start)
            )
            actual = ttnn.to_torch(out).reshape_as(expected)
            physical_count = math.ceil(count / 32) * 32
            chunk = 128 if physical_count % 128 == 0 and start % 128 == 0 else 32
            kchunk = 256 if physical_count >= 256 and start % 256 == 0 else chunk
            row = dict(
                start=start,
                count=count,
                physical=physical_count,
                k_chunk=kchunk,
                path="stock" if start + physical_count <= 256 * kchunk else "accurate",
                pcc=pcc(actual, expected),
                relative_l2=((actual.float() - expected.float()).norm() / expected.float().norm()).item(),
            )
            rows.append(row)
            print(row, flush=True)
            assert row["pcc"] >= 0.995
            for t in [out, tx, *tr, *caches]:
                t.deallocate(True)
            Path(args.output).write_text(
                json.dumps(
                    {
                        "completed": len(rows) == len(cases),
                        "scope": "real weights and checkpoint embeddings; HF-initialized prefix; public optimized prefill",
                        "records": rows,
                        "runtime_audit_completed_calls": audit,
                    },
                    indent=2,
                )
                + "\n"
            )
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
