# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Paged request-slot isolation and traced batch decode at real model shapes."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests.run_decoder import MODEL, REVISION, load_layer
from models.autoports.google_gemma_4_26b_a4b_it.tests.runtime_audit import device_only
from models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder import FunctionalDecoder
from models.common.utility_functions import comp_pcc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decoder", choices=("functional", "fused"), default="functional")
    parser.add_argument("--fusion")
    parser.add_argument("--group-size", type=int)
    args = parser.parse_args()
    decoder_options = {}
    if args.decoder == "fused":
        decoder_options = {
            k: v for k, v in {"fusion": args.fusion, "group_size": args.group_size}.items() if v is not None
        }
    decoder_class = FunctionalDecoder
    if args.decoder == "fused":
        from models.autoports.google_gemma_4_26b_a4b_it.tt.fused_decoder import FusedDecoder

        decoder_class = FusedDecoder
    torch.manual_seed(42)
    torch.set_num_threads(8)
    batch, length, extent, block = args.batch, 33, 128, 32
    config = AutoConfig.from_pretrained(MODEL, revision=REVISION).text_config
    config._attn_implementation = "eager"
    hf = load_layer(config, args.layer, True)
    x = torch.randn(batch, length, config.hidden_size).bfloat16().float()
    dx = torch.randn(batch, 1, config.hidden_size).bfloat16().float()
    cos, sin = Gemma4TextRotaryEmbedding(config)(
        x, torch.arange(extent)[None], layer_type=config.layer_types[args.layer]
    )
    hf_cache = DynamicCache()
    mask = torch.zeros(length, length).masked_fill(
        torch.triu(torch.ones(length, length, dtype=torch.bool), 1), float("-inf")
    )[None, None]
    with torch.no_grad():
        reference = hf(
            x, position_embeddings=(cos[:, :length], sin[:, :length]), attention_mask=mask, past_key_values=hf_cache
        )
        decode_ref = hf(
            dx,
            position_embeddings=(cos[:, length : length + 1], sin[:, length : length + 1]),
            attention_mask=torch.zeros(batch, 1, 1, length + 1),
            past_key_values=hf_cache,
        )
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        layer = decoder_class.from_state_dict(
            hf.state_dict(), hf_config=config, layer_idx=args.layer, mesh_device=mesh, **decoder_options
        )

        def device(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(t, device=mesh, dtype=dtype, layout=layout)

        pages = batch * extent // block
        table = torch.randperm(pages).int().reshape(batch, -1)
        page_table = device(table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        cfg = layer.layer.self_attn.config
        cache = [device(torch.zeros(pages, cfg.num_key_value_heads, block, cfg.head_dim)) for _ in range(2)]
        ropes = tuple(device(t[None]) for t in (cos, sin))
        prefill_rows = []
        for slot in range(batch):
            inp = device(x[slot : slot + 1][None])
            with device_only():
                out = layer.prefill_forward(inp, rope_mats=ropes, page_table=page_table, kv_cache=cache, user_id=slot)
            got = ttnn.to_torch(out).squeeze(0).float()
            passed, pcc = comp_pcc(reference[slot : slot + 1], got, 0.995)
            prefill_rows.append(dict(slot=slot, pcc=float(pcc), passed=bool(passed)))
        dt = device(dx.transpose(0, 1)[None])
        p = torch.zeros(1, max(32, batch), dtype=torch.int32)
        p[0, :batch] = length
        cp = torch.full((batch,), length, dtype=torch.int32)
        rp = device(p, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        dp = device(cp, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        dr = tuple(device(t.squeeze(0)) for t in (cos, sin))

        def forward():
            return layer.decode_forward(
                dt, rope_mats=dr, current_pos=rp, cache_pos=dp, page_table=page_table, kv_cache=cache
            )

        warm = forward()
        ttnn.synchronize_device(mesh)
        warm.deallocate(True)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        with device_only():
            out = forward()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
        got = ttnn.to_torch(out).reshape(batch, 1, config.hidden_size).float()
        decode_rows = []
        for slot in range(batch):
            passed, pcc = comp_pcc(decode_ref[slot], got[slot], 0.995)
            decode_rows.append(dict(slot=slot, pcc=float(pcc), passed=bool(passed)))
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
        repeated = torch.equal(ttnn.to_torch(out).reshape_as(got).float(), got)
        ttnn.release_trace(mesh, trace)
        result = dict(
            layer_type=config.layer_types[args.layer],
            batch=batch,
            length=length,
            prefill=prefill_rows,
            decode=decode_rows,
            repeated_equal=repeated,
            traced=True,
            runtime_audit="clean",
            page_table="random permutation, disjoint pages per slot",
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(result, flush=True)
        assert repeated and all(r["passed"] for r in prefill_rows + decode_rows), result
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
