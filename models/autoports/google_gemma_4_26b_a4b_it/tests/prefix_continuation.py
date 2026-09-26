# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Partial-page continuations preserve prefix bytes and another request slot."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests.create_optimized_long_reference import load_input_fixture
from models.autoports.google_gemma_4_26b_a4b_it.tests.run_decoder import load_layer
from models.autoports.google_gemma_4_26b_a4b_it.tests.runtime_audit import device_only
from models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder import FunctionalDecoder
from models.common.utility_functions import comp_pcc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-fixture", type=Path)
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
    config = AutoConfig.from_pretrained(Path(__file__).parent).text_config
    config._attn_implementation = "eager"
    hf = load_layer(config, args.layer, True)
    length, extent, block, slot = 65, 128, 32, 1
    input_fixture = None
    if args.input_fixture:
        x, input_fixture = load_input_fixture(args.input_fixture, config, args.layer, length)
        input_fixture["position_policy"] = "Recorded positions 0..64, split across the original continuation boundaries"
    else:
        x = torch.randn(1, length, config.hidden_size).bfloat16().float()
    cos, sin = Gemma4TextRotaryEmbedding(config)(
        x, torch.arange(extent)[None], layer_type=config.layer_types[args.layer]
    )
    mask = torch.zeros(length, length).masked_fill(
        torch.triu(torch.ones(length, length, dtype=torch.bool), 1), float("-inf")
    )[None, None]
    with torch.no_grad():
        reference = hf(x, position_embeddings=(cos[:, :length], sin[:, :length]), attention_mask=mask)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        decoder = decoder_class.from_state_dict(
            hf.state_dict(), hf_config=config, layer_idx=args.layer, mesh_device=mesh, **decoder_options
        )

        def device(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(t, device=mesh, dtype=dtype, layout=layout)

        cfg = decoder.layer.self_attn.config
        pages = 2 * extent // block
        table = torch.randperm(pages).int().reshape(2, -1)
        page_table = device(table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        cache = [
            device(
                torch.zeros(pages, cfg.num_key_value_heads, block, cfg.head_dim),
                getattr(decoder, "kv_cache_dtype", ttnn.bfloat16),
            )
            for _ in range(2)
        ]
        ropes = tuple(device(t[None]) for t in (cos, sin))
        outputs, preservation = [], []

        def snapshot():
            return [ttnn.to_torch(t).clone() for t in cache]

        before = snapshot()
        for start, end in [(0, 31), (31, 33), (33, 65)]:
            inp = device(x[:, start:end][None])
            with device_only():
                out = decoder.prefill_forward(
                    inp, rope_mats=ropes, page_table=page_table, kv_cache=cache, user_id=slot, start_pos=start
                )
            outputs.append(ttnn.to_torch(out).squeeze(0).float())
            after = snapshot()
            unchanged_prefix, unchanged_other_slot = True, True
            for old, new in zip(before, after):
                unchanged_other_slot &= torch.equal(old[table[0].long()], new[table[0].long()])
                old_rows = (
                    old[table[slot].long()].permute(1, 0, 2, 3).reshape(cfg.num_key_value_heads, extent, cfg.head_dim)
                )
                new_rows = (
                    new[table[slot].long()].permute(1, 0, 2, 3).reshape(cfg.num_key_value_heads, extent, cfg.head_dim)
                )
                unchanged_prefix &= torch.equal(old_rows[:, :start], new_rows[:, :start])
            preservation.append(
                dict(start=start, end=end, prefix_unchanged=unchanged_prefix, other_slot_unchanged=unchanged_other_slot)
            )
            before = after
        passed, pcc = comp_pcc(reference, torch.cat(outputs, dim=1), 0.995)
        result = dict(
            layer_type=config.layer_types[args.layer],
            length=length,
            slot=slot,
            real_weights=True,
            input_fixture=input_fixture,
            pcc=float(pcc),
            passed=bool(passed),
            runtime_audit="clean",
            cache_preservation=preservation,
            page_table="random disjoint physical pages",
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(result, flush=True)
        assert passed and all(r["prefix_unchanged"] and r["other_slot_unchanged"] for r in preservation), result
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
