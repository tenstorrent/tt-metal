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
from models.autoports.google_gemma_4_26b_a4b_it.tests.create_optimized_long_reference import load_input_fixture
from models.autoports.google_gemma_4_26b_a4b_it.tests.run_decoder import MODEL, REVISION, load_layer
from models.autoports.google_gemma_4_26b_a4b_it.tests.runtime_audit import device_only
from models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder import FunctionalDecoder
from models.common.utility_functions import comp_pcc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-fixture", type=Path)
    parser.add_argument(
        "--heterogeneous-positions", action="store_true", help="Use prompt lengths and decode positions 32 + slot"
    )
    parser.add_argument("--decoder", choices=("functional", "fused"), default="functional")
    parser.add_argument("--fusion")
    parser.add_argument("--group-size", type=int)
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("--batch must be positive")
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
    batch, block = args.batch, 32
    lengths = [32 + slot for slot in range(batch)] if args.heterogeneous_positions else [33] * batch
    length = max(lengths)
    extent = (length + 1 + 127) // 128 * 128
    config = AutoConfig.from_pretrained(MODEL, revision=REVISION).text_config
    config._attn_implementation = "eager"
    hf = load_layer(config, args.layer, True)
    input_fixture = None
    if args.input_fixture:
        values, input_fixture = load_input_fixture(args.input_fixture, config, args.layer, sum(n + 1 for n in lengths))
        if args.heterogeneous_positions:
            x = torch.zeros(batch, length, config.hidden_size, dtype=values.dtype)
            decode_inputs, windows, start = [], [], 0
            for slot, prompt_length in enumerate(lengths):
                x[slot, :prompt_length] = values[0, start : start + prompt_length]
                decode_inputs.append(values[:, start + prompt_length : start + prompt_length + 1])
                windows.append(
                    dict(slot=slot, prefill_start=start, prefill_length=prompt_length, decode_row=start + prompt_length)
                )
                start += prompt_length + 1
            dx = torch.cat(decode_inputs, dim=0)
            input_fixture["windows"] = windows
        else:
            windows = values.reshape(batch, length + 1, config.hidden_size)
            x, dx = windows[:, :length].contiguous(), windows[:, length:].contiguous()
            input_fixture["windows"] = [
                dict(
                    slot=slot,
                    prefill_start=slot * (length + 1),
                    prefill_length=length,
                    decode_row=slot * (length + 1) + length,
                )
                for slot in range(batch)
            ]
        input_fixture[
            "position_policy"
        ] = "Each distinct recorded window is rebased to positions 0..prompt_length for HF and TT"
    else:
        x = torch.randn(batch, length, config.hidden_size).bfloat16().float()
        dx = torch.randn(batch, 1, config.hidden_size).bfloat16().float()
    cos, sin = Gemma4TextRotaryEmbedding(config)(
        x, torch.arange(extent)[None], layer_type=config.layer_types[args.layer]
    )
    with torch.no_grad():
        if args.heterogeneous_positions:
            reference, decode_refs = [], []
            for slot, prompt_length in enumerate(lengths):
                hf_cache = DynamicCache()
                mask = torch.zeros(prompt_length, prompt_length).masked_fill(
                    torch.triu(torch.ones(prompt_length, prompt_length, dtype=torch.bool), 1), float("-inf")
                )[None, None]
                reference.append(
                    hf(
                        x[slot : slot + 1, :prompt_length],
                        position_embeddings=(cos[:, :prompt_length], sin[:, :prompt_length]),
                        attention_mask=mask,
                        past_key_values=hf_cache,
                    )
                )
                decode_refs.append(
                    hf(
                        dx[slot : slot + 1],
                        position_embeddings=(
                            cos[:, prompt_length : prompt_length + 1],
                            sin[:, prompt_length : prompt_length + 1],
                        ),
                        attention_mask=torch.zeros(1, 1, 1, prompt_length + 1),
                        past_key_values=hf_cache,
                    )
                )
            decode_ref = torch.cat(decode_refs, dim=0)
        else:
            hf_cache = DynamicCache()
            mask = torch.zeros(length, length).masked_fill(
                torch.triu(torch.ones(length, length, dtype=torch.bool), 1), float("-inf")
            )[None, None]
            batched_reference = hf(
                x, position_embeddings=(cos[:, :length], sin[:, :length]), attention_mask=mask, past_key_values=hf_cache
            )
            reference = list(batched_reference.split(1, dim=0))
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
        cache = [
            device(
                torch.zeros(pages, cfg.num_key_value_heads, block, cfg.head_dim),
                getattr(layer, "kv_cache_dtype", ttnn.bfloat16),
            )
            for _ in range(2)
        ]
        ropes = tuple(device(t[None]) for t in (cos, sin))
        prefill_rows = []
        for slot in range(batch):
            inp = device(x[slot : slot + 1, : lengths[slot]][None])
            with device_only():
                out = layer.prefill_forward(inp, rope_mats=ropes, page_table=page_table, kv_cache=cache, user_id=slot)
            got = ttnn.to_torch(out).squeeze(0).float()
            passed, pcc = comp_pcc(reference[slot], got, 0.995)
            prefill_rows.append(dict(slot=slot, length=lengths[slot], pcc=float(pcc), passed=bool(passed)))
        dt = device(dx.transpose(0, 1)[None])
        p = torch.zeros(1, max(32, batch), dtype=torch.int32)
        cp = torch.tensor(lengths, dtype=torch.int32)
        p[0, :batch] = cp
        rp = device(p, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        dp = device(cp, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        decode_rope_layout = getattr(layer, "decode_rope_layout", ttnn.TILE_LAYOUT)
        dr = tuple(device(t.squeeze(0), layout=decode_rope_layout) for t in (cos, sin))

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
            decode_rows.append(dict(slot=slot, position=lengths[slot], pcc=float(pcc), passed=bool(passed)))
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
        repeated = torch.equal(ttnn.to_torch(out).reshape_as(got).float(), got)
        ttnn.release_trace(mesh, trace)
        result = dict(
            layer_type=config.layer_types[args.layer],
            batch=batch,
            length=lengths if args.heterogeneous_positions else length,
            lengths=lengths,
            heterogeneous_positions=args.heterogeneous_positions,
            real_weights=True,
            input_fixture=input_fixture,
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
