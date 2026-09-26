# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Increasing/decreasing prompts and changing page ownership on one loaded layer."""

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
    parser.add_argument("--layer", type=int, default=0)
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
    torch.manual_seed(19)
    torch.set_num_threads(8)
    cfg = AutoConfig.from_pretrained(MODEL, revision=REVISION).text_config
    cfg._attn_implementation = "sdpa"
    hf = load_layer(cfg, args.layer, True)
    kind = cfg.layer_types[args.layer]
    extent = 4096
    lengths = [31, 32, 33, 1023, 1024, 1025, 2049, 33, 2047]
    input_fixture = None
    if args.input_fixture:
        windows = [
            dict(request=index, prefill_start=index * 128, length=length) for index, length in enumerate(lengths)
        ]
        used_length = max(window["prefill_start"] + window["length"] + 1 for window in windows)
        values, input_fixture = load_input_fixture(args.input_fixture, cfg, args.layer, used_length)
        input_fixture["windows"] = [
            {**window, "decode_row": window["prefill_start"] + window["length"]} for window in windows
        ]
        input_fixture["position_policy"] = "Each recorded window is rebased to positions 0..length for HF and TT"
    cos, sin = Gemma4TextRotaryEmbedding(cfg)(
        torch.zeros(1, 1, cfg.hidden_size), torch.arange(extent)[None], layer_type=kind
    )
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    results = []
    try:
        layer = decoder_class.from_state_dict(
            hf.state_dict(), hf_config=cfg, layer_idx=args.layer, mesh_device=mesh, **decoder_options
        )

        def device(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(t, device=mesh, dtype=dtype, layout=layout)

        def refresh(t, dest, dtype, layout):
            host = ttnn.from_torch(t, dtype=dtype, layout=layout)
            ttnn.copy_host_to_device_tensor(host, dest)

        ac = layer.layer.self_attn.config
        cache = [
            device(
                torch.zeros(extent // 32, ac.num_key_value_heads, 32, ac.head_dim),
                getattr(layer, "kv_cache_dtype", ttnn.bfloat16),
            )
            for _ in range(2)
        ]
        pt = device(torch.arange(extent // 32, dtype=torch.int32)[None], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        rt = tuple(device(t[None]) for t in (cos, sin))
        decode_rope_layout = getattr(layer, "decode_rope_layout", ttnn.TILE_LAYOUT)
        dr = tuple(device(t.squeeze(0), layout=decode_rope_layout) for t in (cos, sin))
        dt = device(torch.zeros(1, 1, 1, cfg.hidden_size))
        p = torch.zeros(1, 32, dtype=torch.int32)
        cp = torch.zeros(1, dtype=torch.int32)
        rp = device(p, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        dp = device(cp, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        # Program binaries introduced after capture can outlive the prefill call.
        # Initialize this test's exact catalog before keeping a decode trace alive.
        prefill_warmup = []
        for request_index, length in enumerate(lengths):
            if input_fixture is not None:
                start = input_fixture["windows"][request_index]["prefill_start"]
                warm_input = values[:, start : start + length].contiguous()
            else:
                warm_input = torch.zeros(1, length, cfg.hidden_size)
            entries_before = mesh.num_program_cache_entries()
            warm_xt = device(warm_input[None])
            with device_only():
                warm_out = layer.prefill_forward(warm_xt, rope_mats=rt, page_table=pt, kv_cache=cache)
            warm_out.deallocate(True)
            warm_xt.deallocate(True)
            prefill_warmup.append(
                dict(
                    request=request_index,
                    length=length,
                    entries_before=entries_before,
                    entries_after=mesh.num_program_cache_entries(),
                )
            )
        ttnn.synchronize_device(mesh)
        # The validated requests below refill their own pages and refresh inputs.
        trace = None
        captured_program_entries = None
        for request_index, length in enumerate(lengths):
            if input_fixture is not None:
                start = input_fixture["windows"][request_index]["prefill_start"]
                x = values[:, start : start + length].contiguous()
                dx = values[:, start + length : start + length + 1].contiguous()
            else:
                x = torch.randn(1, length, cfg.hidden_size).bfloat16().float()
                dx = torch.randn(1, 1, cfg.hidden_size).bfloat16().float()
            idx = torch.arange(length)
            allowed = idx[:, None] >= idx[None, :]
            if kind == "sliding_attention":
                allowed &= idx[:, None] - idx[None, :] < cfg.sliding_window
            mask = torch.zeros(length, length).masked_fill(~allowed, float("-inf"))[None, None]
            hc = DynamicCache()
            with torch.no_grad():
                ref = hf(
                    x, position_embeddings=(cos[:, :length], sin[:, :length]), attention_mask=mask, past_key_values=hc
                )
                dm = torch.zeros(1, 1, 1, length + 1)
                if kind == "sliding_attention":
                    dm[..., : max(0, length + 1 - cfg.sliding_window)] = float("-inf")
                dref = hf(
                    dx,
                    position_embeddings=(cos[:, length : length + 1], sin[:, length : length + 1]),
                    attention_mask=dm,
                    past_key_values=hc,
                )
            new_table = torch.randperm(extent // 32).int()[None]
            refresh(new_table, pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            xt = device(x[None])
            with device_only():
                out = layer.prefill_forward(xt, rope_mats=rt, page_table=pt, kv_cache=cache)
            actual = ttnn.to_torch(out).squeeze(0).float()
            passed, pcc = comp_pcc(ref, actual, 0.995)
            out.deallocate(True)
            xt.deallocate(True)
            p[0, 0], cp[0] = length, length
            refresh(dx[None], dt, ttnn.bfloat16, ttnn.TILE_LAYOUT)
            refresh(p, rp, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            refresh(cp, dp, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)

            def decode():
                return layer.decode_forward(
                    dt, rope_mats=dr, current_pos=rp, cache_pos=dp, page_table=pt, kv_cache=cache
                )

            if trace is None:
                warm = decode()
                ttnn.synchronize_device(mesh)
                warm.deallocate(True)
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                with device_only():
                    traced = decode()
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                captured_program_entries = mesh.num_program_cache_entries()
                mesh.set_program_cache_misses_allowed(False)
            assert mesh.num_program_cache_entries() == captured_program_entries
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            got = ttnn.to_torch(traced).squeeze(0).float()
            dpass, dpcc = comp_pcc(dref, got, 0.995)
            row = dict(length=length, prefill_pcc=float(pcc), decode_pcc=float(dpcc), passed=bool(passed and dpass))
            results.append(row)
            args.output.write_text(
                json.dumps(
                    dict(
                        layer_type=kind,
                        real_weights=True,
                        input_fixture=input_fixture,
                        requests=results,
                        runtime_audit="clean",
                        trace_reused=True,
                        prefill_warmup=prefill_warmup,
                        program_cache_entries_at_capture=captured_program_entries,
                        program_cache_entries=mesh.num_program_cache_entries(),
                        program_cache_misses_while_trace_live="forbidden",
                        trace_allocation_contract=(
                            "All catalog prefill signatures and the decode signature initialized before capture; "
                            "this test does not admit unseen prefill signatures while the trace is live"
                        ),
                    ),
                    indent=2,
                )
                + "\n"
            )
            print(row, flush=True)
        ttnn.release_trace(mesh, trace)
        mesh.set_program_cache_misses_allowed(True)
        assert all(row["passed"] for row in results), results
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
