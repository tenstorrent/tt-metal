# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full-context execution with a sampled, exact HF layer oracle.

All K/V positions participate. Only the compared query rows are sampled to
avoid a quadratic full-context HF attention matrix; model shapes, weights,
and the TT prefill invocation are not reduced. Sampling is reported explicitly.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding, apply_rotary_pos_emb

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests.run_decoder import MODEL, REVISION, load_layer
from models.autoports.google_gemma_4_26b_a4b_it.tests.runtime_audit import device_only
from models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder import FunctionalDecoder
from models.common.utility_functions import comp_pcc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--reference-file", type=Path)
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
    torch.manual_seed(123)
    torch.set_num_threads(8)
    config = AutoConfig.from_pretrained(MODEL, revision=REVISION).text_config
    config._attn_implementation = "sdpa"
    hf = load_layer(config, args.layer, True)
    kind = config.layer_types[args.layer]
    length = args.length
    x = torch.randn(1, length, config.hidden_size).bfloat16().float()
    rope = Gemma4TextRotaryEmbedding(config)
    extent = (length + 1023) // 1024 * 1024
    cos, sin = rope(x, torch.arange(extent)[None], layer_type=kind)
    if args.reference_file:
        saved = torch.load(args.reference_file, weights_only=True)
        assert (saved["layer"], saved["length"], saved["revision"]) == (args.layer, length, REVISION)
        ref, samples = saved["reference"], saved["samples"]
    else:
        keys, values = [], []
        with torch.no_grad():
            for start in range(0, length, 1024):
                end = min(start + 1024, length)
                norm = hf.input_layernorm(x[:, start:end])
                k = hf.self_attn.k_proj(norm).view(1, end - start, -1, hf.self_attn.head_dim)
                v = hf.self_attn.v_proj(norm).view_as(k) if hf.self_attn.v_proj is not None else k
                k = apply_rotary_pos_emb(hf.self_attn.k_norm(k), cos[:, start:end], sin[:, start:end], unsqueeze_dim=2)
                keys.append(k.transpose(1, 2))
                values.append(hf.self_attn.v_norm(v).transpose(1, 2))
        keys, values = torch.cat(keys, dim=2), torch.cat(values, dim=2)

        class FixedCache:
            def update(self, k, v, layer_idx):
                return keys, values

        samples = sorted(
            set(
                [
                    0,
                    min(31, length - 1),
                    min(32, length - 1),
                    *range(1023, length, 1024),
                    *range(max(0, length - 33), length),
                ]
            )
        )
        refs = []
        with torch.no_grad():
            for offset in range(0, len(samples), 16):
                idx = torch.tensor(samples[offset : offset + 16])
                allowed = torch.arange(length)[None, :] <= idx[:, None]
                if kind == "sliding_attention":
                    allowed &= torch.arange(length)[None, :] > idx[:, None] - config.sliding_window
                mask = torch.zeros(len(idx), length).masked_fill(~allowed, float("-inf"))[None, None]
                refs.append(
                    hf(
                        x[:, idx],
                        position_embeddings=(cos[:, idx], sin[:, idx]),
                        attention_mask=mask,
                        past_key_values=FixedCache(),
                    )
                )
        ref = torch.cat(refs, dim=1)
    print("HF_SAMPLED_REFERENCE_READY", len(samples), flush=True)
    if args.reference_only:
        torch.save(
            {"reference": ref, "samples": samples, "length": length, "layer": args.layer, "revision": REVISION},
            args.output,
        )
        return
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        decoder = decoder_class.from_state_dict(
            hf.state_dict(), hf_config=config, layer_idx=args.layer, mesh_device=mesh, **decoder_options
        )

        def device(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(t, device=mesh, dtype=dtype, layout=layout)

        block = 32
        pages = (extent + block - 1) // block
        table = torch.randperm(pages).int()[None]
        pt = device(table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        cfg = decoder.layer.self_attn.config
        cache = [device(torch.zeros(pages, cfg.num_key_value_heads, block, cfg.head_dim)) for _ in range(2)]
        xt = device(x[None])
        rt = tuple(device(t[None]) for t in (cos, sin))
        print("TT_PREFILL_START", length, flush=True)
        with device_only():
            out = decoder.prefill_forward(xt, rope_mats=rt, page_table=pt, kv_cache=cache)
        actual = ttnn.to_torch(out).squeeze(0).float()[:, samples]
        passed, pcc = comp_pcc(ref, actual, 0.995)
        result = dict(
            layer_type=kind,
            length=length,
            phase="prefill",
            real_weights=True,
            pcc=float(pcc),
            passed=bool(passed),
            compared_query_rows=samples,
            scope="subset",
            cache_positions=length,
            runtime_prefill_audit="passed",
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(result, flush=True)
        dt = device(x[:, -1:][None])
        dr = tuple(device(t.squeeze(0)) for t in (cos, sin))
        p = torch.zeros(1, 32, dtype=torch.int32)
        p[0, 0] = length - 1
        cp = torch.tensor([length - 1], dtype=torch.int32)
        rp = device(p, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        dp = device(cp, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)

        def decode():
            with device_only():
                return decoder.decode_forward(
                    dt, rope_mats=dr, current_pos=rp, cache_pos=dp, page_table=pt, kv_cache=cache
                )

        warm = decode()
        ttnn.synchronize_device(mesh)
        warm.deallocate(True)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        traced = decode()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        decode_results = []
        for position in [length - 1, max(0, length - 2), length - 1]:
            p[0, 0], cp[0] = position, position
            for tensor, dest, dtype, layout in [
                (x[:, position : position + 1][None], dt, ttnn.bfloat16, ttnn.TILE_LAYOUT),
                (p, rp, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
                (cp, dp, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
            ]:
                host = ttnn.from_torch(tensor, dtype=dtype, layout=layout)
                ttnn.copy_host_to_device_tensor(host, dest)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            got = ttnn.to_torch(traced).squeeze(0).float()
            oracle = ref[:, samples.index(position) : samples.index(position) + 1]
            ok, value = comp_pcc(oracle, got, 0.995)
            decode_results.append(dict(position=position, pcc=float(value), passed=bool(ok)))
        ttnn.release_trace(mesh, trace)
        result["decode"] = decode_results
        result["runtime_decode_audit"] = "passed"
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print("DECODE", decode_results, flush=True)
        assert passed and all(r["passed"] for r in decode_results), result
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
