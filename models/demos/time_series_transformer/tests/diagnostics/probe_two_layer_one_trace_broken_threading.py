# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Positive control: deliberately reproduces the ORIGINAL bug that motivated
the per-layer-trace rewrite in tst_model_cached_additions.py.

All prior probes in this series (self-attn alone, self+cross, self+FFN,
full self+cross+FFN) fused two layers into one trace with CORRECT
layer-to-layer threading (each layer's Q/K/V genuinely derived from the
previous layer's real output buffer) and were all bit-exact vs. the
per-layer-trace baseline. That means trace fusion itself is not the
problem.

This probe instead reproduces the documented original mistake: BOTH
layers compute Q/K/V from `captured_dec_input` (the raw single-token
embedding), so layer 1 never actually attends over layer 0's output --
it attends over the same raw input layer 0 did. This is a pure
data-threading bug, orthogonal to whether the two layers share one trace
or two. Expected: FAIL, with a large max_abs_diff, confirming this
(not trace fusion) was the real root cause of the original ~0.31-0.33
PCC regression.
"""
import sys
from pathlib import Path

import torch
from transformers import TimeSeriesTransformerForPrediction

import ttnn

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_fused_trace import _overlay_kv_device, _precompute_kv_selectors_host  # noqa: E402

from models.demos.time_series_transformer.tt.tst_attention import (  # noqa: E402
    allocate_kv_cache,
    precompute_cross_attn_kv,
    tst_cross_attention_with_kv,
)
from models.demos.time_series_transformer.tt.tst_decoder_layer import tst_ffn  # noqa: E402
from models.demos.time_series_transformer.tt.tst_model import D_MODEL, PADDED_WIDTH, load_weights  # noqa: E402
from models.demos.time_series_transformer.tt.tst_model_cached_additions import (  # noqa: E402
    _attend_from_cache,
    _build_causal_mask_1tok,
    _extract_and_write_kv,
    _extract_q_only,
    _zero_kv_caches,
)
from models.demos.time_series_transformer.tt.ttnn_utils import layer_norm_padded  # noqa: E402

MODEL_ID = "huggingface/time-series-transformer-tourism-monthly"
MODEL_REVISION = "2a40ad41f6ffe61e7bef6099b08c6c2fce36ac35"
T_MAX = 8
BS = 1


def block(hidden_in, query, k, v, k_pre, v_pre, causal_mask, w_layer):
    """Full decoder layer: self-attn(cache) + cross-attn(precomputed) + FFN."""
    attn_out = _attend_from_cache(query, k, v, causal_mask, w_layer["self_attn"])
    residual = ttnn.add(hidden_in, attn_out)
    hidden = layer_norm_padded(
        residual, w_layer["self_attn_layer_norm_weight"], w_layer["self_attn_layer_norm_bias"], orig_dim=D_MODEL
    )

    cross_out = tst_cross_attention_with_kv(hidden, k_pre, v_pre, w_layer["encoder_attn"])
    residual = ttnn.add(hidden, cross_out)
    hidden = layer_norm_padded(
        residual, w_layer["encoder_attn_layer_norm_weight"], w_layer["encoder_attn_layer_norm_bias"], orig_dim=D_MODEL
    )

    ffn_out = tst_ffn(hidden, w_layer)
    residual = ttnn.add(hidden, ffn_out)
    return layer_norm_padded(
        residual, w_layer["final_layer_norm_weight"], w_layer["final_layer_norm_bias"], orig_dim=D_MODEL
    )


def main():
    device = ttnn.open_device(device_id=0, l1_small_size=24_576)
    trace_ids = []
    try:
        hf_model = TimeSeriesTransformerForPrediction.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        hf_model.eval()
        weights = load_weights(hf_model, device)
        num_layers = 2
        w_layers = [weights[f"decoder.layers.{i}"] for i in range(num_layers)]

        causal_mask = _build_causal_mask_1tok(device, 0, T_MAX)
        k_sel_host, v_sel_host = _precompute_kv_selectors_host(T_MAX)
        k_sel = ttnn.from_torch(k_sel_host[0], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        v_sel = ttnn.from_torch(v_sel_host[0], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        dec_input = ttnn.from_torch(
            torch.zeros(BS, 1, PADDED_WIDTH, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        encoder_hidden = ttnn.from_torch(
            torch.randn(BS, 24, PADDED_WIDTH, dtype=torch.bfloat16) * 0.1,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        precomputed_kv = [
            precompute_cross_attn_kv(encoder_hidden, w_layers[i]["encoder_attn"]) for i in range(num_layers)
        ]

        # -- Ground truth: TWO SEPARATE per-layer traces, CORRECT threading --
        caches_base = [allocate_kv_cache(device, BS, T_max=T_MAX) for _ in range(num_layers)]
        _zero_kv_caches(caches_base, device, BS, T_MAX)
        hidden_in = dec_input
        layer_bufs_base = [dec_input]
        for i in range(num_layers):
            kc, vc = caches_base[i]
            k_pre, v_pre = precomputed_kv[i]
            q_warm = _extract_and_write_kv(hidden_in, w_layers[i]["self_attn"], kc, vc, step=0, cq_id=0)
            _ = block(hidden_in, q_warm, kc, vc, k_pre, v_pre, causal_mask, w_layers[i])
            ttnn.synchronize_device(device)
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            q_t = _extract_q_only(hidden_in, w_layers[i]["self_attn"])
            out = block(hidden_in, q_t, kc, vc, k_pre, v_pre, causal_mask, w_layers[i])
            ttnn.end_trace_capture(device, tid, cq_id=0)
            trace_ids.append(tid)
            layer_bufs_base.append(out)
            hidden_in = out
        _zero_kv_caches(caches_base, device, BS, T_MAX)
        for i in range(num_layers):
            kc, vc = caches_base[i]
            _extract_and_write_kv(layer_bufs_base[i], w_layers[i]["self_attn"], kc, vc, step=0, cq_id=0)
            ttnn.execute_trace(device, trace_ids[i], cq_id=0, blocking=True)
        out_baseline = layer_bufs_base[num_layers]

        # -- BROKEN: ONE fused trace, BOTH layers read Q/K/V from raw
        #    dec_input instead of the previous layer's real output. This
        #    reproduces the original documented bug verbatim.
        caches_broken = [allocate_kv_cache(device, BS, T_max=T_MAX) for _ in range(num_layers)]
        _zero_kv_caches(caches_broken, device, BS, T_MAX)
        # Untraced warmup: BOTH layers write K/V and compute using dec_input
        # (the raw embedding) -- NOT hidden_in threaded from the previous
        # layer. This is the bug, reproduced deliberately.
        for i in range(num_layers):
            kc, vc = caches_broken[i]
            k_pre, v_pre = precomputed_kv[i]
            q_warm = _extract_and_write_kv(dec_input, w_layers[i]["self_attn"], kc, vc, step=0, cq_id=0)
            _ = block(dec_input, q_warm, kc, vc, k_pre, v_pre, causal_mask, w_layers[i])
        ttnn.synchronize_device(device)
        _zero_kv_caches(caches_broken, device, BS, T_MAX)

        tid_broken = ttnn.begin_trace_capture(device, cq_id=0)
        layer_outs_broken = []
        for i in range(num_layers):
            kc, vc = caches_broken[i]
            k_pre, v_pre = precomputed_kv[i]
            # BUG: both layers derive Q from dec_input, not from the
            # previous layer's real output buffer.
            q_t = _extract_q_only(dec_input, w_layers[i]["self_attn"])
            out = block(dec_input, q_t, kc, vc, k_pre, v_pre, causal_mask, w_layers[i])
            layer_outs_broken.append(out)
        ttnn.end_trace_capture(device, tid_broken, cq_id=0)
        trace_ids.append(tid_broken)
        _zero_kv_caches(caches_broken, device, BS, T_MAX)

        # Per-step replay: write BOTH layers' K/V from dec_input (bug:
        # layer 1's K/V should come from layer 0's real output, but here
        # both read the same raw dec_input), then run the single trace.
        for i in range(num_layers):
            kc, vc = caches_broken[i]
            _extract_and_write_kv(dec_input, w_layers[i]["self_attn"], kc, vc, step=0, cq_id=0)
        ttnn.execute_trace(device, tid_broken, cq_id=0, blocking=True)
        out_broken = layer_outs_broken[num_layers - 1]

        base_t = ttnn.to_torch(out_baseline).float()
        broken_t = ttnn.to_torch(out_broken).float()
        diff = (base_t - broken_t).abs().max().item()
        rel_diff = diff / (base_t.abs().max().item() + 1e-8)
        print(f"[RESULT] BROKEN threading (both layers read raw dec_input), max_abs_diff={diff:.6f} rel={rel_diff:.4f}")
        print(
            "UNEXPECTED PASS -- broken threading did NOT diverge here; root cause must be something else entirely (e.g. real multi-step generation state, not single-step layer wiring)."
            if diff < 0.01
            else "CONFIRMED: broken threading (layer 1 reading raw input instead of layer 0's real output) reproduces a large divergence -- this IS the root cause of the original PCC regression, not trace fusion itself."
        )
    finally:
        for tid in trace_ids:
            try:
                ttnn.release_trace(device, tid)
            except Exception:
                pass
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
