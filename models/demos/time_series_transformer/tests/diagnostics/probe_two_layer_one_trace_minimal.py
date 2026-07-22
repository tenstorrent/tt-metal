# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Minimal isolation: TWO overlay+self-attn+residual+layernorm blocks (no
cross-attn, no FFN) chained inside ONE trace capture, vs the same computed
as TWO SEPARATE per-layer traces (matches the proven-working baseline
pattern). If this alone reproduces divergence, the bug is confirmed to be
buffer aliasing/dependency-tracking specific to same-shape op sequences
repeating inside one capture -- not anything to do with cross-attn or FFN.
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

from models.demos.time_series_transformer.tt.tst_attention import allocate_kv_cache  # noqa: E402
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


def self_attn_block(hidden_in, query, k, v, causal_mask, w_self):
    attn_out = _attend_from_cache(query, k, v, causal_mask, w_self)
    residual = ttnn.add(hidden_in, attn_out)
    return layer_norm_padded(
        residual, w_self["self_attn_layer_norm_weight"], w_self["self_attn_layer_norm_bias"], orig_dim=D_MODEL
    )


def main():
    device = ttnn.open_device(device_id=0, l1_small_size=24_576)
    trace_ids = []
    try:
        hf_model = TimeSeriesTransformerForPrediction.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        hf_model.eval()
        weights = load_weights(hf_model, device)
        num_layers = 2
        ws = [weights[f"decoder.layers.{i}"]["self_attn"] for i in range(num_layers)]
        for w in ws:
            w["self_attn_layer_norm_weight"] = weights["decoder.layers.0"]["self_attn_layer_norm_weight"]
            w["self_attn_layer_norm_bias"] = weights["decoder.layers.0"]["self_attn_layer_norm_bias"]

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

        # -- Baseline: TWO SEPARATE per-layer traces (proven pattern) --
        caches_base = [allocate_kv_cache(device, BS, T_max=T_MAX) for _ in range(num_layers)]
        _zero_kv_caches(caches_base, device, BS, T_MAX)
        hidden_in = dec_input
        layer_bufs_base = [dec_input]
        for i in range(num_layers):
            kc, vc = caches_base[i]
            q_warm = _extract_and_write_kv(hidden_in, ws[i], kc, vc, step=0, cq_id=0)
            _ = self_attn_block(hidden_in, q_warm, kc, vc, causal_mask, ws[i])
            ttnn.synchronize_device(device)
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            q_t = _extract_q_only(hidden_in, ws[i])
            out = self_attn_block(hidden_in, q_t, kc, vc, causal_mask, ws[i])
            ttnn.end_trace_capture(device, tid, cq_id=0)
            trace_ids.append(tid)
            layer_bufs_base.append(out)
            hidden_in = out
        _zero_kv_caches(caches_base, device, BS, T_MAX)
        for i in range(num_layers):
            kc, vc = caches_base[i]
            _extract_and_write_kv(layer_bufs_base[i], ws[i], kc, vc, step=0, cq_id=0)
            ttnn.execute_trace(device, trace_ids[i], cq_id=0, blocking=True)
        out_baseline = layer_bufs_base[num_layers]

        # -- Fused: TWO layers, ONE trace (the thing under test) --
        caches_fused = [allocate_kv_cache(device, BS, T_max=T_MAX) for _ in range(num_layers)]
        _zero_kv_caches(caches_fused, device, BS, T_MAX)
        hidden_in = dec_input
        for i in range(num_layers):
            kc, vc = caches_fused[i]
            q_warm = _extract_and_write_kv(hidden_in, ws[i], kc, vc, step=0, cq_id=0)
            hidden_in = self_attn_block(hidden_in, q_warm, kc, vc, causal_mask, ws[i])
        ttnn.synchronize_device(device)
        hidden_in = dec_input
        for i in range(num_layers):
            kc, vc = caches_fused[i]
            q_o, k_full, v_full = _overlay_kv_device(hidden_in, ws[i], kc, vc, k_sel, v_sel)
            hidden_in = self_attn_block(hidden_in, q_o, k_full, v_full, causal_mask, ws[i])
        ttnn.synchronize_device(device)
        _zero_kv_caches(caches_fused, device, BS, T_MAX)

        tid_fused = ttnn.begin_trace_capture(device, cq_id=0)
        hidden_in = dec_input
        layer_bufs_fused = [dec_input]
        for i in range(num_layers):
            kc, vc = caches_fused[i]
            q_o, k_full, v_full = _overlay_kv_device(hidden_in, ws[i], kc, vc, k_sel, v_sel)
            hidden_in = self_attn_block(hidden_in, q_o, k_full, v_full, causal_mask, ws[i])
            layer_bufs_fused.append(hidden_in)
        ttnn.end_trace_capture(device, tid_fused, cq_id=0)
        trace_ids.append(tid_fused)
        _zero_kv_caches(caches_fused, device, BS, T_MAX)

        ttnn.execute_trace(device, tid_fused, cq_id=0, blocking=True)
        for i in range(num_layers):
            kc, vc = caches_fused[i]
            _extract_and_write_kv(layer_bufs_fused[i], ws[i], kc, vc, step=0, cq_id=0)
        out_fused = layer_bufs_fused[num_layers]

        base_t = ttnn.to_torch(out_baseline).float()
        fused_t = ttnn.to_torch(out_fused).float()
        diff = (base_t - fused_t).abs().max().item()
        print(f"[RESULT] minimal self-attn-only, 2-layer-in-1-trace max_abs_diff={diff:.6f}")
        print(
            "PASS -- even minimal 2-layer-in-1-trace works; bug needs cross-attn/FFN present to manifest."
            if diff < 0.01
            else "FAIL -- confirmed: 2-layer-in-1-trace itself is broken, independent of cross-attn/FFN."
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
