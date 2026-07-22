# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Bisection: TWO decoder layers, chained, fully eager (no trace at all).
Isolates chaining from tracing -- the one thing not yet tested in isolation.
"""
import sys
from pathlib import Path

import torch
from transformers import TimeSeriesTransformerForPrediction

import ttnn

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_fused_trace import (  # noqa: E402
    _overlay_kv_device,
    _precompute_kv_selectors_host,
    _run_single_layer_compute_prebuilt_kv,
)

from models.demos.time_series_transformer.tt.tst_attention import allocate_kv_cache  # noqa: E402
from models.demos.time_series_transformer.tt.tst_attention import precompute_cross_attn_kv  # noqa: E402
from models.demos.time_series_transformer.tt.tst_model import PADDED_WIDTH, load_weights  # noqa: E402
from models.demos.time_series_transformer.tt.tst_model_cached_additions import (  # noqa: E402
    _attend_from_cache,
    _build_causal_mask_1tok,
    _extract_and_write_kv,
    _run_single_layer_compute,
    _zero_kv_caches,
)

MODEL_ID = "huggingface/time-series-transformer-tourism-monthly"
MODEL_REVISION = "2a40ad41f6ffe61e7bef6099b08c6c2fce36ac35"
T_MAX = 8
BS = 1


def main():
    device = ttnn.open_device(device_id=0, l1_small_size=24_576)
    try:
        hf_model = TimeSeriesTransformerForPrediction.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        hf_model.eval()
        weights = load_weights(hf_model, device)
        num_layers = sum(1 for k in weights if k.startswith("decoder.layers."))
        print(f"[setup] num_decoder_layers={num_layers}")

        causal_mask = _build_causal_mask_1tok(device, 0, T_MAX)
        k_sel_host, v_sel_host = _precompute_kv_selectors_host(T_MAX)
        k_sel = ttnn.from_torch(k_sel_host[0], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        v_sel = ttnn.from_torch(v_sel_host[0], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        hidden_1tok = ttnn.from_torch(
            torch.randn(BS, 1, PADDED_WIDTH, dtype=torch.bfloat16) * 0.1,
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
            precompute_cross_attn_kv(encoder_hidden, weights[f"decoder.layers.{i}"]["encoder_attn"])
            for i in range(num_layers)
        ]

        caches_a = [allocate_kv_cache(device, BS, T_max=T_MAX) for _ in range(num_layers)]
        _zero_kv_caches(caches_a, device, BS, T_MAX)
        h = hidden_1tok
        for i in range(num_layers):
            w_layer = weights[f"decoder.layers.{i}"]
            kc, vc = caches_a[i]
            k_pre, v_pre = precomputed_kv[i]
            q = _extract_and_write_kv(h, w_layer["self_attn"], kc, vc, step=0, cq_id=0)
            ttnn.synchronize_device(device)
            h = _run_single_layer_compute(h, q, kc, vc, k_pre, v_pre, causal_mask, w_layer)
        out_a = h

        caches_b = [allocate_kv_cache(device, BS, T_max=T_MAX) for _ in range(num_layers)]
        _zero_kv_caches(caches_b, device, BS, T_MAX)
        h = hidden_1tok
        for i in range(num_layers):
            w_layer = weights[f"decoder.layers.{i}"]
            kc, vc = caches_b[i]
            k_pre, v_pre = precomputed_kv[i]
            q, k_full, v_full = _overlay_kv_device(h, w_layer["self_attn"], kc, vc, k_sel, v_sel)
            h = _run_single_layer_compute_prebuilt_kv(h, q, k_full, v_full, k_pre, v_pre, causal_mask, w_layer)
        out_b = h

        a_t = ttnn.to_torch(out_a).float()
        b_t = ttnn.to_torch(out_b).float()
        diff = (a_t - b_t).abs().max().item()
        print(f"[RESULT] two-layer eager chained max_abs_diff={diff:.6f}")
        print(
            "PASS -- chaining is correct; bug is trace-capture-specific."
            if diff < 0.01
            else "FAIL -- bug is in chaining itself, independent of tracing."
        )
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
