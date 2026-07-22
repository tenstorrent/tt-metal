# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Bisection: real write-then-attend vs overlay-then-attend, ONE decoder layer,
SELF-ATTENTION OUTPUT ONLY (no cross-attn, no FFN, no second layer, no trace).
Everything used here is the REAL production function except the overlay
helper itself, which is imported from probe_fused_trace.py verbatim (not
reimplemented a second time).
"""
import sys
from pathlib import Path

import torch
from transformers import TimeSeriesTransformerForPrediction

import ttnn

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))

from models.demos.time_series_transformer.tt.tst_attention import NUM_HEADS, allocate_kv_cache  # noqa: E402
from models.demos.time_series_transformer.tt.tst_model import PADDED_WIDTH, load_weights  # noqa: E402
from models.demos.time_series_transformer.tt.tst_model_cached_additions import (  # noqa: E402
    _attend_from_cache,
    _build_causal_mask_1tok,
    _extract_and_write_kv,
    _zero_kv_caches,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_fused_trace import _overlay_kv_device, _precompute_kv_selectors_host  # noqa: E402

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
        w_layer = weights["decoder.layers.0"]
        w_self = w_layer["self_attn"]

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

        # -- Arm A: real write then real attend --
        kc_a, vc_a = allocate_kv_cache(device, BS, T_max=T_MAX)
        _zero_kv_caches([(kc_a, vc_a)], device, BS, T_MAX)
        q_a = _extract_and_write_kv(hidden_1tok, w_self, kc_a, vc_a, step=0, cq_id=0)
        ttnn.synchronize_device(device)
        out_a = _attend_from_cache(q_a, kc_a, vc_a, causal_mask, w_self)

        # -- Arm B: overlay then real attend --
        kc_b, vc_b = allocate_kv_cache(device, BS, T_max=T_MAX)
        _zero_kv_caches([(kc_b, vc_b)], device, BS, T_MAX)
        q_b, k_full_b, v_full_b = _overlay_kv_device(hidden_1tok, w_self, kc_b, vc_b, k_sel, v_sel)
        out_b = _attend_from_cache(q_b, k_full_b, v_full_b, causal_mask, w_self)

        a_t = ttnn.to_torch(out_a).float()
        b_t = ttnn.to_torch(out_b).float()
        diff = (a_t - b_t).abs().max().item()
        print(f"[RESULT] single-layer eager self-attn-only max_abs_diff={diff:.6f}")
        print(
            "PASS -- overlay/attend composition itself is correct; bug is multi-layer/trace-specific."
            if diff < 0.01
            else "FAIL -- bug is in the overlay/attend composition itself, before tracing enters the picture."
        )
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
