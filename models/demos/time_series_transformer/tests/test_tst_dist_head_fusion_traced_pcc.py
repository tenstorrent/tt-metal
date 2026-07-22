# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PCC check: fused ttnn Student-T dist head (student_t_params_ttnn) vs the
host torch dist head (_distribution_head), on a REAL hidden state produced
by run_traced_generation_cached's own decoder stack -- not a random tensor.

Scope: Student-T only (student_t_params_ttnn). Does not cover
normal_params_ttnn / negative_binomial_params_ttnn -- see PR comment
from yieldthought: generation currently only exercises Student-T.

Does NOT touch generate_traced() or run_traced_generation_cached's
production call path -- it calls run_traced_generation_cached() once,
UNMODIFIED, purely to populate ctx.traced_out with a real decoder output,
then reads it independently. No debug scaffolding added to
tst_model_cached_additions.py.

No `device` fixture is available under --noconftest (see test_tst_perf.py's
own repro command), so the device is opened/closed manually here, matching
test_tst_perf.py's _open_device() pattern rather than assuming a fixture.
"""

import math
from pathlib import Path

from safetensors import safe_open
from transformers import TimeSeriesTransformerForPrediction

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.time_series_transformer.tt.tst_config import D_MODEL
from models.demos.time_series_transformer.tt.tst_distribution import _distribution_head, student_t_params_ttnn
from models.demos.time_series_transformer.tt.tst_model_cached_additions import (
    build_traced_decoder_context_cached,
    run_traced_generation_cached,
)
from models.demos.time_series_transformer.tt.tst_weights import load_weights

MODEL_ID = "huggingface/time-series-transformer-tourism-monthly"
REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference"
L1_SMALL_SIZE = 24_576
# 0.99 here (not 0.999 like the isolated-head PCC test in
# test_tst_dist_head_fusion_pcc.py): this hidden state comes from the FULL
# traced decoder stack, not an isolated head, so the same op-chain-depth
# rule from ../CHANGELOG.md "PCC threshold policy" places it at 0.99.
PCC_THRESHOLD = 0.99

# Copied verbatim from test_tst_perf.py -- proven-working trace sizing,
# not reinvented here.
TRACE_BYTES_PER_BS_UNIT = 640_615
TRACE_HEADROOM = 1.25


def _trace_region_size(bs: int) -> int:
    return int(math.ceil(TRACE_BYTES_PER_BS_UNIT * bs * TRACE_HEADROOM))


def _open_device() -> ttnn.Device:
    return ttnn.open_device(
        device_id=0,
        l1_small_size=L1_SMALL_SIZE,
        trace_region_size=_trace_region_size(bs=1),
    )


def _load_inputs(b: int) -> dict:
    tensors = {}
    with safe_open(str(REFERENCE_DIR / "inputs.safetensors"), framework="pt", device="cpu") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)[:b]
    return tensors


def test_dist_head_fusion_traced_pcc():
    """
    1. Build a real ctx and run ONE generation pass (unfused, untouched)
       so ctx.traced_out holds a real final-decoder-layer output.
    2. Read that buffer exactly as production does: to_torch + [..., :D_MODEL].
    3. Run host _distribution_head() on it (== production's own reference call).
    4. Rebuild a clean [B,1,D_MODEL] device tensor from the SAME sliced data
       and run student_t_params_ttnn() on it.
    5. Assert shapes match after the required squeeze, then PCC-compare.
    """
    inputs = _load_inputs(b=1)
    device = _open_device()
    try:
        hf_model = TimeSeriesTransformerForPrediction.from_pretrained(MODEL_ID).eval()
        weights = load_weights(hf_model, device)

        ctx = build_traced_decoder_context_cached(
            device,
            weights,
            inputs["input_past_values"],
            inputs["input_past_time_features"],
            inputs["input_future_time_features"],
            inputs["input_past_observed_mask"],
            inputs["input_static_categorical_features"],
            inputs["input_static_real_features"],
        )
        try:
            run_traced_generation_cached(ctx, weights, use_2cq=False)

            # ── Real hidden state, sliced exactly like production (line 1036) ──
            dec_out = ttnn.to_torch(ctx.traced_out).float()[..., :D_MODEL]  # [B, 1, D_MODEL]

            # ── Host reference (== production's own call, line 1039) ──
            df_host, loc_host, scale_host = _distribution_head(dec_out, weights)  # each [B, 1] post-squeeze

            # ── Device fused path, independent tensor built from the same data ──
            hidden_tt = ttnn.from_torch(dec_out, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            df_tt, loc_tt, scale_tt = student_t_params_ttnn(hidden_tt, weights["dist_head_ttnn"])

            df_dev = ttnn.to_torch(df_tt).float()
            loc_dev = ttnn.to_torch(loc_tt).float()
            scale_dev = ttnn.to_torch(scale_tt).float()

            # student_t_params_ttnn docstring: NOT squeezed -> [B, 1, 1]. Verify, then squeeze.
            assert df_dev.shape[-1] == 1, f"expected trailing dim 1 (unsqueezed), got {df_dev.shape}"
            df_dev, loc_dev, scale_dev = (t.squeeze(-1) for t in (df_dev, loc_dev, scale_dev))

            assert df_dev.shape == df_host.shape, f"df shape mismatch: {df_dev.shape} vs {df_host.shape}"
            assert loc_dev.shape == loc_host.shape, f"loc shape mismatch: {loc_dev.shape} vs {loc_host.shape}"
            assert scale_dev.shape == scale_host.shape, f"scale shape mismatch: {scale_dev.shape} vs {scale_host.shape}"

            for name, host_t, dev_t in (
                ("df", df_host, df_dev),
                ("loc", loc_host, loc_dev),
                ("scale", scale_host, scale_dev),
            ):
                passed, pcc_msg = comp_pcc(host_t, dev_t, PCC_THRESHOLD)
                assert passed, f"{name} PCC failed: {pcc_msg}"
        finally:
            ctx.release()
    finally:
        ttnn.close_device(device)
