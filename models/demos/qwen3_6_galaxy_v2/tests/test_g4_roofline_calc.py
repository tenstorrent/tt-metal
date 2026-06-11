# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only validation that the augmented roofline reproduces hardware anchors.

No device. Anchors:
  * G=4 single-axis MLP traced+warmed: ~0.342 ms/layer @ bf8 (measured this repo)
  * TP=32 current 2D impl full decode: ~24 tok/s measured (~41.7 ms/token)
"""
import g4_roofline_calc as rc


def test_decode_weight_params_in_range():
    p = rc.decode_weight_params()
    # decoder layers + lm_head, excludes vision tower + (mostly) embed table.
    # Total model is 27.78B; vision + embed ~= 2.2B, so 24-26B is expected.
    assert 24.0e9 < p < 26.5e9, f"{p/1e9:.2f}B out of expected band"


def test_mlp_fraction_matches_known_split():
    mlp = rc.mlp_params_per_layer() * rc.N_LAYERS
    assert 16.5e9 < mlp < 17.5e9  # 3*H*I*64 = 17.1B


def test_tp32_bf8_weight_floor_about_1p5ms():
    r = rc.compute(32, "bf8")
    # ~25.6 GB over 32 chips * 512 GiB/s -> ~1.5 ms
    assert 1.2 < r.weight_ms < 1.8, r.weight_ms


def test_g4_mlp_only_ccl_anchor():
    # MLP-only: 2 CCLs/layer, no lm_head, no attn. Reproduce ~0.342 ms/layer @ bf8.
    mlp_w_bytes = rc.mlp_params_per_layer() * rc.BYTES["bf8"]
    floor_ms = mlp_w_bytes / (4 * rc.BW_CHIP) * 1000.0
    per_layer = floor_ms + 2 * rc.T_CCL_MS  # 2 CCLs * 0.10ms
    assert 0.28 < per_layer < 0.40, per_layer  # measured 0.342


def test_tp32_full_decode_reproduces_24_tok_s():
    r = rc.compute(32, "bf8", ccl_per_layer=rc.N_CCL_PER_LAYER_2D, socket_hops=0)
    # 1.5ms floor + 7*64*0.10 = 44.8ms -> ~21-26 tok/s, brackets measured ~24
    assert 20.0 < r.total_tok_s < 28.0, r.total_tok_s


def test_no_config_hits_70_tok_s_at_decode():
    # Central finding: CCL latency floor blocks 70 tok/s regardless of G/precision.
    for label, nchips, ccl_pl, hops in rc.PRESETS:
        for prec in ("bf16", "bf8", "bf4"):
            r = rc.compute(nchips, prec, ccl_per_layer=ccl_pl, socket_hops=hops)
            assert r.total_tok_s < 70.0, f"{label}/{prec} = {r.total_tok_s:.0f} tok/s"
