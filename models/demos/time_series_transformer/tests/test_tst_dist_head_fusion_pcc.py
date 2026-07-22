# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent Inc.
"""
Standalone PCC gate for Change 2 (distribution head fusion), step 1.

Compares the new ttnn student_t_params_ttnn / normal_params_ttnn /
negative_binomial_params_ttnn against the existing host torch
student_t_params / normal_params / negative_binomial_params, on the same
random hidden-state input, run through the SAME loaded weights (dist_head
vs dist_head_ttnn) so any transpose/layout mistake shows up as a real PCC
failure rather than passing silently.

Does NOT touch generate_traced() or run_traced_generation_cached() --
this only tests the new functions in isolation. Wiring into the trace
capture paths is step 2, gated on this passing.
"""
import pytest
import torch
from transformers import TimeSeriesTransformerForPrediction
from tt.tst_config import D_MODEL
from tt.tst_distribution import (
    negative_binomial_params,
    negative_binomial_params_ttnn,
    normal_params,
    normal_params_ttnn,
    student_t_params,
    student_t_params_ttnn,
)
from tt.tst_weights import load_weights

import ttnn
from models.common.utility_functions import comp_pcc

MODEL_ID = "huggingface/time-series-transformer-tourism-monthly"


@pytest.fixture
def weights(device):
    hf_model = TimeSeriesTransformerForPrediction.from_pretrained(MODEL_ID).eval()
    return load_weights(hf_model, device)


# 0.999, not 0.99: this tests an isolated 2-3 matmul projection head, on the
# SAME loaded weights (dist_head vs dist_head_ttnn), no attention or layer
# norm in the chain. Far less bfloat16 drift is expected at that depth, so
# the tighter threshold catches a real transpose/layout bug instead of
# hiding it behind a threshold sized for a much deeper op chain. See
# ../CHANGELOG.md "PCC threshold policy".
def _pcc_check(a, b, label, threshold=0.999):
    passed, pcc = comp_pcc(a, b, threshold)
    assert passed, f"{label}: PCC {pcc} below threshold {threshold}"


def test_student_t_params_ttnn_matches_host(device, weights):
    B, T = 2, 1
    torch.manual_seed(0)
    hidden_torch = torch.randn(B, T, D_MODEL, dtype=torch.float32)

    # Host path (existing, untouched)
    dh = weights["dist_head"]
    df_h, loc_h, scale_h = student_t_params(hidden_torch, dh["w0"], dh["b0"], dh["w1"], dh["b1"], dh["w2"], dh["b2"])

    # ttnn path (new)
    hidden_tt = ttnn.from_torch(hidden_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    df_tt, loc_tt, scale_tt = student_t_params_ttnn(hidden_tt, weights["dist_head_ttnn"])
    df_d = ttnn.to_torch(df_tt).float().squeeze(-1)
    loc_d = ttnn.to_torch(loc_tt).float().squeeze(-1)
    scale_d = ttnn.to_torch(scale_tt).float().squeeze(-1)

    _pcc_check(df_h, df_d, "student_t df")
    _pcc_check(loc_h, loc_d, "student_t loc")
    _pcc_check(scale_h, scale_d, "student_t scale")


def test_normal_params_ttnn_matches_host(device, weights):
    B, T = 2, 1
    torch.manual_seed(1)
    hidden_torch = torch.randn(B, T, D_MODEL, dtype=torch.float32)

    dh = weights["dist_head"]
    loc_h, scale_h = normal_params(hidden_torch, dh["w0"], dh["b0"], dh["w1"], dh["b1"])

    hidden_tt = ttnn.from_torch(hidden_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    loc_tt, scale_tt = normal_params_ttnn(hidden_tt, weights["dist_head_ttnn"])
    loc_d = ttnn.to_torch(loc_tt).float().squeeze(-1)
    scale_d = ttnn.to_torch(scale_tt).float().squeeze(-1)

    _pcc_check(loc_h, loc_d, "normal loc")
    _pcc_check(scale_h, scale_d, "normal scale")


def test_negative_binomial_params_ttnn_matches_host(device, weights):
    B, T = 2, 1
    torch.manual_seed(2)
    hidden_torch = torch.randn(B, T, D_MODEL, dtype=torch.float32)

    dh = weights["dist_head"]
    tc_h, logits_h = negative_binomial_params(hidden_torch, dh["w0"], dh["b0"], dh["w1"], dh["b1"])

    hidden_tt = ttnn.from_torch(hidden_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tc_tt, logits_tt = negative_binomial_params_ttnn(hidden_tt, weights["dist_head_ttnn"])
    tc_d = ttnn.to_torch(tc_tt).float().squeeze(-1)
    logits_d = ttnn.to_torch(logits_tt).float().squeeze(-1)

    _pcc_check(tc_h, tc_d, "negative_binomial total_count")
    _pcc_check(logits_h, logits_d, "negative_binomial logits")
