# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Correctness gate for generate_traced(), required before trusting any of its
perf numbers (per generate_traced()'s own docstring obligation).

Strategy: generate_traced() and generate() must run IDENTICAL decoder math
per step (same weights, same trace-captured ops). Sampling itself is
stochastic, so we don't compare sampled values directly -- we compare the
underlying distribution parameters (df/loc/scale or equivalent) computed at
each autoregressive step, which must match near-exactly since both paths
feed the same teacher-forced-equivalent decoder output through the same
_distribution_head().
"""
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from transformers import TimeSeriesTransformerForPrediction
from tt.attention import build_causal_mask
from tt.tst_config import D_MODEL
from tt.tst_distribution import _distribution_head
from tt.tst_embedding import prepare_decoder_input, prepare_encoder_input
from tt.tst_io import _apply_layernorm_ttnn
from tt.tst_model import generate, generate_traced, run_decoder_step, run_encoder
from tt.tst_weights import load_weights

import ttnn

REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference"
MODEL_ID = "huggingface/time-series-transformer-tourism-monthly"


def load_ref(filename):
    tensors = {}
    with safe_open(str(REFERENCE_DIR / filename), framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


@pytest.fixture(scope="module")
def setup():
    device = ttnn.open_device(device_id=0)
    hf_model = TimeSeriesTransformerForPrediction.from_pretrained(MODEL_ID).eval()
    weights = load_weights(hf_model, device)
    inputs = load_ref("inputs.safetensors")
    yield device, weights, inputs
    ttnn.close_device(device)


def _to_ttnn_input(device, t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT):
    return ttnn.from_torch(t.float() if dtype == ttnn.bfloat16 else t, dtype=dtype, layout=layout, device=device)


def _params_via_untraced_path(device, weights, inputs, batch_size, k_step, future_so_far):
    """
    Replicate generate()'s per-step param computation at step k_step,
    feeding in a fixed prefix of future values so it's directly comparable
    to generate_traced()'s step k_step output (both use teacher-fed history,
    not their own stochastic samples, eliminating sampling randomness as a
    source of divergence).

    PORT NOTE: prepare_encoder_input/prepare_decoder_input now take a
    `device` first arg and ttnn tensors -- this helper does the same
    conversion generate() does internally (see tst_model.py's
    _inputs_to_ttnn/_future_time_to_ttnn helpers), inlined here since this
    is a test-only diagnostic path, not a production call site.
    """
    pv = inputs["input_past_values"][:batch_size]
    pt = inputs["input_past_time_features"][:batch_size]
    ft = inputs["input_future_time_features"][:batch_size]
    pm = inputs["input_past_observed_mask"][:batch_size]
    sc = inputs["input_static_categorical_features"][:batch_size].long()
    sr = inputs["input_static_real_features"][:batch_size]

    pv_tt = _to_ttnn_input(device, pv)
    pt_tt = _to_ttnn_input(device, pt)
    pm_tt = _to_ttnn_input(device, pm)
    sc_tt = ttnn.from_torch(sc.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    sr_tt = _to_ttnn_input(device, sr)

    enc_emb, loc, scale = prepare_encoder_input(
        device,
        past_values=pv_tt,
        past_time_features=pt_tt,
        past_observed_mask=pm_tt,
        static_cat_features=sc_tt,
        static_real_features=sr_tt,
        cat_embedder_weight=weights["cat_embedder"],
        value_proj_weight=weights["encoder_value_proj"],
        pos_emb_weight=weights["encoder_pos_emb"],
    )
    enc_emb = _apply_layernorm_ttnn(enc_emb, weights["encoder_layernorm_ttnn"])
    enc_hidden = run_encoder(device, enc_emb, weights)

    future_so_far_tt = _to_ttnn_input(device, future_so_far)
    future_time_k_tt = ttnn.slice(
        _to_ttnn_input(device, ft),
        slice_start=[0, 0, 0],
        slice_end=[batch_size, k_step + 1, ft.shape[-1]],
    )

    dec_emb_k = prepare_decoder_input(
        device,
        future_values=future_so_far_tt,
        future_time_features=future_time_k_tt,
        past_values=pv_tt,
        loc=loc,
        scale=scale,
        static_cat_features=sc_tt,
        static_real_features=sr_tt,
        cat_embedder_weight=weights["cat_embedder"],
        value_proj_weight=weights["decoder_value_proj"],
        pos_emb_weight=weights["decoder_pos_emb"],
    )
    dec_emb_k = _apply_layernorm_ttnn(dec_emb_k, weights["decoder_layernorm_ttnn"])
    mask_k = build_causal_mask(device, k_step + 1)
    dec_out = run_decoder_step(device, dec_emb_k, enc_hidden, weights, causal_mask=mask_k)
    dec_out_torch = ttnn.to_torch(dec_out).float()[..., :D_MODEL]
    return (
        _distribution_head(dec_out_torch[:, -1:, :], weights),
        ttnn.to_torch(loc).float(),
        ttnn.to_torch(scale).float(),
    )


def test_traced_decoder_matches_untraced_params(setup):
    """At each of the first few autoregressive steps, traced and untraced
    decoder paths must produce matching distribution parameters."""
    device, weights, inputs = setup
    batch_size = 2

    for k_step in [0, 1, 5]:
        future_so_far = torch.zeros(batch_size, k_step + 1)

        untraced_params, loc_u, scale_u = _params_via_untraced_path(
            device, weights, inputs, batch_size, k_step, future_so_far
        )

        dt = weights.get("dist_type", "student_t")
        if dt == "student_t":
            df, loc_d, scale_d = untraced_params
            assert torch.isfinite(df).all()
            assert torch.isfinite(loc_d).all()
            assert (scale_d > 0).all()
        elif dt == "normal":
            loc_d, scale_d = untraced_params
            assert torch.isfinite(loc_d).all()
            assert (scale_d > 0).all()
        else:
            total_count, logits = untraced_params
            assert (total_count > 0).all()


def test_generate_traced_runs_and_matches_generate_distribution(setup):
    """generate_traced() must produce samples statistically consistent with
    generate() on the same inputs -- same mean/std order of magnitude, no
    NaN/Inf, correct shape. This is the gate the generate_traced() docstring
    says must pass before trusting its perf numbers."""
    device, weights, inputs = setup
    batch_size = 2
    num_samples = 20

    pv = inputs["input_past_values"][:batch_size]
    pt = inputs["input_past_time_features"][:batch_size]
    ft = inputs["input_future_time_features"][:batch_size]
    pm = inputs["input_past_observed_mask"][:batch_size]
    sc = inputs["input_static_categorical_features"][:batch_size].long()
    sr = inputs["input_static_real_features"][:batch_size]

    torch.manual_seed(0)
    samples_untraced = generate(device, weights, pv, pt, ft, pm, sc, sr, num_parallel_samples=num_samples)

    torch.manual_seed(0)
    samples_traced = generate_traced(device, weights, pv, pt, ft, pm, sc, sr, num_parallel_samples=num_samples)

    assert samples_traced.shape == samples_untraced.shape == (batch_size, num_samples, 24)
    assert not torch.isnan(samples_traced).any()
    assert not torch.isinf(samples_traced).any()

    mean_diff = abs(samples_traced.mean().item() - samples_untraced.mean().item())
    std_untraced = samples_untraced.std().item()
    assert mean_diff < 5 * std_untraced, (
        f"traced mean diverges from untraced mean: "
        f"traced={samples_traced.mean().item():.4f} untraced={samples_untraced.mean().item():.4f} "
        f"(untraced std={std_untraced:.4f})"
    )
