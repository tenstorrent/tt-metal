# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Stage 1 requirement: support student_t, normal, negative_binomial.
Exercises all three dist_type branches in _distribution_head() and generate(),
using synthetic dist_head weights since the pinned checkpoint only ships
Student-T heads.
"""
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from transformers import TimeSeriesTransformerForPrediction
from tt.tst_distribution import _distribution_head
from tt.tst_model import generate
from tt.tst_weights import load_weights

import ttnn

REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference"
MODEL_ID = "huggingface/time-series-transformer-tourism-monthly"
D_MODEL = 26


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


def _swap_dist_head(weights, dist_type):
    """Return a shallow-copied weights dict with dist_type swapped and a
    synthetic dist_head sized correctly for that distribution family.
    student_t/normal/negative_binomial all take a [B,T,D_MODEL] hidden input;
    only the output width (1 param vs 2 vs 3) differs in HF's real proj heads,
    but our student_t_params/normal_params/negative_binomial_params all read
    w0,b0,w1,b1 (,w2,b2) independently, so any [1, D_MODEL] linear works."""
    w = dict(weights)
    torch.manual_seed(0)
    dh = {
        "w0": torch.randn(1, D_MODEL) * 0.01,
        "b0": torch.zeros(1),
        "w1": torch.randn(1, D_MODEL) * 0.01,
        "b1": torch.zeros(1),
        "w2": torch.randn(1, D_MODEL) * 0.01,
        "b2": torch.zeros(1),
    }
    w["dist_head"] = dh
    w["dist_type"] = dist_type
    return w


@pytest.mark.parametrize("dist_type", ["student_t", "normal", "negative_binomial"])
def test_distribution_head_routes_correctly(setup, dist_type):
    device, weights, inputs = setup
    w = _swap_dist_head(weights, dist_type)
    hidden = torch.randn(4, 1, D_MODEL) * 0.1

    params = _distribution_head(hidden, w)

    if dist_type == "student_t":
        df, loc, scale = params
        assert df.shape == (4, 1)
        assert (df > 2.0).all(), "Student-T df must be > 2 after squareplus+2 shift"
        assert (scale > 0).all()
    elif dist_type == "normal":
        loc, scale = params
        assert loc.shape == (4, 1)
        assert (scale > 0).all()
    else:  # negative_binomial
        total_count, logits = params
        assert total_count.shape == (4, 1)
        assert (total_count > 0).all()


@pytest.mark.parametrize("dist_type", ["student_t", "normal", "negative_binomial"])
def test_generate_runs_for_each_distribution(setup, dist_type):
    device, weights, inputs = setup
    w = _swap_dist_head(weights, dist_type)

    pv = inputs["input_past_values"][:2]
    pt = inputs["input_past_time_features"][:2]
    ft = inputs["input_future_time_features"][:2]
    pm = inputs["input_past_observed_mask"][:2]
    sc = inputs["input_static_categorical_features"][:2].long()
    sr = inputs["input_static_real_features"][:2]

    samples = generate(device, w, pv, pt, ft, pm, sc, sr, num_parallel_samples=5)

    assert samples.shape == (2, 5, 24)
    assert not torch.isnan(samples).any(), f"{dist_type}: NaN in samples"
    assert not torch.isinf(samples).any(), f"{dist_type}: Inf in samples"
    if dist_type == "negative_binomial":
        assert (samples >= 0).all(), "negative_binomial samples must be non-negative integers"
        assert torch.allclose(samples, samples.round()), "negative_binomial samples must be integral"
