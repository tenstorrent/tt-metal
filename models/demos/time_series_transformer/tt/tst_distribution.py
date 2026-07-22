# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

"""
Distribution heads for probabilistic forecasting: parameter projection,
sampling, and exact NLL for Student-T, Normal, and Negative Binomial.

Host (torch) implementations below are the reference path, used by every
current generation and NLL call site. A parallel ttnn projection path
(student_t_params_ttnn / normal_params_ttnn / negative_binomial_params_ttnn)
exists at the bottom of this file but is NOT wired into any trace-capture
path — see that section's docstring before using it.
"""

import torch
from torch.distributions import NegativeBinomial, Normal, StudentT


def squareplus(x):
    return 0.5 * (x + torch.sqrt(x * x + 4.0))


# ── Student-T ────────────────────────────────────────────────────────────────


def student_t_params(hidden, w0, b0, w1, b1, w2, b2):
    """Project decoder hidden -> (df, loc, scale) for Student-T.
    proj.0 -> df, proj.1 -> loc, proj.2 -> scale (HF convention)"""
    raw_df = hidden @ w0.T + b0
    loc = hidden @ w1.T + b1
    raw_scale = hidden @ w2.T + b2
    eps = torch.finfo(raw_scale.dtype).eps
    scale = squareplus(raw_scale).clamp_min(eps).squeeze(-1)
    df = (2.0 + squareplus(raw_df)).squeeze(-1)
    loc = loc.squeeze(-1)
    return df, loc, scale


def sample_student_t(df, loc, scale):
    return StudentT(df=df, loc=loc, scale=scale).sample()


def nll_student_t(df, loc, scale, targets):
    return -StudentT(df=df, loc=loc, scale=scale).log_prob(targets)


# ── Normal ───────────────────────────────────────────────────────────────────


def normal_params(hidden, w0, b0, w1, b1, **_ignored):
    """Project decoder hidden -> (loc, scale) for Normal.
    proj.0 -> loc, proj.1 -> scale (proj.2 unused for Normal)"""
    loc = (hidden @ w0.T + b0).squeeze(-1)
    raw_scale = hidden @ w1.T + b1
    eps = torch.finfo(raw_scale.dtype).eps
    scale = squareplus(raw_scale).clamp_min(eps).squeeze(-1)
    return loc, scale


def sample_normal(loc, scale):
    return Normal(loc=loc, scale=scale).sample()


def nll_normal(loc, scale, targets):
    return -Normal(loc=loc, scale=scale).log_prob(targets)


# ── Negative Binomial ────────────────────────────────────────────────────────


def negative_binomial_params(hidden, w0, b0, w1, b1, **_ignored):
    """Project decoder hidden -> (total_count, logits) for NegativeBinomial.
    Matches HF NegativeBinomialOutput.domain_map exactly (transformers/src/
    transformers/time_series_utils.py):
    proj.0 -> total_count (squareplus'd), proj.1 -> logits (raw, unsqueezed).

    Not an affine-transformable family — per HF's own comment, "We cannot
    scale using the affine transformation since negative binomial should
    return integers. Instead we scale the parameters." See generate()'s
    negative_binomial branch for the logits += scale.log() step.
    """
    raw_total_count = hidden @ w0.T + b0
    raw_logits = hidden @ w1.T + b1
    total_count = squareplus(raw_total_count).squeeze(-1)
    logits = raw_logits.squeeze(-1)
    return total_count, logits


def sample_negative_binomial(total_count, logits):
    return NegativeBinomial(total_count=total_count, logits=logits).sample().float()


def nll_negative_binomial(total_count, logits, targets):
    return -NegativeBinomial(total_count=total_count, logits=logits).log_prob(targets.clamp(min=0).round())


# ── ttnn projection path (unwired) ─────────────────────────────────────────
#
# Mirrors student_t_params / normal_params / negative_binomial_params above,
# operating on ttnn tensors so the projection could run inside a trace
# instead of after a full host readback. w0/w1/w2 here are pre-transposed to
# ttnn.linear's [in, out] convention (see tst_model.py load_weights()'s
# dist_head_ttnn), unlike the host dh["w0"] above, which is raw PyTorch
# Linear [out, in] used via hidden @ w0.T.
#
# NOT wired into any trace-capture path. No current call site uses these
# functions. PCC-gated against the host functions above via
# tests/test_tst_dist_head_fusion_pcc.py before any integration.

import ttnn

_SQUAREPLUS_EPS = 1.1920929e-07  # torch.finfo(torch.float32).eps, matches host path


def squareplus_ttnn(x):
    x2 = ttnn.multiply(x, x)
    inner = ttnn.add(x2, 4.0)
    root = ttnn.sqrt(inner)
    summed = ttnn.add(x, root)
    return ttnn.multiply(summed, 0.5)


def student_t_params_ttnn(hidden, dh_ttnn):
    """hidden: ttnn tensor [B, T, D_MODEL]. dh_ttnn: device dist_head_ttnn dict.
    Returns (df, loc, scale) as ttnn tensors, [B, T, 1] — squeeze(-1) is not
    applied here; it happens once after to_torch() in the caller, matching
    how the host projection's output is squeezed downstream.
    """
    raw_df = ttnn.linear(hidden, dh_ttnn["w0"], bias=dh_ttnn["b0"])
    loc = ttnn.linear(hidden, dh_ttnn["w1"], bias=dh_ttnn["b1"])
    raw_scale = ttnn.linear(hidden, dh_ttnn["w2"], bias=dh_ttnn["b2"])

    scale = squareplus_ttnn(raw_scale)
    scale = ttnn.clamp(scale, min=_SQUAREPLUS_EPS)
    df = ttnn.add(squareplus_ttnn(raw_df), 2.0)

    return df, loc, scale


def normal_params_ttnn(hidden, dh_ttnn):
    """Mirrors normal_params. dh_ttnn only needs w0/b0/w1/b1."""
    loc = ttnn.linear(hidden, dh_ttnn["w0"], bias=dh_ttnn["b0"])
    raw_scale = ttnn.linear(hidden, dh_ttnn["w1"], bias=dh_ttnn["b1"])
    scale = squareplus_ttnn(raw_scale)
    scale = ttnn.clamp(scale, min=_SQUAREPLUS_EPS)
    return loc, scale


def negative_binomial_params_ttnn(hidden, dh_ttnn):
    """Mirrors negative_binomial_params. dh_ttnn only needs w0/b0/w1/b1."""
    raw_total_count = ttnn.linear(hidden, dh_ttnn["w0"], bias=dh_ttnn["b0"])
    logits = ttnn.linear(hidden, dh_ttnn["w1"], bias=dh_ttnn["b1"])
    total_count = squareplus_ttnn(raw_total_count)
    return total_count, logits


# ── Dispatch (used by every generation entry point) ────────────────────────


def _distribution_head(hidden, weights):
    """hidden: torch [B, T, D_MODEL] -> distribution params tuple."""
    dh = weights["dist_head"]
    dt = weights.get("dist_type", "student_t")
    if dt == "normal":
        return normal_params(hidden, dh["w0"], dh["b0"], dh["w1"], dh["b1"])
    elif dt == "negative_binomial":
        return negative_binomial_params(hidden, dh["w0"], dh["b0"], dh["w1"], dh["b1"])
    else:
        return student_t_params(hidden, dh["w0"], dh["b0"], dh["w1"], dh["b1"], dh["w2"], dh["b2"])


def _sample_next_step(params, dist_type, _lc, _sc):
    """
    Shared sampling logic for generate() and generate_traced().
    Applies the second squeeze needed to collapse the leftover seq-len-1
    axis from slicing hidden as [:, -1:, :] before sampling.
    """
    if dist_type == "normal":
        loc_d, scale_d = params
        raw_loc = _lc + _sc * loc_d.squeeze(-1)
        raw_scale = _sc * scale_d.squeeze(-1)
        return sample_normal(raw_loc, raw_scale)
    elif dist_type == "negative_binomial":
        total_count, logits = params
        total_count = total_count.squeeze(-1)
        logits = logits.squeeze(-1)
        logits_scaled = logits + _sc.log()
        return sample_negative_binomial(total_count, logits_scaled)
    else:  # student_t
        df, loc_d, scale_d = params
        raw_loc = _lc + _sc * loc_d.squeeze(-1)
        raw_scale = _sc * scale_d.squeeze(-1)
        return sample_student_t(df.squeeze(-1), raw_loc, raw_scale)
