# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

import torch
from torch.distributions import NegativeBinomial, Normal, StudentT


def squareplus(x):
    return 0.5 * (x + torch.sqrt(x * x + 4.0))


# ── Student-T ────────────────────────────────────────────────────────────────


def student_t_params(hidden, w0, b0, w1, b1, w2, b2):
    """Project decoder hidden -> (df, loc, scale) for Student-T.
    proj.0 -> df, proj.1 -> loc, proj.2 -> scale  (HF convention)"""
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
    proj.0 -> loc, proj.1 -> scale  (proj.2 unused for Normal)"""
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
    Matches HF NegativeBinomialOutput.domain_map exactly (verified against
    transformers/src/transformers/time_series_utils.py):
    proj.0 -> total_count (squareplus'd), proj.1 -> logits (raw, unsqueezed).
    NOTE: unlike StudentT/Normal, this is NOT an affine-transformable family --
    HF's own comment: "We cannot scale using the affine transformation since
    negative binomial should return integers. Instead we scale the parameters."
    See generate()'s negative_binomial branch for the logits += scale.log() step."""
    raw_total_count = hidden @ w0.T + b0
    raw_logits = hidden @ w1.T + b1
    total_count = squareplus(raw_total_count).squeeze(-1)
    logits = raw_logits.squeeze(-1)
    return total_count, logits


def sample_negative_binomial(total_count, logits):
    return NegativeBinomial(total_count=total_count, logits=logits).sample().float()


def nll_negative_binomial(total_count, logits, targets):
    return -NegativeBinomial(total_count=total_count, logits=logits).log_prob(targets.clamp(min=0).round())
