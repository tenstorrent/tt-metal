# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

import math
import torch
from torch.distributions import StudentT, Normal, NegativeBinomial


def squareplus(x):
    return 0.5 * (x + torch.sqrt(x * x + 4.0))


def student_t_params(hidden, w0, b0, w1, b1, w2, b2):
    """Project decoder hidden -> (df, loc, scale) for Student-T."""
    raw_df    = hidden @ w0.T + b0          # [..., 1]
    loc       = hidden @ w1.T + b1          # [..., 1]
    raw_scale = hidden @ w2.T + b2          # [..., 1]
    eps = torch.finfo(raw_scale.dtype).eps
    scale = squareplus(raw_scale).clamp_min(eps).squeeze(-1)
    df    = (2.0 + squareplus(raw_df)).squeeze(-1)
    loc   = loc.squeeze(-1)
    return df, loc, scale


def normal_params(hidden, w0, b0, w1, b1):
    """Project decoder hidden -> (loc, scale) for Normal."""
    loc       = (hidden @ w0.T + b0).squeeze(-1)
    raw_scale = (hidden @ w1.T + b1)
    eps = torch.finfo(raw_scale.dtype).eps
    scale = squareplus(raw_scale).clamp_min(eps).squeeze(-1)
    return loc, scale


def sample_student_t(df, loc, scale, num_samples):
    """Sample [num_samples, B, T] from Student-T distribution."""
    dist = StudentT(df=df.unsqueeze(0), loc=loc.unsqueeze(0), scale=scale.unsqueeze(0))
    return dist.sample((num_samples,))   # [S, B, T]


def nll_student_t(df, loc, scale, targets):
    """Negative log-likelihood of targets under Student-T."""
    dist = StudentT(df=df, loc=loc, scale=scale)
    return -dist.log_prob(targets)
