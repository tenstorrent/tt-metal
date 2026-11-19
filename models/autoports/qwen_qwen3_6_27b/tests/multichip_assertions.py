# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed numerical oracle shared by the multichip runners."""

import math

import torch

from models.autoports.qwen_qwen3_6_27b.tests.run_decoder import pcc as correlation


def pcc(a, b):
    assert torch.isfinite(a).all() and torch.isfinite(b).all(), "Non-finite PCC input"
    if torch.equal(a, b):
        return 1.0
    score = correlation(a, b)
    assert math.isfinite(score), "Undefined PCC for unequal constant tensors"
    return score
