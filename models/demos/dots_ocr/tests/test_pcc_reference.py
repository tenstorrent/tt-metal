# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Sanity tests for :mod:`models.demos.dots_ocr.reference.pcc` (shared TT vs HF gate)."""

import torch

from models.demos.dots_ocr.reference.pcc import comp_pcc


def test_comp_pcc_identical_is_one():
    x = torch.randn(3, 5)
    assert abs(comp_pcc(x, x) - 1.0) < 1e-5


def test_comp_pcc_opposite_sign_is_minus_one_1d():
    t = torch.randn(8)
    assert abs(comp_pcc(t, -t) + 1.0) < 1e-4


def test_comp_pcc_matches_corrcoef_1d():
    a = torch.randn(100)
    b = 0.3 * a + torch.randn(100) * 0.1
    p1 = comp_pcc(a, b)
    p2 = torch.corrcoef(torch.stack([a.float().flatten(), b.float().flatten()]))[0, 1].item()
    assert abs(p1 - p2) < 1e-4
