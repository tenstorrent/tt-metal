# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Verifier-added extended tests for groupnorm_sc_N_1_HW_C.

Covers exactly the gaps found in Phase-0 verification, nothing more:
- TILE-layout gamma/beta (promoted into SUPPORTED["affine_layout"] by the
  verifier; acceptance suite only exercises ROW_MAJOR affine).
- Largest single-group L1-fitting footprint (G=1, C=1024, Wg=32 with
  gamma+beta — five 2*Wg stream CBs + 2*Wt affine pages just fit; C=2048
  OOMs, see verification_report.md).
- Many-(n,g) loop accounting with batched reader/writer (8 batches x 4 groups).
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C

PCC_BF16 = 0.995


def torch_groupnorm(x, num_groups, gamma, beta, eps=1e-5):
    N, _, HW, C = x.shape
    x_nchw = x.to(torch.float32).squeeze(1).permute(0, 2, 1)
    w = gamma.to(torch.float32).reshape(C)
    b = beta.to(torch.float32).reshape(C)
    y = torch.nn.functional.group_norm(x_nchw, num_groups, weight=w, bias=b, eps=eps)
    return y.permute(0, 2, 1).unsqueeze(1)


def run(device, shape, num_groups, affine_layout):
    torch.manual_seed(7)
    N, _, HW, C = shape
    x = torch.randn(shape, dtype=torch.bfloat16)
    gamma = torch.randn(1, 1, 1, C, dtype=torch.bfloat16)
    beta = torch.randn(1, 1, 1, C, dtype=torch.bfloat16)
    expected = torch_groupnorm(x, num_groups, gamma, beta)

    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_g = ttnn.from_torch(gamma, dtype=ttnn.bfloat16, layout=affine_layout, device=device)
    tt_b = ttnn.from_torch(beta, dtype=ttnn.bfloat16, layout=affine_layout, device=device)

    out = groupnorm_sc_N_1_HW_C(tt_x, num_groups, gamma=tt_g, beta=tt_b)
    assert out.layout == ttnn.TILE_LAYOUT
    assert_with_pcc(expected, ttnn.to_torch(out).to(torch.float32), pcc=PCC_BF16)


def test_tile_layout_affine(device):
    run(device, (1, 1, 64, 128), 4, ttnn.TILE_LAYOUT)


def test_single_group_max_l1_footprint(device):
    # Wg = 32: five 2*Wg-page stream CBs + 64 affine pages — largest fitting case
    run(device, (1, 1, 32, 1024), 1, ttnn.ROW_MAJOR_LAYOUT)


def test_many_batches_many_groups(device):
    run(device, (8, 1, 64, 128), 4, ttnn.ROW_MAJOR_LAYOUT)
