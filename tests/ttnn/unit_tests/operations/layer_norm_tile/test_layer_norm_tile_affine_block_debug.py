# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 5 debug: deterministic gamma = 1, beta = 1 + (column % 32) / 8 (exact in bf16), so out - normalize(x) reads back beta
per column and shows which tile columns / rows the grouped affine block misplaces."""

import pytest
import torch
import ttnn

from ttnn.operations.layer_norm_tile import layer_norm
from tests.ttnn.unit_tests.operations.layer_norm_tile.test_layer_norm_tile import reference_layer_norm, to_device


@pytest.mark.parametrize("width, stick", [(224, 224), (224, 32), (32, 32), (256, 32)])
def test_beta_readback(device, width, stick):
    torch.manual_seed(7)
    shape = (2, 1, 64, width)
    x = torch.randn(shape).to(torch.bfloat16)
    gamma = torch.ones(width).to(torch.bfloat16)
    beta = (1 + (torch.arange(width) % 32) / 8).to(torch.bfloat16)

    def aff(v):
        return to_device(
            v.reshape(1, 1, width // stick, stick), device, ttnn.BufferType.L1, layout=ttnn.ROW_MAJOR_LAYOUT
        )

    out = ttnn.to_torch(layer_norm(to_device(x, device, ttnn.BufferType.DRAM), aff(gamma), aff(beta), None)).float()
    exp = reference_layer_norm(x, gamma, beta, None).float()
    err = (out - exp).abs()
    bad_cols = (err.amax(dim=(0, 1, 2)) > 0.1).nonzero().flatten().tolist()
    bad_rows = (err.amax(dim=(1, 3)) > 0.1).nonzero().tolist()
    got_beta = (out - (exp - beta))[0, 0, 0, :].tolist()
    # Found: the injector writer's trid barrier right after a single beta read passed early, so the face-1
    # halves of every beta tile (columns 16-31 of each tile) read back 0 with one group per tensor.
    assert not bad_cols, (
        f"bad cols {bad_cols[:64]} ({len(bad_cols)}), bad (b,row) {bad_rows[:8]} ({len(bad_rows)}), "
        f"beta read back at row 0: {[round(v, 1) for v in got_beta[:64]]}"
    )
