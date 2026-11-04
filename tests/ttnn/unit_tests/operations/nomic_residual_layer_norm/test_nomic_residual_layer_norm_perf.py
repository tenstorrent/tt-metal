# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf-case driver for nomic_residual_layer_norm (feature_spec LOOSE_CASES shapes).

Run under `scripts/run_safe_pytest.sh --profile` and read DEVICE KERNEL DURATION from the per-op CSV. Each case
runs the op twice; correctness is checked loosely here (the acceptance test is the correctness gate).
"""

import pytest
import torch
import ttnn

from ttnn.operations.nomic_residual_layer_norm import (
    default_compute_kernel_config,
    input_memory_config,
    nomic_residual_layer_norm,
)
from ttnn.operations.nomic_residual_layer_norm.nomic_residual_layer_norm import model_memory_config

HIDDEN = 768


@pytest.mark.parametrize(
    "batch, seq, with_residual",
    [(8, 512, True), (4, 512, True), (2, 512, True), (8, 512, False), (7, 512, True)],
    ids=["8x512-residual", "4x512-residual", "2x512-residual", "8x512-no_residual", "7x512-residual"],
)
@pytest.mark.parametrize("placement", ["op_layout", "model_layout"])
def test_perf_case(device, batch, seq, with_residual, placement):
    shape = (batch, 1, seq, HIDDEN)
    torch.manual_seed(0)
    x = torch.randn(shape).to(torch.bfloat16)
    r = torch.randn(shape).to(torch.bfloat16) if with_residual else None
    gamma = 1.0 + 0.25 * torch.randn(HIDDEN)
    beta = (0.1 * torch.randn(HIDDEN)).to(torch.bfloat16)
    layout = (model_memory_config if placement == "model_layout" else input_memory_config)(shape, device)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=layout)
    tt_r = (
        ttnn.from_torch(r, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=layout)
        if with_residual
        else None
    )
    tt_g = ttnn.from_torch(gamma.reshape(1, 1, 1, HIDDEN), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tt_b = ttnn.from_torch(beta.reshape(1, 1, 1, HIDDEN), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    for _ in range(2):
        out = nomic_residual_layer_norm(
            tt_x, tt_g, tt_b, residual=tt_r, epsilon=1e-5, compute_kernel_config=default_compute_kernel_config()
        )
    y = x.double() + (r.double() if with_residual else 0)
    expected = torch.nn.functional.layer_norm(y, (HIDDEN,), gamma.double(), beta.double(), eps=1e-5)
    got = ttnn.to_torch(out).double()
    assert torch.corrcoef(torch.stack([got.flatten(), expected.flatten()]))[0, 1] > 0.9999
