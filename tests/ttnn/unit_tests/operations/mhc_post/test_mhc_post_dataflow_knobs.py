# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Dataflow-knob coverage (Refinement 3): both COEF_EXPANDER placements (writer default, reader) and the
block-size policy at its default, at the coarsest L1 fit (no cap, 1 block per core minimum), and at small
caps (ragged tails, many blocks per segment). The shapes put segment boundaries inside core ranges
(T640 C1792: 10-11 units per core over 56-unit rows), cover a non-aligned token dim and n = 5."""

import pytest
import torch
import ttnn

import ttnn.operations.mhc_post.mhc_post_program_descriptor as pd
from ttnn.operations.mhc_post import mhc_post

from .test_mhc_post import reference_mhc_post, to_device

TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16}

BLOCK_POLICIES = [
    pytest.param(None, None, id="default_policy"),
    pytest.param(None, 1, id="coarsest_fit"),
    pytest.param(1, None, id="B1"),
    pytest.param(3, None, id="B3"),
]


@pytest.mark.parametrize("expander", ["writer", "reader"])
@pytest.mark.parametrize("max_block, min_blocks", BLOCK_POLICIES)
@pytest.mark.parametrize(
    "n, T, C, dtype",
    [
        pytest.param(4, 640, 1792, ttnn.bfloat16, id="n4_T640_C1792_bf16"),
        pytest.param(4, 100, 32 * 7, ttnn.float32, id="n4_T100_C224_fp32_non_aligned"),
        pytest.param(5, 64, 32 * 5, ttnn.bfloat16, id="n5_T64_C160_bf16"),
    ],
)
def test_mhc_post_dataflow_knobs(device, monkeypatch, expander, max_block, min_blocks, n, T, C, dtype):
    monkeypatch.setattr(pd, "COEF_EXPANDER", expander)
    if max_block is not None:
        monkeypatch.setattr(pd, "MAX_BLOCK_COL_TILES", max_block)
    if min_blocks is not None:
        monkeypatch.setattr(pd, "MAX_BLOCK_COL_TILES", None)
        monkeypatch.setattr(pd, "MIN_BLOCKS_PER_CORE", min_blocks)
    torch.manual_seed(n + T)
    f = torch.randn(1, T, C).to(TORCH_DTYPE[dtype])
    x = torch.randn(1, T, n * C).to(TORCH_DTYPE[dtype])
    post = 2.0 * torch.rand(1, T, n)
    comb = torch.rand(1, T, n * n)
    out = mhc_post(
        to_device(f, device, dtype),
        to_device(x, device, dtype),
        to_device(post, device, ttnn.float32),
        to_device(comb, device, ttnn.float32),
    )
    got = ttnn.to_torch(out).float()
    ref = reference_mhc_post(f, x, post, comb)
    if dtype == ttnn.float32:
        torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)
    else:
        torch.testing.assert_close(got, ref.bfloat16().float(), rtol=1.6e-2, atol=1e-2)
