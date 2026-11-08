# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

# Shape from the differential probe: 33x70 is not tile-aligned, so the TILE predicate and the
# ROW_MAJOR operand hold their values in different orders in memory.
SHAPE = (1, 1, 33, 70)


def device_tensor(device, tensor, layout):
    return ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=layout, device=device)


@pytest.mark.parametrize("variant", ["TTS", "TST"])
@pytest.mark.parametrize(
    "predicate_layout, operand_layout",
    [(ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT), (ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT)],
)
def test_where_with_scalar_rejects_mixed_layouts(device, expect_error, variant, predicate_layout, operand_layout):
    predicate = device_tensor(device, torch.rand(SHAPE).round(), predicate_layout)
    operand = device_tensor(device, torch.randn(SHAPE), operand_layout)
    with expect_error(RuntimeError, "same layout"):
        if variant == "TTS":
            ttnn.where(predicate, operand, 0.5)
        else:
            ttnn.where(predicate, 0.5, operand)


@pytest.mark.parametrize("variant", ["TTS", "TST"])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_where_with_scalar_same_layout_matches_torch(device, variant, layout):
    torch.manual_seed(0)
    predicate_t = torch.rand(SHAPE).round()
    operand_t = torch.randn(SHAPE)
    scalar = 0.5
    if variant == "TTS":
        golden = torch.where(predicate_t.bool(), operand_t, torch.full(SHAPE, scalar)).bfloat16()
    else:
        golden = torch.where(predicate_t.bool(), torch.full(SHAPE, scalar), operand_t).bfloat16()

    predicate = device_tensor(device, predicate_t.bfloat16(), layout)
    operand = device_tensor(device, operand_t.bfloat16(), layout)
    if variant == "TTS":
        result = ttnn.where(predicate, operand, scalar)
    else:
        result = ttnn.where(predicate, scalar, operand)
    assert torch.equal(ttnn.to_torch(result).reshape(SHAPE).bfloat16(), golden)
