# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Format-state regressions for the experimental Quasar API.

Mixed input formats exercise descriptor kernels on WH/BH; the descriptor
factory is not executable on Quasar yet (QUASAR_PARITY_GAPS.md). Matching formats
also exercise the named-DFB factory. The Quasar-native factory currently rejects
activations. Passing on WH/BH does not validate Quasar-specific LLKs or packers.
"""

import math

import pytest
import torch

import ttnn

pytestmark = pytest.mark.nightly

_GRID = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
_FULL = (1, 1, 96, 96)  # nine tiles: repeated iterations; sharded cases also have a remainder
_SHAPES = [
    (_FULL, _FULL),
    ((1, 1, 1, 96), _FULL),
    (_FULL, (1, 1, 1, 96)),
    ((1, 1, 96, 1), _FULL),
    (_FULL, (1, 1, 96, 1)),
    ((1, 1, 1, 1), _FULL),
    (_FULL, (1, 1, 1, 1)),
    ((1, 1, 1, 96), (1, 1, 96, 1)),
    ((1, 1, 96, 1), (1, 1, 1, 96)),
]


def _input(device, shape, dtype, offset, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    # Small dyadic values make add/multiply exact in BF16, avoiding a PCC oracle
    # that could hide a handful of corrupt tiles. Always use quantized inputs.
    host = ((torch.arange(math.prod(shape)) * 7 + offset) % 17 - 8).float().reshape(shape) / 2
    tensor = ttnn.from_torch(host, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
    return tensor, ttnn.to_torch(tensor).float()


def _activations(side):
    relu = [ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)]
    return dict(
        input_tensor_a_activations=relu if side in ("lhs", "both") else [],
        input_tensor_b_activations=relu if side in ("rhs", "both") else [],
    )


@pytest.mark.parametrize("op_name", ["add", "multiply"])
@pytest.mark.parametrize(
    "a_dtype,b_dtype",
    [
        (ttnn.float32, ttnn.bfloat16),
        (ttnn.bfloat16, ttnn.float32),
        (ttnn.bfloat16, ttnn.bfloat16),
        (ttnn.float32, ttnn.float32),
        (ttnn.bfloat8_b, ttnn.bfloat8_b),
    ],
)
@pytest.mark.parametrize("shapes", _SHAPES)
@pytest.mark.parametrize("side", ["none", "lhs", "rhs", "both"])
def test_fused_activation_formats(device, op_name, a_dtype, b_dtype, shapes, side):
    if device.arch() == ttnn.device.Arch.QUASAR and a_dtype != b_dtype:
        pytest.skip("Mixed dtypes require the descriptor factory, which is not executable on Quasar yet")
    if device.arch() == ttnn.device.Arch.QUASAR and a_dtype != ttnn.bfloat16 and shapes[0] != shapes[1]:
        pytest.skip("Non-BF16 broadcasts require the descriptor factory, not yet executable on Quasar")
    a, ah = _input(device, shapes[0], a_dtype, 1)
    b, bh = _input(device, shapes[1], b_dtype, 5)
    if side in ("lhs", "both"):
        ah = ah.relu()
    if side in ("rhs", "both"):
        bh = bh.relu()
    expected = ah + bh if op_name == "add" else ah * bh
    out = getattr(ttnn.experimental.quasar, op_name)(
        a, b, dtype=ttnn.bfloat16, sub_core_grids=_GRID, **_activations(side)
    )
    torch.testing.assert_close(ttnn.to_torch(out).float(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("op_name", ["subtract", "multiply"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b])
@pytest.mark.parametrize("side", ["none", "lhs", "rhs", "both"])
def test_fused_scalar_activation_formats(device, op_name, dtype, side):
    if device.arch() == ttnn.device.Arch.QUASAR and dtype == ttnn.bfloat8_b:
        pytest.skip("Block-float scalars require the descriptor factory, not yet executable on Quasar")
    # Quasar's experimental API currently exposes tensor-first scalar operations.
    a, ah = _input(device, _FULL, dtype, 3)
    scalar = 2.5  # Keep RHS-RELU cases nonzero so a corrupt tensor cannot hide behind multiply-by-zero.
    if side in ("lhs", "both"):
        ah = ah.relu()
    bh = max(scalar, 0) if side in ("rhs", "both") else scalar
    expected = ah - bh if op_name == "subtract" else ah * bh
    out = getattr(ttnn.experimental.quasar, op_name)(
        a, scalar, dtype=ttnn.bfloat16, sub_core_grids=_GRID, **_activations(side)
    )
    torch.testing.assert_close(ttnn.to_torch(out).float(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("op_name", ["add", "multiply"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("side", ["none", "lhs", "rhs", "both"])
def test_fused_activation_sharded_chunks(device, op_name, dtype, side):
    # Matching dtypes avoid the separate mixed-format FPU batch-capacity bug.
    memory = ttnn.create_sharded_memory_config(
        _FULL, ttnn.CoreGrid(y=1, x=1), ttnn.ShardStrategy.HEIGHT, ttnn.ShardOrientation.ROW_MAJOR
    )
    a, ah = _input(device, _FULL, dtype, 1, memory)
    b, bh = _input(device, _FULL, dtype, 5, memory)
    if side in ("lhs", "both"):
        ah = ah.relu()
    if side in ("rhs", "both"):
        bh = bh.relu()
    expected = ah + bh if op_name == "add" else ah * bh
    out = getattr(ttnn.experimental.quasar, op_name)(
        a, b, dtype=ttnn.bfloat16, memory_config=memory, sub_core_grids=_GRID, **_activations(side)
    )
    torch.testing.assert_close(ttnn.to_torch(out).float(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat4_b])
@pytest.mark.parametrize("shapes", _SHAPES[:3])
def test_activation_intermediate_format(device, dtype, shapes):
    if device.arch() == ttnn.device.Arch.QUASAR and shapes[0] != shapes[1]:
        pytest.skip("Format-changing broadcasts require the descriptor factory, not yet executable on Quasar")
    a, ah = _input(device, shapes[0], dtype, 1)
    b, bh = _input(device, shapes[1], dtype, 5)
    # LOGADDEXP's implicit EXP preprocessing changes block-float inputs to BF16
    # intermediates. The SrcA reference must describe the intermediate, not input.
    out = ttnn.experimental.quasar.logaddexp(a, b, dtype=ttnn.bfloat16, sub_core_grids=_GRID)
    torch.testing.assert_close(ttnn.to_torch(out).float(), torch.logaddexp(ah, bh), rtol=0.03, atol=0.03)
