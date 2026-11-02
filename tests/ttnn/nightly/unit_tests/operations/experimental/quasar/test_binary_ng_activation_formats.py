# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Format-state regressions for the experimental Quasar API.

Every case runs on Quasar and on WH/BH. See
QUASAR_PARITY_GAPS.md for the factory capability boundary. On Quasar, the
no-broadcast tensor-tensor cases that the native gate admits run on the
Quasar-native factory, unless TTNN_QSR_NATIVE=0.
Passing on WH/BH does not validate Quasar-specific LLKs or packers.
"""

import math

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_allclose, assert_equal

# Every test in this module takes the function-scoped `device` fixture without
# parametrizing device_params, so the device is opened once per module rather than
# once per test.
pytestmark = pytest.mark.use_module_device

_GRID = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
_FULL = (1, 1, 96, 96)
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

_SHARED_DFB_CASES = [(ttnn.bfloat16, ttnn.bfloat16, shapes) for shapes in _SHAPES] + [
    (dtype, dtype, (_FULL, _FULL)) for dtype in (ttnn.float32, ttnn.bfloat8_b)
]


def _input(device, shape, dtype, offset, memory_config=ttnn.DRAM_MEMORY_CONFIG):
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
    "a_dtype,b_dtype,shapes",
    [pytest.param(*case, id=f"shared_dfb-{i}") for i, case in enumerate(_SHARED_DFB_CASES)],
)
@pytest.mark.parametrize("side", ["none", "lhs", "rhs", "both"])
def test_fused_activation_formats(device, op_name, a_dtype, b_dtype, shapes, side):
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
    assert_equal(expected, ttnn.to_torch(out).float())


@pytest.mark.parametrize("op_name", ["subtract", "multiply"])
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(ttnn.bfloat16, id="shared_dfb-bf16"),
        pytest.param(ttnn.float32, id="shared_dfb-fp32"),
    ],
)
@pytest.mark.parametrize("side", ["none", "lhs", "rhs", "both"])
def test_fused_scalar_activation_formats(device, op_name, dtype, side):
    a, ah = _input(device, _FULL, dtype, 3)
    scalar = 2.5  # Keep RHS-RELU cases nonzero so a corrupt tensor cannot hide behind multiply-by-zero.
    if side in ("lhs", "both"):
        ah = ah.relu()
    bh = max(scalar, 0) if side in ("rhs", "both") else scalar
    expected = ah - bh if op_name == "subtract" else ah * bh
    out = getattr(ttnn.experimental.quasar, op_name)(
        a, scalar, dtype=ttnn.bfloat16, sub_core_grids=_GRID, **_activations(side)
    )
    assert_equal(expected, ttnn.to_torch(out).float())


@pytest.mark.parametrize("op_name", ["add", "multiply"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("side", ["none", "lhs", "rhs", "both"])
def test_fused_activation_sharded_chunks(device, op_name, dtype, side):
    # Matching dtypes: mixed a and b dtypes are not supported.
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
    assert_equal(expected, ttnn.to_torch(out).float())


@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat4_b])
@pytest.mark.parametrize(
    "shapes",
    [
        pytest.param(_SHAPES[0], id="shared_dfb-full"),
    ],
)
def test_activation_intermediate_format(device, dtype, shapes):
    a, ah = _input(device, shapes[0], dtype, 1)
    b, bh = _input(device, shapes[1], dtype, 5)
    # LOGADDEXP's implicit EXP preprocessing changes block-float inputs to BF16
    # intermediates. The SrcA reference must describe the intermediate, not input.
    out = ttnn.experimental.quasar.logaddexp(a, b, dtype=ttnn.bfloat16, sub_core_grids=_GRID)
    assert_allclose(torch.logaddexp(ah, bh), ttnn.to_torch(out).float(), rtol=0.03, atol=0.03)
