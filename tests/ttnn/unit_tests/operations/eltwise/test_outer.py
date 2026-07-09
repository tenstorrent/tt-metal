# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from functools import partial
from models.common.utility_functions import torch_random
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import assert_allclose, assert_with_pcc

pytestmark = pytest.mark.use_module_device

# Scalar bf8 outer: looser than comp_pcc fallback; seed=0 ~1.125 abs diff, atol covers near-zero products.
_BF8_SCALAR_OUTER_RTOL = 0.05
_BF8_SCALAR_OUTER_ATOL = 2.0


# Contract: a:[..., N], b:[..., M] -> [..., N, M].
# The last dim of each input is the vector; leading dims broadcast.
SHAPE_PAIRS = [
    # Unbatched 1D x 1D
    ([32], [32]),
    ([2048], [64]),
    ([64], [2048]),
    ([30], [30]),  # non-tile-aligned
    ([1], [32]),
    ([1], [1]),
    # Same-rank batched
    ([2, 32], [2, 32]),
    ([4, 8, 32], [4, 8, 32]),
    # Broadcast on batch
    ([2, 32], [1, 32]),
    ([1, 8, 32], [4, 8, 32]),
    # Rank-mismatch broadcast (b's missing leading dims treated as 1)
    ([2, 3, 32], [32]),
    # Rank-4 inputs (mirrors original eager test shapes; output is 5-D)
    ([1, 1, 1, 32], [1, 1, 1, 32]),
    ([1, 1, 1, 2048], [1, 1, 1, 64]),  # LLaMa dimensions
]


def _pcc_threshold(dtype):
    # bfloat8_b quantizes per-tile; batched outer accumulates quantization noise
    # across the broadcast volume, so a 0.98 threshold matches realistic block-float
    # behavior on these sizes.
    return 0.98 if dtype == ttnn.bfloat8_b else 0.9999


def _golden(a_pt, b_pt):
    if a_pt.dim() == 1 and b_pt.dim() == 1:
        return torch.outer(a_pt, b_pt)
    return torch.einsum("...i,...j->...ij", a_pt, b_pt)


@pytest.mark.parametrize("a_shape, b_shape", SHAPE_PAIRS)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32])
@pytest.mark.parametrize(
    "mem_a, mem_b",
    [
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.L1_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
        (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
    ],
)
def test_outer_tile_layout(a_shape, b_shape, dtype, mem_a, mem_b, device):
    torch.manual_seed(0)

    gen = partial(torch_random, low=-100, high=100, dtype=torch.float32)
    a_pt = gen_func_with_cast_tt(gen, dtype)(a_shape)
    b_pt = gen_func_with_cast_tt(gen, dtype)(b_shape)

    a_tt = ttnn.from_torch(a_pt, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mem_a)
    b_tt = ttnn.from_torch(b_pt, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mem_b)

    out_tt = ttnn.outer(a_tt, b_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out_pt = _golden(a_pt, b_pt)
    out_actual = ttnn.to_torch(out_tt)

    assert tuple(out_actual.shape) == tuple(
        out_pt.shape
    ), f"expected shape {tuple(out_pt.shape)}, got {tuple(out_actual.shape)}"
    # BFLOAT8_B scalar output: PCC undefined; use relaxed allclose for quant error.
    if dtype == ttnn.bfloat8_b and out_actual.numel() == 1:
        assert_allclose(
            out_pt,
            out_actual,
            rtol=_BF8_SCALAR_OUTER_RTOL,
            atol=_BF8_SCALAR_OUTER_ATOL,
        )
    else:
        assert_with_pcc(out_pt, out_actual, pcc=_pcc_threshold(dtype))


@pytest.mark.parametrize("a_shape, b_shape", SHAPE_PAIRS)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_outer_row_major_layout(a_shape, b_shape, dtype, device):
    # ROW_MAJOR inputs are internally converted to TILE by binary_ng; verify
    # unsqueeze + multiply on RM-laid-out inputs still works end-to-end.
    # Block-float dtypes require TILE layout and are exercised in the matrix above.
    torch.manual_seed(0)

    gen = partial(torch_random, low=-100, high=100, dtype=torch.float32)
    a_pt = gen_func_with_cast_tt(gen, dtype)(a_shape)
    b_pt = gen_func_with_cast_tt(gen, dtype)(b_shape)

    a_tt = ttnn.from_torch(a_pt, dtype=dtype, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    b_tt = ttnn.from_torch(b_pt, dtype=dtype, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    out_tt = ttnn.outer(a_tt, b_tt)
    out_pt = _golden(a_pt, b_pt)

    assert tuple(ttnn.to_torch(out_tt).shape) == tuple(out_pt.shape)
    assert_with_pcc(out_pt, ttnn.to_torch(out_tt), pcc=_pcc_threshold(dtype))


# Note: ttnn.bfloat4_b is not exercised here because ttnn.outer relies on
# ttnn.unsqueeze, which routes through reshape_device_operation; that op
# only supports bfloat16/float32/int32/uint32. bfloat8_b is the only
# block-float dtype with a working code path through reshape.


def test_outer_default_memory_config(device):
    # Smoke test: ttnn.outer(a, b) with no memory_config kwarg uses the default.
    torch.manual_seed(0)
    a_pt = torch.rand([32], dtype=torch.bfloat16)
    b_pt = torch.rand([64], dtype=torch.bfloat16)

    a_tt = ttnn.from_torch(a_pt, device=device, layout=ttnn.TILE_LAYOUT)
    b_tt = ttnn.from_torch(b_pt, device=device, layout=ttnn.TILE_LAYOUT)

    out_tt = ttnn.outer(a_tt, b_tt)
    out_pt = _golden(a_pt, b_pt)

    assert tuple(ttnn.to_torch(out_tt).shape) == tuple(out_pt.shape)
    assert_with_pcc(out_pt, ttnn.to_torch(out_tt), pcc=0.9999)
