# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""BF16 SFPU tensor-tensor binary comparisons (eq/ne/lt/gt/le/ge) on Blackhole and Wormhole."""

import struct

import pytest
import torch
import ttnn

pytestmark = pytest.mark.use_module_device


# BF16 special values via round-trip through float32 bit patterns.
# BF16 uses the top 16 bits of IEEE-754 float32, so build values via int→float32→bfloat16.
def _bf16_from_bits(bits16: int) -> float:
    """Expand a 16-bit BF16 bit pattern to float32 (zero-pad lower 16 mantissa bits)."""
    packed = struct.pack(">I", bits16 << 16)
    return struct.unpack(">f", packed)[0]


_SPECIAL_BF16_GRID = torch.tensor(
    [
        _bf16_from_bits(0x0000),  # +0.0
        _bf16_from_bits(0x8000),  # -0.0
        _bf16_from_bits(0x3F80),  # +1.0
        _bf16_from_bits(0xBF80),  # -1.0
        _bf16_from_bits(0x45CD),  # ~6576.0 — an arbitrary finite
        _bf16_from_bits(0x7F80),  # +inf
        _bf16_from_bits(0xFF80),  # -inf
        _bf16_from_bits(0x7FC0),  # quiet NaN
    ],
    dtype=torch.float32,
).to(torch.bfloat16)


def _bool_mask(tt: torch.Tensor) -> torch.Tensor:
    """Normalize device output (float/bool/int) to a boolean mask for comparison with torch.*."""
    if tt.dtype == torch.bool:
        return tt
    if tt.is_floating_point():
        return tt != 0.0
    return tt != 0


def _assert_matches_torch(ttnn_op, torch_a: torch.Tensor, torch_b: torch.Tensor, device, mem_cfg=None):
    golden_fn = ttnn.get_golden_function(ttnn_op)
    golden = golden_fn(torch_a, torch_b)
    assert golden.dtype == torch.bool

    ta = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tb = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    kwargs = {}
    if mem_cfg is not None:
        kwargs["memory_config"] = mem_cfg
    out = ttnn_op(ta, tb, **kwargs)
    out_t = ttnn.to_torch(out)
    assert torch.equal(_bool_mask(out_t), golden), f"mismatch for {ttnn_op.__name__}"


@pytest.mark.parametrize(
    "input_shapes",
    (
        torch.Size([64, 64]),
        torch.Size([1, 1, 32, 32]),
        torch.Size([1, 3, 320, 384]),
    ),
)
@pytest.mark.parametrize(
    "mem_cfg",
    (
        None,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
    ),
)
@pytest.mark.parametrize(
    "ttnn_op",
    (ttnn.eq, ttnn.ne, ttnn.gt, ttnn.lt, ttnn.ge, ttnn.le),
)
def test_binary_comp_bf16_tensor_tensor_random(input_shapes, mem_cfg, ttnn_op, device):
    torch.manual_seed(0)
    a = torch.empty(input_shapes, dtype=torch.bfloat16).uniform_(-1000.0, 1000.0)
    b = torch.empty(input_shapes, dtype=torch.bfloat16).uniform_(-1000.0, 1000.0)
    _assert_matches_torch(ttnn_op, a, b, device, mem_cfg=mem_cfg)


@pytest.mark.parametrize(
    "ttnn_op",
    (ttnn.eq, ttnn.ne, ttnn.gt, ttnn.lt, ttnn.ge, ttnn.le),
)
def test_binary_comp_bf16_broadcast_rhs_singleton(ttnn_op, device):
    """RHS broadcasts from a scalar-shaped tensor (1,1,1)."""
    torch.manual_seed(1)
    a = torch.empty((2, 64, 64), dtype=torch.bfloat16).uniform_(-50.0, 50.0)
    b = torch.tensor([[[3.25]]], dtype=torch.bfloat16)
    _assert_matches_torch(ttnn_op, a, b, device)


@pytest.mark.parametrize(
    "ttnn_op",
    (ttnn.eq, ttnn.ne, ttnn.gt, ttnn.lt, ttnn.ge, ttnn.le),
)
def test_binary_comp_bf16_special_floats(ttnn_op, device):
    """All pairs from `_SPECIAL_BF16_GRID` (Cartesian product): zeros, finites, ±inf ties, NaN/unordered."""
    pairs = torch.cartesian_prod(_SPECIAL_BF16_GRID, _SPECIAL_BF16_GRID)
    a = pairs[:, 0].unsqueeze(0).expand(32, -1)
    b = pairs[:, 1].unsqueeze(0).expand(32, -1)
    _assert_matches_torch(ttnn_op, a, b, device)


@pytest.mark.parametrize(
    "ttnn_op",
    (ttnn.eq, ttnn.ne, ttnn.gt, ttnn.lt, ttnn.ge, ttnn.le),
)
def test_binary_comp_bf16_adjacent_ulps(ttnn_op, device):
    """Nearest BF16 neighbors (1 ULP apart) — exercises the border of the 7-bit mantissa."""
    # BF16 ULP at 1.0 is 2^-7 = 0.0078125; just below 1.0 is 0.99609375 (0x3F7F).
    one = torch.tensor(_bf16_from_bits(0x3F80), dtype=torch.bfloat16)  # 1.0
    one_up = torch.tensor(_bf16_from_bits(0x3F81), dtype=torch.bfloat16)  # 1.0 + 1 ULP
    one_dn = torch.tensor(_bf16_from_bits(0x3F7F), dtype=torch.bfloat16)  # 1.0 - 1 ULP
    m_one = torch.tensor(_bf16_from_bits(0xBF80), dtype=torch.bfloat16)  # -1.0
    m_one_up = torch.tensor(_bf16_from_bits(0xBF7F), dtype=torch.bfloat16)  # -1.0 + 1 ULP
    m_one_dn = torch.tensor(_bf16_from_bits(0xBF81), dtype=torch.bfloat16)  # -1.0 - 1 ULP

    vals_a = torch.stack([one, one_up, one_dn, m_one, m_one_dn, m_one])
    vals_b = torch.stack([one_up, one, one, m_one_dn, m_one, m_one_up])
    a = vals_a.unsqueeze(0).expand(32, -1)
    b = vals_b.unsqueeze(0).expand(32, -1)
    _assert_matches_torch(ttnn_op, a, b, device)


@pytest.mark.parametrize(
    "ttnn_op",
    (ttnn.eq, ttnn.ne, ttnn.gt, ttnn.lt, ttnn.ge, ttnn.le),
)
def test_binary_comp_bf16_exact_equal_tile(ttnn_op, device):
    """Two identical tiles (all elements pairwise equal)."""
    torch.manual_seed(2)
    a = torch.empty((128, 128), dtype=torch.bfloat16).uniform_(-10.0, 10.0)
    same = torch.cat([a, a], dim=0)
    _assert_matches_torch(ttnn_op, same[:128], same[128:], device)
