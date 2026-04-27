# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""FP32 SFPU tensor–tensor binary comparisons (eq/ne/lt/gt/le/ge) on Blackhole and Wormhole."""

import pytest
import torch
import ttnn

pytestmark = pytest.mark.use_module_device

# Scalars for Cartesian product in special_floats test: zeros, ±finite, ±inf, NaN.
_SPECIAL_FP32_GRID = torch.tensor(
    (
        0.0,
        -0.0,
        1.0,
        -1.0,
        6573.837573,
        float("inf"),
        float("-inf"),
        float("nan"),
    ),
    dtype=torch.float32,
)


def _bool_mask(tt: torch.Tensor) -> torch.Tensor:
    """Normalize device output (float/bool/int) to a boolean mask for comparison with torch.*."""
    if tt.dtype == torch.bool:
        return tt
    if tt.is_floating_point():
        return tt != 0.0
    return tt != 0


def _assert_matches_torch(ttnn_op, torch_a: torch.Tensor, torch_b: torch.Tensor, device, *, mem_cfg=None):
    golden_fn = ttnn.get_golden_function(ttnn_op)
    golden = golden_fn(torch_a, torch_b)
    assert golden.dtype == torch.bool

    ta = ttnn.from_torch(torch_a, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tb = ttnn.from_torch(torch_b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
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
def test_binary_comp_fp32_tensor_tensor_random(input_shapes, mem_cfg, ttnn_op, device):
    torch.manual_seed(0)
    a = torch.empty(input_shapes, dtype=torch.float32).uniform_(-1000.0, 1000.0)
    b = torch.empty(input_shapes, dtype=torch.float32).uniform_(-1000.0, 1000.0)
    _assert_matches_torch(ttnn_op, a, b, device, mem_cfg=mem_cfg)


@pytest.mark.parametrize(
    "ttnn_op",
    (ttnn.eq, ttnn.ne, ttnn.gt, ttnn.lt, ttnn.ge, ttnn.le),
)
def test_binary_comp_fp32_broadcast_rhs_singleton(ttnn_op, device):
    """RHS broadcasts from a scalar-shaped tensor (1,1,1)."""
    torch.manual_seed(1)
    a = torch.empty((2, 64, 64), dtype=torch.float32).uniform_(-50.0, 50.0)
    b = torch.tensor([[[3.25]]], dtype=torch.float32)
    _assert_matches_torch(ttnn_op, a, b, device)


@pytest.mark.parametrize(
    "ttnn_op",
    (ttnn.eq, ttnn.ne, ttnn.gt, ttnn.lt, ttnn.ge, ttnn.le),
)
def test_binary_comp_fp32_special_floats(ttnn_op, device):
    """All pairs from `_SPECIAL_FP32_GRID` (Cartesian product): zeros, finites, ±inf ties, NaN/unordered."""
    pairs = torch.cartesian_prod(_SPECIAL_FP32_GRID, _SPECIAL_FP32_GRID)
    a = pairs[:, 0].unsqueeze(0).expand(32, -1)
    b = pairs[:, 1].unsqueeze(0).expand(32, -1)
    _assert_matches_torch(ttnn_op, a, b, device)


@pytest.mark.parametrize(
    "ttnn_op",
    (ttnn.eq, ttnn.ne, ttnn.gt, ttnn.lt, ttnn.ge, ttnn.le),
)
def test_binary_comp_fp32_adjacent_ulps(ttnn_op, device):
    """Nearest float32 neighbors (1 ULP apart), not subnormal — border of the mantissa."""
    f32 = torch.float32
    t2 = torch.tensor(2.0, dtype=f32)
    z0 = torch.tensor(0.0, dtype=f32)
    n2 = torch.tensor(-2.0, dtype=f32)

    one = torch.tensor(1.0, dtype=f32)
    one_up = torch.nextafter(one, t2)
    one_dn = torch.nextafter(one, z0)
    m_one = torch.tensor(-1.0, dtype=f32)
    m_one_up = torch.nextafter(m_one, z0)
    m_one_dn = torch.nextafter(m_one, n2)

    vals_a = torch.stack([one, one_up, one_dn, m_one, m_one_dn, m_one])
    vals_b = torch.stack([one_up, one, one, m_one_dn, m_one, m_one_up])
    a = vals_a.unsqueeze(0).expand(32, -1)
    b = vals_b.unsqueeze(0).expand(32, -1)
    _assert_matches_torch(ttnn_op, a, b, device)


@pytest.mark.parametrize(
    "ttnn_op",
    (ttnn.eq, ttnn.ne, ttnn.gt, ttnn.lt, ttnn.ge, ttnn.le),
)
def test_binary_comp_fp32_exact_equal_tile(ttnn_op, device):
    """Two identical tiles (all elements pairwise equal)."""
    torch.manual_seed(2)
    a = torch.empty((128, 128), dtype=torch.float32).uniform_(-10.0, 10.0)
    same = torch.cat([a, a], dim=0)
    _assert_matches_torch(ttnn_op, same[:128], same[128:], device)
