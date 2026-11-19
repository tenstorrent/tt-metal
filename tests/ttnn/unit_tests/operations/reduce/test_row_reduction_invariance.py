# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Row placement must not choose a different FP32 reduction association."""

import pytest
import torch
import ttnn
from tt_py_test_utils_common.utility_functions import is_blackhole

pytestmark = pytest.mark.use_module_device


def _reduce(device, values, dtype, operation):
    tensor = ttnn.from_torch(values, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    return ttnn.to_torch(getattr(ttnn, operation)(tensor, dim=-1, keepdim=True))


def _inputs(width, repeated):
    rng = torch.Generator().manual_seed(8301 + width)
    shape = (1, 1, 1 if repeated else 32, width)
    # Nonuniform magnitudes make floating-point association observable even
    # at width32. All-positive inputs also give a meaningful relative error
    # bound, avoiding a permissive tolerance around cancellation to zero.
    mantissa = torch.rand(shape, generator=rng) + 0.5
    exponent = torch.randint(-12, 12, shape, generator=rng)
    values = torch.ldexp(mantissa, exponent)
    return values.expand(1, 1, 32, width).contiguous()


@pytest.mark.merge_gate
@pytest.mark.skipif(not is_blackhole(), reason="Row-association repair is Blackhole-specific")
@pytest.mark.parametrize("width", [32, 2560])
@pytest.mark.parametrize("operation", ["sum", "mean"])
def test_fp32_row_reduction_placement_invariance(device, width, operation):
    for repeated in (True, False):
        values = _inputs(width, repeated)
        baseline = _reduce(device, values, ttnn.float32, operation)
        golden = getattr(torch, operation)(values.double(), dim=-1, keepdim=True).float()
        torch.testing.assert_close(baseline, golden, rtol=2e-5, atol=1e-6)
        if repeated:
            # One identical operand row occupies every SIMD row group and both
            # vertical face pairs, so no row-position case is sampled away.
            assert torch.equal(baseline, baseline[:, :, :1, :].expand_as(baseline))
        else:
            # Distinct rows catch accidental row mixing while crossing both
            # four-row groups and the16-row face boundary.
            for shift in (1, 4, 16):
                order = torch.arange(32).roll(shift)
                relocated = _reduce(device, values[:, :, order, :], ttnn.float32, operation)
                assert torch.equal(relocated, baseline[:, :, order, :]), f"row shift {shift}"
        split = torch.cat(
            [_reduce(device, values[:, :, a:b, :], ttnn.float32, operation) for a, b in ((0, 4), (4, 32))],
            dim=2,
        )
        assert torch.equal(split, baseline), "Nonaligned continuation must preserve the row reduction"


@pytest.mark.merge_gate
@pytest.mark.parametrize("operation", ["sum", "mean"])
def test_bf16_row_reduction_sanity(device, operation):
    rng = torch.Generator().manual_seed(172)
    values = torch.randint(-4, 5, (1, 1, 32, 32), generator=rng).to(torch.bfloat16)
    result = _reduce(device, values, ttnn.bfloat16, operation)
    golden = getattr(torch, operation)(values.float(), dim=-1, keepdim=True).to(torch.bfloat16)
    # Small integer sums and division by32 are exactly representable in BF16.
    assert torch.equal(result, golden)


@pytest.mark.merge_gate
def test_int32_row_reduction_keeps_low_bits(device):
    values = torch.full((1, 1, 32, 32), 2**20, dtype=torch.int32)
    values[0, 0, :, 0] += torch.arange(1, 33, dtype=torch.int32)
    values[:, :, 1::2, :] *= -1
    result = _reduce(device, values, ttnn.int32, "sum")
    golden = values.sum(-1, keepdim=True, dtype=torch.int32)
    assert torch.equal(result, golden), "Integer sum must not pass through lossy FP32 rounding"


@pytest.mark.merge_gate
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16, ttnn.int32])
def test_row_max_sanity(device, dtype):
    rng = torch.Generator().manual_seed(95)
    values = torch.randint(-120, 121, (1, 1, 32, 32), generator=rng, dtype=torch.int32)
    if dtype != ttnn.int32:
        values = values.to(torch.float32 if dtype == ttnn.float32 else torch.bfloat16)
    result = _reduce(device, values, dtype, "max")
    assert torch.equal(result, values.max(-1, keepdim=True).values)
