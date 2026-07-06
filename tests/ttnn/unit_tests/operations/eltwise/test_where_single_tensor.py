# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


SHAPES = [
    (8,),
    (32,),
    (100,),
    (4, 8),
    (8, 32),
    (2, 3, 4),
    (1, 1, 16),
    (1, 1, 1, 8),
    (1, 1, 1, 32),
    (1, 1, 4, 8),
    (1, 2, 4, 8),
    (2, 1, 4, 8),
    (2, 3, 4, 5),
]


def _run_case(x_torch, device):
    tt_in = ttnn.from_torch(
        x_torch.to(torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    got = ttnn.where(tt_in)
    exp = torch.where(x_torch)
    assert len(got) == len(exp)
    for g, e in zip(got, exp):
        # Same nonzero positions in the same order.
        assert torch.equal(g.to(torch.int64), e.to(torch.int64)), f"mismatch\n got={g}\n exp={e}"


@pytest.mark.parametrize("shape", SHAPES)
def test_where_single_tensor_sparse(device, shape):
    torch.manual_seed(0)
    x = torch.zeros(shape, dtype=torch.int32)
    flat = x.flatten()
    if flat.numel() > 0:
        # 1 in every 8 non-zero
        flat[::8] = 1
    _run_case(flat.reshape(shape), device)


@pytest.mark.parametrize("shape", SHAPES)
def test_where_single_tensor_dense(device, shape):
    x = torch.ones(shape, dtype=torch.int32)
    _run_case(x, device)


@pytest.mark.parametrize("shape", SHAPES)
def test_where_single_tensor_all_zero(device, shape):
    x = torch.zeros(shape, dtype=torch.int32)
    _run_case(x, device)


def test_where_single_tensor_bool(device):
    torch.manual_seed(0)
    x = torch.randint(0, 2, (2, 3, 4, 5), dtype=torch.int32)
    _run_case(x, device)


def test_where_ternary_still_works(device):
    # Regression: the 3-arg overload must still dispatch to the C++ where.
    torch.manual_seed(0)
    cond = torch.randint(0, 2, (1, 1, 32, 32), dtype=torch.uint8).to(torch.bool)
    a = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
    b = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
    tt_cond = ttnn.from_torch(cond, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_a = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_b = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out = ttnn.where(tt_cond, tt_a, tt_b)
    out = ttnn.to_torch(tt_out)
    exp = torch.where(cond, a, b)
    torch.testing.assert_close(out, exp, atol=1e-2, rtol=1e-2)
