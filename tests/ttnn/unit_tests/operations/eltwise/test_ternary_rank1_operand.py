# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# A rank-1 operand ([W]) broadcast against a taller output must be replicated to every row, as torch does.

import pytest
import torch
import ttnn

DTYPES = [(torch.bfloat16, ttnn.bfloat16), (torch.float32, ttnn.float32)]


def _tt(x, ndt, device):
    return ttnn.from_torch(x, dtype=ndt, layout=ttnn.TILE_LAYOUT, device=device)


@pytest.mark.parametrize("tdt, ndt", DTYPES)
@pytest.mark.parametrize(
    "c_shape, t_shape, f_shape",
    [
        ((4, 32), (4, 32), (32,)),  # false operand rank 1
        ((4, 32), (32,), (4, 32)),  # true operand rank 1
        ((32,), (4, 32), (4, 32)),  # predicate rank 1
        ((2, 4, 32), (2, 4, 32), (32,)),  # rank 1 against rank 3
        ((64, 64), (64, 64), (64,)),  # output taller than one tile: reads past the operand's pages
    ],
)
def test_where_rank1_operand(device, tdt, ndt, c_shape, t_shape, f_shape):
    torch.manual_seed(0)
    c = (torch.rand(c_shape) > 0.5).to(tdt)
    t = torch.randn(t_shape).to(tdt)
    f = torch.randn(f_shape).to(tdt)
    golden = torch.where(c.bool(), t, f)
    out = ttnn.to_torch(ttnn.where(_tt(c, ndt, device), _tt(t, ndt, device), _tt(f, ndt, device)))
    assert torch.equal(out, golden)


@pytest.mark.parametrize("tdt, ndt", DTYPES)
def test_where_rank1_and_row_operand_share_a_cache_entry(device, tdt, ndt):
    # compute_program_hash keys a rank-1 operand as H = 1, the same as [1, W]: both orders must stay exact.
    c = torch.zeros(4, 32).to(tdt)
    a = torch.full((4, 32), -99.0).to(tdt)
    row = (torch.arange(32, dtype=torch.float32) + 1).to(tdt)
    golden = row.expand(4, 32)
    for first, second in ((row, row.reshape(1, 32)), (row.reshape(1, 32), row)):
        device.clear_program_cache()
        for b in (first, second):
            out = ttnn.to_torch(ttnn.where(_tt(c, ndt, device), _tt(a, ndt, device), _tt(b, ndt, device)))
            assert torch.equal(out.reshape(4, 32), golden)


@pytest.mark.parametrize("tdt, ndt", DTYPES)
def test_ternary_family_rank1_operand(device, tdt, ndt):
    a = torch.randn(4, 32).to(tdt)
    v = torch.rand(32).to(tdt) + 1.0  # rank 1, kept away from 0 for addcdiv
    m = torch.randn(4, 32).to(tdt)
    cases = [
        (
            lambda: ttnn.addcmul(_tt(a, ndt, device), _tt(v, ndt, device), _tt(m, ndt, device), value=1.0),
            torch.addcmul(a.float(), v.float(), m.float()),
        ),
        (
            lambda: ttnn.addcdiv(_tt(a, ndt, device), _tt(m, ndt, device), _tt(v, ndt, device), value=1.0),
            torch.addcdiv(a.float(), m.float(), v.float()),
        ),
        (lambda: ttnn.lerp(_tt(a, ndt, device), _tt(v, ndt, device), 0.5), torch.lerp(a.float(), v.float(), 0.5)),
        (
            lambda: ttnn.mac(_tt(a, ndt, device), _tt(v, ndt, device), _tt(m, ndt, device)),
            a.float() * v.float() + m.float(),
        ),
    ]
    for run, golden in cases:
        out = ttnn.to_torch(run()).float()
        assert torch.isfinite(out).all()
        torch.testing.assert_close(out, golden, rtol=2e-2, atol=5e-2)
