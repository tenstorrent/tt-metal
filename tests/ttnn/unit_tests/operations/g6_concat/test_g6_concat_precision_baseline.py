# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline of g6_concat on Phase 0 production shapes (bfloat16, TILE, DRAM).

The op moves values and computes nothing, so the contract is bit-exactness: PCC 1, zero abs
error, zero relative RMS, zero ULP and a got/true ratio of exactly 1. The metrics are printed
(`-s`) so the verification report can record them; the gate is the exact contract.
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.g6_concat import g6_concat

BASELINE_CASES = [
    # small: one tile row, trimmed to v = 1 on read and write
    pytest.param([(1, 1, 1, 128)] * 2, -1, id="flux23_small_w"),
    # medium: 31-row images (padding row never read), two inputs of different width
    pytest.param([(1, 6, 31, 64), (1, 6, 31, 192)], -1, id="qwen109_medium_w"),
    # row_gather: 16 height-1 inputs share every output tile
    pytest.param([(1, 1, 1, 62080)] * 16, 2, id="qwen79_row_gather_h"),
    # larger: 8 x 6144 tiles, many blocks per core
    pytest.param([(1, 4096, 1536)] * 8, 0, id="qwen23_large_outer"),
]


def _ordered(bits16):
    """bfloat16 bit patterns -> integers ordered like the values (ULP distance = difference)."""
    b = bits16.to(torch.int32) & 0xFFFF
    return torch.where(b >= 0x8000, 0x8000 - b, b)


@pytest.mark.parametrize("shapes, dim", BASELINE_CASES)
def test_g6_concat_precision_baseline(shapes, dim, device):
    torch.manual_seed(1234)
    torch_inputs = [torch.randn(s, dtype=torch.float32).to(torch.bfloat16) for s in shapes]
    tt_inputs = [ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device) for t in torch_inputs]
    expected = torch.cat(torch_inputs, dim=dim)
    actual = ttnn.to_torch(g6_concat(tt_inputs, dim)).to(torch.bfloat16)
    assert list(actual.shape) == list(expected.shape)

    _, pcc = comp_pcc(expected, actual, pcc=0.0)
    _, allclose_msg = comp_allclose(expected, actual)
    exp32, act32 = expected.float(), actual.float()
    err = act32 - exp32
    max_abs = err.abs().max().item()
    mean_abs = err.abs().mean().item()
    rel_rms = (err.pow(2).mean().sqrt() / exp32.std()).item()
    ulp = (_ordered(actual.view(torch.int16)) - _ordered(expected.view(torch.int16))).abs()
    nonzero = exp32 != 0
    ratio = act32[nonzero] / exp32[nonzero]
    q = torch.quantile(ratio[: 1 << 20].double(), torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64))
    print(
        f"\nBASELINE {shapes[0]}x{len(shapes)} dim={dim}: pcc={pcc} max_abs={max_abs} mean_abs={mean_abs} "
        f"rel_rms={rel_rms} max_ulp={int(ulp.max())} ulp_mismatches={int((ulp != 0).sum())} "
        f"ratio_p5/p50/p95={q.tolist()} [{allclose_msg}]"
    )

    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16)), "logical values not bit-identical"
    assert max_abs == 0.0 and rel_rms == 0.0 and int(ulp.max()) == 0
