# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for g6_unsqueeze (verifier).

The op is a pure bit move, so the baseline is bit-exactness: every logical output word equals its
input word (NaN payloads, -0.0, inf, subnormals included). PCC / abs errors / relative RMS and the
got/true ratio spread are recorded for the report; the gate is bit equality.
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.g6_unsqueeze import g6_unsqueeze

_SPECIALS = [float("nan"), float("inf"), float("-inf"), -0.0, 1e-45, -1e-45, 1.17e-38, 3.4e38, -3.4e38]


@pytest.mark.parametrize("shape", [(1, 32), (1, 256), (1, 512), (1, 1, 4100)])
def test_precision_baseline(device, shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32)
    flat = x.view(-1)
    for i, v in enumerate(_SPECIALS):
        flat[(i * 37) % flat.numel()] = v
    nan_payload = torch.tensor([0x7FC0_1234], dtype=torch.int32).view(torch.float32)
    flat[flat.numel() - 1] = nan_payload[0]

    tt_x = ttnn.from_torch(
        x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    got = ttnn.to_torch(g6_unsqueeze(tt_x, -1))
    want = torch.unsqueeze(x, -1)
    assert list(got.shape) == list(want.shape)

    bits_equal = torch.equal(got.contiguous().view(torch.int32), want.contiguous().view(torch.int32))

    finite = torch.isfinite(want) & (want != 0)
    g, w = got[finite], want[finite]
    _, pcc = comp_pcc(w, g)
    _, allclose_msg = comp_allclose(w, g)
    err = (g - w).double()
    max_abs = err.abs().max().item()
    mean_abs = err.abs().mean().item()
    rel_rms = (err.pow(2).mean().sqrt() / w.double().std()).item()
    r = (g / w).double()
    q = torch.quantile(r, torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64))
    print(
        f"\nBASELINE shape={shape} bit_exact={bits_equal} pcc={pcc} max_abs={max_abs:.3e} "
        f"mean_abs={mean_abs:.3e} rel_rms={rel_rms:.3e} ratio p5/p50/p95={q[0]:.6f}/{q[1]:.6f}/{q[2]:.6f} "
        f"[{allclose_msg}]"
    )
    assert bits_equal, "g6_unsqueeze must be bit-exact"
