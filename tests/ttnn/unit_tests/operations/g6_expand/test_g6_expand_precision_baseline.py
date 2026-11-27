# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for g6_expand (pure byte movement: the expectation is bit-exact).

Measures PCC, max / mean abs error, relative RMS (RMS error / std of the reference) and the
got/true ratio spread over the production shapes, and gates on bit identity.
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.g6_expand import g6_expand

# small (36 tiles, non-aligned), medium (64 tiles), non-aligned 49 tiles, large (256 tiles, perf target)
WIDTHS = [185, 196, 256, 512]


@pytest.mark.parametrize("width", WIDTHS)
def test_g6_expand_precision_baseline(device, width):
    torch.manual_seed(0)
    x = torch.randn((1, 1, 1, width), dtype=torch.float32).to(torch.bfloat16)
    sizes = [1, 1, width, width]
    expected = x.expand(*sizes).contiguous()
    tt_x = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    got = ttnn.to_torch(g6_expand(tt_x, sizes))

    pcc_ok, pcc = comp_pcc(expected, got, 0.9999)
    _, allclose_msg = comp_allclose(expected, got)
    e, g = expected.float(), got.float()
    err = g - e
    max_abs = err.abs().max().item()
    mean_abs = err.abs().mean().item()
    rel_rms = (err.pow(2).mean().sqrt() / e.std()).item()
    mask = e != 0
    r = g[mask] / e[mask]
    r_med = r.median().item()
    r_p5, r_p95 = torch.quantile(r, torch.tensor([0.05, 0.95])).tolist()
    bit_diff = int((got.view(torch.int16) != expected.view(torch.int16)).sum())
    print(
        f"\nBASELINE W={width}: pcc={pcc} max_abs={max_abs:.3g} mean_abs={mean_abs:.3g} "
        f"rel_rms={rel_rms:.3g} ratio median={r_med:.6f} p5={r_p5:.6f} p95={r_p95:.6f} "
        f"bit_diff={bit_diff} ({allclose_msg})"
    )
    assert pcc_ok
    assert rel_rms == 0.0
    assert bit_diff == 0, f"{bit_diff} elements differ in bits"
