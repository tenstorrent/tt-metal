# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC: the fused post-mix kernel (ttnn.experimental.deepseek_prefill.mhc_post) against the
composite ttnn form it replaces, TtMHCWrap.hc_post.

Both arms consume the same device tensors, so any difference is the kernel's own arithmetic:
the coefficient expansion through E_k and the FPU multiply / SFPU accumulate chain. The
composite is itself checked against torch in test_mhc.py::test_hc_post, so agreement here
carries the reference across.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.reference.mhc.mhc_reference import MHCConfig, sinkhorn_knopp
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import TtMHCWrap

PCC = 0.9999


def _up(device, t):
    return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)


@pytest.mark.parametrize("T", [1, 32, 96, 640], ids=["T1", "T32", "T96", "T640"])
@pytest.mark.parametrize("C", [256, 7168], ids=["C256", "C7168"])
def test_mhc_post_vs_composite(device, T, C):
    torch.manual_seed(0)
    cfg = MHCConfig(dim=C, n=4)
    n = cfg.n
    g = torch.Generator().manual_seed(4)
    fn = torch.randn(cfg.mix_hc, n * C, generator=g) * 0.02
    wrap = TtMHCWrap(device, cfg, fn, torch.randn(cfg.mix_hc, generator=g), torch.full((3,), 1.0))

    residual = _up(device, torch.randn(1, 1, T, n * C, generator=g))
    f_out = _up(device, torch.randn(1, 1, T, C, generator=g))
    post = _up(device, 2 * torch.sigmoid(torch.randn(1, 1, T, n, generator=g)))
    comb = _up(
        device,
        sinkhorn_knopp(torch.randn(1, T, n, n, generator=g), cfg.sinkhorn_iters, cfg.eps).reshape(1, 1, T, n * n),
    )

    ref = ttnn.to_torch(wrap.hc_post(f_out, residual, post, comb)).float().flatten()
    got = ttnn.to_torch(wrap.hc_post_fused(f_out, residual, post, comb)).float().flatten()

    md = (ref - got).abs().max().item()
    passed, val = comp_pcc(ref, got, PCC)
    logger.info(f"mhc_post T={T} C={C}: pcc={val} | max|Δ|={md:.2e}")
    assert passed, f"pcc={val} | max|Δ|={md:.2e}"
