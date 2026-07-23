# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Correctness gate for the fused mHC op (and the wrapped block) vs the fp32 torch reference.

Single-device so it runs on every DeepSeek_PREFILL_OP_TESTS sku (p150 / p300 / loudbox).
Exhaustive coverage — multi-chip, sharded, 524k-token prefill scale — lives in the nightly
tests/ttnn/nightly/.../deepseek_prefill/test_mhc_split_sinkhorn.py.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.reference.mhc.mhc_reference import MHCConfig, MHCWrap, parametrize
from models.demos.deepseek_v3_d_p.tt.mhc.mhc_block import TtMHCBlock
from models.demos.deepseek_v3_d_p.tt.mhc.mhc_kernel import mhc_split_sinkhorn

PCC = 0.999


def _check(name, ref, dev, pcc=PCC):
    ref, dev = ref.float().flatten(), dev.float().flatten()
    passed, val = comp_pcc(ref, dev, pcc)
    logger.info(f"{name}: pcc={val} | max|Δ|={(ref - dev).abs().max().item():.2e}")
    assert passed, f"{name}: pcc={val} (threshold {pcc})"


@pytest.mark.parametrize("T", [1, 32], ids=["T1", "T32"])
@pytest.mark.parametrize("scale_val", [0.01, 1.0], ids=["s0.01", "s1.0"])
def test_mhc_split_sinkhorn(device, T, scale_val):
    torch.manual_seed(0)
    cfg = MHCConfig(dim=64, n=4)  # parametrization is hidden-dim-independent
    g = torch.Generator().manual_seed(1)
    mixes = torch.randn(T, cfg.mix_hc, generator=g)
    scale = torch.full((3,), float(scale_val))
    base = torch.randn(cfg.mix_hc, generator=g)

    r_pre, r_post, r_comb = parametrize(mixes.reshape(1, T, cfg.mix_hc), scale, base, cfg, constraint="sinkhorn")
    d_pre, d_post, d_comb = mhc_split_sinkhorn(device, mixes, scale, base, cfg)

    _check("pre", r_pre.reshape(T, cfg.n), d_pre)
    _check("post", r_post.reshape(T, cfg.n), d_post)
    _check("comb", r_comb.reshape(T, cfg.n, cfg.n), d_comb)


def test_mhc_block(device):
    """Full X' = H_res·X + H_postᵀ·F(H_pre·X) around an identity sublayer."""
    torch.manual_seed(0)
    T, cfg = 32, MHCConfig(dim=256, n=4)
    g = torch.Generator().manual_seed(7)
    fn = torch.randn(cfg.mix_hc, cfg.n * cfg.dim, generator=g) * 0.02
    base = torch.randn(cfg.mix_hc, generator=g)
    scale = torch.full((3,), 1.0)
    x = torch.randn(1, T, cfg.n, cfg.dim)

    wrap = MHCWrap(cfg, constraint="sinkhorn")
    wrap.fn.data, wrap.base.data, wrap.scale.data = fn, base, scale
    ref = wrap(x, lambda z: z)

    block = TtMHCBlock(device, cfg, fn, base, scale)
    x_tt = ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)
    dev = ttnn.to_torch(block(x_tt, lambda z: z))
    _check("block", ref, dev)
