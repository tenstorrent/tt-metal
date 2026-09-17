# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC: fused mHC kernel (ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn) vs the
pure-torch ground truth (models/demos/deepseek_v3_d_p/reference/mhc/mhc_reference.py::parametrize).

The op is per-token: it reads mixes[T, (2+n)*n] and writes that token's H matrices, with no
dependency between tokens and none on the hidden dim. Coverage is therefore fixed at the
per-device token count the model runs, in both the interleaved and the sharded layout.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, is_blackhole
from models.demos.deepseek_v3_d_p.reference.mhc.mhc_reference import MHCConfig, parametrize
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import build_consts

# The op ships for Blackhole prefill. The suite that folder-collects this directory also runs on
# Wormhole under -x, where a failure off the target arch would abort other teams' tests.
pytestmark = pytest.mark.skipif(not is_blackhole(), reason="mhc_split_sinkhorn targets Blackhole")

PCC = 0.999

# A 5120-token prefill split over SP=8 lands 640 tokens (20 tiles) on each device, and the op is
# token-flattened, so that count is the only shape the model ever asks of it.
TOKENS = 640


def _check(name, ref, dev, pcc=PCC):
    ref = ref.float().flatten()
    dev = dev.float().flatten()
    md = (ref - dev).abs().max().item()
    passed, val = comp_pcc(ref, dev, pcc)
    logger.info(f"{name}: pcc={val} | max|Δ|={md:.2e}")
    assert passed, f"{name}: pcc={val} | max|Δ|={md:.2e} (threshold {pcc})"


def _params(cfg, scale_val, T):
    g = torch.Generator().manual_seed(1)
    mixes = torch.randn(T, cfg.mix_hc, generator=g)
    scale = torch.full((3,), float(scale_val))
    base = torch.randn(cfg.mix_hc, generator=g)
    return mixes, scale, base


def _run(device, mixes, scale, base, cfg):
    """mixes: [T, (2+n)*n] -> (pre [T,n], post [T,n], comb [T,n,n]) as torch tensors."""
    n, T = cfg.n, mixes.shape[0]
    mixes_tt = ttnn.from_torch(mixes.float(), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)
    consts_tt = ttnn.from_torch(
        build_consts(cfg, scale, base), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32
    )
    pre, post, comb = ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn(
        mixes_tt, consts_tt, n, int(cfg.sinkhorn_iters), float(cfg.eps)
    )
    return ttnn.to_torch(pre), ttnn.to_torch(post), ttnn.to_torch(comb).reshape(T, n, n)


@pytest.mark.parametrize("scale_val", [0.01, 1.0], ids=["s0.01", "s1.0"])
def test_mhc_split_sinkhorn(device, scale_val):
    torch.manual_seed(0)
    cfg = MHCConfig(dim=64, n=4)  # dim is irrelevant to parametrization
    T = TOKENS
    mixes, scale, base = _params(cfg, scale_val, T)

    r_pre, r_post, r_comb = parametrize(mixes.reshape(1, T, cfg.mix_hc), scale, base, cfg, constraint="sinkhorn")
    d_pre, d_post, d_comb = _run(device, mixes, scale, base, cfg)

    _check("pre", r_pre.reshape(T, cfg.n), d_pre)
    _check("post", r_post.reshape(T, cfg.n), d_post)
    _check("comb", r_comb.reshape(T, cfg.n, cfg.n), d_comb)


# Sharded input (#40720): mixes L1 height-sharded across cores; the op aliases input/output
# CBs to the shards (zero-copy, no DRAM round-trip). Outputs come back sharded on the same grid.
# TOKENS is 20 tiles, so the grid has to divide 20 to keep every shard tile-aligned: 20 cores
# hold a tile each (the per-core minimum, and 20 only fits as 5x4 on an 8-wide Wormhole grid),
# 4 cores hold five. The op walks the shard grid as a CoreRangeSet, so its shape is free.
@pytest.mark.parametrize("cores_x, cores_y", [(5, 4), (4, 1)], ids=["c20_tpc1", "c4_tpc5"])
def test_mhc_split_sinkhorn_sharded(device, cores_x, cores_y):
    torch.manual_seed(0)
    cfg = MHCConfig(dim=64, n=4)
    T = TOKENS
    mixes, scale, base = _params(cfg, 1.0, T)

    r_pre, r_post, r_comb = parametrize(mixes.reshape(1, T, cfg.mix_hc), scale, base, cfg, constraint="sinkhorn")

    # Sharded TILE tensors need tile-aligned shard width -> pad mixes 24->32; outputs come
    # back 32-wide too (kernel emits 32-wide tiles), sliced below.
    mixes32 = torch.zeros(T, 32)
    mixes32[:, : cfg.mix_hc] = mixes
    mem = ttnn.create_sharded_memory_config(
        [T, 32], ttnn.CoreGrid(y=cores_y, x=cores_x), ttnn.ShardStrategy.HEIGHT, ttnn.ShardOrientation.ROW_MAJOR
    )
    mt = ttnn.from_torch(mixes32, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32, memory_config=mem)
    ct = ttnn.from_torch(build_consts(cfg, scale, base), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)

    pre, post, comb = ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn(
        mt, ct, cfg.n, int(cfg.sinkhorn_iters), float(cfg.eps)
    )
    _check("pre", r_pre.reshape(T, cfg.n), ttnn.to_torch(pre)[:, : cfg.n])
    _check("post", r_post.reshape(T, cfg.n), ttnn.to_torch(post)[:, : cfg.n])
    _check("comb", r_comb.reshape(T, cfg.n, cfg.n), ttnn.to_torch(comb)[:, : cfg.n * cfg.n].reshape(T, cfg.n, cfg.n))
