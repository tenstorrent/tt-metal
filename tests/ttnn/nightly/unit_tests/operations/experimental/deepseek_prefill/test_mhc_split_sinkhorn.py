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

# The kernel has only ever been exercised on Blackhole. The group suite that folder-collects
# this directory also runs on Wormhole under -x, where an unproven failure would abort tests
# belonging to other teams.
pytestmark = pytest.mark.skipif(not is_blackhole(), reason="mhc_split_sinkhorn is validated on Blackhole only")

PCC = 0.999

# A 5120-token prefill split over SP=8 lands 640 tokens (20 tiles) on each device, and the op is
# token-flattened, so that count is the only shape the model ever asks of it.
TOKENS = 640


@pytest.fixture(autouse=True)
def _p150_only(device):
    """The kernel is brought up on P150 only.

    The cluster query hangs off the open device rather than a collection-time skipif because
    get_cluster_type() takes the chip lock, which would strand anything that later forks.
    """
    if ttnn.cluster.get_cluster_type() == ttnn.cluster.ClusterType.P100:
        pytest.skip("mhc_split_sinkhorn is validated on P150 only")


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
# TOKENS is 20 tiles, so only a grid that divides 20 keeps every shard tile-aligned.
@pytest.mark.parametrize("cores_x", [4, 5], ids=["x4", "x5"])
def test_mhc_split_sinkhorn_sharded(device, cores_x):
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
        [T, 32], ttnn.CoreGrid(y=1, x=cores_x), ttnn.ShardStrategy.HEIGHT, ttnn.ShardOrientation.ROW_MAJOR
    )
    mt = ttnn.from_torch(mixes32, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32, memory_config=mem)
    ct = ttnn.from_torch(build_consts(cfg, scale, base), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)

    pre, post, comb = ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn(
        mt, ct, cfg.n, int(cfg.sinkhorn_iters), float(cfg.eps)
    )
    _check("pre", r_pre.reshape(T, cfg.n), ttnn.to_torch(pre)[:, : cfg.n])
    _check("post", r_post.reshape(T, cfg.n), ttnn.to_torch(post)[:, : cfg.n])
    _check("comb", r_comb.reshape(T, cfg.n, cfg.n), ttnn.to_torch(comb)[:, : cfg.n * cfg.n].reshape(T, cfg.n, cfg.n))
