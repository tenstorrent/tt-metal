# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC: fused mHC kernel (ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn) vs the
pure-torch ground truth (models/demos/deepseek_v3_d_p/reference/mhc/mhc_reference.py::parametrize).

Unit (B=1,S=1 -> T=1) is issue #40707; T=32 exercises a full tile of tokens.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.reference.mhc.mhc_reference import MHCConfig, parametrize
from models.demos.deepseek_v3_d_p.tt.mhc.mhc_kernel import mhc_split_sinkhorn

PCC = 0.999


def _check(name, ref, dev, pcc=PCC):
    ref = ref.float().flatten()
    dev = dev.float().flatten()
    md = (ref - dev).abs().max().item()
    passed, val = comp_pcc(ref, dev, pcc)
    logger.info(f"{name}: pcc={val} | max|Δ|={md:.2e}")
    assert passed, f"{name}: pcc={val} | max|Δ|={md:.2e} (threshold {pcc})"


# T=1 unit (#40707), 32 one full tile, 100 multi-tile+partial, 2048 spans ~64 cores (multi-core #40719/#40722)
@pytest.mark.parametrize("T", [1, 32, 100, 2048], ids=["T1", "T32", "T100", "T2048"])
@pytest.mark.parametrize("scale_val", [0.01, 1.0], ids=["s0.01", "s1.0"])
def test_mhc_split_sinkhorn(device, T, scale_val):
    torch.manual_seed(0)
    cfg = MHCConfig(dim=64, n=4)  # dim is irrelevant to parametrization
    g = torch.Generator().manual_seed(1)
    mixes = torch.randn(T, cfg.mix_hc, generator=g)
    scale = torch.full((3,), float(scale_val))
    base = torch.randn(cfg.mix_hc, generator=g)

    r_pre, r_post, r_comb = parametrize(mixes.reshape(1, T, cfg.mix_hc), scale, base, cfg, constraint="sinkhorn")

    d_pre, d_post, d_comb = mhc_split_sinkhorn(device, mixes, scale, base, cfg)

    _check("pre", r_pre.reshape(T, cfg.n), d_pre)
    _check("post", r_post.reshape(T, cfg.n), d_post)
    _check("comb", r_comb.reshape(T, cfg.n, cfg.n), d_comb)


# General multi-core at prefill scale (#40722). The op is token-flattened (T = B*S), so B vs S
# is irrelevant here -- only the token count matters. 524288 tokens (16384 tiles) is the design
# doc's B=32/S=16k target and dwarfs any realistic B=1 prefill (S=16k -> T=16384). Guards
# correctness + no-OOM across the full grid, DRAM-interleaved.
def test_mhc_split_sinkhorn_prefill_scale(device):
    torch.manual_seed(0)
    cfg = MHCConfig(dim=64, n=4)
    T = 32 * 16384  # = B*S tokens; the kernel never distinguishes B from S
    g = torch.Generator().manual_seed(1)
    mixes = torch.randn(T, cfg.mix_hc, generator=g)
    scale = torch.full((3,), 1.0)
    base = torch.randn(cfg.mix_hc, generator=g)

    r_pre, r_post, r_comb = parametrize(mixes.reshape(1, T, cfg.mix_hc), scale, base, cfg, constraint="sinkhorn")
    d_pre, d_post, d_comb = mhc_split_sinkhorn(device, mixes, scale, base, cfg)

    _check("pre", r_pre.reshape(T, cfg.n), d_pre)
    _check("post", r_post.reshape(T, cfg.n), d_post)
    _check("comb", r_comb.reshape(T, cfg.n, cfg.n), d_comb)


# Sharded input (#40720): mixes L1 height-sharded across cores; the op aliases input/output
# CBs to the shards (zero-copy, no DRAM round-trip). Outputs come back sharded on the same grid.
@pytest.mark.parametrize("cores_x", [8], ids=["x8"])
@pytest.mark.parametrize("tiles_per_core", [1, 2], ids=["tpc1", "tpc2"])
def test_mhc_split_sinkhorn_sharded(device, cores_x, tiles_per_core):
    from models.demos.deepseek_v3_d_p.tt.mhc.mhc_kernel import build_consts

    torch.manual_seed(0)
    cfg = MHCConfig(dim=64, n=4)
    T = cores_x * tiles_per_core * 32
    g = torch.Generator().manual_seed(1)
    mixes = torch.randn(T, cfg.mix_hc, generator=g)
    scale = torch.full((3,), 1.0)
    base = torch.randn(cfg.mix_hc, generator=g)

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


# Multi-chip (#40723): parametrization is per-token-independent -> shard tokens across the
# mesh, replicate the constant tiles, run the multi-core kernel per device, no CCL.
@pytest.mark.parametrize("device_params", [{}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4), (2, 4)], indirect=True, ids=["mesh1x4", "mesh2x4"])
@pytest.mark.parametrize("T_per_dev", [32, 256], ids=["Td32", "Td256"])
def test_mhc_split_sinkhorn_multichip(mesh_device, T_per_dev, device_params):
    from models.demos.deepseek_v3_d_p.tt.mhc.mhc_kernel import build_consts

    torch.manual_seed(0)
    D = mesh_device.get_num_devices()
    T = T_per_dev * D
    cfg = MHCConfig(dim=64, n=4)
    g = torch.Generator().manual_seed(1)
    mixes = torch.randn(T, cfg.mix_hc, generator=g)
    scale = torch.full((3,), 1.0)
    base = torch.randn(cfg.mix_hc, generator=g)

    r_pre, r_post, r_comb = parametrize(mixes.reshape(1, T, cfg.mix_hc), scale, base, cfg, constraint="sinkhorn")

    mt = ttnn.from_torch(
        mixes,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.float32,
    )
    ct = ttnn.from_torch(
        build_consts(cfg, scale, base),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.float32,
    )
    pre, post, comb = ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn(
        mt, ct, cfg.n, int(cfg.sinkhorn_iters), float(cfg.eps)
    )
    cat0 = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    d_pre = ttnn.to_torch(pre, mesh_composer=cat0)
    d_post = ttnn.to_torch(post, mesh_composer=cat0)
    d_comb = ttnn.to_torch(comb, mesh_composer=cat0).reshape(T, cfg.n, cfg.n)

    _check("pre", r_pre.reshape(T, cfg.n), d_pre)
    _check("post", r_post.reshape(T, cfg.n), d_post)
    _check("comb", r_comb.reshape(T, cfg.n, cfg.n), d_comb)
