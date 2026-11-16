# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""DSpark parallel verify: ``ttnn.experimental.matmul_decode`` on a block of ``S`` rows.

A block of ``S`` rows is a TILE tensor with an ``[h, 32]`` tile, ``h`` in {2, 4, 8} (``S`` padded up to ``h``),
so the ``S`` rows are one custom_mm in0 tile per K step; the output is the same ``[h, 32]``-tiled block on the
same placement. ``S = 1`` stays ROW_MAJOR. The tests cover the V4.1 decode call sites at their TP4 per-chip
shapes and grids:

- replicated A (the HEIGHT_SHARDED replica ``all_gather_for_matmul`` / an output mcast leaves): q_b, o_b,
  Engram's wkv, the compressor's wkv|wgate, the indexer's wq_b / weights_proj;
- q_a and kv with the fused RMSNorm epilogue (vector gamma) and their output mcast;
- batched o_a (groups as the batch, output mcast onto o_b's grid), with an L1 and a DRAM-read weight;
- WIDTH_SHARDED A (the two-hub gather).

Each row ``j`` of the block is checked against ``S = 1`` ROW_MAJOR calls on row ``j`` (bit-for-bit where the
arithmetic is the same) and against a torch reference.

Profile::

    python -m tracy -r -p -v --no-web-server -t 8123 -o <out> -m pytest <this file> -k profile -s
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pytest
import torch
from loguru import logger

import ttnn

pytestmark = pytest.mark.skipif(not ttnn.device.is_blackhole(), reason="custom_mm is Blackhole-only")

ITERATIONS = 10
EPS = 1e-6


def _signpost(name: str) -> None:
    try:
        from tracy import signpost

        signpost(name)
    except ImportError:
        pass


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def rect(num_cores: int, device, origin=(0, 0), preferred_width: Optional[int] = None) -> ttnn.CoreRangeSet:
    """A filled rectangle of ``num_cores`` from ``origin`` (``LinearDecode(rectangle_b_grid=True)``'s rule)."""
    grid = device.compute_with_storage_grid_size()
    x0, y0 = origin
    cols, rows = grid.x - x0, grid.y - y0

    def fits(w):
        return w is not None and 0 < w <= cols and num_cores % w == 0 and num_cores // w <= rows

    w = preferred_width if fits(preferred_width) else next(w for w in range(cols, 0, -1) if fits(w))
    return ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x0 + w - 1, y0 + num_cores // w - 1))}
    )


def row_wise(num_cores: int, device) -> ttnn.CoreRangeSet:
    return ttnn.num_cores_to_corerangeset(num_cores, device.compute_with_storage_grid_size(), row_wise=True)


def bbox(grid: ttnn.CoreRangeSet) -> ttnn.CoreRangeSet:
    return ttnn.CoreRangeSet({grid.bounding_box()})


def shard_cfg(shape, grid, strategy):
    return ttnn.create_sharded_memory_config(
        shape,
        core_grid=grid,
        strategy=strategy,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def tile_height(rows: int) -> int:
    """The block's tile height: 1 (ROW_MAJOR) for one row, else ``rows`` padded up to 2 / 4 / 8."""
    return 1 if rows == 1 else next(h for h in (2, 4, 8) if rows <= h)


def make_replica(device, rows_kn: torch.Tensor, grid: ttnn.CoreRangeSet) -> ttnn.Tensor:
    """A HEIGHT_SHARDED replica of ``rows_kn [R, K]`` (R already padded to h) on every core of ``grid``.

    One row: ROW_MAJOR. More: TILE with an ``[h, 32]`` tile.
    """
    r, k = rows_kn.shape
    h = tile_height(r)
    full = rows_kn.repeat(grid.num_cores(), 1)
    kw = dict(dtype=ttnn.bfloat16, device=device, memory_config=shard_cfg((r, k), grid, ttnn.ShardStrategy.HEIGHT))
    if h == 1:
        return ttnn.from_torch(full, layout=ttnn.ROW_MAJOR_LAYOUT, **kw)
    return ttnn.from_torch(full, layout=ttnn.TILE_LAYOUT, tile=ttnn.Tile((h, 32)), **kw)


def l1_weight(device, w_kn: torch.Tensor, grid: ttnn.CoreRangeSet, dtype) -> ttnn.Tensor:
    k, n = w_kn.shape[-2:]
    return ttnn.from_torch(
        w_kn,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=shard_cfg((k, n // grid.num_cores()), grid, ttnn.ShardStrategy.WIDTH),
    )


def dram_weight(device, w_kn: torch.Tensor, num_slabs: int, dtype) -> ttnn.Tensor:
    """DRAM ND-sharded, one ``[K, N / num_slabs]`` slab per weight core (``dram_sharded_weight``)."""
    k, n = w_kn.shape[-2:]
    banks = device.dram_grid_size().x
    cfg = ttnn.MemoryConfig(
        ttnn.BufferType.DRAM,
        ttnn.NdShardSpec(
            ttnn.Shape([k, n // num_slabs]),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))}),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
        ),
    )
    return ttnn.from_torch(w_kn, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=cfg)


def gamma_tensor(device, gamma: torch.Tensor, grid: ttnn.CoreRangeSet) -> ttnn.Tensor:
    n = gamma.numel()
    return ttnn.from_torch(
        gamma.reshape(1, n),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        tile=ttnn.Tile((1, 32)),
        device=device,
        memory_config=shard_cfg((1, n // grid.num_cores()), grid, ttnn.ShardStrategy.WIDTH),
    )


def out_rows(out: ttnn.Tensor, rows: int, n: int) -> torch.Tensor:
    """``[copies, rows, N]``: every replica (mcast) or the one width-sharded result, padding rows dropped."""
    t = ttnn.to_torch(out).float()
    t = t.reshape(-1, t.shape[-1])
    if out.memory_config().memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        shard_h = out.memory_config().shard_spec.shape[0]
        return t.reshape(-1, shard_h, t.shape[-1])[:, :rows, :n]
    return t[None, :rows, :n]


# ---------------------------------------------------------------------------------------------------------------------
# Full-width call sites (TP4 per-chip shapes). ``a_grid`` / ``out_grid`` mirror the model's placements.
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class Site:
    name: str
    K: int
    N: int
    n_blocks: int
    wdtype: object = ttnn.bfloat8_b
    rms: bool = False
    out: str = "none"  # "none" (WIDTH_SHARDED on the B grid), or a dest grid built in grids()
    dram: bool = True
    extra: dict = field(default_factory=dict)


SITES = [
    # q_a: 8 B cores, fused RMSNorm (vector gamma), its replica mcast onto q_b's 8x8 B grid.
    Site("q_a", 5120, 1280, 8, rms=True, out="q_b"),
    # kv: 16 B cores, fused RMSNorm, one replica on core (0, 0).
    Site("kv", 5120, 512, 16, rms=True, out="core00"),
    # q_b: N = 32768 / TP4 on 64 cores, reading q_a's replica.
    Site("q_b", 1280, 8192, 64),
    # o_b: K = 8192 / TP4, N = D on 80 cores (row-parallel), reading o_a's replica.
    Site("o_b", 2048, 5120, 80),
    # Engram wkv: K = 6144 / TP4, N = (hc + 1) * D on 80 cores.
    Site("engram_wkv", 1536, 25600, 80),
    # The compressor's column cut (ratio 2: wkv | wgate, Dh / TP each), 8 cores, bfloat8_b.
    Site("compressor", 5120, 256, 8),
    # Indexer wq_b (TP4 cut of 32 heads x 128) on 64 cores and weights_proj (one tile of heads).
    Site("indexer_wq_b", 1280, 4096, 64),
    Site("indexer_weights_proj", 5120, 32, 1),
]


def site_grids(site: Site, device):
    b_grid = rect(site.n_blocks, device)
    out_grid = None
    if site.out == "q_b":
        out_grid = rect(64, device)
    elif site.out == "core00":
        out_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    return b_grid, out_grid


def site_tensors(site: Site, device, rows: int, *, l1: bool, seed: int = 0):
    torch.manual_seed(seed)
    b_grid, out_grid = site_grids(site, device)
    w = torch.randn(site.K, site.N, dtype=torch.bfloat16) / site.K**0.5
    x = torch.randn(rows, site.K, dtype=torch.bfloat16)
    gamma = (1.0 + 0.1 * torch.randn(site.N)).to(torch.bfloat16) if site.rms else None
    weight = l1_weight(device, w, b_grid, site.wdtype) if l1 else dram_weight(device, w, site.n_blocks, site.wdtype)
    kwargs = {}
    if not l1:
        kwargs["weight_core_grid"] = b_grid
    if out_grid is not None:
        kwargs["output_core_grid"] = out_grid
        if out_grid.num_cores() >= 2:
            kwargs["output_mcast_two_hub"] = True
    if site.rms:
        kwargs.update(rms_norm=True, rms_norm_gamma=gamma_tensor(device, gamma, b_grid), rms_norm_epsilon=EPS)
    return w, x, gamma, weight, b_grid, kwargs


def reference(x, w, gamma):
    y = x.float() @ w.float()
    if gamma is not None:
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + EPS) * gamma.float()
    return y


def run_block(device, site: Site, x_rows: torch.Tensor, weight, b_grid, kwargs) -> ttnn.Tensor:
    rows = x_rows.shape[0]
    h = tile_height(rows)
    x_pad = torch.zeros(h if h > 1 else 1, site.K, dtype=torch.bfloat16)
    x_pad[:rows] = x_rows
    a = make_replica(device, x_pad, b_grid)
    out = ttnn.experimental.matmul_decode(a, weight, **kwargs)
    ttnn.deallocate(a)
    return out


@pytest.mark.parametrize("rows", [4, 2, 5])
@pytest.mark.parametrize("site", SITES, ids=[s.name for s in SITES])
def test_full_width_rows(device, site, rows):
    """Replicated A as an ``[h, 32]`` row block == ``rows`` single-row ROW_MAJOR calls, at the site's config."""
    _, out_grid = site_grids(site, device)
    w, x, gamma, weight, b_grid, kwargs = site_tensors(site, device, rows, l1=not site.dram)
    out = run_block(device, site, x, weight, b_grid, kwargs)
    h = tile_height(rows)
    assert out.layout == ttnn.TILE_LAYOUT and out.tile.tile_shape == [h, 32], (out.layout, out.tile)
    assert out.memory_config().shard_spec.shape[0] == h
    got = out_rows(out, rows, site.N)
    if out_grid is not None:
        assert got.shape[0] == out_grid.num_cores()
    ttnn.deallocate(out)

    ref = reference(x, w, gamma)
    singles = []
    for j in range(rows):
        o1 = run_block(device, site, x[j : j + 1], weight, b_grid, kwargs)
        assert o1.layout == ttnn.ROW_MAJOR_LAYOUT
        singles.append(out_rows(o1, 1, site.N))
        ttnn.deallocate(o1)
    single = torch.cat(singles, dim=1)
    for j in range(rows):
        p_ref = min(pcc(ref[j], g[j]) for g in got)
        p_one = min(pcc(single[0, j], g[j]) for g in got)
        diff = max((single[0, j] - g[j]).abs().max().item() for g in got)
        logger.info(f"{site.name} rows={rows} row {j}: pcc vs torch {p_ref:.5f}, vs 1-row {p_one:.6f}, max diff {diff}")
        assert p_ref > 0.99, (site.name, j, p_ref)
        assert p_one > 0.9995, (site.name, j, p_one)
    # Replicas are identical.
    for g in got[1:]:
        assert torch.equal(g, got[0])


def test_full_width_rows_scalar_gamma_groups(device):
    """Fused RMSNorm with a scalar gamma and N split into groups, on 4-row blocks."""
    torch.manual_seed(1)
    K, N, cores, group = 2048, 1024, 16, 256
    b_grid = rect(cores, device)
    w = torch.randn(K, N, dtype=torch.bfloat16) / K**0.5
    x = torch.randn(4, K, dtype=torch.bfloat16)
    weight = l1_weight(device, w, b_grid, ttnn.bfloat16)
    kwargs = dict(rms_norm=True, rms_norm_gamma=0.75, rms_norm_epsilon=EPS, rms_norm_group_size=group)
    a = make_replica(device, x, b_grid)
    got = out_rows(ttnn.experimental.matmul_decode(a, weight, **kwargs), 4, N)[0]
    y = x.float() @ w.float()
    yg = y.reshape(4, N // group, group)
    ref = (yg * torch.rsqrt(yg.square().mean(-1, keepdim=True) + EPS) * 0.75).reshape(4, N)
    for j in range(4):
        p = pcc(ref[j], got[j])
        logger.info(f"grouped rms row {j}: pcc {p:.5f}")
        assert p > 0.99


@pytest.mark.parametrize("k, n, a_cores", [(1024, 4096, 8), (4096, 1024, 8), (7168, 2048, 8)])
@pytest.mark.parametrize("rows", [4, 8])
def test_width_sharded_rows(device, k, n, a_cores, rows):
    """WIDTH_SHARDED A (two-hub gather) as an ``[h, 32]`` row block: custom_mm, rows == single-row calls."""
    torch.manual_seed(0)
    b_grid = row_wise(n // 64, device)
    a_grid = rect(a_cores, device)
    w = torch.randn(k, n, dtype=torch.bfloat16) / k**0.5
    x = torch.randn(rows, k, dtype=torch.bfloat16)
    weight = l1_weight(device, w, b_grid, ttnn.bfloat16)
    h = tile_height(rows)
    a = ttnn.from_torch(
        x,
        layout=ttnn.TILE_LAYOUT,
        tile=ttnn.Tile((h, 32)),
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=shard_cfg((rows, k // a_cores), a_grid, ttnn.ShardStrategy.WIDTH),
    )
    got = ttnn.to_torch(ttnn.experimental.matmul_decode(a, weight)).float().reshape(-1, n)[:rows]
    ref = x.float() @ w.float()
    for j in range(rows):
        a1 = ttnn.from_torch(
            x[j : j + 1],
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            device=device,
            memory_config=shard_cfg((1, k // a_cores), a_grid, ttnn.ShardStrategy.WIDTH),
        )
        one = ttnn.to_torch(ttnn.experimental.matmul_decode(a1, weight)).float().reshape(-1, n)[0]
        p_ref, p_one = pcc(ref[j], got[j]), pcc(one, got[j])
        logger.info(f"width-sharded {k}x{n} row {j}: pcc vs torch {p_ref:.5f}, vs 1-row {p_one:.6f}")
        assert p_ref > 0.99 and p_one > 0.9995


# ---------------------------------------------------------------------------------------------------------------------
# Batched o_a: groups as the batch. TP4: local_o_groups = 2, K = H*Dh / o_groups = 4096, N = o_lora_rank = 1024,
# b_blocks = 2, n_blocks = 32 (64 cores, row-wise), replica on its bbox, output mcast onto o_b's B grid bbox.
# ---------------------------------------------------------------------------------------------------------------------
OA = dict(batch=2, K=4096, N=1024, b_blocks=2, n_blocks=32)


def oa_tensors(device, rows: int, *, l1: bool, wdtype=ttnn.bfloat8_b, seed: int = 0):
    torch.manual_seed(seed)
    batch, K, N, n_blocks = OA["batch"], OA["K"], OA["N"], OA["n_blocks"]
    num_b = OA["b_blocks"] * n_blocks
    b_grid = row_wise(num_b, device)
    w = torch.randn(batch, K, N, dtype=torch.bfloat16) / K**0.5
    folded = w.permute(1, 0, 2).reshape(1, 1, K, batch * N)  # Bc = 1: [K, b_blocks * N]
    if l1:
        weight = l1_weight(device, folded, b_grid, wdtype)
        kwargs = {}
    else:
        weight = dram_weight(device, folded, num_b, wdtype)
        kwargs = {"weight_core_grid": b_grid}
    dest = bbox(rect(80, device))  # o_b's B grid
    kwargs.update(output_core_grid=dest, output_mcast_two_hub=True)
    x = torch.randn(rows, batch, K, dtype=torch.bfloat16)  # row r: every group's K side by side
    return w, x, weight, bbox(b_grid), kwargs


def oa_block(device, x: torch.Tensor, weight, a_grid, kwargs) -> ttnn.Tensor:
    rows, batch, K = x.shape
    h = tile_height(rows)
    m = max(h, 1)
    x_pad = torch.zeros(m, batch, K, dtype=torch.bfloat16)
    x_pad[:rows] = x
    ncores = a_grid.num_cores()
    if h == 1:
        # ROW_MAJOR replica rows ordered [M][batch].
        replica = x_pad.reshape(m * batch, K)
        layout_kw = dict(layout=ttnn.ROW_MAJOR_LAYOUT)
    else:
        # [h, 32]-tile replica: tile row b holds group b's h rows (the bytes of the [h, batch * K] block).
        replica = x_pad.permute(1, 0, 2).reshape(batch * m, K)
        layout_kw = dict(layout=ttnn.TILE_LAYOUT, tile=ttnn.Tile((h, 32)))
    full = replica.repeat(ncores, 1).reshape(1, batch, ncores * m, K)
    a = ttnn.from_torch(
        full,
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=shard_cfg((batch * m, K), a_grid, ttnn.ShardStrategy.HEIGHT),
        **layout_kw,
    )
    out = ttnn.experimental.matmul_decode(a, weight, **kwargs)
    ttnn.deallocate(a)
    return out


@pytest.mark.parametrize("l1", [True, False], ids=["l1_weight", "dram_weight"])
@pytest.mark.parametrize("rows", [4, 5])
def test_batched_o_a_rows(device, rows, l1):
    w, x, weight, a_grid, kwargs = oa_tensors(device, rows, l1=l1)
    batch, N = OA["batch"], OA["N"]
    out = oa_block(device, x, weight, a_grid, kwargs)
    h = tile_height(rows)
    assert out.layout == ttnn.TILE_LAYOUT and out.tile.tile_shape == [h, 32]
    got = out_rows(out, rows, batch * N)
    ref = torch.einsum("rbk,bkn->rbn", x.float(), w.float()).reshape(rows, batch * N)
    for j in range(rows):
        o1 = oa_block(device, x[j : j + 1], weight, a_grid, kwargs)
        one = out_rows(o1, 1, batch * N)[0, 0]
        p_ref = min(pcc(ref[j], g[j]) for g in got)
        p_one = min(pcc(one, g[j]) for g in got)
        logger.info(f"o_a rows={rows} row {j}: pcc vs torch {p_ref:.5f}, vs 1-row {p_one:.6f}")
        assert p_ref > 0.99 and p_one > 0.9995


def test_batched_o_a_from_gather(device):
    """The model's chain: flash_mqa's (4, 32) block ``[4, batch * K]`` on core (0, 0) ->
    ``all_gather_for_matmul(num_groups=batch)`` (labelled ``[1, batch, cores * 4, K]``, shard ``[batch * 4, K]``) ->
    batched o_a, equal to the hand-built replica."""
    rows = 4
    w, x, weight, a_grid, kwargs = oa_tensors(device, rows, l1=True)
    batch, K, N = OA["batch"], OA["K"], OA["N"]
    core = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    block = ttnn.from_torch(
        x.reshape(1, 1, rows, batch * K),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        tile=ttnn.Tile((rows, 32)),
        device=device,
        memory_config=shard_cfg((rows, batch * K), core, ttnn.ShardStrategy.HEIGHT),
    )
    gathered = ttnn.experimental.deepseek.all_gather_for_matmul(block, a_grid, num_groups=batch)
    ncores = a_grid.num_cores()
    assert list(gathered.shape) == [1, batch, ncores * rows, K], gathered.shape
    assert list(gathered.memory_config().shard_spec.shape) == [batch * rows, K]
    out = ttnn.experimental.matmul_decode(gathered, weight, **kwargs)
    got = out_rows(out, rows, batch * N)
    ref_out = oa_block(device, x, weight, a_grid, kwargs)
    ref_dev = out_rows(ref_out, rows, batch * N)
    assert torch.equal(got, ref_dev), "gather-fed o_a differs from the hand-built replica"
    ref = torch.einsum("rbk,bkn->rbn", x.float(), w.float()).reshape(rows, batch * N)
    for j in range(rows):
        p_ref = min(pcc(ref[j], g[j]) for g in got)
        logger.info(f"o_a from gather row {j}: pcc vs torch {p_ref:.5f}")
        assert p_ref > 0.99


# ---------------------------------------------------------------------------------------------------------------------
# Profile: S = 1 (ROW_MAJOR) vs S = 4 ((4, 32) block) at the production configs.
# ---------------------------------------------------------------------------------------------------------------------
PERF_SITES = [s for s in SITES]


def _timed(label: str, fn) -> None:
    out = fn()  # warmup / compile
    ttnn.deallocate(out)
    ttnn.synchronize_device(fn.device)
    _signpost(label)
    for _ in range(ITERATIONS):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(fn.device)


@pytest.mark.parametrize("weight_place", ["l1", "dram"])
def test_profile_matmul_decode_rows(device, weight_place):
    l1 = weight_place == "l1"
    for site in PERF_SITES:
        for wdtype in (ttnn.bfloat4_b,) if site.name != "compressor" else (ttnn.bfloat8_b,):
            site.wdtype = wdtype
            _, x, _, weight, b_grid, kwargs = site_tensors(site, device, 4, l1=l1)
            for rows in (1, 4):
                xr = x[:rows]
                h = tile_height(rows)
                x_pad = torch.zeros(h, site.K, dtype=torch.bfloat16)
                x_pad[:rows] = xr
                a = make_replica(device, x_pad, b_grid)

                def fn():
                    return ttnn.experimental.matmul_decode(a, weight, **kwargs)

                fn.device = device
                _timed(f"{site.name}-{weight_place}-s{rows}", fn)
                ttnn.deallocate(a)
            ttnn.deallocate(weight)
    for rows in (1, 4):
        _, x, weight, a_grid, kwargs = oa_tensors(device, rows, l1=l1, wdtype=ttnn.bfloat4_b)

        def fn():
            return oa_block(device, x, weight, a_grid, kwargs)

        fn.device = device
        _timed(f"o_a-{weight_place}-s{rows}", fn)
        ttnn.deallocate(weight)
    # The brief's WIDTH_SHARDED shapes (L1 weight, 8 gather cores).
    if l1:
        for k, n in [(1024, 4096), (4096, 1024), (7168, 2048)]:
            torch.manual_seed(0)
            b_grid, a_grid = row_wise(n // 64, device), rect(8, device)
            weight = l1_weight(device, torch.randn(k, n, dtype=torch.bfloat16), b_grid, ttnn.bfloat16)
            for rows in (1, 4):
                h = tile_height(rows)
                kw = dict(
                    dtype=ttnn.bfloat16,
                    device=device,
                    memory_config=shard_cfg((rows, k // 8), a_grid, ttnn.ShardStrategy.WIDTH),
                )
                x = torch.randn(rows, k, dtype=torch.bfloat16)
                a = (
                    ttnn.from_torch(x, layout=ttnn.ROW_MAJOR_LAYOUT, **kw)
                    if h == 1
                    else ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, tile=ttnn.Tile((h, 32)), **kw)
                )

                def fn():
                    return ttnn.experimental.matmul_decode(a, weight)

                fn.device = device
                _timed(f"ws{k}x{n}-s{rows}", fn)
                ttnn.deallocate(a)
            ttnn.deallocate(weight)
