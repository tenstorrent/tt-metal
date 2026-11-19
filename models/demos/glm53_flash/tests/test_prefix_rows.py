# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The whole-prefix latent copy of the sparse prefill (``TtLatentCache.user_rows_rm``: the sparse_sdpa
row-major kv contract) in isolation on one chip: the old path (paged TILE cache -> ``ttnn.slice`` ->
``ttnn.reshape`` -> ``ttnn.to_layout`` ROW_MAJOR) split into its ops, the direct path (``untilize_with_unpadding``
of the leading block range, no slice), their wall time per prefix length, the bitwise comparison of the two kv
tensors and of the ``sparse_sdpa`` output computed from each (random queries, random valid index rows).

    TT_VISIBLE_DEVICES=0 pytest models/demos/glm53_flash/tests/test_prefix_rows.py -s --timeout 1500

Knobs: GLM53_PREFIX_N (prefix lengths, default 16384,131072,262144,524288,1048576; the cache is allocated for
the largest + 64), GLM53_PREFIX_REPEAT (timed repetitions after the compile pass, default 2).
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir
from models.demos.glm53_flash.tests.utils import log_dram
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.dsa import TtLatentCache

TILE = 32
CURVE = [int(v) for v in os.environ.get("GLM53_PREFIX_N", "16384,131072,262144,524288,1048576").split(",")]
REPEAT = int(os.environ.get("GLM53_PREFIX_REPEAT", "2"))
SEED_CHUNK = 32768
SENTINEL = 0xFFFFFFFF
Q_ROWS, Q_HEADS, TOPK, IDX_W = 128, 32, 2048, 2176  # the prefill's sparse_sdpa geometry (2048 + 3 tail, padded to 128)


def _seed(cache: TtLatentCache, device, n_rows: int, seed: int = 1234) -> None:
    """Random bf16 latent rows [0, n_rows) into slot 0 (the page order of the prefill)."""
    g = torch.Generator().manual_seed(seed)
    for start in range(0, n_rows, SEED_CHUNK):
        T = min(SEED_CHUNK, n_rows - start)
        x = torch.randn(1, 1, T, cache.latent_dim, generator=g).to(torch.bfloat16)
        x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        cache.fill(x_tt, 0, start=start)
        ttnn.deallocate(x_tt)
    ttnn.synchronize_device(device)


def timed(device, fn, repeat: int = REPEAT) -> tuple[float, object]:
    """min wall time over ``repeat`` runs after one compile run; returns (seconds, last result)."""
    out = fn()
    ttnn.synchronize_device(device)
    best = float("inf")
    for _ in range(repeat):
        if isinstance(out, ttnn.Tensor):
            ttnn.deallocate(out)
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(device)
        best = min(best, time.perf_counter() - t0)
    return best, out


def _bits(t: ttnn.Tensor) -> torch.Tensor:
    return ttnn.to_torch(t).contiguous().view(torch.int16)


def _sparse_inputs(device, N: int, L: int, seed: int = 7) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Random row-major queries [1, 32, Q_ROWS, L] and valid index rows [1, 1, Q_ROWS, IDX_W] (2048 distinct ids
    < N per row, ascending, then sentinels) for ``sparse_sdpa`` over an N-row kv."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, Q_HEADS, Q_ROWS, L, generator=g).to(torch.bfloat16)
    ids = torch.full((1, 1, Q_ROWS, IDX_W), SENTINEL, dtype=torch.int64)
    for r in range(Q_ROWS):
        ids[0, 0, r, :TOPK] = torch.randperm(N, generator=g)[:TOPK].sort().values
    q_tt = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    i_tt = ttnn.from_torch(ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    return q_tt, i_tt


def _sparse(q_tt: ttnn.Tensor, kv: ttnn.Tensor, i_tt: ttnn.Tensor, L: int) -> ttnn.Tensor:
    return ttnn.transformer.sparse_sdpa(
        q_tt,
        kv,
        i_tt,
        v_dim=L,
        kv_format=ttnn.transformer.SparseKVFormat.BF16,
        scale=L**-0.5,
        k_chunk_size=128,
    )


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_prefix_rows_timing(mesh_device):
    cfg = GLM53FlashConfig.from_hf_config_json(default_snapshot_dir() / "config.json")
    n_max = CURVE[-1]
    cache = TtLatentCache(mesh_device, cfg, max_batch=1, max_seq_len=n_max + 64)
    L = cache.latent_dim
    log_dram(mesh_device, f"1-slot cache for {n_max + 64} positions")
    t0 = time.perf_counter()
    _seed(cache, mesh_device, n_max)
    logger.info(f"seeded {n_max} random rows in {time.perf_counter() - t0:.1f} s")
    rows = []
    for N in CURVE:
        nb = N // TILE
        res = {"N": N}
        # ---- the old path, op by op
        t_slice, blocks = timed(mesh_device, lambda: ttnn.slice(cache.cache, [0, 0, 0, 0], [nb, 1, TILE, L]))
        # the reshape is a view of ``blocks`` (deallocating it would free the blocks): timed once, kept
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        rows_t = ttnn.reshape(blocks, [1, 1, N, L])
        ttnn.synchronize_device(mesh_device)
        t_reshape = time.perf_counter() - t0
        t_untilize, rm_old = timed(mesh_device, lambda: ttnn.to_layout(rows_t, ttnn.ROW_MAJOR_LAYOUT))
        res.update(slice=t_slice, reshape=t_reshape, untilize=t_untilize)
        ttnn.deallocate(rm_old)
        ttnn.deallocate(blocks)
        # ---- the two whole paths
        t_old, rm_ref = timed(mesh_device, lambda: cache.user_rows_rm(0, N, direct=False))
        t_new, rm_new = timed(mesh_device, lambda: cache.user_rows_rm(0, N))
        res.update(old=t_old, new=t_new, direct=N >= cache.DIRECT_UNTILIZE_MIN_ROWS)
        assert list(rm_new.shape) == [1, 1, N, L] and rm_new.layout == ttnn.ROW_MAJOR_LAYOUT, rm_new.shape
        ref_bits = _bits(rm_ref)
        res["kv_equal"] = bool(torch.equal(ref_bits, _bits(rm_new)))
        # ---- sparse_sdpa from each kv (the consumer of the copy)
        q_tt, i_tt = _sparse_inputs(mesh_device, N, L)
        t_sdpa, o_ref = timed(mesh_device, lambda: _sparse(q_tt, rm_ref, i_tt, L))
        _, o_new = timed(mesh_device, lambda: _sparse(q_tt, rm_new, i_tt, L))
        res["sdpa_equal"] = bool(torch.equal(_bits(o_ref), _bits(o_new)))
        res["sdpa_ms"] = t_sdpa * 1e3
        for t in (o_ref, o_new, q_tt, i_tt, rm_ref, rm_new):
            ttnn.deallocate(t)
        alloc, free = log_dram(mesh_device, f"N={N} done")
        res["dram_alloc_gib"] = alloc
        rows.append(res)
        logger.info(
            f"N={N}: old {t_old * 1e3:.1f} ms (slice {t_slice * 1e3:.1f} + reshape {t_reshape * 1e3:.2f} + untilize "
            f"{t_untilize * 1e3:.1f}); new {t_new * 1e3:.1f} ms (direct {res['direct']}); kv bitwise equal "
            f"{res['kv_equal']}; sparse_sdpa({Q_ROWS} rows) {t_sdpa * 1e3:.1f} ms, output bitwise equal {res['sdpa_equal']}"
        )
    logger.info(
        "| prefix N | GB (RM out) | old ms | = slice | + reshape | + untilize | new ms | direct | old GB/s | new GB/s | "
        "kv == | sdpa == |"
    )
    for r in rows:
        gb = r["N"] * L * 2 / 1e9
        logger.info(
            f"| {r['N']} | {gb:.3f} | {r['old'] * 1e3:.1f} | {r['slice'] * 1e3:.1f} | {r['reshape'] * 1e3:.2f} | "
            f"{r['untilize'] * 1e3:.1f} | {r['new'] * 1e3:.1f} | {r['direct']} | {gb / r['old']:.1f} | "
            f"{gb / r['new']:.1f} | {r['kv_equal']} | {r['sdpa_equal']} |"
        )
    assert all(r["kv_equal"] and r["sdpa_equal"] for r in rows), rows
