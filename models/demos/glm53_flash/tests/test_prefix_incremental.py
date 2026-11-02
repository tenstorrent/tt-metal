# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The incremental whole-prefix latent copy of the sparse prefill (``TtLatentCache.prefix_rows_rm``,
``GLM53_PREFIX_INCREMENTAL=1``): the layer's ROW_MAJOR arena, appended with every chunk's latent through
``ttnn.experimental.slice_write``, must hold bit-exactly the rows the per-chunk rebuild (``user_rows_rm``: the
untilize of the paged cache) produces, on the owner column of the heads2users layout, after every chunk of a
multi-chunk prompt; the layer-3 attention output of a 2-chunk prompt must be bit-equal with the knob on and off.

* ``test_prefix_arena_budget_host``: host only (no device): the ``auto`` budget rule and the block planner's
  invariants (runs on CPU now).
* ``test_prefix_arena_append_random``: random latent rows, a 3-chunk prompt into every user of a heads2users
  cache on the (rows, cols) block: append / rebuild / release / user switch / knob off, bit-equal rows on the
  owner column, timing of the append against the rebuild.
* ``test_dsa_attention_prefix_incremental_long[65536]``: ``TtDSA.prefill_forward(slab=True)`` on the real
  ``R16385`` rows, a [8192, 8192] prompt at the end of a 64 K synthetic prefix, knob 0 vs 1: outputs bit-equal,
  selections identical, the arena == the pages after every chunk, the ``prefill_state`` seeding path appends.

    TT_VISIBLE_DEVICES=11,10,15,14,7,6,3,2 TT_MESH_GRAPH_DESC_PATH=<p150_x8 descriptor> MESH_DEVICE=P150x8 \
        GLM53_MESH_SHAPE=2,4 pytest models/demos/glm53_flash/tests/test_prefix_incremental.py -s --timeout 1500
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

ttnn = pytest.importorskip("ttnn")
pytest.importorskip("models.demos.glm53_flash.tt.dsa")
# isort: off
from models.demos.glm53_flash.tt.dsa import TtLatentCache, sparse_block_plan  # noqa: E402
from models.demos.glm53_flash.tt.mesh import GLMMesh, mesh_device_params  # noqa: E402

# isort: on

TILE = 32
MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "2,4").split(","))
LONG_N = [int(v) for v in os.environ.get("GLM53_PFX_LONG_N", "65536").split(",")]
SEED_CHUNK = 32768
L = 512


# ------------------------------------------------------------------------------------------ host only
class _CacheStub(TtLatentCache):
    """The arena bookkeeping of ``TtLatentCache`` without a device (the budget rule reads three attributes)."""

    def __init__(self, rows_per_user: int, latent_dim: int, n_dsa: int):  # noqa: D401 - no device
        self.rows_per_user, self.latent_dim, self.n_dsa_layers = rows_per_user, latent_dim, n_dsa


def test_prefix_arena_budget_host(monkeypatch):
    # the auto rule: every DSA layer's arena (rows x 512 bf16) within the budget
    monkeypatch.setenv("GLM53_PREFIX_INCREMENTAL", "auto")
    monkeypatch.setenv("GLM53_PREFIX_INCREMENTAL_BUDGET_MB", "1024")
    assert _CacheStub(65536, L, 11).prefix_arena_bytes == 64 * 2**20
    assert _CacheStub(65536, L, 11).prefix_incremental, "11 x 64 MiB = 704 MiB fits a 1 GiB budget"
    assert not _CacheStub(131072, L, 11).prefix_incremental, "11 x 128 MiB = 1.4 GiB does not"
    assert not _CacheStub(1048576, L, 11).prefix_incremental, "11 GiB at 1 M"
    monkeypatch.setenv("GLM53_PREFIX_INCREMENTAL_BUDGET_MB", "12288")
    assert _CacheStub(1048576, L, 11).prefix_incremental
    monkeypatch.setenv("GLM53_PREFIX_INCREMENTAL", "0")
    assert not _CacheStub(1024, L, 11).prefix_incremental
    monkeypatch.setenv("GLM53_PREFIX_INCREMENTAL", "1")
    assert _CacheStub(1048576, L, 11).prefix_incremental
    monkeypatch.setenv("GLM53_PREFIX_INCREMENTAL", "yes")
    with pytest.raises(ValueError):  # allow-pytest.raises: a host-only knob parse, no device fixture involved
        _CacheStub(1024, L, 11).prefix_incremental
    # the block planner: exact cover, sub-blocks <= qblock, attention blocks a multiple of qblock
    for r0, r1, qb, ab in ((0, 2048, 512, 2048), (1024, 16384, 2048, 2048), (0, 2048, 256, 1000), (32, 4128, 512, 32)):
        plan = sparse_block_plan(r0, r1, qb, ab)
        assert [a for a, _, _ in plan][0] == r0 and plan[-1][1] == r1
        assert all(a1 == b0 for (_, a1, _), (b0, _, _) in zip(plan, plan[1:]))
        for a0, a1, subs in plan:
            assert subs[0][0] == a0 and subs[-1][1] == a1 and all(s1 - s0 <= qb for s0, s1 in subs)
            assert all(s1 == t0 for (_, s1), (t0, _) in zip(subs, subs[1:]))
            assert a1 - a0 <= max(qb, (ab // qb) * qb)
    assert sparse_block_plan(0, 2048, 512, 512) == [(a, a + 512, [(a, a + 512)]) for a in range(0, 2048, 512)]
    assert sparse_block_plan(8, 8, 32, 32) == []


# ------------------------------------------------------------------------------------------ helpers
def _cfg():
    from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir
    from models.demos.glm53_flash.tt.config import GLM53FlashConfig

    return GLM53FlashConfig.from_hf_config_json(default_snapshot_dir() / "config.json")


def _bits(mesh: GLMMesh, t: ttnn.Tensor, chip: int, n_rows: int) -> torch.Tensor:
    """Rows [0, n_rows) of a [1,1,N,L] tensor on ``chip`` as raw bf16 bits."""
    return ttnn.to_torch(mesh.device_tensors(t)[chip])[0, 0, :n_rows].contiguous().view(torch.int16)


def _timed(device, fn, repeat: int = 2):
    out = fn()
    ttnn.synchronize_device(device)
    best = float("inf")
    for _ in range(repeat):
        if isinstance(out, ttnn.Tensor) and out.is_allocated():
            ttnn.deallocate(out)
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(device)
        best = min(best, time.perf_counter() - t0)
    return best, out


# -------------------------------------------------------------------------------------- random rows
@pytest.mark.timeout(900)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_prefix_arena_append_random(mesh_device, monkeypatch):
    monkeypatch.setenv("GLM53_PREFIX_INCREMENTAL", "1")
    cfg = _cfg()
    mesh = GLMMesh.from_device(mesh_device)
    T, n_chunks = 4096, 3
    part = mesh.cols > 1
    cache = TtLatentCache(
        mesh_device,
        cfg,
        max_batch=mesh.cols if part else 1,
        max_seq_len=n_chunks * T + 64,
        mesh=mesh,
        partition_users=part,
        prefill_partition="heads2users" if part else None,
    )
    assert cache.prefix_incremental and cache._prefix is None
    g = torch.Generator().manual_seed(11)
    users = list(range(mesh.cols)) if part else [0]
    hosts: dict[int, list[torch.Tensor]] = {}
    for u in users:
        hosts[u] = []
        for k in range(n_chunks):
            c_host = torch.randn(1, 1, T, L, generator=g).to(torch.bfloat16)
            hosts[u].append(c_host)
            c = mesh.upload(c_host, ttnn.bfloat16)  # replicated: the chunk latent every chip computes
            start = k * T
            cache.fill(c, u, start=start)
            kv, owned = cache.prefix_rows_rm(u, start, T, chunk_c=c)
            ttnn.deallocate(c)
            assert not owned and kv is cache._prefix.rows and cache._prefix.user == u
            assert cache._prefix.n_valid == start + T, (cache._prefix.n_valid, start + T)
            assert list(kv.shape) == [1, 1, cache.rows_per_user, L] and kv.layout == ttnn.ROW_MAJOR_LAYOUT
            ref = cache.user_rows_rm(u, start + T)  # the per-chunk rebuild: the untilize of the pages
            owner = cache.owner_of(u)
            for chip in range(mesh.rows * mesh.cols):
                col = chip % mesh.cols
                got = _bits(mesh, kv, chip, start + T)
                if owner is None or col == owner:
                    assert torch.equal(got, _bits(mesh, ref, chip, start + T)), f"user {u} chunk {k} chip {chip}"
                else:
                    # a non-owner column appended the chunk latent into its own arena (its slot holds another
                    # user; the attention it computes from this kv is discarded)
                    want = torch.cat(hosts[u][: k + 1], dim=2)[0, 0].contiguous().view(torch.int16)
                    assert torch.equal(got, want), f"user {u} chunk {k} non-owner chip {chip}"
            ttnn.deallocate(ref)
        logger.info(f"user {u}: {n_chunks} chunks of {T} rows appended, arena == pages on the owner column")
    # the rebuild path: forget the arena's rows (a seeded prefix it did not see), the next chunk rebuilds from the pages
    u = users[-1]
    cache.prefix_sync(u, 0, None)  # a device-slot fill: released
    assert cache._prefix is None
    c = mesh.upload(hosts[u][-1], ttnn.bfloat16)
    kv, owned = cache.prefix_rows_rm(u, (n_chunks - 1) * T, T, chunk_c=c)
    assert not owned and cache._prefix.n_valid == n_chunks * T
    ref = cache.user_rows_rm(u, n_chunks * T)
    owner = cache.owner_of(u)
    for chip in range(mesh.rows * mesh.cols):
        if owner is None or chip % mesh.cols == owner:
            assert torch.equal(_bits(mesh, kv, chip, n_chunks * T), _bits(mesh, ref, chip, n_chunks * T)), chip
    ttnn.deallocate(ref)
    # prefix_sync of a fill that does not continue the arena marks it stale; a continuing one appends
    cache.prefix_sync(u, T, c)  # start T != n_valid: stale
    assert cache._prefix is not None and cache._prefix.n_valid == 0
    kv, _ = cache.prefix_rows_rm(u, (n_chunks - 1) * T, T, chunk_c=c)  # rebuilt
    assert cache._prefix.n_valid == n_chunks * T
    # a new prompt of the same user starts at 0: the arena is reused from row 0
    cache.fill(c, u, start=0)
    kv, _ = cache.prefix_rows_rm(u, 0, T, chunk_c=c)
    assert cache._prefix.n_valid == T
    ref = cache.user_rows_rm(u, T)
    for chip in range(mesh.rows * mesh.cols):
        if owner is None or chip % mesh.cols == owner:
            assert torch.equal(_bits(mesh, kv, chip, T), _bits(mesh, ref, chip, T)), chip
    ttnn.deallocate(ref)
    # another user takes the layer's arena (one prompt in flight per chip)
    if len(users) > 1:
        old = cache._prefix.rows
        cache.fill(c, users[0], start=0)
        kv2, _ = cache.prefix_rows_rm(users[0], 0, T, chunk_c=c)
        assert cache._prefix.user == users[0] and not old.is_allocated() and kv2.is_allocated()
    # timing: append vs rebuild at the longest prefix of this cache
    n_max = n_chunks * T
    cache.release_prefix()
    cache.fill(c, u, start=n_max - T)
    t_rebuild, ref = _timed(mesh_device, lambda: cache.user_rows_rm(u, n_max))
    ttnn.deallocate(ref)
    cache.release_prefix()
    cache.prefix_rows_rm(u, 0, n_max - T)  # the arena holds [0, n_max - T) (a rebuild)

    def append():  # returns a host int: the timer must not deallocate the arena between runs
        cache._prefix.n_valid = n_max - T
        cache.prefix_rows_rm(u, n_max - T, T, chunk_c=c)
        return cache._prefix.n_valid

    t_append, _ = _timed(mesh_device, append)
    ttnn.deallocate(c)
    cache.release_prefix()
    assert cache._prefix is None
    # knob off: a fresh, owned copy of the exact prefix length
    monkeypatch.setenv("GLM53_PREFIX_INCREMENTAL", "0")
    cache2 = TtLatentCache(mesh_device, cfg, max_batch=1, max_seq_len=T + 64, mesh=mesh)
    c0 = mesh.upload(hosts[users[0]][0], ttnn.bfloat16)
    cache2.fill(c0, 0, start=0)
    kv, owned = cache2.prefix_rows_rm(0, 0, T, chunk_c=c0)
    assert owned and list(kv.shape) == [1, 1, T, L] and cache2._prefix is None
    ttnn.deallocate(kv)
    ttnn.deallocate(c0)
    logger.info(
        f"TIMING {MESH} prefix {n_max} rows: rebuild (user_rows_rm) {t_rebuild * 1e3:.2f} ms | append of {T} rows "
        f"(untilize + slice_write) {t_append * 1e3:.2f} ms"
    )


# ------------------------------------------------------------------------------------- the attention
@pytest.mark.timeout(1500)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
@pytest.mark.parametrize("n_pos", LONG_N)
def test_dsa_attention_prefix_incremental_long(mesh_device, monkeypatch, n_pos):
    from models.demos.glm53_flash.tests.test_cache_1m import _x_stats, synthetic_rows
    from models.demos.glm53_flash.tests.test_mesh_heads2users_prefill import (
        R16385,
        _attention,
        _caches,
        _load,
        compare_user,
        prefill_user,
    )
    from models.demos.glm53_flash.tests.utils import attention_input, log_dram

    raw, cfg = _load()
    n_real = 16384
    x_in = torch.load(R16385 / "layer3_prefill.pt")["x_in"][:, :n_real]
    rows_t = attention_input(raw, cfg, x_in)
    start0 = n_pos - n_real
    attn, mesh = _attention(mesh_device, raw, cfg)
    # two heads2users caches of the same geometry: the knob is read at a cache's first sparse chunk
    _, cache_ref = _caches(mesh_device, cfg, mesh, n_pos + 64 * mesh.rows)
    _, cache_inc = _caches(mesh_device, cfg, mesh, n_pos + 64 * mesh.rows)
    user = 0
    monkeypatch.setenv("GLM53_PREFIX_INCREMENTAL", "0")
    assert not cache_ref.prefix_incremental
    monkeypatch.setenv("GLM53_PREFIX_INCREMENTAL", "1")
    assert cache_inc.prefix_incremental
    log_dram(mesh_device, "attention + two heads2users caches")
    if start0:
        mu, sigma = _x_stats(raw, cfg)
        filled, ci = 0, 0
        while filled < start0:
            T = min(SEED_CHUNK, start0 - filled)
            x_tt = mesh.upload(synthetic_rows(mu, sigma, ci, T).reshape(1, 1, T, -1), ttnn.bfloat16)
            attn.prefill_state(x_tt, cache_ref, user, T, start=filled)
            attn.prefill_state(x_tt, cache_inc, user, T, start=filled)
            ttnn.deallocate(x_tt)
            filled += T
            ci += 1
        ttnn.synchronize_device(mesh_device)
    assert cache_inc._prefix is None, "seeding alone never allocates the arena"
    chunks, qb = (8192, 8192), int(os.environ.get("GLM53_PFX_QBLOCK", "2048"))
    failures = []
    ref, ref_idx, w_ref = prefill_user(attn, cache_ref, mesh, rows_t, user, chunks, start0, qb, monkeypatch)
    assert cache_ref._prefix is None
    # chunk by chunk with the arena: after chunk 1 the arena was REBUILT (the seeded prefix predates it), after
    # chunk 2 APPENDED; both must equal the pages' untilize on the owner column
    outs, walls = [], []
    rec_rows: dict[int, torch.Tensor] = {}
    start = start0
    for i, n in enumerate(chunks):
        got_i, rec_i, w_i = prefill_user(
            attn, cache_inc, mesh, rows_t[start - start0 :], user, (n,), start, qb, monkeypatch
        )
        outs.append(got_i)
        walls.append(w_i[0])
        rec_rows.update(rec_i.rows)
        a = cache_inc._prefix
        assert a is not None and a.user == user and a.n_valid == start + n, (a and a.n_valid, start + n)
        pages = cache_inc.user_rows_rm(user, start + n)
        owner = cache_inc.owner_of(user)
        for r in range(mesh.rows):
            chip = r * mesh.cols + owner
            eq = torch.equal(_bits(mesh, a.rows, chip, start + n), _bits(mesh, pages, chip, start + n))
            if not eq:
                failures.append(f"chunk {i}: arena != pages on chip {chip}")
        ttnn.deallocate(pages)
        start += n
    got = torch.cat(outs, dim=0)

    class _Rec:
        rows = rec_rows

    failures += compare_user(ref, ref_idx, got, _Rec, start0, chunks, mesh, f"N={n_pos} incremental vs rebuild")
    bit = torch.equal(ref, got)
    logger.info(
        f"N={n_pos} chunks {list(chunks)}: output bit-equal {bit}; walls rebuild {w_ref} s, incremental {walls} s"
    )
    if not bit:
        failures.append("the attention output differs bitwise between the arena and the per-chunk copy")
    # the seeding path appends to a live arena (prefix_sync) instead of leaving it stale
    mu, sigma = _x_stats(raw, cfg)
    T = 4096
    x_tt = mesh.upload(synthetic_rows(mu, sigma, 99, T).reshape(1, 1, T, -1), ttnn.bfloat16)
    attn.prefill_state(x_tt, cache_inc, user, T, start=n_pos)
    ttnn.deallocate(x_tt)
    a = cache_inc._prefix
    assert a is not None and a.n_valid == n_pos + T, a.n_valid
    pages = cache_inc.user_rows_rm(user, n_pos + T)
    owner = cache_inc.owner_of(user)
    for r in range(mesh.rows):
        chip = r * mesh.cols + owner
        if not torch.equal(_bits(mesh, a.rows, chip, n_pos + T), _bits(mesh, pages, chip, n_pos + T)):
            failures.append(f"prefill_state append: arena != pages on chip {chip}")
    ttnn.deallocate(pages)
    cache_inc.release_prefix(user)
    assert cache_inc._prefix is None
    log_dram(mesh_device, "done")
    assert not failures, "\n".join(failures)
