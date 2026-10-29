# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Heads2users prefill of the user-partitioned DSA cache on a (rows, 4) block (doc/mesh/sparse_1m.md section 7:
``GLM53_PREFILL_PARTITION=heads2users``, no prefill scratch slot). Column c of a row owns user c; during the prefill
of user u the owner column attends every sub-block with all 64 heads (gathered over the columns) over its own slot,
the other columns write the chunk into dummy blocks and compute discarded rows of the same shapes; the inverse
all-to-all + slice hands every column the owner's output for its head shard.

* ``test_owner_exchange_primitives``: the gather / inverse all-to-all / slice and the prefix gather on random data
  (host emulation of what every column must end up with).
* ``test_dsa_attention_heads2users_r4097``: layer-3 attention (``TtDSA.prefill_forward(slab=True)``) on the real
  4097-token prompt ``R4097``: reference = a non-partitioned cache (one replicated slot); under test = a heads2users
  partitioned cache (4 users per row, one slot per chip) with the same rows prefilled into users 0..3 one after the
  other (single shot, [2048, 2048, 1], [1024, 2048, 1025]: a chunk starting inside the dense region needs the owner's
  prefix). Per user: output rows vs the reference, the sparse selections (read on the owner column), and after the
  four prefills the state of user c on column c vs the reference slot (the dummy writes leave live slots alone).
* ``test_dsa_attention_heads2users_long[65536]``: 16384 real rows (``R16385``) at the end of a 64 K prefix (seeded
  synthetically), [16384] and [8192, 8192], users 0 and 1, plus warm walls reference vs heads2users.

    TT_VISIBLE_DEVICES=11,10,15,14,7,6,3,2 TT_MESH_GRAPH_DESC_PATH=<p150_x8 descriptor> MESH_DEVICE=P150x8 \
        GLM53_MESH_SHAPE=2,4 pytest models/demos/glm53_flash/tests/test_mesh_heads2users_prefill.py -s --timeout 1500
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir, load_layer_raw, shard_headers
from models.demos.glm53_flash.tests.test_cache_1m import _attn_keys, _x_stats, synthetic_rows
from models.demos.glm53_flash.tests.test_mesh_slab_prefill import (
    DENSE_ROWS,
    SENTINEL,
    _chunk_to_device,
    _rows_out,
    _slabs_of,
    class_report,
)
from models.demos.glm53_flash.tests.test_prefill_chunked import compare_indices, row_pcc
from models.demos.glm53_flash.tests.utils import attention_input, pcc
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.dsa import DUMMY_BLOCKS, TtDSA, TtLatentCache
from models.demos.glm53_flash.tt.mesh import GLMMesh, mesh_device_params

TILE = 32
LAYER = 3
MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "2,4").split(","))
REF = Path("bringup/references/glm53_flash")
R4097, R16385 = REF / "R4097", REF / "R16385"
ROW_PCC_BAR = float(os.environ.get("GLM53_H2U_PCC_BAR", "0.99999"))
SEED_CHUNK = 32768
LONG_N = [int(v) for v in os.environ.get("GLM53_H2U_LONG_N", "65536").split(",")]


# ------------------------------------------------------------------------------------------ helpers
def _load():
    snapshot = default_snapshot_dir()
    raw = load_layer_raw(snapshot, LAYER, shard_headers(snapshot), name_filter=_attn_keys)
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    return raw, cfg


def _attention(mesh_device, raw, cfg) -> tuple[TtDSA, GLMMesh]:
    mesh = GLMMesh.from_device(mesh_device)
    attn = TtDSA(
        mesh_device,
        cfg,
        {k[len("self_attn.") :]: v for k, v in raw.items() if k.startswith("self_attn.")},
        layer_idx=LAYER,
        mesh=mesh,
    )
    return attn, mesh


def _caches(mesh_device, cfg, mesh: GLMMesh, max_seq_len: int) -> tuple[TtLatentCache, TtLatentCache]:
    """(reference: one replicated slot; heads2users: ``cols`` users per row = one slot per chip + the dummies)."""
    ref = TtLatentCache(mesh_device, cfg, max_batch=1, max_seq_len=max_seq_len, mesh=mesh)
    part = TtLatentCache(
        mesh_device,
        cfg,
        max_batch=mesh.cols,
        max_seq_len=max_seq_len,
        mesh=mesh,
        partition_users=True,
        prefill_partition="heads2users",
    )
    assert part.partition and part.prefill_mode == "heads2users" and part.scratch is None
    assert part.users_per_chip == 1 and part.slots == 1 and part.dummy_blocks == DUMMY_BLOCKS
    assert part.max_blocks == part.blocks_per_seq + DUMMY_BLOCKS and tuple(part.page_table_host.shape) == (
        2,
        part.blocks_per_seq,
    )
    assert (
        part.index.dummy
        and part.index.dummy_slot == 1
        and part.index.pools_per_user - TILE >= -(-max_seq_len // part.index.P)
    )
    for u in range(mesh.cols):
        assert part.owner_of(u) == u and part.prefill_slot(u) == 0 and part.index.prefill_slot(u) == 0
    assert ref.owner_of(0) is None
    return ref, part


class OwnerIndexRecorder:
    """``TtDSA.on_indices`` sink reading the index rows on column ``owner`` of each mesh row (the slab path hands
    ``(row_starts, indices)``; positions below the dense boundary are the superset's clamped rows, skipped)."""

    def __init__(self, mesh: GLMMesh, max_pos: int, owner: int):
        self.mesh, self.max_pos, self.owner = mesh, max_pos, owner
        self.rows: dict[int, torch.Tensor] = {}

    def __call__(self, query_start, indices: ttnn.Tensor):
        devs = self.mesh.device_tensors(indices)
        starts = query_start if isinstance(query_start, tuple) else (query_start,)
        for r, s in enumerate(starts):
            t = ttnn.to_torch(devs[r * self.mesh.cols + self.owner]).reshape(-1, indices.shape[-1]).to(torch.int64)
            t = t & SENTINEL
            for i in range(t.shape[0]):
                pos = s + i
                if pos < DENSE_ROWS or pos >= self.max_pos:
                    continue
                row = t[i]
                sent = (row == SENTINEL).nonzero()
                first = int(sent[0]) if sent.numel() else row.numel()
                assert bool((row[first:] == SENTINEL).all()), f"row {pos}: a sentinel inside the valid prefix"
                self.rows[pos] = row[:first].sort().values


def prefill_user(attn, cache, mesh, rows_t, user, chunks, start0, qblock, env, record=True):
    """Prefill rows [start0, start0 + sum(chunks)) into ``user`` of ``cache`` chunk by chunk (slab-local path).
    Returns (output [n, 4096] fp32, OwnerIndexRecorder, per-chunk wall seconds)."""
    env.setenv("GLM53_PREFILL_QBLOCK", str(qblock))
    owner = cache.owner_of(user)
    rec = OwnerIndexRecorder(mesh, start0 + sum(chunks), 0 if owner is None else owner)
    attn.on_indices = rec if record else None
    outs, walls, start = [], [], start0
    for n in chunks:
        x = _chunk_to_device(mesh, rows_t, start - start0, n, True)
        ttnn.synchronize_device(mesh.device)
        t0 = time.perf_counter()
        y = attn.prefill_forward(x, cache, user, n, start=start, slab=True)
        ttnn.synchronize_device(mesh.device)
        walls.append(time.perf_counter() - t0)
        outs.append(_rows_out(mesh, y, n, True))
        ttnn.deallocate(y)
        ttnn.deallocate(x)
        start += n
    attn.on_indices = None
    return torch.cat(outs, dim=0), rec, walls


def _snapshot_chip(cache: TtLatentCache, mesh: GLMMesh, user: int, n_pos: int, chip: int):
    """(latent rows, complete pools, ring k/g rows) of ``user``'s slot as held on ``chip`` (row-major chip order)."""
    P = cache.index.P
    n_pools, tail = n_pos // P, n_pos % P
    slot = cache.index.prefill_slot(user)
    lat_t = cache.user_rows(user, n_pos)
    lat = ttnn.to_torch(mesh.device_tensors(lat_t)[chip])[0, 0, :n_pos].float()
    ttnn.deallocate(lat_t)
    pools = ttnn.to_torch(mesh.device_tensors(cache.index.pools_batched())[chip])[slot, 0, :n_pools].float()
    rk = ttnn.to_torch(mesh.device_tensors(cache.index.ring_k)[chip])[slot, 0, :tail].float()
    rg = ttnn.to_torch(mesh.device_tensors(cache.index.ring_g)[chip])[slot, 0, :tail].float()
    return lat, pools, rk, rg


def compare_user_state(ref: TtLatentCache, part: TtLatentCache, mesh: GLMMesh, user: int, n_pos: int):
    """User ``user``'s slot on its owner column (every mesh row) vs the reference slot 0 on the same chip."""
    owner = part.owner_of(user)
    names = ("latent", "pools", "ring_k", "ring_g")
    parts, ok = [], True
    for r in range(mesh.rows):
        chip = r * mesh.cols + owner
        a, b = _snapshot_chip(ref, mesh, 0, n_pos, chip), _snapshot_chip(part, mesh, user, n_pos, chip)
        for name, x, y in zip(names, a, b):
            if x.numel() == 0:
                continue
            eq = torch.equal(x, y)
            ok &= eq
            if not eq:
                d = (x - y).abs()
                parts.append(
                    f"row {r} {name} {tuple(x.shape)} DIFFER: {int((d > 0).sum())} of {d.numel()}, max|diff| {d.max():.3g}, "
                    f"pcc {pcc(x, y):.7f}"
                )
    return (
        "; ".join(parts) if parts else f"user {user}: latent / pools / rings equal to the reference on column {owner}"
    ), ok


def compare_user(ref, ref_idx, got, got_idx, start0, chunks, mesh, label) -> list[str]:
    r = row_pcc(ref, got)
    below = int((r < ROW_PCC_BAR).sum())
    exact = int((ref == got).all(dim=1).sum())
    itxt, n_sparse, n_diff = (
        compare_indices(ref_idx.rows, got_idx.rows) if ref_idx.rows else ("no sparse rows recorded", 0, 0)
    )
    logger.info(
        f"{label}: rows {ref.shape[0]}: pcc min {r.min():.7f} (row {int(r.argmin()) + start0}) median {r.median():.7f}; "
        f"rows < {ROW_PCC_BAR} {below}; bit-exact rows {exact}; max|diff| {(ref - got).abs().max():.3g}\n      "
        + class_report(ref, got, start0, _slabs_of(chunks, mesh.rows))
    )
    logger.info(f"    selections: {itxt}")
    failures = []
    if below:
        failures.append(f"{label}: {below} rows below {ROW_PCC_BAR} (min {r.min():.7f})")
    if n_diff:
        failures.append(f"{label}: {n_diff} of {n_sparse} sparse rows select differently")
    return failures


# ------------------------------------------------------------------------------------- primitives
@pytest.mark.timeout(900)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_owner_exchange_primitives(mesh_device):
    mesh = GLMMesh.from_device(mesh_device)
    rows, cols = mesh.rows, mesh.cols
    assert cols > 1, "the owner exchange needs a multi-column mesh"
    H_loc, H, n, L = 16, 64, 128, 512
    torch.manual_seed(0)
    # heads -> every column: chip (r, c) holds heads [16c, 16c+16) of row r's queries
    q_host = torch.randn(rows, H, n, L).bfloat16()
    q = mesh.upload(q_host, ttnn.bfloat16, rows=0, cols=1)  # [1,16,n,L] per chip
    g = mesh.all_gather_cols(q, dim=1)  # [1,64,n,L]
    for chip, t in enumerate(mesh.device_tensors(g)):
        r = chip // cols
        assert torch.equal(ttnn.to_torch(t), q_host[r : r + 1]), f"gather: chip {chip} differs"
    # owner's output -> head shards: chip (r, c) holds a distinct [1,64,n,L] "attention output"; after the all-to-all
    # (heads split by column, sources stacked on dim 0) entry ``owner`` is the owner's heads [16d, 16d+16) on column d
    o_host = torch.randn(rows, cols * H, n, L).bfloat16()  # [r, c*64:(c+1)*64] = column c's output
    o = mesh.upload(o_host, ttnn.bfloat16, rows=0, cols=1)  # [1,64,n,L] per chip
    o_x = mesh.all_to_all_cols(o, in_dim=1, out_dim=0)  # [cols,16,n,L]
    assert tuple(o_x.shape) == (cols, H_loc, n, L), tuple(o_x.shape)
    shards = [ttnn.to_torch(t) for t in mesh.device_tensors(o_x)]
    for owner in range(cols):
        sl = ttnn.slice(o_x, [owner, 0, 0, 0], [owner + 1, H_loc, n, L])
        for chip, t in enumerate(mesh.device_tensors(sl)):
            r, d = divmod(chip, cols)
            want = o_host[r : r + 1, owner * H + d * H_loc : owner * H + (d + 1) * H_loc]
            assert torch.equal(shards[chip][owner : owner + 1], want), f"a2a: chip {chip} entry {owner} differs"
            assert torch.equal(ttnn.to_torch(t), want), f"slice: chip {chip} owner {owner} differs"
        ttnn.deallocate(sl)
    # the prefix gather: chip (r, c) holds its own [1,1,start,L]; the owner's entry reaches every column
    start = 1024
    p_host = torch.randn(rows, cols, start, L).bfloat16()
    p = mesh.upload(p_host, ttnn.bfloat16, rows=0, cols=1)  # [1,1,start,L] per chip
    pg = mesh.all_gather_cols(p, dim=0)  # [cols,1,start,L]
    for chip, t in enumerate(mesh.device_tensors(pg)):
        r = chip // cols
        assert torch.equal(ttnn.to_torch(t), p_host[r].reshape(cols, 1, start, L)), f"prefix gather: chip {chip}"
    logger.info(f"owner exchange primitives on {MESH}: gather, all-to-all + slice and the prefix gather exact")


# ----------------------------------------------------------------------------------- layer, R4097
@pytest.mark.timeout(1500)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_dsa_attention_heads2users_r4097(mesh_device, monkeypatch):
    raw, cfg = _load()
    x_in = torch.load(R4097 / f"layer{LAYER}_prefill.pt")["x_in"]
    S = x_in.shape[1]
    rows_t = attention_input(raw, cfg, x_in)  # [S, 4096] bf16
    attn, mesh = _attention(mesh_device, raw, cfg)
    ref_cache, part = _caches(mesh_device, cfg, mesh, S + 64 * mesh.rows)
    failures = []
    for chunks, qb in (((S,), 2048), ((2048, 2048, 1), 2048), ((1024, 2048, 1025), 2048)):
        ref, ref_idx, w_ref = prefill_user(attn, ref_cache, mesh, rows_t, 0, chunks, 0, qb, monkeypatch)
        walls = []
        for u in range(mesh.cols):
            got, got_idx, w_got = prefill_user(attn, part, mesh, rows_t, u, chunks, 0, qb, monkeypatch)
            walls.append(sum(w_got))
            failures += compare_user(
                ref,
                ref_idx,
                got,
                got_idx,
                0,
                chunks,
                mesh,
                f"chunks {list(chunks)} user {u} (column {part.owner_of(u)})",
            )
        # every user's slot on its own column holds exactly the reference state (the later prefills' dummy writes
        # did not touch it)
        for u in range(mesh.cols):
            txt, ok = compare_user_state(ref_cache, part, mesh, u, S)
            logger.info(f"    state after all {mesh.cols} prefills: {txt}")
            if not ok:
                failures.append(f"chunks {list(chunks)}: user {u} state: {txt}")
        logger.info(
            f"chunks {list(chunks)}: wall reference {sum(w_ref) * 1e3:.0f} ms, heads2users per user "
            f"{[f'{w * 1e3:.0f}' for w in walls]} ms"
        )
    assert not failures, "\n".join(failures)


# ------------------------------------------------------------------------------------- long prefix
@pytest.mark.timeout(1500)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
@pytest.mark.parametrize("n_pos", LONG_N)
def test_dsa_attention_heads2users_long(mesh_device, monkeypatch, n_pos):
    raw, cfg = _load()
    L = 16384
    x_in = torch.load(R16385 / f"layer{LAYER}_prefill.pt")["x_in"][:, :L]
    rows_t = attention_input(raw, cfg, x_in)  # [L, 4096]
    start0 = n_pos - L
    attn, mesh = _attention(mesh_device, raw, cfg)
    ref_cache, part = _caches(mesh_device, cfg, mesh, n_pos + 64 * mesh.rows)
    # GLM53_H2U_USERS: "all" = every column's user (the 1 M x 32-user run's column-3 divergence), else "0,1"
    _us = os.environ.get("GLM53_H2U_USERS", "0,1")
    users = list(range(mesh.cols)) if _us == "all" else [u for u in map(int, _us.split(",")) if u < mesh.cols]
    if start0:
        mu, sigma = _x_stats(raw, cfg)
        filled, ci = 0, 0
        t0 = time.perf_counter()
        while filled < start0:
            T = min(SEED_CHUNK, start0 - filled)
            x = synthetic_rows(mu, sigma, ci, T)
            x_tt = mesh.upload(x.reshape(1, 1, T, -1), ttnn.bfloat16)
            attn.prefill_state(x_tt, ref_cache, 0, T, start=filled)
            for u in users:
                attn.prefill_state(x_tt, part, u, T, start=filled)
            ttnn.deallocate(x_tt)
            filled += T
            ci += 1
        ttnn.synchronize_device(mesh_device)
        logger.info(
            f"seeded [0, {start0}) synthetically into the reference slot and users {users} ({ci} chunks, "
            f"{time.perf_counter() - t0:.1f} s)"
        )
    failures = []
    qb = int(os.environ.get("GLM53_H2U_QBLOCK", "2048"))
    for chunks in ((L,), (8192, 8192)):
        ref, ref_idx, w_ref = prefill_user(attn, ref_cache, mesh, rows_t, 0, chunks, start0, qb, monkeypatch)
        for u in users:
            got, got_idx, w_got = prefill_user(attn, part, mesh, rows_t, u, chunks, start0, qb, monkeypatch)
            failures += compare_user(
                ref, ref_idx, got, got_idx, start0, chunks, mesh, f"N={n_pos} chunks {list(chunks)} user {u}"
            )
        for u in users:
            txt, ok = compare_user_state(ref_cache, part, mesh, u, n_pos)
            logger.info(f"    state: {txt}")
            if not ok:
                failures.append(f"N={n_pos} chunks {list(chunks)}: user {u} state: {txt}")
        # warm walls: the same chunking again into the same slots (identical state), no index recorder
        _, _, w_ref2 = prefill_user(attn, ref_cache, mesh, rows_t, 0, chunks, start0, qb, monkeypatch, record=False)
        _, _, w_got2 = prefill_user(attn, part, mesh, rows_t, 0, chunks, start0, qb, monkeypatch, record=False)
        for i, n in enumerate(chunks):
            s0 = start0 + sum(chunks[:i])
            logger.info(
                f"TIMING N={n_pos} chunk [{s0}, {s0 + n}) (sub-block {qb}) on {MESH}: reference (replicated slot, "
                f"heads->sequence a2a) warm {w_ref2[i] * 1e3:.0f} ms | heads2users warm {w_got2[i] * 1e3:.0f} ms "
                f"({n / w_got2[i]:.0f} tok/s) -> x{w_ref2[i] / w_got2[i]:.2f}"
            )
    assert not failures, "\n".join(failures)


# ------------------------------------------------------------------------------------- waves (per-row chunks)
def _wave_rows_in(mesh: GLMMesh, chunks: list[torch.Tensor], T: int) -> ttnn.Tensor:
    """Row r's chunk rows ``chunks[r]`` [n_r, 4096] bf16 padded to ``T`` -> [1,1,T,4096] per row (row-sharded)."""
    x = torch.zeros(1, 1, mesh.rows * T, chunks[0].shape[-1], dtype=torch.bfloat16)
    for r, c in enumerate(chunks):
        x[0, 0, r * T : r * T + c.shape[0]] = c
    return mesh.upload(x, ttnn.bfloat16, rows=2)


def _wave_rows_out(mesh: GLMMesh, y: ttnn.Tensor, valids: tuple[int, ...]) -> list[torch.Tensor]:
    """Row r's first ``valids[r]`` output rows [n_r, 4096] fp32 (its columns must agree: the all-reduced o_proj)."""
    grid = _shards(mesh, y)
    out = []
    for r in range(mesh.rows):
        for c in range(1, mesh.cols):
            assert torch.equal(grid[r][c], grid[r][0]), (r, c, "columns differ")
        out.append(grid[r][0][0, 0, : valids[r]].float())
    return out


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_dsa_attention_wave_vs_slab(mesh_device, monkeypatch):
    """Layer-3 attention through a WAVE (``TtDSA.prefill_forward(wave=WaveInfo)``: every mesh row its own prompt in slot
    ``user`` from its own start, per-chip page-table / pool views, the shape-static dense prefix with row-sharded Q
    offsets, the sparse path with the rows' true positions) against the slab-local prefill of each prompt ALONE in the
    same user slot (its owner column): (1) dense regime, prompt A (dump rows [0, 200)) on row 0 and prompt B (rows
    [200, 500)) on row 1 at T = 384, then the prompts swapped across the rows; (2) the sparse regime: A = rows [0,
    2500) and B = rows [1500, 3800) in two waves (A[:2048] | B[:1024] at start 0, then A[2048:] at 2048 | B[1024:] at
    1024, T = 1280). Per-row PCC of every prompt's rows vs its slab prefill > ROW_PCC_BAR."""
    from models.demos.glm53_flash.tt.decoder_layer import WaveInfo

    raw, cfg = _load()
    x_in = torch.load(R4097 / f"layer{LAYER}_prefill.pt")["x_in"]
    rows_t = attention_input(raw, cfg, x_in)  # [S, 4096] bf16
    attn, mesh = _attention(mesh_device, raw, cfg)
    assert mesh.rows == 2, "the wave case below pairs two rows"
    _, part = _caches(mesh_device, cfg, mesh, 4096 + 64 * mesh.rows)
    monkeypatch.setenv("GLM53_PREFILL_QBLOCK", "2048")
    user = 1  # owner column 1
    failures = []

    def wave(chunks, starts, valids, T):
        x = _wave_rows_in(mesh, chunks, T)
        info = WaveInfo(T=T, starts=starts, valids=valids, start_t=None, end_t=None)
        y = attn.prefill_forward(x, part, user, T, start=0, wave=info)
        out = _wave_rows_out(mesh, y, valids)
        ttnn.deallocate(y)
        ttnn.deallocate(x)
        return out

    def check(label, ref, got):
        r = row_pcc(ref, got)
        below = int((r < ROW_PCC_BAR).sum())
        exact = int((ref == got).all(dim=1).sum())
        logger.info(
            f"{label}: rows {ref.shape[0]}: pcc min {r.min():.7f} (row {int(r.argmin())}) median {r.median():.7f}; "
            f"rows < {ROW_PCC_BAR} {below}; bit-exact rows {exact}; max|diff| {(ref - got).abs().max():.3g}"
        )
        if below:
            failures.append(f"{label}: {below} rows below {ROW_PCC_BAR} (min {r.min():.7f})")

    # (1) dense regime, both orders
    a, b = rows_t[0:200], rows_t[200:500]
    ref_a, _, _ = prefill_user(attn, part, mesh, a, user, (200,), 0, 2048, monkeypatch, record=False)
    ref_b, _, _ = prefill_user(attn, part, mesh, b, user, (300,), 0, 2048, monkeypatch, record=False)
    got = wave([a, b], (0, 0), (200, 300), 384)
    check("dense wave A on row 0", ref_a, got[0])
    check("dense wave B on row 1", ref_b, got[1])
    got = wave([b, a], (0, 0), (300, 200), 384)
    check("dense wave B on row 0", ref_b, got[0])
    check("dense wave A on row 1", ref_a, got[1])
    # (2) the sparse regime in two waves with per-row starts
    A, B = rows_t[0:2500], rows_t[1500:3800]
    ref_A, _, _ = prefill_user(attn, part, mesh, A, user, (2048, 452), 0, 2048, monkeypatch, record=False)
    ref_B, _, _ = prefill_user(attn, part, mesh, B, user, (1024, 1276), 0, 2048, monkeypatch, record=False)
    got1 = wave([A[:2048], B[:1024]], (0, 0), (2048, 1024), 2048)
    check("sparse wave 1: A[:2048] on row 0", ref_A[:2048], got1[0])
    check("sparse wave 1: B[:1024] on row 1", ref_B[:1024], got1[1])
    got2 = wave([A[2048:], B[1024:]], (2048, 1024), (452, 1276), 1280)
    check("sparse wave 2: A[2048:2500] on row 0 (start 2048)", ref_A[2048:], got2[0])
    check("sparse wave 2: B[1024:2300] on row 1 (start 1024)", ref_B[1024:], got2[1])
    assert not failures, "\n".join(failures)
