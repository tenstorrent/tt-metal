# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Slab-local sequence-parallel DSA prefill (``TtDSA._prefill_slab``) vs the v1 path on a (rows, 4) block.

v1 (``GLM53_SP_ATTN=v1``): every mesh row gathers the chunk's hidden, attends over ALL T queries and keeps
its slab through a reduce-scatter (rows x the attention work). Slab-local (default): every row attends its
own T/rows queries against the full prefix; only the chunk's latent and index stems are gathered.

* ``test_slab_primitives``: the building blocks on the block -- the windowed SDPA with a row-sharded
  ``windowed_q_token_offset_tensor`` == the plain causal kernel's rows of each slab (bit-exact), ``where``
  with a row-sharded [n,1] mask, ``all_gather_rows`` of the narrow latent / index stems.
* ``test_dsa_attention_slab_r4097``: layer-3 attention (``TtDSA.prefill_forward`` on the normalized attention
  inputs of the real 4097-token prompt ``R4097``), chunkings that produce all-dense, mixed and all-sparse
  chunks (and a mixed slab): slab-local (slot 1) vs v1 (slot 0) row by row and per row class, the sparse
  rows' selections, the two slots' cache / pool / ring state.
* ``test_dsa_layer_slab_r4097``: the whole layer 3 through ``FunctionalDecoder`` (mHC + attention + MoE),
  slab vs v1 and both vs the HF dump, then 2 decode steps on both slots.
* ``test_dsa_attention_slab_long[16384|65536]``: 16384 real rows (``R16385``) at the end of a 16 K / 64 K
  context (prefix seeded synthetically through ``prefill_state``), one 16384 chunk and 8192-token chunks,
  slab vs v1, plus the warmed per-chunk wall time and stage breakdown of both paths (the 8 K-chunk timing).

    TT_VISIBLE_DEVICES=27,26,31,30,23,22,19,18 TT_MESH_GRAPH_DESC_PATH=<p150_x8 descriptor> MESH_DEVICE=P150x8 \
        GLM53_MESH_SHAPE=2,4 pytest models/demos/glm53_flash/tests/test_mesh_slab_prefill.py -s --timeout 1500
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
from models.demos.glm53_flash.tests.test_prefill_chunked import STAGE_GROUPS, compare_indices, row_pcc
from models.demos.glm53_flash.tests.utils import attention_input, pcc
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.dsa import StageTimer, TtDSA, TtLatentCache
from models.demos.glm53_flash.tt.functional_decoder import (
    FunctionalDecoder,
    device_streams_to_host,
    host_streams_to_device,
)
from models.demos.glm53_flash.tt.mesh import GLMMesh, mesh_device_params

TILE = 32
LAYER = 3
MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "2,4").split(","))
REF = Path("bringup/references/glm53_flash")
R4097, R16385 = REF / "R4097", REF / "R16385"
SENTINEL = 0xFFFFFFFF
DENSE_ROWS = 2048
ROW_PCC_BAR = float(os.environ.get("GLM53_SLAB_PCC_BAR", "0.99999"))
HF_PCC_BAR = 0.995
SEED_CHUNK = 32768
LONG_N = [int(v) for v in os.environ.get("GLM53_SLAB_LONG_N", "16384,65536").split(",")]


# ------------------------------------------------------------------------------------------ helpers
def _load(attn_only: bool):
    snapshot = default_snapshot_dir()
    raw = load_layer_raw(snapshot, LAYER, shard_headers(snapshot), name_filter=_attn_keys if attn_only else None)
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    return raw, cfg


def _attention(mesh_device, raw, cfg, max_batch: int, max_seq_len: int):
    mesh = GLMMesh.from_device(mesh_device)
    attn = TtDSA(
        mesh_device,
        cfg,
        {k[len("self_attn.") :]: v for k, v in raw.items() if k.startswith("self_attn.")},
        layer_idx=LAYER,
        mesh=mesh,
    )
    return attn, TtLatentCache(mesh_device, cfg, max_batch=max_batch, max_seq_len=max_seq_len, mesh=mesh), mesh


def _chunk_to_device(
    mesh: GLMMesh, rows_t: torch.Tensor, start: int, n: int, slab: bool, t_pad: int | None = None
) -> ttnn.Tensor:
    """rows [S, 4096] bf16 -> the chunk [start, start + n) as [1,1,T_pad,4096] bf16 TILE, T_pad a multiple of
    32 * rows (or the given ``t_pad``: a served bucket): replicated over the mesh (v1 feeds the gathered chunk) or
    row-sharded on dim 2 (the slabs)."""
    unit = TILE * mesh.rows
    T = -(-n // unit) * unit if t_pad is None else int(t_pad)
    assert T >= n and T % unit == 0, (T, n, unit)
    x = torch.zeros(1, 1, T, rows_t.shape[-1], dtype=torch.bfloat16)
    x[0, 0, :n] = rows_t[start : start + n]
    return mesh.upload(x, ttnn.bfloat16, rows=2 if slab else None)


def _shards(mesh: GLMMesh, t: ttnn.Tensor) -> list[list[torch.Tensor]]:
    sh = [ttnn.to_torch(d) for d in mesh.device_tensors(t)]
    return [[sh[r * mesh.cols + c] for c in range(mesh.cols)] for r in range(mesh.rows)]


def _rows_out(mesh: GLMMesh, y: ttnn.Tensor, n: int, slab: bool) -> torch.Tensor:
    """The chunk's first ``n`` output rows [n, 4096] fp32: slab = the rows' shards concatenated, v1 = row 0's
    (every row holds the whole chunk). The columns of a row must agree (the all-reduced o_proj)."""
    grid = _shards(mesh, y)
    for r in range(mesh.rows):
        for c in range(1, mesh.cols):
            assert torch.equal(grid[r][c], grid[r][0]), (r, c, "columns differ")
    if slab:
        out = torch.cat([grid[r][0] for r in range(mesh.rows)], dim=-2)
    else:
        for r in range(1, mesh.rows):
            assert torch.equal(grid[r][0], grid[0][0]), (r, "v1 rows differ")
        out = grid[0][0]
    return out[0, 0, :n].float()


class IndexRecorder:
    """``TtDSA.on_indices`` sink on a mesh: the valid-id prefix of every sparse query row by absolute position.
    v1 hands ``(query_start, indices)`` with every chip holding the same rows (device 0 is read); the slab path
    hands ``(row_starts, indices)`` with mesh row r's chip holding its slab (positions below the dense boundary
    are the superset's clamped rows and are skipped)."""

    def __init__(self, mesh: GLMMesh, max_pos: int):
        self.mesh, self.max_pos = mesh, max_pos
        self.rows: dict[int, torch.Tensor] = {}

    def __call__(self, query_start, indices: ttnn.Tensor):
        devs = self.mesh.device_tensors(indices)
        starts = query_start if isinstance(query_start, tuple) else (query_start,)
        for r, s in enumerate(starts):
            t = ttnn.to_torch(devs[r * self.mesh.cols]).reshape(-1, indices.shape[-1]).to(torch.int64) & SENTINEL
            for i in range(t.shape[0]):
                pos = s + i
                if pos < DENSE_ROWS or pos >= self.max_pos:
                    continue
                row = t[i]
                sent = (row == SENTINEL).nonzero()
                first = int(sent[0]) if sent.numel() else row.numel()
                assert bool((row[first:] == SENTINEL).all()), f"row {pos}: a sentinel inside the valid prefix"
                self.rows[pos] = row[:first].sort().values


def attn_prefill(attn, cache, mesh, rows_t, user, chunks, start0, slab, qblock, env, timer=None, record=True):
    """Prefill rows [start0, start0 + sum(chunks)) into slot ``user`` chunk by chunk (slab-local or v1).
    Returns (output [n, 4096] fp32, IndexRecorder, per-chunk wall seconds, per-chunk stage groups). ``record``:
    install the index recorder (its per-sub-block host readback sits inside the timed window: off for timing)."""
    env.setenv("GLM53_PREFILL_QBLOCK", str(qblock))
    rec = IndexRecorder(mesh, start0 + sum(chunks))
    attn.on_indices = rec if record else None
    attn.set_stage_timer(timer)
    outs, walls, stages, start = [], [], [], start0
    for n in chunks:
        if timer is not None:
            timer.reset()
        x = _chunk_to_device(mesh, rows_t, start - start0, n, slab)  # rows_t holds the rows from start0 on
        ttnn.synchronize_device(mesh.device)
        t0 = time.perf_counter()
        y = attn.prefill_forward(x, cache, user, n, start=start, slab=slab)
        ttnn.synchronize_device(mesh.device)
        walls.append(time.perf_counter() - t0)
        if timer is not None:
            stages.append({g: sum(timer.times.get(k, 0.0) for k in ks) for g, ks in STAGE_GROUPS.items()})
            for extra in ("slab_gather", "a2a_heads_to_seq", "a2a_seq_to_heads"):
                stages[-1][extra] = timer.times.get(extra, 0.0)
        outs.append(_rows_out(mesh, y, n, slab))
        ttnn.deallocate(y)
        ttnn.deallocate(x)
        start += n
    attn.on_indices = None
    attn.set_stage_timer(None)
    return torch.cat(outs, dim=0), rec, walls, stages


def _snapshot(cache: TtLatentCache, mesh: GLMMesh, user: int, n_pos: int):
    """(latent rows, complete pools, ring k/g rows) of ``user`` on every chip (row-major chip order)."""
    P = cache.index.P
    n_pools, tail = n_pos // P, n_pos % P
    lat_t = cache.user_rows(user, n_pos)
    out = []
    for chip in range(mesh.rows * mesh.cols):
        lat = ttnn.to_torch(mesh.device_tensors(lat_t)[chip])[0, 0, :n_pos].float()
        pools = ttnn.to_torch(mesh.device_tensors(cache.index.pools_batched())[chip])[user, 0, :n_pools].float()
        rk = ttnn.to_torch(mesh.device_tensors(cache.index.ring_k)[chip])[user, 0, :tail].float()
        rg = ttnn.to_torch(mesh.device_tensors(cache.index.ring_g)[chip])[user, 0, :tail].float()
        out.append((lat, pools, rk, rg))
    ttnn.deallocate(lat_t)
    return out


def compare_state(cache, mesh, n_pos: int, u_ref: int = 0, u_got: int = 1) -> tuple[str, bool]:
    """Slot ``u_got`` (slab) vs slot ``u_ref`` (v1) on chip 0, and slot ``u_got`` equal on every chip."""
    a, b = _snapshot(cache, mesh, u_ref, n_pos), _snapshot(cache, mesh, u_got, n_pos)
    names = ("latent", "pools", "ring_k", "ring_g")
    parts, ok = [], True
    for name, x, y in zip(names, a[0], b[0]):
        if x.numel() == 0:
            parts.append(f"{name} (empty)")
            continue
        eq = torch.equal(x, y)
        ok &= eq
        if eq:
            parts.append(f"{name} {tuple(x.shape)} equal")
        else:
            d = (x - y).abs()
            parts.append(
                f"{name} {tuple(x.shape)} DIFFER: {int((d > 0).sum())} of {d.numel()} elements, max|diff| {d.max():.3g} "
                f"(|ref| max {x.abs().max():.3g}), pcc {pcc(x, y):.7f}"
            )
    chips_eq = all(torch.equal(x, y) for chip in b[1:] for x, y in zip(b[0], chip))
    ok &= chips_eq
    parts.append("slab slot identical on all chips" if chips_eq else "slab slot DIFFERS between chips")
    return "; ".join(parts), ok


def class_report(ref: torch.Tensor, got: torch.Tensor, start0: int, slabs: list[tuple[int, int, int]]) -> str:
    """Per (chunk, mesh row, class) minimum row PCC and bit-exact counts; ``slabs`` = (row index in ``ref``,
    row count, mesh row) per slab in order. Returns the text; raises nothing (the caller asserts)."""
    r = row_pcc(ref, got)
    exact = (ref == got).all(dim=1)
    lines = []
    for i0, n, mesh_row in slabs:
        pos = torch.arange(i0, i0 + n) + start0
        for name, m in (("dense", pos < DENSE_ROWS), ("sparse", pos >= DENSE_ROWS)):
            if not bool(m.any()):
                continue
            rr, ee = r[i0 : i0 + n][m], exact[i0 : i0 + n][m]
            lines.append(
                f"rows [{int(pos[m][0])}, {int(pos[m][-1]) + 1}) mesh row {mesh_row} {name}: pcc min {rr.min():.7f} "
                f"median {rr.median():.7f}, bit-exact {int(ee.sum())}/{int(m.sum())}"
            )
    return "\n      ".join(lines)


def _slabs_of(chunks, mesh_rows: int) -> list[tuple[int, int, int]]:
    """(first row index, rows, mesh row) of every slab of every chunk (the padded tail rows are excluded)."""
    out, i0 = [], 0
    for n in chunks:
        unit = TILE * mesh_rows
        T_loc = (-(-n // unit) * unit) // mesh_rows
        for r in range(mesh_rows):
            lo, hi = r * T_loc, min(n, (r + 1) * T_loc)
            if hi > lo:
                out.append((i0 + lo, hi - lo, r))
        i0 += n
    return out


# ------------------------------------------------------------------------------------- primitives
@pytest.mark.timeout(900)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_slab_primitives(mesh_device):
    mesh = GLMMesh.from_device(mesh_device)
    rows = mesh.rows
    torch.manual_seed(0)
    # (1) the dense/sparse select with a row-sharded [n, 1] mask
    n_d, N = 64, 256
    a, b = torch.randn(1, 1, n_d, N), torch.randn(1, 1, n_d, N)
    m = torch.zeros(rows, 1, n_d, 1)
    for r in range(rows):
        m[r, 0, : 32 * (r + 1)] = 1.0
    a_t = mesh.upload(a, ttnn.bfloat16)
    b_t = mesh.upload(b, ttnn.bfloat16)
    sel = ttnn.where(mesh.upload(m, ttnn.bfloat16, rows=0), a_t, b_t)
    grid = _shards(mesh, sel)
    for r in range(rows):
        want = torch.where(m[r] > 0, a.bfloat16().float(), b.bfloat16().float())
        assert all(torch.equal(grid[r][c].float(), want) for c in range(mesh.cols)), f"where select row {r}"
    logger.info("where with a row-sharded [n,1] mask: exact on every chip")
    # (2) the windowed SDPA with a row-sharded Q offset tensor == the plain causal kernel's slab rows
    H, D, S = 16, 256, 512
    T_loc = S // rows
    q, k, v = torch.randn(1, H, S, D), torch.randn(1, H, S, D), torch.randn(1, H, S, D)
    q_t, k_t, v_t = mesh.upload(q, ttnn.bfloat16, rows=2), mesh.upload(k, ttnn.bfloat16), mesh.upload(v, ttnn.bfloat16)
    cu = mesh.upload(torch.tensor([0, S], dtype=torch.int32), ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    off = mesh.upload(
        torch.tensor([[r * T_loc] for r in range(rows)], dtype=torch.int32),
        ttnn.int32,
        rows=0,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    grid_size = mesh_device.compute_with_storage_grid_size()
    cfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_size, q_chunk_size=128, k_chunk_size=128, exp_approx_mode=False
    )
    ck = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    o_t = ttnn.transformer.scaled_dot_product_attention(
        q_t,
        k_t,
        v_t,
        is_causal=True,
        scale=1 / 16,
        program_config=cfg,
        compute_kernel_config=ck,
        cu_window_seqlens=cu,
        windowed_q_token_offset=0,
        windowed_q_token_offset_tensor=off,
    )
    o_ref = ttnn.transformer.scaled_dot_product_attention(
        mesh.upload(q, ttnn.bfloat16),
        k_t,
        v_t,
        is_causal=True,
        scale=1 / 16,
        program_config=cfg,
        compute_kernel_config=ck,
    )
    ref = _shards(mesh, o_ref)[0][0].float()
    got = _shards(mesh, o_t)
    for r in range(rows):
        want = ref[:, :, r * T_loc : (r + 1) * T_loc]
        for c in range(mesh.cols):
            g = got[r][c].float()
            p = pcc(want, g)
            logger.info(
                f"windowed SDPA mesh row {r} col {c} (Q offset {r * T_loc}): bit-equal {torch.equal(g, want)} pcc {p:.8f} "
                f"max|diff| {(g - want).abs().max():.3g}"
            )
            assert torch.equal(g, want), (r, c)
    # (3) the gathers of the narrow latent / index stems
    for width in (512, 128):
        xs = torch.randn(1, 1, 64 * rows, width)
        g = mesh.all_gather_rows(mesh.upload(xs, ttnn.bfloat16, rows=2), dim=2)
        grid = _shards(mesh, g)
        assert all(torch.equal(grid[r][c], xs.bfloat16()) for r in range(rows) for c in range(mesh.cols)), width
        logger.info(f"all_gather_rows of [1,1,{64 * rows},{width}] bf16: exact on every chip")


# ------------------------------------------------------------------------------- attention, R4097
def _run_variant(attn, cache, mesh, rows_t, chunks, start0, qblock, env, timer=None, record=True):
    """v1 into slot 0, slab-local into slot 1 (same rows, same chunking); returns the comparison lines."""
    ref, ref_idx, w_ref, _ = attn_prefill(
        attn, cache, mesh, rows_t, 0, chunks, start0, False, qblock, env, None, record
    )
    got, got_idx, w_got, st = attn_prefill(
        attn, cache, mesh, rows_t, 1, chunks, start0, True, qblock, env, timer, record
    )
    r = row_pcc(ref, got)
    below = int((r < ROW_PCC_BAR).sum())
    exact = int((ref == got).all(dim=1).sum())
    itxt, n_sparse, n_diff = (
        compare_indices(ref_idx.rows, got_idx.rows) if ref_idx.rows else ("no sparse rows recorded", 0, 0)
    )
    state, state_ok = compare_state(cache, mesh, start0 + sum(chunks))
    head = (
        f"chunks {list(chunks)} from {start0} x sub-block {qblock}: rows {ref.shape[0]}: pcc min {r.min():.7f} "
        f"(row {int(r.argmin()) + start0}) median {r.median():.7f}; rows < {ROW_PCC_BAR} {below}; bit-exact rows {exact}; "
        f"max|diff| {(ref - got).abs().max():.3g}; wall v1 {sum(w_ref) * 1e3:.0f} ms slab {sum(w_got) * 1e3:.0f} ms"
    )
    classes = class_report(ref, got, start0, _slabs_of(chunks, mesh.rows))
    logger.info(head + "\n      " + classes)
    logger.info(f"    selections: {itxt}")
    logger.info(f"    state slot 0 (v1) vs slot 1 (slab): {state}")
    failures = []
    if below:
        failures.append(f"{list(chunks)} from {start0}: {below} rows below {ROW_PCC_BAR} (min {r.min():.7f})")
    if n_diff:
        failures.append(f"{list(chunks)} from {start0}: {n_diff} of {n_sparse} sparse rows select differently")
    if not state_ok:
        failures.append(f"{list(chunks)} from {start0}: state differs: {state}")
    return failures, (w_ref, w_got, st)


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_dsa_attention_slab_r4097(mesh_device, monkeypatch):
    raw, cfg = _load(attn_only=True)
    x_in = torch.load(R4097 / f"layer{LAYER}_prefill.pt")["x_in"]
    S = x_in.shape[1]
    rows_t = attention_input(raw, cfg, x_in)  # [S, 4096] bf16
    attn, cache, mesh = _attention(mesh_device, raw, cfg, max_batch=2, max_seq_len=S + 64 * mesh_device.shape[0])
    failures = []
    # single shot (one mixed chunk: slab 0 = 2048 dense + 32 sparse rows, slab 1 all sparse on 2 rows),
    # 1024-chunks (all-dense, all-dense, all-sparse, all-sparse, a 1-token tail), 2048 + 2048 + 1
    for chunks, qb in (((S,), 2048), ((1024, 1024, 1024, 1024, 1), 512), ((2048, 2048, 1), 2048)):
        f, _ = _run_variant(attn, cache, mesh, rows_t, chunks, 0, qb, monkeypatch)
        failures += f
    # a chunk that starts inside the dense region and ends in the sparse one (start 1024, 2048 rows)
    f, _ = _run_variant(attn, cache, mesh, rows_t, (1024, 2048, 1025), 0, 2048, monkeypatch)
    failures += f
    assert not failures, "\n".join(failures)


@pytest.mark.timeout(900)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
@pytest.mark.parametrize("S", [128, 1024])
def test_dsa_attention_slab_dense(mesh_device, monkeypatch, S):
    """All-dense chunks (the TTFT shapes: S tokens, S/rows per slab), equality + warm walls v1 vs slab + stages."""
    raw, cfg = _load(attn_only=True)
    x_in = torch.load(R4097 / f"layer{LAYER}_prefill.pt")["x_in"][:, :S]
    rows_t = attention_input(raw, cfg, x_in)
    attn, cache, mesh = _attention(mesh_device, raw, cfg, max_batch=2, max_seq_len=S + 64 * mesh_device.shape[0])
    timer = StageTimer(mesh_device)
    failures, _ = _run_variant(attn, cache, mesh, rows_t, (S,), 0, 2048, monkeypatch)
    walls = {"v1": [], "slab": []}
    for it in range(3):
        for name, slab in (("v1", False), ("slab", True)):
            _, _, w, _ = attn_prefill(
                attn, cache, mesh, rows_t, int(slab), (S,), 0, slab, 2048, monkeypatch, None, False
            )
            walls[name].append(w[0])
    _, _, _, st = attn_prefill(attn, cache, mesh, rows_t, 1, (S,), 0, True, 2048, monkeypatch, timer, False)
    stg = "  ".join(f"{g} {v * 1e3:.2f}" for g, v in st[0].items() if v)
    logger.info(
        f"TIMING dense chunk S={S} on {MESH}: v1 warm min {min(walls['v1']) * 1e3:.1f} ms (all {[f'{w * 1e3:.1f}' for w in walls['v1']]}) | "
        f"slab warm min {min(walls['slab']) * 1e3:.1f} ms (all {[f'{w * 1e3:.1f}' for w in walls['slab']]}); slab stages ms: {stg}"
    )
    assert not failures, "\n".join(failures)


# ----------------------------------------------------------------------------------- layer, R4097
@pytest.mark.timeout(1500)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_dsa_layer_slab_r4097(mesh_device, monkeypatch):
    raw, cfg = _load(attn_only=False)
    pre = torch.load(R4097 / f"layer{LAYER}_prefill.pt")
    x_in, x_out = pre["x_in"], pre["x_out"]
    S = x_in.shape[1]
    rows = mesh_device.shape[0]
    dec = FunctionalDecoder.from_state_dict(
        raw, hf_config=cfg, layer_idx=LAYER, mesh_device=mesh_device, max_batch=2, max_seq_len=S + 64 * rows
    )
    mesh = dec.mesh

    def layer_prefill(user: int, mode: str, chunks):
        monkeypatch.setenv("GLM53_SP_ATTN", mode)
        rec = IndexRecorder(mesh, S)
        dec.layer.attn.on_indices = rec
        outs, start = [], 0
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        for n in chunks:
            y = dec.prefill_forward(
                host_streams_to_device(x_in[:, start : start + n], mesh_device, mesh=mesh),
                user=user,
                seq_len=n,
                start=start,
            )
            outs.append(device_streams_to_host(y, 4, n, mesh=mesh))
            ttnn.deallocate(y)
            start += n
        ttnn.synchronize_device(mesh_device)
        dt = time.perf_counter() - t0
        dec.layer.attn.on_indices = None
        return torch.cat(outs, dim=1), rec, dt

    failures = []
    r2 = lambda t: t[0].reshape(S, -1)
    for chunks in ((S,), (2048, 2048, 1)):
        ref, ref_idx, t_ref = layer_prefill(0, "v1", chunks)
        got, got_idx, t_got = layer_prefill(1, "slab", chunks)
        p_ref, p_got = pcc(x_out, ref), pcc(x_out, got)
        r = row_pcc(r2(ref), r2(got))
        below = int((r < ROW_PCC_BAR).sum())
        itxt, n_sparse, n_diff = compare_indices(ref_idx.rows, got_idx.rows)
        state, state_ok = compare_state(dec.cache, mesh, S)
        logger.info(
            f"layer {LAYER} chunks {list(chunks)}: pcc vs HF v1 {p_ref:.6f} slab {p_got:.6f}; slab vs v1 rows: pcc min {r.min():.7f} "
            f"median {r.median():.7f}, rows < {ROW_PCC_BAR} {below}, bit-exact {int((r2(ref) == r2(got)).all(dim=1).sum())}; "
            f"wall v1 {t_ref:.2f} s slab {t_got:.2f} s (first calls compile)\n      "
            + class_report(r2(ref), r2(got), 0, _slabs_of(chunks, rows))
        )
        logger.info(f"    selections: {itxt}")
        logger.info(f"    state slot 0 (v1) vs slot 1 (slab): {state}")
        if p_got < HF_PCC_BAR or below or n_diff or not state_ok:
            failures.append(
                f"chunks {list(chunks)}: hf {p_got:.6f}, rows below bar {below}, differing selections {n_diff}, state ok {state_ok}"
            )
    # decode on both slots at the dump's positions (the same token for both; rows replicate the slot)
    for step in range(2):
        d = torch.load(R4097 / f"layer{LAYER}_decode{step}.pt")
        pos = d["position"]
        U = dec.total_users
        dp = dec.make_decode_positions([pos] * U)
        xin = d["x_in"].repeat(1, U, 1, 1)
        yd = device_streams_to_host(
            dec.decode_forward(host_streams_to_device(xin, mesh_device, pad_to=U, mesh=mesh), positions=dp),
            4,
            U,
            mesh=mesh,
        )
        ps = [pcc(d["x_out"], yd[:, u : u + 1]) for u in range(U)]
        p01 = min(pcc(yd[:, r * 2 : r * 2 + 1], yd[:, r * 2 + 1 : r * 2 + 2]) for r in range(rows))
        logger.info(
            f"  decode pos {pos}: pcc vs HF per (row, slot) {['%.6f' % p for p in ps]}, slot 0 vs 1 min {p01:.6f}"
        )
        assert min(ps) > HF_PCC_BAR and p01 > 0.9999
    assert not failures, "\n".join(failures)


# ------------------------------------------------------------------------------ attention, long
@pytest.mark.timeout(1500)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
@pytest.mark.parametrize("n_pos", LONG_N)
def test_dsa_attention_slab_long(mesh_device, monkeypatch, n_pos):
    raw, cfg = _load(attn_only=True)
    L = 16384
    x_in = torch.load(R16385 / f"layer{LAYER}_prefill.pt")["x_in"][:, :L]
    rows_t = attention_input(raw, cfg, x_in)  # [L, 4096]
    start0 = n_pos - L
    attn, cache, mesh = _attention(mesh_device, raw, cfg, max_batch=2, max_seq_len=n_pos + 64 * mesh_device.shape[0])
    if start0:
        mu, sigma = _x_stats(raw, cfg)
        filled, ci = 0, 0
        t0 = time.perf_counter()
        while filled < start0:
            T = min(SEED_CHUNK, start0 - filled)
            x = synthetic_rows(mu, sigma, ci, T)
            x_tt = mesh.upload(x.reshape(1, 1, T, -1), ttnn.bfloat16)
            for user in (0, 1):
                attn.prefill_state(x_tt, cache, user, T, start=filled)
            ttnn.deallocate(x_tt)
            filled += T
            ci += 1
        ttnn.synchronize_device(mesh_device)
        logger.info(f"seeded [0, {start0}) synthetically in both slots ({ci} chunks, {time.perf_counter() - t0:.1f} s)")
    timer = StageTimer(mesh_device)
    failures = []
    a2a_modes = [m for m in os.environ.get("GLM53_SLAB_A2A_MODES", "1,0").split(",") if m]
    for chunks, qb in (((L,), 2048), ((8192, 8192), 2048)):
        for a2a in a2a_modes:  # the heads -> sequence exchange of the sparse attention on / off (GLM53_SPARSE_A2A)
            monkeypatch.setenv("GLM53_SPARSE_A2A", a2a)
            f, (w_ref, w_got, _) = _run_variant(attn, cache, mesh, rows_t, chunks, start0, qb, monkeypatch)
            failures += [f"a2a={a2a}: {x}" for x in f]
            # warmed walls: the same chunking again into the same slots (identical state), no index recorder (its
            # host readback per sub-block would sit inside the timed window) and no stage timer (its per-mark mesh
            # synchronizations would inflate the slab pass only); then a slab-only pass with the stage timer
            f2, (w_ref2, w_got2, _) = _run_variant(
                attn, cache, mesh, rows_t, chunks, start0, qb, monkeypatch, record=False
            )
            failures += [f"a2a={a2a}: {x}" for x in f2]
            _, _, _, st2 = attn_prefill(
                attn, cache, mesh, rows_t, 1, chunks, start0, True, qb, monkeypatch, timer, False
            )
            for i, n in enumerate(chunks):
                s0 = start0 + sum(chunks[:i])
                kind = "dense" if s0 + n <= DENSE_ROWS else ("sparse" if s0 >= DENSE_ROWS else "mixed")
                stg = "  ".join(f"{g} {v * 1e3:.1f}" for g, v in st2[i].items() if v)
                logger.info(
                    f"TIMING N={n_pos} chunk [{s0}, {s0 + n}) ({kind}, sub-block {qb}, sparse a2a {a2a}) on {MESH}: v1 cold "
                    f"{w_ref[i] * 1e3:.0f} warm {w_ref2[i] * 1e3:.0f} ms | slab cold {w_got[i] * 1e3:.0f} warm "
                    f"{w_got2[i] * 1e3:.0f} ms ({n / w_got2[i]:.0f} tok/s) -> x{w_ref2[i] / w_got2[i]:.2f}; "
                    f"slab stages ms (separate timed pass, stage-synchronized): {stg}"
                )
    assert not failures, "\n".join(failures)
