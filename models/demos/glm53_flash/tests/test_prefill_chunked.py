# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Chunked prefill == single-shot prefill (the 1 M-token mesh run's precondition), one chip, real weights.

Every test prefills the same prompt twice into two user slots of one state: single-shot into slot 0,
chunked (chunk length x ``GLM53_PREFILL_QBLOCK`` query sub-block, several sub-blocks per chunk) into slot 1,
then compares, row by row, the output, the sparse rows' top-k selections (the index blocks the DSA prefill
hands to ``sparse_sdpa``, through ``TtDSA.on_indices``), the latent cache pages, the pooled-key index and
the ring tail (DSA) or the recurrent + conv carries (KDA) of the two slots.

* ``test_dsa_attention_r4097``: layer-3 attention (``TtDSA.prefill_forward`` on the normalized attention
  inputs of the non-repetitive 4097-token dump ``R4097``): chunks 1024 / 2048 x sub-blocks 512 / 2048.
* ``test_dsa_layer_r4097``: the whole layer 3 (``FunctionalDecoder``, mHC + attention + MoE), the same
  chunkings, vs the HF dump too, then 2 decode steps on both slots (positions 4097, 4098) vs the dump.
* ``test_dsa_attention_long[16384|65536]``: 16384 real rows (dump ``R16385``) single-shot (one 16384 chunk,
  7-8 sub-blocks) vs 8192-token chunks x sub-blocks 512 / 2048; at 65536 both slots first get the same
  synthetic 49152-position prefix through ``TtDSA.prefill_state`` (the ``test_cache_1m`` seeding).
* ``test_kda_layer_r4097``: layer 0 (KDA + dense MLP) in chunks 1024 / 2048 / 4096 (+ the 1-token tail).
* ``test_dsa_prefill_chunk_cost``: one DSA layer chunk of 16384 tokens at 16 K, 64 K, 256 K and 1 M
  positions (prefix seeded synthetically): wall time per stage (``StageTimer``: whole-prefix copy, scorer,
  top-k, sparse_sdpa, ...), tokens/s, and the 1 M-prompt projection; results to ``GLM53_LONGCTX_RESULTS``.

    TT_VISIBLE_DEVICES=0 pytest models/demos/glm53_flash/tests/test_prefill_chunked.py -s --timeout 1500

Knobs: GLM53_LONGCTX_N (cost curve, default 16384,65536,262144,1048576), GLM53_LONGCTX_CHUNK (16384),
GLM53_LONGCTX_QBLOCK (2048), GLM53_LONGCTX_FULL_LAYER=1 (the cost test on the whole layer instead of the
attention only), GLM53_LONGCTX_KDA_LAYER (0), GLM53_LONGCTX_RESULTS (a JSON lines file for the doc).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir, load_layer_raw, shard_headers
from models.demos.glm53_flash.tests.test_cache_1m import _attn_keys, _x_stats, synthetic_rows
from models.demos.glm53_flash.tests.utils import attention_input, log_dram, pcc
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.dsa import StageTimer, TtDSA, TtLatentCache
from models.demos.glm53_flash.tt.functional_decoder import (
    FunctionalDecoder,
    device_streams_to_host,
    host_streams_to_device,
)

TILE = 32
LAYER_DSA = 3
LAYER_KDA = int(os.environ.get("GLM53_LONGCTX_KDA_LAYER", "0"))
REF = Path("bringup/references/glm53_flash")
R4097, R16385 = REF / "R4097", REF / "R16385"
ROW_PCC_BAR = 0.9999  # every row of the chunked output vs the single-shot one
HF_PCC_BAR = 0.995
SENTINEL = 0xFFFFFFFF
DENSE_ROWS = 2048
CURVE = [int(v) for v in os.environ.get("GLM53_LONGCTX_N", "16384,65536,262144,1048576").split(",")]
COST_CHUNK = int(os.environ.get("GLM53_LONGCTX_CHUNK", "16384"))
COST_QBLOCK = int(os.environ.get("GLM53_LONGCTX_QBLOCK", "2048"))
SEED_CHUNK = 32768
RESULTS = os.environ.get("GLM53_LONGCTX_RESULTS")


# ------------------------------------------------------------------------------------------ helpers
def _load(layer_idx: int, attn_only: bool = False):
    snapshot = default_snapshot_dir()
    raw = load_layer_raw(snapshot, layer_idx, shard_headers(snapshot), name_filter=_attn_keys if attn_only else None)
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    return raw, cfg


def _attention(device, raw, cfg, max_batch: int, max_seq_len: int) -> tuple[TtDSA, TtLatentCache]:
    attn = TtDSA(
        device,
        cfg,
        {k[len("self_attn.") :]: v for k, v in raw.items() if k.startswith("self_attn.")},
        layer_idx=LAYER_DSA,
    )
    return attn, TtLatentCache(device, cfg, max_batch=max_batch, max_seq_len=max_seq_len)


def _rows_to_device(device, rows: torch.Tensor, start: int, n: int) -> ttnn.Tensor:
    """rows [S, 4096] bf16 -> [1,1,T_pad,4096] bf16 TILE of rows [start, start + n) (zero padding)."""
    T = -(-n // TILE) * TILE
    x = torch.zeros(1, 1, T, rows.shape[-1], dtype=torch.bfloat16)
    x[0, 0, :n] = rows[start : start + n]
    return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


class IndexRecorder:
    """``TtDSA.on_indices`` sink: the valid-id prefix of every sparse query row by absolute position."""

    def __init__(self, max_pos: int):
        self.max_pos = max_pos
        self.rows: dict[int, torch.Tensor] = {}

    def __call__(self, query_start: int, indices: ttnn.Tensor):
        t = ttnn.to_torch(indices).reshape(indices.shape[-2], indices.shape[-1]).to(torch.int64) & SENTINEL
        for i in range(t.shape[0]):
            pos = query_start + i
            if pos >= self.max_pos:
                break  # padding rows
            row = t[i]
            sent = (row == SENTINEL).nonzero()
            first = int(sent[0]) if sent.numel() else row.numel()
            assert bool((row[first:] == SENTINEL).all()), f"row {pos}: a sentinel inside the valid prefix"
            self.rows[pos] = row[:first].sort().values


def row_pcc(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-row PCC of two [S, D] tensors."""
    a = a.float() - a.float().mean(1, keepdim=True)
    b = b.float() - b.float().mean(1, keepdim=True)
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-30)


def row_report(ref: torch.Tensor, got: torch.Tensor, bar: float = ROW_PCC_BAR) -> tuple[str, float, int]:
    """(text, min row pcc, rows below bar) for [S, D] outputs."""
    r = row_pcc(ref, got)
    below = int((r < bar).sum())
    exact = int((ref.float() == got.float()).all(dim=1).sum())
    q = torch.quantile(r, torch.tensor([0.0, 0.01, 0.5]))
    worst = int(r.argmin())
    txt = (
        f"rows {ref.shape[0]}: pcc min {q[0]:.6f} (row {worst}) p1 {q[1]:.6f} median {q[2]:.6f}; "
        f"rows < {bar} {below}; bit-exact rows {exact}; max|diff| {(ref.float() - got.float()).abs().max():.3g}"
    )
    return txt, float(q[0]), below


def compare_indices(ref: dict[int, torch.Tensor], got: dict[int, torch.Tensor]) -> tuple[str, int, int]:
    """(text, rows compared, rows whose selected-id sets differ) of two IndexRecorder row maps."""
    common = sorted(set(ref) & set(got))
    assert common and set(ref) == set(got), (len(ref), len(got), len(common))
    diff_rows, worst = 0, 0
    worst_pos = None
    for p in common:
        a, b = ref[p], got[p]
        if a.numel() != b.numel() or not torch.equal(a, b):
            diff_rows += 1
            sa, sb = set(a.tolist()), set(b.tolist())
            d = len(sa ^ sb)
            if d > worst:
                worst, worst_pos = d, p
    txt = f"sparse rows {len(common)} [{common[0]}, {common[-1]}]: identical selections {len(common) - diff_rows}, differing {diff_rows}"
    if diff_rows:
        txt += f" (largest symmetric difference {worst} ids at position {worst_pos})"
    return txt, len(common), diff_rows


def snapshot_dsa(cache: TtLatentCache, user: int, n_pos: int):
    """(latent rows [n_pos, 512], complete pools [n_pos // P, D], ring k/g rows [tail, D]) of ``user``."""
    lat = ttnn.to_torch(cache.user_rows(user, n_pos))[0, 0, :n_pos].float()
    P = cache.index.P
    n_pools, tail = n_pos // P, n_pos % P
    pools = ttnn.to_torch(cache.index.pools_batched())[user, 0, :n_pools].float()
    rk = ttnn.to_torch(cache.index.ring_k)[user, 0, :tail].float()
    rg = ttnn.to_torch(cache.index.ring_g)[user, 0, :tail].float()
    return lat, pools, rk, rg


def compare_dsa_state(cache: TtLatentCache, n_pos: int, u_ref: int = 0, u_got: int = 1) -> tuple[str, bool]:
    """(text, all equal) for the latent rows, complete pools and ring tail of the two slots."""
    a, b = snapshot_dsa(cache, u_ref, n_pos), snapshot_dsa(cache, u_got, n_pos)
    names = ("latent", "pools", "ring_k", "ring_g")
    parts, ok = [], True
    for name, x, y in zip(names, a, b):
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
    return "; ".join(parts), ok


def attn_prefill(attn: TtDSA, cache: TtLatentCache, device, rows, user: int, chunks, start0: int, qblock: int, env):
    """Prefill rows [start0, start0 + sum(chunks)) of ``rows`` into slot ``user`` chunk by chunk under
    GLM53_PREFILL_QBLOCK=qblock. Returns (output [n, 4096] fp32, IndexRecorder, seconds)."""
    env.setenv("GLM53_PREFILL_QBLOCK", str(qblock))
    n_total = sum(chunks)
    rec = IndexRecorder(start0 + n_total)
    attn.on_indices = rec
    outs, start = [], start0
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for n in chunks:
        x = _rows_to_device(device, rows, start, n)
        y = attn.prefill_forward(x, cache, user, n, start=start)
        outs.append(ttnn.to_torch(y)[0, 0, :n].float())
        ttnn.deallocate(y)
        ttnn.deallocate(x)
        start += n
    ttnn.synchronize_device(device)
    dt = time.perf_counter() - t0
    attn.on_indices = None
    return torch.cat(outs, dim=0), rec, dt


def layer_prefill(dec: FunctionalDecoder, device, x_in: torch.Tensor, user: int, chunks, qblock: int, env):
    """x_in [1,S,4,4096] through ``prefill_forward`` in chunks -> (output [1,S,4,4096], IndexRecorder, s)."""
    env.setenv("GLM53_PREFILL_QBLOCK", str(qblock))
    S = x_in.shape[1]
    assert sum(chunks) == S, (chunks, S)
    rec = IndexRecorder(S)
    attn = dec.layer.attn
    if hasattr(attn, "on_indices"):
        attn.on_indices = rec
    outs, start = [], 0
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for n in chunks:
        y = dec.prefill_forward(
            host_streams_to_device(x_in[:, start : start + n], device), user=user, seq_len=n, start=start
        )
        outs.append(device_streams_to_host(y, 4, n))
        ttnn.deallocate(y)
        start += n
    ttnn.synchronize_device(device)
    dt = time.perf_counter() - t0
    if hasattr(attn, "on_indices"):
        attn.on_indices = None
    return torch.cat(outs, dim=1), rec, dt


def _chunking(S: int, chunk: int) -> tuple[int, ...]:
    out = []
    while sum(out) < S:
        out.append(min(chunk, S - sum(out)))
    return tuple(out)


def _record(kind: str, **kw):
    if RESULTS:
        with open(RESULTS, "a") as f:
            f.write(json.dumps({"kind": kind, **kw}) + "\n")


# ------------------------------------------------------------------------------- DSA attention
DSA_VARIANTS = [(1024, 512), (1024, 2048), (2048, 512), (2048, 2048)]


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_dsa_attention_r4097(mesh_device, monkeypatch):
    raw, cfg = _load(LAYER_DSA, attn_only=True)
    x_in = torch.load(R4097 / f"layer{LAYER_DSA}_prefill.pt")["x_in"]
    S = x_in.shape[1]
    rows = attention_input(raw, cfg, x_in)  # [S, 4096] bf16 (HF attn_hc collapse + input_layernorm)
    attn, cache = _attention(mesh_device, raw, cfg, max_batch=2, max_seq_len=S + 64)
    ref, ref_idx, t_ref = attn_prefill(attn, cache, mesh_device, rows, 0, (S,), 0, 2048, monkeypatch)
    logger.info(f"layer 3 attention single-shot S={S} (slot 0, sub-block 2048): {t_ref:.2f} s (compile included)")
    results = []
    for chunk, qb in DSA_VARIANTS:
        chunks = _chunking(S, chunk)
        got, got_idx, dt = attn_prefill(attn, cache, mesh_device, rows, 1, chunks, 0, qb, monkeypatch)
        txt, pmin, below = row_report(ref, got)
        itxt, n_sparse, n_diff = compare_indices(ref_idx.rows, got_idx.rows)
        state, state_ok = compare_dsa_state(cache, S)
        logger.info(f"chunks {list(chunks)} x sub-block {qb} (slot 1, {dt:.2f} s): output {txt}")
        logger.info(f"    selections: {itxt}")
        logger.info(f"    state slot 0 vs slot 1: {state}")
        results.append((chunk, qb, pmin, below, n_diff, n_sparse, state_ok, state))
        _record(
            "dsa_attention_r4097",
            chunk=chunk,
            qblock=qb,
            row_pcc_min=pmin,
            rows_below=below,
            sel_diff=n_diff,
            sel_rows=n_sparse,
            state_ok=state_ok,
        )
    failures = []
    for chunk, qb, pmin, below, n_diff, n_sparse, state_ok, state in results:
        if below:
            failures.append(f"chunk {chunk} sub-block {qb}: {below} rows below {ROW_PCC_BAR} (min {pmin:.6f})")
        if n_diff:
            failures.append(f"chunk {chunk} sub-block {qb}: {n_diff} of {n_sparse} sparse rows select differently")
        if not state_ok:
            failures.append(f"chunk {chunk} sub-block {qb}: state differs: {state}")
    assert not failures, "\n".join(failures)


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_dsa_layer_r4097(mesh_device, monkeypatch):
    raw, cfg = _load(LAYER_DSA)
    pre = torch.load(R4097 / f"layer{LAYER_DSA}_prefill.pt")
    x_in, x_out = pre["x_in"], pre["x_out"]
    S = x_in.shape[1]
    dec = FunctionalDecoder.from_state_dict(
        raw, hf_config=cfg, layer_idx=LAYER_DSA, mesh_device=mesh_device, max_batch=2, max_seq_len=S + 64
    )
    ref, ref_idx, t_ref = layer_prefill(dec, mesh_device, x_in, 0, (S,), 2048, monkeypatch)
    p_hf = pcc(x_out, ref)
    logger.info(f"layer 3 single-shot S={S} (slot 0): pcc vs HF {p_hf:.6f}, {t_ref:.1f} s (compile included)")
    assert p_hf > HF_PCC_BAR
    r2 = lambda t: t[0].reshape(S, -1)
    failures = []
    for chunk, qb in ((1024, 512), (2048, 2048)):
        chunks = _chunking(S, chunk)
        got, got_idx, dt = layer_prefill(dec, mesh_device, x_in, 1, chunks, qb, monkeypatch)
        txt, pmin, below = row_report(r2(ref), r2(got))
        itxt, n_sparse, n_diff = compare_indices(ref_idx.rows, got_idx.rows)
        state, state_ok = compare_dsa_state(dec.cache, S)
        logger.info(
            f"layer 3 chunks {list(chunks)} x sub-block {qb} (slot 1, {dt:.1f} s): pcc vs HF {pcc(x_out, got):.6f}; "
            f"vs single-shot {txt}"
        )
        logger.info(f"    selections: {itxt}")
        logger.info(f"    state slot 0 vs slot 1: {state}")
        _record(
            "dsa_layer_r4097",
            chunk=chunk,
            qblock=qb,
            row_pcc_min=pmin,
            rows_below=below,
            sel_diff=n_diff,
            sel_rows=n_sparse,
            state_ok=state_ok,
        )
        if below or n_diff or not state_ok:
            failures.append(
                f"chunk {chunk} sub-block {qb}: rows below bar {below} (min {pmin:.6f}), differing selections {n_diff}, "
                f"state {state}"
            )
    # decode on both slots (the same token at the same position: slot 0 = single-shot state, slot 1 = chunked)
    for step in range(2):
        d = torch.load(R4097 / f"layer{LAYER_DSA}_decode{step}.pt")
        pos = d["position"]
        dp = dec.make_decode_positions([pos, pos])
        xin = torch.cat([d["x_in"], d["x_in"]], dim=1)
        yd = device_streams_to_host(
            dec.decode_forward(host_streams_to_device(xin, mesh_device, pad_to=2), positions=dp), 4, 2
        )
        p0, p1 = pcc(d["x_out"], yd[:, 0:1]), pcc(d["x_out"], yd[:, 1:2])
        p01 = pcc(yd[:, 0:1], yd[:, 1:2])
        logger.info(
            f"  decode pos {pos}: slot 0 (single-shot) pcc vs HF {p0:.6f}, slot 1 (chunked) {p1:.6f}, slot 0 vs 1 {p01:.6f}"
        )
        assert p0 > HF_PCC_BAR and p1 > HF_PCC_BAR and p01 > 0.9999
    assert not failures, "\n".join(failures)


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("n_pos", [16384, 65536])
def test_dsa_attention_long(mesh_device, monkeypatch, n_pos):
    raw, cfg = _load(LAYER_DSA, attn_only=True)
    L = 16384  # the real rows (dump R16385) at the end of the context
    x_in = torch.load(R16385 / f"layer{LAYER_DSA}_prefill.pt")["x_in"][:, :L]
    rows = attention_input(raw, cfg, x_in)  # [L, 4096]
    start0 = n_pos - L
    attn, cache = _attention(mesh_device, raw, cfg, max_batch=2, max_seq_len=n_pos + 64)
    log_dram(mesh_device, f"attention + 2-slot cache for {n_pos + 64} positions")
    if start0:
        mu, sigma = _x_stats(raw, cfg)
        filled, ci = 0, 0
        t0 = time.perf_counter()
        while filled < start0:
            T = min(SEED_CHUNK, start0 - filled)
            x = synthetic_rows(mu, sigma, ci, T)
            for user in (0, 1):
                x_tt = ttnn.from_torch(
                    x.reshape(1, 1, T, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device
                )
                attn.prefill_state(x_tt, cache, user, T, start=filled)
                ttnn.deallocate(x_tt)
            filled += T
            ci += 1
        ttnn.synchronize_device(mesh_device)
        logger.info(f"seeded [0, {start0}) synthetically in both slots ({ci} chunks, {time.perf_counter() - t0:.1f} s)")
    # a prefix seeded through prefill_state must leave the two slots identical before the real rows
    ref, ref_idx, t_ref = attn_prefill(attn, cache, mesh_device, rows, 0, (L,), start0, 2048, monkeypatch)
    logger.info(
        f"layer 3 attention rows [{start0}, {n_pos}) single-shot (one {L} chunk, sub-block 2048, slot 0): {t_ref:.2f} s"
    )
    failures = []
    for chunk, qb in ((8192, 512), (8192, 2048)):
        chunks = _chunking(L, chunk)
        got, got_idx, dt = attn_prefill(attn, cache, mesh_device, rows, 1, chunks, start0, qb, monkeypatch)
        txt, pmin, below = row_report(ref, got)
        itxt, n_sparse, n_diff = compare_indices(ref_idx.rows, got_idx.rows)
        state, state_ok = compare_dsa_state(cache, n_pos)
        logger.info(f"chunks {list(chunks)} x sub-block {qb} (slot 1, {dt:.2f} s): output {txt}")
        logger.info(f"    selections: {itxt}")
        logger.info(f"    state slot 0 vs slot 1 over [0, {n_pos}): {state}")
        _record(
            "dsa_attention_long",
            n_pos=n_pos,
            chunk=chunk,
            qblock=qb,
            row_pcc_min=pmin,
            rows_below=below,
            sel_diff=n_diff,
            sel_rows=n_sparse,
            state_ok=state_ok,
        )
        if below or n_diff or not state_ok:
            failures.append(
                f"N={n_pos} chunk {chunk} sub-block {qb}: rows below bar {below} (min {pmin:.6f}), differing selections "
                f"{n_diff}, state {state}"
            )
    log_dram(mesh_device, f"N={n_pos} done")
    assert not failures, "\n".join(failures)


# ------------------------------------------------------------------------------------------- KDA
@pytest.mark.timeout(1500)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_kda_layer_r4097(mesh_device, monkeypatch):
    raw, cfg = _load(LAYER_KDA)
    pre = torch.load(R4097 / f"layer{LAYER_KDA}_prefill.pt")
    x_in, x_out = pre["x_in"], pre["x_out"]
    S = x_in.shape[1]
    dec = FunctionalDecoder.from_state_dict(
        raw, hf_config=cfg, layer_idx=LAYER_KDA, mesh_device=mesh_device, max_batch=2, max_seq_len=S + 64
    )
    assert dec.is_kda
    attn = dec.layer.attn
    ref, _, t_ref = layer_prefill(dec, mesh_device, x_in, 0, (S,), 2048, monkeypatch)
    p_hf = pcc(x_out, ref)
    logger.info(
        f"layer {LAYER_KDA} ({'KDA+MoE' if dec.is_moe else 'KDA+dense'}) single-shot S={S} (slot 0): pcc vs HF {p_hf:.6f}, {t_ref:.1f} s"
    )
    assert p_hf > HF_PCC_BAR
    rec_ref = ttnn.to_torch(dec.kda_batched.recurrent)[0].float().clone()
    conv_ref = attn.conv_history_torch(dec.kda_batched)[0].float().clone()
    r2 = lambda t: t[0].reshape(S, -1)
    failures = []
    for chunk in (1024, 2048, 4096):
        chunks = _chunking(S, chunk)
        got, _, dt = layer_prefill(dec, mesh_device, x_in, 1, chunks, 2048, monkeypatch)
        txt, pmin, below = row_report(r2(ref), r2(got))
        rec_got = ttnn.to_torch(dec.kda_batched.recurrent)[1].float()
        conv_got = attn.conv_history_torch(dec.kda_batched)[1].float()
        rec_eq, conv_eq = torch.equal(rec_ref, rec_got), torch.equal(conv_ref, conv_got)
        rec_p, conv_p = pcc(rec_ref, rec_got), pcc(conv_ref, conv_got)
        rec_d, conv_d = (rec_ref - rec_got).abs().max().item(), (conv_ref - conv_got).abs().max().item()
        logger.info(
            f"layer {LAYER_KDA} chunks {list(chunks)} (slot 1, {dt:.1f} s): pcc vs HF {pcc(x_out, got):.6f}; vs single-shot {txt}"
        )
        logger.info(
            f"    recurrent state {tuple(rec_ref.shape)} slot 0 vs 1: {'equal' if rec_eq else f'pcc {rec_p:.7f} max|diff| {rec_d:.3g} (|state| max {rec_ref.abs().max():.3g})'}; "
            f"conv history {tuple(conv_ref.shape)}: {'equal' if conv_eq else f'pcc {conv_p:.7f} max|diff| {conv_d:.3g}'}"
        )
        _record(
            "kda_layer_r4097",
            layer=LAYER_KDA,
            chunk=chunk,
            row_pcc_min=pmin,
            rows_below=below,
            rec_pcc=rec_p,
            rec_maxdiff=rec_d,
            conv_pcc=conv_p,
            conv_maxdiff=conv_d,
        )
        if below or rec_p < ROW_PCC_BAR or conv_p < ROW_PCC_BAR:
            failures.append(
                f"chunk {chunk}: rows below bar {below} (min {pmin:.6f}), recurrent pcc {rec_p:.7f}, conv pcc {conv_p:.7f}"
            )
    # the slots' states must also agree on what comes next: the dump's decode step on both slots
    d = torch.load(R4097 / f"layer{LAYER_KDA}_decode0.pt")
    pos = d["position"]
    dp = dec.make_decode_positions([pos, pos])
    xin = torch.cat([d["x_in"], d["x_in"]], dim=1)
    yd = device_streams_to_host(
        dec.decode_forward(host_streams_to_device(xin, mesh_device, pad_to=2), positions=dp), 4, 2
    )
    p0, p1, p01 = pcc(d["x_out"], yd[:, 0:1]), pcc(d["x_out"], yd[:, 1:2]), pcc(yd[:, 0:1], yd[:, 1:2])
    logger.info(
        f"  decode pos {pos}: slot 0 (single-shot) pcc vs HF {p0:.6f}, slot 1 (chunked 4096+1) {p1:.6f}, slot 0 vs 1 {p01:.6f}"
    )
    assert p0 > HF_PCC_BAR and p1 > HF_PCC_BAR and p01 > ROW_PCC_BAR
    assert not failures, "\n".join(failures)


# ------------------------------------------------------------------------------------------ cost
STAGE_GROUPS = {
    "stems": ("prefill_stems",),
    "cache_fill+index": ("cache_fill", "index_maintain"),
    "dense_attention": ("dense_attention",),
    "kv_copy": ("kv_copy",),
    "index_queries": ("index_queries",),
    "scorer": ("index_score",),
    "topk": ("score_untilize", "topk", "score_expand", "ids_concat"),
    "absorb_q": ("absorb_q",),
    "sparse_sdpa": ("sparse_sdpa",),
    "sparse_out": ("sparse_out",),
    "out_proj": ("out_proj",),
}


def _grouped(timer: StageTimer) -> dict[str, float]:
    return {g: sum(timer.times.get(k, 0.0) for k in ks) for g, ks in STAGE_GROUPS.items()}


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_dsa_prefill_chunk_cost(mesh_device, monkeypatch):
    assert CURVE == sorted(CURVE) and all(n % TILE == 0 for n in CURVE) and COST_CHUNK % TILE == 0
    assert CURVE[0] >= COST_CHUNK
    monkeypatch.setenv("GLM53_PREFILL_QBLOCK", str(COST_QBLOCK))
    full_layer = os.environ.get("GLM53_LONGCTX_FULL_LAYER", "0") == "1"
    row_sim = int(os.environ.get("GLM53_LONGCTX_ROW_SIM", "0"))
    n_max = CURVE[-1] + row_sim * COST_QBLOCK  # headroom for the emulated row's chunk-start offset
    raw, cfg = _load(LAYER_DSA, attn_only=not full_layer)
    x_in = torch.load(R16385 / f"layer{LAYER_DSA}_prefill.pt")["x_in"][:, :COST_CHUNK]
    rows = attention_input(raw, cfg, x_in)  # real rows for the measured chunk
    mu, sigma = _x_stats(raw, cfg)
    log_dram(mesh_device, "before build")
    t0 = time.perf_counter()
    dec = None
    if full_layer:
        dec = FunctionalDecoder.from_state_dict(
            raw, hf_config=cfg, layer_idx=LAYER_DSA, mesh_device=mesh_device, max_batch=1, max_seq_len=n_max + 64
        )
        attn, cache = dec.layer.attn, dec.cache
    else:
        attn, cache = _attention(mesh_device, raw, cfg, max_batch=1, max_seq_len=n_max + 64)
    log_dram(
        mesh_device,
        f"layer 3 {'full layer' if full_layer else 'attention only'} + 1-slot cache for {n_max + 64} positions built ({time.perf_counter() - t0:.0f} s)",
    )
    timer = StageTimer(mesh_device)
    # GLM53_LONGCTX_ROW_SIM=r: emulate mesh row r of a multi-row mesh (the shared scorer's chunk-start offset
    # r*Sq + the per-row bounded top-k, tests/test_select_rows.py) for the 8x4 prefill cost on one chip
    attn.indexer.row_offset_sim = row_sim
    filled, ci = 0, 0
    table = []
    for N in CURVE:
        start0 = N - COST_CHUNK
        t0 = time.perf_counter()
        while filled < start0:
            T = min(SEED_CHUNK, start0 - filled)
            x_tt = ttnn.from_torch(
                synthetic_rows(mu, sigma, ci, T).reshape(1, 1, T, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
            )
            attn.prefill_state(x_tt, cache, 0, T, start=filled)
            ttnn.deallocate(x_tt)
            filled += T
            ci += 1
        ttnn.synchronize_device(mesh_device)
        t_seed = time.perf_counter() - t0
        if dec is not None:
            dec.next_start[0] = start0
        walls = {}
        for variant in ("compile", "timed", "free"):
            attn.set_stage_timer(timer if variant == "timed" else None)
            if variant == "timed":
                timer.reset()  # (a reset on the later "free" pass would wipe the recorded stages)
            if dec is not None:
                streams = host_streams_to_device(x_in, mesh_device)
                dec.next_start[0] = start0
            else:
                x_tt = _rows_to_device(mesh_device, rows, 0, COST_CHUNK)
            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter()
            if dec is not None:
                y = dec.prefill_forward(streams, user=0, seq_len=COST_CHUNK, start=start0)
            else:
                y = attn.prefill_forward(x_tt, cache, 0, COST_CHUNK, start=start0)
            ttnn.synchronize_device(mesh_device)
            walls[variant] = time.perf_counter() - t0
            ttnn.deallocate(y)
            ttnn.deallocate(streams if dec is not None else x_tt)
        attn.set_stage_timer(None)
        filled = N
        alloc, free = log_dram(mesh_device, f"N={N} after the measured chunk")
        groups = _grouped(timer)
        peak = max(timer.dram.values()) / 2**30 if timer.dram else float("nan")
        rec = {
            "N": N,
            "start": start0,
            "chunk": COST_CHUNK,
            "qblock": COST_QBLOCK,
            "full_layer": full_layer,
            "seed_s": t_seed,
            "wall_compile_s": walls["compile"],
            "wall_timed_s": walls["timed"],
            "wall_free_s": walls["free"],
            "tok_s": COST_CHUNK / walls["free"],
            "stages_s": groups,
            "stage_total_s": timer.total,
            "dram_peak_gib": peak,
        }
        table.append(rec)
        logger.info(
            f"N={N}: chunk [{start0}, {N}) of {COST_CHUNK} tokens (sub-block {COST_QBLOCK}): wall compile {walls['compile']:.2f} s, "
            f"timed {walls['timed']:.2f} s, free {walls['free']:.3f} s = {COST_CHUNK / walls['free']:.0f} tok/s; seeding {t_seed:.1f} s; "
            f"DRAM peak at the stage boundaries {peak:.2f} GiB"
        )
        logger.info(
            "    stages: "
            + "  ".join(f"{g} {v * 1e3:.1f}ms" for g, v in groups.items())
            + f"  | total {timer.total * 1e3:.1f}ms"
        )
        logger.info(f"    raw: {timer.report()}")
        _record("dsa_prefill_chunk_cost", **rec)
    # projection for a 1 M prompt in chunks of COST_CHUNK: the per-chunk cost as a function of the context
    # end N is interpolated linearly between the measured points (extrapolated flat below / above)
    pts = sorted((r["N"], r["wall_free_s"]) for r in table)

    def cost_at(n):
        if n <= pts[0][0]:
            return pts[0][1]
        if n >= pts[-1][0]:
            return pts[-1][1]
        for (n0, c0), (n1, c1) in zip(pts, pts[1:]):
            if n0 <= n <= n1:
                return c0 + (c1 - c0) * (n - n0) / (n1 - n0)
        return pts[-1][1]

    total_1m = 1 << 20
    n_chunks = total_1m // COST_CHUNK
    proj = sum(cost_at((i + 1) * COST_CHUNK) for i in range(n_chunks))
    logger.info(
        "| positions N | chunk wall s (free) | tok/s | kv_copy ms | scorer ms | topk ms | sparse_sdpa ms | stages total ms |"
    )
    for r in table:
        g = r["stages_s"]
        logger.info(
            f"| {r['N']} | {r['wall_free_s']:.3f} | {r['tok_s']:.0f} | {g['kv_copy'] * 1e3:.1f} | {g['scorer'] * 1e3:.1f} | "
            f"{g['topk'] * 1e3:.1f} | {g['sparse_sdpa'] * 1e3:.1f} | {r['stage_total_s'] * 1e3:.1f} |"
        )
    logger.info(
        f"projection: a {total_1m}-token prompt in {n_chunks} chunks of {COST_CHUNK} on one chip, this layer alone: "
        f"{proj:.1f} s (interpolated over the measured points {[n for n, _ in pts]}; x 11 DSA layers = {11 * proj:.0f} s)"
    )
    _record("dsa_prefill_1m_projection", chunk=COST_CHUNK, qblock=COST_QBLOCK, layer_s=proj, points=pts)
