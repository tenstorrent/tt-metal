# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of GEMMA4_SDPA_PAIR_SPLIT (tt/sdpa_pair_split.py): the per-chip pair QKV / o_proj layout, the
partner selector and the LSE merge, against a torch emulation of the (1,8) row. Pure torch: neither this file nor
the modules it imports touch ttnn.

    pytest models/demos/gemma4/tests/unit/test_pair_split_host.py --noconftest -p no:cacheprovider
"""
from __future__ import annotations

import math

import pytest
import torch

from models.demos.gemma4.tt import kv_pair_shard as kps
from models.demos.gemma4.tt import mesh_orientation as mo
from models.demos.gemma4.tt import sdpa_pair_split as ps

# A small 31B-shaped row: 32 q heads, 4 KV heads, TP 8 over one mesh row (4 q heads per chip in the 4-head layout,
# 8 = one KV group per chip under the pair split), K=V tying.
NQ, NKV, TP, D, HIDDEN = 32, 4, 8, 16, 64
G = NQ // NKV
CHUNK = 8  # keys per chunk of the emulated halves
CUR_POS = (0, 3, 37, 100)  # one chunk (the second half empty), a partial chunk, several chunks
S = 128


def _rand(*shape, seed):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=gen, dtype=torch.float32)


def _model(seed=0):
    """HF-shaped weights (``q_proj [NQ*D, HIDDEN]``, ``k_proj [NKV*D, HIDDEN]``, V = K, ``o_proj [HIDDEN, NQ*D]``),
    the decode rows x [B, HIDDEN] and each KV head's cache history [B, NKV, S, D] (K = V)."""
    q_w = _rand(NQ * D, HIDDEN, seed=seed) / math.sqrt(HIDDEN)
    k_w = _rand(NKV * D, HIDDEN, seed=seed + 1) / math.sqrt(HIDDEN)
    o_w = _rand(HIDDEN, NQ * D, seed=seed + 2) / math.sqrt(NQ * D)
    x = _rand(len(CUR_POS), HIDDEN, seed=seed + 3)
    cache = _rand(len(CUR_POS), NKV, S, D, seed=seed + 4)
    return q_w, k_w, o_w, x, cache


def _with_new_token(cache, b, g, k_new, pos):
    """KV head g of user b after the decode step's cache write: the new K (= V) row at ``pos``."""
    kv = cache[b, g].clone()
    kv[pos] = k_new
    return kv


def _attend(q, kv, keys, scale):
    """One chip's attention of q [H, D] over the rows ``keys`` of kv [S, D] (K = V), as SdpaDecodeLse writes it:
    O normalised over those keys, m the raw-score max, l = sum exp((s - m) * scale); no keys: O = 0, m = -30000,
    l = 1."""
    H = q.shape[0]
    if len(keys) == 0:
        return torch.zeros(H, D), torch.full((H,), ps.EMPTY_HALF_M), torch.ones(H)
    s = q @ kv[keys].T
    m = s.max(-1).values
    p = torch.exp((s - m[:, None]) * scale)
    l = p.sum(-1)
    return (p @ kv[keys]) / l[:, None], m, l


def _half_keys(pos, half):
    """Keys 0..pos in CHUNK-key chunks, the first ceil(n / 2) chunks to half 0 and the rest to half 1."""
    n = (pos + 1 + CHUNK - 1) // CHUNK
    lo, hi = (0, (n + 1) // 2) if half == 0 else ((n + 1) // 2, n)
    return list(range(lo * CHUNK, min(hi * CHUNK, pos + 1)))


def _half_block_keys(pos, half):
    """The keys pair half ``half`` attends under GEMMA4_KV_PAIR_SHARD, through the production helpers: positions
    0..pos written through its write table (``kv_pair_shard.half_block_write_table``: CHUNK-token device pages,
    logical blocks of 2 CHUNK tokens, the partner's half into the null block), then read back as the decode kernel
    does -- the compacted sequence 0..``half_block_cur_pos`` over the logical table, entry j // CHUNK, offset
    j % CHUNK. They are exactly the positions of the chunks of parity ``half``."""
    n_logical = S // (2 * CHUNK)
    gen = torch.Generator().manual_seed(7)
    table = (torch.randperm(4 * n_logical, generator=gen)[:n_logical] + 1).unsqueeze(0)  # 0 is the null block
    wt = kps.half_block_write_table(table, half)[0]
    assert wt.shape == (2 * n_logical,) and set(wt[1 - half :: 2].tolist()) == {kps.NULL_BLOCK}
    pages = {}
    for p in range(pos + 1):
        pages.setdefault(int(wt[p // CHUNK]), {})[p % CHUNK] = p
    eff = int(kps.half_block_cur_pos(torch.tensor([pos]), half, CHUNK)[0])
    keys = [pages[int(table[0, j // CHUNK])][j % CHUNK] for j in range(eff + 1)]
    assert keys == [p for p in range(pos + 1) if (p // CHUNK) % 2 == half], (pos, half)
    return keys


def _reference(q_w, k_w, o_w, x, cache, scale):
    """Full GQA attention of every q head over keys 0..pos (the new K written at pos), then o_proj."""
    out = torch.zeros(len(CUR_POS), NQ * D)
    for b, pos in enumerate(CUR_POS):
        for g in range(NKV):
            kv = _with_new_token(cache, b, g, x[b] @ k_w[g * D : (g + 1) * D].T, pos)
            for h in range(g * G, (g + 1) * G):
                s = (x[b] @ q_w[h * D : (h + 1) * D].T) @ kv[: pos + 1].T * scale
                out[b, h * D : (h + 1) * D] = torch.softmax(s, -1) @ kv[: pos + 1]
    return out @ o_w.T


def _regular_layout(q_w, k_w, o_w, x, cache, scale):
    """The 4-head layout of tt/attention/weights.py (kv_replicated): device i holds q heads [4i, 4i + 4), KV head
    (i * 4) * NKV // NQ and the o_proj rows of its q heads; the row all-reduce sums the per-device products."""
    per = NQ // TP
    y = torch.zeros(len(CUR_POS), HIDDEN)
    for i in range(TP):
        kv_idx = (i * per) * NKV // NQ
        wq = torch.chunk(q_w, TP, dim=0)[i]
        wk = k_w[kv_idx * D : (kv_idx + 1) * D]
        wo = o_w.T[i * per * D : (i + 1) * per * D]
        for b, pos in enumerate(CUR_POS):
            q = (x[b] @ wq.T).reshape(per, D)
            kv = _with_new_token(cache, b, kv_idx, x[b] @ wk.T, pos)
            o, _, _ = _attend(q, kv, list(range(pos + 1)), scale)
            y[b] += o.reshape(-1) @ wo
    return y


def _pair_layout(q_w, k_w, o_w, x, cache, scale, keys_of=_half_keys):
    """The pair layout through the production helpers: per device column i the pair QKV block and o_proj block,
    half i % 2 of the keys (``keys_of``), the LSE merge (pair_merge_reference), the row sum. Returns (y, per-device
    K rows)."""
    kw = dict(num_q_heads=NQ, num_kv_heads=NKV, head_dim=D, tp=TP)
    wqkv = ps.pair_qkv_weight(q_w, k_w, k_w, **kw)  # [HIDDEN, TP * (G + 2) * D]
    wo = ps.pair_o_proj_weight(o_w.T, **kw)  # [TP * G * D, HIDDEN]
    width = (G + 2) * D
    assert wqkv.shape == (HIDDEN, TP * width) and wo.shape == (TP * G * D, HIDDEN)
    B = len(CUR_POS)
    out, m, l = torch.zeros(TP, B, G, D), torch.zeros(TP, B, G), torch.zeros(TP, B, G)
    k_rows = torch.zeros(TP, B, D)
    for i in range(TP):
        proj = x @ wqkv[:, i * width : (i + 1) * width]  # [B, (G + 2) * D]
        q, k_new, v_new = proj[:, : G * D], proj[:, G * D : (G + 1) * D], proj[:, (G + 1) * D :]
        assert torch.equal(k_new, v_new), "K=V tying: the V block is the K block"
        k_rows[i] = k_new
        for b, pos in enumerate(CUR_POS):
            kv = _with_new_token(cache, b, i // 2, k_new[b], pos)
            out[i, b], m[i, b], l[i, b] = _attend(q[b].reshape(G, D), kv, keys_of(pos, i % 2), scale)
    merged = ps.pair_merge_reference(out, m, l, scale=scale)  # [TP, B, G, D]
    y = sum(merged[i].reshape(B, G * D) @ wo[i * G * D : (i + 1) * G * D] for i in range(TP))
    return y, k_rows


@pytest.mark.parametrize("scale", (1.0, 0.25))
def test_pair_layout_sums_to_the_regular_layout_and_full_attention(scale):
    """(a) sum over the row of (w * O_half) @ Wo_pair[i] == the 4-head layout's sum of O_own @ Wo[i] == full attention
    @ o_proj, and both chips of a pair write the same K / V (the KV head the 4-head layout writes there); (b) the same
    with the half-block ownership of the pair-sharded global KV (GEMMA4_KV_PAIR_SHARD: each chip keeps the chunks of
    its parity, written through its write table and read through the logical table, ``_half_block_keys``)."""
    q_w, k_w, o_w, x, cache = _model()
    ref = _reference(q_w, k_w, o_w, x, cache, scale)
    regular = _regular_layout(q_w, k_w, o_w, x, cache, scale)
    pair, k_rows = _pair_layout(q_w, k_w, o_w, x, cache, scale)
    tol = dict(rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(regular, ref, **tol)
    torch.testing.assert_close(pair, ref, **tol)
    torch.testing.assert_close(pair, regular, **tol)
    half_block, _ = _pair_layout(q_w, k_w, o_w, x, cache, scale, keys_of=_half_block_keys)
    torch.testing.assert_close(half_block, ref, **tol)
    for i in range(TP):
        kv_idx = (i * (NQ // TP)) * NKV // NQ  # the 4-head layout's KV head on column i
        assert kv_idx == i // 2
        torch.testing.assert_close(k_rows[i], x @ k_w[kv_idx * D : (kv_idx + 1) * D].T, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(k_rows[i], k_rows[i ^ 1], rtol=0, atol=0)


def test_pair_blocks_hold_the_group_heads_in_natural_order():
    """Column i's QKV block is [Wq of heads G*(i//2) .. G*(i//2)+G-1 | Wk of head i//2 | Wv of head i//2] and its
    o_proj block the rows of the same heads; the two columns of a pair hold the same blocks."""
    q_w, k_w, o_w, _, _ = _model(seed=7)
    v_w = _rand(NKV * D, HIDDEN, seed=11)  # untied V, to see the V block land
    kw = dict(num_q_heads=NQ, num_kv_heads=NKV, head_dim=D, tp=TP)
    wqkv = ps.pair_qkv_weight(q_w, k_w, v_w, **kw)
    wo = ps.pair_o_proj_weight(o_w.T, **kw)
    width = (G + 2) * D
    for i in range(TP):
        g = i // 2
        block = wqkv[:, i * width : (i + 1) * width]
        for j in range(G):
            h = g * G + j
            assert torch.equal(block[:, j * D : (j + 1) * D], q_w[h * D : (h + 1) * D].T)
        assert torch.equal(block[:, G * D : (G + 1) * D], k_w[g * D : (g + 1) * D].T)
        assert torch.equal(block[:, (G + 1) * D :], v_w[g * D : (g + 1) * D].T)
        assert torch.equal(wo[i * G * D : (i + 1) * G * D], o_w.T[g * G * D : (g + 1) * G * D])
        assert torch.equal(block, wqkv[:, (i ^ 1) * width : ((i ^ 1) + 1) * width])
    assert ps.pair_heads(NQ, NKV) == 2 * (NQ // TP)


@pytest.mark.parametrize("tp", (2, 4, 8))
def test_selector_picks_the_partner_column(tp):
    """(b) The selector shard of device column c (dim 1 of the [1, tp*tp, 1, 1] host tensor, split in tp) is
    e_c - e_{c^1}: the sum over the gathered device dim of gathered x selector is lse_c - lse_{c^1}, the partner c^1
    being the other chip of c's pair."""
    sel = ps.pair_selector(tp)
    assert sel.shape == (1, tp * tp, 1, 1) and sel.dtype == torch.float32
    shards = torch.chunk(sel, tp, dim=1)  # what the mesh mapper hands device column c
    gathered = _rand(tp, 5, 3, seed=tp)  # [tp, B*H rows...]: every device's LSEs after the all-gather
    for c, shard in enumerate(shards):
        row = shard.reshape(tp)
        partner = c ^ 1
        assert partner != c and partner // 2 == c // 2
        expect = torch.zeros(tp)
        expect[c], expect[partner] = 1.0, -1.0
        assert torch.equal(row, expect)
        picked = (gathered * row[:, None, None]).sum(0)
        torch.testing.assert_close(picked, gathered[c] - gathered[partner], rtol=0, atol=0)


@pytest.mark.parametrize("empty", (0, 1))
def test_merge_with_an_empty_half_returns_the_other_half(empty):
    """(c) A half with no keys writes O = 0, m = -30000, l = 1: its weight is 0 and the live half's O comes back
    unchanged (the clamp keeps the sigmoid away from exp overflow); the pair sum is the live half's attention."""
    B, H = 3, G
    live = 1 - empty
    out = torch.zeros(2, B, H, D)
    m = torch.zeros(2, B, H)
    l = torch.ones(2, B, H)
    out[live] = _rand(B, H, D, seed=3)
    m[live] = _rand(B, H, seed=4) * 20.0
    l[live] = 1.0 + _rand(B, H, seed=5).abs() * 100.0
    out[empty], m[empty], l[empty] = 0.0, ps.EMPTY_HALF_M, 1.0
    merged = ps.pair_merge_reference(out, m, l, scale=1.0)
    assert torch.isfinite(merged).all()
    torch.testing.assert_close(merged[live], out[live], rtol=0, atol=0)
    assert torch.equal(merged[empty], torch.zeros(B, H, D))
    torch.testing.assert_close(merged.sum(0), out[live], rtol=0, atol=0)


def test_merge_matches_the_logsumexp_merge():
    """Two live halves: w_self = sigmoid(lse_self - lse_partner) is the normalised exp(lse) weight of the
    two-way LSE merge (``SdpaDecodeLse.merge_torch``'s formula)."""
    B, H = 4, G
    out = _rand(2, B, H, D, seed=21)
    m = _rand(2, B, H, seed=22) * 10.0
    l = 1.0 + _rand(2, B, H, seed=23).abs() * 50.0
    merged = ps.pair_merge_reference(out, m, l, scale=0.5).sum(0)
    lse = ps.lse_from_m_l(m, l, 0.5)
    top = torch.maximum(lse[0], lse[1])
    wa, wb = torch.exp(lse[0] - top), torch.exp(lse[1] - top)
    expect = (wa[..., None] * out[0] + wb[..., None] * out[1]) / (wa + wb)[..., None]
    torch.testing.assert_close(merged, expect, rtol=1e-5, atol=1e-6)


def test_layers_that_qualify():
    """The 31B global layer on the (1,8) row qualifies; sliding layers, other meshes, split KV heads and a pair that
    does not hold exactly one KV group do not."""
    base = dict(is_sliding=False, num_q_heads=32, num_kv_heads=4, tp=8, mesh_shape=(1, 8), kv_replicated=True)
    assert ps.pair_split_reason(**base) is None
    assert ps.pair_split_reason(**{**base, "num_q_heads": 16}) is None  # 16 / 4: 2 per chip, 4 per pair
    assert ps.pair_split_reason(**{**base, "is_sliding": True}) == "sliding layer"
    assert "one row" in ps.pair_split_reason(**{**base, "mesh_shape": (2, 4)})
    assert "one row" in ps.pair_split_reason(**{**base, "mesh_shape": (1, 7), "tp": 7})
    assert "fracture" in ps.pair_split_reason(**{**base, "fractured": True})
    assert "TP" in ps.pair_split_reason(**{**base, "tp_axis": 0})
    assert "replicated" in ps.pair_split_reason(**{**base, "tp": 4, "mesh_shape": (1, 4), "kv_replicated": False})
    assert "KV group" in ps.pair_split_reason(**{**base, "num_kv_heads": 2})  # 16 per group, 8 per pair
    assert ps.pair_split_mesh((1, 8)) and ps.pair_split_mesh((1, 2))
    assert not ps.pair_split_mesh((1, 1)) and not ps.pair_split_mesh(None)


def test_knob_is_registered_and_off_by_default(monkeypatch):
    """GEMMA4_SDPA_PAIR_SPLIT sits in the decode-flag registry; unset it is off under every build profile except the
    pair-qualified dense (1,8) line (GALAXY_LINE_PAIR_ON), and the env wins both ways."""
    assert ps.PAIR_SPLIT_ENV == mo.SDPA_PAIR_SPLIT_ENV == "GEMMA4_SDPA_PAIR_SPLIT"
    assert mo.DECODE_FUSION_OFF[ps.PAIR_SPLIT_ENV] == "0"
    monkeypatch.delenv(ps.PAIR_SPLIT_ENV, raising=False)
    try:
        for shape, moe in (((1, 8), False), ((1, 4), False), ((1, 4), True), ((8, 4), True)):
            mo.set_decode_fusion_profile(shape, blackhole=True, moe=moe)
            assert not ps.pair_split_enabled()
        mo.set_decode_fusion_profile((1, 8), blackhole=True, moe=False, kv_pairs=True)
        assert ps.pair_split_enabled()
        for shape, moe in (((1, 4), False), ((1, 8), True)):  # the line profile needs a dense build on (1,8)
            mo.set_decode_fusion_profile(shape, blackhole=True, moe=moe, kv_pairs=True)
            assert not ps.pair_split_enabled()
        monkeypatch.setenv(ps.PAIR_SPLIT_ENV, "0")
        mo.set_decode_fusion_profile((1, 8), blackhole=True, moe=False, kv_pairs=True)
        assert not ps.pair_split_enabled()
        mo.clear_decode_fusion_profile()
        monkeypatch.setenv(ps.PAIR_SPLIT_ENV, "1")
        assert ps.pair_split_enabled()
    finally:
        mo.clear_decode_fusion_profile()
