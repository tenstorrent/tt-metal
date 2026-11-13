# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU reference of the Hy4 column sequence-sharding (design task kv-design; numpy only): token -> (page, column, local row), per-column filtering of the DSA top-k ids, the exact merge of per-column partial
softmax results (flash-decoding style, attention sink applied once) and the two-stage top-k. DESIGN.md s2. These functions define the contract the device kernels are tested against.
"""
import numpy as np

CHUNK, COLS, SLICE = 512, 8, 64
SENTINEL = 0xFFFFFFFF


def locate(t):
    """global token position -> (pool page, column, local row on that column). Column c of page p holds tokens [p*512 + 64c, p*512 + 64c + 64) at local rows [64p, 64p + 64)."""
    t = np.asarray(t, dtype=np.int64)
    page = t // CHUNK
    col = (t % CHUNK) // SLICE
    return page, col, page * SLICE + t % SLICE


def global_pos(page, col, local_row):
    page, col, local_row = (np.asarray(x, dtype=np.int64) for x in (page, col, local_row))
    return page * CHUNK + col * SLICE + (local_row - page * SLICE)


def split_ids(ids, cols=COLS):
    """ids: [..., K] natural positions (SENTINEL = masked, contiguous tail). -> [cols, ..., K] uint32 local rows owned by each column with the sentinels moved to a contiguous tail (stock sparse_sdpa precondition),
    and the per-column count of valid keys [cols, ...] (may be 0: the op must then emit max=-inf, sum=0 or the merge must skip the column)
    """
    ids = np.asarray(ids, dtype=np.int64)
    valid = ids != SENTINEL
    _, col, loc = locate(np.where(valid, ids, 0))
    out = np.full((cols,) + ids.shape, SENTINEL, dtype=np.int64)
    cnt = np.zeros((cols,) + ids.shape[:-1], dtype=np.int64)
    for c in range(cols):
        m = valid & (col == c)
        cnt[c] = m.sum(-1)
        order = np.argsort(~m, axis=-1, kind="stable")  # owned first, stable
        packed = np.take_along_axis(np.where(m, loc, SENTINEL), order, axis=-1)
        out[c] = packed
    return out.astype(np.uint32), cnt


def partial_attention(q, kv, local_ids, v_dim, scale):
    """one column's partial result for ONE query row: q [H, D], kv [T_local, D], local_ids [K] (SENTINEL tail). -> (o [H, v_dim] normalised by the local sum, m [H] row max (raw*scale), l [H] sum exp(s - m))
    sum = 0 and m = -inf when the column owns no selected key."""
    sel = local_ids[local_ids != SENTINEL].astype(np.int64)
    H = q.shape[0]
    if sel.size == 0:
        return np.zeros((H, v_dim)), np.full(H, -np.inf), np.zeros(H)
    k = kv[sel]
    s = (q @ k.T) * scale
    m = s.max(-1)
    p = np.exp(s - m[:, None])
    l = p.sum(-1)
    o = (p @ k[:, :v_dim]) / l[:, None]
    return o, m, l


def merge(parts, sink=None):
    """exact combine of per-column (o, m, l): M = max m_c; w_c = l_c exp(m_c - M); out = sum o_c w_c / (sum w_c + exp(sink - M)); the sink (per head, in the scaled-logit domain) enters ONCE."""
    o = np.stack([p[0] for p in parts])
    m = np.stack([p[1] for p in parts])
    l = np.stack([p[2] for p in parts])
    M = m.max(0)
    Msafe = np.where(np.isfinite(M), M, 0.0)
    w = l * np.exp(np.where(np.isfinite(m), m - Msafe, -np.inf))
    den = w.sum(0)
    if sink is not None:
        den = den + np.exp(np.asarray(sink) - Msafe)
    return (o * w[..., None]).sum(0) / den[:, None]


def reference_attention(q, kv, ids, v_dim, scale, sink=None):
    sel = ids[ids != SENTINEL].astype(np.int64)
    s = (q @ kv[sel].T) * scale
    M = s.max(-1)
    p = np.exp(s - M[:, None])
    den = p.sum(-1)
    if sink is not None:
        den = den + np.exp(np.asarray(sink) - M)
    return (p @ kv[sel][:, :v_dim]) / den[:, None]


def shard_kv(kv_natural, n_pages, cols=COLS):
    """natural [n_pages*512, D] -> per column [cols, n_pages*64, D] following ``locate``"""
    t = np.arange(n_pages * CHUNK)
    _, col, loc = locate(t)
    out = np.zeros((cols, n_pages * SLICE, kv_natural.shape[1]), dtype=kv_natural.dtype)
    out[col, loc] = kv_natural
    return out


def two_stage_topk(scores, k, cols=COLS):
    """scores [T] natural order -> ids of the global top-k computed as per-column local top-k then a merge (what the sharded indexer does). Exact for distinct scores."""
    t = np.arange(scores.size)
    _, col, _ = locate(t)
    cand = []
    for c in range(cols):
        idx = t[col == c]
        take = idx[np.argsort(-scores[idx], kind="stable")[:k]]
        cand.append(take)
    cand = np.concatenate(cand)
    return cand[np.argsort(-scores[cand], kind="stable")[:k]]
