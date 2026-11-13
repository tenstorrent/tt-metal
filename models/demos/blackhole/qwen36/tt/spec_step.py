# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Packed per-step inputs of the MTP verify and drafter traces (host side, no device).

Each trace reads its per-step integers from ONE uint32 ROW_MAJOR buffer that the host stages with one host-to-device
copy: segment i is row i, zero padded to a multiple of PACK_ALIGN words, and the trace slices row i, columns [0, n).
The traces look up RoPE cos and sin on the device by table row (Qwen36Model.rope_lookup), so no float table is staged
per step.

Verify step buffer (Qwen36Model._vfy_step_buf), rows of B * T words, row r = u * T + j:
  VERIFY_TOKENS     the token of verify row r
  VERIFY_POSITIONS  its KV position (doubles as the decode SDPA's cur_pos)
  VERIFY_ROPE       its RoPE table row (rope_rows: position plus the row's M-RoPE delta)

Drafter pack (MTPSpecEngine._dr_pack), 2K + 5 rows (drafter_pack):
  0                 pending token of each user [B] (the token draft step 0 embeds)
  1 .. K            KV position of draft step k [B]
  K + 1, K + 2      reseed tokens and KV positions [B * T] (user-major)
  K + 3             anchor row u * T + m[u] of the verify feed [B]
  K + 4             RoPE table rows of the reseed rows [B * T]
  K + 5 .. 2K + 4   RoPE table rows of draft step k [B]
"""

import torch

PACK_ALIGN = 32  # words per packed row are a multiple of this (128 B)
# RoPE table rows past max_seq_len: a verify or drafter window that starts near the end still finds its rows.
ROPE_TABLE_SLACK = 64

VERIFY_TOKENS, VERIFY_POSITIONS, VERIFY_ROPE = range(3)
VERIFY_STEP_ROWS = 3

DRAFT_PENDING = 0


def draft_step_row(k):
    return 1 + k


def draft_reseed_tokens(K):
    return K + 1


def draft_reseed_positions(K):
    return K + 2


def draft_anchor(K):
    return K + 3


def draft_reseed_rope(K):
    return K + 4


def draft_step_rope(K, k):
    return K + 5 + k


def drafter_pack_rows(K):
    return 2 * K + 5


def rope_table_rows(max_seq_len):
    """Rows of the device RoPE tables of a model of ``max_seq_len`` positions (Qwen36Model.rope_rm_tables)."""
    return int(max_seq_len) + ROPE_TABLE_SLACK


def pack_width(n):
    """Words per packed row for segments of at most ``n`` words."""
    return -(-max(int(n), 1) // PACK_ALIGN) * PACK_ALIGN


def pack_u32_rows(segments, width=None):
    """Integer segments -> int32 [len(segments), width] (staged as uint32): row i holds segment i, zero padded.
    ``width`` defaults to pack_width of the longest segment. Raises on a negative value or a segment longer than
    ``width``, which the uint32 buffer cannot hold."""
    segs = [torch.as_tensor(s).reshape(-1).to(torch.int64) for s in segments]
    width = pack_width(max(int(s.numel()) for s in segs)) if width is None else int(width)
    out = torch.zeros(len(segs), width, dtype=torch.int64)
    for i, s in enumerate(segs):
        if s.numel() > width:
            raise ValueError(f"packed segment {i} has {s.numel()} words, the row holds {width}")
        out[i, : s.numel()] = s
    if bool((out < 0).any()) or bool((out >= 2**31).any()):
        raise ValueError("a packed uint32 input holds a negative value or one at or above 2**31")
    return out.to(torch.int32)


def rope_rows(positions, deltas, table_rows):
    """RoPE table row of each position: position + its M-RoPE delta (one per position, or one int for all), clamped
    into [0, table_rows). Inside the table the row holds the cos and sin the host computes for that position; only
    padding and junk rows (a negative sum, or a window past the table) are clamped."""
    pos = torch.as_tensor(positions).reshape(-1).to(torch.int64)
    if isinstance(deltas, (list, tuple, torch.Tensor)):
        d = torch.as_tensor(deltas).reshape(-1).to(torch.int64)
        if d.numel() != pos.numel():
            raise ValueError(f"{d.numel()} M-RoPE deltas for {pos.numel()} positions")
    else:
        d = int(deltas)
    return (pos + d).clamp(0, int(table_rows) - 1)


def drafter_pack(committed, m, p0, declined, keep, rope_base, K, table_rows):
    """The drafter pack (module doc) of one propose, int32 [2K + 5, width].

    ``committed`` [B, T]: the step's committed block; only columns <= m[u] (the accepted drafts and the bonus) are
    read. ``m`` [B]: accepted drafts per user. ``p0`` [B]: position of the user's verify column 0, so committed
    token j sits at KV position p0 + 1 + j and the drafter's anchor slot is p0 + m. ``declined`` [B]: rows that draft
    nothing; they draft at positions 0 .. K-1 from token 0 (over the scratch table the caller stages). ``keep`` [B]:
    declined rows whose anchor slot still gets its drafter KV (one more reseed row, i = m). ``rope_base`` [B]: the
    user's RoPE table row of position p0 (rope_rows of p0 and its delta).

    Reseed row u * T + i is real for i < m[u] + (declined and keep): token committed[u, i] at position p0 + i; the
    others carry token 0 at position 0 (the caller points them at the scratch block). RoPE rows: reseed row i takes
    rope_base + i and draft step k takes rope_base + m + k, which are the rows of the real positions (p0 + i, and
    p0 + m + k for a drafting row); a padding or declined row's RoPE row is junk that only reaches the scratch block."""
    committed = torch.as_tensor(committed).to(torch.int64)
    B, T = int(committed.shape[0]), int(committed.shape[1])
    m = torch.as_tensor(m).reshape(-1).to(torch.int64)
    p0 = torch.as_tensor(p0).reshape(-1).to(torch.int64)
    dec = torch.as_tensor(declined).reshape(-1).to(torch.bool)
    keep = torch.as_tensor(keep).reshape(-1).to(torch.bool)
    rb = torch.as_tensor(rope_base).reshape(-1).to(torch.int64)
    if T != K + 1 or not all(int(t.numel()) == B for t in (m, p0, dec, keep, rb)):
        raise ValueError(f"drafter_pack needs [B, {K + 1}] committed and B entries per user")
    if bool(((m < 0) | (m > K)).any()):
        raise ValueError(f"accepted drafts {m.tolist()} outside [0, {K}]")
    j = torch.arange(T).reshape(1, T)
    k = torch.arange(K).reshape(1, K)
    m_kv = m + (dec & keep).to(torch.int64)
    real = j < m_kv.reshape(B, 1)
    pending = torch.where(dec, 0, committed.gather(1, m.reshape(B, 1)).reshape(B))
    p = torch.where(dec, 0, p0 + m)
    steps = p.reshape(B, 1) + k  # [B, K]
    rtok = torch.where(real, committed, 0)
    rpos = torch.where(real, p0.reshape(B, 1) + j, 0)
    top = int(table_rows) - 1
    rope_reseed = (rb.reshape(B, 1) + j).clamp(max=top)
    rope_steps = (rb.reshape(B, 1) + m.reshape(B, 1) + k).clamp(max=top)
    anchor = torch.arange(B) * T + m
    return pack_u32_rows(
        [pending, *steps.T, rtok, rpos, anchor, rope_reseed, *rope_steps.T],
        width=pack_width(B * T),
    )


def reseed_real_rows(m, declined, keep, T):
    """[B * T] bool: the reseed rows drafter_pack makes real (user-major)."""
    m_kv = torch.as_tensor(m).reshape(-1).to(torch.int64) + (
        torch.as_tensor(declined).reshape(-1).to(torch.bool) & torch.as_tensor(keep).reshape(-1).to(torch.bool)
    ).to(torch.int64)
    return (torch.arange(T).reshape(1, T) < m_kv.reshape(-1, 1)).reshape(-1)


def reseed_table(pt, real, T, scratch_block):
    """The reseed page table [B * T, nb]: real row u * T + i takes pt[u], the others the scratch block."""
    pt = torch.as_tensor(pt).to(torch.int32)
    rows = pt.repeat_interleave(T, dim=0)
    return torch.where(torch.as_tensor(real).reshape(-1, 1), rows, torch.full_like(rows, int(scratch_block)))
