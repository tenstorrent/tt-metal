# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Global KV pair split under vLLM (no ttnn): the layout, the per-chip positions and fill tables, and a torch reference
of the split attention with its LSE merge.

MiMo's 9 global layers have 4 KV heads over the 8 chips of a lane, so chips 2g and 2g + 1 (a pair, rep = 2) both serve
KV head g. Under vLLM a block can be shared by requests in slots of either parity (prefix caching), so the pair cannot
split the users (the standalone model's kv.py layout). Instead each chip of the pair stores half of every global
block: the chip of parity ``c = chip % 2`` keeps the global positions p with ``own(p, c)``, the 64-token halves
[128 j + 64 c, 128 j + 64 c + 64) of every 128-token block j.

* vLLM's global KV cache group has block size 128 (PAIR). Its page has the bytes of the 64-token SWA page (4 KV heads x
  128 tokens against 8 x 64), so both group kinds draw from one pool, and a block id names the same physical page
  ``[1, 64, dim]`` per chip (SUB rows) on every layer that shares it.
* Global block j of a request (table entry j, block id b) covers global positions [128 j, 128 j + 128). Chip c keeps
  positions [128 j + 64 c, 128 j + 64 c + 64) in rows 0..63 of page b: local index ``local(p) = 64 (p >> 7) + (p & 63)``.
  So both chips read and write through vLLM's table unchanged, one entry per 64 local tokens: the paged ops see a
  page table indexed by local position.
* Decode on chip c writes a token at p to ``local(p)`` when it owns p, else nowhere (-1), and its SDPA attends to the
  local keys visible at p: the first ``visible(p, c)`` local indices, so its SDPA position is ``visible(p, c) - 1``
  (-1: none).
* Each chip computes its query heads over its local keys: a normalized partial output ``o_c`` and the log of its
  softmax denominator ``lse_c`` (``m_c + log l_c``, logits ``scale * q.k``). The pair exchanges the LSE and each chip
  scales its partial by ``alpha_c = exp(lse_c) / (exp(lse_c) + exp(lse_p))``; the row all-reduce after the o
  projection then sums ``W (alpha_A o_A + alpha_B o_B)``, which is full attention. A query with no visible key on a
  chip has ``o_c = 0`` and ``lse_c = NO_KEY_LSE``, so ``alpha_c = 0``; with none on either chip alpha is 1/2 of zeros.

``positions_step`` mirrors kernels/attn/pair_positions.cpp, ``merge_weights_device`` the device merge
(tt/pair_split.py), ``fill_table`` the prefill fill tables (tt/kv.py).
"""

import math

import torch

SUB = 64  # tokens of a physical page: a chip's local block
PAIR = 2 * SUB  # global tokens of one block of vLLM's global group (half on each chip of the pair)
REP = 2  # chips per global KV head (tp / num_kv_heads)
NO_KEY_LSE = -1e30  # the LSE of a query that sees no key on a chip (finite: the merge stays NaN-free)
LSE_CLAMP = 80.0  # the device merge clamps lse_p - lse_c to +-80 before exp (exp(80) < fp32 max)


# ---------------------------------------------------------------------- scalar layout
def parity(chip):
    """The pair parity of chip ``chip`` of the lane's row (chips 2g, 2g + 1 serve KV head g)."""
    return chip % REP


def own(p, c):
    """Whether chip parity ``c`` keeps global position ``p``."""
    return p >= 0 and ((p >> 6) & 1) == c


def local(p):
    """The local index of global position ``p`` on the chip that keeps it."""
    return SUB * (p >> 7) + (p & (SUB - 1))


def global_position(l, c):
    """The global position of local index ``l`` on chip parity ``c`` (the inverse of ``local``)."""
    return PAIR * (l >> 6) + SUB * c + (l & (SUB - 1))


def visible(p, c):
    """Local keys of chip parity ``c`` at global positions <= ``p`` (0 for p < 0): they are local indices 0 .. L - 1."""
    if p < 0:
        return 0
    return SUB * (p >> 7) + min(max((p & (PAIR - 1)) + 1 - SUB * c, 0), SUB)


def write_position(p, c):
    """Where chip parity ``c`` writes the K / V of a token at ``p``: ``local(p)``, or -1 (the cache update skips it)."""
    return local(p) if own(p, c) else -1


def sdpa_position(p, c):
    """The decode SDPA position of a user at ``p`` on chip parity ``c``: its last visible local index (-1: none)."""
    return visible(p, c) - 1


# ---------------------------------------------------------------------- decode positions (the device kernel's mirror)
def positions_step(positions, c, serial_slots=0):
    """kernels/attn/pair_positions.cpp on chip parity ``c``: global positions [B] (-1 an idle slot) -> (write [B],
    sdpa [B], serial: ``serial_slots`` lists [B], list k keeping the write positions of the slots j with j %
    serial_slots == k: spec_decode.serial_positions' columns), Python ints."""
    write = [write_position(int(p), c) if int(p) >= 0 else -1 for p in positions]
    sdpa = [sdpa_position(int(p), c) for p in positions]
    serial = [[w if j % serial_slots == k else -1 for j, w in enumerate(write)] for k in range(serial_slots)]
    return write, sdpa, serial


def chip_positions(positions, tp):
    """positions [B] -> (write int32 [tp, B], sdpa int32 [tp, B]): every chip's decode positions."""
    rows = [positions_step(positions.tolist(), parity(chip)) for chip in range(tp)]
    write = torch.tensor([r[0] for r in rows], dtype=torch.int32)
    sdpa = torch.tensor([r[1] for r in rows], dtype=torch.int32)
    return write, sdpa


# ---------------------------------------------------------------------- prefill fill tables
def fill_table(row, start, padded, c, scratch=0):
    """The fill table of one chunk of ``padded`` tokens from ``start`` (a multiple of PAIR) on chip parity ``c``:
    entry k is where ``paged_fill_cache`` writes the chunk's 64-token sub-block k (positions start + 64 k ..
    start + 64 k + 63), the block of vLLM's global table ``row`` (one id per 128 global tokens) that holds it when the
    chip keeps it, else ``scratch`` (also past the end of ``row``). Row r of the fill input then lands at local index
    ``local(start + r)`` of the chip that keeps it."""
    assert start % PAIR == 0, f"a pair-split chunk starts on a {PAIR}-token block, got {start}"
    out = []
    for k in range(-(-padded // SUB)):
        q = start + SUB * k
        j = q // PAIR
        out.append(int(row[j]) if own(q, c) and j < len(row) else scratch)
    return out


# ---------------------------------------------------------------------- capacity
def global_pool_tokens(max_model_len):
    """Pool tokens (64-token pages x 64) vLLM's global group takes for one request of ``max_model_len`` tokens: one
    page per 128 tokens."""
    return SUB * math.ceil(max_model_len / PAIR)


# ---------------------------------------------------------------------- attention reference
def local_keys(k, v, c):
    """Chip parity ``c``'s cache of a context k [T, d], v [T, dv] (global positions 0 .. T - 1): its local keys in
    local order and their global positions."""
    positions = [p for p in range(k.shape[0]) if own(p, c)]
    idx = torch.tensor(positions, dtype=torch.long)
    return k[idx], v[idx], idx


def full_attention(q, k, v, scale, q_positions):
    """Causal attention of queries q [H, S, d] at global positions ``q_positions`` [S] over k [T, d], v [T, dv]
    (positions 0 .. T - 1): [H, S, dv]."""
    logits = scale * torch.einsum("hsd,td->hst", q, k)
    mask = torch.arange(k.shape[0]).reshape(1, 1, -1) <= torch.as_tensor(q_positions).reshape(1, -1, 1)
    logits = logits.masked_fill(~mask, float("-inf"))
    return torch.einsum("hst,te->hse", torch.softmax(logits, dim=-1), v)


def partial_attention(q, k, v, scale, q_positions, c):
    """Chip parity ``c``'s part of the split attention: its query heads q [H, S, d] at ``q_positions`` over its local
    keys (local key l visible to query i iff global_position(l, c) <= q_positions[i]). Returns (o [H, S, dv]: the
    softmax-normalized partial, 0 where no key is visible; lse [H, S]: max logit + log of the sum of exp, NO_KEY_LSE
    where none is), as the ops return them with ``return_lse``."""
    kl, vl, gpos = local_keys(k, v, c)
    if kl.shape[0] == 0:
        o = torch.zeros(q.shape[0], q.shape[1], v.shape[-1], dtype=q.dtype)
        return o, torch.full(q.shape[:2], NO_KEY_LSE, dtype=q.dtype)
    pos = torch.as_tensor(q_positions).reshape(1, -1, 1)
    mask = gpos.reshape(1, 1, -1) <= pos
    logits = scale * torch.einsum("hsd,td->hst", q, kl)
    logits = logits.masked_fill(~mask, float("-inf"))
    any_key = mask.any(dim=-1).expand(q.shape[0], -1)
    m = logits.amax(dim=-1, keepdim=True)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    e = torch.exp(logits - m)
    den = e.sum(dim=-1, keepdim=True)
    o = torch.einsum("hst,te->hse", e / den.clamp_min(torch.finfo(den.dtype).tiny), vl)
    o = torch.where(any_key.unsqueeze(-1), o, torch.zeros_like(o))
    lse = (m + torch.log(den)).squeeze(-1)
    lse = torch.where(any_key, lse, torch.full_like(lse, NO_KEY_LSE))
    return o, lse


def merge_weights(lse_c, lse_p):
    """alpha_c = exp(lse_c) / (exp(lse_c) + exp(lse_p)), computed against the larger LSE (the spec's form)."""
    m = torch.maximum(lse_c, lse_p)
    w_c, w_p = torch.exp(lse_c - m), torch.exp(lse_p - m)
    return w_c / (w_c + w_p)


def merge_weights_device(lse_c, lse_p):
    """The device merge's form (tt/pair_split.py): 1 / (1 + exp(clamp(lse_p - lse_c, -80, 80))). Equal to
    ``merge_weights`` up to exp(-80) ~ 2e-35 absolute: with NO_KEY_LSE on this chip only it is ~2e-35 (its output is
    0, so the product is), on the partner only 1 in fp32, on both 1/2."""
    d = torch.clamp(lse_p - lse_c, -LSE_CLAMP, LSE_CLAMP)
    return torch.reciprocal(torch.exp(d) + 1.0)


def merge(o_a, lse_a, o_b, lse_b, weights=merge_weights):
    """The pair's merged attention: alpha_a o_a + alpha_b o_b (what the row all-reduce sums after the o projection)."""
    alpha_a = weights(lse_a, lse_b).unsqueeze(-1)
    alpha_b = weights(lse_b, lse_a).unsqueeze(-1)
    return alpha_a * o_a + alpha_b * o_b


def pair_attention(q, k, v, scale, q_positions, weights=merge_weights):
    """Split attention of the pair: each chip's partial over its local keys, merged. Equals ``full_attention``."""
    o0, lse0 = partial_attention(q, k, v, scale, q_positions, 0)
    o1, lse1 = partial_attention(q, k, v, scale, q_positions, 1)
    return merge(o0, lse0, o1, lse1, weights)
