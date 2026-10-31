# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Long-lived MTP speculative-decode engine for Qwen3.6-27B (TP), driven one call at a time.

``prepare`` (= prepare_prefill + prepare_decode) allocates and compiles everything, ``capture`` (= record_prefill +
capture_decode) records the prefill chunk trace (B == 1) and the verify trace, then ``prefill`` /
``verify`` / ``propose`` serve any number of requests, with page tables passed on every call. Nothing
persistent is allocated and nothing compiles after ``capture`` (#54032: late or program-cache buffers can
land on trace-owned memory).

Rows are user-major, r = u*T + j with T = K+1, in one 32-row decode tile; row u is state slot u.
``accepted_counts`` passed to ``verify`` describes the PREVIOUS verify (state index = accepted_counts - 1).
Verify row j of user u processes tokens[u, j] at start_pos[u] + j. After m accepted drafts,
committed_tokens[u, :m+1] = accepted drafts + bonus token, at committed_positions[u, :m+1] =
start_pos + 1 .. start_pos + m + 1. The bonus is the next ``pending``; the anchor hidden is verify row m.
The drafter pairs (hidden at slot s, token at s+1) -> slot s.

Slot map (QWEN36_SPEC_SLOTMAP, default 1): the runner front-packs its rows and sends a ``slot_remap`` after every
condense. The engine keeps every request in the engine row its prefill chose and only permutes the host map
``_phys`` (runner slot -> engine row). The runner-facing calls ``decode_forward`` / ``propose_draft_tokens`` /
``prefill`` / ``release`` take runner slots: per-row inputs are scattered to engine rows, outputs gathered back.
``verify`` / ``propose`` (the demo API) and every ``_fresh`` / ``_active`` / ... set are in engine rows; the demo never
remaps, so there the two numberings coincide. With the flag at 0 (not the default) a remap permutes the GDN state on the device.
Every per-row device input the verify stages (tokens, positions, mi_prev -> state_blk_idx / conv_sel, or under
QWEN36_VERIFY_SPEC_STEP=1 the gdn_spec_step ctrl page {parity, mi, ring block}) is built from ENGINE rows.

Fast join (QWEN36_SPEC_FAST_JOIN, default 1): with the flag at 1 a fresh row's first verify is NOT a separate eager
T = 1 forward (_seed / _join: model.seed_spec_step over all 64 layers, ~3000 dispatches, plus the ring rebuild) but the
normal verify replay itself: the row's ring slot 0 / E_prev are first seeded from its prefill state
(TPGatedDeltaNet.seed_spec_rows_select, ~6 exact data-movement ops per GDN layer), then it runs as an ordinary 0-draft
row (tokens [t_Tp, pad..], mi_prev 0) next to the running rows, in ONE replay (_seed_fresh_rows). Token 0 of a T-row
verify is one plain decode step of the same kernels, so the state and logits are those of the eager seed; rec_state is
no longer advanced by a seed, so a fresh row left out of a step keeps its prefill state (no re-prefill).

Fast prefill (QWEN36_SPEC_FAST_PREFILL, default 0): with the flag at 1 ``prefill`` replays captured traces like the
non-spec class instead of the eager masked-bucket forward (~2500 dispatches, 158 host writes, 48 eager GDN slot
writes per request): a prompt < 2048 tokens is ONE replay of the bucket's spec short trace (the base forward + the
MTP KV warm-up over the segment + the slot Tp-1 feed row, model.capture_prefill_traces_short(spec=True)), a longer
prompt replays the 2048-token chunk trace per full chunk (recorded on the B=1 GDN scratch also at B > 1, the MTP
warm-up of a full chunk stays eager) and the spec short trace for its tail, and the GDN scratch -> slot copy is the
per-slot slot-write trace. Same MTP KV semantics as the eager path: slot i = (post-norm base hidden h_i, token_{i+1})
for every prompt chunk, slot Tp-1 written at the first verify (_warm_mtp_last)."""

import math
import os
import threading
import time
import weakref
from dataclasses import dataclass
from typing import Callable

import torch
from loguru import logger
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.demos.blackhole.qwen36.tt import prefill_plan
from models.demos.blackhole.qwen36.tt.mtp import draft_argmax
from models.demos.blackhole.qwen36.tt.spec_bucket import apply_row_swap, pick_bucket, plan_row_swap
from models.demos.blackhole.qwen36.tt.spec_chain import (
    ChainDevice,
    fused_fit_mode_bounds,
    fused_fit_mode_range,
    pack_ovr_host,
)
from models.demos.blackhole.qwen36.tt.spec_fused import (
    AUX_ACTIVE,
    AUX_DECLINED,
    AUX_KEEP,
    AUX_P0,
    FUSED_AUX_ROWS,
    FusedGreedyDevice,
    aux_row_deltas,
    fused_accept_aux_host,
    fused_greedy_walk,
)
from models.demos.blackhole.qwen36.tt.spec_fused_sampled import (
    FusedSampledDevice,
    audit_boundary_users,
    fused_sampled_reference,
    host_redo_walk,
)
from models.demos.blackhole.qwen36.tt.tp_common import (
    GDN_SNAP_SLOTS,
    engine_spec_buckets,
    join_logits_one_copy_enabled,
    pack_u32_rows,
    prefill_host_tolist_enabled,
    prefill_masked_big_enabled,
    prefill_mtp_lean_enabled,
    rope_row_index,
    spec_bucket_hold,
    spec_buckets_flag,
    spec_chain_audit_enabled,
    spec_chain_audit_every,
    spec_chain_enabled,
    spec_chain_sampled_enabled,
    spec_fast_join_enabled,
    spec_fast_prefill_enabled,
    spec_fused_audit_enabled,
    spec_fused_audit_every,
    spec_fused_enabled,
    spec_fused_sampled_enabled,
    spec_hostfix_enabled,
    spec_hostset_enabled,
    spec_hostset_trace_enabled,
    spec_join_fast_enabled,
    spec_join_trace_enabled,
    spec_row_delta_enabled,
    verify_device_topk_m,
)

# Prompt length above which the reseed goes back to the per-slot loop (B == 1 only).
EAGER_RESEED_PROMPT_LEN = 131072
# Verify rows B*(K+1) must fit one decode tile (QKV heads, matmul configs, GDN core budget).
MAX_SPEC_ROWS = ttnn.TILE_SIZE
# Prompt lengths warmed in prepare: exact bucket lengths take the no-mask prefill path, the others the masked one.
_WARM_PROMPT_LENS = (100, 128, 200, 256, 400, 512, 900, 1024, 1900, 2048, 2148, 2248, 2448, 2948, 3948, 4096)
# Runner contract: token id for unused tail columns of a returned block.
PLACEHOLDER_TOKEN_ID = -1
# "topk" (QWEN36_VERIFY_DEVICE_TOPK): the exact per-position top-M candidates instead of the full logits, with the
# full logits fetched lazily when a row cannot be served from them (note 26).
SPEC_MODES = ("argmax_ids", "logits", "topk", "fused_sample")
# Token ids are packed below the value in the exact top-k sort key (exact_topk): the vocabulary must fit.
_TOPK_ID_BITS = 20


@dataclass
class TopKCandidates:
    """Exact per-position top-M of the verify logits under (value desc, token id asc): ``values`` fp32 [B, T, M]
    are the raw bf16 logit values (no temperature), ``ids`` int32 [B, T, M] the global token ids."""

    values: torch.Tensor
    ids: torch.Tensor


@dataclass
class VerifyResult:
    """decode_forward result: argmax_ids int32 [B, 1+K] (unused tail columns = -1); logits [B, 1+K, V] fp32 in
    "logits" mode; in "topk" mode topk_values / topk_ids [B, 1+K, M] (TopKCandidates, in runner slots) and
    ``fetch_logits``, which returns the step's full [B, 1+K, V] fp32 logits in runner slots when called (one
    device round trip in steady state and under fast join, a host copy after an eager seed / join step with
    QWEN36_SPEC_FAST_JOIN=0)."""

    spec_mode: str
    argmax_ids: torch.Tensor
    logits: torch.Tensor | None = None
    topk_values: torch.Tensor | None = None
    topk_ids: torch.Tensor | None = None
    fetch_logits: Callable[[], torch.Tensor] | None = None
    # "fused_sample" (QWEN36_SPEC_FUSED, greedy rows): the greedy walk done here. committed [B, 1+K] int32 (-1 past the
    # count) and counts [B] int32 in [1, 1+K]; a padding row commits its column-0 argmax with count 1.
    accepted_token_ids: torch.Tensor | None = None
    accepted_counts: torch.Tensor | None = None
    # decode_forward(defer_read=True): the FusedHandle of the enqueued step (no host tensors above; see MTPSpecEngine._decode_forward_deferred)
    handle: object = None


@dataclass
class FusedStep:
    """What ``verify(fused=True)`` hands back besides the ids: committed / counts in ENGINE rows and the DraftResult
    the fused step made for them (None after a fallback)."""

    committed: torch.Tensor
    counts: torch.Tensor
    drafts: "DraftResult | None" = None


@dataclass
class DraftResult:
    """propose_draft_tokens result: draft_token_ids int32 [B, K], num_valid int32 [B] in [0, K]."""

    draft_token_ids: torch.Tensor
    num_valid: torch.Tensor


# --------------------------------------------------------------------------------------------- #
# Pure host helpers (no device needed)
# --------------------------------------------------------------------------------------------- #
def _free_maint(*ts):
    """Deallocate the outputs of an MTP KV-maintenance forward. With QWEN36_MTP_KV_ONLY=1 its hidden is None (the layer
    stopped after the KV write); everything else is a real tensor."""
    for t in ts:
        if t is not None:
            ttnn.deallocate(t)


def anchor_selector(m, B, T):
    """One-hot [1,1,B,B*T] bf16 with a single 1.0 at (u, u*T + m[u]): ``sel @ feed`` gathers each
    user's accepted-prefix row of the verify feed. QWEN36_SPEC_HOSTSET (default 1): one index assignment."""
    if not spec_hostset_enabled():
        return _anchor_selector_old(m, B, T)
    sel = torch.zeros(1, 1, B, B * T, dtype=torch.bfloat16)
    rows = torch.arange(B)
    sel[0, 0, rows, rows * T + torch.as_tensor([int(x) for x in m], dtype=torch.int64)] = 1.0
    return sel


def _anchor_selector_old(m, B, T):
    sel = torch.zeros(1, 1, B, B * T, dtype=torch.bfloat16)
    for u in range(B):
        sel[0, 0, u, u * T + int(m[u])] = 1.0
    return sel


def _row_vision_of(eng):
    """engine row -> the packed vision rows of the multimodal request in it (absent for text rows): what a re-prefill of
    the row needs. Lives in the engine's __dict__ so an engine built without __init__ (tests) has it too."""
    return eng.__dict__.setdefault("_row_vision_map", {})


def _record_row_delta(eng, row, delta):
    """Store engine row ``row``'s M-RoPE delta (request-level: 0 for text, the grid's rope_delta for a multimodal prompt)
    where the verify / draft / join builders read it (model.set_slot_rope_delta). No-op for a model without it."""
    model = getattr(eng, "model", None)
    f = getattr(model, "set_slot_rope_delta", None)
    if f is not None and getattr(model, "_slot_rope_delta", None) is not None:
        f(int(row), 0 if delta is None else int(delta))


def join_warm_plan(fresh_now, tokens, pt):
    """Host plan of the join's MTP slot Tp-1 warm (QWEN36_SPEC_JOIN_FAST): for the fresh rows in ascending engine-row
    order, (rows, first tokens, slots Tp-1, page-table rows [R, nb] int32). Row i is exactly what the per-row
    ``_warm_mtp_last(_last_hidden[u], tokens[u, 0], Tp - 1, _mtp_pt_rows[u])`` call uses."""
    rows = sorted(fresh_now)
    toks = [int(tokens[u, 0]) for u in rows]
    slots = [int(fresh_now[u]) - 1 for u in rows]
    pt_rows = torch.as_tensor(pt)[rows].to(torch.int32).contiguous()
    return rows, toks, slots, pt_rows


def join_mtp_warm_host(fresh_now, tokens, pt, phys, B, scratch_block, rope_delta, table_rows):
    """Host inputs of the traced B-row MTP join warm (QWEN36_SPEC_JOIN_TRACE): (pack int32 [3, W], page table int32
    [B, nb]), in PHYSICAL rows (the order of the hidden buffers the trace concatenates). ``phys[u]`` = the physical row
    of fresh engine row u. Pack rows (uint32, as pack_u32_rows): 0 tokens [B]; 1 positions [B]; 2 RoPE table rows [B].
    A fresh row carries its first token, slot Tp-1 and its page-table row (join_warm_plan); every other physical row
    carries token 0, position 0 and the scratch block in every table entry, so its KV write lands in the scratch block.
    ``rope_delta``: a scalar (every row) or one M-RoPE delta per ENGINE row (a list indexed like ``fresh_now``: a fresh row
    takes its own, the padding rows 0).
    """
    rows, toks, slots, pt_rows = join_warm_plan(fresh_now, tokens, pt)
    tok, pos = [0] * B, [0] * B
    per_row = isinstance(rope_delta, (list, tuple, torch.Tensor))
    dl = [0] * B if per_row else int(rope_delta)
    table = torch.full((B, pt_rows.shape[1]), int(scratch_block), dtype=torch.int32)
    for i, u in enumerate(rows):
        p = int(phys[u])
        tok[p], pos[p], table[p] = toks[i], slots[i], pt_rows[i]
        if per_row:
            dl[p] = int(rope_delta[u])
    ridx = rope_row_index(pos, dl, table_rows)
    return pack_u32_rows([tok, pos, ridx.tolist()]), table


def last_row_selector(i, bucket):
    """One-hot [1,1,1,bucket] bf16 with a 1.0 at column i: ``sel @ feed`` picks row i of a bucket."""
    sel = torch.zeros(1, 1, 1, bucket, dtype=torch.bfloat16)
    sel[0, 0, 0, int(i)] = 1.0
    return sel


def accepted_plan(committed_tokens, committed_positions, accepted_counts, B, T):
    """accepted_counts -> (m, anchor_rows, pending, anchor_pos), each a list of B ints.

    m[u] = accepted_counts[u] - 1 (accepted drafts); anchor row = u*T + m[u] of the verify feed;
    pending[u] = committed_tokens[u, m[u]] (the bonus token, at position p+1); anchor_pos[u] = p =
    committed_positions[u, m[u]] - 1 (the slot whose KV the first draft step writes).
    QWEN36_SPEC_HOSTSET (default 1): the two lookups go through plain lists (no tensor index per row)."""
    m = [int(a) - 1 for a in accepted_counts]
    assert len(m) == B and all(0 <= x < T for x in m), f"accepted_counts must be in [1, {T}], got {m}"
    anchor_rows = [u * T + m[u] for u in range(B)]
    if (
        spec_hostset_enabled()
        and isinstance(committed_tokens, torch.Tensor)
        and isinstance(committed_positions, torch.Tensor)
    ):
        tl, pl = committed_tokens.tolist(), committed_positions.tolist()
        return m, anchor_rows, [int(tl[u][m[u]]) for u in range(B)], [int(pl[u][m[u]]) - 1 for u in range(B)]
    pending = [int(committed_tokens[u][m[u]]) for u in range(B)]
    anchor_pos = [int(committed_positions[u][m[u]]) - 1 for u in range(B)]
    return m, anchor_rows, pending, anchor_pos


def reseed_row_plan(committed_tokens, committed_positions, m, B, T):
    """Per-row (token, position, real) for the B*T-row reseed forward. Row u*T+i is REAL for i < m[u]
    (token committed_tokens[u, i] written at slot committed_positions[u, i] - 1) and PADDING otherwise
    (token 0, position 0, redirected to the scratch block by the caller)."""
    plan = []
    for u in range(B):
        for i in range(T):
            if i < int(m[u]):
                plan.append((int(committed_tokens[u][i]), int(committed_positions[u][i]) - 1, True))
            else:
                plan.append((0, 0, False))
    return plan


def reseed_rows_host(plan, pt, T, nb, scratch_block):
    """reseed_row_plan -> (tok [R,1] int32, pos [R] int32, pt_rows [R,nb] int32). Padding rows get position 0
    and an all-scratch page-table row; real row r takes pt[r // T] (``pt`` is not read when all rows pad)."""
    tok = torch.tensor([[t] for t, _, _ in plan], dtype=torch.int32)
    pos = torch.tensor([q for _, q, _ in plan], dtype=torch.int32)
    pt_rows = torch.full((len(plan), nb), int(scratch_block), dtype=torch.int32)
    for r, (_, _, real) in enumerate(plan):
        if real:
            pt_rows[r] = pt[r // T]
    return tok, pos, pt_rows


def reseed_rows_vec(committed_tokens, committed_positions, m, pt, T, nb, scratch_block, need_pt=True):
    """reseed_row_plan + reseed_rows_host in tensor ops (QWEN36_SPEC_HOSTSET): (tok [R,1] int32, pos [R] int32,
    pt_rows [R,nb] int32, real [R] bool), element for element what the two loops return (row u*T+i is real for
    i < m[u]; padding rows are token 0, position 0, an all-scratch table row; ``pt`` is not read when all pad).
    ``need_pt=False`` skips the [R,nb] table (pt_rows is None)."""
    ct, cp = torch.as_tensor(committed_tokens).to(torch.int64), torch.as_tensor(committed_positions).to(torch.int64)
    B = ct.shape[0]
    mm = torch.as_tensor([int(x) for x in m], dtype=torch.int64)
    real = (torch.arange(T)[None, :] < mm[:, None]).reshape(-1)
    zero = torch.zeros((), dtype=torch.int64)
    tok = torch.where(real, ct.reshape(-1), zero).to(torch.int32).reshape(B * T, 1)
    pos = torch.where(real, cp.reshape(-1) - 1, zero).to(torch.int32)
    if not need_pt:
        return tok, pos, None, real
    pt_rows = torch.full((B * T, nb), int(scratch_block), dtype=torch.int32)
    ridx = real.nonzero().reshape(-1)
    if ridx.numel():
        pt_rows[ridx] = torch.as_tensor(pt)[ridx // T].to(torch.int32)
    return tok, pos, pt_rows, real


def exact_topk(values, m, ids=None):
    """Top-``m`` per row under (value desc, token id asc), exact and deterministic. ``values`` [R, C] (any float
    dtype; -0.0 ties with +0.0, as torch.sort does; no NaN), ``ids`` [R, C] int token ids (None = the column
    index). Returns (values fp32 [R, m], ids int32 [R, m]). One torch.topk over a packed int64 key (the
    float's order-preserving integer image above, ``2**_TOPK_ID_BITS - 1 - id`` below), so ties never depend on
    torch.topk's own tie order: this is the plugin's fast-path tie rule (spec_accept._constrained_table)."""
    v = torch.as_tensor(values).to(torch.float32)
    R, C = v.shape
    v = torch.where(v == 0, torch.zeros_like(v), v).contiguous()
    key = v.view(torch.int32).to(torch.int64)
    key = key ^ ((key >> 63) & 0x7FFFFFFF)  # negative floats: flip the magnitude bits (int order == float order)
    span = 1 << _TOPK_ID_BITS
    if ids is None:
        id_key = (span - 1) - torch.arange(C, dtype=torch.int64)
    else:
        ids = torch.as_tensor(ids).to(torch.int64)
        if ids.shape != v.shape or bool((ids < 0).any()) or bool((ids >= span).any()):
            raise ValueError(f"exact_topk: ids must be [{R}, {C}] in [0, {span}), got {tuple(ids.shape)}")
        id_key = (span - 1) - ids
    key = (key << _TOPK_ID_BITS) | id_key
    top, pos = torch.topk(key, int(m), dim=1)
    out_ids = (span - 1) - (top & (span - 1))
    return v.gather(1, pos), out_ids.to(torch.int32)


def merge_shard_topk(values, local_ids, per_shard, m):
    """Host merge of the topk trace's readback (model._forward_verify_topk_body): ``values`` [R, D*m] bf16 and
    ``local_ids`` [R, D*m] (device d's block at columns d*m .. d*m+m-1, ids local to that vocab shard) ->
    (values fp32 [R, m], ids int32 [R, m]) = the exact global top-m per row under (value desc, id asc).
    Exact for stable AND unstable per-shard order: a token among the global top-m has fewer than m tokens
    above it in its own shard, so it is in that shard's list (note 26, 3.1). Ties truncated at a shard's cut
    are handled by the plugin's overflow rule (merged[m-1] >= the request's k-th value -> full logits).
    Raises when a local id is not below ``per_shard``: only a -1e30 pad lane can produce one, and a pad can
    only rank in the top-m of a padding row, which the caller never passes."""
    values = torch.as_tensor(values)
    local_ids = torch.as_tensor(local_ids).to(torch.int64)
    R, W = values.shape
    if local_ids.shape != (R, W) or W % m != 0:
        raise ValueError(f"merge_shard_topk: values {tuple(values.shape)}, ids {tuple(local_ids.shape)}, m={m}")
    if bool((local_ids >= per_shard).any()) or bool((local_ids < 0).any()):
        bad = local_ids[(local_ids >= per_shard) | (local_ids < 0)]
        raise ValueError(f"merge_shard_topk: local ids outside [0, {per_shard}) came back (pad lanes?): {bad[:8]}")
    gid = local_ids + (torch.arange(W, dtype=torch.int64) // m) * per_shard
    return exact_topk(values, m, gid)


def parked_rows_host(ring, e_prev, s, T):
    """A parked row's host state (see TPGatedDeltaNet.save_spec_user) whose last commit used ring slot s ->
    (ring slot 0 [D, Nv, Dk, Dv], E_prev [D, kc-1+T, C] shifted by s so a mi = 0 verify reads its rows s+1 ..)."""
    D = ring.shape[0] // T
    slot = ring.reshape(D, T, *ring.shape[1:])[:, s].contiguous()
    e = torch.zeros_like(e_prev)
    e[:, : e_prev.shape[1] - s] = e_prev[:, s:]
    return slot, e


def draft_step_plan(pending, p, K):
    """(pending tok [B,1] int32, [K] position vectors [B] int32): step k writes slot p[u] + k."""
    tok = torch.tensor([[int(t)] for t in pending], dtype=torch.int32)
    return tok, [torch.tensor([int(pu) + k for pu in p], dtype=torch.int32) for k in range(K)]


def _int_block(x, B, T, name):
    t = torch.as_tensor(x)
    if tuple(t.shape) != (B, T):
        raise ValueError(f"{name} must have shape [{B}, {T}], got {tuple(t.shape)}")
    return t.to(torch.int64)


def _int_vec(x, B, name):
    t = torch.as_tensor(x)
    if tuple(t.shape) != (B,):
        raise ValueError(f"{name} must have shape [{B}], got {tuple(t.shape)}")
    return t.to(torch.int64)


def sanitize_verify_block(tokens, start_pos, num_valid_drafts, K):
    """Validate a runner verify block [B, 1+K] -> (clean_tokens [B,T] int64, start0 [B] int64, active [B] bool).
    Active row (start_pos[u,0] >= 0): columns 0..n are real (tokens >= 0, consecutive positions); the padding
    tail gets tokens[u,0]. Inactive row: start_pos[u,0] < 0, n == 0.
    QWEN36_SPEC_HOSTFIX (default 1): checks and padding on plain lists (no tensor op per row); a block with any error
    is re-run through the per-row code so the raised error is the same."""
    if not spec_hostfix_enabled():
        return _sanitize_verify_block_old(tokens, start_pos, num_valid_drafts, K)
    T = K + 1
    tokens = torch.as_tensor(tokens)
    if tokens.ndim != 2 or tokens.shape[1] != T:
        raise ValueError(f"tokens must have shape [B, {T}], got {tuple(tokens.shape)}")
    B = tokens.shape[0]
    tokens = tokens.to(torch.int64)
    pos = _int_block(start_pos, B, T, "start_pos")
    nv = _int_vec(num_valid_drafts, B, "num_valid_drafts")
    tl, pl, nl = tokens.tolist(), pos.tolist(), nv.tolist()
    clean, start0, active = [], [], []
    for u in range(B):
        n, s0 = nl[u], pl[u][0]
        if s0 < 0:
            ok, row = n == 0, [0] * T
        else:
            ok = 0 <= n <= K and min(tl[u][: n + 1]) >= 0 and pl[u][: n + 1] == list(range(s0, s0 + n + 1))
            row = tl[u][: n + 1] + [tl[u][0]] * (T - n - 1)
        if not ok:
            return _sanitize_verify_block_old(tokens, start_pos, num_valid_drafts, K)  # raises the row's error
        clean.append(row)
        start0.append(max(s0, 0))
        active.append(s0 >= 0)
    if B == 0:
        return _sanitize_verify_block_old(tokens, start_pos, num_valid_drafts, K)
    return torch.tensor(clean, dtype=torch.int64), torch.tensor(start0, dtype=torch.int64), torch.tensor(active)


def _sanitize_verify_block_old(tokens, start_pos, num_valid_drafts, K):
    """Validate a runner verify block [B, 1+K] -> (clean_tokens [B,T] int64, start0 [B] int64, active [B] bool).
    Active row (start_pos[u,0] >= 0): columns 0..n are real (tokens >= 0, consecutive positions); the padding
    tail gets tokens[u,0]. Inactive row: start_pos[u,0] < 0, n == 0."""
    T = K + 1
    tokens = torch.as_tensor(tokens)
    if tokens.ndim != 2 or tokens.shape[1] != T:
        raise ValueError(f"tokens must have shape [B, {T}], got {tuple(tokens.shape)}")
    B = tokens.shape[0]
    tokens = tokens.to(torch.int64)
    pos = _int_block(start_pos, B, T, "start_pos")
    nv = _int_vec(num_valid_drafts, B, "num_valid_drafts")
    clean = tokens.clone()
    start0 = pos[:, 0].clone()
    active = start0 >= 0
    for u in range(B):
        n = int(nv[u])
        if not 0 <= n <= K:
            raise ValueError(f"row {u}: num_valid_drafts {n} outside [0, {K}]")
        if not active[u]:
            if n != 0:
                raise ValueError(f"row {u}: inactive row (start_pos -1) has num_valid_drafts {n}")
            clean[u] = 0
            start0[u] = 0
            continue
        real = tokens[u, : n + 1]
        if bool((real < 0).any()):
            raise ValueError(f"row {u}: negative token among real columns 0..{n}: {real.tolist()}")
        want = start0[u] + torch.arange(n + 1)
        if not torch.equal(pos[u, : n + 1], want):
            raise ValueError(f"row {u}: positions {pos[u, : n + 1].tolist()} are not consecutive from {int(start0[u])}")
        clean[u, n + 1 :] = tokens[u, 0]
    return clean, start0, active


def mask_unused_columns(ids, num_valid_drafts):
    """ids [B, T] with columns > num_valid_drafts[u] set to PLACEHOLDER_TOKEN_ID."""
    ids = ids.clone()
    nv = torch.as_tensor(num_valid_drafts).to(torch.int64).reshape(-1, 1)
    ids[torch.arange(ids.shape[1]).reshape(1, -1) > nv] = PLACEHOLDER_TOKEN_ID
    return ids


NO_DRAFT_STREAK = 16  # consecutive withheld offers (never verified with drafts) before a slot is marked no_draft


def note_withheld_drafts(offered, marked, speculated, streak, num_valid_drafts, limit=NO_DRAFT_STREAK):
    """Drafter-skip bookkeeping at a verify. ``offered``: slots given drafts by the last propose; ``marked``: slots
    whose drafts the runner withholds; ``speculated``: slots that ever verified with n > 0 since prefill;
    ``streak``: slot -> consecutive withheld offers. A row with n > 0 marks the slot speculated and resets its streak;
    an offered row with n == 0 bumps the streak. A never-speculated slot reaching ``limit`` is marked.
    Mutates ``offered``/``speculated``/``streak`` (offered is consumed); returns the new ``marked`` set."""
    nv = torch.as_tensor(num_valid_drafts).reshape(-1)
    marked = set(marked)
    if spec_hostfix_enabled():  # QWEN36_SPEC_HOSTFIX: one tolist() instead of a tensor index per row
        nv = nv.tolist()
    for u in range(len(nv)):
        if int(nv[u]) > 0:
            speculated.add(u)
            streak[u] = 0
        elif u in offered:
            streak[u] = streak.get(u, 0) + 1
            if u not in speculated and streak[u] >= limit:
                marked.add(u)
    offered.clear()
    return marked


def reset_draft_skip(engine, slot):
    """Forget all per-slot drafter-skip state (release / prefill)."""
    engine._offered.discard(slot)
    engine._no_draft.discard(slot)
    engine._speculated.discard(slot)
    engine._withheld_streak.pop(slot, None)


def propose_skip_plan(marked, B):
    """(decline, skip_all) for a propose: marked rows draft nothing; skip_all when every row is marked."""
    decline = [u in marked for u in range(B)]
    return decline, all(decline)


def spec_round_up(n, mult=32):
    return -(-int(n) // mult) * mult


def spec_table_width(max_seq_len, draft_len, block_size=64):
    """Page-table width (blocks) the engine needs: max_seq_len + K + 1 lookahead (no padding; fits vLLM's pool)."""
    return -(-(int(max_seq_len) + int(draft_len) + 1) // int(block_size))


def table_capacity(width_blocks, block_size):
    """Token capacity of the caller's table."""
    return int(width_blocks) * int(block_size)


def draft_rows_fit(anchor_pos, K, capacity):
    """Per row: may it draft? Its K drafter writes (p..p+K-1) and next verify window (p+1..p+1+K)
    must stay in capacity."""
    return [int(p) + 1 + K < capacity for p in anchor_pos]


def _prompt_ids_fast(prompt_ids):
    """QWEN36_PREFILL_HOST_TOLIST: (the prompt ids as Python ints, an int32 [1, Tp] row of them) with ONE tolist() and one
    dtype copy, equal to today's [int(t) for t in ...] and torch.tensor([ids], dtype=torch.int32); (None, None) for a
    floating / complex / bool input, which keeps the per-token path (int() truncation, bool -> 0/1)."""
    t = torch.as_tensor(prompt_ids).reshape(-1)
    if t.is_floating_point() or t.is_complex() or t.dtype == torch.bool:
        return None, None
    return t.tolist(), t.to(torch.int32, copy=True).reshape(1, -1)


def mtp_prefill_inputs(prompt_ids, chunk_start, bucket, page_table_row, block_size, scratch_block, buf_blocks):
    """Host inputs of the MTP drafter's KV warm-up over the prompt segment [chunk_start, chunk_start + bucket) of a
    ``bucket``-row prefill (the spec short trace's ``spec_inputs``): the same values _warm_mtp_chunk /
    Qwen36Model.ttnn_mtp_prefill_forward upload eagerly.

    Returns int32 torch (tokens [1, bucket], full table [1, buf_blocks], chunk table [1, bucket // block_size]):
    tokens: slot i (segment row i) fuses with prompt token chunk_start + i + 1, so row i holds
    prompt_ids[chunk_start + 1 + i] for i < n = min(bucket, Tp - 1 - chunk_start) and 0 past it (slot Tp-1 is left to
    _warm_mtp_last); chunk table: the user's blocks of the window, entries past ceil((chunk_start + n) / block_size)
    redirected to the MTP scratch block so padding rows never write K/V into real blocks; full table: the user's row
    (zero = null block past its allocation) padded to buf_blocks with the same window redirect. n may be 0 (a segment
    that only holds slot Tp-1): every row then pads to the scratch block."""
    Tp = len(prompt_ids)
    row = torch.as_tensor(page_table_row).reshape(1, -1).to(torch.int32)
    bpb = bucket // block_size
    blk0 = chunk_start // block_size
    assert chunk_start % block_size == 0 and bucket % block_size == 0
    width = max(buf_blocks, blk0 + bpb)
    if row.shape[1] < width:
        row = torch.cat([row, torch.zeros(1, width - row.shape[1], dtype=torch.int32)], dim=1)
    full = row.clone()
    toks = torch.zeros(1, bucket, dtype=torch.int32)
    n = max(0, min(bucket, Tp - 1 - chunk_start))
    if n > 0:
        toks[0, :n] = torch.as_tensor(prompt_ids[chunk_start + 1 : chunk_start + 1 + n], dtype=torch.int32)
    keep = -(-(chunk_start + n) // block_size) - blk0  # blocks of the window that hold real MTP rows
    full[0, blk0 + keep : blk0 + bpb] = int(scratch_block)
    chunk = full[:, blk0 : blk0 + bpb].clone()
    return toks, full[:, :buf_blocks].contiguous(), chunk.contiguous()


def verify_write_table(page_table, start_pos, num_valid_drafts, block_size):
    """[B, nb] table with each row's entries past the block of its last real column (start_pos + n) set to the
    null block 0, so padded verify columns write only that row's current block or the null block.
    QWEN36_SPEC_HOSTSET (default 1): one slice assignment per row instead of a whole-table boolean mask."""
    pt = torch.as_tensor(page_table).clone()
    last = torch.as_tensor(start_pos).reshape(-1).to(torch.int64)
    last = last + torch.as_tensor(num_valid_drafts).reshape(-1).to(torch.int64)
    if spec_hostset_enabled() and pt.dim() == 2 and last.shape[0] == pt.shape[0]:
        for u, c in enumerate((last // int(block_size)).tolist()):
            pt[u, max(c + 1, 0) :] = 0
        return pt
    cols = torch.arange(pt.shape[1]).reshape(1, -1)
    pt[cols > (last // int(block_size)).reshape(-1, 1)] = 0
    return pt


def slot_map_after_remap(phys, remap):
    """Runner slot -> engine row after a runner slot_remap: runner row i now holds the request that sat at runner
    row remap[i], and that request's state stays in engine row phys[remap[i]]."""
    return [phys[int(j)] for j in remap]


def slot_map_inverse(phys):
    """Engine row -> runner slot (the inverse permutation of a slot map)."""
    inv = [0] * len(phys)
    for slot, row in enumerate(phys):
        inv[int(row)] = slot
    return inv


def rows_to_engine(x, inv):
    """Scatter a runner-ordered per-row tensor into engine row order: out[phys[s]] = x[s] (``inv`` = slot_map_inverse)."""
    return torch.as_tensor(x)[torch.as_tensor(inv, dtype=torch.int64)]


def rows_to_runner(x, phys):
    """Gather an engine-ordered per-row tensor back into runner slot order: out[s] = x[phys[s]]."""
    return torch.as_tensor(x)[torch.as_tensor(phys, dtype=torch.int64)]


# --------------------------------------------------------------------------------------------- #
# Spec-decode memory cost (pure: config time, no device)
# --------------------------------------------------------------------------------------------- #
# Mirror of TPAttention._SPEC_SDPA_L1_FIT group counts per user (T -> groups); other T -> 1 group total.
_SPEC_SDPA_GROUPS_PER_USER = {4: 1, 8: 2, 12: 3}
_MASK_BUCKETS_DEFAULT = (128, 256, 512, 1024, 2048)  # Qwen36Model._PREFILL_MASK_BUCKETS


def _pad32(n):
    return -(-int(n) // 32) * 32


def _tile_bytes(shape, esize):
    """Per-device bytes of a TILE tensor: the last two dims round up to 32 (ttnn padded shape)."""
    s = [int(d) for d in shape]
    s[-2], s[-1] = _pad32(s[-2]), _pad32(s[-1])
    return math.prod(s) * esize


def _rm_bytes(shape, esize):
    return math.prod(int(d) for d in shape) * esize


def _cfg(tc, name, default=None):
    return tc.get(name, default) if isinstance(tc, dict) else getattr(tc, name, default)


def spec_memory_costs(
    text_config,
    tp,
    batch_size,
    draft_len,
    block_size=64,
    kv_dtype_bytes=2,
    nb=None,
    max_seq_len=None,
    num_layers=None,
    mask_buckets=None,
    buckets=None,
):
    """PER-DEVICE bytes of the MTP spec-decode state beyond plain decode. Pure: HF text config + tp only.

    Returns {"per_seq": {gdn_ring, gdn_conv_windows, drafter_hidden, total},   # x number of users B
             "fixed": {verify_io, mtp_scratch_block, engine_buffers, total},   # per engine
             "per_token": {mtp_kv, total}}                                     # per drafter KV token
    Counted: GDN fp32 per-token state ring and bf16 conv windows (_verify_win_buf, _spec_win_pad, _conv_win_buf),
    the engine's per-user _last_hidden, the verify trace's persistent inputs/outputs, the MTP cache's extra
    scratch block, the engine's anchor/selector buffers. NOT counted: rec_state, conv_states, base KV, the MTP
    KV blocks themselves (per_token x tokens), weights, trace-internal intermediates.
    Dims follow the model: TILE tensors pad their last two dims to 32; ROW_MAJOR are exact.
    ``fixed`` scales with B only through page tables / selectors / feed rows (all small). Page-table width nb =
    ``nb`` or ceil(max_seq_len / block_size); with neither, page tables count 0. ``num_layers`` overrides
    num_hidden_layers (truncated test models); ``mask_buckets`` the engine's last-row selector buckets.
    ``buckets``: the verify/drafter row buckets (QWEN36_SPEC_BUCKETS; None reads the env, default "1,2,4,8" = on, "8" = off = none beyond B). Each bucket
    b < B adds to ``fixed.verify_io`` its persistent trace outputs (normed, logits TILE, RM copy, shard logits; b*T rows, TILE pads
    to 32) and to ``engine_buffers`` its anchor hidden [1,1,b,dim/tp]. Per-bucket page-table slices are not persistent.
    """
    tp, B, K = int(tp), int(batch_size), int(draft_len)
    T, rows = K + 1, B * (K + 1)
    dim = int(_cfg(text_config, "hidden_size"))
    vocab = int(_cfg(text_config, "vocab_size"))
    n_heads = int(_cfg(text_config, "num_attention_heads"))
    n_kv = int(_cfg(text_config, "num_key_value_heads"))
    head_dim = int(_cfg(text_config, "head_dim", None) or dim // n_heads)
    rp = _cfg(text_config, "rope_parameters", None) or {}
    prf = rp.get("partial_rotary_factor", _cfg(text_config, "partial_rotary_factor", 1.0))
    rope_dim = int(head_dim * prf)
    nk = int(_cfg(text_config, "linear_num_key_heads", 16))
    nv = int(_cfg(text_config, "linear_num_value_heads", 32))
    dk = int(_cfg(text_config, "linear_key_head_dim", 128))
    dv = int(_cfg(text_config, "linear_value_head_dim", 128))
    kconv = int(_cfg(text_config, "linear_conv_kernel_dim", 4))
    assert nk % tp == 0 and nv % tp == 0 and dim % tp == 0, "GDN heads / hidden must divide by tp"
    nv_tp = nv // tp
    C = (nk * dk * 2 + nv * dv) // tp  # gdn_qkv_dim_tp
    n_local_kv = max(1, n_kv // tp)
    dim_frac = dim // tp
    L = int(num_layers if num_layers is not None else _cfg(text_config, "num_hidden_layers"))
    types = _cfg(text_config, "layer_types", None) or ["linear_attention"] * 3 + ["full_attention"]
    n_gdn = sum(1 for i in range(L) if types[i % len(types)] != "full_attention")
    bf, f32, i32 = 2, 4, 4

    ring = n_gdn * B * _tile_bytes([T * nv_tp, dk, dv], f32)  # [T*B*Nv, Dk, Dv]
    win_rows = [kconv - 1 + T, kconv] + ([T - 1] if T > 1 else [])  # _verify_win_buf, _conv_win_buf, _spec_win_pad
    if os.environ.get("QWEN36_VERIFY_SPEC_STEP", "1") == "1":  # gdn_spec_step keeps E_prev as a ping-pong pair
        win_rows.append(kconv - 1 + T)
    windows = n_gdn * B * sum(_tile_bytes([r, C], bf) for r in win_rows)
    hidden = B * _tile_bytes([1, 1, 1, dim_frac], bf)
    per_seq = {"gdn_ring": ring, "gdn_conv_windows": windows, "drafter_hidden": hidden}

    nb = int(nb) if nb else (-(-int(max_seq_len) // int(block_size)) if max_seq_len else 0)
    groups = B * _SPEC_SDPA_GROUPS_PER_USER[T] if T in _SPEC_SDPA_GROUPS_PER_USER else 1
    logits_w = tp * _pad32(vocab // tp) if vocab % tp == 0 else _pad32(vocab)  # vocab-sharded head gathers
    verify_io = (
        _rm_bytes([1, rows], i32)  # token_buf
        + _rm_bytes([rows], i32)  # kvpos_buf
        + _rm_bytes([rows, nb], i32)
        + _rm_bytes([B, nb], i32)
        + _rm_bytes([groups, nb], i32)  # page tables
        + 2 * _tile_bytes([1, rows, 1, rope_dim], bf)  # cos, sin
        + _rm_bytes([B * nv_tp], i32)  # state_idx
        + _tile_bytes([B, kconv - 1 + T, kconv - 1 + 2 * T], bf)  # conv_sel
        + _tile_bytes([1, 1, rows, logits_w], bf)
        + _rm_bytes([1, 1, rows, logits_w], bf)  # logits TILE + RM copy
        + _tile_bytes([1, 1, rows, dim_frac], bf)  # rows_out (drafter feed)
        + _rm_bytes([1, 1, rows], i32)  # ids_out
    )
    topk_m = verify_device_topk_m()
    if topk_m and vocab % tp == 0:  # QWEN36_VERIFY_DEVICE_TOPK: shard logits kept alive + the gathered candidates
        verify_io += (
            _tile_bytes([1, 1, rows, vocab // tp], bf)  # _vfy_shard_logits_out
            + _tile_bytes([1, 1, 32, tp * topk_m], bf)  # _vfy_topk_vals_out
            + _tile_bytes([1, 1, 32, tp * topk_m], i32)  # _vfy_topk_ids_out
        )
    if buckets is None:  # the flag; "8" means off at any engine width (as in MTPSpecEngine.__init__)
        raw = spec_buckets_flag()
        buckets = () if raw in ("", "8") else engine_spec_buckets(B)
    row_bks = sorted(int(b) for b in buckets if int(b) < B)
    for b in row_bks:  # bucket-local persistent trace outputs (the hand-off targets are the B-row outputs above)
        rb = b * T
        verify_io += (
            _tile_bytes([1, 1, rb, dim_frac], bf)  # normed
            + _tile_bytes([1, 1, rb, logits_w], bf)
            + _rm_bytes([1, 1, rb, logits_w], bf)  # logits TILE + RM copy
        )
        if topk_m and vocab % tp == 0:
            verify_io += _tile_bytes([1, 1, rb, vocab // tp], bf)  # shard logits
    block_bytes = 2 * n_local_kv * int(block_size) * head_dim * int(kv_dtype_bytes)
    sel_buckets = _MASK_BUCKETS_DEFAULT if mask_buckets is None else mask_buckets
    _big = int(os.environ.get("QWEN36_PREFILL_BIG_CHUNK", "0") or 0)
    sel_buckets = tuple(sel_buckets) + ((_big,) if _big else ())
    if _big == 8192 and prefill_masked_big_enabled():  # QWEN36_PREFILL_MASKED_BIG: the 4096 selector (8192 is _big)
        sel_buckets = sel_buckets + (4096,)
    engine = (
        sum(_tile_bytes([1, 1, b, dim_frac], bf) for b in row_bks)  # _hp_bufs
        + sum(_tile_bytes([1, 1, 1, b], bf) for b in sel_buckets)  # _last_sel
        + _rm_bytes([B, nb], i32)
        + B * _rm_bytes([1, nb], i32)  # _mtp_pt, _mtp_pt_rows
        + _tile_bytes([1, 1, B, dim_frac], bf)  # _hp_buf
        + _tile_bytes([1, 1, B, rows], bf)  # _anchor_sel
        + _tile_bytes([1, 1, 1, rows], bf)  # _eager_sel
        + (_tile_bytes([1, 1, rows, B], bf) + _tile_bytes([1, 1, rows, rows], bf) if B > 1 else 0)  # _join_take/keep
    )
    fixed = {"verify_io": verify_io, "mtp_scratch_block": block_bytes, "engine_buffers": engine}
    per_token = {"mtp_kv": 2 * n_local_kv * head_dim * int(kv_dtype_bytes)}
    for d in (per_seq, fixed, per_token):
        d["total"] = sum(d.values())
    return {"per_seq": per_seq, "fixed": fixed, "per_token": per_token}


_ELEM_BYTES = {ttnn.bfloat16: 2, ttnn.float32: 4, ttnn.int32: 4, ttnn.uint32: 4, ttnn.uint16: 2, ttnn.uint8: 1}


def _elem_bytes(dtype):
    """Bytes per element; block-float and unknown dtypes raise rather than guess."""
    if dtype not in _ELEM_BYTES:
        raise ValueError(f"no element size for dtype {dtype} (block-float or unsupported)")
    return _ELEM_BYTES[dtype]


def _device_bytes(t):
    """Allocated per-device bytes of a ttnn tensor: padded shape volume x element size (0 for None)."""
    if t is None:
        return 0
    shape = getattr(t, "padded_shape", None) or t.shape
    return math.prod(int(d) for d in shape) * _elem_bytes(t.dtype)


class _CorePlan:
    """Host plan of one ``propose`` (see MTPSpecEngine._propose_core_plan)."""

    def __init__(self, **kw):
        self.drafts = None
        self.recorded = False
        self.dpt = None
        self.__dict__.update(kw)


class FusedHandle:
    """A deferred fused step (``decode_forward(defer_read=True)``): everything its read and its later engine bookkeeping need,
    captured at enqueue. ``tip_id`` (int, increasing per fused step) is what ``chain_tip`` names. ``wait()`` (reader thread) blocks on
    the step's device event and returns ``(accepted_token_ids [B,1+K] int32, accepted_counts [B] int32, p0 [B] int32)`` in
    runner slots (p0 negative on padding rows); it makes no device call other than the event wait and mutates only this handle
    (its lock and result cache). A *ready* handle (``device`` False) stands for a step that ran synchronously."""

    def __init__(self, engine, tip_id, kind):
        self._eng, self.tip_id, self.kind = engine, int(tip_id), kind
        self._lock = threading.Lock()
        self._data = self._readout = None  # per-handle result cache (engine rows / runner slots)
        self.device, self.ready, self.chainable = True, False, True
        self.event = self.res_t = self.aux_t = self.ids_t = self.union = self.audit = None
        self.dr_ts, self.res_rows = [], 2
        self.active = self.declined = self.cmask = self.prev = self.succ = None
        self.nvv = (
            self.nv_own
        ) = self.tokens = self.start_host = self.mi_host = self.aux = self.pt = self.sampling = None
        self.lo = self.hi = self.exact = self.fin = self.draft_result = self.perm = None
        self.parity, self.cap, self.skip, self.finished = 0, 0, [], False
        self.gen = None  # per-engine-row generations at creation (see MTPSpecEngine._row_gens)
        self.bucket = None  # verify row bucket of the step (QWEN36_SPEC_BUCKETS; None while the flag is off)

    def wait(self):
        return self._eng._def_wait(self)


class _ProposePlan:
    """Host plan of one ``propose_draft_tokens`` (see MTPSpecEngine._propose_plan)."""

    def __init__(self, **kw):
        self.skip_all = False
        self.core = None
        self.decline = None
        self.keep = None
        self.__dict__.update(kw)


def _chunk_prog_of(eng):
    """engine row -> (e, digest of prompt[:e], page-table signature): see MTPSpecEngine._chunk_prog."""
    return eng.__dict__.setdefault("_chunk_progress", {})


def chunk_exact_resume_enabled():
    """QWEN36_SPEC_CHUNK_EXACT_RESUME (default 1): chunked prefill resumes exactly at the previous chunk end (Option B);
    0 = Option A (snapshot at e-128, recompute [e-128, e))."""
    return os.environ.get("QWEN36_SPEC_CHUNK_EXACT_RESUME", "1") == "1"


class _SnapRun:
    """QWEN36_PREFILL_INTRACE_SNAP: the save points of ONE single-segment spec fast prefill and their saves. The replay
    loops ask inside(cs, rows) for the points strictly inside a replay (staged as in-replay snapshots) and call
    after(cs, rows, inside) right after it: snapshot k of the replay goes to the prefix cache from the snapshot buffers,
    a point at the replay's last row from the bound state, in increasing position order (the legacy planner's order,
    with the same low-priority / end-of-prompt marks)."""

    def __init__(self, model, cache, tokens_row, points):
        self.model, self.cache, self.tokens = model, cache, tokens_row
        self.points = sorted(int(p) for p in points)
        self._set = frozenset(self.points)
        self.low = frozenset(getattr(points, "low", ()))
        self.end = frozenset(getattr(points, "end", ()))
        self.saved = []
        self.n_inside = 0

    def inside(self, cs, rows):
        return [p for p in self.points if cs < p < cs + rows]

    def after(self, cs, rows, inside):
        for k, p in enumerate(inside):
            self._save(p, self.model.gdn_snap_copy_fn(k))
            self.n_inside += 1
        if cs + rows in self._set:
            self._save(cs + rows, None)

    def _save(self, p, copy_fn):
        kw = {} if copy_fn is None else {"copy_fn": copy_fn}
        self.cache.save(self.tokens, p, low_prio=p in self.low, protect=p in self.end, **kw)
        self.saved.append(p)


class MTPSpecEngine:
    """MTP speculative decode engine over B users (see module docstring)."""

    _dr_hostset = False  # class default (engines built without __init__ in tests); set by _alloc_drafter_inputs

    def __init__(self, model, batch_size, draft_len):
        assert model.mtp is not None, "model has no MTP head (has_mtp / mtp.* weights?)"
        assert model.num_devices > 1, "MTPSpecEngine is TP-only for now"
        self.model = model
        self.mesh = model.mesh_device
        self.args = model.args
        self.vocab = model.args.vocab_size
        self.B = int(batch_size)
        self.K = int(draft_len)
        self.T = self.K + 1
        assert self.B == self.args.max_batch_size, (
            f"batch_size {self.B} != model max_batch_size {self.args.max_batch_size}; the verify trace's "
            f"row count is baked in at capture"
        )
        assert self.B * self.T <= MAX_SPEC_ROWS, (
            f"B={self.B} users x T={self.T} rows = {self.B * self.T} > {MAX_SPEC_ROWS}: the verify is one "
            f"decode tile. Lower K or the batch."
        )
        # Verify runs the fused GDN op; plain decode must use the same math or greedy near-ties flip. Under
        # QWEN36_VERIFY_SPEC_STEP=1 the verify runs gdn_spec_step and plain decode the gdn_decode_step op (one
        # kernel family), so set_gdn_fused_decode(True) is a no-op there (model.set_gdn_fused_decode).
        assert model.gdn_fused_decode or model.gdn_verify_spec_step, (
            "MTPSpecEngine needs model.set_gdn_fused_decode(True) before construction: verify runs the "
            "fused GDN op, and decode must use the same math"
        )
        self.mtp = model.mtp
        self._gdn = [layer.attention for layer in model.layers if not layer.is_full_attention]
        # One-hot selects are matmuls and must be exact: HiFi4 + fp32 acc (packer_l1_acc off: it rounds in bf16).
        self._anchor_cc = ttnn.init_device_compute_kernel_config(
            self.mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self._timing = bool(int(os.environ.get("QWEN36_SPEC_TIMING", "0")))
        self.last_verify_timing = None  # (device_s, readback_s) of the last verify when timing is on
        self.mtp_extra_steps = 0  # drafter forwards spent on KV maintenance (reseed)

        self.prefill_prepared = False
        self.prefill_recorded = False
        # debug/A-B knob: 1 = read prompts eagerly even when the chunk program is recorded
        self.recorded_prefill = os.environ.get("QWEN36_SPEC_EAGER_PREFILL", "0") != "1"
        self.prepared = False  # decode prepared
        self.captured = False  # verify trace recorded
        self._prep_pt = None  # fitted host table from prepare_prefill (verify prepare reuses it)
        self._last_cap_blocks = None  # caller's real table width of the most recent verify
        self.nb = None  # caller's table width: verify trace, drafter, MTP staging
        self.nb_prefill = None  # nb rounded up to 32: only the recorded prefill chunk program
        self.block_size = None
        self._dim_frac = self.args.dim // model.num_devices
        self._feed_dtype = ttnn.bfloat16
        self._scratch_block = None
        self._last_hidden = []  # [B] persistent [1,1,1,dim/tp]: base feed row of slot Tp-1
        self._last_sel = {}  # bucket -> persistent one-hot selector [1,1,1,bucket]
        self._mtp_pt = None  # persistent [B, nb] MTP page table (draft + batched reseed)
        self._mtp_pt_rows = []  # persistent [1, nb] per user (eager prompt warm / slot Tp-1 write / eager reseed)
        self._hp_buf = None  # persistent anchor hidden [1,1,B,dim/tp]
        self._anchor_sel = None  # persistent one-hot [1,1,B,B*T]
        self._eager_sel = None  # persistent one-hot [1,1,1,B*T]: eager reseed row gather
        self._join_take = None  # persistent one-hot [1,1,B*T,B]: seed feed row u -> verify-feed row u*T
        self._join_keep = None  # persistent one-hot [1,1,B*T,B*T]: verify-feed rows kept by a join
        self.eager_reseed_len = EAGER_RESEED_PROMPT_LEN  # B == 1 anchors past this reseed eagerly
        self._vfeed = None  # the verify trace's own output rows; valid until the next replay
        self._topk_logits = None  # [B, T, vocab] fp32 host logits of a topk-mode seed / join step (fetch_verify_logits)
        # A/B + safety switch: 1 = run the drafter eagerly even when its trace is recorded
        self.recorded_drafter = os.environ.get("QWEN36_SPEC_EAGER_DRAFTER", "0") != "1"
        self._dr_bufs = []  # persistent drafter-trace inputs (all of the _dr_* tensors below)
        self._dr_tok0 = None  # [B,1] uint32 pending tokens
        self._dr_pos, self._dr_cos, self._dr_sin = [], [], []  # per draft step k: [B] int32, [1,B,1,rd] bf16 x2
        self._dr_rs_tok = (
            self._dr_rs_pos
        ) = self._dr_rs_pt = self._dr_rs_cos = self._dr_rs_sin = None  # reseed rows (B*T)
        # QWEN36_SPEC_HOSTSET (+ _TRACE), decided when the drafter inputs are allocated: ONE packed uint32 input buffer
        # (_dr_pack) replaces tok0 / pos / rs_tok / rs_pos / anchor selector / cos / sin (16 H2D copies -> 1); the trace
        # slices it, looks cos/sin up from the RoPE tables and builds the anchor one-hot from an index (_dr_eye).
        self._dr_hostset = False
        self._dr_pack = None
        self._dr_eye = None
        self._dr_rs_cache = None  # (real-row mask, table or None) the reseed page-table buffer holds
        # QWEN36_SPEC_FUSED (stage S1): the device accept (FusedGreedyDevice bound to the live buffers), its trace, the
        # DraftResult of the last fused step with the commit it was made for, and the host mirror of pt_rows_in
        self._fa = None
        self._fa_trace = None
        self._fused_cache = None
        self._fa_pt_host = None
        # stage S2: the SAMPLED accept (FusedSampledDevice sharing the S1 buffers) and its trace; audit boundary count
        self._fs = None
        self._fs_trace = None
        self.fused_boundary_diffs = 0
        self.fused_fallbacks = {}  # reason -> count of fused_sample steps that ran the argmax path
        self.fused_steps = 0  # fused_sample verify steps (eligible + fallback)
        self.row_delta_on = (
            spec_row_delta_enabled()
        )  # QWEN36_SPEC_ROW_DELTA: per-row rope delta words, no mixed decline
        self.row_delta_mixed_steps = 0  # fused steps that ran with different live-row deltas (flag on)
        # QWEN36_SPEC_CHAIN (stage S3c): two ChainDevice instances (the S1 / S2 result buffer they read), their traces, the state of
        # the last fused step a chain step may continue (_chain_prev), counters
        self._ch_g = self._ch_s = self._ch_trace_g = self._ch_trace_s = None
        self._ch_standins = ()
        self._chain_prev = self._chain_in = self._chain_cand = self._chain_ev = None
        self._chain_dropped = self._chain_why = None
        self.chain_steps = self.chain_audits = 0
        self.chain_breaks = {}  # reason -> count of fused steps that could have chained and ran the host-staged path
        self._dr_trace = None  # trace id: anchor gather + batched reseed + K draft steps
        # QWEN36_SPEC_BUCKETS (default "1,2,4,8" = on, "8" = off): the verify / drafter row buckets, the drafter trace id per bucket < B that exists
        raw = spec_buckets_flag()
        self._buckets = (
            (self.B,) if raw in ("", "8") else tuple(engine_spec_buckets(self.B))
        )  # "8" = off at any engine width
        self._dr_traces = {}  # bucket b < B -> drafter trace id of b rows (filled when a per-bucket capture exists)
        self._dr_outs = []  # K persistent [1,1,B] uint32 ids outputs of that trace
        self._last_pt = None  # fitted host page table of the most recent verify
        self._fresh = {}  # slot -> prompt length, prefilled and awaiting the seed verify
        self._active = set()  # slots holding a live request
        self._parked = {}  # slot -> per-GDN-layer host state of a live row left out of a verify
        self._finishing = set()  # engine rows freed by decode_forward(finishing_slots) and not prefilled since
        self._prompts = {}  # slot -> (prompt ids, page-table row) of a fresh slot, for a re-prefill
        self._fresh_dirty = set()  # fresh slots left out of a seed, which overwrote their prefill state
        self._row_active = None  # [B] bool: rows that held a request in the last verify (None = all)
        self._offered = set()  # slots the last propose gave drafts (num_valid > 0)
        self._no_draft = set()  # slots whose drafts the runner withholds: propose skips them
        self._speculated = set()  # slots that verified with n > 0 since prefill (never marked)
        self._withheld_streak = {}  # slot -> consecutive offers answered with n == 0
        self.no_draft_streak = NO_DRAFT_STREAK
        self.skipped_propose_calls = 0  # proposes skipped because every row was marked
        # Host slot map (module docstring): 1 = a runner slot_remap permutes _phys only; 0 = the GDN state moves on
        # the device (remap_spec_slots per layer) and the per-slot host state is renumbered, as before.
        self.host_slotmap = os.environ.get("QWEN36_SPEC_SLOTMAP", "1") != "0"
        self._phys = list(range(self.B))  # runner slot -> engine row
        self._phys_perm = None  # (phys, inv) int64 index tensors while the map is not the identity
        # Fast join (module docstring): 1 = a fresh row is seeded in place and verified inside the normal replay;
        # 0 = the eager T = 1 seed forward (_seed / _join), as before.
        self.fast_join = spec_fast_join_enabled()
        # Fast prefill (module docstring): 1 = prompts replay the spec short traces / chunk trace / slot-write traces;
        # 0 = the eager prefill_for_spec path (B > 1) or the B == 1 chunk-trace path, as before.
        self.fast_prefill = spec_fast_prefill_enabled()
        self._kc = self._gdn[0].K if self._gdn else 4  # GDN conv kernel size (E_prev carry rows 1 .. kc-1)
        self._fj_sel = None  # persistent one-hot [B, kc-1+T, kc-1+T + kc-1]: fresh_row_select, shared by every layer
        # QWEN36_SPEC_JOIN_TRACE: persistent fresh-row mask (fresh_row_mask), the two seed traces (one per conv-window
        # parity), the buffers they were captured against, and the replay / fallback counters
        self._fj_mask = None
        self._seed_trace = None
        self._seed_trace_bufs = None
        self._fj_mask_dims = None
        self.join_trace_replays = 0
        self.join_trace_fallbacks = {}  # reason -> count of joins whose GDN seed ran the eager loop
        # QWEN36_SPEC_JOIN_TRACE, MTP part of the seed: ONE fixed B-row MTP decode (slot Tp-1 warm) replayed per join.
        # Inputs: packed uint32 tokens / positions / RoPE rows, a [B, nb] page table (the row mask: a row that is not
        # fresh holds the scratch block), the B per-row hidden buffers concatenated in the trace in the fixed physical
        # order they had at capture (_mtpw_hidden; _mtpw_phys maps id(buffer) -> physical row).
        self._mtpw_pack = self._mtpw_pt = self._mtpw_trace = self._mtpw_rope = None
        self._mtpw_hidden, self._mtpw_phys = (), {}
        self.join_mtp_trace_replays = 0

    # --------------------------------------------------------------------- #
    # Host helpers
    # --------------------------------------------------------------------- #
    # ---- QWEN36_SPEC_BUCKETS: the row bucket of a step, its drafter trace, the lowest-free-row placement
    def _bucketed(self):
        """True when more than one bucket is configured (flag != "8"); every bucket hook is a no-op otherwise."""
        return self._buckets is not None and len(self._buckets) > 1

    def _bucket_has_trace(self, b):
        """Only B-row traces exist until the per-bucket captures land (_dr_traces: bucket -> drafter trace id)."""
        return b == self.B or b in (self._dr_traces or {})

    def _bucket_of(self, live_rows):
        """The bucket a step that runs engine rows ``live_rows`` replays: pick_bucket over the configured buckets, B when that
        bucket has no trace yet. Pure (no counters). Flag off: B."""
        if not self._bucketed():
            return self.B
        b = pick_bucket(self._buckets, live_rows)
        return b if self._bucket_has_trace(b) else self.B

    @staticmethod
    def _active_rows(active):
        return torch.nonzero(torch.as_tensor(active, dtype=torch.bool).reshape(-1)).reshape(-1).tolist()

    def _bucket_step(self, live_rows):
        """Per step: the bucket the step runs into ``_vb`` (+ whether a missing trace forced B). Flag off: nothing.
        QWEN36_SPEC_BUCKET_HOLD: the engine keeps a current bucket ``_bk_cb``. A wanted bucket above it switches up at once. A
        wanted bucket below it switches down only after HOLD consecutive such steps (to the largest wanted of the streak, so the
        new bucket fits every step of it), or earlier, free, when the step breaks the chain anyway (_bucket_free_note).
        HOLD = 0: every down-shift is immediate (the bucket of the live rows)."""
        if not self._bucketed():
            return
        chosen = pick_bucket(self._buckets, live_rows)
        wanted = self._bucket_of(live_rows)
        self._vb_fell = chosen != wanted
        cb, hold = self._bk_cb, spec_bucket_hold()
        self._bk_cand = None
        if cb is None or wanted >= cb:
            cb = wanted
            self._bk_streak = self._bk_smax = 0
        else:
            self._bk_streak += 1
            self._bk_smax = max(self._bk_smax, wanted)
            if self._bk_streak >= hold:
                cb = self._bk_smax
                self.bucket_downs_hold += 1
                self._bk_streak = self._bk_smax = 0
            else:
                self._bk_cand = self._bk_smax  # the free switch target, if this step breaks the chain anyway
        self._bk_cb = self._vb = cb

    def _bucket_free_note(self, prev_bucket, kw=None):
        """The step breaks the chain for a reason other than ``bucket_change``: a pending down-shift happens now, at no extra cost
        (``kw["bucket"]`` and the step count follow), and a bucket that differs from the previous step's counts as a free switch.
        """
        if not self._bucketed():
            return
        cand = self._bk_cand
        if cand is not None:
            old = self._vb
            self._bk_cb = self._vb = cand
            self._bk_cand = None
            self._bk_streak = self._bk_smax = 0
            self.bucket_downs_free += 1
            if kw is not None:
                kw["bucket"] = cand
                bs = self.__dict__.setdefault("bucket_steps", {})
                if bs.get(old, 0) > 0:
                    bs[old] -= 1
                    bs[cand] = bs.get(cand, 0) + 1
        if prev_bucket is not None and prev_bucket != self._vb:
            self.bucket_free_switch += 1

    def _bucket_kw(self, kw):
        """Put the step's bucket into the verify kwargs and count the step (flag off: kw unchanged)."""
        if self._bucketed():
            kw["bucket"] = self._vb
            bs = self.__dict__.setdefault("bucket_steps", {})
            bs[self._vb] = bs.get(self._vb, 0) + 1
            if self._vb_fell:
                self.bucket_fallbacks += 1
        return kw

    def _dr_trace_for(self, b):
        """Drafter trace id for bucket ``b`` (B, None or no per-bucket capture: the B-row trace)."""
        if b is None or b == self.B or not self._dr_traces:
            return self._dr_trace
        return self._dr_traces.get(b, self._dr_trace)

    def _bucket_note_text(self):
        """Tail of the periodic fused line: steps per bucket, fallbacks, row swaps (flag on only)."""
        if not self._bucketed():
            return ""
        bs = self.__dict__.get("bucket_steps", {})
        return (
            "; buckets "
            + " ".join(f"{b}:{bs.get(b, 0)}" for b in self._buckets)
            + f", fallbacks {self.bucket_fallbacks}, row swaps {self.row_swaps}"
            + f", down-shifts hold {self.bucket_downs_hold} free {self.bucket_downs_free}"
            + f", free switches {self.bucket_free_switch}"
        )

    def _pending_rows(self):
        """Engine rows that a deferred step still in flight (not finished) describes."""
        rows, h, seen = set(), self._def_last, 0
        while h is not None and seen < 64:
            if not h.finished and h.active is not None:
                rows |= set(self._active_rows(h.active))
            h, seen = h.prev, seen + 1
        return rows

    def _claim_low_row(self, slot):
        """QWEN36_SPEC_BUCKETS, host slot map only: before a NEW request is prefilled into runner slot ``slot``, swap the slot map
        entries of two FREE runner slots so the request lands on the lowest free engine row (small buckets stay usable). A live
        row never moves (a re-prefill of a live request does nothing), and nothing moves while a deferred step in flight describes
        either of the two rows. Counts ``row_swaps``."""
        if not self._bucketed() or not self.host_slotmap:
            return
        # Rows with pending chunk progress (Option B: a non-final chunk's snapshot pin + _last_hidden[row] belong to a live
        # request between its chunks) are reserved: never free for another slot, and a slot sitting on one never moves.
        free = set(range(self.B)) - self._active - self._chunk_reserved_rows()
        swap = plan_row_swap(int(slot), free, self._phys)
        if swap is None:
            return
        a, b = swap
        if {self._phys[a], self._phys[b]} & self._pending_rows():
            return
        logger.info(
            f"[spec_engine] row swap: slot {a} row {self._phys[a]} <-> slot {b} row {self._phys[b]} (lowest free row)"
        )
        self._set_slot_map(apply_row_swap(self._phys, swap))
        self.row_swaps += 1

    def _chunk_reserved_rows(self):
        """Engine rows whose request is between two chunks of one chunked prefill (``_chunk_progress`` keys). Invariant: such a
        row is never claimed / swapped by _claim_low_row, so the request keeps row, snapshot pin and hidden row."""
        return set(_chunk_prog_of(self))

    def _chunk_cache(self):
        return getattr(getattr(self, "model", None), "_prefix_cache", None)

    def _chunk_unpin(self, row):
        c = self._chunk_cache()
        if c is not None and hasattr(c, "unpin"):
            c.unpin(row)

    def _chunk_bump(self, name):
        c = self._chunk_cache()
        if c is not None and hasattr(c, "bump"):
            c.bump(name)

    def _note_prefill_done(self):
        """Every QWEN36_PREFIX_SUMMARY_EVERY (default 1000) prefills: one INFO line with the chunk / prefix-cache counters
        (visible without debug; includes the true resume metrics of GdnPrefixStateCache.record_resume)."""
        n = self.__dict__.get("_n_prefills", 0) + 1
        self.__dict__["_n_prefills"] = n
        every = max(1, int(os.environ.get("QWEN36_PREFIX_SUMMARY_EVERY", "1000")))
        if n % every == 0:
            c = self._chunk_cache()
            if c is not None and hasattr(c, "summary"):
                logger.info(
                    f"[spec_engine] chunked-prefill summary after {n} prefills: {c.summary()} "
                    f"row_swaps={getattr(self, 'row_swaps', 0)} chunk_exact_misses={getattr(self, 'chunk_exact_misses', 0)}"
                )

    def _set_slot_map(self, phys):
        self._phys = [int(r) for r in phys]
        if self._phys == list(range(self.B)):
            self._phys_perm = None
        else:
            inv = slot_map_inverse(self._phys)
            self._phys_perm = (torch.tensor(self._phys, dtype=torch.int64), torch.tensor(inv, dtype=torch.int64))

    def _to_rows(self, *xs):
        """Runner-ordered per-row tensors -> engine row order (the identity map passes them through, no copy).
        Scattered this way: tokens, start_pos, active, num_valid_drafts, accepted_counts (-> mi_prev, state_blk_idx,
        conv_sel, or the gdn_spec_step ctrl page under QWEN36_VERIFY_SPEC_STEP=1) and the page-table rows (verify
        write table, drafter / MTP tables, reseed rows). The GDN ring,
        E_prev, rec_state, conv taps, _last_hidden, _vfeed, _mtp_pt_rows and every per-slot set stay in engine rows."""
        if self._phys_perm is None:
            return xs
        inv = self._phys_perm[1]
        return tuple(None if x is None else torch.as_tensor(x)[inv] for x in xs)

    def _to_slots(self, *xs):
        """Engine-ordered per-row outputs -> runner slot order (ids, logits, drafts, num_valid; no copy at the
        identity). In logits mode this gathers [B, T, V] fp32: up to 32 MB at B=8, T=4, so only off the identity."""
        if self._phys_perm is None:
            return xs
        phys = self._phys_perm[0]
        return tuple(None if x is None else torch.as_tensor(x)[phys] for x in xs)

    def _slot_of(self, row):
        """Runner slot of an engine row (for messages)."""
        return row if self._phys_perm is None else int(self._phys_perm[1][row])

    def _fit(self, pt):
        pt = self.model._fit_spec_page_table(pt, self.nb)
        assert pt.shape[0] == self.B, f"page_table has {pt.shape[0]} rows, engine has B={self.B}"
        return pt

    def _ints(self, x, shape):
        t = torch.as_tensor(x).to(torch.int64)
        assert (
            t.numel() == shape[0] * shape[1] if len(shape) == 2 else t.numel() == shape[0]
        ), f"expected {shape}, got {tuple(t.shape)}"
        return t.reshape(*shape)

    def _stage(self, host, dev, dtype, layout):
        """Copy a torch tensor into a persistent device tensor (address preserved)."""
        h = ttnn.from_torch(
            host, dtype=dtype, layout=layout, device=None, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh)
        )
        ttnn.copy_host_to_device_tensor(h, dev)

    def _stage_table(self, host, dev, dtype, layout):
        """_stage for a page table that mostly repeats from step to step. QWEN36_SPEC_HOSTFIX (default 1): skip the
        copy when ``dev`` already holds this exact table. Every write to a buffer must go through here, since the
        cache is the host copy of what that buffer holds (keyed by the buffer, dropped when it is freed)."""
        if not spec_hostfix_enabled():
            return self._stage(host, dev, dtype, layout)
        cache = self.__dict__.setdefault("_stage_cache", {})
        hit = cache.get(id(dev))
        if hit is not None and hit[0] is dev and hit[1].shape == host.shape and torch.equal(hit[1], host):
            return
        self._stage(host, dev, dtype, layout)
        cache[id(dev)] = (dev, host.clone())

    # class defaults of the join-seed trace state (set per instance in __init__; engines built without it read these)
    _fj_mask = _seed_trace = _seed_trace_bufs = _fj_mask_dims = None
    join_trace_replays = 0
    # the traced MTP part of the join seed (_mtpw_*): packed inputs, page table, hidden buffers, trace, id -> row map
    _mtpw_pack = _mtpw_pt = _mtpw_trace = _mtpw_rope = None
    _mtpw_hidden, _mtpw_phys = (), {}
    join_mtp_trace_replays = 0

    # QWEN36_SPEC_BUCKETS class defaults (set per instance in __init__): None = the flag is off, every hook does nothing
    _buckets = _dr_traces = _vb = None
    _hp_bufs = None  # bucket b < B -> persistent anchor hidden [1,1,b,dim/tp] of its drafter body
    _dr_bks = ()  # the buckets < B whose verify + drafter traces this engine builds (descending), see _bucket_plan
    _dr_standins = ()  # warm-pass stand-ins of the shared K ids outputs, freed after the captures
    _addr_snap = None
    _vb_fell = False  # the last _bucket_step chose a bucket that has no trace yet and ran at B
    bucket_fallbacks = row_swaps = 0
    # QWEN36_SPEC_BUCKET_HOLD state: current bucket, the streak of steps wanting less, its largest wanted bucket, the pending free
    # down-shift target of this step; counters: down-shifts by cause (hold reached / free with another break), switches absorbed
    _bk_cb = _bk_cand = None
    _bk_streak = _bk_smax = 0
    bucket_downs_hold = bucket_downs_free = bucket_free_switch = 0
    _dr_rs_dirty = False  # the device wrote _dr_rs_pt (fused step): _dr_rs_cache no longer mirrors it
    # class defaults of the fused-step state (set per instance in __init__; engines built without it read these)
    _fa = _fa_trace = _fused_cache = _fa_pt_host = None
    _fs = _fs_trace = None
    _fs_standin = ()
    fused_steps = fused_boundary_diffs = 0
    fused_sampled_steps = fused_audited_steps = 0
    # QWEN36_SPEC_CHAIN (stage S3c) class defaults: with the flag off every one of these stays None / 0 and no hook does anything
    _ch_g = _ch_s = _ch_trace_g = _ch_trace_s = None
    _ch_standins = ()
    _chain_prev = _chain_in = _chain_cand = _chain_ev = _chain_dropped = _chain_why = None
    chain_steps = chain_audits = 0
    # deferred serving (stage S4b): the newest FusedHandle (chain_tip must name it), the context verify() reads, tip counter
    _def_last = _def_ctx = None
    _tip_seq = 0
    _row_gen = None  # per-engine-row generation, bumped by prefill / release (see _row_gens)
    fused_greedy_checked = 0

    def _alloc_host_replicated(self, host, dtype, layout):
        return ttnn.from_torch(
            host,
            dtype=dtype,
            layout=layout,
            device=self.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _stage_mtp_tables(self, pt):
        """Re-stage the persistent MTP page tables from the fitted host table [B, nb]."""
        self._stage_table(pt, self._mtp_pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        for u in range(self.B):
            self._stage_table(pt[u : u + 1].contiguous(), self._mtp_pt_rows[u], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)

    def _stage_mtp_tables_rows(self, pt, rows):
        """QWEN36_SPEC_JOIN_FAST: _stage_mtp_tables for the fresh ``rows`` only. The full [B, nb] table and the other
        rows' tables are staged by the next propose before anything reads them (_stage_table keys on content)."""
        for u in rows:
            self._stage_table(pt[u : u + 1].contiguous(), self._mtp_pt_rows[u], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)

    # --- per-engine-row M-RoPE deltas (multimodal rows keep their own delta; text rows 0) ---
    def _row_deltas(self, default=None):
        """The rows' M-RoPE deltas: a list [B] of ints (engine row u = device row u; what the prefill recorded with
        model.set_slot_rope_delta), or, for a model without the per-row vector (a test fake), the scalar
        ``rope.rope_delta`` (``default`` when given), as before."""
        f = getattr(self.model, "_spec_row_deltas", None)
        if f is not None:
            return f(self.B)
        return int(self.model.rope.rope_delta if default is None else default)

    def _has_row_deltas(self):
        return getattr(getattr(self, "model", None), "_spec_row_deltas", None) is not None

    def _row_delta(self, u):
        d = self._row_deltas()
        return int(d[u]) if isinstance(d, list) else int(d)

    def _mtp_rd_kw(self, rows):
        """kwargs of model.ttnn_mtp_decode_forward for the engine ``rows`` (None = delta 0): their own deltas; {} for a model
        without the per-row vector (the call is then what it always was)."""
        if not self._has_row_deltas():
            return {}
        return {"rope_deltas": [0 if u is None else self._row_delta(u) for u in rows]}

    def _row_kw(self, u):
        """``row=`` keyword of _warm_mtp_last for engine row ``u`` (only a model with per-row deltas needs it)."""
        return {"row": u} if self._has_row_deltas() else {}

    def _pos_rd_kw(self, per_row):
        """Positional ``deltas`` argument of model._rope_tp_cos_sin_decode_torch for a vector of B * ``per_row`` rows
        (user-major): each row's delta repeated ``per_row`` times; {} for a model without the per-row vector."""
        if not self._has_row_deltas():
            return {}
        return {"deltas": [x for x in self._row_deltas() for _ in range(per_row)]}

    def _live_rope_delta(self, active):
        """(delta, mixed) for a fused / chain step: the delta word of the device-built rope rows. It is one scalar for the
        whole step, so it is the COMMON delta of the live rows; ``mixed`` is True when the live rows differ (the step must
        then take the host-built path, which has a delta per row). No live row: (0, False)."""
        d = self._row_deltas()
        if not isinstance(d, list):
            return int(d), False
        live = {d[u] for u in range(self.B) if bool(active[u])}
        if len(live) > 1:
            return 0, True
        return (live.pop() if live else 0), False

    def _row_delta_flag(self):
        """QWEN36_SPEC_ROW_DELTA, read once (engine init; an engine built without __init__ reads it on first use)."""
        v = self.__dict__.get("row_delta_on")
        if v is None:
            v = self.__dict__["row_delta_on"] = spec_row_delta_enabled()
        return v

    def _step_rope_delta(self, active):
        """(rope_delta, common, mixed) for a fused / chain step. ``rope_delta`` is what the aux / overlay builders get:
        flag on and a per-row model: the B own deltas of every engine row (live or not); else the common delta (an int).
        ``common`` goes to row 5 col 1; ``mixed`` = the live rows differ."""
        common, mixed = self._live_rope_delta(active)
        if self._row_delta_flag() and self._has_row_deltas():
            return [int(x) for x in self._row_deltas()], common, mixed
        return common, common, mixed

    def _warm_mtp_last_rows(self, fresh_now, tokens, pt):
        """QWEN36_SPEC_JOIN_FAST: _warm_mtp_last for every fresh row. One row: the old single-row call. Two or more:
        ONE drafter decode over R rows (hidden rows concatenated, tokens / slots Tp-1 / page-table rows in the
        ``join_warm_plan`` order). The rows are independent users on distinct blocks, so the KV written per row is the
        one the per-row calls write; the logits are not computed (need_logits=False) and the output is dropped."""
        rows, toks, slots, pt_rows = join_warm_plan(fresh_now, tokens, pt)
        if len(rows) == 1:
            u = rows[0]
            self._warm_mtp_last(self._last_hidden[u], toks[0], slots[0], self._mtp_pt_rows[u], **self._row_kw(u))
            return
        mesh = self.mesh
        parts = [ttnn.to_layout(self._last_hidden[u], ttnn.ROW_MAJOR_LAYOUT) for u in rows]
        cat = ttnn.concat(parts, dim=2)  # [1,1,R,dim/tp]
        hid = ttnn.to_layout(cat, ttnn.TILE_LAYOUT)
        pt_tt = ttnn.from_torch(
            pt_rows,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        _, h_next = self.model.ttnn_mtp_decode_forward(
            hid, toks, slots, pt_tt, need_logits=False, **self._mtp_rd_kw(rows)
        )
        _free_maint(*parts, cat, hid, pt_tt, h_next)

    def _join_warm_warmup(self):
        """QWEN36_SPEC_JOIN_FAST: compile the R-row batched warm (R = 2 .. B) before capture, so a join never compiles a
        program. Every row writes slot 0 of the MTP scratch block (write-only garbage)."""
        if not spec_join_fast_enabled():
            return
        B = self.B
        pt = torch.full((B, self.nb), self._scratch_block, dtype=torch.int32)
        tokens = torch.zeros(B, self.T, dtype=torch.int64)
        for R in range(2, B + 1):
            self._warm_mtp_last_rows({u: 1 for u in range(R)}, tokens, pt)
        ttnn.synchronize_device(self.mesh)

    def _mtp_warm_prepare(self):
        """QWEN36_SPEC_JOIN_TRACE: allocate the persistent inputs of the traced MTP join warm (scratch-only content:
        token 0, position 0, every table entry the scratch block) and run its body once so every program is compiled
        before capture forbids cache misses."""
        if not (self.fast_join and spec_join_fast_enabled() and spec_join_trace_enabled()):
            return
        B, rope = self.B, self.model.rope
        rope.decode_rm_tables()  # the resident RoPE tables exist before the trace is captured
        self._mtpw_rope = (rope.rope_delta, rope.max_seq_len)
        zero = torch.zeros(B, dtype=torch.int32)
        self._mtpw_pack = self._alloc_host_replicated(pack_u32_rows([zero] * 3), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self._mtpw_pt = self._alloc_host_replicated(
            torch.full((B, self.nb), self._scratch_block, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT
        )
        self._mtpw_hidden = tuple(self._last_hidden)
        self._mtp_warm_body()
        ttnn.synchronize_device(self.mesh)

    def _mtp_warm_body(self):
        """The traced MTP join warm: ONE B-row MTP decoder-layer decode, no lm_head. Inputs are persistent buffers: the
        pack (rows as join_mtp_warm_host; sliced, typecast and RoPE-looked-up in the trace as the drafter does, via
        model._dl_rope_cos_sin), the page table, and the hidden buffers of ``_mtpw_hidden`` concatenated in that order.
        alias_kv_write=False: the rows are independent users (the batched eager warm does the same). The output is
        dropped; the MTP KV of each row's slot is the effect."""
        B, pk, model = self.B, self._mtpw_pack, self.model

        def seg(row):
            return ttnn.slice(pk, [row, 0], [row + 1, B])

        tok = ttnn.reshape(seg(0), (B, 1))
        p_u32 = seg(1)
        p_i32 = ttnn.typecast(p_u32, ttnn.int32)
        ttnn.deallocate(p_u32)
        pos = ttnn.reshape(p_i32, (B,))
        idx = seg(2)
        cos, sin = model._dl_rope_cos_sin(idx, B)
        parts = [ttnn.to_layout(h, ttnn.ROW_MAJOR_LAYOUT) for h in self._mtpw_hidden]
        cat = ttnn.concat(parts, dim=2)  # [1,1,B,dim/tp]
        hid = ttnn.to_layout(cat, ttnn.TILE_LAYOUT)
        _, hn = self.mtp.forward_decode(
            hid, tok, pos, cos, sin, self._mtpw_pt, need_logits=False, alias_kv_write=False, spec_n_users=1
        )
        _free_maint(tok, pos, idx, cos, sin, *parts, cat, hid, hn)

    def _capture_mtp_warm_trace(self):
        """QWEN36_SPEC_JOIN_TRACE, inside the guarded scope: capture the MTP join warm once, against the hidden buffers
        as they are ordered now (the trace bakes their addresses; _mtpw_phys records id(buffer) -> physical row)."""
        self._mtpw_trace = None
        if self._mtpw_pack is None or not spec_join_trace_enabled():
            return
        self._mtpw_hidden = tuple(self._last_hidden)
        self._mtpw_phys = {id(h): p for p, h in enumerate(self._mtpw_hidden)}
        self._mtpw_trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        self._mtp_warm_body()
        ttnn.end_trace_capture(self.mesh, self._mtpw_trace, cq_id=0)
        ttnn.synchronize_device(self.mesh)
        logger.info("[spec_engine] join MTP warm trace captured")

    def _mtp_warm_trace_why(self):
        """None when the MTP part of this join can replay _mtpw_trace; else why it runs eagerly: "flag off" (not
        counted), "mtp no trace", "mtp stale buffers" (the pack / page table, or a hidden buffer that is not one of
        the captured ones or sits at another row than recorded). Without allocated inputs the feature is not in use
        and nothing is counted ("")."""
        if not spec_join_trace_enabled():
            return "flag off"
        if self._mtpw_pack is None:
            return ""
        if self._mtpw_trace is None:
            return "mtp no trace"
        h = self._last_hidden
        if len(h) != len(self._mtpw_hidden) or any(self._mtpw_phys.get(id(x)) is None for x in h):
            return "mtp stale buffers"
        if len({id(x) for x in h}) != len(h) or any(self._mtpw_hidden[self._mtpw_phys[id(x)]] is not x for x in h):
            return "mtp stale buffers"
        return None

    def _replay_mtp_warm(self, fresh_now, tokens, pt):
        """Stage the pack and the page table, enqueue the MTP warm trace (no host sync)."""
        phys = {u: self._mtpw_phys[id(self._last_hidden[u])] for u in fresh_now}
        # the rows' own deltas now (the scalar captured at trace prep goes stale at the next multimodal prefill)
        pack, table = join_mtp_warm_host(
            fresh_now,
            tokens,
            pt,
            phys,
            self.B,
            self._scratch_block,
            self._row_deltas(self._mtpw_rope[0]),
            self._mtpw_rope[1],
        )
        self._stage(pack, self._mtpw_pack, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self._stage_table(table, self._mtpw_pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.execute_trace(self.mesh, self._mtpw_trace, cq_id=0, blocking=False)
        self.join_mtp_trace_replays += 1

    def _check_scratch(self, pt):
        assert int(pt.max()) < self._scratch_block, (
            f"page table names block {int(pt.max())}, but the MTP cache's scratch block is "
            f"{self._scratch_block}; the MTP cache needs one block past every block the tables use"
        )

    # --------------------------------------------------------------------- #
    # Lifecycle
    # --------------------------------------------------------------------- #
    def prepare(self, page_table):
        """Warmup phase 1 (prefill + decode). Idempotent per table width."""
        self.prepare_prefill(page_table)
        self.prepare_decode()

    def capture(self):
        """Warmup phase 2: record the prefill chunk trace (B == 1) and the verify trace. Idempotent."""
        self.record_prefill()
        self.capture_decode()

    def prepare_prefill(self, page_table):
        """Allocate persistent engine buffers and compile every prefill program (eager warm prompts; B == 1 also
        the recorded chunk program). Idempotent per table width."""
        model, B, T = self.model, self.B, self.T
        pt0 = torch.as_tensor(page_table)
        width = int(pt0.shape[-1])
        if self.prefill_prepared:
            assert width == self.nb, f"engine prepared for nb={self.nb}, got a table of width {width}"
            return
        t0 = time.perf_counter()
        assert not self.captured and getattr(model, "_vfy_trace_id", None) is None, "prepare after capture"
        # The B=1 GDN prefill scratch's first bind can free spec buffers, so it must come first.
        if B > 1:
            model.warmup_gdn_slot_ops()

        self.nb = width  # must stay <= the cache block count (verify / drafter paged_update_cache)
        self.nb_prefill = spec_round_up(width)  # the recorded chunk program needs a multiple of 32
        pt = self._fit(pt0)
        self._prep_pt = pt
        self.block_size = int(self.mtp.attention.paged_k.shape[-2])
        self._scratch_block = int(self.mtp.attention.paged_k.shape[0]) - 1
        self._check_scratch(pt)
        self._last_hidden = [
            ttnn.zeros(
                [1, 1, 1, self._dim_frac],
                device=self.mesh,
                dtype=self._feed_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for _ in range(B)
        ]
        big = getattr(model, "_prefill_big_chunk", 0)
        self._last_sel = {
            b: self._alloc_host_replicated(
                torch.zeros(1, 1, 1, b, dtype=torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT
            )
            for b in tuple(model._PREFILL_MASK_BUCKETS)
            + ((4096,) if big == 8192 and prefill_masked_big_enabled() else ())
            + ((big,) if big else ())
        }
        self._mtp_pt = self._alloc_host_replicated(pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        self._mtp_pt_rows = [
            self._alloc_host_replicated(pt[u : u + 1].contiguous(), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT) for u in range(B)
        ]
        self._anchor_warmup(self._dim_frac, self._feed_dtype)

        cap = width * self.block_size
        # Compile every masked + no-mask bucket program and KV fill width (B=1 scratch bound for B>1).
        buckets = [b for b in model._PREFILL_MASK_BUCKETS if b <= cap]
        if buckets:
            prev = model._bind_gdn_prefill_scratch() if B > 1 else None
            try:
                model.warmup_prefill_masked_buckets(pt[0:1], buckets=buckets)
            finally:
                if prev is not None:
                    model._unbind_gdn_prefill_scratch(prev)
        warm_lens = [L for L in _WARM_PROMPT_LENS if L + T + 1 <= cap]
        for L in warm_lens:  # dummy prompts through the real (eager) prefill path on engine row 0
            self._prefill_row(0, [1] * L, pt[0:1], _cap_blocks=width)
            self._fresh.pop(0, None)
            self._active.discard(0)

        pt_prefill = model._fit_spec_page_table(pt[0:1], self.nb_prefill)
        if B == 1:  # allocate + compile the recorded chunk program (re-runs masked-bucket warmup itself)
            model.prepare_prefill_trace_chunked(self.mesh, pt_prefill, chunk_size=2048)
        if self.fast_prefill:
            # QWEN36_SPEC_FAST_PREFILL=1: allocate + compile the chunk program (B > 1: on the B=1 GDN scratch the
            # requests run on) and the spec short traces of every bucket. Allocation only here; record_prefill captures.
            prev = model._bind_gdn_prefill_scratch() if B > 1 else None
            try:
                if B > 1:
                    model.prepare_prefill_trace_chunked(self.mesh, pt_prefill, chunk_size=2048)
                # QWEN36_PREFILL_BIG_CHUNK: the big chunk set, its drafter warm and last-row capture compiled on a real
                # big-chunk output (the replay loop runs them per big chunk; nothing may compile after capture).
                prep_big = getattr(model, "prepare_prefill_trace_chunked_big", None)  # absent on test fakes
                if prep_big is not None:
                    prep_big(self.mesh, pt_prefill, warm_cb=lambda h: self._warm_big_chunk(h, pt[0:1]))
                # vLLM's null block absorbs the base's padded K/V (the MTP padding rows go to the MTP scratch block).
                model.prefill_trash_block = 0
                # QWEN36_PREFILL_MASKED_BIG: the masked 4096 / 8192 spec short traces join the default buckets in this
                # one prepare call (all persistent inputs allocated and every program warmed before any capture).
                extra = getattr(model, "masked_big_buckets", None)  # absent on test fakes
                extra = tuple(extra(pt_prefill)) if extra is not None else ()
                if extra:
                    model.prepare_prefill_traces_short(
                        self.mesh, pt_prefill, buckets=tuple(model.short_prefill_trace_buckets()) + extra, spec=True
                    )
                else:
                    model.prepare_prefill_traces_short(self.mesh, pt_prefill, spec=True)
            finally:
                if prev is not None:
                    model._unbind_gdn_prefill_scratch(prev)

        self.prefill_prepared = True
        logger.info(
            f"[spec_engine] prefill prepared B={B} K={self.K} nb={self.nb} block={self.block_size} "
            f"warm_prompts={warm_lens} fast_prefill={int(self.fast_prefill)} in {time.perf_counter() - t0:.1f}s"
        )

    def record_prefill(self):
        """Record the 2048-token chunk prefill trace (B == 1, or any B under QWEN36_SPEC_FAST_PREFILL=1, where the spec
        short traces and the GDN slot-write traces are captured too; otherwise B > 1 prefill stays eager). Idempotent.
        """
        assert self.prefill_prepared, "record_prefill before prepare_prefill"
        if self.prefill_recorded:
            return
        model = self.model
        if self.fast_prefill:
            prev = model._bind_gdn_prefill_scratch() if self.B > 1 else None
            try:
                model.record_prefill_trace_chunked(self.mesh)
                rec_big = getattr(model, "record_prefill_trace_chunked_big", None)  # absent on test fakes
                if rec_big is not None:
                    rec_big(self.mesh)
                model.record_prefill_traces_short(self.mesh)
            finally:
                if prev is not None:
                    model._unbind_gdn_prefill_scratch(prev)
            if self.B > 1:
                fmts = {dn._conv_fmt for dn in self._gdn}
                if fmts == {"both"}:
                    model.capture_gdn_slot_write_traces(conv_fmt="both")  # skipped (logged) when not applicable
                else:
                    logger.info(f"[spec_engine] GDN slot-write traces not captured: conv formats {sorted(fmts)}")
        elif self.B == 1:
            model.record_prefill_trace_chunked(self.mesh)
        self.prefill_recorded = True

    def prepare_decode(self):
        """Prepare the verify trace and warm the drafter programs. Idempotent."""
        assert self.prefill_prepared, "prepare_decode before prepare_prefill"
        if self.prepared:
            return
        model, B, T = self.model, self.B, self.T
        assert not self.captured and getattr(model, "_vfy_trace_id", None) is None, "prepare_decode after capture"
        # a B == 1 prefill trace replay may overwrite buffers allocated after it was recorded
        assert not (B == 1 and self.prefill_recorded), "prepare_decode after record_prefill"
        t0 = time.perf_counter()
        # Compile the seed's programs (T = 1 forward, feed-row scatter and copy) before any capture.
        _, f = model.seed_spec_step([0] * B, [0] * B, self._prep_pt)
        r0, r1 = self._seed_rows(f), self._seed_rows(f)
        ttnn.copy(r0, r1)
        for t in (r0, r1):
            ttnn.deallocate(t)
        model.prepare_verify_trace(self._prep_pt, T, warm_positions=[0] * B)
        self._dr_bks = self._bucket_plan()
        if (
            self._dr_bks
        ):  # every verify bucket program is compiled here, before any capture (capture forbids cache misses)
            model.prepare_verify_buckets(self._dr_bks)
        for dn in self._gdn:
            # The seed's sync_conv_win from the packed conv format (the gdn_decode_step op is the plain decode, e.g.
            # QWEN36_VERIFY_SPEC_STEP=1): compile its unpack / copies before capture_decode forbids cache misses.
            dn.warmup_packed_sync()
        if self.fast_join:
            # QWEN36_SPEC_FAST_JOIN=1: the per-row ring writes at every offset, the whole-slot write, the window select
            # and both taps routes, compiled now (the ring and window pair exist since prepare_verify_trace).
            from models.demos.blackhole.qwen36.tt.gdn.tp import fresh_row_mask, fresh_row_select

            self._fj_sel = self._alloc_host_replicated(
                fresh_row_select(range(B), B, T, self._kc), ttnn.bfloat16, ttnn.TILE_LAYOUT
            )
            if self._gdn and spec_join_trace_enabled():
                # QWEN36_SPEC_JOIN_TRACE: the mask buffer (shared by every layer, so every layer must agree on Nv, Dk);
                # warmup compiles the masked seed for both window parities before any capture
                dims = {(dn.Nv, dn.Dk) for dn in self._gdn}
                assert len(dims) == 1, f"join trace: GDN layers disagree on (Nv, Dk): {sorted(dims)}"
                self._fj_mask_dims = dims.pop()
                if self._fj_mask is not None:
                    ttnn.deallocate(self._fj_mask)
                self._fj_mask = self._alloc_host_replicated(
                    fresh_row_mask([], B, *self._fj_mask_dims), ttnn.float32, ttnn.TILE_LAYOUT
                )
            for dn in self._gdn:
                dn.warmup_fast_seed(self._fj_sel, self._fj_mask)
            ttnn.synchronize_device(self.mesh)
        if B > 1:  # compile the join programs (per-user seed, feed merge)
            self._join_take = self._alloc_host_replicated(
                torch.zeros(1, 1, B * T, B, dtype=torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT
            )
            self._join_keep = self._alloc_host_replicated(
                torch.zeros(1, 1, B * T, B * T, dtype=torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT
            )
            for dn in self._gdn:
                dn.seed_spec_rows(range(0, B, 2))
                dn.seed_spec_rows(range(1, B, 2))
                # Also with the host slot map: the per-user ring / E_prev slices this compiles are the ones
                # save_spec_user (parking) reuses after capture.
                dn.remap_spec_slots(list(range(B))[::-1])  # a full reversal slices every row offset once
                if B % 2 == 1:  # odd B: the reversal fixes the middle row; a cyclic shift moves every row
                    dn.remap_spec_slots([(i + 1) % B for i in range(B)])
            z = ttnn.zeros(
                [1, 1, B * T, self._dim_frac],
                device=self.mesh,
                dtype=self._feed_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            rows_w = model.spec_feed_rows(z)
            if rows_w is not z:
                ttnn.deallocate(z)
            self._merge_seed_rows(f, [0], rows_w)
            ttnn.deallocate(rows_w)
            ttnn.synchronize_device(self.mesh)
        ttnn.deallocate(f)

        # Drafter programs.
        zrow = ttnn.zeros(
            [1, 1, 1, self._dim_frac],
            device=self.mesh,
            dtype=self._feed_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._warm_mtp_last(zrow, 0, 0, self._mtp_pt_rows[0])
        ttnn.deallocate(zrow)
        self._reseed_warmup(B * T, self._dim_frac, self._feed_dtype)
        self._eager_warmup()
        self._join_warm_warmup()
        self._mtp_warm_prepare()
        self._draft_warmup([0] * B, self._hp_buf, [0] * B)
        if self.recorded_drafter:
            self._alloc_drafter_inputs()
            z = ttnn.zeros(
                [1, 1, B * T, self._dim_frac],
                device=self.mesh,
                dtype=self._feed_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            # Warm feed = spec_feed_rows of a zero DRAM bf16 TILE residual: the same op/config that produces the
            # verify trace's _vfy_rows_out, so the body compiles for the exact capture-time input.
            feed = self.model.spec_feed_rows(z)
            if feed is not z:
                ttnn.deallocate(z)
            for o in self._drafter_body(feed):  # compiles every program of the trace body (scratch-only inputs)
                ttnn.deallocate(o)
            if self._dr_bks:  # each drafter bucket body once, handing its ids over to stand-ins of the shared outputs
                self._dr_standins = tuple(
                    ttnn.zeros(
                        [1, 1, B, 1],
                        device=self.mesh,
                        dtype=ttnn.uint32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    for _ in range(self.K)
                )
                for b in self._dr_bks:
                    self._drafter_body(feed, b, list(self._dr_standins))
            ttnn.synchronize_device(self.mesh)
            ttnn.deallocate(feed)
            if spec_fused_enabled():
                self._prepare_fused()

        ttnn.synchronize_device(self.mesh)
        self.prepared = True
        logger.info(
            f"[spec_engine] decode prepared in {time.perf_counter() - t0:.1f}s "
            f"(slot remaps: {'host slot map' if self.host_slotmap else 'device GDN remap, QWEN36_SPEC_SLOTMAP=0'}; "
            f"fresh rows: {'seeded in the verify replay (QWEN36_SPEC_FAST_JOIN=1)' if self.fast_join else 'eager T=1 seed forward'})"
        )

    def _prepare_fused(self):
        """QWEN36_SPEC_FUSED: allocate the device accept bound to the LIVE buffers (verify token buffer, drafter pack, reseed
        page table; the verify ids output is a trace output, bound in capture_decode) and run its body eagerly once so
        every program is compiled before capture forbids cache misses."""
        model = self.model
        if not (self.recorded_drafter and self._dr_hostset and getattr(model, "_vfy_sharded", False)):
            logger.warning(
                "[spec_engine] QWEN36_SPEC_FUSED=1 needs the recorded packed drafter and the sharded verify argmax; "
                "fused_sample steps will run the argmax path"
            )
            return
        W = int(self._dr_pack.shape[-1])
        self._fa = FusedGreedyDevice.allocate(
            self.mesh,
            self.B,
            self.K,
            self.nb,
            W,
            tok_in=model._vfy_token_buf,
            pack_out=self._dr_pack,
            rs_pt_out=self._dr_rs_pt,
        )
        self._fa.body()  # writes the live pack / rs_pt from zero inputs: the mirrors below forget them
        ttnn.synchronize_device(self.mesh)
        self._prepare_fused_sampled(W)
        self._dr_rs_dirty = True
        self._fa_pt_host = None
        self._prepare_chain()

    def _prepare_chain(self):
        """QWEN36_SPEC_CHAIN (stage S3c): allocate the chain device (tt/spec_chain.py) bound to the LIVE verify token buffer, packed
        aux buffer and the fused aux (shared by S1 / S2), once per result buffer (S1 ``fa.result_out`` and, when the sampled accept
        exists, S2 ``fs.result_out``), with stand-in draft inputs (the live ones are the drafter trace's outputs, bound in
        capture_decode), and compile the bodies eagerly. Any shape mismatch leaves chain mode off (host-staged steps as before).
        """
        if not spec_chain_enabled():
            return
        model, fa, fs = self.model, self._fa, self._fs
        if not (
            getattr(model, "_vfy_hostset", False)
            and getattr(model, "_vfy_fused", False)
            and getattr(model, "_vfy_aux_buf", None) is not None
        ):
            logger.warning(
                "[spec_engine] QWEN36_SPEC_CHAIN=1 needs the packed verify aux buffer (QWEN36_SPEC_HOSTSET) and the fused GDN op; chain off"
            )
            return
        kw = dict(aux_prev_in=fa.aux_in, tok_out=model._vfy_token_buf, vaux_out=model._vfy_aux_buf)
        Nv = int(model._vfy_Nv)
        ch_g = ch_s = None
        try:
            ch_g = ChainDevice.allocate(self.mesh, self.B, self.K, Nv, result_in=fa.result_out, **kw)
            if fs is not None:
                ch_s = ChainDevice.allocate(self.mesh, self.B, self.K, Nv, result_in=fs.result_out, **kw)
        except AssertionError as e:
            logger.warning(f"[spec_engine] chain mode disabled: {e}")
            return
        for ch in (ch_g, ch_s):
            if ch is not None:
                ch.body()  # compiles every program (stand-in drafts, zero inputs); writes the live token / aux buffers
        ttnn.synchronize_device(self.mesh)
        model._vfy_aux_staged = None  # the body overwrote the device words the host mirror describes
        self._ch_g, self._ch_s = ch_g, ch_s
        self._ch_standins = tuple(t for ch in (ch_g, ch_s) if ch is not None for t in ch.draft_ins)

    def _prepare_fused_sampled(self, W):
        """Stage S2: allocate the sampled accept (FusedSampledDevice) over the S1 buffers (ids stand-in, token buffer, aux, pt
        rows, drafter pack, reseed table are SHARED; only samp_in, result_out [3, W] and the sampled body are new) with stand-in
        union buffers (the live ones are the topk trace outputs, bound in capture_decode), and compile its body eagerly. Needs
        the verify topk trace (QWEN36_VERIFY_DEVICE_TOPK) and TP * M <= 1024; otherwise sampled fused steps take the
        full-logits host walk (reason "no_sampled_trace")."""
        model, fa = self.model, self._fa
        if not (spec_fused_sampled_enabled() and getattr(model, "_vfy_topk", False)):
            logger.info("[spec_engine] fused sampled accept off (QWEN36_SPEC_FUSED_SAMPLED=0 or no verify topk trace)")
            return
        TP, M = int(model.num_devices), int(model._vfy_topk_m)
        try:
            self._fs = FusedSampledDevice.allocate(
                self.mesh,
                self.B,
                self.K,
                self.nb,
                W,
                TP,
                M,
                int(model.vocab_size) // TP,
                ids_in=fa.ids_in,
                tok_in=model._vfy_token_buf,
                pack_out=self._dr_pack,
                rs_pt_out=self._dr_rs_pt,
                aux_in=fa.aux_in,
                pt_rows_in=fa.pt_rows_in,
            )
        except AssertionError as e:
            logger.warning(f"[spec_engine] fused sampled accept disabled (TP={TP}, M={M}): {e}")
            self._fs = None
            return
        self._fs_standin = (
            self._fs.union_vals_in,
            self._fs.union_ids_in,
        )  # freed at shutdown (the live ones are trace outputs)
        self._fs.body()  # compiles every program of the sampled body (stand-in union / ids, zero inputs)
        ttnn.synchronize_device(self.mesh)

    def _capture_fused(self):
        """Inside capture_decode's guarded scope, after the verify trace exists: bind its ids output and capture the
        accept trace. A verify trace without the [1,1,1,32] ids row (unsharded argmax) disables the fused step."""
        fa = self._fa
        try:
            fa.bind_ids(self.model._vfy_ids_out)
        except AssertionError as e:
            logger.warning(f"[spec_engine] fused accept disabled: {e}")
            self._fa = None
            return
        self._fa_trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        fa.body()
        ttnn.end_trace_capture(self.mesh, self._fa_trace, cq_id=0)
        ttnn.synchronize_device(self.mesh)
        self._dr_rs_dirty = True
        logger.info("[spec_engine] fused accept trace captured (greedy walk + drafter inputs on device)")
        self._log_trace_memory("accept")
        self._capture_fused_sampled()
        self._capture_chain()

    def _capture_chain(self):
        """QWEN36_SPEC_CHAIN, inside the guarded scope after the fused traces: bind the live drafter id outputs and capture the chain
        trace of each variant (S1 always, S2 when its sampled accept trace exists)."""
        if self._ch_g is None:
            return
        try:
            for ch in (self._ch_g, self._ch_s):
                if ch is not None:
                    ch.bind_draft_ins(self._dr_outs)
        except AssertionError as e:
            logger.warning(f"[spec_engine] chain mode disabled: {e}")
            self._ch_g = self._ch_s = None
            return
        self._ch_trace_g = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        self._ch_g.body()
        ttnn.end_trace_capture(self.mesh, self._ch_trace_g, cq_id=0)
        if self._ch_s is not None and self._fs is not None and self._fs_trace is not None:
            self._ch_trace_s = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            self._ch_s.body()
            ttnn.end_trace_capture(self.mesh, self._ch_trace_s, cq_id=0)
        else:
            self._ch_s = None
        ttnn.synchronize_device(self.mesh)
        logger.info(
            "[spec_engine] chain trace captured (device builds the next verify's tokens / aux from the previous fused step"
            f"{'; greedy + sampled variants' if self._ch_trace_s is not None else '; greedy variant'})"
        )
        self._log_trace_memory("chain")

    def _capture_fused_sampled(self):
        """Stage S2, inside the guarded scope: bind the live verify ids and the live topk trace outputs and capture the sampled
        accept trace. A model without the topk outputs (or of another shape) leaves sampled steps on the host walk."""
        fs, model = self._fs, self.model
        if fs is None:
            return
        try:
            fs.bind_ids(model._vfy_ids_out)
            fs.bind_union(model._vfy_topk_vals_out, model._vfy_topk_ids_out)
        except (AssertionError, AttributeError) as e:
            logger.warning(f"[spec_engine] fused sampled accept disabled: {e}")
            self._fs = None
            return
        self._fs_trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        fs.body()
        ttnn.end_trace_capture(self.mesh, self._fs_trace, cq_id=0)
        ttnn.synchronize_device(self.mesh)
        logger.info("[spec_engine] fused sampled accept trace captured (union top-k -> sampled walk + drafter inputs)")
        self._log_trace_memory("accept_s")

    def _seed_buffers(self):
        """The device buffers a seed trace bakes in, per GDN layer (compared by identity at replay)."""
        return [
            (dn.rec_state, dn._conv_hist_rm, dn._spec_ring, *(getattr(dn, "_verify_win_pair", None) or ()))
            for dn in self._gdn
        ]

    def _capture_seed_traces(self):
        """QWEN36_SPEC_JOIN_TRACE, inside the guarded scope, after the chain traces: the GDN seed traces, then the
        MTP warm trace (independent of the GDN guards)."""
        self._capture_gdn_seed_traces()
        self._capture_mtp_warm_trace()

    def _capture_gdn_seed_traces(self):
        """Capture the 48-layer masked seed once per conv-window parity (the parity picks the window half the seed
        writes, and a trace bakes one address). Without the window pair (QWEN36_VERIFY_SPEC_STEP=0) there is one parity and one trace. Needs every layer's
        seed_masked_ok() now; otherwise no trace and every join runs the eager seed."""
        self._seed_trace = self._seed_trace_bufs = None
        if not spec_join_trace_enabled():
            return
        if not self._gdn or self._fj_mask is None or not self.fast_join:
            logger.info("[spec_engine] join seed trace not captured: no GDN layers / mask buffer / fast join")
            return
        bad = [i for i, dn in enumerate(self._gdn) if not dn.seed_masked_ok()]
        if bad:
            logger.info(f"[spec_engine] join seed trace not captured: seed_masked_ok() is False on layer(s) {bad[:4]}")
            return
        fused = all(dn._spec_fused for dn in self._gdn)
        saved = [dn._spec_win_par for dn in self._gdn]
        traces = []
        try:
            for p in (0, 1) if fused else (0,):
                for dn in self._gdn:
                    dn._spec_win_par = p
                tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
                for dn in self._gdn:
                    dn._seed_masked_impl(self._fj_mask, self._fj_sel)
                ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
                traces.append(tid)
        finally:
            for dn, par in zip(self._gdn, saved):
                dn._spec_win_par = par
        ttnn.synchronize_device(self.mesh)
        self._seed_trace = traces if fused else [traces[0], traces[0]]
        self._seed_trace_bufs = self._seed_buffers()
        logger.info(f"[spec_engine] join seed trace captured ({len(traces)} window parity variant(s))")

    def _release_seed_traces(self):
        for tid in {t for t in (self._seed_trace or ())}:
            ttnn.release_trace(self.mesh, tid)
        self._seed_trace = self._seed_trace_bufs = None
        if self._mtpw_trace is not None:
            ttnn.release_trace(self.mesh, self._mtpw_trace)
        self._mtpw_trace = None

    def capture_decode(self):
        """Record the verify trace, then the drafter trace. Idempotent; requires prepare_decode."""
        assert self.prepared, "capture_decode before prepare_decode"
        if self.captured:
            return
        # Verify outputs are read right after a verify replay, before any prefill replay can overwrite them.
        # Cache misses disallowed: a program not compiled by the warm passes fails here, not inside a capture.
        self.mesh.set_program_cache_misses_allowed(False)
        try:
            with trace_allocation_tracker.corruptible_allocation_scope(self.mesh):
                self.model.record_verify_trace()  # V8: its outputs are the shared buffers
                if not self._dr_bks:  # with buckets the model logged V8 itself
                    self._log_trace_memory("V8")
                snap = self._shared_addrs()
                if (
                    self._dr_bks
                ):  # V4, V2, V1 right after V8 (each main + logits + topk), before any drafter / accept capture
                    self.model.record_verify_buckets(self._dr_bks)
                    self._check_shared_addrs(snap, "after the verify buckets")
                if self.recorded_drafter:  # ids outputs are read right after each replay
                    feed = self.model._vfy_rows_out
                    self._dr_trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
                    self._dr_outs = self._drafter_body(feed)
                    ttnn.end_trace_capture(self.mesh, self._dr_trace, cq_id=0)
                    ttnn.synchronize_device(self.mesh)
                    logger.info(f"[spec_engine] drafter trace captured (anchor + reseed + {self.K} draft steps)")
                    self._log_trace_memory("D8")
                    snap.update(self._shared_addrs())
                    for b in self._dr_bks:  # D4, D2, D1: no new persistent outputs, they copy into _dr_outs in place
                        tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
                        self._drafter_body(feed, b, self._dr_outs)
                        ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
                        self._dr_traces[b] = tid
                        ttnn.synchronize_device(self.mesh)
                        logger.info(f"[spec_engine] drafter bucket trace captured (b={b}: {b * self.T} reseed rows)")
                        self._log_trace_memory(f"D{b}")
                    self._check_shared_addrs(snap, "after the drafter buckets")
                    for t in self._dr_standins:  # warm-pass stand-ins of the shared K ids, used by no trace
                        ttnn.deallocate(t)
                    self._dr_standins = ()
                    if self._fa is not None:
                        snap.update(self._shared_addrs())
                        self._capture_fused()
                else:
                    logger.info("[spec_engine] drafter runs eagerly (QWEN36_SPEC_EAGER_DRAFTER=1)")
                self._capture_seed_traces()  # last: every earlier trace keeps its allocation layout
                self._log_trace_memory("seed")
                self._check_shared_addrs(snap, "after every capture")
                self._zero_shared_outputs()
                self._addr_snap = snap
        finally:
            self.mesh.set_program_cache_misses_allowed(True)
        self.captured = True

    def _zero_shared_outputs(self):
        """QWEN36_SPEC_BUCKETS: trace capture does not run the ops, so the shared B-row outputs (verify ids / drafter-feed rows /
        top-k, drafter K ids) hold uninitialized memory until a B-row replay writes them. A bucket replay writes only its own rows,
        so the rows >= b would stay garbage (the host reads all B rows of the K ids). Write zeros once, in place (no new
        allocation: the addresses the traces baked in do not move). Flag off: nothing."""
        if not self._bucketed():
            return
        m = self.model
        outs = [
            getattr(m, n, None) for n in ("_vfy_ids_out", "_vfy_rows_out", "_vfy_topk_vals_out", "_vfy_topk_ids_out")
        ]
        torch_dt = {ttnn.bfloat16: torch.bfloat16, ttnn.uint32: torch.int32}
        for t in (*outs, *(self._dr_outs or ())):
            if t is not None:
                self._stage(torch.zeros(*t.shape, dtype=torch_dt[t.dtype]), t, t.dtype, t.layout)
        ttnn.synchronize_device(self.mesh)

    def _bucket_plan(self):
        """The row buckets < B this engine builds verify + drafter traces for (descending); () = none. QWEN36_SPEC_BUCKETS
        "8" (default) gives (). A flag value the engine cannot serve (no recorded packed drafter, no sharded verify argmax)
        warns and falls back to the B-row traces only (``_buckets`` becomes (B,))."""
        if not self._bucketed():
            return ()
        why = None
        if not self.recorded_drafter:
            why = "the drafter is not recorded (QWEN36_SPEC_EAGER_DRAFTER=1)"
        elif not spec_hostset_trace_enabled():
            why = "the packed drafter inputs are off (QWEN36_SPEC_HOSTSET=0)"
        elif not getattr(self.model, "_vfy_sharded", False):
            why = "the verify argmax is not sharded (QWEN36_VERIFY_SHARDED_ARGMAX)"
        bks = tuple(sorted((b for b in self._buckets if b < self.B), reverse=True))
        if why is not None or not bks:
            logger.warning(
                f"[spec_engine] QWEN36_SPEC_BUCKETS={self._buckets} ignored: {why or 'no bucket below B'}; B-row traces only"
            )
            self._buckets = (self.B,)
            return ()
        logger.info(f"[spec_engine] row buckets {bks} + {self.B}: verify and drafter traces per bucket")
        return bks

    def _log_trace_memory(self, label):
        """Log the trace-region / DRAM bytes after the capture ``label`` (model._log_trace_memory; never raises)."""
        return self.model._log_trace_memory(label)

    def _shared_addrs(self):
        """name -> buffer address of every persistent buffer a trace baked in that more than one trace (or the host) shares:
        the verify outputs (ids / drafter-feed rows / topk pair), its token / aux inputs, the drafter outputs and inputs,
        the accept and chain inputs / outputs. Buffers that do not exist yet are left out. Bound-in buffers (the ids and
        union inputs of the accept traces) are the verify outputs themselves and are listed once, under the model."""
        model, out = self.model, {}

        def add(name, t):
            if t is not None:
                out[name] = int(t.buffer_address())

        for n in (
            "_vfy_ids_out",
            "_vfy_rows_out",
            "_vfy_topk_vals_out",
            "_vfy_topk_ids_out",
            "_vfy_token_buf",
            "_vfy_aux_buf",
        ):
            add(f"model.{n}", getattr(model, n, None))
        for k, t in enumerate(self._dr_outs or ()):
            add(f"dr_outs[{k}]", t)
        for n in ("_dr_pack", "_dr_eye", "_dr_rs_pt", "_hp_buf", "_mtp_pt", "_anchor_sel"):
            add(n, getattr(self, n, None))
        for b, t in (self._hp_bufs or {}).items():
            add(f"_hp_bufs[{b}]", t)
        for tag, obj, names in (
            ("fa", self._fa, ("aux_in", "pt_rows_in", "result_out")),
            ("fs", self._fs, ("samp_in", "result_out")),
            ("ch_g", self._ch_g, ("ovr_in", "result_in", "aux_prev_in", "tok_out", "vaux_out")),
            ("ch_s", self._ch_s, ("ovr_in", "result_in", "aux_prev_in", "tok_out", "vaux_out")),
        ):
            for n in names:
                if obj is not None:
                    add(f"{tag}.{n}", getattr(obj, n, None))
        return out

    def _check_shared_addrs(self, snap, label):
        """Every buffer in ``snap`` (from _shared_addrs) must still sit at its address: a hand-off that re-binds its
        destination (slice_write into a TILE buffer) leaves earlier traces writing the old address, and later captures
        may place scratch on the new one."""
        now = self._shared_addrs()
        moved = {n: (hex(a), hex(now[n]) if n in now else None) for n, a in snap.items() if now.get(n) != a}
        assert not moved, f"[spec_engine] shared buffers moved {label}: {moved}"
        logger.info(f"[spec_engine] {len(snap)} shared buffer addresses unchanged {label}")

    def shutdown(self):
        """Release the traces and every persistent engine buffer; a later prepare starts over."""
        model = self.model
        t0 = time.time()
        logger.info("[spec_engine] shutdown: releasing traces")
        model.release_verify_trace()
        model._vfy_prepared_key = None
        model.__dict__.pop("_vfy_mem_log", None)  # the next prepare / capture logs its memory from zero
        if self.prefill_recorded and getattr(model, "_chunked_trace_id", None) is not None:
            ttnn.release_trace(model.device, model._chunked_trace_id)
            model._chunked_trace_id = None
            model._chunked_prepared_key = None
        if self.fast_prefill:
            model.release_prefill_trace_short()  # the spec short traces (captured or only prepared) and their inputs
            model.release_gdn_slot_write_traces()
        self._release_seed_traces()
        for name in ("_ch_trace_g", "_ch_trace_s"):
            if getattr(self, name) is not None:
                ttnn.release_trace(self.mesh, getattr(self, name))
                setattr(self, name, None)
        for ch in (self._ch_g, self._ch_s):
            if ch is not None:  # the shared buffers (token / aux / result_in / aux_prev_in) belong to their owners
                for t in (ch.ovr_in, *(v for n, v in vars(ch).items() if n.startswith("c_"))):
                    ttnn.deallocate(t)
        for t in self._ch_standins:
            ttnn.deallocate(t)
        self._ch_g = self._ch_s = None
        self._ch_standins = ()
        self._chain_prev = self._chain_in = self._chain_cand = None
        if self._fa_trace is not None:
            ttnn.release_trace(self.mesh, self._fa_trace)
            self._fa_trace = None
        if self._fs_trace is not None:
            ttnn.release_trace(self.mesh, self._fs_trace)
            self._fs_trace = None
        if self._fs is not None:
            fs = self._fs
            for t in (
                fs.samp_in,
                fs.result_out,
                *self._fs_standin,
                *(v for n, v in vars(fs).items() if n.startswith("c_")),
            ):
                if t is not None:  # the live union buffers are the topk trace's outputs: they died with that trace
                    ttnn.deallocate(t)
        self._fs, self._fs_standin = None, ()
        if self._fa is not None:
            for name in ("aux_in", "pt_rows_in", "result_out", "c_iota_t", "c_iota_k", "c_ut", "c_one"):
                ttnn.deallocate(getattr(self._fa, name))
            if self._fa.ids_standin is not None:  # the warm-pass stand-in (the live ids buffer dies with its trace)
                ttnn.deallocate(self._fa.ids_standin)
        self._fa, self._fused_cache, self._fa_pt_host, self._dr_rs_dirty = None, None, None, False
        for tid in (self._dr_traces or {}).values():  # the drafter bucket traces (D4, D2, D1)
            ttnn.release_trace(self.mesh, tid)
        self._dr_traces = {}
        if self._dr_trace is not None:
            ttnn.release_trace(self.mesh, self._dr_trace)
            self._dr_trace = None
        for t in (*self._dr_outs, *self._dr_bufs, *(self._hp_bufs or {}).values(), *self._dr_standins):
            ttnn.deallocate(t)
        self._dr_outs, self._dr_bufs, self._hp_bufs, self._dr_standins, self._dr_bks = [], [], None, (), ()
        self._addr_snap = None
        for t in (self._mtpw_pack, self._mtpw_pt):
            if t is not None:
                ttnn.deallocate(t)
        self._mtpw_pack = self._mtpw_pt = self._mtpw_rope = None
        self._mtpw_hidden, self._mtpw_phys = (), {}
        self._dr_tok0 = self._dr_rs_tok = self._dr_rs_pos = self._dr_rs_pt = self._dr_rs_cos = self._dr_rs_sin = None
        self._dr_pack = self._dr_eye = self._dr_rs_cache = None
        self._dr_hostset = False
        self._dr_pos, self._dr_cos, self._dr_sin = [], [], []
        self.__dict__.pop("_stage_cache", None)  # QWEN36_SPEC_HOSTFIX table cache: its buffers are freed here
        bufs = [
            *self._last_hidden,
            *self._last_sel.values(),
            self._mtp_pt,
            *self._mtp_pt_rows,
            self._hp_buf,
            self._anchor_sel,
            self._eager_sel,
            self._join_take,
            self._join_keep,
            self._fj_sel,
            self._fj_mask,
        ]
        for t in bufs:
            if t is not None:
                ttnn.deallocate(t)
        self._fj_sel = self._fj_mask = None
        self._last_hidden, self._last_sel = [], {}
        self._mtp_pt, self._mtp_pt_rows, self._hp_buf, self._anchor_sel, self._eager_sel, self._vfeed = (
            None,
            [],
            None,
            None,
            None,
            None,
        )
        self._join_take = self._join_keep = None
        self._prep_pt = None
        self._fresh.clear()
        self._active.clear()
        self.__dict__.setdefault("_finishing", set()).clear()
        self._set_slot_map(range(self.B))
        self.prefill_prepared = self.prefill_recorded = self.prepared = self.captured = False
        if getattr(model, "_spec_engine", None) is self:
            model._spec_engine = None
        logger.info(f"[spec_engine] shutdown done in {time.time() - t0:.2f} s")

    def measured_spec_bytes(self):
        """ACTUAL allocated per-device bytes of the buffers spec_memory_costs models (same dict structure).
        Call after prepare (and capture, for the trace outputs)."""
        m, nbytes = self.model, _device_bytes
        per_seq = {"gdn_ring": 0, "gdn_conv_windows": 0}
        for dn in self._gdn:
            per_seq["gdn_ring"] += nbytes(dn._spec_ring)
            per_seq["gdn_conv_windows"] += sum(
                nbytes(getattr(dn, n)) for n in ("_verify_win_buf", "_spec_win_pad", "_conv_win_buf")
            ) + sum(nbytes(w) for w in (getattr(dn, "_verify_win_pair", None) or ()))
        per_seq["drafter_hidden"] = sum(nbytes(t) for t in self._last_hidden)
        vfy = (
            "token_buf",
            "kvpos_buf",
            "kvpt_buf",
            "kvpt_users",
            "kvpt1_buf",
            "cos_buf",
            "sin_buf",
            "state_idx",
            "conv_sel",
            "logits_out",
            "rows_out",
            "ids_out",
            "logits_rm_out",
        )
        att = self.mtp.attention
        kv_bytes = nbytes(att.paged_k) + nbytes(att.paged_v)
        n_blocks = int(att.paged_k.shape[0])
        bks = (getattr(m, "_vfy_bk", None) or {}).values()
        fixed = {
            "verify_io": sum(nbytes(getattr(m, f"_vfy_{n}", None)) for n in vfy)
            + sum(
                nbytes(getattr(bk, n, None))
                for bk in bks
                for n in ("normed", "shard_logits", "logits_out", "logits_rm_out")
            ),
            "mtp_scratch_block": kv_bytes // n_blocks,
            "engine_buffers": sum(
                nbytes(t)
                for t in (
                    *self._last_sel.values(),
                    *(self._hp_bufs or {}).values(),
                    self._mtp_pt,
                    *self._mtp_pt_rows,
                    self._hp_buf,
                    self._anchor_sel,
                    self._eager_sel,
                    self._join_take,
                    self._join_keep,
                    self._fj_sel,
                )
            ),
        }
        per_token = {"mtp_kv": kv_bytes // (n_blocks * int(att.paged_k.shape[-2]))}
        for d in (per_seq, fixed, per_token):
            d["total"] = sum(d.values())
        return {"per_seq": per_seq, "fixed": fixed, "per_token": per_token}

    def _park_save(self, us):
        """{row: [per-GDN-layer (ring, e_prev) host state]} of the rows ``us``. With layers that offer
        spec_users_save_enqueue (TPGatedDeltaNet) every layer's D2H copy is enqueued non-blocking and the device is
        synchronized ONCE; otherwise one blocking save_spec_user per layer and row. Same bytes either way."""
        gdn = self._gdn
        if gdn and all(hasattr(dn, "spec_users_save_enqueue") for dn in gdn):
            pend = [dn.spec_users_save_enqueue(us) for dn in gdn]
            ttnn.synchronize_device(self.mesh)
            per_layer = [dn.spec_users_save_finish(p) for dn, p in zip(gdn, pend)]
            return {u: [d[u] for d in per_layer] for u in us}
        return {u: [dn.save_spec_user(u) for dn in gdn] for u in us}

    def _park_load(self, states):
        """Restore parked rows: ``states`` = per GDN layer {row: (ring slot, e_prev)}. Batched like _park_save: all H2D
        writes enqueued, one synchronize (keeps the host tensors alive until the copies are done)."""
        gdn = self._gdn
        if gdn and all(hasattr(dn, "spec_users_load_enqueue") for dn in gdn):
            for dn, st in zip(gdn, states):
                dn.spec_users_load_enqueue(st)
            ttnn.synchronize_device(self.mesh)
            return
        for dn, st in zip(gdn, states):
            dn.load_spec_users(st)

    def _park_note(self, kind, t0):
        d = self.__dict__.setdefault("park_stats", {})
        n, ms = d.get(kind, (0, 0.0))
        self._park_last_ms = (time.perf_counter() - t0) * 1e3
        d[kind] = (n + 1, ms + self._park_last_ms)

    def _check_finishing_rows(self, active_rows):
        bad = sorted(u for u in active_rows if u in self.__dict__.get("_finishing", ()))
        if bad:
            raise RuntimeError(
                f"finishing engine rows {bad} are also scheduled in this step: they were freed as finishing (their request ends in the "
                "step in flight) and carry a request again without a prefill: their GDN state was not parked, so they cannot resume"
            )

    def _apply_finishing(self, finishing_slots):
        """decode_forward(finishing_slots): requests left out of this step only because their in-flight step reaches
        max_tokens. ``finishing_slots`` are in the PRE-remap runner numbering (the slot the request held before this step's
        ``slot_remap``; the plugin pops them before it computes the remap, so the remap may reuse that slot number for another
        request). They are resolved to engine rows with the slot map as it is NOW, so call this BEFORE ``remap_slots``; the
        rows are freed as a finished request is (release, no park), remembered in ``_finishing`` until a prefill reassigns
        them, and are inactive overlay rows of the chain. remap_slots then permutes ``_finishing`` with the other per-row
        sets. The "also scheduled" check runs on engine rows after the remap and the scatter (``_check_finishing_rows``).
        Order in decode_forward / _decode_forward_deferred: validate block -> _apply_finishing (pre-remap slots) -> remap_slots
        -> _to_rows (post-remap slots -> engine rows) -> _check_finishing_rows -> verify (parked / fresh rows as usual).
        """
        if finishing_slots is None:
            return
        slots = [int(s) for s in finishing_slots]
        for s in slots:
            if not 0 <= s < self.B:
                raise ValueError(f"finishing slot {s} out of range [0,{self.B})")
        rows = [self._phys[s] for s in slots]
        if len(set(rows)) != len(rows):
            raise RuntimeError(f"finishing slots {slots} name an engine row twice")
        fin = self.__dict__.setdefault("_finishing", set())
        for s, row in zip(slots, rows):
            fin.add(row)
            self.release(s)
        self.finishing_freed = self.__dict__.get("finishing_freed", 0) + len(slots)

    def _row_gens(self):
        """Per-engine-row generation: bumped when a prefill or a release gives the row to another request. A deferred handle
        records it at creation; its later finish / materialize skip rows whose generation moved (a prefill ran over the pending
        step into the slot, e.g. a claimed finishing slot reused by a new request)."""
        if self._row_gen is None:
            self._row_gen = [0] * self.B
        return self._row_gen

    def release(self, slot):
        """Mark runner slot ``slot`` (engine row _phys[slot]) empty and re-offer drafts to every row (no device work)."""
        slot = self._phys[int(slot)]
        self._row_gens()[slot] += 1
        self._active.discard(slot)
        self._fresh.pop(slot, None)
        self._parked.pop(slot, None)
        self._prompts.pop(slot, None)
        if _chunk_prog_of(self).pop(slot, None) is not None:  # a released row's chunk-resume entry (Option B) is void
            self._chunk_unpin(slot)  # ... and so is its pinned snapshot
        _row_vision_of(self).pop(slot, None)
        _record_row_delta(self, slot, None)  # the row's M-RoPE delta goes with its request (text rows are 0)
        self._fresh_dirty.discard(slot)
        reset_draft_skip(self, slot)
        # the leaving request may be why other rows' drafts were withheld: offer them again
        self._no_draft.clear()
        self._withheld_streak.clear()

    def remap_slots(self, remap):
        """Apply a runner slot_remap (runner row i takes the request that sat at row remap[i]). Host slot map
        (QWEN36_SPEC_SLOTMAP=1): only ``_phys`` changes, no device work, every engine row and per-row host set
        keeps its state. Otherwise the GDN state of every layer and the per-slot host state are permuted on the
        device; paged KV follows the page tables either way. Identity is a no-op."""
        idx = [int(x) for x in remap]
        if sorted(idx) != list(range(self.B)):
            raise ValueError(f"slot_remap {idx} is not a permutation of range({self.B})")
        if idx == list(range(self.B)):
            return
        if self.host_slotmap:
            self._set_slot_map(slot_map_after_remap(self._phys, idx))
            return
        if getattr(self, "_ch_g", None) is not None:  # QWEN36_SPEC_CHAIN
            self._chain_drop("slot_change")  # the device remap moves the engine rows the chain state describes
        for dn in self._gdn:
            dn.remap_spec_slots(idx)
        if (
            getattr(getattr(self, "model", None), "_slot_rope_delta", None) is not None
        ):  # the per-row M-RoPE deltas move with the rows
            self.model.remap_slot_rope_delta(idx)
        new_of = {old: new for new, old in enumerate(idx)}
        self.__dict__["_row_vision_map"] = {new_of[u]: v for u, v in _row_vision_of(self).items()}
        self._last_hidden = [self._last_hidden[j] for j in idx]
        self._fresh = {new_of[u]: n for u, n in self._fresh.items()}
        self._withheld_streak = {new_of[u]: n for u, n in self._withheld_streak.items()}
        self._parked = {new_of[u]: v for u, v in self._parked.items()}
        self._finishing = {new_of[u] for u in self.__dict__.get("_finishing", ())}
        self._prompts = {new_of[u]: v for u, v in self._prompts.items()}
        self.__dict__["_chunk_progress"] = {new_of[u]: v for u, v in _chunk_prog_of(self).items()}
        c = self._chunk_cache()
        if c is not None and hasattr(
            c, "remap_owners"
        ):  # the pins follow their rows (_last_hidden moved with idx above)
            c.remap_owners(new_of)
        self._fresh_dirty = {new_of[u] for u in self._fresh_dirty}
        self._active = {new_of[u] for u in self._active}
        self._offered = {new_of[u] for u in self._offered}
        self._no_draft = {new_of[u] for u in self._no_draft}
        self._speculated = {new_of[u] for u in self._speculated}

    # --------------------------------------------------------------------- #
    # Prefill
    # --------------------------------------------------------------------- #
    def prefill(
        self, slot, prompt_ids, page_table_row, _cap_blocks=None, start_pos=None, vision_tokens=None, final=True
    ):
        """Eager prefill of runner slot ``slot`` (engine row _phys[slot]), warming its MTP KV; returns host float
        logits [vocab]. Replays the recorded chunk trace when B == 1 and record_prefill ran, else the eager path.
        The GDN seed and the MTP KV of slot Tp-1 are written by the first ``verify``.
        ``start_pos`` (vLLM num_computed_tokens; None = not part of prefix caching, the unchanged path): with a GDN
        prefix-state cache on the model (B > 1) the prefill resumes from a snapshot <= start_pos (see
        Qwen36Model.plan_spec_prefix) on the traced fast path and on the eager path; B == 1 recomputes from 0.
        ``vision_tokens`` (multimodal request: the packed vision rows the tower produced; None = text): spliced into the
        image/video placeholder positions and the M-RoPE table is built with the grid. Such a request never uses the
        prefix cache (no plan, no snapshot save / lookup): the caller passes start_pos=None, and a start_pos given anyway
        is ignored. The drafter keeps the placeholder embedding at those positions.
        ``final`` False (QWEN36_SPEC_CHUNKED_PREFILL: a non-final chunk [start_pos, e) of a longer prompt, e = len(prompt_ids), a
        multiple of 128): the chunk is prefilled (KV + drafter KV, GDN snapshot at e-128 into the prefix cache) but the row is
        not registered (no _prompts / _fresh / _active, no GDN slot write, no last-row capture, no host logits read) and a
        zero row is returned. Text rows only.
        Option A (QWEN36_SPEC_CHUNK_EXACT_RESUME=0): the snapshot is at e-128 and the next chunk (start_pos = e) resumes
        there, recomputing [e-128, e). Option B (default): the snapshot is at the chunk end e and the target hidden row of
        slot e-1 is captured into _last_hidden[slot]; the next chunk (start_pos = e, same prompt prefix and page table, see
        _chunk_prog) resumes exactly at e and first warms the drafter slot e-1 with (that row, token e)."""
        assert final or vision_tokens is None, "a multimodal prompt cannot be prefilled in non-final chunks"
        assert 0 <= slot < self.B, f"slot {slot} out of range [0,{self.B})"
        if getattr(self, "_ch_g", None) is not None:  # QWEN36_SPEC_CHAIN
            self._chain_drop("prefill")
        if (
            getattr(self, "_def_last", None) is not None
        ):  # deferred serving: nothing chains on the pending step across a prefill
            self._def_last.chainable = False
            if not self._def_last.device and self._def_last.fin is None:
                # a fallback step ran on the host: its propose runs the drafter over the page table of its own step, which the new
                # request may reuse; finish it now, as the drained order does
                self._def_finish_any(self._def_last)
        if len(getattr(self, "_buckets", None) or ()) > 1:  # QWEN36_SPEC_BUCKETS
            self._claim_low_row(slot)
        self._row_gens()[self._phys[int(slot)]] += 1
        # text: the call is unchanged (no vision kwarg)
        vis_kw = {} if vision_tokens is None else dict(vision_tokens=vision_tokens)
        if not final:
            vis_kw["final"] = False
        return self._prefill_row(self._phys[int(slot)], prompt_ids, page_table_row, _cap_blocks, start_pos, **vis_kw)

    def _prefill_row(
        self, slot, prompt_ids, page_table_row, _cap_blocks=None, start_pos=None, vision_tokens=None, final=True
    ):
        """``prefill`` of engine row ``slot`` (``final`` False: a non-final chunk, see ``prefill``)."""
        reset_draft_skip(self, slot)
        self._parked.pop(slot, None)
        # A prefill of the row ends its pending progress (see _chunk_exact_resume): the request it belongs to consumes it, any
        # other request drops it (warning there; _claim_low_row keeps that from happening).
        prog = _chunk_prog_of(self).pop(slot, None)
        self.__dict__.setdefault("_finishing", set()).discard(slot)  # a prefill gives the slot a new request
        self._fresh_dirty.discard(slot)
        model = self.model
        assert 0 <= slot < self.B, f"slot {slot} out of range [0,{self.B})"
        assert self._last_hidden, "prefill before prepare"
        prompt_row = None
        if prefill_host_tolist_enabled():
            # QWEN36_PREFILL_HOST_TOLIST: one tolist() and an int32 copy of the same ids (see _prompt_ids_fast)
            prompt_list, prompt_row = _prompt_ids_fast(prompt_ids)
        if prompt_row is None:
            prompt_list = [int(t) for t in torch.as_tensor(prompt_ids).reshape(-1)]
        Tp = len(prompt_list)
        if final:
            self._prompts[slot] = (prompt_list, torch.as_tensor(page_table_row).clone()) + (
                () if start_pos is None else (int(start_pos),)
            )
        else:
            assert vision_tokens is None, "a multimodal prompt cannot be prefilled in non-final chunks"
            assert Tp % 128 == 0, f"non-final prefill chunk end {Tp} is not a multiple of 128"
            assert (
                start_pos is not None and 0 <= int(start_pos) < Tp
            ), f"non-final chunk start_pos {start_pos} vs end {Tp}"
        # A multimodal row keeps its vision rows until it is released / joined: a re-prefill (_reprefill,
        # _reconcile_fresh_rows) needs them. A text prefill on the row clears a previous request's.
        if vision_tokens is None:
            _row_vision_of(self).pop(slot, None)
        else:
            _row_vision_of(self)[slot] = vision_tokens
        cap_blocks = int(torch.as_tensor(page_table_row).shape[-1]) if _cap_blocks is None else _cap_blocks
        # a non-final chunk runs no verify rows past its end
        assert Tp + (self.T if final else 0) <= table_capacity(
            cap_blocks, self.block_size
        ), f"prompt {Tp} + {self.T if final else 0} verify rows does not fit {cap_blocks} blocks x {self.block_size}"
        pt_row = model._fit_spec_page_table(page_table_row, self.nb)
        assert pt_row.shape[0] == 1, "prefill takes ONE page-table row"
        self._check_scratch(pt_row)
        prompt = torch.tensor([prompt_list], dtype=torch.int32) if prompt_row is None else prompt_row
        # the ids the chunk-resume digests hash: the list (today), or the int32 row's bytes (QWEN36_PREFILL_HOST_TOLIST)
        digest_ids = prompt_list if prompt_row is None else prompt_row[0]
        captured_row = [False]
        # Prefix cache (B > 1, start_pos given, cache built): (s, (cache, slot) | None, save points) or None.
        # A multimodal request bypasses the prefix cache (mirrors Qwen36Model.prefill_paged_slots): no lookup, no saves.
        # A non-final chunk also snapshots the GDN state at e-128 (e = Tp): the next chunk (start_pos = e) resumes there
        # (spec_resume_cap(e, L) = e-128) and recomputes [e-128, e), which also warms the drafter slot e-1.
        force = ()
        exact_mode = chunk_exact_resume_enabled()
        save_end = (not final) and exact_mode  # Option B: snapshot the chunk end + capture the hidden row of slot e-1
        if not final:
            if self.B <= 1 or getattr(model, "_prefix_cache", None) is None:
                raise RuntimeError("non-final prefill chunk needs the batched spec engine and the GDN prefix cache")
            if save_end:
                pass
            elif Tp - 128 > int(start_pos):
                force = (Tp - 128,)
            else:
                logger.info(
                    f"[spec_engine] chunked prefill: chunk [{start_pos}, {Tp}) is <= 128 tokens: no e-128 snapshot"
                )
        exact = self._chunk_exact_resume(prog, digest_ids, start_pos, Tp, pt_row, vision_tokens, exact_mode)
        plan_kw = {}
        if force:
            plan_kw["force_points"] = force
        if exact is not None:
            plan_kw["exact_resume"] = exact
        if save_end:
            plan_kw["nonfinal"] = True
        plan = (
            model.plan_spec_prefix(prompt[0], Tp, start_pos, **plan_kw)
            if start_pos is not None and self.B > 1 and vision_tokens is None
            else None
        )
        if not final and plan is None and (force or save_end):
            raise RuntimeError(f"non-final chunk [{start_pos}, {Tp}): no prefix plan (forced save point {force[:1]})")
        if exact is not None and (plan is None or int(plan[0]) != exact):
            # the snapshot at e is gone (evicted): the plan fell back to the normal cap; the hidden row of e-1 is unusable
            exact = None
            self.chunk_exact_misses = getattr(self, "chunk_exact_misses", 0) + 1
            self._chunk_bump("refusals")
            logger.warning(f"[spec_engine] chunk exact resume: snapshot at {start_pos} evicted, Option A resume")
        self._note_resume_fallback(prompt_list, start_pos, Tp, plan, final)
        vis_kw = {} if vision_tokens is None else dict(vision_tokens=vision_tokens)  # text: the calls are unchanged
        # The prompt the drafter fuses its next-token column from. A non-final fast-prefill segment [a, b) narrows it
        # to prompt[:b+1] so padded bucket rows never fuse (and write drafter KV for) tokens past b.
        mtp_prompt = [prompt_list]

        def _on_chunk(hidden, chunk_start, valid_len):
            # Never frees `hidden` (the caller does, or it is the trace's persistent output).
            feed = model.spec_feed_rows(hidden)
            self._warm_mtp_chunk(feed, chunk_start, valid_len, mtp_prompt[0], pt_row)
            if (final or save_end) and chunk_start + valid_len >= Tp:  # the chunk holding slot Tp-1
                self._capture_last_row(feed, Tp - 1 - chunk_start, slot)
                captured_row[0] = True
            if feed is not hidden:
                ttnn.deallocate(feed)

        if exact is not None:
            # Drafter slot e-1 (unwarmed after the previous chunk): (target hidden row of e-1, token e), through THIS
            # request's page table. Slots >= e are warmed by the segments below.
            self._warm_resume_slot(slot, prompt_list[exact], exact - 1, pt_row)
            self.chunk_exact_resumes = getattr(self, "chunk_exact_resumes", 0) + 1
            self._chunk_bump("exact_resumes")
        logits_dev = None
        if self.fast_prefill and self.prefill_recorded and self.recorded_prefill:
            fast_kw = {} if plan is None else dict(plan=plan, mtp_prompt=mtp_prompt)
            fast_kw.update(vis_kw)
            if not final:
                fast_kw["final_row"] = False
                fast_kw["save_end"] = save_end
            logits_dev = self._prefill_fast(slot, prompt_list, prompt, pt_row, Tp, _on_chunk, captured_row, **fast_kw)
        if logits_dev is None and self.B == 1 and self.prefill_recorded and self.recorded_prefill:
            pt_rec = model._fit_spec_page_table(pt_row, self.nb_prefill)  # MTP chunk fill keeps the nb-wide pt_row
            logits_dev = model.prefill_traced_chunked(prompt, pt_rec, Tp, on_chunk=_on_chunk, **vis_kw)
        if logits_dev is None:
            if plan is None:
                logits_dev = model.prefill_for_spec(prompt, pt_row, Tp, _on_chunk, slot=slot, **vis_kw)
            elif final:
                logits_dev = model.prefill_for_spec(prompt, pt_row, Tp, _on_chunk, slot=slot, plan=plan)
            else:
                model.prefill_for_spec(
                    prompt, pt_row, Tp, _on_chunk, slot=slot, plan=plan, final=False, save_end=save_end
                )
        self._chunk_unpin(slot)  # the previous chunk's snapshot was restored above (or refused): its pin ends
        self._note_prefill_done()
        if not final:
            # nothing is registered for the row. Option A: the next chunk prefills again from the e-128 snapshot.
            # Option B: the next chunk resumes at e from the snapshot at e and the row captured into _last_hidden[slot].
            if save_end:
                assert captured_row[0], f"non-final chunk never delivered the chunk holding slot {Tp - 1}"
                _chunk_prog_of(self)[slot] = (Tp, self._chunk_digest(digest_ids, Tp), self._chunk_pt_sig(pt_row, Tp))
                c = self._chunk_cache()
                if c is not None and hasattr(c, "pin"):
                    c.max_pinned = self.B  # one live chunk per engine row
                    if not c.pin(prompt, Tp, slot):
                        logger.warning(
                            f"[spec_engine] chunk-end snapshot at {Tp} of row {slot} is not in the cache: not pinned"
                        )
            return torch.zeros(self.vocab)
        assert captured_row[0], f"prefill never delivered the chunk holding slot {Tp - 1}"
        if isinstance(logits_dev, torch.Tensor):
            lt = logits_dev
        else:
            lt = self._replicated_to_torch(logits_dev)
            ttnn.deallocate(logits_dev)
        self._fresh[slot] = Tp
        self._active.add(slot)
        _record_row_delta(
            self, slot, None if vision_tokens is None else getattr(getattr(model, "rope", None), "rope_delta", 0)
        )
        return lt.reshape(-1)[: self.vocab].float()

    def _prefill_fast(
        self,
        slot,
        prompt_list,
        prompt,
        pt_row,
        Tp,
        on_chunk,
        captured_row,
        plan=None,
        mtp_prompt=None,
        vision_tokens=None,
        final_row=True,
        save_end=False,
    ):
        """QWEN36_SPEC_FAST_PREFILL=1 body of _prefill_row: traced prefill of one prompt on the GDN binding the traces
        were captured against (the B=1 scratch at B > 1, the decode buffers at B == 1), then the scratch -> slot write.
        Returns host float logits [vocab], or None when no trace serves the prompt (the caller runs the eager path;
        nothing has been staged or written then).

        ``plan`` (Qwen36Model.plan_spec_prefix: (s, (cache, snapshot slot) | None, save points); B > 1 only): the
        prefix-cached prefill. The GDN snapshot at s is restored into the scratch (s == 0: the first segment zeroes it)
        and the prompt runs as segments [s, p1), [p1, p2), ..., [pk, Tp) at absolute positions, each through the same
        traced chunk replays + short-trace tail as an uncached prompt (chunk_start = the segment's absolute start,
        GDN state carried). The scratch is snapshotted after every segment that ends at a save point pi. The drafter
        KV of [s, Tp) is warmed in the same replays (on_chunk / the spec short trace); [0, s) is already in the
        request's cached blocks (see doc 40).

        ``vision_tokens`` (multimodal prompt; plan is None then): the packed vision rows, spliced per segment by
        _fast_segment.

        ``final_row`` False (non-final chunk): the scratch is not written to the row's GDN slot and no host logits are read
        (the returned value is meaningless). ``save_end`` (with final_row False; Option B): the state after tokens[:Tp]
        is also snapshotted into the prefix cache (no extra segment: Tp is the last segment's end)."""
        model = self.model
        batched = self.B > 1
        if plan is None:
            s0, src, points = 0, None, []
        else:
            s0, src, points = plan
        bounds = list(points) + [Tp]
        run, seg_kw = None, {}  # QWEN36_PREFILL_INTRACE_SNAP: the single segment's _SnapRun
        # The traces' GDN binding check (_short_trace_for) needs the binding the requests run on, so bind first.
        prev = model._bind_gdn_prefill_scratch() if batched else None
        try:
            self._skip_logits_read = not final_row
            if plan is not None:
                assert vision_tokens is None, "a multimodal prompt bypasses the prefix cache"
                model._build_request_rope(prompt, None)  # one table for the whole prompt: every segment slices it
                if s0 > 0:
                    src[0].restore(src[1])
                run, mb_choice = self._snap_run_plan(prompt, pt_row, s0, Tp, points)
                if run is not None:
                    bounds = [Tp]  # ONE segment: every save point is a replay end or an in-replay snapshot
                    seg_kw = dict(snap=run, mb_choice=mb_choice)
            host, saved, a = None, [], s0
            for i, b in enumerate(bounds):
                final = i == len(bounds) - 1
                if mtp_prompt is not None:
                    mtp_prompt[0] = prompt_list if final else prompt_list[: b + 1]
                if final:
                    captured_row[0] = False  # only the segment holding slot Tp-1 may satisfy the capture assert
                host = self._fast_segment(
                    slot,
                    prompt_list if final else prompt_list[: b + 1],
                    prompt,
                    pt_row,
                    a,
                    b,
                    final,
                    carry=a > 0,
                    on_chunk=on_chunk,
                    captured_row=captured_row,
                    decline_ok=(i == 0 and s0 == 0),
                    **({} if vision_tokens is None else dict(vision_tokens=vision_tokens)),
                    **seg_kw,
                )
                if host is None:
                    return None  # only the first from-0 segment may decline, before any side effect
                if not final:
                    # the bound scratch = the state after tokens[:b]
                    model._prefix_cache.save(
                        prompt[0],
                        b,
                        low_prio=b in getattr(points, "low", ()),
                        protect=b in getattr(points, "end", ()),
                    )
                    saved.append(b)
                a = b
            if run is not None:
                saved = list(run.saved)  # in position order, as the per-segment saves above would have been
            if save_end:
                assert plan is not None
                model._prefix_cache.save(prompt[0], Tp)  # the bound scratch = the state after tokens[:Tp]
                saved.append(Tp)
            if plan is not None:
                model._prefix_cache.log(Tp, -1, s0, saved)
        finally:
            self._skip_logits_read = False
            if mtp_prompt is not None:
                mtp_prompt[0] = prompt_list
            if prev is not None:
                model._unbind_gdn_prefill_scratch(prev)
        if batched and final_row:
            # Device-to-device: scratch (B=1) -> row `slot` of the batched GDN buffers, other rows untouched (the slot-write
            # trace when captured and still valid, else TPGatedDeltaNet.write_slot(consume=False) eagerly).
            model._write_gdn_slot_from_scratch(slot)
            if os.environ.get("QWEN36_SPEC_FAST_PREFILL_SYNC", "0") == "1":
                # Measurement aid (default off): wait for the slot-write replay inside the prefill call. Measured on the
                # Galaxy: ~1.5 ms of device time at ISL 1600, and no effect on the first decode step after the prefill
                # (test_spec_continuous join step: 88.6 ms without vs 89.7 / 91.2 ms with the wait, eager prefill 61 ms).
                ttnn.synchronize_device(self.mesh)
        chunk = model._chunked_chunk_size or 2048
        num_full, tail = (Tp - s0) // chunk, (Tp - s0) % chunk
        path = "short trace" if num_full == 0 and s0 == 0 else f"{num_full} chunk replay(s), tail {tail}"
        msg = f"[spec_engine] fast prefill: slot {slot} Tp={Tp} ({path})" + (
            "" if plan is None else f" prefix resume s={s0} segments={len(bounds)}"
        )
        if run is not None:
            msg += f" in-replay snapshots={run.n_inside} saves={len(run.saved)}"
            self.intrace_prefills = getattr(self, "intrace_prefills", 0) + 1
            self.intrace_snapshots = getattr(self, "intrace_snapshots", 0) + run.n_inside
        if getattr(self, "_fast_prefill_logged", False):
            logger.debug(msg)
        else:
            self._fast_prefill_logged = True
            logger.info(msg)
        return host

    def _snap_run_plan(self, prompt, pt_row, s0, Tp, points):
        """QWEN36_PREFILL_INTRACE_SNAP: (_SnapRun, masked-big choice) when the prefill [s0, Tp) can run as ONE segment
        with every save point a replay end or an in-replay snapshot, else (None, None) (flag off, no point to save, or a
        point would fall in an eager replay / exceed the GDN_SNAP_SLOTS of a replay: the legacy split serves it). The
        replays are the ones _fast_segment runs for [s0, Tp) (prefill_plan.segment_replays with the same masked-big
        choice, passed on to it so the decision is made once)."""
        model = self.model
        cache = getattr(model, "_prefix_cache", None)
        active = getattr(model, "gdn_snap_active", None)  # absent on test fakes
        if cache is None or active is None or not active() or not points:
            return None, None
        chunk = model._chunked_chunk_size or 2048
        n = Tp - s0
        big = model.big_chunk_rows()
        choice = self._masked_big_choice(n, s0) if n >= chunk else None
        mb = None if choice is None or choice[0] == "old" else int(choice[0][1:])
        try:
            replays = prefill_plan.segment_replays(s0, Tp, big, mb, chunk)
            where = prefill_plan.assign_points(replays, points)
        except (AssertionError, ValueError) as e:
            logger.debug(f"[spec_engine] in-replay snapshots: legacy split for [{s0}, {Tp}) ({e})")
            return None, None
        for rep, (inside, _end) in zip(replays, where):
            if not inside:
                continue
            if len(inside) > GDN_SNAP_SLOTS:
                return None, None
            if rep.kind == "tail":
                if rep.rows < 3 or model._short_trace_for(rep.rows) is None or not model.snap_rows_ok(rep.bucket):
                    return None, None
            elif rep.kind == "mbig":
                if model._short_trace_for(rep.rows, bucket=rep.bucket) is None or not model.snap_rows_ok(rep.bucket):
                    return None, None
            elif not model.snap_rows_ok(rep.rows):  # a chunk-trace replay
                return None, None
        if any(rep.kind in ("tail", "mbig") for rep in replays):
            trash = getattr(model, "prefill_trash_block", None)
            nreal = -(-Tp // int(self.block_size))
            if trash is not None and bool((torch.as_tensor(pt_row).reshape(-1)[:nreal] == int(trash)).any()):
                return None, None  # a short trace would decline this request (trash block collision): legacy split
        # The cheaper of the two plans by the replay cost model: merging small segments can round them up to a bigger
        # short-trace bucket, and the whole call's tail can be 1-2 rows (an eager masked bucket) where the last segment
        # of the split plan never is.
        def _mb(n_, a_):
            ch = self._masked_big_choice(n_, a_)
            return None if ch is None or ch[0] == "old" else int(ch[0][1:])

        cost_one = prefill_plan.plan_cost(s0, Tp, points, True, big=big, chunk=chunk, masked_choice=_mb)
        cost_split = prefill_plan.plan_cost(s0, Tp, points, False, big=big, chunk=chunk, masked_choice=_mb)
        if cost_one >= cost_split:
            self.intrace_declined = getattr(self, "intrace_declined", 0) + 1
            return None, None
        return _SnapRun(model, cache, prompt[0], points), choice

    _MASKED_BIG_LOG_FIRST = 20  # INFO-log the first N plan decisions
    _MASKED_BIG_COUNT_EVERY = 64  # then the plan counters every N decisions

    def _masked_big_choice(self, n, a):
        """QWEN36_PREFILL_MASKED_BIG choice for a segment [a, a + n), n >= 2048, without side effects: after the
        n // 8192 full big blocks, the remainder r runs by the cheapest plan of prefill_plan.plan_remainder (A[M] +
        B[M] * depth, depth = absolute start of the remainder) over the captured masked buckets. Returns (plan, costs,
        r, depth) or None (flag off, no big chunk trace of 8192, no remainder, no masked bucket captured)."""
        if not prefill_masked_big_enabled():
            return None
        model = self.model
        big = model.big_chunk_rows()
        if big != 8192:
            return None
        n_big = n // big
        r = n - n_big * big
        if r <= 0:
            return None
        depth = a + n_big * big
        avail = tuple(bk for bk in prefill_plan.MASKED_BIG_BUCKETS if model._short_trace_for(r, bucket=bk) is not None)
        if not avail:
            return None
        plan, costs = prefill_plan.plan_remainder(r, depth, buckets=avail)
        return plan, costs, r, depth

    def _masked_big_plan(self, n, a, choice=None):
        """Bucket (4096 / 8192) the masked-big remainder of segment [a, a + n) replays, or None (the existing 2048 chunk
        replays + masked tail). ``choice``: _masked_big_choice's result when the caller already made it. Counts every
        decision in self.masked_big_plans and logs the first 20 at INFO (then the counters every 64)."""
        if choice is None:
            choice = self._masked_big_choice(n, a)
        if choice is None:
            return None
        plan, costs, r, depth = choice
        plans = self.__dict__.setdefault("masked_big_plans", {"old": 0, "m4096": 0, "m8192": 0})
        plans[plan] = plans.get(plan, 0) + 1
        total = sum(plans.values())
        if total <= self._MASKED_BIG_LOG_FIRST:
            logger.info(
                f"[spec_engine] masked-big split #{total}: r={r} depth={depth} -> {plan} costs "
                + " ".join(f"{k}={v:.3f}s" for k, v in costs.items())
            )
        elif total % self._MASKED_BIG_COUNT_EVERY == 0:
            logger.info(f"[spec_engine] masked-big plans after {total} decisions{self._masked_big_note_text()}")
        return None if plan == "old" else int(plan[1:])

    def _intrace_note_text(self):
        """Tail of the periodic spec log line (QWEN36_PREFILL_INTRACE_SNAP, once a prefill ran single-segment): the
        single-segment prefills and the in-replay snapshots they saved."""
        n = self.__dict__.get("intrace_prefills")
        if not n:
            return ""
        return (
            f"; in-replay snapshots {self.__dict__.get('intrace_snapshots', 0)} in {n} single-segment prefills"
            f" (split kept as cheaper: {self.__dict__.get('intrace_declined', 0)})"
        )

    def _masked_big_note_text(self):
        """Tail of the periodic spec log line: counts of the remainder plans chosen (flag on, once a decision was made)."""
        plans = self.__dict__.get("masked_big_plans")
        if not plans:
            return ""
        return "; masked-big plans " + " ".join(f"{k}={v}" for k, v in sorted(plans.items()))

    def _fast_segment(
        self,
        slot,
        mtp_list,
        prompt,
        pt_row,
        a,
        b,
        final,
        carry,
        on_chunk,
        captured_row,
        decline_ok,
        vision_tokens=None,
        snap=None,
        mb_choice=None,
        _masked_big=True,
    ):
        """Traced prefill of the prompt segment [a, b) (a % 128 == 0) on the bound B=1 GDN scratch: full chunks by the
        chunk trace, the tail by the spec short trace (drafter warm-up fused) or the eager masked bucket. carry: continue
        the bound GDN state (a > 0 or a restored snapshot) instead of zeroing it. ``mtp_list`` = the token list the
        drafter fuses from (the whole prompt for the final segment, prompt[:b+1] otherwise). Returns host float logits
        [vocab] (meaningful for the final segment only), or None when decline_ok and no trace serves a short segment.
        ``vision_tokens``: the multimodal prompt's packed vision rows; each replay / chunk splices the rows after the
        image placeholders before its own start (model._vis_row_offset_for over the whole prompt).

        ``snap`` (QWEN36_PREFILL_INTRACE_SNAP, the single-segment plan of _snap_run_plan): every replay below stages
        the save points inside it as in-replay snapshots and saves them (and a point at its end) right after it.
        QWEN36_PREFILL_MASKED_BIG (text only, n >= 2048): after the n // 8192 full big blocks the remainder runs as ONE
        masked 4096 / 8192 spec short trace when the cost model finds it cheaper than 2048 replays + the masked tail
        (``mb_choice``: the caller's _masked_big_choice for this segment). ``_masked_big=False`` (internal) keeps the
        existing plan, used for the fallback after a declined replay."""
        model = self.model
        vis_kw = {} if vision_tokens is None else dict(vision_tokens=vision_tokens)
        snap_kw = {} if snap is None else dict(snap=snap)
        chunk = model._chunked_chunk_size or 2048
        n = b - a
        num_full, tail = n // chunk, n % chunk
        mb = None  # masked big bucket (4096 / 8192) serving the remainder after the full big blocks (flag only)
        if num_full > 0 and _masked_big and vision_tokens is None and prefill_masked_big_enabled():
            mb = self._masked_big_plan(n, a, choice=mb_choice)
        if mb is not None:
            big = model.big_chunk_rows()
            n_big = n // big
            cs = a + n_big * big
            r = b - cs
            if n_big:
                if a == 0 and not carry:
                    model._build_request_rope(prompt, vision_tokens)
                model._prefill_traced_chunked_tp(
                    prompt,
                    model._fit_spec_page_table(pt_row, self.nb_prefill),
                    b,
                    n_big * big // chunk,
                    chunk,
                    0,
                    on_chunk=on_chunk,
                    run_tail=False,
                    start=a,
                    reset_gdn=not carry,
                    allow_big=True,
                    **snap_kw,
                )
            host = self._replay_spec_short(
                slot,
                mtp_list,
                prompt[:, cs:] if n_big else prompt[:, a:],
                pt_row,
                r,
                cs,
                True if n_big else carry,
                captured_row,
                bucket=mb,
                **snap_kw,
            )
            if host is not None:
                return host
            # trash-block collision (nothing was staged): the existing plan serves the rest of the segment
            return self._fast_segment(
                slot,
                mtp_list,
                prompt,
                pt_row,
                cs,
                b,
                final,
                True if n_big else carry,
                on_chunk,
                captured_row,
                decline_ok if not n_big else False,
                vision_tokens=vision_tokens,
                _masked_big=False,
                **snap_kw,
            )
        if num_full == 0:
            # `prompt[:, a:]` (not [:, a:b]): the replay slices the first n tokens, and at a == 0 builds the request RoPE
            # from what it gets, which must be the whole prompt when later segments follow.
            host = self._replay_spec_short(
                slot,
                mtp_list,
                prompt[:, a:],
                pt_row,
                n,
                a,
                carry,
                captured_row,
                **self._vis_replay_kw(prompt, a, vision_tokens),
                **snap_kw,
            )
            if host is None:
                if decline_ok:
                    return None  # no spec short trace for this length / trash-block collision: nothing staged, eager
                assert snap is None or not snap.inside(a, n), "in-replay snapshot points in an eager replay"
                pt_rec = model._fit_spec_page_table(pt_row, self.nb_prefill)
                logits, hidden = model.prefill_masked_bucket(
                    prompt[:, a:b],
                    pt_rec,
                    actual_len=n,
                    chunk_start=a,
                    return_hidden=True,
                    **self._vis_replay_kw(prompt, a, vision_tokens),
                )
                on_chunk(hidden, a, n)
                ttnn.deallocate(hidden)
                host = self._read_replicated_logits(logits)
            return host
        if a == 0 and not carry:
            # text: 1D RoPE, rope_delta 0; multimodal: M-RoPE with the grid (the chunk loop slices it)
            model._build_request_rope(prompt, vision_tokens)
        pt_rec = model._fit_spec_page_table(pt_row, self.nb_prefill)
        cs = a + num_full * chunk
        tail_tr = model._short_trace_for(tail) if tail >= 3 else None
        # Full chunks by replay (with on_chunk). For a > 0 the tail is run here, never by run_tail=True (its tail slice
        # ignores `start`).
        if a == 0 and tail > 0 and tail_tr is None:
            # The uncached long-prompt path: the chunk trace replays every full chunk and the eager masked-bucket tail
            # runs inside it (its tail slice is right for start == 0).
            assert snap is None or not snap.inside(cs, tail), "in-replay snapshot points in an eager tail"
            logits = model._prefill_traced_chunked_tp(
                prompt,
                pt_rec,
                b,
                num_full,
                chunk,
                tail,
                on_chunk=on_chunk,
                run_tail=True,
                allow_big=True,
                **vis_kw,
                **snap_kw,
            )
            return self._read_replicated_logits(logits)
        model._prefill_traced_chunked_tp(
            prompt,
            pt_rec,
            b,
            num_full,
            chunk,
            tail,
            on_chunk=on_chunk,
            run_tail=False,
            start=a,
            reset_gdn=not carry,
            allow_big=True,
            **vis_kw,
            **snap_kw,
        )
        if tail == 0:
            if (
                not final
            ):  # on_chunk already saw the last chunk in the replay loop; a mid-prompt segment has no logits use
                return torch.zeros(0)
            if prefill_mtp_lean_enabled():
                # QWEN36_PREFILL_MTP_LEAN: the replay loop above already ran on_chunk on the last chunk (drafter K/V warm +
                # the slot b-1 feed-row capture), so it is not run again; a non-final vLLM chunk reads no logits.
                if getattr(self, "_skip_logits_read", False):
                    return torch.zeros(0)
                last = getattr(model, "_last_chunk", None)
                out, rows = last if last is not None else (model._chunked_trace_output, chunk)
                return self._read_replicated_logits(model._masked_bucket_logits_tp(out, rows, rows))
            # Exact multiple: the last chunk's output gives the logits and the slot b-1 feed row.
            last = getattr(model, "_last_chunk", None)  # (output, rows); fakes leave it unset
            out, rows = last if last is not None else (model._chunked_trace_output, chunk)
            logits = model._masked_bucket_logits_tp(out, rows, rows)
            on_chunk(out, cs - rows, rows)
            return self._read_replicated_logits(logits)
        host = None
        if tail_tr is not None:
            host = self._replay_spec_short(
                slot,
                mtp_list,
                prompt[:, cs:b],
                pt_row,
                tail,
                cs,
                True,
                captured_row,
                **self._vis_replay_kw(prompt, cs, vision_tokens),
                **snap_kw,
            )
        if host is None:  # no short trace for the tail (< 3 tokens / uncaptured bucket) or a trash-block collision
            assert snap is None or not snap.inside(cs, tail), "in-replay snapshot points in an eager tail"
            logits, hidden = model.prefill_masked_bucket(
                prompt[:, cs:b],
                pt_rec,
                actual_len=tail,
                chunk_start=cs,
                return_hidden=True,
                **self._vis_replay_kw(prompt, cs, vision_tokens),
            )
            on_chunk(hidden, cs, tail)
            ttnn.deallocate(hidden)
            host = self._read_replicated_logits(logits)
        return host

    def _vis_replay_kw(self, prompt, start, vision_tokens):
        """Vision kwargs (vision_tokens, vis_row_offset) for model._replay_short_trace / prefill_masked_bucket over the
        segment starting at absolute ``start`` of the whole ``prompt``: {} for text (the call is unchanged), else the
        rows and the number of placeholder tokens before ``start``."""
        if vision_tokens is None:
            return {}
        return dict(vision_tokens=vision_tokens, vis_row_offset=self.model._vis_row_offset_for(prompt, start))

    def _replay_spec_short(
        self,
        slot,
        prompt_list,
        tokens,
        pt_row,
        L,
        chunk_start,
        carry,
        captured_row,
        vision_tokens=None,
        vis_row_offset=0,
        bucket=None,
        snap=None,
    ):
        """One replay of the spec short trace over the prompt segment [chunk_start, chunk_start + L) = the prompt's last
        segment (slot Tp-1 is its last real row): base forward + MTP KV warm-up + feed row, then the feed row of slot
        Tp-1 is copied into _last_hidden[slot]. Returns host float logits [vocab] or None (no trace / trash collision).
        ``bucket``: a masked big bucket (QWEN36_PREFILL_MASKED_BIG). ``snap`` (QWEN36_PREFILL_INTRACE_SNAP): the save
        points inside the replay are snapshotted by it and saved right after it.
        """
        model = self.model
        bkt_kw = {} if bucket is None else dict(bucket=bucket)  # only the masked-big replay passes it
        tr = model._short_trace_for(L, **bkt_kw)
        if tr is None:
            return None
        inputs = mtp_prefill_inputs(
            prompt_list, chunk_start, tr.bucket, pt_row, self.block_size, self._scratch_block, tr.buf_blocks
        )
        vis_kw = {} if vision_tokens is None else dict(vision_tokens=vision_tokens, vis_row_offset=vis_row_offset)
        inside = snap.inside(chunk_start, L) if snap is not None else []
        if snap is not None:
            vis_kw["snap_rows"] = [p - chunk_start for p in inside]
        tr = model._replay_short_trace(
            tokens,
            pt_row,
            L,
            strict=False,
            chunk_start=chunk_start,
            carry_gdn=carry,
            spec_inputs=inputs,
            **bkt_kw,
            **vis_kw,
        )
        if tr is None:
            # The trash block (vLLM's null block 0) is one of this request's real blocks: the traced prefill would park
            # the bucket's padded K/V in it, so the eager path serves this request. vLLM never allocates block 0; a
            # harness that does must also keep it out of every OTHER request's blocks, which the engine cannot check.
            if not getattr(self, "_trash_collision_warned", False):
                self._trash_collision_warned = True
                logger.warning(
                    f"[spec_engine] fast prefill declined for slot {slot} (Tp {len(prompt_list)}): trash block "
                    f"{getattr(model, 'prefill_trash_block', None)} is one of the request's real blocks; eager prefill "
                    "for this request"
                )
            assert not inside, "in-replay snapshot points in a declined short replay"
            return None
        if snap is not None:
            snap.after(chunk_start, L, inside)  # device-side saves, ordered behind the (drained) replay
        assert tr.last_feed.dtype == self._feed_dtype and tr.last_feed.shape[-1] == self._dim_frac, (
            f"feed row {tr.last_feed.dtype} {tuple(tr.last_feed.shape)} != buffer {self._feed_dtype} "
            f"dim/tp={self._dim_frac}"
        )
        ttnn.copy(tr.last_feed, self._last_hidden[slot])
        captured_row[0] = True
        if getattr(self, "_skip_logits_read", False):  # non-final chunk: no logits use
            return torch.zeros(0)
        if model._logits_read_comp is None:
            model._logits_read_comp = ttnn.ConcatMeshToTensor(self.mesh, dim=-1)
        lg = ttnn.to_torch(tr.logits, mesh_composer=model._logits_read_comp)
        return lg.reshape(-1)[: self.vocab].float()

    def _read_replicated_logits(self, logits_dev):
        """Replicated device logits [1, 1, vocab] (one all-gathered copy per device) -> host float [vocab], freed."""
        if getattr(self, "_skip_logits_read", False):  # non-final prefill chunk: no logits use
            ttnn.deallocate(logits_dev)
            return torch.zeros(0)
        lt = self._replicated_to_torch(logits_dev)
        ttnn.deallocate(logits_dev)
        return lt.reshape(-1)[: self.vocab].float()

    def _replicated_to_torch(self, logits_dev):
        """Host copy of replicated device logits. Today: every device's copy concatenated on dim 0 (the callers keep the
        first ``vocab`` values = device 0's copy). QWEN36_JOIN_LOGITS_ONE_COPY: device 0's copy alone (the same values;
        1/num_devices of the D2H)."""
        if join_logits_one_copy_enabled():
            return ttnn.to_torch(ttnn.get_device_tensors(logits_dev)[0])
        return ttnn.to_torch(logits_dev, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0))

    def _note_resume_fallback(self, prompt_list, start_pos, Tp, plan, final):
        """Count (and log) a prefill with start_pos > 0 that resumes from a GDN snapshot below spec_resume_cap (the
        snapshot at the cap was evicted or never saved): correct but slower. Chunked prefill expects the e-128 snapshot.
        """
        if start_pos is None or int(start_pos) <= 0 or self.B <= 1:
            return
        from models.demos.blackhole.qwen36.tt.prefix_cache import spec_resume_cap

        cap = spec_resume_cap(int(start_pos), Tp)
        s = 0 if plan is None else int(plan[0])
        c = self._chunk_cache()
        if c is not None and hasattr(c, "record_resume"):
            c.record_resume(start_pos, cap, s, None if plan is None or plan[1] is None else plan[1][1])
        if s >= cap:
            return
        self.resume_fallbacks = getattr(self, "resume_fallbacks", 0) + 1
        # a plain APC miss is routine; with chunked prefill it means the e-128 snapshot was evicted
        (logger.warning if os.environ.get("QWEN36_SPEC_CHUNKED_PREFILL", "0") == "1" else logger.debug)(
            f"[spec_engine] prefix resume fallback #{self.resume_fallbacks}: start_pos={start_pos} Tp={Tp} "
            f"({'final' if final else 'non-final'}) resumes at {s}, wanted {cap} (snapshot evicted?)"
        )

    # --- Option B (QWEN36_SPEC_CHUNK_EXACT_RESUME): exact resume of a chunked prefill at the previous chunk end ---
    def _chunk_prog(self):
        """engine row -> (e, digest of prompt[:e], page-table signature) of a row whose last prefill was a non-final chunk
        [., e) run by THIS engine in Option B mode (its snapshot at e and _last_hidden[row] = target hidden row of slot e-1).
        Set by a non-final chunk, consumed (popped) by the next prefill of the row, dropped on release."""
        return _chunk_prog_of(self)

    @staticmethod
    def _chunk_digest(prompt_list, e):
        import hashlib

        import numpy as np

        if isinstance(prompt_list, torch.Tensor):
            # QWEN36_PREFILL_HOST_TOLIST: the contiguous int32 prompt row: its bytes are np.asarray(ids, int32)'s
            return hashlib.blake2b(prompt_list.reshape(-1)[:e].contiguous().numpy().tobytes(), digest_size=16).digest()
        return hashlib.blake2b(np.asarray(prompt_list[:e], dtype=np.int32).tobytes(), digest_size=16).digest()

    def _chunk_pt_sig(self, pt_row, e):
        """The page-table blocks holding positions [0, e): the KV (and drafter KV) the exact resume relies on."""
        n = -(-int(e) // int(self.block_size))
        return tuple(int(x) for x in torch.as_tensor(pt_row).reshape(-1)[:n].tolist())

    def _chunk_exact_resume(self, prog, prompt_list, start_pos, Tp, pt_row, vision_tokens, exact_mode):
        """e when this prefill may resume exactly at e = start_pos (see _chunk_prog), else None (normal resume path:
        Option A). Checked: Option B on, text row, B > 1, the previous chunk was run here for this row and ended at e,
        same prompt tokens [0, e) and same page-table blocks, >= 3 tokens left (the short-tail floor of the plans)."""
        if not exact_mode or prog is None or start_pos is None or vision_tokens is not None or self.B <= 1:
            return None
        e0, dig, sig = prog
        ok = (
            int(start_pos) == e0
            and Tp - e0 >= 3
            and self._chunk_digest(prompt_list, e0) == dig
            and self._chunk_pt_sig(pt_row, e0) == sig
        )
        if not ok:
            self.chunk_exact_misses = getattr(self, "chunk_exact_misses", 0) + 1
            self._chunk_bump("refusals")
            logger.warning(
                f"[spec_engine] chunk exact resume refused: start_pos={start_pos} progress end={e0} Tp={Tp} "
                f"(prompt / page table / length mismatch): Option A resume; pending progress of another request dropped: "
                f"progress digest {dig.hex()[:8]} (end {e0}), this prompt digest[:{e0}] "
                f"{self._chunk_digest(prompt_list, min(e0, Tp)).hex()[:8]} (Tp {Tp})"
            )
            return None
        return e0

    def _warm_resume_slot(self, slot, token, pos, pt_row):
        """Write the drafter KV of slot ``pos`` (= e-1) from (_last_hidden[slot], ``token`` = prompt[e]) through the
        request's own page table ``pt_row`` ([1, nb] host int32). The one-row drafter decode that warms slot Tp-1 today
        (_warm_mtp_last); text rows: M-RoPE delta 0."""
        pt_tt = ttnn.from_torch(
            torch.as_tensor(pt_row).to(torch.int32).reshape(1, -1).contiguous(),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        try:
            self._warm_mtp_last(self._last_hidden[slot], int(token), int(pos), pt_tt)
        finally:
            ttnn.deallocate(pt_tt)

    def _reprefill(self, slot):
        """Re-run a fresh slot's prefill (a seed overwrote its prefill GDN state while it was left out)."""
        prompt, row, *rest = self._prompts[slot]
        logger.info(f"[spec_engine] re-prefill engine row {slot} ({len(prompt)} tokens)")
        # Same vLLM start_pos as the original prefill (its cached blocks are still the request's).
        vis_kw = {} if slot not in _row_vision_of(self) else dict(vision_tokens=_row_vision_of(self)[slot])
        if rest:
            self._prefill_row(slot, prompt, row, start_pos=rest[0], **vis_kw)
        else:
            self._prefill_row(slot, prompt, row, **vis_kw)

    def _reconcile_fresh_rows(self, fresh_now, start_pos):
        """Make every fresh row's prefilled length equal to the position its first verify starts at.

        A fresh row's first verify carries start_pos = vLLM's num_computed_tokens = the prompt length. vLLM pads a
        new request whose prefix-cache hit leaves exactly ONE uncached token to 1 + K scheduled tokens ("pad new
        decode requests to uniform spec decoding size", placeholder drafts [-1] * K that update_from_output rolls
        back as rejected), and the runner turns that into prompt_lens = num_computed_tokens + 1 + K: the row was
        prefilled over K tokens past the prompt (whatever sat in the token buffer). The durable GDN state cannot
        un-read tokens, so the row is prefilled again at the true length: the stored prompt cut at start_pos with the
        same vLLM start_pos (the cached blocks are the request's; the GDN snapshots the long prefill saved are keyed by
        the real tokens, so the re-prefill usually resumes from one). The paged KV and drafter KV past the prompt are
        rewritten by the first verify / drafter steps. A start_pos ABOVE the prefilled length is a prompt the engine
        never saw: raise. Returns ``fresh_now`` with the corrected lengths (``_fresh`` is updated by the re-prefill)."""
        out = dict(fresh_now)
        for u, Tp in sorted(fresh_now.items()):
            sp = int(start_pos[u])
            if sp == Tp:
                continue
            if sp > Tp or sp <= 0:
                raise ValueError(
                    f"[spec_engine] slot {u}: first verify at start_pos {sp} but the row was prefilled with {Tp} "
                    f"tokens; the engine never saw tokens [{Tp}, {sp}) (a partial prompt is not supported)"
                )
            if u not in self._prompts:
                raise RuntimeError(f"[spec_engine] slot {u}: prefilled {Tp} tokens, verify at {sp}, no stored prompt")
            prompt, row, *rest = self._prompts[u]
            vstart = rest[0] if rest else None
            if vstart is not None and vstart >= sp:
                raise ValueError(
                    f"[spec_engine] slot {u}: vLLM start_pos {vstart} of the prefill is not below the prompt length "
                    f"{sp} (prefilled {Tp})"
                )
            logger.warning(
                f"[spec_engine] slot {u}: prefilled {Tp} tokens but the first verify starts at {sp}: the last {Tp - sp} "
                f"prefill token(s) were not part of the prompt (vLLM pad_spec_decode placeholders); re-prefilling the "
                f"row with {sp} tokens (vLLM start_pos {vstart})"
            )
            vis_kw = {} if u not in _row_vision_of(self) else dict(vision_tokens=_row_vision_of(self)[u])
            self._prefill_row(u, prompt[:sp], row, start_pos=vstart, **vis_kw)
            out[u] = self._fresh[u]
        return out

    def _capture_last_row(self, feed, i, slot):
        """feed row i -> self._last_hidden[slot], fixed-shape programs only (a per-length slice would
        compile per prompt length). Exact like _set_anchor."""
        bucket = feed.shape[-2]
        assert bucket in self._last_sel, f"no last-row selector for bucket {bucket}"
        assert (
            feed.dtype == self._feed_dtype and feed.shape[-1] == self._dim_frac
        ), f"feed {feed.dtype} {tuple(feed.shape)} != buffer {self._feed_dtype} dim/tp={self._dim_frac}"
        self._stage(last_row_selector(i, bucket), self._last_sel[bucket], ttnn.bfloat16, ttnn.TILE_LAYOUT)
        row = ttnn.matmul(self._last_sel[bucket], feed, compute_kernel_config=self._anchor_cc)
        ttnn.copy(row, self._last_hidden[slot])
        ttnn.deallocate(row)

    # --------------------------------------------------------------------- #
    # Verify
    # --------------------------------------------------------------------- #
    def verify(
        self,
        tokens,
        start_pos,
        accepted_counts,
        page_table,
        read_logits=False,
        active=None,
        num_valid_drafts=None,
        read_topk=False,
        narrow=False,
        fused=False,
        sampling=None,
    ):
        """Replay the verify trace over [B, T] tokens at start_pos[u] + j. The first verify after prefill is
        the seed (see _seed): row 0 holds the first generated token, other rows are filler, accepted_counts = 1.
        Fresh rows may join running ones (see _join). A live row left out is parked on host and restored when it
        returns; a fresh row left out defers its seed.
        ``active`` (bool [B], None = all): False rows are padding and run over the null block.
        ``num_valid_drafts`` ([B], None = all columns real): the target's KV writes use verify_write_table;
        the drafter keeps the full table.
        Returns (ids int32 [B, T], logits [B, T, vocab] or None). With ``read_topk`` (QWEN36_VERIFY_DEVICE_TOPK;
        exclusive with ``read_logits``) the second item is a TopKCandidates [B, T, M] instead: in steady state
        from the topk trace (merge_shard_topk) -- under fast join that covers a fresh row's first step too, since
        it runs as a 0-draft row inside the same replay (column 0 real, the filler columns ignored by the runner
        like its argmax / logits columns) -- and on an eager seed / join step (QWEN36_SPEC_FAST_JOIN=0) from the
        host logits those steps read anyway (exact_topk), which are then cached for fetch_verify_logits.
        ``narrow`` (with ``read_logits``; QWEN36_SPEC_HOSTSET): only column 0 of each row is wanted (a no-draft step), so
        the replay reads [B, 1, V] logits (column 0 sliced on the device) instead of [B, T, V]; the returned logits
        are [B, 1, V] when that read was served (otherwise [B, T, V] as always).
        ``fused`` (spec_mode "fused_sample", greedy rows; needs ``num_valid_drafts``): returns (ids, None, FusedStep)
        instead; the step either ran the device accept + drafter in this call (``FusedStep.drafts`` set) or fell back
        to the argmax verify plus the host walk (see _verify_fused). ``sampling`` (with ``fused``, engine rows; stage S2): the
        sampled-row controls and pre-drawn uniforms of decode_forward's ``spec_sampling``."""
        assert self.captured, "verify before capture"
        if fused and (read_logits or read_topk or num_valid_drafts is None):
            raise ValueError("verify(fused=True) needs num_valid_drafts and no logits / topk read")
        if sampling is not None and not fused:
            raise ValueError("verify(sampling=...) needs fused=True")
        if read_topk and read_logits:
            raise ValueError("verify: read_logits and read_topk are exclusive")
        B, T = self.B, self.T
        tokens = self._ints(tokens, (B, T))
        start_pos = self._ints(start_pos, (B,))
        mi_prev = self._ints(accepted_counts, (B,)) - 1
        assert bool(((mi_prev >= 0) & (mi_prev < T)).all()), f"accepted_counts must be in [1, {T}]"
        self._last_cap_blocks = int(torch.as_tensor(page_table).shape[-1])
        pt = self._fit(page_table)
        active = (
            torch.ones(B, dtype=torch.bool) if active is None else torch.as_tensor(active, dtype=torch.bool).reshape(B)
        )
        # QWEN36_SPEC_HOSTSET: plain bools for the per-row loops (a tensor index per row costs ~2 us each)
        act = active.tolist() if spec_hostset_enabled() else active
        rows = [u for u in range(B) if act[u]]
        self._check_finishing_rows(rows)
        unknown = [u for u in rows if u not in self._active]
        if unknown:
            raise ValueError(f"rows {unknown} carry a request that was never prefilled into that slot")
        if (
            self._def_ctx is None
        ):  # deferred serving decided the bucket in _decode_forward_deferred (before the chain reason)
            self._bucket_step(rows)
        # contract 4e: a live row left out of a step keeps its GDN state on host until it returns
        to_park = [
            u for u in range(B) if not act[u] and u in self._active and u not in self._fresh and u not in self._parked
        ]
        if to_park:
            t0 = time.perf_counter()
            saved = self._park_save(to_park)
            for u in to_park:
                self._parked[u] = saved[u]
            self._park_note("park", t0)
            for u in to_park:
                logger.info(
                    f"[spec_engine] parked slot {u} (left out of a step), {self._park_last_ms:.1f} ms for {len(to_park)} row(s)"
                )
            self._chain_ev = self._chain_ev or "park"
        back = [u for u in rows if u in self._parked]
        if back:
            self._chain_ev = self._chain_ev or "unpark"
        if back:
            t0 = time.perf_counter()
            states = [
                {u: parked_rows_host(*self._parked[u][li], int(mi_prev[u]), T) for u in back}
                for li in range(len(self._gdn))
            ]
            self._park_load(states)
            for u in back:
                del self._parked[u]
            self._park_note("unpark", t0)
            logger.info(f"[spec_engine] unparked slot(s) {back}, {self._park_last_ms:.1f} ms")
            # restored into ring slot 0: the returning row verifies from mi = 0
            mi_prev = mi_prev.clone()
            mi_prev[back] = 0
        if not bool(active.all()):
            pt = pt.clone()
            pt[~active] = 0  # vLLM null block
            mi_prev = mi_prev.clone()
            mi_prev[~active] = 0
        self._row_active = active.clone()
        self._last_pt = pt
        # padded columns may sit past the runner's KV reservation (max_model_len, ngram)
        vpt = pt if num_valid_drafts is None else verify_write_table(pt, start_pos, num_valid_drafts, self.block_size)
        assert int(start_pos.max()) + T <= table_capacity(
            self._last_cap_blocks, self.block_size
        ), "verify window past the paged KV capacity"

        fresh_now = {u: Tp for u, Tp in self._fresh.items() if act[u]}
        if fresh_now:
            self._chain_ev = self._chain_ev or "fresh"
            # a row prefilled past its prompt (vLLM pad_spec_decode placeholders) is re-prefilled at the true length
            fresh_now = self._reconcile_fresh_rows(fresh_now, start_pos)
        if fresh_now and self.fast_join:
            # QWEN36_SPEC_FAST_JOIN=1: seed the fresh rows' ring slot 0 / E_prev in place, then verify them in THIS
            # replay as 0-draft rows at mi = 0 next to the running rows (one forward for the whole step).
            fresh = self._seed_fresh_rows(tokens, start_pos, pt, fresh_now, rows)
            mi_prev = mi_prev.clone()
            mi_prev[fresh] = 0
        elif fresh_now:
            if self._bucketed():
                self._vb = self.B  # the eager seed / join runs the B-row traces
            # the seed rewrites every row's durable GDN state: fresh rows left out must re-prefill before their own seed
            self._fresh_dirty |= set(self._fresh) - set(fresh_now)
            for u in sorted(fresh_now):
                if u in self._fresh_dirty:
                    self._reprefill(u)
            # Each fresh user's slot Tp-1 MTP KV, then the seed (its filler rows are never run).
            self._stage_mtp_tables(pt)
            for u, Tp in sorted(fresh_now.items()):
                assert int(start_pos[u]) == Tp, f"slot {u}: start_pos {int(start_pos[u])} != prompt length {Tp}"
                self._warm_mtp_last(
                    self._last_hidden[u], int(tokens[u, 0]), Tp - 1, self._mtp_pt_rows[u], **self._row_kw(u)
                )
            fresh = sorted(fresh_now)
            for u in fresh:
                del self._fresh[u]
                self._prompts.pop(u, None)
                _row_vision_of(self).pop(u, None)  # no re-prefill from here on
                self._fresh_dirty.discard(u)
            if set(fresh) == set(rows):
                out = self._seed(tokens, start_pos, vpt, read_logits or read_topk or sampling is not None)
            else:
                out = self._join(
                    tokens, start_pos, mi_prev, vpt, active, fresh, read_logits or read_topk or sampling is not None
                )
            if fused:  # QWEN36_SPEC_FAST_JOIN=0: the eager seed / join read host logits; walk them on the host
                return self._fused_fallback_result(
                    out[0], tokens, num_valid_drafts, "eager_join", logits=out[1], sampling=sampling
                )
            return self._topk_from_logits(out) if read_topk else out

        args = (tokens.tolist(), start_pos.tolist(), mi_prev.tolist())
        kw = dict(read_logits=read_logits, page_tables=vpt, inactive=(~active).tolist())
        self._bucket_kw(kw)
        if read_topk:
            kw["read_topk"] = True  # only when asked: an older model.verify_traced has no such argument
        if narrow and read_logits and spec_hostset_enabled():
            kw["read_col0"] = True
        if fused:
            return self._verify_fused(tokens, start_pos, pt, active, num_valid_drafts, args, kw, sampling)
        if self._timing:
            ids, vfeed, lt = self._verify_split(*args, **kw)
        else:
            ids, vfeed, lt = self.model.verify_traced(*args, **kw)
        self._vfeed = vfeed
        if lt is not None and kw.get("read_col0") and getattr(self.model, "_vfy_last_col0", False):
            return torch.tensor(ids, dtype=torch.int32).reshape(B, T), lt.reshape(B, 1, -1)
        ids = torch.tensor(ids, dtype=torch.int32).reshape(B, T)
        if read_topk:
            self._topk_logits = None  # steady state: fetch_verify_logits replays the logits trace
            vals, idx = merge_shard_topk(lt[0], lt[1], self.vocab // self.model.num_devices, self.model._vfy_topk_m)
            return ids, TopKCandidates(vals.reshape(B, T, -1), idx.reshape(B, T, -1))
        return ids, (None if lt is None else lt.reshape(B, T, -1))

    def _seed_fresh_rows(self, tokens, start_pos, pt, fresh_now, rows):
        """QWEN36_SPEC_FAST_JOIN=1: make the fresh rows in ``fresh_now`` (engine row -> prompt length) ordinary verify
        rows. Writes each one's MTP KV of slot Tp-1 (_warm_mtp_last, as the eager seed did), then seeds its GDN ring
        slot 0 / E_prev row from the prefill state (rec_state, conv taps) in every GDN layer through ONE shared
        one-hot selector (fresh_row_select, staged once per call). The whole-slot ring write is used when every active
        row (``rows``) is fresh. Nothing advances rec_state or the taps, so a fresh row left out of this step keeps
        its prefill state and needs no re-prefill. Returns the sorted fresh engine rows; the caller runs them in the
        normal verify replay with mi_prev = 0 (their tokens are [t_Tp, pad..] and the runner reads column 0)."""
        from models.demos.blackhole.qwen36.tt.gdn.tp import fresh_row_mask, fresh_row_select

        B, T = self.B, self.T
        if spec_join_fast_enabled():
            # QWEN36_SPEC_JOIN_FAST: only the fresh rows' tables, one batched warm call (note 55)
            for u, Tp in fresh_now.items():
                assert int(start_pos[u]) == Tp, f"slot {u}: start_pos {int(start_pos[u])} != prompt length {Tp}"
            mwhy = self._mtp_warm_trace_why()
            if mwhy is None:
                # QWEN36_SPEC_JOIN_TRACE: the fixed B-row warm replays; _mtp_pt_rows[u] stays unstaged (propose restages
                # every table it reads before any use, the same as for the rows this eager branch leaves out)
                self._replay_mtp_warm(fresh_now, tokens, pt)
            else:
                if mwhy not in ("", "flag off"):
                    fb = self.__dict__.setdefault("join_trace_fallbacks", {})
                    fb[mwhy] = fb.get(mwhy, 0) + 1
                self._stage_mtp_tables_rows(pt, sorted(fresh_now))
                self._warm_mtp_last_rows(fresh_now, tokens, pt)
        else:
            self._stage_mtp_tables(pt)
            for u, Tp in sorted(fresh_now.items()):
                assert int(start_pos[u]) == Tp, f"slot {u}: start_pos {int(start_pos[u])} != prompt length {Tp}"
                self._warm_mtp_last(
                    self._last_hidden[u], int(tokens[u, 0]), Tp - 1, self._mtp_pt_rows[u], **self._row_kw(u)
                )
        fresh = sorted(fresh_now)
        for u in fresh:
            del self._fresh[u]
            self._prompts.pop(u, None)
            _row_vision_of(self).pop(u, None)  # no re-prefill from here on
            self._fresh_dirty.discard(u)
        if self._gdn:
            self._stage(fresh_row_select(fresh, B, T, self._kc), self._fj_sel, ttnn.bfloat16, ttnn.TILE_LAYOUT)
            why = self._join_trace_why()
            if why is None:
                # one trace replay for all layers: the mask (engine rows, as fresh_row_select) selects the fresh rows
                # only, equal to the eager partial path (also when every active row is fresh)
                p = self._gdn[0]._spec_win_par
                self._stage(
                    fresh_row_mask(fresh, B, *self._fj_mask_dims), self._fj_mask, ttnn.float32, ttnn.TILE_LAYOUT
                )
                ttnn.execute_trace(self.mesh, self._seed_trace[p], cq_id=0, blocking=False)
                self.join_trace_replays += 1
            else:
                fb = self.__dict__.setdefault("join_trace_fallbacks", {})
                fb[why] = fb.get(why, 0) + 1
                seed_rows = range(B) if set(fresh) == set(rows) else fresh
                for dn in self._gdn:
                    dn.seed_spec_rows_select(seed_rows, self._fj_sel)
        return fresh

    def _join_trace_why(self):
        """None when the GDN seed of this join can replay the seed trace; else the reason it runs eagerly:
        "flag off", "no trace", "taps route layer <i>" (seed_masked_ok() False), "mixed parity",
        "stale buffers" (a layer's ring / window / state buffer is not the one the trace was captured against)."""
        if not spec_join_trace_enabled():
            return "flag off"
        if self._seed_trace is None:
            return "no trace"
        for i, dn in enumerate(self._gdn):
            if not dn.seed_masked_ok():
                return f"taps route layer {i}"
        if len({dn._spec_win_par for dn in self._gdn}) != 1:
            return "mixed parity"
        cur = self._seed_buffers()
        if len(cur) != len(self._seed_trace_bufs) or any(
            len(a) != len(b) or any(x is not y for x, y in zip(a, b)) for a, b in zip(cur, self._seed_trace_bufs)
        ):
            return "stale buffers"
        return None

    def _topk_from_logits(self, out):
        """(ids, logits [B, T, vocab] fp32) of an eager seed / join step (QWEN36_SPEC_FAST_JOIN=0) -> (ids,
        TopKCandidates); the logits are kept (engine rows) for fetch_verify_logits, since no device replay can
        reproduce an eager seed step's rows. Under fast join a fresh row rides the normal replay, so its candidates
        come from the topk trace like every other row's and this helper is never reached."""
        ids, logits = out
        B, T = self.B, self.T
        self._topk_logits = logits
        vals, idx = exact_topk(logits.reshape(B * T, -1), self.model._vfy_topk_m)
        return ids, TopKCandidates(vals.reshape(B, T, -1), idx.reshape(B, T, -1))

    def fetch_verify_logits(self):
        """The full logits [B, T, vocab] fp32, in runner slots, of the last ``decode_forward(spec_mode="topk")``:
        the lazy fallback for rows the candidates cannot serve. Steady state (and every fast-join step, fresh rows
        included) replays the logits trace (model.fetch_verify_logits: one device round trip); after an eager seed /
        join step (QWEN36_SPEC_FAST_JOIN=0) the host copy is returned. Valid until the next decode_forward; the
        plugin walks acceptance before it proposes."""
        B, T = self.B, self.T
        if self._topk_logits is not None:
            logits = self._topk_logits
        else:
            logits = self.model.fetch_verify_logits().reshape(B, T, -1)[..., : self.vocab].to(torch.float32)
        return self._to_slots(logits)[0]

    def _seed_rows(self, feed):
        """[1,1,B,dim/tp] seed feed -> [1,1,B*T,dim/tp] with user u's row at u*T (exact one-hot matmul)."""
        B, T = self.B, self.T
        sel = torch.zeros(1, 1, B * T, B, dtype=torch.bfloat16)
        sel[0, 0, torch.arange(B) * T, torch.arange(B)] = 1
        sel_tt = ttnn.from_torch(
            sel,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        rows = ttnn.matmul(
            sel_tt, feed, compute_kernel_config=self._anchor_cc, dtype=feed.dtype, memory_config=feed.memory_config()
        )
        ttnn.deallocate(sel_tt)
        return rows

    def _seed(self, tokens, start_pos, pt, read_logits):
        """The seed: one eager T = 1 forward per user (model.seed_spec_step) advances the durable GDN state and
        writes KV at Tp, then the ring and E_prev are seeded from it. Its feed row lands in verify-feed row u*T,
        the next propose's anchor. (Seeding through row 0 of the verify trace drafts measurably worse.)
        Returns (ids, logits) like verify(); only column 0 is real."""
        B, T = self.B, self.T
        for dn in self._gdn:
            dn.sync_conv_win()  # a prefill replay writes the taps without flagging the window stale
        lt, feed = self.model.seed_spec_step(tokens[:, 0].tolist(), start_pos.tolist(), pt)
        for dn in self._gdn:
            dn.seed_spec_state()
        rows = self._seed_rows(feed)
        ttnn.copy(rows, self.model._vfy_rows_out)
        ttnn.deallocate(rows)
        ttnn.deallocate(feed)
        self._vfeed = self.model._vfy_rows_out
        lt = lt[:, : self.vocab]
        ids = torch.zeros(B, T, dtype=torch.int32)
        ids[:, 0] = lt.argmax(-1).to(torch.int32)
        logits = None
        if read_logits:
            logits = torch.zeros(B, T, self.vocab, dtype=torch.float32)
            logits[:, 0] = lt
        return ids, logits

    def _merge_seed_rows(self, feed, fresh, rows_out):
        """rows_out row u*T (u in fresh) <- feed row u; other rows kept.
        Exact one-hot matmuls (one program for any ``fresh``)."""
        B, T = self.B, self.T
        take = torch.zeros(1, 1, B * T, B, dtype=torch.bfloat16)
        keep = torch.zeros(1, 1, B * T, B * T, dtype=torch.bfloat16)
        for u in fresh:
            take[0, 0, u * T, u] = 1
        taken = {u * T for u in fresh}
        for r in range(B * T):
            if r not in taken:
                keep[0, 0, r, r] = 1
        self._stage(take, self._join_take, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        self._stage(keep, self._join_keep, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        kw = dict(compute_kernel_config=self._anchor_cc, dtype=rows_out.dtype, memory_config=rows_out.memory_config())
        a = ttnn.matmul(self._join_take, feed, **kw)
        b = ttnn.matmul(self._join_keep, rows_out, **kw)
        s = ttnn.add(a, b, memory_config=rows_out.memory_config())
        ttnn.copy(s, rows_out)
        for t in (a, b, s):
            ttnn.deallocate(t)

    def _join(self, tokens, start_pos, mi_prev, pt, active, fresh, read_logits):
        """Mixed step: verify the running rows, seed the fresh ones (see _seed), merge their feed rows.
        Returns (ids, logits) like verify()."""
        B, T = self.B, self.T
        fmask = torch.zeros(B, dtype=torch.bool)
        fmask[fresh] = True
        run = active & ~fmask
        vt, vs, vm, vpt = tokens.clone(), start_pos.clone(), mi_prev.clone(), pt.clone()
        vt[~run], vs[~run], vm[~run], vpt[~run] = 0, 0, 0, 0  # fresh rows run as padding
        args = (vt.tolist(), vs.tolist(), vm.tolist())
        kw = dict(read_logits=read_logits, page_tables=vpt, inactive=(~run).tolist())
        if self._timing:
            ids_v, _, lt_v = self._verify_split(*args, **kw)
        else:
            ids_v, _, lt_v = self.model.verify_traced(*args, **kw)
        ids = torch.tensor(ids_v, dtype=torch.int32).reshape(B, T)
        logits = None
        if read_logits:
            logits = lt_v.reshape(B, T, -1)[..., : self.vocab].to(torch.float32)
        for dn in self._gdn:
            dn.sync_conv_win()
        st, sp, spt = tokens[:, 0].clone(), start_pos.clone(), pt.clone()
        st[~fmask], sp[~fmask], spt[~fmask] = 0, 0, 0
        lt, feed = self.model.seed_spec_step(st.tolist(), sp.tolist(), spt)
        for dn in self._gdn:
            dn.seed_spec_rows(fresh)
        self._merge_seed_rows(feed, fresh, self.model._vfy_rows_out)
        ttnn.deallocate(feed)
        self._vfeed = self.model._vfy_rows_out
        lt = lt[:, : self.vocab]
        ids[fresh] = 0
        ids[fresh, 0] = lt[fresh].argmax(-1).to(torch.int32)
        if read_logits:
            logits[fresh] = 0
            logits[fresh, 0] = lt[fresh].to(torch.float32)
        return ids, logits

    def decode_forward(
        self,
        tokens,
        start_pos,
        *,
        num_valid_drafts,
        accepted_counts,
        page_table,
        spec_mode="argmax_ids",
        slot_remap=None,
        narrow_col0=False,
        spec_sampling=None,
        defer_read=False,
        chain_rows=None,
        chain_tip=None,
        finishing_slots=None,
    ):
        """Runner verify contract over a [B, 1+K] block (see sanitize_verify_block) in runner slots; slot_remap is
        applied after validation, then every per-row input is scattered to engine rows (_to_rows) and the outputs
        are gathered back (_to_slots). Returns VerifyResult. ``narrow_col0`` (spec_mode "logits", the narrow
        no-draft step; QWEN36_SPEC_HOSTSET): the caller uses only column 0, so ``logits`` may come back [B, 1, V].
        ``spec_sampling`` (spec_mode "fused_sample", stage S2; runner slots): dict temperature fp32 [B], top_k int32 [B],
        top_p fp32 [B], greedy bool [B], uniforms fp32 [B, K+2] (the plugin's pre-drawn table: col 0 bonus, 1..K accepts, K+1
        correction). None, or every row greedy: the S1 greedy step.
        ``defer_read`` (spec_mode "fused_sample"; QWEN36_SPEC_CHAIN + QWEN36_SPEC_FUSED, stage S4b): enqueue the whole step, no
        synchronize, and return VerifyResult(handle=FusedHandle); ``chain_rows`` (bool [B], runner order) / ``chain_tip`` (the tip_id of the
        newest fused step): rows whose placeholder tokens (-1), provisional positions and counts (-1) the device replaces from that step.
        See _decode_forward_deferred. ``finishing_slots`` (runner slots, QWEN36 plugin contract): requests left out of this step
        only because the step in flight ends them at max_tokens; freed without parking (see _apply_finishing)."""
        if spec_mode not in SPEC_MODES:
            raise ValueError(f"spec_mode {spec_mode!r} not in {SPEC_MODES}")
        if spec_sampling is not None and spec_mode != "fused_sample":
            raise ValueError(f"spec_sampling needs spec_mode 'fused_sample', got {spec_mode!r}")
        if defer_read:
            if (
                spec_sampling is not None
                and not spec_chain_sampled_enabled()
                and not bool(torch.as_tensor(spec_sampling["greedy"]).all())
            ):
                raise ValueError(
                    "defer_read=True with a non-greedy row in spec_sampling needs QWEN36_SPEC_CHAIN_SAMPLED=1 (sampled overflow is not "
                    "handled for deferred steps); run a sampled step synchronously (defer_read=False)"
                )
            return self._decode_forward_deferred(
                tokens,
                start_pos,
                num_valid_drafts,
                accepted_counts,
                page_table,
                spec_mode,
                slot_remap,
                spec_sampling,
                chain_rows,
                chain_tip,
                finishing_slots,
            )
        if chain_rows is not None or chain_tip is not None:
            raise ValueError("chain_rows / chain_tip need defer_read=True")
        self._fused_cache = None  # a fused step's drafts are good for the propose that follows it, nothing older
        if self._ch_g is not None:  # QWEN36_SPEC_CHAIN: take what the last fused step left (consumed by _chain_plan)
            self._chain_in, self._chain_prev = self._chain_prev, None
            self._chain_why, self._chain_dropped = self._chain_dropped, None
            self._chain_cand = self._chain_ev = None
            if spec_mode != "fused_sample":
                self._chain_in = None
                self._chain_drop("non_fused_step")
        clean, start0, active = sanitize_verify_block(tokens, start_pos, num_valid_drafts, self.K)
        if clean.shape[0] != self.B:
            raise ValueError(f"block has {clean.shape[0]} rows, engine has B={self.B}")
        if not bool(active.any()):
            raise ValueError("no active row in the verify block")
        self._apply_finishing(finishing_slots)  # pre-remap slots, before the slot map moves
        if slot_remap is not None:
            self.remap_slots(slot_remap)
            if (
                self._chain_in is not None
                and not self.host_slotmap
                and tuple(int(x) for x in slot_remap) != tuple(range(self.B))
            ):
                self._chain_in = None  # the device remap moved the engine rows
                self._chain_why = "slot_change"
        clean, start0, active, nv_rows, acc_rows, pt_rows = self._to_rows(
            clean, start0, active, num_valid_drafts, accepted_counts, page_table
        )
        self._check_finishing_rows([u for u in range(self.B) if bool(active[u])])
        self._no_draft = note_withheld_drafts(
            self._offered,
            self._no_draft,
            self._speculated,
            self._withheld_streak,
            nv_rows,
            self.no_draft_streak,
        )
        kw = {"read_topk": True} if spec_mode == "topk" else {}
        if narrow_col0 and spec_mode == "logits":
            kw["narrow"] = True
        if spec_mode == "fused_sample":
            sampling = None
            if spec_sampling is not None:
                names = ("temperature", "top_k", "top_p", "greedy", "uniforms")
                sampling = dict(zip(names, self._to_rows(*(spec_sampling[n] for n in names))))
                if bool(torch.as_tensor(sampling["greedy"]).all()):
                    sampling = None  # nothing sampled: the S1 step
            ids, _, fz = self.verify(
                clean,
                start0,
                acc_rows,
                pt_rows,
                active=active,
                num_valid_drafts=nv_rows,
                fused=True,
                sampling=sampling,
            )
            ids, committed, counts = self._to_slots(ids, fz.committed, fz.counts)
            if fz.drafts is not None:  # cache what the plugin's propose will ask for, in runner slots
                cpos = self._committed_positions(start0, active, fz.committed, fz.counts)
                cpos, act = self._to_slots(cpos, active)
                self._fused_cache = (fz.drafts, committed.clone(), cpos, counts.clone(), act.clone())
            if self._ch_g is not None:  # the next step may chain on this one once its propose took the cache
                self._chain_in = None
                if fz.drafts is not None and self._chain_cand is not None:
                    self._chain_prev, self._chain_cand = self._chain_cand, None
                else:
                    self._chain_drop("prev_not_fused")
            return VerifyResult(
                spec_mode,
                mask_unused_columns(ids, num_valid_drafts),
                accepted_token_ids=committed.to(torch.int32),
                accepted_counts=counts.to(torch.int32),
            )
        ids, logits = self.verify(
            clean,
            start0,
            acc_rows,
            pt_rows,
            read_logits=spec_mode == "logits",
            active=active,
            num_valid_drafts=nv_rows,
            **kw,
        )
        if spec_mode == "topk":
            ids, vals, idx = self._to_slots(ids, logits.values, logits.ids)
            return VerifyResult(
                spec_mode,
                mask_unused_columns(ids, num_valid_drafts),
                topk_values=vals,
                topk_ids=idx,
                fetch_logits=self.fetch_verify_logits,
            )
        ids, logits = self._to_slots(ids, logits)
        return VerifyResult(spec_mode, mask_unused_columns(ids, num_valid_drafts), logits)

    def _verify_split(self, tokens, positions, mi_prev, **kw):
        """verify_traced with the readback timed separately (hooks ttnn.get_device_tensors);
        sets ``last_verify_timing``."""
        orig = ttnn.get_device_tensors
        mark = []

        def hooked(*a, **k):
            if not mark:
                mark.append(time.perf_counter())
            return orig(*a, **k)

        ttnn.get_device_tensors = hooked
        t0 = time.perf_counter()
        try:
            out = self.model.verify_traced(tokens, positions, mi_prev, **kw)
        finally:
            ttnn.get_device_tensors = orig
        ttnn.synchronize_device(self.mesh)
        t1 = time.perf_counter()
        t_mark = mark[0] if mark else t1
        self.last_verify_timing = (t_mark - t0, t1 - t_mark)
        return out

    # --------------------------------------------------------------------- #
    # Fused greedy step (QWEN36_SPEC_FUSED, spec_mode "fused_sample")
    # --------------------------------------------------------------------- #
    @staticmethod
    def _committed_positions(start0, active, committed, counts):
        """The plugin's committed_positions for a commit: [B, T] p0 + 1 + j for j < count, -1 past it and on padding rows
        (model_runner._committed_positions), int64."""
        T = committed.shape[1]
        cpos = torch.as_tensor(start0).reshape(-1, 1) + 1 + torch.arange(T)[None, :]
        keep = (torch.arange(T)[None, :] < torch.as_tensor(counts).reshape(-1, 1)) & torch.as_tensor(active).reshape(
            -1, 1
        )
        return torch.where(keep, cpos, torch.full_like(cpos, -1)).to(torch.int64)

    def _fused_note(self, reason):
        if reason is not None:
            fb = self.__dict__.setdefault("fused_fallbacks", {})
            fb[reason] = fb.get(reason, 0) + 1
            self._chain_drop(reason)
        self.fused_steps += 1
        if self.fused_steps % 256 == 0:
            fb = self.__dict__.setdefault("fused_fallbacks", {})
            logger.info(
                f"[spec_engine] fused steps {self.fused_steps}, fallbacks {sum(fb.values())}: "
                + (", ".join(f"{k}={v}" for k, v in sorted(fb.items())) or "none")
                + f"; audited {self.fused_audited_steps}, boundary_diffs {self.fused_boundary_diffs}"
                + f", row_delta_mixed_steps {self.__dict__.get('row_delta_mixed_steps', 0)}"
                + (
                    f"; chain steps {self.chain_steps}, breaks "
                    + (
                        ", ".join(f"{k}={v}" for k, v in sorted(self.__dict__.get("chain_breaks", {}).items()))
                        or "none"
                    )
                    + f", audits {self.chain_audits}"
                    if self._ch_g is not None
                    else ""
                )
                + self._park_note_text()
                + self._join_trace_note_text()
                + self._bucket_note_text()
                + self._masked_big_note_text()
                + self._intrace_note_text()
            )

    def _join_trace_note_text(self):
        """Tail of the periodic fused line: seed-trace replays and the eager fallbacks by reason (flag on only)."""
        if not spec_join_trace_enabled():
            return ""
        fb = self.__dict__.setdefault("join_trace_fallbacks", {})
        return f"; join trace replays {self.join_trace_replays}, fallbacks " + (
            ", ".join(f"{k}={v}" for k, v in sorted(fb.items())) or "none"
        )

    def _park_note_text(self):
        """Tail of the periodic fused line: deferred chain breaks by reason, park / unpark counts and times, finishing rows."""
        out = ""
        db = self.__dict__.get("def_breaks", {})
        if db:
            out += (
                f"; deferred breaks {sum(db.values())} (" + ", ".join(f"{k}={v}" for k, v in sorted(db.items())) + ")"
            )
        for kind, (n, ms) in sorted(self.__dict__.get("park_stats", {}).items()):
            out += f"; {kind} n={n} total={ms:.1f} ms avg={ms / max(n, 1):.1f} ms"
        fin = self.__dict__.get("finishing_freed", 0)
        if fin:
            out += f"; finishing rows freed {fin}"
        return out

    def _fused_fallback_result(self, ids, tokens, num_valid, reason, logits=None, sampling=None):
        """Fallback tail: ``ids`` [B, T] (from the argmax verify) walked on the host with the greedy rule; no drafts cached.
        A step with ``sampling`` (stage S2) walks the full ``logits`` [B, T, V] or [B*T, V] instead (host_redo_walk, the
        plugin's uniforms)."""
        self._fused_note(reason)
        if sampling is not None:
            return ids, None, self._host_walk_step(logits, tokens, num_valid, sampling)
        committed, counts = fused_greedy_walk(ids, tokens, num_valid, self.K)
        return ids, None, FusedStep(committed.to(torch.int32), counts.to(torch.int32), None)

    def _host_walk_step(self, logits, tokens, num_valid, sampling):
        """The exact host walk of a sampled fused step on the full logits (engine rows): a FusedStep without drafts, so the
        propose that follows runs the normal path."""
        B, K = self.B, self.K
        committed, counts = host_redo_walk(
            torch.as_tensor(logits).reshape(B * (K + 1), -1)[:, : self.vocab],
            tokens,
            sampling,
            sampling["uniforms"],
            B=B,
            K=K,
            num_valid=num_valid,
        )
        return FusedStep(committed, counts, None)

    def _fused_ineligible(self, start_pos, nv, active):
        """(reason, aux): reason None when this step can run the device accept. The decline / skip decisions are all known
        before the step (no row's fit depends on m), else the step falls back."""
        B, K = self.B, self.K
        if self._fa is None or self._fa_trace is None or not self.captured or self._dr_trace is None:
            return "not_captured", None
        if B == 1 and int(start_pos[0]) + K > self.eager_reseed_len:  # the eager drafter may be chosen, whatever m is
            return "eager_drafter", None
        skip, skip_all = propose_skip_plan(self._no_draft, B)
        if skip_all and B == 1:
            return "skip_all", None
        delta, common, mixed = self._step_rope_delta(active)
        if (
            mixed and not self._row_delta_flag()
        ):  # flag 0: the device words carry one delta, the host verify one per row
            return "mixed_rope_delta", None
        try:
            aux = fused_accept_aux_host(
                nv,
                start_pos,
                active,
                skip,
                table_capacity(self._last_cap_blocks, self.block_size),
                K,
                int(self._scratch_block),
                delta,
                self.model.rope.max_seq_len,
                common_delta=common,
            )
        except ValueError:
            return "mixed_fit", None
        if bool(aux[2, :B].all()):
            return "all_declined", None
        if mixed:
            self.row_delta_mixed_steps = self.__dict__.get("row_delta_mixed_steps", 0) + 1
        return None, aux

    def _verify_fused(self, tokens, start_pos, pt, active, num_valid, args, kw, sampling=None):
        """The fused step after verify's host pre-work. Eligible: stage aux (+ page-table rows, + the MTP table), enqueue
        verify, accept and drafter traces, ONE synchronize, then read ids, the device commit and the draft ids, and finish the
        propose bookkeeping from the device commit. Else the argmax verify + host walk (no drafts).
        ``sampling`` (stage S2, engine rows, at least one non-greedy row): the sampled accept trace runs instead of the greedy
        one (_verify_fused_sampled); an ineligible sampled step walks the FULL logits on the host (never the greedy walk).
        """
        B, T, K = self.B, self.T, self.K
        nv = torch.as_tensor(num_valid).to(torch.int64).reshape(B)
        reason, aux = self._fused_ineligible(start_pos, nv, active)
        if sampling is not None:
            reason = reason or self._sampled_ineligible(sampling)
            if reason is not None:
                ids, vfeed, lt = self.model.verify_traced(*args, **dict(kw, read_logits=True))
                self._vfeed = vfeed
                ids = torch.tensor(ids, dtype=torch.int32).reshape(B, T)
                return self._fused_fallback_result(ids, tokens, nv, reason, logits=lt, sampling=sampling)
            return self._verify_fused_sampled(tokens, start_pos, pt, active, nv, args, kw, sampling, aux)
        if reason is not None:
            ids, vfeed, _ = self.model.verify_traced(*args, **kw)
            self._vfeed = vfeed
            return self._fused_fallback_result(torch.tensor(ids, dtype=torch.int32).reshape(B, T), tokens, nv, reason)
        if self._def_ctx is not None:  # decode_forward(defer_read=True): enqueue, no synchronize
            return None, None, self._def_enqueue("g", tokens, start_pos, pt, active, nv, args, kw, None, aux)
        fa, model = self._fa, self.model
        self._check_scratch(pt)
        declined = aux[2, :B].bool()
        chain = self._chain_plan(tokens, start_pos, active, nv, args, kw)
        if chain is None:
            fa.stage_aux(aux)
        else:  # QWEN36_SPEC_CHAIN: the overlay replaces the aux copy; the chain trace writes aux_in, the token and aux buffers
            expect = self._chain_stage(chain, tokens, start_pos, active, aux, args)
        if self._fa_pt_host is None or not torch.equal(self._fa_pt_host, pt):
            fa.stage_pt_rows(pt.repeat_interleave(T, dim=0))
            self._fa_pt_host = pt.clone()
        dpt = pt
        if bool(declined.any()):  # declined rows draft over an all-scratch table (as the propose path stages it)
            dpt = pt.clone()
            dpt[declined] = self._scratch_block
        self._stage_table(dpt, self._mtp_pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        if chain is None:
            handle = model.verify_stage_and_enqueue(*args, **kw)
        else:
            self._chain_exec(chain)
            handle = model.verify_stage_and_enqueue(*args, **kw, skip_tokens_h2d=True, skip_aux_h2d=True)
        ttnn.execute_trace(self.mesh, self._fa_trace, cq_id=0, blocking=False)
        ttnn.execute_trace(self.mesh, self._dr_trace_for(self._vb), cq_id=0, blocking=False)
        # Every non-blocking D2H read is enqueued BEFORE the single synchronize (a from_device(blocking=False) issued
        # after it is never waited on: its host tensor is unfilled). The reads below it are blocking to_torch.
        dr_handles = self._ids_enqueue_reads(self._dr_outs)
        ttnn.synchronize_device(self.mesh)  # the only one of the step
        ids, vfeed, _ = model.verify_after_external_sync(handle)
        self._vfeed = vfeed
        self._dr_rs_dirty = True  # the accept trace rewrote the reseed page table (and the pack)
        res = ttnn.to_torch(ttnn.get_device_tensors(fa.result_out)[0]).reshape(2, -1).to(torch.int64)
        steps = self._ids_from_handles(dr_handles)
        ids = torch.tensor(ids, dtype=torch.int32).reshape(B, T)
        committed, counts = res[0, : B * T].reshape(B, T), res[1, :B]
        if chain is not None:
            self._chain_audit(expect)
        # debug check (always on in S1): the device commit is the host greedy walk of the ids it read
        w_c, w_n = fused_greedy_walk(ids, tokens, nv, K)
        if not (torch.equal(w_c, committed) and torch.equal(w_n, counts)):
            raise RuntimeError(
                f"fused accept != host greedy walk: device committed {committed.tolist()} counts {counts.tolist()}; "
                f"host committed {w_c.tolist()} counts {w_n.tolist()}; ids {ids.tolist()} tokens {tokens.tolist()}"
            )
        self._chain_note_step("g", committed, counts, steps, start_pos, active)
        return self._fused_finish(ids, committed, counts, steps, start_pos, active, declined)

    def _fused_finish(self, ids, committed, counts, steps, start_pos, active, declined):
        """Tail shared by the greedy and the sampled fused step: the propose bookkeeping from the device commit."""
        K = self.K
        committed, counts = committed.to(torch.int32), counts.to(torch.int32)
        cpos = self._committed_positions(start_pos, active, committed, counts)
        sl = self._to_slots(committed, cpos, counts)  # the plan maps runner slots back to rows
        plan = self._propose_plan(K, *sl, None, build_core=False)
        if plan.skip_all or list(plan.decline) != declined.tolist():
            raise RuntimeError(f"fused aux declined {declined.tolist()} != propose plan {plan.decline}")
        drafts = self._drafts_from_steps(steps)
        if bool(declined.any()):
            drafts[declined] = 0
        if any(int(c) > 1 for c in counts):
            self.mtp_extra_steps += 1
        out = self._propose_finish(plan, drafts)
        self._fused_note(None)
        return ids, None, FusedStep(committed, counts, out)

    # --------------------------------------------------------------------- #
    # Chain mode (QWEN36_SPEC_CHAIN, stage S3c): the device builds the next verify's inputs
    # --------------------------------------------------------------------- #
    def _chain_drop(self, reason):
        """Forget the chain state of the last fused step (something changed that the device words do not know about). The first
        reason stands until the next fused step reads it (_chain_plan). No-op with chain mode off."""
        if self._ch_g is None:
            return
        self._chain_prev = None
        if self._chain_dropped is None:
            self._chain_dropped = reason

    def _chain_note_step(self, kind, committed, counts, steps, start_pos, active):
        """After a fused step that ran the device accept + drafter: remember its result (engine rows) so the next step can chain.
        It becomes ``_chain_prev`` in decode_forward when the step produced drafts, and usable once propose took the cache.
        """
        if self._ch_g is None:
            return
        self._chain_cand = dict(
            kind=kind,
            committed=torch.as_tensor(committed).clone().to(torch.int64),
            counts=torch.as_tensor(counts).clone().to(torch.int64),
            steps=[[int(x) for x in st] for st in steps],
            p0=torch.as_tensor(start_pos).clone().to(torch.int64),
            active=torch.as_tensor(active).clone().bool(),
            taken=False,
        )
        if self._bucketed():
            self._chain_cand["bucket"] = self._vb

    def _chain_plan(self, tokens, start_pos, active, nv, args, kw=None):
        """The previous-step record when THIS fused step (already eligible for the device accept) can take its verify inputs from the
        device, else None (reason counted in ``chain_breaks``; the step then stages tokens / aux on the host exactly as without
        chain mode). Needs: the previous step was a fused step that ran the device accept + drafter, the plugin took its drafts,
        no park / unpark / join / fresh row / slot change / prefill since, the live rows are a subset of the previous live rows
        (a finished row becomes an overlay row), no 'mixed' fit row, and the plugin's counts, positions and tokens for the
        continuing rows equal what the device derives (a plugin that capped a count or changed a draft breaks the chain).
        """
        if self._ch_g is None:
            return None
        prev, self._chain_in = self._chain_in, None
        why, self._chain_why = self._chain_why, None
        ev, self._chain_ev = self._chain_ev, None
        reason = self._chain_reason(prev, why, ev, tokens, start_pos, active, nv, args)
        if reason is not None and reason != "bucket_change":  # a break anyway: a pending bucket down-shift is free now
            self._bucket_free_note(None if prev is None else prev.get("bucket"), kw)
        if reason is not None:
            cb = self.__dict__.setdefault("chain_breaks", {})
            cb[reason] = cb.get(reason, 0) + 1
            return None
        return prev

    def _chain_reason(self, prev, why, ev, tokens, start_pos, active, nv, args, skip_bucket=False):
        B, K, T = self.B, self.K, self.T
        if prev is None:
            return why or "first"
        if not prev["taken"]:
            return "no_propose"
        if ev:
            return ev
        pa = prev["active"]
        if bool((active & ~pa).any()):
            return "row_set"
        if not skip_bucket and self._bucketed() and prev.get("bucket") != self._vb:
            # the step's bucket (held, see _bucket_step) differs: a break of its own only when nothing else breaks the chain
            return (
                self._chain_reason(prev, why, ev, tokens, start_pos, active, nv, args, skip_bucket=True)
                or "bucket_change"
            )
        if (self._ch_trace_g if prev["kind"] == "g" else self._ch_trace_s) is None:
            return "not_captured"
        cont = active & pa
        cap = table_capacity(self._last_cap_blocks, self.block_size)
        modes = fused_fit_mode_range(prev["p0"], cap, pa, K)
        if any(modes[u] == "mixed" for u in range(B) if bool(cont[u])):
            return "mixed_fit"
        mi = torch.as_tensor(args[2]).to(torch.int64).reshape(B)
        cn, p0 = prev["counts"], prev["p0"]
        if not torch.equal(mi[cont], cn[cont] - 1):
            return "counts_differ"
        if not torch.equal(torch.as_tensor(start_pos).to(torch.int64)[cont], (p0 + cn)[cont]):
            return "position_differs"
        tok0 = prev["committed"].gather(1, (cn - 1)[:, None])  # [B, 1] the bonus / correction token
        drafts = torch.tensor(prev["steps"], dtype=torch.int64).T.reshape(B, K)
        nvv = torch.as_tensor(nv).to(torch.int64).reshape(B)
        cols = torch.where(torch.arange(K)[None, :] < nvv[:, None], drafts, tok0)
        pred = torch.cat([tok0, cols], dim=1)
        if not torch.equal(torch.as_tensor(tokens).to(torch.int64)[cont], pred[cont]):
            return "tokens_differ"
        return None

    def _chain_stage(self, prev, tokens, start_pos, active, aux, args):
        """Stage the overlay (ONE H2D copy) of a chain step: next-step nv / active / declined / keep, host words (tokens, p0, mi)
        for every row that does not continue (mask = 1: finished or padding), and the GDN window parity the next verify
        reads. Returns what today's path would stage (tokens, aux words, fused aux) when this step is audited, else None.
        """
        B, K, model = self.B, self.K, self.model
        cont = active & prev["active"]
        rope_delta, common_delta, _ = self._step_rope_delta(active)
        ovr = pack_ovr_host(
            aux[0, :B],
            aux[1, :B],
            aux[2, :B],
            aux[3, :B],
            start_pos,
            (~cont).to(torch.int64),
            torch.as_tensor(tokens).to(torch.int64),
            args[2],
            model.vfy_gdn_parity(),
            K=K,
            scratch_block=int(self._scratch_block),
            rope_delta=rope_delta,
            rope_rows=model.rope.max_seq_len,
            common_delta=common_delta,
        )
        (self._ch_g if prev["kind"] == "g" else self._ch_s).stage_ovr(ovr)
        n = self.chain_steps
        self.chain_steps = n + 1
        if not (spec_chain_audit_enabled() and n % spec_chain_audit_every() == 0):
            return None
        exp_tok = torch.as_tensor(args[0]).to(torch.int64).reshape(-1)
        # at the parity the replay reads, with the per-row deltas the trace reads (the common delta with the flag off; inactive rows too)
        exp_aux = model._vfy_aux_host(model._vfy_row_positions(args[1]), args[2], rope_delta=rope_delta)
        return exp_tok, exp_aux, torch.as_tensor(aux).clone()

    def _chain_exec(self, prev):
        """Enqueue the chain trace of the variant that ran the previous step; it rewrites the verify token buffer, the packed aux
        buffer and the fused aux, so the model's host mirror of the aux words is stale from here."""
        ttnn.execute_trace(
            self.mesh, self._ch_trace_g if prev["kind"] == "g" else self._ch_trace_s, cq_id=0, blocking=False
        )
        self.model._vfy_aux_staged = None

    def _chain_audit(self, expect):
        """After the step's synchronize: the device-built token buffer, packed aux buffer and fused aux must be bit-equal to the words
        the host staging would have copied (built before the replay). Raises RuntimeError naming the first differing row.
        """
        if expect is None:
            return
        self.chain_audits += 1
        model, fa = self.model, self._fa
        read = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0])
        self._chain_compare(expect, read(model._vfy_token_buf), read(model._vfy_aux_buf), read(fa.aux_in))

    def _chain_compare(self, expect, tok_buf, aux_buf, fa_aux):
        """Pure half of the chain audit: host copies of the device token buffer, packed aux buffer and fused aux against the
        expected words (exp_tok, exp_aux, exp_fa). Raises RuntimeError naming the first differing word."""
        B, T = self.B, self.T
        exp_tok, exp_aux, exp_fa = expect
        mask32 = 0xFFFFFFFF
        R = B * T
        got = tok_buf.reshape(-1)[:R].to(torch.int64) & mask32
        want = exp_tok & mask32
        bad = (got != want).nonzero().reshape(-1)
        if bad.numel():
            i = int(bad[0])
            raise RuntimeError(
                f"chain audit: verify token buffer differs at engine row {i // T} column {i % T}: device {int(got[i])} "
                f"host {int(want[i])}; device {got.tolist()} host {want.tolist()}"
            )
        want = exp_aux.to(torch.int64) & mask32
        got = (aux_buf.to(torch.int64) & mask32).reshape(want.shape[0], -1)
        if got.shape != want.shape:
            raise RuntimeError(f"chain audit: aux buffer shape {tuple(got.shape)} != host {tuple(want.shape)}")
        bad = (got != want).nonzero()
        if bad.numel():
            r, c = (int(x) for x in bad[0])
            name = ("kv position", "rope row", "gdn ctrl word")[r]
            raise RuntimeError(
                f"chain audit: verify aux row {r} ({name}) differs at word {c} (engine row {c // T if r < 2 else c}): "
                f"device {int(got[r, c])} host {int(want[r, c])}"
            )
        want = torch.as_tensor(exp_fa).to(torch.int64) & mask32
        got = (fa_aux.to(torch.int64) & mask32).reshape(want.shape)
        bad = (got != want).nonzero()
        if bad.numel():
            r, c = (int(x) for x in bad[0])
            raise RuntimeError(
                f"chain audit: fused aux row {r} differs at row/user {c}: device {int(got[r, c])} host {int(want[r, c])}"
            )

    # --------------------------------------------------------------------- #
    # Deferred serving (QWEN36_SPEC_CHAIN + QWEN36_SPEC_FUSED, stage S4b): enqueue-only fused steps, event reads
    # --------------------------------------------------------------------- #
    # decode_forward(defer_read=True) on a fused_sample step ENQUEUES the whole step (the S3c chain step for rows that chain, the
    # S1 / S2 step for the first step of a chain), enqueues the non-blocking reads, records a device event and returns a
    # FusedHandle. There is no synchronize_device. The reader thread calls ``handle.wait()`` (event_synchronize, then host
    # conversions and the pure debug checks); the engine thread later calls propose_draft_tokens(hidden=handle) for the engine
    # bookkeeping that needed the commit (_deferred_finish). A step that cannot chain (see _def_pre_reason / _def_aux_for_chain)
    # first MATERIALIZES the host inputs from the pending step (one synchronize on the engine thread) and then runs as an ordinary
    # step. Thread rule: the reader thread touches only the handle (its lock, its read tensors, its result cache), ttnn.event_synchronize
    # and ttnn.to_torch on host tensors, and pure functions; every other engine field is engine-thread state.
    def _def_next_tip(self):
        self._tip_seq += 1
        return self._tip_seq

    @staticmethod
    def _perm_slots(x, perm):
        return x if perm is None else x[perm[0]]

    @staticmethod
    def _perm_rows(x, perm):
        return x if perm is None else x[perm[1]]

    def _def_chain_args(self, chain_rows, chain_tip, prev):
        """Validate ``chain_rows`` / ``chain_tip``; the bool [B] runner-order mask (None = nothing chains). The model refuses a tip
        that is not its newest fused step."""
        B = self.B
        if chain_rows is None:
            if chain_tip is not None:
                raise ValueError("chain_tip without chain_rows")
            return None
        cr = torch.as_tensor(chain_rows)
        if cr.dtype != torch.bool or tuple(cr.shape) != (B,):
            raise ValueError(f"chain_rows must be a bool tensor of shape [{B}], got {cr.dtype} {tuple(cr.shape)}")
        if not bool(cr.any()):
            return None
        if chain_tip is None or prev is None or int(chain_tip) != prev.tip_id:
            raise ValueError(
                f"chain_tip {chain_tip} is not the newest fused step of the engine "
                f"({None if prev is None else prev.tip_id}); a chained step must follow its tip"
            )
        return cr

    def _def_pre_reason(self, prev):
        """Reason a chain step cannot take its inputs from ``prev`` that does not depend on this step's rows (None = it can)."""
        if prev is None or not prev.device or not prev.chainable or prev.kind is None:
            return "tip_not_chainable"
        if self._ch_g is None or (self._ch_trace_g if prev.kind == "g" else self._ch_trace_s) is None:
            return "not_captured"
        return None

    def _def_events(self, active):
        """verify()'s pre-work events that need the device idle or the host state (park / unpark / fresh / unknown row)."""
        act = [bool(a) for a in active.tolist()]
        rows = [u for u in range(self.B) if act[u]]
        if any(u not in self._active for u in rows):
            return "unknown_row"
        for u in range(self.B):
            if not act[u] and u in self._active and u not in self._fresh and u not in self._parked:
                return "park"
        if any(u in self._parked for u in rows):
            return "unpark"
        if any(u in self._fresh for u in rows):
            return "fresh"
        return None

    def _def_next_bounds(self, prev):
        """(lo, hi) int64 [B]: bounds of the NEXT step's column-0 position when it chains on ``prev`` (exact once prev was finished)."""
        if prev.exact is not None:
            nxt = prev.exact[0] + prev.exact[1]
            return nxt, nxt.clone()
        return prev.lo + 1, prev.hi + self.K + 1

    def _def_window_blocks_disjoint(self, pt, active, lo, hi, nv):
        """The KV-write invariant that replaces the per-step collision check of chain rows: the blocks a row may write
        (positions lo .. hi + nv, so every real column whatever m was) are non-null and shared with no other active row.
        """
        bs, nb = int(self.block_size), int(pt.shape[1])
        seen = {}
        act, lo_l, hi_l, nv_l = active.tolist(), lo.tolist(), hi.tolist(), nv.tolist()
        rows = pt.tolist()
        for u in range(self.B):
            if not act[u]:
                continue
            for b in range(int(lo_l[u]) // bs, min(nb - 1, (int(hi_l[u]) + int(nv_l[u])) // bs) + 1):
                blk = rows[u][b]
                if blk == 0:
                    continue
                if seen.setdefault(blk, u) != u:
                    return False
        return True

    def _def_aux_for_chain(self, prev, cmask, start0, active, nv, cap):
        """(reason, aux, lo, hi): the fused aux of a chain step (host rows exact, chain rows from the bounds of their start) or the
        reason it cannot run (the caller materializes). Same word layout as fused_accept_aux_host."""
        B, K = self.B, self.K
        lo_c, hi_c = self._def_next_bounds(prev)
        lo = torch.where(cmask, lo_c, start0)
        hi = torch.where(cmask, hi_c, start0)
        if self._fa is None or self._fa_trace is None or not self.captured or self._dr_trace is None:
            return "not_captured", None, lo, hi
        if B == 1 and int(hi[0]) + K > self.eager_reseed_len:
            return "eager_drafter", None, lo, hi
        skip, skip_all = propose_skip_plan(self._no_draft, B)
        if skip_all and B == 1:
            return "skip_all", None, lo, hi
        host_act = active & ~cmask
        delta, common, mixed = self._step_rope_delta(active)
        if mixed and not self._row_delta_flag():
            return "mixed_rope_delta", None, lo, hi
        try:
            aux = fused_accept_aux_host(
                nv,
                start0,
                host_act,
                skip,
                cap,
                K,
                int(self._scratch_block),
                delta,
                self.model.rope.max_seq_len,
                common_delta=common,
            )
        except ValueError:
            return "mixed_fit", None, lo, hi
        modes = fused_fit_mode_bounds(lo, hi, cap, cmask, K)
        aux = aux.clone()
        for u in range(B):
            if not bool(cmask[u]):
                continue
            if modes[u] == "mixed" and not skip[u]:
                return "mixed_fit", None, lo, hi
            aux[AUX_ACTIVE, u] = 1
            aux[AUX_DECLINED, u] = int(skip[u] or modes[u] == "fits_none")
            aux[AUX_KEEP, u] = int(skip[u])
            aux[AUX_P0, u] = int(hi[u])
        if bool(aux[2, :B].all()):
            return "all_declined", None, lo, hi
        if mixed:
            self.row_delta_mixed_steps = self.__dict__.get("row_delta_mixed_steps", 0) + 1
        return None, aux, lo, hi

    def _def_count_break(self, reason):
        cb = self.__dict__.setdefault("chain_breaks", {})
        cb[reason] = cb.get(reason, 0) + 1
        db = self.__dict__.setdefault("def_breaks", {})
        db[reason] = db.get(reason, 0) + 1

    def _decode_forward_deferred(
        self,
        tokens,
        start_pos,
        num_valid_drafts,
        accepted_counts,
        page_table,
        spec_mode,
        slot_remap,
        spec_sampling,
        chain_rows,
        chain_tip,
        finishing_slots=None,
    ):
        """decode_forward(defer_read=True): see the section comment. Returns VerifyResult(spec_mode, None, handle=FusedHandle)."""
        B, K, T = self.B, self.K, self.T
        if spec_mode != "fused_sample":
            raise ValueError(f"defer_read needs spec_mode 'fused_sample', got {spec_mode!r}")
        if not self.host_slotmap:
            raise ValueError(
                "deferred serving (defer_read) needs QWEN36_SPEC_SLOTMAP=1 (a device slot remap would move the rows a pending step describes)"
            )
        self._fused_cache = None
        if self._ch_g is not None:  # the S3c sync chain state is not used while serving deferred
            self._chain_in = (
                self._chain_prev
            ) = self._chain_cand = self._chain_ev = self._chain_why = self._chain_dropped = None
        prev = self._def_last
        cr = self._def_chain_args(chain_rows, chain_tip, prev)
        tok_in = _int_block(tokens, B, T, "tokens")
        pos = _int_block(start_pos, B, T, "start_pos")
        nv_in = _int_vec(num_valid_drafts, B, "num_valid_drafts")
        acc = _int_vec(accepted_counts, B, "accepted_counts")
        tok = tok_in
        if cr is not None:
            cr = cr & (pos[:, 0] >= 0)
            if not bool(cr.any()):
                cr = None
        if cr is not None:
            if not bool((tok_in[cr] == PLACEHOLDER_TOKEN_ID).all()) or not bool((acc[cr] == -1).all()):
                raise ValueError("a chain row carries placeholder tokens (-1) and accepted_counts -1")
            tok, pos, acc = tok_in.clone(), pos.clone(), acc.clone()
            for u in torch.nonzero(cr).reshape(-1).tolist():
                n = int(nv_in[u])
                if not 0 <= n <= K:
                    raise ValueError(f"chain row {u}: num_valid_drafts {n} outside [0, {K}]")
                tok[u] = 0
                pos[u, : n + 1] = int(pos[u, 0]) + torch.arange(n + 1)
            acc[cr] = 1
        pre = self._def_pre_reason(prev) if cr is not None else None
        if pre is not None:
            self._def_finish_any(
                prev
            )  # before the slot map moves: the ready handle's propose maps with the map of its own step
        clean, start0, active = sanitize_verify_block(tok, pos, nv_in, K)
        if clean.shape[0] != B:
            raise ValueError(f"block has {clean.shape[0]} rows, engine has B={B}")
        if not bool(active.any()):
            raise ValueError("no active row in the verify block")
        self._apply_finishing(finishing_slots)  # pre-remap slots, before the slot map moves
        if slot_remap is not None:
            self.remap_slots(slot_remap)
        clean, start0, active, nv_rows, acc_rows, pt_rows = self._to_rows(clean, start0, active, nv_in, acc, page_table)
        self._check_finishing_rows([u for u in range(B) if bool(active[u])])
        cmask = None if cr is None else (self._to_rows(cr)[0] & active)
        if cmask is not None and not bool(cmask.any()):
            cmask = None
        sampling = None
        if spec_sampling is not None:
            names = ("temperature", "top_k", "top_p", "greedy", "uniforms")
            sampling = dict(zip(names, self._to_rows(*(spec_sampling[n] for n in names))))
            if bool(torch.as_tensor(sampling["greedy"]).all()):
                sampling = None
        kind_s = sampling is not None
        # ---- which rows chain
        if self._bucketed():
            self._bucket_step(self._active_rows(active))
        reason = pre
        if cmask is not None and reason is None:
            if bool((active & ~cmask).any()):
                reason = "host_row"  # an active row that does not continue (a join, a seed): the host owns its inputs
            elif bool((cmask & ~prev.active).any()):
                reason = "row_set"
            else:
                reason = self._def_events(active)
            if reason is None and kind_s:
                reason = self._sampled_ineligible(sampling)
            if reason is None and self._bucketed() and prev.bucket != self._vb:
                reason = "bucket_change"  # only when no other reason breaks the chain (that one is counted instead)
        if self._bucketed() and (cmask is None or (reason is not None and reason != "bucket_change")):
            self._bucket_free_note(
                None if prev is None else prev.bucket
            )  # a break anyway: a pending down-shift is free now
        hint = nv_rows.clone()
        if cmask is not None and reason is None:
            nv_rows = nv_rows.clone()
            nv_rows[cmask] = torch.minimum(hint, prev.nv_own)[cmask]
        elif (
            cmask is not None
        ):  # materialize now: the tokens, positions and counts of the chain rows from the pending step
            self._def_count_break(reason)
            clean, start0, acc_rows, nv_rows = self._def_materialize(
                prev, cmask, clean, start0, acc_rows, nv_rows, reason
            )
            cmask = None
        self._no_draft = note_withheld_drafts(
            self._offered, self._no_draft, self._speculated, self._withheld_streak, nv_rows, self.no_draft_streak
        )
        chain = None
        if cmask is not None:
            cap = table_capacity(int(torch.as_tensor(pt_rows).shape[-1]), self.block_size)
            why, aux, lo, hi = self._def_aux_for_chain(prev, cmask, start0, active, nv_rows, cap)
            if why is None:
                pt = self._fit(pt_rows)
                if not self._def_window_blocks_disjoint(
                    torch.where(active[:, None], pt, torch.zeros_like(pt)), active, lo, hi, nv_rows
                ):
                    why = "shared_block"
            if why is not None:
                self._def_count_break(why)
                clean, start0, acc_rows, nv_rows = self._def_materialize(
                    prev, cmask, clean, start0, acc_rows, nv_rows, why
                )
                cmask = None
            else:
                chain = dict(prev=prev, cmask=cmask, lo=lo, hi=hi, aux=aux)
        if chain is not None:
            h = self._def_chain_step(chain, clean, start0, active, acc_rows, nv_rows, pt_rows, sampling)
            return VerifyResult("fused_sample", None, handle=h)
        self._def_ctx = {"sampling": sampling}
        try:
            ids, _, fz = self.verify(
                clean, start0, acc_rows, pt_rows, active=active, num_valid_drafts=nv_rows, fused=True, sampling=sampling
            )
        finally:
            self._def_ctx = None
        if ids is None and isinstance(fz, FusedHandle):
            return VerifyResult("fused_sample", None, handle=fz)
        h = self._def_ready_handle(
            fz, start0, active
        )  # a fallback step ran synchronously: its result is already on the host
        return VerifyResult("fused_sample", None, handle=h)

    def _def_chain_step(self, chain, clean, start0, active, acc_rows, nv_rows, pt_rows, sampling):
        """The chain step: verify()'s tail for rows whose positions are bounds, then the deferred enqueue."""
        B, K, T = self.B, self.K, self.T
        cmask, lo, hi, aux = chain["cmask"], chain["lo"], chain["hi"], chain["aux"]
        self._last_cap_blocks = int(torch.as_tensor(pt_rows).shape[-1])
        pt = self._fit(pt_rows)
        mi_prev = acc_rows - 1
        mi_prev = torch.where(active & ~cmask, mi_prev, torch.zeros_like(mi_prev))
        assert bool(((mi_prev >= 0) & (mi_prev < T)).all()), f"accepted_counts must be in [1, {T}]"
        pt = pt.clone()
        pt[~active] = 0  # vLLM null block
        self._row_active = active.clone()
        self._last_pt = pt
        start_hi = torch.where(cmask, hi, start0)
        vpt = verify_write_table(pt, start_hi, nv_rows, self.block_size)
        assert int(start_hi.max()) + T <= table_capacity(
            self._last_cap_blocks, self.block_size
        ), "verify window past the paged KV capacity"
        args = (clean.tolist(), start_hi.tolist(), mi_prev.tolist())
        kw = dict(read_logits=False, page_tables=vpt, inactive=(~active).tolist())
        self._bucket_kw(kw)  # the bucket was decided in _decode_forward_deferred
        chain = dict(chain, lo=lo, hi=hi)
        return self._def_enqueue(
            "s" if sampling is not None else "g", clean, start_hi, pt, active, nv_rows, args, kw, sampling, aux, chain
        )

    def _def_enqueue(self, kind, tokens, start_pos, pt, active, nv, args, kw, sampling, aux, chain=None):
        """The enqueue half of a fused step (greedy 'g' or sampled 's'; ``chain`` None = host-staged first step of a chain): the call
        order of _verify_fused / _verify_fused_sampled up to the drafter trace, the GDN parity flipped at enqueue, then the
        non-blocking reads and one event. NO synchronize. Returns the FusedHandle."""
        B, T, K = self.B, self.T, self.K
        fa, fs, model = self._fa, self._fs, self.model
        self._check_scratch(pt)
        declined = aux[2, :B].bool()
        parity = model.vfy_gdn_parity()
        do_audit = False
        if chain is None:
            fa.stage_aux(aux)
            prev, cmask = None, None
        else:
            prev, cmask = chain["prev"], chain["cmask"]
            ovr_delta, ovr_common, _ = self._step_rope_delta(active)
            ovr = pack_ovr_host(
                aux[0, :B],
                aux[1, :B],
                aux[2, :B],
                aux[3, :B],
                start_pos,
                (~cmask).to(torch.int64),
                torch.as_tensor(tokens).to(torch.int64),
                args[2],
                parity,
                K=K,
                scratch_block=int(self._scratch_block),
                rope_delta=ovr_delta,
                rope_rows=model.rope.max_seq_len,
                common_delta=ovr_common,
            )
            (self._ch_g if prev.kind == "g" else self._ch_s).stage_ovr(ovr)
            n = self.chain_steps
            self.chain_steps = n + 1
            do_audit = spec_chain_audit_enabled() and n % spec_chain_audit_every() == 0
        if kind == "s":
            fs.stage_samp(
                sampling["temperature"], sampling["top_k"], sampling["top_p"], sampling["greedy"], sampling["uniforms"]
            )
        if self._fa_pt_host is None or not torch.equal(self._fa_pt_host, pt):
            fa.stage_pt_rows(pt.repeat_interleave(T, dim=0))
            self._fa_pt_host = pt.clone()
        dpt = pt
        if bool(declined.any()):
            dpt = pt.clone()
            dpt[declined] = self._scratch_block
        self._stage_table(dpt, self._mtp_pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        extra = {"execute_topk": True} if kind == "s" else {}
        read = lambda t: ttnn.from_device(ttnn.get_device_tensors(t)[0], blocking=False)
        audit = None
        if chain is None:
            model.verify_stage_and_enqueue(*args, **kw, **extra, flip_parity=True)
        else:
            ttnn.execute_trace(
                self.mesh, self._ch_trace_g if prev.kind == "g" else self._ch_trace_s, cq_id=0, blocking=False
            )
            model._vfy_aux_staged = None
            if (
                do_audit
            ):  # the device-built words of THIS step, copied right behind the chain trace; checked at this step's read
                audit = dict(tok=read(model._vfy_token_buf), aux=read(model._vfy_aux_buf), fa=read(fa.aux_in))
            model.verify_stage_and_enqueue(
                *args, **kw, **extra, skip_tokens_h2d=True, skip_aux_h2d=True, skip_kv_check=True, flip_parity=True
            )
        ttnn.execute_trace(self.mesh, self._fa_trace if kind == "g" else self._fs_trace, cq_id=0, blocking=False)
        ttnn.execute_trace(self.mesh, self._dr_trace_for(self._vb), cq_id=0, blocking=False)
        self._vfeed = model._vfy_rows_out
        self._dr_rs_dirty = True  # the accept trace rewrote the reseed page table (and the pack)
        res_t = read(fa.result_out if kind == "g" else fs.result_out)
        aux_t = read(fa.aux_in)  # still this step's aux: the next chain trace is enqueued after these reads
        dr_ts = [read(t) for t in self._dr_outs]
        ids_t = union = None
        if kind == "g":
            n_g = self.fused_greedy_checked
            self.fused_greedy_checked = n_g + 1
            if spec_fused_audit_enabled() and n_g % spec_fused_audit_every() == 0:
                ids_t = read(model._vfy_ids_out)
        else:
            n_s = self.fused_sampled_steps
            self.fused_sampled_steps = n_s + 1
            if spec_fused_audit_enabled() and n_s % spec_fused_audit_every() == 0:
                ids_t = read(model._vfy_ids_out)
                union = (read(fs.union_vals_in), read(fs.union_ids_in))
        event = ttnn.record_event(self.mesh, 0)
        h = FusedHandle(self, self._def_next_tip(), kind)
        h.event, h.res_t, h.aux_t, h.dr_ts, h.ids_t, h.union, h.audit = event, res_t, aux_t, dr_ts, ids_t, union, audit
        h.res_rows = 2 if kind == "g" else 3
        h.active, h.declined, h.cmask, h.prev = active.clone(), declined.clone(), cmask, prev
        h.nvv = torch.as_tensor(nv).to(torch.int64).reshape(B).clone()
        h.nv_own = torch.where(declined, torch.zeros_like(h.nvv), torch.full_like(h.nvv, K))
        h.tokens = torch.as_tensor(tokens).to(torch.int64).clone()
        h.start_host = torch.as_tensor(start_pos).to(torch.int64).clone()
        h.mi_host = torch.as_tensor(args[2]).to(torch.int64).clone()
        h.parity, h.aux, h.pt, h.sampling = parity, aux.clone(), pt.clone(), sampling
        h.gen = list(self._row_gens())
        h.lo = (chain["lo"] if chain is not None else h.start_host).clone()
        h.hi = (chain["hi"] if chain is not None else h.start_host).clone()
        h.skip = [bool(s) for s in propose_skip_plan(self._no_draft, B)[0]]
        h.cap = table_capacity(self._last_cap_blocks, self.block_size)
        h.perm = self._phys_perm
        if self._bucketed():
            h.bucket = self._vb
        if prev is not None:
            prev.succ = weakref.ref(h)
        self._offered = {u for u in range(B) if not bool(declined[u])}
        self._fused_note(None)
        self._def_last = h
        return h

    def _def_ready_handle(self, fz, start0, active):
        """A handle for a step that ran synchronously (a fallback): its result is on the host already, no drafts were made (the
        propose of the plugin, or a chain onto it, runs the ordinary propose), and nothing can chain on it."""
        B, T = self.B, self.T
        h = FusedHandle(self, self._def_next_tip(), None)
        h.device, h.ready, h.chainable = False, True, False
        h.gen = list(self._row_gens())
        committed = torch.as_tensor(fz.committed).to(torch.int64).reshape(B, T)
        counts = torch.as_tensor(fz.counts).to(torch.int64).reshape(B)
        p0 = torch.as_tensor(start0).to(torch.int64).reshape(B).clone()
        h.active, h.perm = torch.as_tensor(active).bool().clone(), self._phys_perm
        h._data = dict(committed=committed, counts=counts, p0=p0, steps=None)
        h.exact = (p0, counts.clone())
        h._readout = self._def_readout(h, h._data)
        self._def_last = h
        return h

    def _def_readout(self, h, d):
        """(accepted_token_ids [B, 1+K] int32, accepted_counts [B] int32, p0 [B] int32; negative on padding rows), runner slots."""
        p0 = torch.where(h.active, d["p0"], torch.full_like(d["p0"], -1))
        return tuple(
            self._perm_slots(x, h.perm)
            for x in (d["committed"].to(torch.int32), d["counts"].to(torch.int32), p0.to(torch.int32))
        )

    # ---- reads (reader thread) and checks (pure)
    def _def_wait(self, h):
        self._def_data(h)
        return h._readout

    def _def_data(self, h, engine_thread=False):
        """The step's host data (engine rows), after its device work finished: event_synchronize on the reader thread, or one
        synchronize_device on the engine thread (materialize / lazy finish before the read). Cached per handle; runs the checks once.
        """
        with h._lock:
            if h._data is not None:
                return h._data
            if engine_thread:
                ttnn.synchronize_device(self.mesh)
            else:
                ttnn.event_synchronize(h.event)
            d = self._def_parse(h)
            self._def_check(h, d)
            h._data = d
            h._readout = self._def_readout(h, d)
            h.prev = None  # the prediction needed it only for the checks
            return d

    def _def_parse(self, h):
        B, T = self.B, self.T
        res = ttnn.to_torch(h.res_t).reshape(h.res_rows, -1).to(torch.int64)
        d = dict(
            committed=res[0, : B * T].reshape(B, T).clone(),
            counts=res[1, :B].clone(),
            overflow=res[2, :B].clone() if h.res_rows == 3 else None,
            p0=(ttnn.to_torch(h.aux_t).reshape(FUSED_AUX_ROWS, -1)[AUX_P0, :B].to(torch.int64) & 0xFFFFFFFF).clone(),
            steps=[[int(v) for v in ttnn.to_torch(t).reshape(-1)[:B]] for t in h.dr_ts],
            ids=None,
            n_chain_audit=0,
            n_s_audit=0,
            boundary=0,
        )
        if h.ids_t is not None:
            d["ids"] = torch.as_tensor(
                [int(v) for v in ttnn.to_torch(h.ids_t).reshape(-1)[: B * T]], dtype=torch.int32
            ).reshape(B, T)
        return d

    def _def_tokens(self, h, pd):
        """The verify token block the device ran this step with: host rows as staged, chain rows from the previous step's commit."""
        B, K = self.B, self.K
        tok = h.tokens.clone()
        if h.cmask is not None:
            committed, counts = pd["committed"], pd["counts"]
            tok0 = committed.gather(1, (counts - 1)[:, None])
            drafts = torch.tensor(pd["steps"], dtype=torch.int64).T.reshape(B, K)
            cols = torch.where(torch.arange(K)[None, :] < h.nvv[:, None], drafts, tok0)
            pred = torch.cat([tok0, cols], dim=1)
            tok[h.cmask] = pred[h.cmask]
        return tok * h.active[:, None].to(torch.int64)

    def _def_check(self, h, d):
        """Debug checks of a deferred step, at read time: pure (host tensors only), raising RuntimeError on a mismatch.
        S1 greedy walk, S2 sampled audit, device p0 vs bounds and vs the previous step, deferred chain audit."""
        B, T, K = self.B, self.T, self.K
        committed, counts, p0 = d["committed"], d["counts"], d["p0"]
        if d["overflow"] is not None and bool(d["overflow"].any()):
            raise RuntimeError(
                "fused sampled step: a top-k tie group reached the last union candidate in deferred serving; the step cannot be redone "
                f"on the host once a chained step followed it (overflow rows {d['overflow'].tolist()})"
            )
        bad = h.active & ((p0 < h.lo) | (p0 > h.hi))
        if bool(bad.any()):
            raise RuntimeError(
                f"deferred fused step {h.tip_id}: device p0 {p0.tolist()} outside the host bounds [{h.lo.tolist()}, {h.hi.tolist()}]"
            )
        pd = None
        if h.cmask is not None:
            pd = self._def_data(h.prev)
            want = pd["p0"] + pd["counts"]
            if bool((h.cmask & (p0 != want)).any()):
                raise RuntimeError(
                    f"deferred fused step {h.tip_id}: device p0 {p0.tolist()} != previous p0 + count {want.tolist()}"
                )
        tokens = self._def_tokens(h, pd)
        ids = d["ids"]
        if ids is not None and h.kind == "g":
            w_c, w_n = fused_greedy_walk(ids, tokens, h.nvv, K)
            if not (torch.equal(w_c, committed) and torch.equal(w_n, counts)):
                raise RuntimeError(
                    f"fused accept != host greedy walk: device committed {committed.tolist()} counts {counts.tolist()}; "
                    f"host committed {w_c.tolist()} counts {w_n.tolist()}; ids {ids.tolist()} tokens {tokens.tolist()}"
                )
        if ids is not None and h.kind == "s" and h.union is not None:
            fs = self._fs
            width = fs.TP * fs.M
            vals = ttnn.to_torch(h.union[0]).reshape(-1, width)[: B * T]
            idx = ttnn.to_torch(h.union[1]).reshape(-1, width)[: B * T]
            aux = h.aux.clone()
            aux[AUX_P0, :B] = p0.to(torch.int32)
            d["n_s_audit"] = 1
            d["boundary"] = self._audit_sampled_core(
                vals, idx, ids, committed, counts, tokens, h.nvv, h.pt, aux, h.sampling
            )
        if h.audit is not None:
            d["n_chain_audit"] = 1
            self._def_chain_audit(h, d, pd)

    def _def_chain_audit(self, h, d, pd):
        """The deferred CHAIN audit: the token buffer, packed aux buffer and fused aux the chain trace wrote for THIS step (copied
        right behind it) against the words the host predicts from the previous step's commit, tokens, counts, p0 and drafts.
        """
        B, T, model = self.B, self.T, self.model
        cm, act = h.cmask, h.active
        p0n = torch.where(cm, pd["p0"] + pd["counts"], h.start_host) * act
        min_ = torch.where(cm, pd["counts"] - 1, h.mi_host) * act
        exp_tok = self._def_tokens(h, pd).reshape(-1)
        delta = aux_row_deltas(h.aux, B)  # the per-row deltas this step's fused aux carries (row AUX_DELTA)
        exp_aux = model._vfy_aux_host(
            model._vfy_row_positions(p0n.tolist()), min_.tolist(), parity=h.parity, rope_delta=delta
        )
        exp_fa = h.aux.clone().to(torch.int64)
        exp_fa[AUX_P0, :B] = p0n
        read = lambda t: ttnn.to_torch(t)
        self._chain_compare((exp_tok, exp_aux, exp_fa), read(h.audit["tok"]), read(h.audit["aux"]), read(h.audit["fa"]))

    # ---- engine-thread bookkeeping
    def _deferred_finish(self, h):
        """propose_draft_tokens(hidden=handle) of a device step, on the engine thread: the bookkeeping the sync path did in
        _fused_finish (mtp_extra_steps, the decline cross-check, the DraftResult) from the handle's own data, never from state the
        steps enqueued since then changed (page table, row map, skip set). Idempotent."""
        if h.draft_result is not None:
            return h.draft_result
        B, K = self.B, self.K
        d = self._def_data(h, engine_thread=True)
        committed, counts, p0 = d["committed"], d["counts"], d["p0"]
        c32, n32 = committed.to(torch.int32), counts.to(torch.int32)
        anchor = (p0 + counts - 1).tolist()
        fit = draft_rows_fit(anchor, K, h.cap)
        want = [(not bool(a)) or bool(s) or (not f) for a, s, f in zip(h.active.tolist(), h.skip, fit)]
        if want != h.declined.tolist():
            raise RuntimeError(f"fused aux declined {h.declined.tolist()} != propose plan {want}")
        drafts = self._drafts_from_steps(d["steps"])
        drafts[h.declined] = 0
        if any(int(c) > 1 for c in n32):
            self.mtp_extra_steps += 1
        nv = torch.tensor([0 if x else K for x in h.declined.tolist()], dtype=torch.int32)
        h.fin = dict(committed=committed, counts=counts, p0=p0, drafts=drafts, nv=nv)
        h.draft_result = DraftResult(self._perm_slots(drafts, h.perm), self._perm_slots(nv, h.perm))
        h.exact = (p0.clone(), counts.clone())
        h.finished = True
        s = None if h.succ is None else h.succ()
        if s is not None and s.cmask is not None:  # the step chained on this one: its start is now exact
            nxt = p0 + counts
            s.lo, s.hi = torch.where(s.cmask, nxt, s.lo), torch.where(s.cmask, nxt, s.hi)
        self.chain_audits += d["n_chain_audit"]
        self.fused_audited_steps += d["n_s_audit"]
        self.fused_boundary_diffs += d["boundary"]
        return h.draft_result

    def _def_finish_any(self, h):
        """Finish ``h`` whatever it is (device step: _deferred_finish; ready fallback step: the ordinary propose from its host
        result) and return its engine-row data (committed, counts, p0, drafts, nv)."""
        if h.device:
            self._deferred_finish(h)
            return h.fin
        if h.fin is None:
            committed, counts, p0 = h._readout
            cpos = self._committed_positions(torch.clamp(p0, min=0), p0 >= 0, committed, counts)
            if h.draft_result is None:
                h.draft_result = self.propose_draft_tokens(self.K, committed, cpos, counts)
            dr = h.draft_result
            h.fin = dict(
                committed=h._data["committed"],
                counts=h._data["counts"],
                p0=h._data["p0"],
                drafts=self._perm_rows(dr.draft_token_ids.to(torch.int64), h.perm),
                nv=self._perm_rows(dr.num_valid.to(torch.int64), h.perm),
            )
            h.finished = True
        return h.fin

    def _def_materialize(self, prev, cmask, clean, start0, acc_rows, nv_rows, why):
        """A chain step that cannot chain: build its chain rows' host inputs from the pending step (engine thread: synchronize,
        read, finish it) and run as an ordinary step. Returns copies of (clean, start0, acc_rows, nv_rows) in engine rows.
        """
        if prev is None:
            raise RuntimeError(f"chain step cannot chain ({why}) and has no pending step to build its inputs from")
        K, T = self.K, self.T
        f = self._def_finish_any(prev)
        committed, counts, p0 = f["committed"], f["counts"], f["p0"]
        clean, start0, acc_rows, nv_rows = clean.clone(), start0.clone(), acc_rows.clone(), nv_rows.clone()
        gen = self._row_gens()
        for u in torch.nonzero(cmask).reshape(-1).tolist():
            if prev.gen is not None and prev.gen[u] != gen[u]:
                continue  # a prefill / release gave the row to another request since the pending step: its output was dropped
            c = int(counts[u])
            n = min(int(nv_rows[u]), int(f["nv"][u]))
            tok0 = int(committed[u, c - 1])
            clean[u] = torch.tensor(
                [tok0] + [int(x) for x in f["drafts"][u, :n]] + [tok0] * (T - 1 - n), dtype=torch.int64
            )
            start0[u] = int(p0[u]) + c
            acc_rows[u] = c
            nv_rows[u] = n
        return clean, start0, acc_rows, nv_rows

    def _sampled_ineligible(self, sampling):
        """Reason a sampled fused step cannot run the sampled accept trace (None = it can), beyond _fused_ineligible's."""
        fs = self._fs
        if fs is None or self._fs_trace is None:
            return "no_sampled_trace"
        k = torch.as_tensor(sampling["top_k"]).to(torch.int64)
        greedy = torch.as_tensor(sampling["greedy"]).bool()
        if bool((~greedy & ((k < 1) | (k > fs.M - 1))).any()):
            return "top_k_range"
        return None

    def _verify_fused_sampled(self, tokens, start_pos, pt, active, nv, args, kw, sampling, aux):
        """Stage S2 step. Call order: stage aux, samp (pack_samp_host), pt rows / MTP table when changed; enqueue verify, topk
        (execute only, no host read), sampled accept trace, drafter trace; enqueue the non-blocking drafter-id reads; ONE
        synchronize; read ids, result [3, W] (committed, counts, overflow) and the draft ids (+ the union in audit mode).
        An ``overflow`` flag on any user (a top-k tie group reached the last union candidate) redoes the whole step on the host:
        the full logits (logits trace on the intact verify rows) walked by host_redo_walk with the SAME uniforms; the result
        carries no drafts, so the propose that follows runs normally (and rewrites what the drafter trace wrote)."""
        B, T, K = self.B, self.T, self.K
        if self._def_ctx is not None:  # decode_forward(defer_read=True): enqueue, no synchronize
            return None, None, self._def_enqueue("s", tokens, start_pos, pt, active, nv, args, kw, sampling, aux)
        fa, fs, model = self._fa, self._fs, self.model
        self._check_scratch(pt)
        declined = aux[2, :B].bool()
        chain = self._chain_plan(tokens, start_pos, active, nv, args, kw)
        if chain is None:
            fa.stage_aux(aux)  # aux_in / pt_rows_in are shared with the S1 device
        else:
            expect = self._chain_stage(chain, tokens, start_pos, active, aux, args)
        fs.stage_samp(
            sampling["temperature"], sampling["top_k"], sampling["top_p"], sampling["greedy"], sampling["uniforms"]
        )
        if self._fa_pt_host is None or not torch.equal(self._fa_pt_host, pt):
            fa.stage_pt_rows(pt.repeat_interleave(T, dim=0))
            self._fa_pt_host = pt.clone()
        dpt = pt
        if bool(declined.any()):
            dpt = pt.clone()
            dpt[declined] = self._scratch_block
        self._stage_table(dpt, self._mtp_pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        if chain is None:
            handle = model.verify_stage_and_enqueue(*args, **kw, execute_topk=True)  # verify, then topk (no host read)
        else:
            self._chain_exec(chain)
            handle = model.verify_stage_and_enqueue(
                *args, **kw, execute_topk=True, skip_tokens_h2d=True, skip_aux_h2d=True
            )
        ttnn.execute_trace(self.mesh, self._fs_trace, cq_id=0, blocking=False)
        ttnn.execute_trace(self.mesh, self._dr_trace_for(self._vb), cq_id=0, blocking=False)
        dr_handles = self._ids_enqueue_reads(self._dr_outs)
        ttnn.synchronize_device(self.mesh)  # the only one of the step
        ids, vfeed, _ = model.verify_after_external_sync(handle)
        self._vfeed = vfeed
        self._dr_rs_dirty = True  # the accept trace rewrote the reseed page table (and the pack)
        res = ttnn.to_torch(ttnn.get_device_tensors(fs.result_out)[0]).reshape(3, -1).to(torch.int64)
        steps = self._ids_from_handles(dr_handles)
        ids = torch.tensor(ids, dtype=torch.int32).reshape(B, T)
        committed, counts, overflow = res[0, : B * T].reshape(B, T), res[1, :B], res[2, :B]
        if chain is not None:
            self._chain_audit(expect)
        if bool(overflow.any()):
            self._fused_note("overflow_redo")
            logits = model.fetch_verify_logits()  # logits trace over the intact _vfy_normed_out: one more round trip
            return ids, None, self._host_walk_step(logits, tokens, nv, sampling)
        n_sampled = self.fused_sampled_steps
        self.fused_sampled_steps = n_sampled + 1
        if spec_fused_audit_enabled() and n_sampled % spec_fused_audit_every() == 0:
            self.fused_audited_steps += 1
            self._audit_sampled(ids, committed, counts, tokens, nv, pt, aux, sampling)
        self._chain_note_step("s", committed, counts, steps, start_pos, active)
        return self._fused_finish(ids, committed, counts, steps, start_pos, active, declined)

    def _audit_sampled(self, ids, committed, counts, tokens, nv, pt, aux, sampling):
        """QWEN36_SPEC_FUSED_AUDIT (default on): read the union back (two extra reads) and recompute the step with
        fused_sampled_reference on the same parameters and uniforms. A differing user whose difference is a float32 boundary
        (|p_d - u| < 1e-5, or the inverse-CDF threshold within 1e-5 of a bucket edge) is counted in ``fused_boundary_diffs``;
        any other difference raises."""
        B, T, K = self.B, self.T, self.K
        fs = self._fs
        width = fs.TP * fs.M
        vals = ttnn.to_torch(ttnn.get_device_tensors(fs.union_vals_in)[0]).reshape(-1, width)[: B * T]
        idx = ttnn.to_torch(ttnn.get_device_tensors(fs.union_ids_in)[0]).reshape(-1, width)[: B * T]
        n = self._audit_sampled_core(vals, idx, ids, committed, counts, tokens, nv, pt, aux, sampling)
        if n:
            self.fused_boundary_diffs += n
            logger.info(
                f"[spec_engine] fused sampled audit: float32 boundary difference users (total {self.fused_boundary_diffs})"
            )

    def _audit_sampled_core(self, vals, idx, ids, committed, counts, tokens, nv, pt, aux, sampling):
        """The check of _audit_sampled on host tensors (also the deferred read-time audit). Returns the number of users whose
        difference is a float32 boundary (explained); raises RuntimeError on any other difference."""
        B, T, K = self.B, self.T, self.K
        fs = self._fs
        tok_flat = torch.as_tensor(tokens).reshape(-1)
        ref = fused_sampled_reference(
            vals,
            idx,
            tok_flat,
            ids.reshape(-1),
            aux,
            pt,
            B=B,
            K=K,
            per_shard=fs.per_shard,
            M=fs.M,
            greedy=sampling["greedy"],
            temperature=sampling["temperature"],
            top_k=sampling["top_k"],
            top_p=sampling["top_p"],
            uniforms=sampling["uniforms"],
            num_valid=nv,
        )
        same = torch.equal(ref.committed.to(torch.int64), committed) and torch.equal(ref.counts.to(torch.int64), counts)
        if same:
            return 0
        drafts = torch.as_tensor(tokens).reshape(B, T)[:, 1:]
        explained = audit_boundary_users(ref, sampling["uniforms"], committed, counts, drafts, nv, B=B, K=K, M=fs.M)
        bad = [
            u
            for u in range(B)
            if not (
                torch.equal(ref.committed[u].to(torch.int64), committed[u]) and int(ref.counts[u]) == int(counts[u])
            )
            and u not in explained
        ]
        if bad:
            raise RuntimeError(
                f"fused sampled accept != host reference for users {bad}: device committed {committed.tolist()} counts "
                f"{counts.tolist()}; reference committed {ref.committed.tolist()} counts {ref.counts.tolist()}; "
                f"uniforms {torch.as_tensor(sampling['uniforms']).tolist()} tokens {tokens.tolist()}"
            )
        return len(explained)

    def _fused_cache_take(self, num_drafts, committed_tokens, committed_positions, accepted_counts, page_table):
        """The cached DraftResult when this propose asks for the commit the last fused step made it for (active rows:
        count, committed tokens within the count, their positions); else drop the cache and return None."""
        cache, self._fused_cache = self._fused_cache, None
        if cache is None or page_table is not None or int(num_drafts) != self.K:
            return None
        res, ctok, cpos, counts, act = cache
        try:
            B, T = self.B, self.T
            tok = _int_block(committed_tokens, B, T, "committed_tokens")
            pos = _int_block(committed_positions, B, T, "committed_positions")
            acc = _int_vec(accepted_counts, B, "accepted_counts")
        except ValueError:
            return None
        act = act.bool()
        used = (torch.arange(T)[None, :] < counts.reshape(-1, 1)) & act.reshape(-1, 1)
        same = (
            torch.equal(acc[act], counts.to(acc.dtype)[act])
            and torch.equal(tok.to(torch.int64)[used], ctok.to(torch.int64)[used])
            and torch.equal(pos.to(torch.int64)[used], cpos[used])
        )
        if not same:
            return None
        return DraftResult(res.draft_token_ids.clone(), res.num_valid.clone())

    # --------------------------------------------------------------------- #
    # Propose
    # --------------------------------------------------------------------- #
    def propose(
        self,
        committed_tokens,
        committed_positions,
        accepted_counts,
        page_table,
        decline=None,
        cap_blocks=None,
        keep_kv=None,
    ):
        """Refresh the anchor, reseed the MTP KV of the committed slots, then draft K tokens per user.
        ``decline`` (list[bool] [B]) rows draft nothing: their drafter writes go to scratch and their ids are 0;
        without it every row must fit. ``keep_kv`` (list[bool] [B]): declined rows that still get their anchor
        slot's MTP KV, so a row offered drafts again later drafts from a complete history.
        ``cap_blocks``: the caller's real table width (default: page_table's).
        Returns torch.int32 [B, K]."""
        if getattr(self, "_ch_g", None) is not None:  # QWEN36_SPEC_CHAIN
            self._chain_drop("propose")  # the drafter replay rewrites the id outputs a chain step would read
        core = self._propose_core_plan(
            committed_tokens, committed_positions, accepted_counts, page_table, decline, cap_blocks, keep_kv
        )
        self._propose_core_enqueue(core)
        drafts = self._propose_core_read(core)
        return self._propose_core_zero_declined(core, drafts)

    # The propose flow in pieces (propose / propose_draft_tokens are their composition; a fused step calls them apart):
    #   _propose_core_plan    host only: accepted_plan, fit / decline / keep, page-table fit, eager-vs-recorded decision
    #   _propose_core_enqueue table + pack H2D copies, drafter execute_trace (non-blocking), mtp_extra_steps, prestage
    #                         (the eager fallbacks run whole here and keep their own reads)
    #   _propose_core_read    the draft ids to host (one synchronize unless sync=False: the caller synced)
    def _propose_core_plan(
        self, committed_tokens, committed_positions, accepted_counts, page_table, decline, cap_blocks, keep_kv
    ):
        """Host planning of ``propose`` (no device call). Returns a ``_CorePlan``."""
        assert self.captured and self._vfeed is not None, "propose needs a verify first"
        B, T, K = self.B, self.T, self.K
        ctok = self._ints(committed_tokens, (B, T))
        cpos = self._ints(committed_positions, (B, T))
        m, anchor_rows, pending, p = accepted_plan(ctok, cpos, accepted_counts, B, T)
        if cap_blocks is None:
            cap_blocks = int(torch.as_tensor(page_table).shape[-1])
        fits = draft_rows_fit(p, K, table_capacity(cap_blocks, self.block_size))
        if decline is None:
            assert all(fits), f"draft window past the paged KV capacity ({cap_blocks} blocks x {self.block_size})"
            decline = [False] * B
        else:
            decline = [bool(d) for d in decline]
            assert len(decline) == B and all(d or f for d, f in zip(decline, fits)), "a non-declined row does not fit"
        keep = [False] * B if keep_kv is None else [bool(k) for k in keep_kv]
        # draft step 0 writes the anchor slot p; a declined row in keep gets it from one more reseed row (i = m)
        m_kv = [mu + int(d and k) for mu, d, k in zip(m, decline, keep)]
        pt = self._fit(page_table)
        self._check_scratch(pt)
        eager = B == 1 and p[0] > self.eager_reseed_len
        c = _CorePlan(ctok=ctok, cpos=cpos, m=m, pending=pending, p=p, pt=pt, m_kv=m_kv, decline=decline, eager=eager)
        c.recorded = bool(
            self.recorded_drafter and self.captured and self._dr_trace is not None and not all(decline) and not eager
        )
        c.dpt = pt
        if any(decline):  # declined rows draft at position 0 over an all-scratch table; their ids are discarded
            c.dpt = pt.clone()
            c.dpt[torch.tensor(decline)] = self._scratch_block
            c.p = [0 if d else pu for d, pu in zip(decline, p)]
            c.pending = [0 if d else t for d, t in zip(decline, pending)]
        return c

    def _propose_core_enqueue(self, c):
        """Device half of ``propose``: stage + enqueue the drafter trace (non-blocking), or run an eager fallback whole
        (its drafts land in ``c.drafts``)."""
        B, K = self.B, self.K
        if c.recorded:
            self._stage_table(c.dpt, self._mtp_pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            self._propose_recorded_enqueue(c.ctok, c.cpos, c.m, c.pending, c.p, c.pt, c.m_kv)
            return
        decline = c.decline
        self._stage_mtp_tables(c.pt)
        self._set_anchor(self._vfeed, c.m)
        if c.eager:
            self._reseed_mtp(c.ctok, c.cpos, c.m_kv)
        else:
            self._reseed_mtp_batched(c.ctok, c.cpos, c.m_kv, c.pt)
        if all(decline):
            c.drafts = torch.zeros(B, K, dtype=torch.int32)
            return
        if any(decline):
            self._stage_table(c.dpt, self._mtp_pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        c.drafts = self._draft(c.pending, self._hp_buf, c.p)

    def _propose_core_read(self, c):
        """Draft ids int32 [B, K] (before declined rows are zeroed), with one synchronize."""
        if not c.recorded:
            return c.drafts
        return self._drafts_from_steps(self._ids_to_host(self._dr_outs))

    def _propose_core_enqueue_reads(self, c):
        """Two-phase read, phase 1: enqueue the non-blocking D2H copies of the draft ids (None: nothing to read)."""
        return self._ids_enqueue_reads(self._dr_outs) if c.recorded else None

    def _propose_core_from_handles(self, c, handles):
        """Phase 2 (after the caller's synchronize): the draft ids int32 [B, K] from the phase-1 handles."""
        if not c.recorded:
            return c.drafts
        return self._drafts_from_steps(self._ids_from_handles(handles))

    def _propose_core_zero_declined(self, c, drafts):
        if any(c.decline):
            drafts[torch.tensor(c.decline)] = 0
        return drafts

    def _drafts_from_steps(self, steps):
        B, K = self.B, self.K
        return torch.tensor([[steps[k][u] for k in range(K)] for u in range(B)], dtype=torch.int32)

    def propose_draft_tokens(
        self, num_drafts, committed_tokens, committed_positions, accepted_counts, page_table=None, hidden=None
    ):
        """Runner propose contract in runner slots: committed block [B, 1+K] padded with -1 past accepted_counts;
        page_table None = the table of the last verify (already in engine rows). Rows whose
        window would leave the paged KV decline (num_valid 0). Returns DraftResult.
        = _propose_plan + _propose_stage_and_enqueue + _propose_read + _propose_finish (``self.propose`` is the
        staged flow's engine-level hook). A fused step's DraftResult is returned as is when the plugin's commit equals the
        one it was made for. ``hidden`` a FusedHandle (deferred serving): the engine bookkeeping of that step on this (engine) thread
        and its DraftResult, whether or not a later step was already enqueued."""
        if isinstance(hidden, FusedHandle):
            h, hidden = hidden, None
            if h.draft_result is not None:
                return h.draft_result
            if h.device:
                return self._deferred_finish(h)
            h.draft_result = self.propose_draft_tokens(
                num_drafts, committed_tokens, committed_positions, accepted_counts, page_table, None
            )
            return h.draft_result
        hit = self._fused_cache_take(num_drafts, committed_tokens, committed_positions, accepted_counts, page_table)
        if (
            self._ch_g is not None
        ):  # QWEN36_SPEC_CHAIN: the drafter outputs the next chain step reads are good only if the plugin took them
            if hit is not None and self._chain_prev is not None:
                self._chain_prev["taken"] = True
            elif hit is None:
                self._chain_drop("cache_miss")
        if hit is not None:
            return hit
        plan = self._propose_plan(
            num_drafts,
            committed_tokens,
            committed_positions,
            accepted_counts,
            page_table,
            hidden,
            build_core=False,
        )
        if plan.skip_all:
            return self._propose_finish(plan, None)
        drafts = self.propose(
            plan.ctok,
            plan.cpos,
            plan.acc,
            plan.page_table,
            decline=plan.decline,
            cap_blocks=plan.cap_blocks,
            keep_kv=plan.keep,
        )
        return self._propose_finish(plan, drafts)

    def _propose_plan(
        self,
        num_drafts,
        committed_tokens,
        committed_positions,
        accepted_counts,
        page_table=None,
        hidden=None,
        build_core=True,
    ):
        """All host planning of ``propose_draft_tokens`` (no device call): argument validation, runner-slot to row
        mapping, padding / fit / skip decisions and (``build_core``) the pack and table inputs. Returns a
        ``_ProposePlan``; ``plan.skip_all`` marks the B == 1 early return (no device work at all)."""
        B, T, K = self.B, self.T, self.K
        if int(num_drafts) != K:
            raise ValueError(f"num_drafts {num_drafts} != engine K {K}")
        if page_table is None:
            if self._last_pt is None:
                raise ValueError("page_table is None and no verify has run")
            page_table, cap_blocks = self._last_pt, self._last_cap_blocks
        else:
            cap_blocks = int(torch.as_tensor(page_table).shape[-1])
            (page_table,) = self._to_rows(page_table)
        if hidden is not None:
            raise ValueError("hidden must be None (the drafter's hidden handoff is on device)")
        ctok = _int_block(committed_tokens, B, T, "committed_tokens")
        cpos = _int_block(committed_positions, B, T, "committed_positions")
        acc = _int_vec(accepted_counts, B, "accepted_counts")
        if bool(((acc < 1) | (acc > T)).any()):
            raise ValueError(f"accepted_counts must be in [1, {T}], got {acc.tolist()}")
        ctok, cpos, acc = self._to_rows(ctok, cpos, acc)
        pad = [False] * B if self._row_active is None else [not bool(a) for a in self._row_active]
        acc = acc.clone()
        acc[torch.tensor(pad)] = 1  # a padding row's count is 1
        check_rows = range(B)
        if spec_hostfix_enabled():  # QWEN36_SPEC_HOSTFIX: validate on plain lists; the per-row loop only names an error
            ctl, cpl, al = ctok.tolist(), cpos.tolist(), acc.tolist()
            ok = all(
                pad[u]
                or (
                    min(ctl[u][: al[u]]) >= 0
                    and cpl[u][0] >= 1
                    and cpl[u][: al[u]] == list(range(cpl[u][0], cpl[u][0] + al[u]))
                )
                for u in range(B)
            )
            check_rows = () if ok else range(B)
        for u in check_rows:
            if pad[u]:
                continue
            c = int(acc[u])
            if bool((ctok[u, :c] < 0).any()):
                raise ValueError(f"row {self._slot_of(u)}: negative committed token in columns 0..{c - 1}")
            if not torch.equal(cpos[u, :c], cpos[u, 0] + torch.arange(c)) or int(cpos[u, 0]) < 1:
                raise ValueError(
                    f"row {self._slot_of(u)}: committed positions {cpos[u, :c].tolist()} are not consecutive from >= 1"
                )
        _, _, _, p = accepted_plan(ctok, cpos, acc, B, T)
        fit = draft_rows_fit(p, K, table_capacity(cap_blocks, self.block_size))
        decline = [(not f) or pd for f, pd in zip(fit, pad)]
        skip, skip_all = propose_skip_plan(self._no_draft, B)
        plan = _ProposePlan(ctok=ctok, cpos=cpos, acc=acc, page_table=page_table, cap_blocks=cap_blocks)
        # B > 1 still writes skipped rows' MTP KV (keep_kv): a row re-offered after a release has its full history
        if skip_all and B == 1:  # the runner withholds every row's drafts: no anchor/reseed/drafter work
            plan.skip_all = True
            return plan
        plan.keep = [s and not pd for s, pd in zip(skip, pad)]
        plan.decline = [d or s for d, s in zip(decline, skip)]
        if build_core:
            plan.core = self._propose_core_plan(ctok, cpos, acc, page_table, plan.decline, cap_blocks, plan.keep)
        return plan

    def _propose_stage_and_enqueue(self, plan):
        """Device half of the plan: table / pack H2D copies, drafter execute_trace (non-blocking), mtp_extra_steps and
        the next-verify prestage, in the order ``propose`` always had. Eager fallbacks run whole here."""
        if plan.skip_all:
            return
        self._propose_core_enqueue(plan.core)

    def _propose_read(self, plan):
        """Draft ids int32 [B, K] to host (_ids_to_host with its synchronize)."""
        if plan.skip_all:
            return None
        return self._propose_core_read(plan.core)

    def _propose_enqueue_reads(self, plan):
        """Phase 1 of the caller-synchronized read: enqueue the D2H copies. The caller then synchronizes ONCE and calls
        ``_propose_read_from_handles``; the copies must be enqueued before that synchronize."""
        if plan.skip_all:
            return None
        return self._propose_core_enqueue_reads(plan.core)

    def _propose_read_from_handles(self, plan, handles):
        if plan.skip_all:
            return None
        return self._propose_core_from_handles(plan.core, handles)

    def _propose_finish(self, plan, drafts):
        """Everything after the readback: decline zeroing, num_valid, _offered bookkeeping, _to_slots -> DraftResult."""
        B, K = self.B, self.K
        if plan.skip_all:
            self.skipped_propose_calls += 1
            self._offered.clear()
            return DraftResult(torch.zeros(B, K, dtype=torch.int32), torch.zeros(B, dtype=torch.int32))
        decline = plan.decline
        if plan.core is not None:
            drafts = self._propose_core_zero_declined(plan.core, drafts)
        num_valid = torch.tensor([0 if d else K for d in decline], dtype=torch.int32)
        self._offered = {u for u in range(B) if decline[u] is False}
        drafts, num_valid = self._to_slots(drafts, num_valid)
        return DraftResult(drafts, num_valid)

    def _propose_run_staged(self, plan):
        """The staged flow end to end (what a fused step splits around its single synchronize)."""
        self._propose_stage_and_enqueue(plan)
        return self._propose_finish(plan, self._propose_read(plan))

    # --------------------------------------------------------------------- #
    # Draft
    # --------------------------------------------------------------------- #
    def _draft(self, pending, anchor_hidden, p):
        """Draft K tokens per user from the MTP head at slot p_u: step 0 fuses (h_{p_u}, pending_u), step k
        (own hidden, previous draft). The chain stays on device; only K x B ids are read back. Returns int32 [B, K]."""
        B = self.B
        tok_tt = ttnn.from_torch(
            torch.tensor([[int(t)] for t in pending], dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        owned_tok = [tok_tt]
        h = anchor_hidden
        for k in range(self.K):
            logits, h_next = self.model.ttnn_mtp_decode_forward(
                h, tok_tt, [pu + k for pu in p], self._mtp_pt, **self._mtp_rd_kw(range(self.B))
            )
            idx = draft_argmax(self.mtp, logits)  # [1,1,B,1] uint32 ROW_MAJOR
            ttnn.deallocate(logits)
            tok_tt = ttnn.reshape(idx, (B, 1))
            owned_tok.append(tok_tt)
            if h is not anchor_hidden:
                ttnn.deallocate(h)
            h = h_next
        if h is not anchor_hidden:
            ttnn.deallocate(h)
        steps = [self._id_to_host(t) for t in owned_tok[1:]]  # steps[k][u]
        for t in owned_tok:
            ttnn.deallocate(t)
        return torch.tensor([[steps[k][u] for k in range(self.K)] for u in range(B)], dtype=torch.int32)

    # --------------------------------------------------------------------- #
    # Recorded drafter (anchor gather + batched reseed + K draft steps as one trace)
    # --------------------------------------------------------------------- #
    def _alloc_drafter_inputs(self):
        """Persistent drafter-trace inputs, zero-filled (a padding-only reseed + position-0 drafts: scratch-safe)."""
        B, K, R, nb, rd = self.B, self.K, self.B * self.T, self.nb, self.args.rope_head_dim
        al, z, bf, i32 = self._alloc_host_replicated, torch.zeros, ttnn.bfloat16, torch.int32
        rm, tile = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT
        self._dr_hostset = spec_hostset_trace_enabled()
        self._hp_bufs = {  # per-bucket anchor hidden [1,1,b,dim/tp]: allocated before any capture, like _hp_buf
            b: ttnn.zeros(
                [1, 1, b, self._hp_buf.shape[-1]],
                device=self.mesh,
                dtype=self._hp_buf.dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for b in self._dr_bks
        }
        if self._dr_hostset:
            self.model.rope.decode_rm_tables()  # the resident RoPE tables exist before the trace is captured
            self._dr_pack = al(pack_u32_rows([z(n, dtype=i32) for n in self._dr_seg_lens()]), ttnn.uint32, rm)
            self._dr_eye = al(torch.eye(R, dtype=torch.bfloat16).reshape(1, 1, R, R), bf, rm)
            self._dr_tok0, self._dr_pos, self._dr_cos, self._dr_sin = None, [], [], []
            self._dr_rs_tok = self._dr_rs_pos = self._dr_rs_cos = self._dr_rs_sin = None
            self._dr_rs_pt = al(torch.full((R, nb), self._scratch_block, dtype=i32), ttnn.int32, rm)
            self._dr_rs_cache = (
                torch.zeros(R, dtype=torch.bool),
                None,
                torch.full((R, nb), self._scratch_block, dtype=i32),
            )
            self._dr_bufs = [self._dr_pack, self._dr_eye, self._dr_rs_pt]
            return
        self._dr_tok0 = al(z(B, 1, dtype=i32), ttnn.uint32, rm)
        self._dr_pos = [al(z(B, dtype=i32), ttnn.int32, rm) for _ in range(K)]
        self._dr_cos = [al(z(1, B, 1, rd, dtype=torch.bfloat16), bf, tile) for _ in range(K)]
        self._dr_sin = [al(z(1, B, 1, rd, dtype=torch.bfloat16), bf, tile) for _ in range(K)]
        self._dr_rs_tok = al(z(R, 1, dtype=i32), ttnn.uint32, rm)
        self._dr_rs_pos = al(z(R, dtype=i32), ttnn.int32, rm)
        self._dr_rs_pt = al(torch.full((R, nb), self._scratch_block, dtype=i32), ttnn.int32, rm)
        self._dr_rs_cos = al(z(1, R, 1, rd, dtype=torch.bfloat16), bf, tile)
        self._dr_rs_sin = al(z(1, R, 1, rd, dtype=torch.bfloat16), bf, tile)
        self._dr_bufs = [
            self._dr_tok0,
            *self._dr_pos,
            *self._dr_cos,
            *self._dr_sin,
            self._dr_rs_tok,
            self._dr_rs_pos,
            self._dr_rs_pt,
            self._dr_rs_cos,
            self._dr_rs_sin,
        ]

    def _drafter_body(self, feed, b=None, dst_outs=None):
        """Anchor gather from ``feed`` [1,1,B*T,dim/tp], batched reseed, K chained draft steps; every input is a
        persistent buffer (device-input twin of _set_anchor + _reseed_mtp_batched + _draft). Returns the K ids
        tensors [1,1,B] uint32 (step k's drafts).

        ``b`` (a row bucket < B, QWEN36_SPEC_BUCKETS): the same body over engine rows 0..b-1 only (b*T reseed rows, b drafts
        per step), on the shared B-row inputs (sliced in the trace). Its K ids [1,1,b,1] are handed over IN PLACE (concat of
        the b new rows with the old rows b..B-1, then ttnn.copy: the destination keeps its address) into ``dst_outs`` (the
        shared K [1,1,B,1] outputs of the B-row trace, or stand-ins while warming), and ``dst_outs`` is returned. b = B or
        None is the B-row body, unchanged."""
        B, mtp = self.B, self.mtp
        b = B if b is None or int(b) == B else int(b)
        bucket = b < B
        assert not bucket or dst_outs is not None, "a row-bucket drafter body needs the shared K ids outputs"
        d = self._dr_inputs(b)
        owned = list(d["owned"])
        if bucket:
            rb = b * self.T
            dim = feed.shape[-1]
            feed = ttnn.slice(feed, [0, 0, 0, 0], [1, 1, rb, dim])
            rs_pt = ttnn.slice(self._dr_rs_pt, [0, 0], [rb, self.nb])
            mtp_pt = ttnn.slice(self._mtp_pt, [0, 0], [b, self.nb])
            owned += [feed, rs_pt, mtp_pt]
            hp = self._hp_bufs[b]
        else:
            rs_pt, mtp_pt, hp = self._dr_rs_pt, self._mtp_pt, self._hp_buf
        Hp = ttnn.matmul(d["anchor_sel"], feed, compute_kernel_config=self._anchor_cc)
        ttnn.copy(Hp, hp)
        ttnn.deallocate(Hp)
        _, hn = mtp.forward_decode(
            feed,
            d["rs_tok"],
            d["rs_pos"],
            d["rs_cos"],
            d["rs_sin"],
            rs_pt,
            need_logits=False,
            alias_kv_write=True,
            spec_n_users=b,
        )
        _free_maint(hn)
        h, tok, outs = hp, d["tok0"], []
        for k in range(self.K):
            logits, hn = mtp.forward_decode(h, tok, d["pos"][k], d["cos"][k], d["sin"][k], mtp_pt)
            idx = draft_argmax(mtp, logits)
            ttnn.deallocate(logits)
            tok = ttnn.reshape(idx, (b, 1))
            outs.append(idx)
            if h is not hp:
                ttnn.deallocate(h)
            h = hn
        ttnn.deallocate(h)
        for t in owned:
            ttnn.deallocate(t)
        if not bucket:
            return outs
        from models.demos.blackhole.qwen36.tt.spec_bucket import handoff_ids_concat

        for idx, dst in zip(outs, dst_outs):
            handoff_ids_concat(idx, dst, b)  # in place (ttnn.copy), never slice_write: dst keeps its address
            ttnn.deallocate(idx)
        return dst_outs

    def _dr_seg_lens(self):
        B, K, R = self.B, self.K, self.B * self.T
        return [B] + [B] * K + [R, R, B, R] + [B] * K

    def _dr_inputs(self, b=None):
        """The drafter body's inputs as a dict (anchor_sel, rs_tok, rs_pos, rs_cos, rs_sin, tok0, pos[k], cos[k], sin[k],
        owned). Without the packed layout: the persistent buffers (owned = []). With it (``_dr_hostset``): derived from
        ``_dr_pack`` in the trace. Rows of the pack (uint32 [2K+5, W], segment i at row i, columns [0, len)):
        0 pending tok0 [B]; 1..K step positions [B]; K+1 reseed tokens [R]; K+2 reseed positions [R]; K+3 anchor row
        u*T+m[u] [B]; K+4 reseed RoPE rows [R]; K+5..2K+4 RoPE rows of the step positions [B].
        Positions: typecast(uint32 -> int32) of the slice. cos/sin: ttnn.embedding on the RoPE tables
        (model._dl_rope_cos_sin) = the host's bf16 values (rope_row_index). The anchor one-hot: embedding of the anchor
        row index into an identity table (a row copy: 1.0 / 0.0 in bf16, exactly the host's anchor_selector)."""
        b = self.B if b is None else int(b)
        if not self._dr_hostset:
            assert b == self.B, "row-bucket drafter bodies need the packed drafter inputs (QWEN36_SPEC_HOSTSET)"
            return dict(
                anchor_sel=self._anchor_sel,
                rs_tok=self._dr_rs_tok,
                rs_pos=self._dr_rs_pos,
                rs_cos=self._dr_rs_cos,
                rs_sin=self._dr_rs_sin,
                tok0=self._dr_tok0,
                pos=self._dr_pos,
                cos=self._dr_cos,
                sin=self._dr_sin,
                owned=[],
            )
        B, K, R, pk, model = b, self.K, b * self.T, self._dr_pack, self.model
        owned = []

        def seg(row, n):
            return ttnn.slice(pk, [row, 0], [row + 1, n])

        def i32(t, shape):
            c = ttnn.typecast(t, ttnn.int32)
            ttnn.deallocate(t)
            v = ttnn.reshape(c, shape)
            owned.append(v)
            return v

        def rope(row, n):
            idx = seg(row, n)
            cos, sin = model._dl_rope_cos_sin(idx, n)
            owned.extend([idx, cos, sin])
            return cos, sin

        tok0 = ttnn.reshape(seg(0, B), (B, 1))
        pos = [i32(seg(1 + k, B), (B,)) for k in range(K)]
        rs_tok = ttnn.reshape(seg(K + 1, R), (R, 1))
        rs_pos = i32(seg(K + 2, R), (R,))
        aidx = seg(K + 3, B)
        sel = ttnn.embedding(aidx, self._dr_eye, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        anchor_sel = ttnn.reshape(sel, (1, 1, B, self.B * self.T))
        if b < self.B:  # the one-hot columns of rows >= b*T are all zero (anchor rows are u*T+m, u < b)
            full = anchor_sel
            anchor_sel = ttnn.slice(full, [0, 0, 0, 0], [1, 1, B, R])
            owned.append(full)
        rs_cos, rs_sin = rope(K + 4, R)
        steps = [rope(K + 5 + k, B) for k in range(K)]
        owned.extend([tok0, rs_tok, aidx, anchor_sel])
        return dict(
            anchor_sel=anchor_sel,
            rs_tok=rs_tok,
            rs_pos=rs_pos,
            rs_cos=rs_cos,
            rs_sin=rs_sin,
            tok0=tok0,
            pos=pos,
            cos=[c for c, _ in steps],
            sin=[sn for _, sn in steps],
            owned=owned,
        )

    def _propose_recorded(self, ctok, cpos, m, pending, p, pt, m_kv=None):
        """Stage the drafter trace's inputs (host->device copies only; _mtp_pt already staged), replay, read ids.
        The reseed rows always run (padding-only when every m == 0). ``m_kv``: reseed row counts (default m)."""
        self._propose_recorded_enqueue(ctok, cpos, m, pending, p, pt, m_kv)
        return self._drafts_from_steps(self._ids_to_host(self._dr_outs))  # steps[k][u]

    def _propose_recorded_enqueue(self, ctok, cpos, m, pending, p, pt, m_kv=None):
        """_propose_recorded up to and including the non-blocking drafter replay (and prestage): no read."""
        B, T, K, model = self.B, self.T, self.K, self.model
        if self._dr_hostset:
            return self._propose_recorded_packed_enqueue(ctok, cpos, m, pending, p, pt, m_kv)
        stage, bf, rm, tile = self._stage, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT
        stage(anchor_selector(m, B, T), self._anchor_sel, bf, tile)
        plan = reseed_row_plan(ctok, cpos, m if m_kv is None else m_kv, B, T)
        rtok, rpos, rpt = reseed_rows_host(plan, pt, T, self.nb, self._scratch_block)
        rcos, rsin = model._rope_tp_cos_sin_decode_torch(rpos, **self._pos_rd_kw(T))
        stage(rtok, self._dr_rs_tok, ttnn.uint32, rm)
        stage(rpos, self._dr_rs_pos, ttnn.int32, rm)
        self._stage_table(rpt, self._dr_rs_pt, ttnn.int32, rm)
        stage(rcos, self._dr_rs_cos, bf, tile)
        stage(rsin, self._dr_rs_sin, bf, tile)
        tok0, poss = draft_step_plan(pending, p, K)
        stage(tok0, self._dr_tok0, ttnn.uint32, rm)
        for k, pv in enumerate(poss):
            c, s = model._rope_tp_cos_sin_decode_torch(pv, **self._pos_rd_kw(1))
            stage(pv, self._dr_pos[k], ttnn.int32, rm)
            stage(c, self._dr_cos[k], bf, tile)
            stage(s, self._dr_sin[k], bf, tile)
        ttnn.execute_trace(self.mesh, self._dr_trace_for(self._vb), cq_id=0, blocking=False)
        if any(m):
            self.mtp_extra_steps += 1

    def _dr_pack_host(self, ctok, cpos, m, pending, p, m_kv):
        """The packed drafter inputs (see _dr_inputs) for one propose: (pack int32 [2K+5, W], tok [R,1], pos [R], real [R]).
        Element for element the values the unpacked path stages (draft_step_plan, reseed_row_plan, anchor_selector,
        _rope_tp_cos_sin_decode_torch), as indices: tests/test_spec_hostset_host.py proves it."""
        B, T, K, model = self.B, self.T, self.K, self.model
        rtok, rpos, _, real = reseed_rows_vec(ctok, cpos, m if m_kv is None else m_kv, None, T, 0, 0, need_pt=False)
        rows = model.rope.max_seq_len
        R = B * T
        steps = torch.as_tensor([int(x) for x in p], dtype=torch.int64)[None, :] + torch.arange(K)[:, None]  # [K, B]
        d = self._row_deltas()
        if isinstance(
            d, list
        ):  # the rows' own deltas: reseed rows are user-major (T per user), steps are k-major [K, B]
            delta = [x for x in d for _ in range(T)] + d * K
        else:
            delta = d
        ridx = rope_row_index(
            torch.cat([rpos.to(torch.int64), steps.reshape(-1)]), delta, rows
        )  # reseed rows, then steps
        segs = [
            [int(t) for t in pending],
            *steps.tolist(),
            rtok.reshape(-1),
            rpos,
            [u * T + int(m[u]) for u in range(B)],
            ridx[:R],
            *ridx[R:].reshape(K, B).tolist(),
        ]
        return pack_u32_rows(segs), rtok, rpos, real

    def _propose_recorded_packed(self, ctok, cpos, m, pending, p, pt, m_kv):
        self._propose_recorded_packed_enqueue(ctok, cpos, m, pending, p, pt, m_kv)
        return self._drafts_from_steps(self._ids_to_host(self._dr_outs))  # steps[k][u]

    def _propose_recorded_packed_enqueue(self, ctok, cpos, m, pending, p, pt, m_kv):
        """_propose_recorded with the packed inputs (QWEN36_SPEC_HOSTSET): ONE pack copy + the reseed page table (skipped
        while the real-row mask and the table repeat, which is most steps of a steady batch). _mtp_pt was staged by
        the caller. After the replay is enqueued the host would only wait for the first id read, so the next verify's
        inputs are prestaged then (_prestage_next_verify)."""
        B, T, K = self.B, self.T, self.K
        pack, _, _, real = self._dr_pack_host(ctok, cpos, m, pending, p, m_kv)
        h = ttnn.from_torch(
            pack,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=None,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        ttnn.copy_host_to_device_tensor(h, self._dr_pack)
        table = self._dr_rs_table(real, pt)
        if table is not None:
            self._stage(table, self._dr_rs_pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.execute_trace(self.mesh, self._dr_trace_for(self._vb), cq_id=0, blocking=False)
        if any(m):
            self.mtp_extra_steps += 1
        self._prestage_next_verify(cpos, m)

    def _dr_rs_table(self, real, pt):
        """The reseed page table [R, nb] the drafter trace must see for this step, or None when the device buffer already
        holds it. Row r = pt[r // T] if real[r] else the scratch block (what reseed_rows_vec builds). The host mirror of
        the buffer (_dr_rs_cache = (real mask, the pt its real rows were taken from or None, mirror)) is updated in
        place: only the rows whose real flag toggled are rewritten while the page table repeats; a changed table
        rebuilds the real rows. Returns a copy of the mirror (the copy to the device may outlive this call)."""
        T, R = self.T, self.B * self.T
        cached_real, cached_pt, host = self._dr_rs_cache
        anyreal = bool(real.any())
        if self._dr_rs_dirty:  # the device rewrote the buffer: restage it whole, whatever the mirror says
            self._dr_rs_dirty = False
            host.fill_(int(self._scratch_block))
            ridx = real.nonzero().reshape(-1)
            if ridx.numel():
                host[ridx] = pt[ridx // T].to(torch.int32)
            self._dr_rs_cache = (real.clone(), pt.clone() if anyreal else None, host)
            return host.clone()
        same_pt = cached_pt is not None and torch.equal(pt, cached_pt)
        if torch.equal(real, cached_real) and (same_pt or (not anyreal)):
            return None
        scratch = int(self._scratch_block)
        if same_pt or cached_pt is None:  # no stale real row: rewrite the toggled rows only
            for r in (real != cached_real).nonzero().reshape(-1).tolist():
                if bool(real[r]):
                    host[r] = pt[r // T]
                else:
                    host[r] = scratch
        else:  # the table changed under rows that were real: rebuild them all
            host.fill_(scratch)
            ridx = real.nonzero().reshape(-1)
            if ridx.numel():
                host[ridx] = pt[ridx // T].to(torch.int32)
        new_pt = cached_pt if same_pt else (pt.clone() if anyreal else None)
        if not anyreal:
            new_pt = None
        self._dr_rs_cache = (real.clone(), new_pt, host)
        return host.clone()

    def _prestage_next_verify(self, cpos, m):
        """While the drafter trace runs (the host would idle ~6 ms in the first id read), stage what the NEXT verify can
        know before the drafts exist: its row positions, RoPE rows and GDN ctrl page. They follow from the commit:
        user u's next start position is the bonus token's slot cpos[u, m[u]] and its mi_prev is m[u]; a row left out of
        the step runs at position 0 / mi 0 (verify's padding rule). NOT prestageable: the 32 verify token ids (the
        drafts), the page tables (the runner owns them and may add a block) and the KV-write check (it needs the
        next num_valid_drafts), and anything that depends on a request joining, finishing or being parked. A wrong
        guess is harmless: verify_traced stages its own words unless they equal what is already there."""
        model = self.model
        if not getattr(model, "_vfy_hostset", False):
            return
        if (
            self._ch_g is not None
        ):  # chain mode: the next verify's words come from the chain trace (they need m), nothing to prestage
            return
        act = [True] * self.B if self._row_active is None else [bool(a) for a in self._row_active]
        cp = torch.as_tensor(cpos).tolist()
        pos = [int(cp[u][int(m[u])]) if act[u] else 0 for u in range(self.B)]
        mi = [int(m[u]) if act[u] else 0 for u in range(self.B)]
        model.prestage_verify_aux(pos, mi)

    def _draft_warmup(self, pending, Hp, p):
        """One need_logits=True draft step before capture (compiles head norm, LM head, argmax, mesh_partition)."""
        logits, h = self.model.ttnn_mtp_decode_forward(Hp, [int(t) for t in pending], list(p), self._mtp_pt)
        idx = draft_argmax(self.mtp, logits)
        for t in (logits, idx, h):
            ttnn.deallocate(t)
        ttnn.synchronize_device(self.mesh)

    def _id_to_host(self, id_tt):
        """[*, B] uint32 device ids -> B ints, read from the device-0 replica."""
        t = ttnn.to_torch(ttnn.get_device_tensors(id_tt)[0])
        return [int(v) for v in t.reshape(-1)[: self.B]]

    def _ids_enqueue_reads(self, id_tts):
        """Phase 1 of a two-phase read. QWEN36_SPEC_HOSTFIX (default 1): enqueue every device-0 copy without
        blocking and return the host tensors. They are filled only by a synchronize that comes AFTER this call. Off:
        return the device-0 tensors (``_ids_from_handles`` then reads them blocking)."""
        devs = [ttnn.get_device_tensors(t)[0] for t in id_tts]
        if not spec_hostfix_enabled():
            return devs
        return [ttnn.from_device(d, blocking=False) for d in devs]

    def _ids_from_handles(self, handles):
        """Phase 2: handles from ``_ids_enqueue_reads`` -> B ints per tensor. Call after the synchronize."""
        return [[int(v) for v in ttnn.to_torch(h).reshape(-1)[: self.B]] for h in handles]

    def _ids_to_host(self, id_tts):
        """_id_to_host for several tensors. QWEN36_SPEC_HOSTFIX (default 1): all device-0 copies are enqueued
        without blocking, then one synchronize, instead of one blocking read round trip per tensor."""
        if not spec_hostfix_enabled():
            return [self._id_to_host(t) for t in id_tts]
        handles = self._ids_enqueue_reads(id_tts)
        ttnn.synchronize_device(self.mesh)
        return self._ids_from_handles(handles)

    # --------------------------------------------------------------------- #
    # MTP KV maintenance
    # --------------------------------------------------------------------- #
    def _warm_mtp_chunk(self, hidden, chunk_start, valid_len, prompt_ids, page_table_torch):
        """Warm one user's MTP KV over one prompt chunk: slot i fuses (base_hidden_i, token_{i+1}). Slot Tp-1
        is left to ``_warm_mtp_last``. Rows past valid_len write to the MTP scratch block.
        ``page_table_torch`` is this user's [1, nb]."""
        Tp = len(prompt_ids)
        if chunk_start >= Tp - 1:
            return
        bucket = hidden.shape[-2]
        toks = torch.zeros(1, bucket, dtype=torch.int32)
        n = min(bucket, Tp - 1 - chunk_start)
        toks[0, :n] = torch.tensor(prompt_ids[chunk_start + 1 : chunk_start + 1 + n], dtype=torch.int32)
        self.model.ttnn_mtp_prefill_forward(hidden, toks, chunk_start, page_table_torch, valid_len=n)

    def _warm_mtp_last(self, last_hidden, first_tok, slot, page_table_tt, row=None):
        """Write the MTP KV of a user's slot Tp-1 from (its feed row, the first generated token); the
        first draft attends to it. ``row``: the engine row (its M-RoPE delta positions the write; None = delta 0)."""
        _, h_next = self.model.ttnn_mtp_decode_forward(
            last_hidden,
            int(first_tok),
            slot,
            page_table_tt,
            need_logits=False,
            **self._mtp_rd_kw([row]),
        )
        _free_maint(h_next)

    def _reseed_mtp(self, committed_tokens, committed_positions, m):
        """Per-slot reseed (B == 1 past EAGER_RESEED_PROMPT_LEN): one 1-row drafter decode per committed slot."""
        T = self.T
        for u, (tok_row, pos_row) in enumerate(zip(committed_tokens.tolist(), committed_positions.tolist())):
            for i in range(m[u]):
                r = u * T + i
                row = self._eager_row(self._vfeed, r)
                _, h_next = self.model.ttnn_mtp_decode_forward(
                    row,
                    tok_row[i],
                    pos_row[i] - 1,
                    self._mtp_pt_rows[u],
                    need_logits=False,
                    **self._mtp_rd_kw([u]),
                )
                ttnn.deallocate(row)
                _free_maint(h_next)
                self.mtp_extra_steps += 1

    def _eager_row(self, feed, r):
        """feed row r as [1,1,1,dim/tp] via a fixed-shape one-hot matmul (exact; one program for every r;
        a sliced row would compile per r after capture)."""
        self._stage(last_row_selector(r, self.B * self.T), self._eager_sel, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        return ttnn.matmul(self._eager_sel, feed, compute_kernel_config=self._anchor_cc)

    def _eager_warmup(self):
        """Compile the eager reseed programs (selector matmul + 1-row drafter decode) so that setting
        eager_reseed_len later never compiles."""
        if self._eager_sel is None:
            self._eager_sel = self._alloc_host_replicated(
                last_row_selector(0, self.B * self.T), ttnn.bfloat16, ttnn.TILE_LAYOUT
            )
        z = ttnn.zeros(
            [1, 1, self.B * self.T, self._dim_frac],
            device=self.mesh,
            dtype=self._feed_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        row = self._eager_row(z, 0)
        self._warm_mtp_last(row, 0, 0, self._mtp_pt_rows[0])
        ttnn.deallocate(row)
        ttnn.deallocate(z)

    def _reseed_mtp_batched(self, committed_tokens, committed_positions, m, pt, scratch_only=False, feed=None):
        """Reseed as ONE fixed-shape drafter forward over all B*T independent rows. Row u*T+i is real for
        i < m[u], else padding (scratch block, position 0). ``scratch_only`` makes every row padding
        (warmup); ``pt`` is the fitted host table; ``feed`` defaults to the verify trace's rows."""
        B, T = self.B, self.T
        R = B * T
        feed = self._vfeed if feed is None else feed
        assert feed.shape[-2] == R, f"reseed wants {R} rows (B={B} x T={T}), got {feed.shape[-2]}"
        if scratch_only:
            m = [0] * B
        elif not any(m):
            return
        mesh, rep = self.mesh, ttnn.ReplicateTensorToMesh(self.mesh)
        plan = reseed_row_plan(committed_tokens, committed_positions, m, B, T)
        tok, pos, pt_rows = reseed_rows_host(plan, pt, T, self.nb, self._scratch_block)
        cos_t, sin_t = self.model._rope_tp_cos_sin_decode_torch(pos, **self._pos_rd_kw(T))
        tok_tt = ttnn.from_torch(tok, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
        pos_tt = ttnn.from_torch(pos, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
        pt_tt = ttnn.from_torch(pt_rows, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=rep)
        cos = ttnn.from_torch(cos_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
        sin = ttnn.from_torch(sin_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
        # spec_n_users=B: the KV write goes out as T calls of B rows, not B*T single-row calls. Padding rows
        # of different users collide on the scratch block (write-only garbage, SDPA read discarded).
        _, h_next = self.mtp.forward_decode(
            feed, tok_tt, pos_tt, cos, sin, pt_tt, need_logits=False, alias_kv_write=True, spec_n_users=B
        )
        _free_maint(tok_tt, pos_tt, pt_tt, cos, sin, h_next)
        self.mtp_extra_steps += 1

    def _reseed_warmup(self, rows, dim_frac, dtype):
        """Compile the batched-reseed program before capture; every row targets the scratch block."""
        z = ttnn.zeros(
            [1, 1, rows, dim_frac],
            device=self.mesh,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        zeros = torch.zeros(self.B, self.T, dtype=torch.int64)
        self._reseed_mtp_batched(zeros, zeros, [0] * self.B, None, scratch_only=True, feed=z)
        self.mtp_extra_steps -= 1  # warmup is not a loop cost
        ttnn.synchronize_device(self.mesh)
        ttnn.deallocate(z)

    # --------------------------------------------------------------------- #
    # Anchor
    # --------------------------------------------------------------------- #
    def _warm_big_chunk(self, hidden, page_table_torch):
        """Compile the per-chunk drafter warm and the last-row capture at the big chunk size on a real big-chunk output
        (engine row 0 scratch writes, as the warm prompts do)."""
        big = int(hidden.shape[-2])
        feed = self.model.spec_feed_rows(hidden)
        self._warm_mtp_chunk(feed, 0, big, [1] * (big + 2), page_table_torch)
        self._capture_last_row(feed, big - 1, 0)
        if feed is not hidden:
            ttnn.deallocate(feed)

    def _anchor_warmup(self, dim_frac, dtype):
        """Allocate the persistent anchor buffers and compile the refill matmul before capture (a per-iteration
        clone could land on trace-owned memory). One program serves every m."""
        B, T = self.B, self.T
        self._hp_buf = ttnn.zeros(
            [1, 1, B, dim_frac],
            device=self.mesh,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._anchor_sel = self._alloc_host_replicated(anchor_selector([0] * B, B, T), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        z = ttnn.zeros(
            [1, 1, B * T, dim_frac],
            device=self.mesh,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._set_anchor(z, [0] * B)
        ttnn.synchronize_device(self.mesh)
        ttnn.deallocate(z)

    def _set_anchor(self, feed_rows, m):
        """Refill the anchor buffer: row u = verify feed row u*T + m[u], as a one-hot matmul. Exact
        (bf16 one-hot, HiFi4 + fp32 accumulate), so the anchor is bit-identical to the row it names."""
        self._stage(anchor_selector(m, self.B, self.T), self._anchor_sel, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        Hp = ttnn.matmul(self._anchor_sel, feed_rows, compute_kernel_config=self._anchor_cc)
        ttnn.copy(Hp, self._hp_buf)
        ttnn.deallocate(Hp)


def get_spec_engine(model, batch_size, draft_len):
    """The engine cached on ``model._spec_engine``, keyed (B, K); a different key shuts the old one
    down first."""
    key = (int(batch_size), int(draft_len))
    eng = getattr(model, "_spec_engine", None)
    if eng is not None and (eng.B, eng.K) == key:
        return eng
    if eng is not None:
        eng.shutdown()
    eng = MTPSpecEngine(model, *key)
    model._spec_engine = eng
    return eng
