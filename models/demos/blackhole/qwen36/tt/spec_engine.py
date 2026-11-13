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
The drafter pairs (hidden at slot s, token at s+1) -> slot s."""
import dataclasses
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from loguru import logger
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.demos.blackhole.qwen36.tt.mtp import draft_argmax
from models.demos.blackhole.qwen36.tt.spec_accept import StepAccept, greedy_counts
from models.demos.blackhole.qwen36.tt.spec_chain import ChainInputs, chain_overlay
from models.demos.blackhole.qwen36.tt.spec_sample import sampling_plan, with_draws
from models.demos.blackhole.qwen36.tt.spec_step import (
    DRAFT_PENDING,
    STEP_DECLINED,
    STEP_DRAWN,
    STEP_KEEP,
    STEP_NUM_VALID,
    STEP_P0,
    STEP_ROPE_BASE,
    VERIFY_STEP_ROWS,
    draft_anchor,
    draft_reseed_positions,
    draft_reseed_rope,
    draft_reseed_tokens,
    draft_step_rope,
    draft_step_row,
    drafter_pack,
    drafter_pack_rows,
    pack_width,
    reseed_real_rows,
    reseed_table,
    rope_rows,
)

# Prompt length above which the reseed goes back to the per-slot loop (B == 1 only).
EAGER_RESEED_PROMPT_LEN = 131072
# Verify rows B*(K+1) must fit one decode tile (QKV heads, matmul configs, GDN core budget).
MAX_SPEC_ROWS = ttnn.TILE_SIZE
# Prompt lengths warmed in prepare: exact bucket lengths take the no-mask prefill path, the others the masked one.
_WARM_PROMPT_LENS = (100, 128, 200, 256, 400, 512, 900, 1024, 1900, 2048, 2148, 2248, 2448, 2948, 3948, 4096)
# MTPBlockEngine logs its one-enqueue counts at every this many verifies.
ONE_ENQUEUE_LOG_EVERY = 1000
# Runner contract: token id for unused tail columns of a returned block.
PLACEHOLDER_TOKEN_ID = -1
SPEC_MODES = ("argmax_ids", "logits")


@dataclass
class VerifyResult:
    """decode_forward result: argmax_ids int32 [B, 1+K] (unused tail columns = -1), logits [B, 1+K, V] or None."""

    spec_mode: str
    argmax_ids: torch.Tensor
    logits: torch.Tensor | None = None


class DeviceReads:
    """Non-blocking reads of device tensors, and the event recorded after them: result() waits for the event once and
    returns the host torch tensors."""

    def __init__(self, reads, event):
        self.reads, self.event, self._host = reads, event, None

    def result(self):
        if self._host is None:
            ttnn.event_synchronize(self.event)
            self._host = [ttnn.to_torch(r) for r in self.reads]
        return self._host


class PendingVerify:
    """One argmax_ids verify of MTPBlockEngine and what the propose_draft_tokens after it needs: its reads (the ids, the
    sampler's draws on a device-sampled step, and on a one-enqueue step the drafter pack and the drafts), the rows it
    ran (``num_valid``, ``active``, the fitted table ``pt`` and its width ``cap_blocks``), the slots whose drafts the
    runner withholds at that verify (``no_draft``) and its one-enqueue ``plan``. wait() reads once, on whichever thread
    first asks, and touches no engine state; result() returns the ids with columns past each row's valid drafts set to
    PLACEHOLDER_TOKEN_ID, as a synchronous verify returns them."""

    def __init__(
        self,
        B,
        K,
        num_valid,
        active,
        pt,
        cap_blocks,
        no_draft,
        plan=None,
        sample_plan=None,
        reads=None,
        rope_deltas=None,
    ):
        self.B, self.K, self.T = int(B), int(K), int(K) + 1
        # The slots' M-RoPE deltas the step ran with: a request released before its propose resets its slot's.
        self.rope_deltas = None if rope_deltas is None else list(rope_deltas)
        self.num_valid = torch.as_tensor(num_valid).reshape(-1).to(torch.int64).clone()
        self.active = torch.as_tensor(active, dtype=torch.bool).reshape(-1).clone()
        self.pt, self.cap_blocks, self.no_draft = pt, cap_blocks, set(no_draft)
        self.plan, self.sample_plan, self.reads = plan, sample_plan, reads
        self.ids = self.pack = self.drafts = None
        # The block's tokens [B, T] (drafts in columns 1..K) and column-0 positions [B]: known at a host-staged verify,
        # filled by the predecessor's propose on a chained one.
        self.tokens = self.p0 = None
        self.successor = None  # the chained verify submitted over this one, still to be filled by its propose
        self.materialized = None  # (committed, counts, DraftResult) the engine computed itself for a chained successor
        self._lock = threading.Lock()

    @property
    def event(self):
        return None if self.reads is None else self.reads.event

    def wait(self):
        """The merged ids [B, T] int64 (draws in a drawn row's columns), unmasked."""
        with self._lock:
            if self.ids is None:
                B, T = self.B, self.T
                host = self.reads.result()
                ids = host[0].reshape(-1)[: B * T].to(torch.int64).reshape(B, T)
                k = 1
                if self.sample_plan is not None:
                    ids = with_draws(ids, host[1], self.sample_plan)
                    k = 2
                if self.plan is not None:
                    self.pack = host[k].to(torch.int64)
                    steps = [r.reshape(-1)[:B].to(torch.int64) for r in host[k + 1 :]]
                    self.drafts = torch.stack(steps, dim=1).to(torch.int32)
                self.ids = ids
            return self.ids

    def result(self):
        return mask_unused_columns(self.wait(), self.num_valid).to(torch.int32)


@dataclass
class DraftResult:
    """propose_draft_tokens result: draft_token_ids int32 [B, K], num_valid int32 [B] in [0, K]."""

    draft_token_ids: torch.Tensor
    num_valid: torch.Tensor


# --------------------------------------------------------------------------------------------- #
# Pure host helpers (no device needed)
# --------------------------------------------------------------------------------------------- #
def free_tensors(*tensors):
    """Deallocate every tensor given, skipping None: a KV-maintenance drafter forward returns no hidden under TP."""
    for t in tensors:
        if t is not None:
            ttnn.deallocate(t)


def anchor_selector(m, B, T):
    """One-hot [1,1,B,B*T] bf16 with a single 1.0 at (u, u*T + m[u]): ``sel @ feed`` gathers each
    user's accepted-prefix row of the verify feed."""
    sel = torch.zeros(1, 1, B, B * T, dtype=torch.bfloat16)
    for u in range(B):
        sel[0, 0, u, u * T + int(m[u])] = 1.0
    return sel


def last_row_selector(i, bucket):
    """One-hot [1,1,1,bucket] bf16 with a 1.0 at column i: ``sel @ feed`` picks row i of a bucket."""
    sel = torch.zeros(1, 1, 1, bucket, dtype=torch.bfloat16)
    sel[0, 0, 0, int(i)] = 1.0
    return sel


def accepted_plan(committed_tokens, committed_positions, accepted_counts, B, T):
    """accepted_counts -> (m, anchor_rows, pending, anchor_pos), each a list of B ints.

    m[u] = accepted_counts[u] - 1 (accepted drafts); anchor row = u*T + m[u] of the verify feed;
    pending[u] = committed_tokens[u, m[u]] (the bonus token, at position p+1); anchor_pos[u] = p =
    committed_positions[u, m[u]] - 1 (the slot whose KV the first draft step writes)."""
    m = [int(a) - 1 for a in accepted_counts]
    assert len(m) == B and all(0 <= x < T for x in m), f"accepted_counts must be in [1, {T}], got {m}"
    anchor_rows = [u * T + m[u] for u in range(B)]
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
    for u in range(nv.numel()):
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


def verify_write_table(page_table, start_pos, num_valid_drafts, block_size):
    """[B, nb] table with each row's entries past the block of its last real column (start_pos + n) set to the
    null block 0, so padded verify columns write only that row's current block or the null block."""
    pt = torch.as_tensor(page_table).clone()
    last = torch.as_tensor(start_pos).reshape(-1).to(torch.int64)
    last = last + torch.as_tensor(num_valid_drafts).reshape(-1).to(torch.int64)
    cols = torch.arange(pt.shape[1]).reshape(1, -1)
    pt[cols > (last // int(block_size)).reshape(-1, 1)] = 0
    return pt


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
    gdn_state_in_blocks=False,
    rope_table_rows=None,
):
    """PER-DEVICE bytes of the MTP spec-decode state beyond plain decode. Pure: HF text config + tp only.

    Returns {"per_seq": {gdn_ring, gdn_conv_windows, drafter_hidden, total},   # x number of users B
             "fixed": {verify_io, mtp_scratch_block, engine_buffers, total},   # per engine
             "per_token": {mtp_kv, total}}                                     # per drafter KV token
    Counted: GDN fp32 per-token state ring and bf16 conv windows (_verify_win_buf, _spec_win_pad, _conv_win_buf),
    the engine's per-user _last_hidden, the verify trace's persistent inputs/outputs, the MTP cache's extra
    scratch block, the engine's anchor/selector buffers. NOT counted: rec_state, conv_states, base KV, the MTP
    KV blocks themselves (per_token x tokens), weights, trace-internal intermediates.
    ``gdn_state_in_blocks`` (MTPBlockEngine): the GDN state is in vLLM's KV cache blocks, so there is no ring and no
    per-user window; ``fixed`` counts instead the verify's per-group ctrl and index tables (gdn_verify), and verify_io
    drops the ring selectors.
    ``rope_table_rows``: the rows of the device RoPE tables the verify and drafter traces look up
    (spec_step.rope_table_rows of the model's max_seq_len); ``fixed`` counts them as rope_tables when given.
    Dims follow the model: TILE tensors pad their last two dims to 32; ROW_MAJOR are exact.
    ``fixed`` scales with B only through page tables / selectors / feed rows (all small). Page-table width nb =
    ``nb`` or ceil(max_seq_len / block_size); with neither, page tables count 0. ``num_layers`` overrides
    num_hidden_layers (truncated test models); ``mask_buckets`` the engine's last-row selector buckets."""
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
    windows = n_gdn * B * sum(_tile_bytes([r, C], bf) for r in win_rows)
    if gdn_state_in_blocks:
        ring = windows = 0
    hidden = B * _tile_bytes([1, 1, 1, dim_frac], bf)
    per_seq = {"gdn_ring": ring, "gdn_conv_windows": windows, "drafter_hidden": hidden}

    nb = int(nb) if nb else (-(-int(max_seq_len) // int(block_size)) if max_seq_len else 0)
    groups = B * _SPEC_SDPA_GROUPS_PER_USER[T] if T in _SPEC_SDPA_GROUPS_PER_USER else 1
    logits_w = tp * _pad32(vocab // tp) if vocab % tp == 0 else _pad32(vocab)  # vocab-sharded head gathers
    verify_io = (
        _rm_bytes([VERIFY_STEP_ROWS, pack_width(rows)], i32)  # step_buf
        + _rm_bytes([rows, nb], i32)
        + _rm_bytes([B, nb], i32)
        + _rm_bytes([groups, nb], i32)  # page tables
        + (0 if gdn_state_in_blocks else _rm_bytes([B * nv_tp], i32))  # state_idx
        + (0 if gdn_state_in_blocks else _tile_bytes([B, kconv - 1 + T, kconv - 1 + 2 * T], bf))  # conv_sel
        + _tile_bytes([1, 1, rows, logits_w], bf)
        + _rm_bytes([1, 1, rows, logits_w], bf)  # logits TILE + RM copy
        + _tile_bytes([1, 1, rows, dim_frac], bf)  # rows_out (drafter feed)
        + _rm_bytes([1, 1, rows], i32)  # ids_out
    )
    block_bytes = 2 * n_local_kv * int(block_size) * head_dim * int(kv_dtype_bytes)
    buckets = _MASK_BUCKETS_DEFAULT if mask_buckets is None else mask_buckets
    engine = (
        sum(_tile_bytes([1, 1, 1, b], bf) for b in buckets)  # _last_sel
        + _rm_bytes([B, nb], i32)
        + B * _rm_bytes([1, nb], i32)  # _mtp_pt, _mtp_pt_rows
        + _tile_bytes([1, 1, B, dim_frac], bf)  # _hp_buf
        + _tile_bytes([1, 1, B, rows], bf)  # _anchor_sel
        + _tile_bytes([1, 1, 1, rows], bf)  # _eager_sel
        + (_tile_bytes([1, 1, rows, B], bf) + _tile_bytes([1, 1, rows, rows], bf) if B > 1 else 0)  # _join_take/keep
    )
    fixed = {"verify_io": verify_io, "mtp_scratch_block": block_bytes, "engine_buffers": engine}
    if rope_table_rows is not None:
        fixed["rope_tables"] = 2 * _rm_bytes([int(rope_table_rows), rope_dim], bf)  # cos, sin
    if gdn_state_in_blocks:
        groups = -(-n_gdn // max(1, L - n_gdn))  # MambaSpec groups: one GDN layer per attention layer slot each
        ctrl_words = -(-(1 + B + 2 * B * nv_tp) // 16) * 16
        fixed["gdn_verify"] = groups * (_rm_bytes([ctrl_words], i32) + _rm_bytes([B, 2 * T * nv_tp], i32))
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


class MTPSpecEngine:
    """MTP speculative decode engine over B users (see module docstring)."""

    sample_on_device = False  # the verify sampler is prepared (MTPBlockEngine, a device-sampling launch)

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
        # Verify runs the fused GDN op; plain decode must use the same math or greedy near-ties flip. With the GDN
        # state in vLLM KV cache blocks no model-owned decode exists.
        assert model.gdn_blocks is not None or model.gdn_fused_decode, (
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
        self._rope_deltas = [0] * self.B  # per slot: the M-RoPE delta of its request (0 for text)
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
        # A/B + safety switch: 1 = run the drafter eagerly even when its trace is recorded
        self.recorded_drafter = os.environ.get("QWEN36_SPEC_EAGER_DRAFTER", "0") != "1"
        self._dr_bufs = []  # persistent drafter-trace inputs (all of the _dr_* tensors below)
        self._dr_pack = None  # [2K+5, W] uint32: the per-step integers (spec_step drafter pack)
        self._dr_rs_pt = None  # [B*T, nb] int32: the reseed rows' page table (spec_step.reseed_table)
        self._dr_iota = None  # [1, B*T] float32 TILE 1 .. B*T: the anchor one-hot's column numbers
        self._dr_rs_staged = None  # host table _dr_rs_pt holds, None when unknown
        self._mtp_pt_staged = None  # host table _mtp_pt holds, None when unknown
        self._dr_trace = None  # trace id: anchor gather + batched reseed + K draft steps
        self._dr_outs = []  # K persistent [1,1,B] uint32 ids outputs of that trace
        self._last_pt = None  # fitted host page table of the most recent verify
        self._fresh = {}  # slot -> prompt length, prefilled and awaiting the seed verify
        self._active = set()  # slots holding a live request
        self._parked = {}  # slot -> per-GDN-layer host state of a live row left out of a verify
        self._prompts = {}  # slot -> (prompt ids, page-table row) of a fresh slot, for a re-prefill
        self._fresh_dirty = set()  # fresh slots left out of a seed, which overwrote their prefill state
        self._row_active = None  # [B] bool: rows that held a request in the last verify (None = all)
        self._offered = set()  # slots the last propose gave drafts (num_valid > 0)
        self._no_draft = set()  # slots whose drafts the runner withholds: propose skips them
        self._speculated = set()  # slots that verified with n > 0 since prefill (never marked)
        self._withheld_streak = {}  # slot -> consecutive offers answered with n == 0
        self.no_draft_streak = NO_DRAFT_STREAK
        self.skipped_propose_calls = 0  # proposes skipped because every row was marked

    # --------------------------------------------------------------------- #
    # Host helpers
    # --------------------------------------------------------------------- #
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

    def _row_rope_deltas(self, per_user):
        """The users' M-RoPE deltas (_rope_deltas) for a user-major block of B * ``per_user`` rows."""
        return [d for d in self._rope_deltas for _ in range(per_user)]

    def _stage(self, host, dev, dtype, layout):
        """Copy a torch tensor into a persistent device tensor (address preserved)."""
        h = ttnn.from_torch(
            host, dtype=dtype, layout=layout, device=None, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh)
        )
        ttnn.copy_host_to_device_tensor(h, dev)

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
        self._stage_table(pt, "_mtp_pt", "_mtp_pt_staged")
        for u in range(self.B):
            self._stage(pt[u : u + 1].contiguous(), self._mtp_pt_rows[u], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)

    def _stage_table(self, host, buf, mirror):
        """Copy the int32 table ``host`` into the persistent buffer named ``buf`` unless the host mirror named ``mirror``
        says the buffer already holds it."""
        host = torch.as_tensor(host).to(torch.int32)
        held = getattr(self, mirror)
        if held is not None and torch.equal(held, host):
            return
        self._stage(host, getattr(self, buf), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        setattr(self, mirror, host.clone())

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
        pt = self._alloc_engine_buffers(pt0)

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
        for L in warm_lens:  # dummy prompts through the real (eager) prefill path on slot 0
            self.prefill(0, [1] * L, pt[0:1], _cap_blocks=width)
            self._fresh.pop(0, None)
            self._active.discard(0)

        if B == 1:  # allocate + compile the recorded chunk program (re-runs masked-bucket warmup itself)
            pt_prefill = model._fit_spec_page_table(pt[0:1], self.nb_prefill)
            model.prepare_prefill_trace_chunked(self.mesh, pt_prefill, chunk_size=2048)

        self.prefill_prepared = True
        logger.info(
            f"[spec_engine] prefill prepared B={B} K={self.K} nb={self.nb} block={self.block_size} "
            f"warm_prompts={warm_lens} in {time.perf_counter() - t0:.1f}s"
        )

    def _alloc_engine_buffers(self, pt0):
        """Allocate the persistent engine buffers for the caller's table width; returns the fitted table."""
        B = self.B
        self.nb = int(pt0.shape[-1])  # must stay <= the cache block count (verify / drafter paged_update_cache)
        self.nb_prefill = spec_round_up(self.nb)  # the recorded chunk program needs a multiple of 32
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
        self._last_sel = {
            b: self._alloc_host_replicated(
                torch.zeros(1, 1, 1, b, dtype=torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT
            )
            for b in self.model._PREFILL_MASK_BUCKETS
        }
        self._mtp_pt = self._alloc_host_replicated(pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        self._mtp_pt_rows = [
            self._alloc_host_replicated(pt[u : u + 1].contiguous(), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT) for u in range(B)
        ]
        self._anchor_warmup(self._dim_frac, self._feed_dtype)
        return pt

    def record_prefill(self):
        """Record the 2048-token chunk prefill trace (B == 1; B > 1 prefill stays eager). Idempotent."""
        assert self.prefill_prepared, "record_prefill before prepare_prefill"
        if self.prefill_recorded:
            return
        if self.B == 1:
            self.model.record_prefill_trace_chunked(self.mesh)
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
        self._prepare_verify()

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
            ttnn.synchronize_device(self.mesh)
            ttnn.deallocate(feed)

        ttnn.synchronize_device(self.mesh)
        self.prepared = True
        logger.info(f"[spec_engine] decode prepared in {time.perf_counter() - t0:.1f}s")

    def _prepare_verify(self):
        """Compile the seed's programs, prepare the verify trace and compile the join programs."""
        model, B, T = self.model, self.B, self.T
        # Compile the seed's programs (T = 1 forward, feed-row scatter and copy) before any capture.
        _, f = model.seed_spec_step([0] * B, [0] * B, self._prep_pt)
        r0, r1 = self._seed_rows(f), self._seed_rows(f)
        ttnn.copy(r0, r1)
        for t in (r0, r1):
            ttnn.deallocate(t)
        model.prepare_verify_trace(self._prep_pt, T, warm_positions=[0] * B)
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
                dn.remap_spec_slots(list(range(B))[::-1])  # a full reversal slices every row offset once
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
                self.model.record_verify_trace()
                if self.recorded_drafter:  # ids outputs are read right after each replay
                    feed = self.model._vfy_rows_out
                    self._dr_trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
                    self._dr_outs = self._drafter_body(feed)
                    ttnn.end_trace_capture(self.mesh, self._dr_trace, cq_id=0)
                    ttnn.synchronize_device(self.mesh)
                    logger.info(f"[spec_engine] drafter trace captured (anchor + reseed + {self.K} draft steps)")
                else:
                    logger.info("[spec_engine] drafter runs eagerly (QWEN36_SPEC_EAGER_DRAFTER=1)")
                self._capture_step()
        finally:
            self.mesh.set_program_cache_misses_allowed(True)
        self.captured = True

    def _capture_step(self):
        """Traces captured after the drafter's, inside capture_decode's scope (MTPBlockEngine's step trace)."""

    def shutdown(self):
        """Release the traces and every persistent engine buffer; a later prepare starts over."""
        model = self.model
        t0 = time.time()
        logger.info("[spec_engine] shutdown: releasing traces")
        model.release_verify_trace()
        model._vfy_prepared_key = None
        if self.prefill_recorded and getattr(model, "_chunked_trace_id", None) is not None:
            ttnn.release_trace(model.device, model._chunked_trace_id)
            model._chunked_trace_id = None
            model._chunked_prepared_key = None
        if self._dr_trace is not None:
            ttnn.release_trace(self.mesh, self._dr_trace)
            self._dr_trace = None
        for t in (*self._dr_outs, *self._dr_bufs):
            ttnn.deallocate(t)
        self._dr_outs, self._dr_bufs = [], []
        self._dr_pack = self._dr_rs_pt = self._dr_iota = None
        self._dr_rs_staged = self._mtp_pt_staged = None
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
        ]
        for t in bufs:
            if t is not None:
                ttnn.deallocate(t)
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
            )
        per_seq["drafter_hidden"] = sum(nbytes(t) for t in self._last_hidden)
        vfy = (
            "step_buf",
            "kvpt_buf",
            "kvpt_users",
            "kvpt1_buf",
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
        fixed = {
            "verify_io": sum(nbytes(getattr(m, f"_vfy_{n}", None)) for n in vfy),
            "mtp_scratch_block": kv_bytes // n_blocks,
            "engine_buffers": sum(
                nbytes(t)
                for t in (
                    *self._last_sel.values(),
                    self._mtp_pt,
                    *self._mtp_pt_rows,
                    self._hp_buf,
                    self._anchor_sel,
                    self._eager_sel,
                    self._join_take,
                    self._join_keep,
                )
            ),
        }
        if getattr(m, "_rope_rm", None) is not None:
            fixed["rope_tables"] = sum(nbytes(t) for t in m._rope_rm)
        per_token = {"mtp_kv": kv_bytes // (n_blocks * int(att.paged_k.shape[-2]))}
        for d in (per_seq, fixed, per_token):
            d["total"] = sum(d.values())
        return {"per_seq": per_seq, "fixed": fixed, "per_token": per_token}

    def release(self, slot):
        """Mark a slot empty and re-offer drafts to every row (no device work)."""
        self._active.discard(slot)
        self._fresh.pop(slot, None)
        self._parked.pop(slot, None)
        self._prompts.pop(slot, None)
        self._fresh_dirty.discard(slot)
        self._rope_deltas[slot] = 0
        reset_draft_skip(self, slot)
        # the leaving request may be why other rows' drafts were withheld: offer them again
        self._no_draft.clear()
        self._withheld_streak.clear()

    def remap_slots(self, remap):
        """Apply a runner slot_remap (row i takes slot remap[i]'s state) to every GDN layer and the per-slot host
        state; paged KV follows the page tables. Identity is a no-op."""
        idx = [int(x) for x in remap]
        if sorted(idx) != list(range(self.B)):
            raise ValueError(f"slot_remap {idx} is not a permutation of range({self.B})")
        if idx == list(range(self.B)):
            return
        self._remap_gdn(idx)
        new_of = {old: new for new, old in enumerate(idx)}
        self._last_hidden = [self._last_hidden[j] for j in idx]
        self._rope_deltas = [self._rope_deltas[j] for j in idx]
        self._fresh = {new_of[u]: n for u, n in self._fresh.items()}
        self._withheld_streak = {new_of[u]: n for u, n in self._withheld_streak.items()}
        self._parked = {new_of[u]: v for u, v in self._parked.items()}
        self._prompts = {new_of[u]: v for u, v in self._prompts.items()}
        self._fresh_dirty = {new_of[u] for u in self._fresh_dirty}
        self._active = {new_of[u] for u in self._active}
        self._offered = {new_of[u] for u in self._offered}
        self._no_draft = {new_of[u] for u in self._no_draft}
        self._speculated = {new_of[u] for u in self._speculated}

    def _remap_gdn(self, idx):
        for dn in self._gdn:
            dn.remap_spec_slots(idx)

    # --------------------------------------------------------------------- #
    # Prefill
    # --------------------------------------------------------------------- #
    def prefill(self, slot, prompt_ids, page_table_row, _cap_blocks=None):
        """Eager prefill into state slot ``slot``, warming its MTP KV; returns host float logits [vocab].
        Replays the recorded chunk trace when B == 1 and record_prefill ran, else the eager path.
        The GDN seed and the MTP KV of slot Tp-1 are written by the first ``verify``."""
        reset_draft_skip(self, slot)
        self._parked.pop(slot, None)
        self._fresh_dirty.discard(slot)
        model = self.model
        assert 0 <= slot < self.B, f"slot {slot} out of range [0,{self.B})"
        assert self._last_hidden, "prefill before prepare"
        prompt_list = [int(t) for t in torch.as_tensor(prompt_ids).reshape(-1)]
        Tp = len(prompt_list)
        self._prompts[slot] = (prompt_list, torch.as_tensor(page_table_row).clone())
        cap_blocks = int(torch.as_tensor(page_table_row).shape[-1]) if _cap_blocks is None else _cap_blocks
        assert Tp + self.T <= table_capacity(
            cap_blocks, self.block_size
        ), f"prompt {Tp} + {self.T} verify rows does not fit {cap_blocks} blocks x {self.block_size}"
        pt_row = model._fit_spec_page_table(page_table_row, self.nb)
        assert pt_row.shape[0] == 1, "prefill takes ONE page-table row"
        self._check_scratch(pt_row)
        prompt = torch.tensor([prompt_list], dtype=torch.int32)
        captured_row = [False]

        def _on_chunk(hidden, chunk_start, valid_len):
            # Never frees `hidden` (the caller does, or it is the trace's persistent output).
            feed = model.spec_feed_rows(hidden)
            self._warm_mtp_chunk(feed, chunk_start, valid_len, prompt_list, pt_row)
            if chunk_start + valid_len >= Tp:  # the chunk holding slot Tp-1
                self._capture_last_row(feed, Tp - 1 - chunk_start, slot)
                captured_row[0] = True
            if feed is not hidden:
                ttnn.deallocate(feed)

        if self.B == 1 and self.prefill_recorded and self.recorded_prefill:
            pt_rec = model._fit_spec_page_table(pt_row, self.nb_prefill)  # MTP chunk fill keeps the nb-wide pt_row
            logits_dev = model.prefill_traced_chunked(prompt, pt_rec, Tp, on_chunk=_on_chunk)
        else:
            logits_dev = model.prefill_for_spec(prompt, pt_row, Tp, _on_chunk, slot=slot)
        assert captured_row[0], f"prefill never delivered the chunk holding slot {Tp - 1}"
        if isinstance(logits_dev, torch.Tensor):
            lt = logits_dev
        else:
            lt = ttnn.to_torch(logits_dev, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0))
            ttnn.deallocate(logits_dev)
        self._fresh[slot] = Tp
        self._active.add(slot)
        return lt.reshape(-1)[: self.vocab].float()

    def _reprefill(self, slot):
        """Re-run a fresh slot's prefill (a seed overwrote its prefill GDN state while it was left out)."""
        prompt, row = self._prompts[slot]
        logger.info(f"[spec_engine] re-prefill slot {slot} ({len(prompt)} tokens)")
        self.prefill(slot, prompt, row)

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
        self, tokens, start_pos, accepted_counts, page_table, read_logits=False, active=None, num_valid_drafts=None
    ):
        """Replay the verify trace over [B, T] tokens at start_pos[u] + j. The first verify after prefill is
        the seed (see _seed): row 0 holds the first generated token, other rows are filler, accepted_counts = 1.
        Fresh rows may join running ones (see _join). A live row left out is parked on host and restored when it
        returns; a fresh row left out defers its seed.
        ``active`` (bool [B], None = all): False rows are padding and run over the null block.
        ``num_valid_drafts`` ([B], None = all columns real): the target's KV writes use verify_write_table;
        the drafter keeps the full table.
        Returns (ids int32 [B, T], logits [B, T, vocab] or None)."""
        assert self.captured, "verify before capture"
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
        rows = [u for u in range(B) if active[u]]
        unknown = [u for u in rows if u not in self._active]
        if unknown:
            raise ValueError(f"rows {unknown} carry a request that was never prefilled into that slot")
        # contract 4e: a live row left out of a step keeps its GDN state on host until it returns
        for u in range(B):
            if not active[u] and u in self._active and u not in self._fresh and u not in self._parked:
                self._parked[u] = [dn.save_spec_user(u) for dn in self._gdn]
                logger.info(f"[spec_engine] parked slot {u} (left out of a step)")
        back = [u for u in rows if u in self._parked]
        if back:
            for li, dn in enumerate(self._gdn):
                dn.load_spec_users({u: parked_rows_host(*self._parked[u][li], int(mi_prev[u]), T) for u in back})
            for u in back:
                del self._parked[u]
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

        fresh_now = {u: Tp for u, Tp in self._fresh.items() if active[u]}
        if fresh_now:
            # the seed rewrites every row's durable GDN state: fresh rows left out must re-prefill before their own seed
            self._fresh_dirty |= set(self._fresh) - set(fresh_now)
            for u in sorted(fresh_now):
                if u in self._fresh_dirty:
                    self._reprefill(u)
            # Each fresh user's slot Tp-1 MTP KV, then the seed (its filler rows are never run).
            self._stage_mtp_tables(pt)
            for u, Tp in sorted(fresh_now.items()):
                assert int(start_pos[u]) == Tp, f"slot {u}: start_pos {int(start_pos[u])} != prompt length {Tp}"
                self._warm_mtp_last(self._last_hidden[u], int(tokens[u, 0]), Tp - 1, self._mtp_pt_rows[u])
            fresh = sorted(fresh_now)
            for u in fresh:
                del self._fresh[u]
                self._prompts.pop(u, None)
                self._fresh_dirty.discard(u)
            if set(fresh) == set(rows):
                return self._seed(tokens, start_pos, vpt, read_logits)
            return self._join(tokens, start_pos, mi_prev, vpt, active, fresh, read_logits)

        args = (tokens.tolist(), start_pos.tolist(), mi_prev.tolist())
        kw = dict(read_logits=read_logits, page_tables=vpt, inactive=(~active).tolist())
        if self._timing:
            ids, vfeed, lt = self._verify_split(*args, **kw)
        else:
            ids, vfeed, lt = self.model.verify_traced(*args, **kw)
        self._vfeed = vfeed
        ids = torch.tensor(ids, dtype=torch.int32).reshape(B, T)
        return ids, (None if lt is None else lt.reshape(B, T, -1))

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
        mamba_state=None,
        sampling=None,
        one_enqueue=True,
        defer=False,
        predecessor=None,
        keep_pending=False,
    ):
        """Runner verify contract over a [B, 1+K] block (see sanitize_verify_block); slot_remap is applied after
        validation. ``defer`` (MTPBlockEngine, argmax_ids): return the verify unread, as a PendingVerify in
        ``argmax_ids``. ``predecessor`` (MTPBlockEngine): the unread PendingVerify a chained verify continues; the
        live rows of the block are placeholders (_decode_chained). ``mamba_state``: the GDN state blocks of the step (MTPBlockEngine only). ``sampling``: the step's
        device sampling parameters (the plugin's ``sampling_params``, one entry per row) or None; a row they sample
        returns the device sampler's draws in place of its argmax ids (spec_sample). ``one_enqueue``: an argmax_ids
        verify may also run the device acceptance and the drafter (MTPBlockEngine); a narrow decode step, which the
        runner follows with a propose over positions it derives from its own request state, passes False.
        Returns VerifyResult."""
        if spec_mode not in SPEC_MODES:
            raise ValueError(f"spec_mode {spec_mode!r} not in {SPEC_MODES}")
        if predecessor is not None:
            if spec_mode != "argmax_ids" or not defer:
                raise ValueError("a chained verify is an argmax_ids verify read later (defer)")
            return self._decode_chained(
                predecessor, start_pos, num_valid_drafts, page_table, slot_remap, mamba_state, sampling
            )
        clean, start0, active = sanitize_verify_block(tokens, start_pos, num_valid_drafts, self.K)
        if clean.shape[0] != self.B:
            raise ValueError(f"block has {clean.shape[0]} rows, engine has B={self.B}")
        if not bool(active.any()):
            raise ValueError("no active row in the verify block")
        plan = None
        if sampling is not None:
            plan = sampling_plan(sampling, torch.where(active, start0, -1).tolist(), self.T)
            plan = plan if plan.any_sampled else None
        if plan is not None and not self.sample_on_device:
            raise ValueError(
                "a verify of a device-sampled step needs the verify sampler, which the engine prepares only after "
                "warmup_model_decode(can_sample_on_device=True)"
            )
        if slot_remap is not None:
            self.remap_slots(slot_remap)
        self._no_draft = note_withheld_drafts(
            self._offered,
            self._no_draft,
            self._speculated,
            self._withheld_streak,
            num_valid_drafts,
            self.no_draft_streak,
        )
        ids, logits = self.verify(
            clean,
            start0,
            accepted_counts,
            page_table,
            read_logits=spec_mode == "logits",
            active=active,
            num_valid_drafts=num_valid_drafts,
            **({} if mamba_state is None else {"mamba_state": mamba_state}),
            **({} if plan is None else {"sample_plan": plan}),
            **(
                {"one_enqueue": True} if one_enqueue and spec_mode == "argmax_ids" and self._one_enqueue_ready() else {}
            ),
            **({"defer": True} if defer and spec_mode == "argmax_ids" else {}),
            **({"keep_pending": True} if keep_pending else {}),
        )
        if isinstance(ids, PendingVerify):
            return VerifyResult(spec_mode, ids, logits)
        return VerifyResult(spec_mode, mask_unused_columns(ids, num_valid_drafts), logits)

    def _one_enqueue_ready(self):
        """Whether a verify can also run the device acceptance and the drafter (MTPBlockEngine once captured)."""
        return False

    def _decode_chained(self, predecessor, start_pos, num_valid_drafts, page_table, slot_remap, mamba_state, sampling):
        raise NotImplementedError(f"{type(self).__name__} builds no chained verify")

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
        if self.recorded_drafter and self.captured and self._dr_trace is not None and not all(decline) and not eager:
            # declined rows draft at position 0 over an all-scratch table; their ids are discarded
            self._stage_table(self._draft_table(pt, decline), "_mtp_pt", "_mtp_pt_staged")
            p0 = [max(pu - mu, 0) for pu, mu in zip(p, m)]  # position of verify column 0; 0 for a padding row
            drafts = self._propose_recorded(ctok, m, p0, decline, keep, pt)
            if any(decline):
                drafts[torch.tensor(decline)] = 0
            return drafts
        self._stage_mtp_tables(pt)

        self._set_anchor(self._vfeed, m)
        if eager:
            self._reseed_mtp(ctok, cpos, m_kv)
        else:
            self._reseed_mtp_batched(ctok, cpos, m_kv, pt)
        if all(decline):
            return torch.zeros(B, K, dtype=torch.int32)
        if any(decline):  # declined rows draft at position 0 over an all-scratch table; their ids are discarded
            self._stage_table(self._draft_table(pt, decline), "_mtp_pt", "_mtp_pt_staged")
            p = [0 if d else pu for d, pu in zip(decline, p)]
            pending = [0 if d else t for d, t in zip(decline, pending)]
        drafts = self._draft(pending, self._hp_buf, p)
        drafts[torch.tensor(decline)] = 0
        return drafts

    def propose_draft_tokens(
        self, num_drafts, committed_tokens, committed_positions, accepted_counts, page_table=None, hidden=None
    ):
        """Runner propose contract: committed block [B, 1+K] padded with -1 past accepted_counts; page_table None =
        the table of the last verify. Rows whose
        window would leave the paged KV decline (num_valid 0). Returns DraftResult."""
        B, K = self.B, self.K
        q = self._propose_request(
            num_drafts, committed_tokens, committed_positions, accepted_counts, page_table, hidden
        )
        # B > 1 still writes skipped rows' MTP KV (keep_kv): a row re-offered after a release has its full history
        if q.skip_all and B == 1:  # the runner withholds every row's drafts: no anchor/reseed/drafter work
            self.skipped_propose_calls += 1
            self._offered.clear()
            return DraftResult(torch.zeros(B, K, dtype=torch.int32), torch.zeros(B, dtype=torch.int32))
        drafts = self.propose(
            q.ctok, q.cpos, q.acc, q.page_table, decline=q.decline, cap_blocks=q.cap_blocks, keep_kv=q.keep
        )
        return self._draft_result(drafts, q.decline)

    def _propose_request(
        self, num_drafts, committed_tokens, committed_positions, accepted_counts, page_table, hidden, step=None
    ):
        """propose_draft_tokens' checked inputs and its decisions: committed block ``ctok`` and positions ``cpos``,
        counts ``acc`` (1 on a padding row), the table and its width, the padding rows, the declined rows (a padding row,
        a row whose window leaves the paged KV, a row whose drafts the runner withholds), the declined rows that keep
        their anchor slot's drafter KV, and whether the runner withholds every row's drafts. ``step`` (a PendingVerify):
        decide for that verify's rows, table and withheld slots rather than the last verify's (a chained verify ran
        after it)."""
        B, T, K = self.B, self.T, self.K
        if int(num_drafts) != K:
            raise ValueError(f"num_drafts {num_drafts} != engine K {K}")
        last_pt = self._last_pt if step is None else step.pt
        if page_table is None:
            if last_pt is None:
                raise ValueError("page_table is None and no verify has run")
            page_table, cap_blocks = last_pt, (self._last_cap_blocks if step is None else step.cap_blocks)
        else:
            cap_blocks = int(torch.as_tensor(page_table).shape[-1])
        if hidden is not None:
            raise ValueError("hidden must be None (the drafter's hidden handoff is on device)")
        ctok = _int_block(committed_tokens, B, T, "committed_tokens")
        cpos = _int_block(committed_positions, B, T, "committed_positions")
        acc = _int_vec(accepted_counts, B, "accepted_counts")
        if bool(((acc < 1) | (acc > T)).any()):
            raise ValueError(f"accepted_counts must be in [1, {T}], got {acc.tolist()}")
        row_active = self._row_active if step is None else step.active
        pad = [False] * B if row_active is None else [not bool(a) for a in row_active]
        acc = acc.clone()
        acc[torch.tensor(pad)] = 1  # a padding row's count is 1
        for u in range(B):
            if pad[u]:
                continue
            c = int(acc[u])
            if bool((ctok[u, :c] < 0).any()):
                raise ValueError(f"row {u}: negative committed token in columns 0..{c - 1}")
            if not torch.equal(cpos[u, :c], cpos[u, 0] + torch.arange(c)) or int(cpos[u, 0]) < 1:
                raise ValueError(f"row {u}: committed positions {cpos[u, :c].tolist()} are not consecutive from >= 1")
        _, _, _, p = accepted_plan(ctok, cpos, acc, B, T)
        fit = draft_rows_fit(p, K, table_capacity(cap_blocks, self.block_size))
        skip, skip_all = propose_skip_plan(self._no_draft if step is None else step.no_draft, B)
        keep = [s and not pd for s, pd in zip(skip, pad)]
        decline = [(not f) or pd or s for f, pd, s in zip(fit, pad, skip)]
        return SimpleNamespace(
            ctok=ctok,
            cpos=cpos,
            acc=acc,
            page_table=page_table,
            cap_blocks=cap_blocks,
            pad=pad,
            decline=decline,
            keep=keep,
            skip_all=skip_all,
        )

    def _draft_result(self, drafts, decline, offer=True):
        """DraftResult of a propose: K drafts on every row that is not declined; those rows are offered. ``offer``
        False: a chained verify already ran these drafts, and its own verify noted the offer."""
        num_valid = torch.tensor([0 if d else self.K for d in decline], dtype=torch.int32)
        if offer:
            self._offered = {u for u in range(self.B) if not decline[u]}
        return DraftResult(drafts, num_valid)

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
                h, tok_tt, [pu + k for pu in p], self._mtp_pt, rope_deltas=self._rope_deltas
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
        steps = self._read_ids(owned_tok[1:])  # steps[k][u]
        for t in owned_tok:
            ttnn.deallocate(t)
        return self._drafts_to_host(steps)

    # --------------------------------------------------------------------- #
    # Recorded drafter (anchor gather + batched reseed + K draft steps as one trace)
    # --------------------------------------------------------------------- #
    def _alloc_drafter_inputs(self):
        """Persistent drafter-trace inputs: the zero pack (a padding-only reseed and position-0 drafts), the reseed table
        on the scratch block, the anchor one-hot's column numbers, and the RoPE tables the trace looks up."""
        B, K, R, nb = self.B, self.K, self.B * self.T, self.nb
        al, rm = self._alloc_host_replicated, ttnn.ROW_MAJOR_LAYOUT
        self.model.rope_rm_tables()
        self._dr_pack = al(torch.zeros(drafter_pack_rows(K), pack_width(R), dtype=torch.int32), ttnn.uint32, rm)
        self._dr_rs_pt = al(torch.full((R, nb), self._scratch_block, dtype=torch.int32), ttnn.int32, rm)
        self._dr_rs_staged = None
        iota = torch.arange(1, R + 1, dtype=torch.float32).reshape(1, R)
        self._dr_iota = al(iota, ttnn.float32, ttnn.TILE_LAYOUT)
        self._dr_bufs = [self._dr_pack, self._dr_rs_pt, self._dr_iota]

    def _drafter_inputs(self):
        """The drafter trace's per-step inputs, derived inside the trace from _dr_pack (spec_step layout): tokens as
        [1, n] uint32 slices, KV positions as [n] int32, RoPE cos and sin looked up by table row, and the anchor one-hot
        [1,1,B,B*T] bf16, 1.0 at (u, u*T + m[u]). Returns (inputs, owned): every tensor in ``owned`` is freed after the
        body.

        The one-hot compares column number + 1 (_dr_iota, zero in the tile padding) with anchor + 1 >= 1, so its padding
        columns are 0 and the anchor matmul stays exact when B*T < 32 and the feed's padding rows hold any value."""
        B, K, R, pk, model = self.B, self.K, self.B * self.T, self._dr_pack, self.model
        owned = []

        def seg(row, n):
            t = ttnn.slice(pk, (row, 0), (row + 1, n))
            owned.append(t)
            return t

        def positions(row, n):
            t = ttnn.reshape(ttnn.typecast(seg(row, n), ttnn.int32), (n,))
            owned.append(t)
            return t

        def rope(row, n):
            cos, sin = model.rope_lookup(seg(row, n), n)
            owned.extend((cos, sin))
            return cos, sin

        tiled = ttnn.to_layout(seg(draft_anchor(K), B), ttnn.TILE_LAYOUT)
        anchor = ttnn.typecast(tiled, ttnn.float32)  # [1, B]
        anchor_t = ttnn.transpose(anchor, 0, 1)
        column = ttnn.add(anchor_t, 1.0)  # [B, 1]
        onehot = ttnn.eq(self._dr_iota, column, dtype=ttnn.bfloat16)  # [B, B*T]
        anchor_sel = ttnn.reshape(onehot, (1, 1, B, R))
        owned.extend((tiled, anchor, anchor_t, column, onehot, anchor_sel))
        rs_cos, rs_sin = rope(draft_reseed_rope(K), R)
        steps = [rope(draft_step_rope(K, k), B) for k in range(K)]
        inputs = dict(
            anchor_sel=anchor_sel,
            rs_tok=seg(draft_reseed_tokens(K), R),
            rs_pos=positions(draft_reseed_positions(K), R),
            rs_cos=rs_cos,
            rs_sin=rs_sin,
            tok0=seg(DRAFT_PENDING, B),
            pos=[positions(draft_step_row(k), B) for k in range(K)],
            cos=[c for c, _ in steps],
            sin=[s for _, s in steps],
        )
        return inputs, owned

    def _drafter_body(self, feed):
        """Anchor gather from ``feed`` [1,1,B*T,dim/tp], batched reseed, K chained draft steps, over the inputs
        _drafter_inputs derives from the persistent pack (device-input twin of _set_anchor + _reseed_mtp_batched +
        _draft). Returns the K ids tensors [1,1,B] uint32 (step k's drafts)."""
        B, mtp = self.B, self.mtp
        x, owned = self._drafter_inputs()
        Hp = ttnn.matmul(x["anchor_sel"], feed, compute_kernel_config=self._anchor_cc)
        ttnn.copy(Hp, self._hp_buf)
        ttnn.deallocate(Hp)
        _, hn = mtp.forward_decode(
            feed,
            x["rs_tok"],
            x["rs_pos"],
            x["rs_cos"],
            x["rs_sin"],
            self._dr_rs_pt,
            need_logits=False,
            alias_kv_write=True,
            spec_n_users=B,
        )
        free_tensors(hn)
        h, tok, outs = self._hp_buf, x["tok0"], []
        for k in range(self.K):
            logits, hn = mtp.forward_decode(h, tok, x["pos"][k], x["cos"][k], x["sin"][k], self._mtp_pt)
            idx = draft_argmax(mtp, logits)
            ttnn.deallocate(logits)
            tok = ttnn.reshape(idx, (B, 1))
            outs.append(idx)
            if h is not self._hp_buf:
                ttnn.deallocate(h)
            h = hn
        ttnn.deallocate(h)
        free_tensors(*owned)
        return outs

    def _draft_table(self, pt, decline):
        """The draft steps' MTP table: ``pt`` with every declined row on the scratch block."""
        if not any(decline):
            return pt
        dpt = pt.clone()
        dpt[torch.tensor([bool(d) for d in decline])] = self._scratch_block
        return dpt

    def _rope_base(self, p0, deltas=None):
        """Each user's RoPE table row of position ``p0[u]`` with its M-RoPE delta (spec_step.drafter_pack): the slots'
        current deltas, or ``deltas``."""
        return rope_rows(p0, self._rope_deltas if deltas is None else deltas, self.model.rope_table_rows())

    def _propose_recorded(self, ctok, m, p0, decline, keep, pt):
        """One copy of the drafter pack (spec_step.drafter_pack of the committed block ``ctok``, the accepted drafts
        ``m``, the verify column-0 positions ``p0``, ``decline`` and ``keep``), the reseed table when it changed, the
        replay, and one wait for the draft ids. _mtp_pt is staged by the caller. The reseed rows always run (padding-only
        when every m == 0). Returns int32 [B, K]."""
        T, K = self.T, self.K
        pack = drafter_pack(ctok, m, p0, decline, keep, self._rope_base(p0), K, self.model.rope_table_rows())
        self._stage(pack, self._dr_pack, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        real = reseed_real_rows(m, decline, keep, T)
        self._stage_table(reseed_table(pt, real, T, self._scratch_block), "_dr_rs_pt", "_dr_rs_staged")
        ttnn.execute_trace(self.mesh, self._dr_trace, cq_id=0, blocking=False)
        if any(m):
            self.mtp_extra_steps += 1
        return self._drafts_to_host(self._read_ids(self._dr_outs))

    def _drafts_to_host(self, steps):
        """steps[k][u] -> int32 [B, K]."""
        return torch.tensor([[steps[k][u] for k in range(self.K)] for u in range(self.B)], dtype=torch.int32)

    def _draft_warmup(self, pending, Hp, p):
        """One need_logits=True draft step before capture (compiles head norm, LM head, argmax, mesh_partition)."""
        logits, h = self.model.ttnn_mtp_decode_forward(Hp, [int(t) for t in pending], list(p), self._mtp_pt)
        idx = draft_argmax(self.mtp, logits)
        for t in (logits, idx, h):
            ttnn.deallocate(t)
        ttnn.synchronize_device(self.mesh)

    def _read_ids(self, id_tts):
        """[*, B] uint32 device ids of each tensor -> B ints per tensor (_read_device0)."""
        return [[int(v) for v in t.reshape(-1)[: self.B]] for t in self._read_device0(id_tts)]

    def _read_device0_async(self, tensors):
        """Each tensor's device-0 replica, read without blocking; the reads land once the returned DeviceReads' event
        has passed."""
        reads = [ttnn.from_device(ttnn.get_device_tensors(t)[0], blocking=False) for t in tensors]
        return DeviceReads(reads, ttnn.record_event(self.mesh, 0))

    def _read_device0(self, tensors):
        """Each tensor's device-0 replica as a torch tensor, with one wait: every read is enqueued without blocking,
        then one synchronize fills them all."""
        reads = [ttnn.from_device(ttnn.get_device_tensors(t)[0], blocking=False) for t in tensors]
        ttnn.synchronize_device(self.mesh)
        return [ttnn.to_torch(r) for r in reads]

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

    def _warm_mtp_last(self, last_hidden, first_tok, slot, page_table_tt, rope_delta=None):
        """Write the MTP KV of a user's slot Tp-1 from (its feed row, the first generated token); the
        first draft attends to it. ``rope_delta``: the user's M-RoPE delta (None: rope.rope_delta)."""
        _, h_next = self.model.ttnn_mtp_decode_forward(
            last_hidden,
            int(first_tok),
            slot,
            page_table_tt,
            need_logits=False,
            rope_deltas=None if rope_delta is None else [rope_delta],
        )
        free_tensors(h_next)

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
                    rope_deltas=[self._rope_deltas[u]],
                )
                ttnn.deallocate(row)
                free_tensors(h_next)
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
        cos_t, sin_t = self.model._rope_tp_cos_sin_decode_torch(pos, self._row_rope_deltas(T))
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
        free_tensors(tok_tt, pos_tt, pt_tt, cos, sin, h_next)
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


class MTPBlockEngine(MTPSpecEngine):
    """MTPSpecEngine for a model whose GDN state is in vLLM KV cache blocks (``model.gdn_blocks`` with verify tables
    for T = K + 1 tokens): Qwen36MTPContractForCausalLM.

    The engine keeps no GDN state. The model's prefill (Qwen36ForCausalLM.prefill_forward) writes each row's end state
    into its destination block and calls ``prefill_hook`` per prompt chunk. A verify reads every row's start state from
    mamba_state.start_blocks and writes the state after each query token into mamba_state.windows
    (GdnBlockState.set_verify_blocks), so a row that moves, leaves a step or resumes needs nothing here, and the
    accepted count selects nothing: the runner points the next step's start_blocks at the accepted position's block.
    What remains per state slot is the drafter's: the feed row of a prompt's last token (``_last_hidden``), kept from
    the final prefill chunk to the row's first verify, which writes that position's drafter KV with the first
    generated token, and the M-RoPE delta of the slot's request (``_rope_deltas``, 0 for text), which offsets the
    RoPE positions of the slot's verify rows and drafter rows.

    One-enqueue steps: an argmax_ids verify (greedy rows, and device-sampled rows whose ids are draws) whose next propose
    is decided before the verify (no row whose drafting depends on its count) enqueues the verify trace, the sampler
    trace, the step trace (spec_accept.StepAccept: the greedy walk and the drafter's inputs on the device) and the
    drafter trace, then reads the ids, the drafter pack and the drafts with one wait. The next propose_draft_tokens
    checks that the runner committed what the device accepted and returns those drafts."""

    def __init__(self, model, batch_size, draft_len):
        super().__init__(model, batch_size, draft_len)
        blocks = model.gdn_blocks
        assert blocks is not None and blocks.verify_tokens == self.T, (
            f"MTPBlockEngine needs the GDN verify tables for T = {self.T} (GdnBlockState(verify_tokens=K + 1)), "
            f"got {None if blocks is None else blocks.verify_tokens}"
        )
        self._accept = None  # spec_accept.StepAccept, once prepared
        self._acc_trace = None  # the step trace
        self._acc_pt_staged = None  # host table rows the step trace's pt_rows holds, None when unknown
        self._pending = deque()  # PendingVerify of each verify whose propose_draft_tokens has not run, oldest first
        self.one_enqueue_steps = 0  # verifies that ran the device acceptance and the drafter
        self.verify_steps = 0  # every verify after capture
        self.one_enqueue_host_proposes = 0  # one-enqueue verifies whose propose drafted again on the host
        self._chain = None  # spec_chain.ChainInputs, once prepared
        self._chain_trace = None  # the chain program's trace
        self.chained_steps = 0  # chained verifies whose inputs the chain program built
        self.materialized_steps = 0  # chained verifies the engine built from a read predecessor instead

    def prepare_prefill(self, page_table):
        """Allocate the engine buffers and compile the drafter's prompt programs (the drafter KV warm and the last-row
        capture of every masked bucket and of a full 2048-token chunk) on the eager prefill path of the batch-1 GDN
        workspace. The model's own prefill programs and chunk trace are prepared by the caller. Idempotent per table
        width."""
        model, T = self.model, self.T
        pt0 = torch.as_tensor(page_table)
        width = int(pt0.shape[-1])
        if self.prefill_prepared:
            assert width == self.nb, f"engine prepared for nb={self.nb}, got a table of width {width}"
            return
        t0 = time.perf_counter()
        assert not self.captured and getattr(model, "_vfy_trace_id", None) is None, "prepare after capture"
        pt = self._alloc_engine_buffers(pt0)
        cap = width * self.block_size
        warm_lens = [L for L in _WARM_PROMPT_LENS if L + T + 1 <= cap]
        for L in warm_lens:
            prompt = torch.ones(1, L, dtype=torch.int32)
            logits = model._prefill_for_spec_b1(prompt, pt[0:1], L, self.prefill_hook(0, prompt[0], pt[0:1], True))
            if not isinstance(logits, torch.Tensor):
                ttnn.deallocate(logits)
        self.release(0)
        self.prefill_prepared = True
        logger.info(
            f"[spec_engine] prefill prepared (GDN state in KV cache blocks) B={self.B} K={self.K} nb={self.nb} "
            f"block={self.block_size} warm_prompts={warm_lens} in {time.perf_counter() - t0:.1f}s"
        )

    def record_prefill(self):
        """The chunk trace belongs to the model's prefill, which records it."""
        assert self.prefill_prepared, "record_prefill before prepare_prefill"
        self.prefill_recorded = True

    def _prepare_verify(self):
        self.model.prepare_verify_trace(
            self._prep_pt, self.T, warm_positions=[0] * self.B, sample=self.sample_on_device
        )

    def prepare_decode(self):
        """The base prepare, then the step trace's buffers and programs (compiled on a stand-in for the verify's ids,
        before any capture). One-enqueue steps need the recorded drafter and the verify's ids as the fused argmax's
        lanes; without either, every step stages the drafter from the host."""
        prepared = self.prepared
        super().prepare_decode()
        if prepared:
            return
        model = self.model
        if not (self.recorded_drafter and getattr(model, "_vfy_ids_lanes", False)):
            logger.info(
                "[spec_engine] one-enqueue steps off: they need the recorded drafter and the fused verify argmax"
            )
            return
        sampler = getattr(model, "_vfy_sampler", None)
        self._accept = StepAccept(
            self.mesh,
            self.B,
            self.K,
            self.nb,
            model.rope_table_rows(),
            self._scratch_block,
            model._vfy_step_buf,
            self._dr_pack,
            self._dr_rs_pt,
            draws=None if sampler is None else sampler.draws,
        )
        self._accept.body()
        ttnn.synchronize_device(self.mesh)
        self._dr_rs_staged = None  # the warm pass wrote the reseed table
        blocks = model.gdn_blocks
        self._chain = ChainInputs(
            self.mesh,
            self.B,
            self.K,
            model.rope_table_rows(),
            model._vfy_step_buf,
            self._dr_pack,
            blocks._verify_ctrl,
            blocks.dims.nv,
        )
        self._chain.body()
        ttnn.synchronize_device(self.mesh)
        blocks.clear_verify_blocks()  # the warm pass wrote the ctrl pages

    def _capture_step(self):
        """The step trace, over the verify trace's ids output (which exists from the verify capture on)."""
        if self._accept is None or self._dr_trace is None:
            return
        self._accept.bind_ids(self.model._vfy_ids_out)
        self._acc_trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        self._accept.body()
        ttnn.end_trace_capture(self.mesh, self._acc_trace, cq_id=0)
        ttnn.synchronize_device(self.mesh)
        logger.info("[spec_engine] step trace captured: one-enqueue steps on (verify, acceptance, drafter, one read)")
        if self._chain is not None:
            self._chain.bind(self.model._vfy_ids_out, self._dr_outs)
            self._chain_trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            self._chain.body()
            ttnn.end_trace_capture(self.mesh, self._chain_trace, cq_id=0)
            ttnn.synchronize_device(self.mesh)
            logger.info("[spec_engine] chain trace captured: chained verifies on")

    def _one_enqueue_ready(self):
        return self._acc_trace is not None and not self._timing

    def _log_one_enqueue_counts(self):
        logger.info(
            f"[spec_engine] one-enqueue steps: {self.one_enqueue_steps} of {self.verify_steps} verifies; "
            f"{self.one_enqueue_host_proposes} of them drafted again on the host; chained verifies: "
            f"{self.chained_steps} built on the device, {self.materialized_steps} from a read predecessor"
        )

    def shutdown(self):
        self._log_one_enqueue_counts()
        if self._acc_trace is not None:
            ttnn.release_trace(self.mesh, self._acc_trace)
            self._acc_trace = None
        if self._accept is not None:
            self._accept.release()
            self._accept = None
        if self._chain_trace is not None:
            ttnn.release_trace(self.mesh, self._chain_trace)
            self._chain_trace = None
        if self._chain is not None:
            self._chain.release()
            self._chain = None
        self._acc_pt_staged = None
        self._pending.clear()
        super().shutdown()

    def capture_decode(self):
        """The base capture, then zero the null block 0: the warm passes wrote GDN state there (every row of a warm
        verify points at it). Serving never writes GDN state into it (verify_tables holds padding rows), and a verify's
        padded columns past a row's last real block read and write block 0, where attention must find finite values."""
        captured = self.captured
        super().capture_decode()
        if not captured:
            self.model.gdn_blocks.zero_blocks([0])

    def _remap_gdn(self, idx):
        """The GDN state follows mamba_state, which already names each row's blocks after the move."""

    def prefill_hook(self, slot, row_tokens, page_table_row, final):
        """on_chunk of Qwen36Model.prefill_traced_chunked for a prefill row of state slot ``slot`` whose tokens
        [0, end) are ``row_tokens``. Writes the drafter KV of the chunk's positions: position i pairs the hidden at i
        with token i + 1, so position end - 1 stays for the next chunk's first token or, after the prompt's last chunk
        (``final``), for the first generated token at the row's first verify, from the feed row kept here. A next chunk
        does not rewrite position end - 1 (its hidden is not kept), nor does a prefix-cache hit at end; the drafter
        then attends to whatever that position's drafter KV holds, which lowers acceptance and never changes the
        verified tokens. The prompt's last chunk also keeps the request's M-RoPE delta for the slot."""
        model = self.model
        pt_row = model._fit_spec_page_table(page_table_row, self.nb)
        self._check_scratch(pt_row)
        prompt = [int(t) for t in torch.as_tensor(row_tokens).reshape(-1)]
        end = len(prompt)

        def on_chunk(hidden, chunk_start, valid_len):
            # Never frees `hidden` (the caller does, or it is the trace's persistent output).
            feed = model.spec_feed_rows(hidden)
            self._warm_mtp_chunk(feed, chunk_start, valid_len, prompt, pt_row)
            if final and chunk_start + valid_len >= end:
                self._capture_last_row(feed, end - 1 - chunk_start, slot)
                # The prefill staged this row's RoPE table before its first chunk (Qwen36Model._build_request_rope).
                self._rope_deltas[slot] = int(model.rope.rope_delta)
            if feed is not hidden:
                ttnn.deallocate(feed)

        return on_chunk

    def finish_prefill(self, slot, end, final):
        """Prefill row of state slot ``slot`` wrote its state after ``end`` tokens. Only a row that ends its prompt
        (``final``) decodes next."""
        reset_draft_skip(self, slot)
        self._fresh.pop(slot, None)
        self._active.discard(slot)
        if final:
            self._fresh[slot] = int(end)
            self._active.add(slot)

    def verify(
        self,
        tokens,
        start_pos,
        accepted_counts,
        page_table,
        read_logits=False,
        active=None,
        num_valid_drafts=None,
        mamba_state=None,
        sample_plan=None,
        one_enqueue=False,
        defer=False,
        keep_pending=False,
    ):
        """Replay the verify trace over [B, T] tokens at start_pos[u] + j, every GDN layer on the blocks of
        ``mamba_state``. A row's first verify after its prompt first writes the drafter KV of the prompt's last
        position. ``accepted_counts`` selects no state. Returns (ids int32 [B, T], logits [B, T, vocab] or None).
        ``sample_plan`` (spec_sample.sampling_plan): each row it samples returns the device sampler's draws instead of
        its argmax ids. ``one_enqueue`` (argmax_ids only): run the step as a one-enqueue step when its next propose is
        decided before the verify (_one_enqueue_plan). ``defer`` (argmax_ids only): enqueue the reads without waiting
        and return a PendingVerify in place of the ids. ``keep_pending``: the verify continues an unread one whose
        propose is still to come (_materialize)."""
        assert self.captured, "verify before capture"
        if mamba_state is None:
            raise ValueError("a verify on GDN state blocks needs mamba_state")
        # Each propose_draft_tokens follows its own verify; one never run leaves nothing to return.
        if not keep_pending:
            self._pending.clear()
        if self.verify_steps % ONE_ENQUEUE_LOG_EVERY == 0:  # a server's engine is killed without a shutdown
            self._log_one_enqueue_counts()
        self.verify_steps += 1
        B, T = self.B, self.T
        tokens = self._ints(tokens, (B, T))
        start_pos = self._ints(start_pos, (B,))
        self._last_cap_blocks = int(torch.as_tensor(page_table).shape[-1])
        pt = self._fit(page_table)
        active = (
            torch.ones(B, dtype=torch.bool) if active is None else torch.as_tensor(active, dtype=torch.bool).reshape(B)
        )
        unknown = [u for u in range(B) if active[u] and u not in self._active]
        if unknown:
            raise ValueError(f"rows {unknown} carry a request whose prompt was never prefilled into that slot")
        if not bool(active.all()):
            pt = pt.clone()
            pt[~active] = 0  # vLLM null block
        self._row_active = active.clone()
        self._last_pt = pt
        # padded columns may sit past the runner's KV reservation (max_model_len, ngram)
        vpt = pt if num_valid_drafts is None else verify_write_table(pt, start_pos, num_valid_drafts, self.block_size)
        assert int(start_pos.max()) + T <= table_capacity(
            self._last_cap_blocks, self.block_size
        ), "verify window past the paged KV capacity"
        fresh = {u: n for u, n in self._fresh.items() if active[u]}
        if fresh:
            self._stage_mtp_tables(pt)
            for u, n in sorted(fresh.items()):
                if int(start_pos[u]) != n:
                    raise ValueError(f"slot {u}: first verify at {int(start_pos[u])}, its prompt has {n} tokens")
                self._warm_mtp_last(
                    self._last_hidden[u], int(tokens[u, 0]), n - 1, self._mtp_pt_rows[u], self._rope_deltas[u]
                )
                del self._fresh[u]
        self.model.gdn_blocks.set_verify_blocks(mamba_state)
        args = (tokens.tolist(), start_pos.tolist(), [0] * B)
        kw = dict(
            read_logits=read_logits,
            page_tables=vpt,
            inactive=(~active).tolist(),
            rope_deltas=list(self._rope_deltas),
        )
        if sample_plan is not None:
            kw["sample_plan"] = sample_plan
        plan = None
        if one_enqueue and not read_logits and num_valid_drafts is not None:
            plan = self._one_enqueue_plan(start_pos, active, num_valid_drafts, sample_plan)
        nv = torch.zeros(B, dtype=torch.int64) if num_valid_drafts is None else num_valid_drafts
        step = PendingVerify(
            B,
            self.K,
            nv,
            active,
            pt,
            self._last_cap_blocks,
            self._no_draft,
            plan=plan,
            sample_plan=sample_plan,
            rope_deltas=self._rope_deltas,
        )
        step.tokens, step.p0 = tokens.clone(), torch.where(active, start_pos, 0)
        self._pending.append(step)
        if plan is not None:
            self._verify_one_enqueue(args, kw, step)
            return (step if defer else step.result()), None
        if defer and not read_logits and not self._timing:
            self.model.verify_enqueue(*args, **kw)
            draws = [] if sample_plan is None else [self.model._vfy_sampler.draws]
            step.reads = self._read_device0_async([self.model._vfy_ids_out, *draws])
            self._vfeed = self.model._vfy_rows_out
            return step, None
        if self._timing:
            ids, vfeed, lt = self._verify_split(*args, **kw)
        else:
            ids, vfeed, lt = self.model.verify_traced(*args, **kw)
        self._vfeed = vfeed
        ids = torch.tensor(ids, dtype=torch.int32).reshape(B, T)
        return ids, (None if lt is None else lt.reshape(B, T, -1))

    def _one_enqueue_plan(self, start_pos, active, num_valid_drafts, sample_plan, no_draft=None):
        """The acceptance rows of a one-enqueue step (spec_step STEP_*) and the propose's decisions it fixes, or None
        when the step must stage its drafter from the host. Decided as propose_draft_tokens decides after the verify,
        for every count the row can accept: a padding row, a row whose drafts the runner withholds and a row whose
        window leaves the paged KV at every count decline; a row that fits at some counts only, a propose that declines
        every row, a B == 1 propose the runner withholds or one that could take the eager reseed run on the host."""
        B, K, T = self.B, self.K, self.T
        nv = [int(n) for n in torch.as_tensor(num_valid_drafts).reshape(B).tolist()]
        act = [bool(a) for a in active.tolist()]
        p0 = [int(start_pos[u]) if act[u] else 0 for u in range(B)]
        capacity = table_capacity(self._last_cap_blocks, self.block_size)
        skip, skip_all = propose_skip_plan(self._no_draft if no_draft is None else no_draft, B)
        if skip_all and B == 1:
            return None
        declined, keep = [], []
        for u in range(B):
            fits = draft_rows_fit([p0[u] + m for m in range(nv[u] + 1)], K, capacity)
            if act[u] and not skip[u] and any(fits) and not all(fits):
                return None
            declined.append(not act[u] or skip[u] or not all(fits))
            keep.append(skip[u] and act[u])
        if all(declined) or (B == 1 and p0[0] + nv[0] > self.eager_reseed_len):
            return None
        self._check_scratch(self._last_pt)
        drawn = [bool(sample_plan is not None and sample_plan.sampled[u * T]) for u in range(B)]
        rows = {
            STEP_NUM_VALID: nv,
            STEP_DRAWN: [int(d) for d in drawn],
            STEP_P0: p0,
            STEP_ROPE_BASE: self._rope_base(p0).tolist(),
            STEP_DECLINED: [int(d) for d in declined],
            STEP_KEEP: [int(k) for k in keep],
        }
        return SimpleNamespace(rows=rows, p0=p0, declined=declined, keep=keep)

    def _verify_one_enqueue(self, args, kw, step):
        """Stage, enqueue the verify, [sampler], step and drafter traces, and enqueue the reads of the ids, the drafter
        pack and the drafts into ``step`` (a PendingVerify), which the caller waits on or defers; the drafts wait for
        the next propose_draft_tokens."""
        model, plan, T = self.model, step.plan, self.T
        self._stage_table(self._draft_table(self._last_pt, plan.declined), "_mtp_pt", "_mtp_pt_staged")
        pt_rows = self._last_pt.repeat_interleave(T, dim=0)
        if self._acc_pt_staged is None or not torch.equal(self._acc_pt_staged, pt_rows):
            self._accept.stage_pt_rows(pt_rows)
            self._acc_pt_staged = pt_rows
        kw = {k: v for k, v in kw.items() if k != "read_logits"}
        model.verify_enqueue(*args, accept_rows=plan.rows, **kw)
        self._enqueue_step_and_drafter(step)

    def _enqueue_step_and_drafter(self, step):
        """Replay the step and drafter traces after a one-enqueue verify and enqueue its reads."""
        model = self.model
        ttnn.execute_trace(self.mesh, self._acc_trace, cq_id=0, blocking=False)
        ttnn.execute_trace(self.mesh, self._dr_trace, cq_id=0, blocking=False)
        self._dr_rs_staged = None  # the step trace wrote the reseed table
        draws = [] if step.sample_plan is None else [model._vfy_sampler.draws]
        step.reads = self._read_device0_async([model._vfy_ids_out, *draws, self._dr_pack, *self._dr_outs])
        self._vfeed = model._vfy_rows_out
        self.one_enqueue_steps += 1

    def propose_draft_tokens(
        self, num_drafts, committed_tokens, committed_positions, accepted_counts, page_table=None, hidden=None
    ):
        """After a one-enqueue verify: the drafts the device made, once the runner's commit is the one the device
        accepted (every live row's count and committed tokens give the drafter pack the device built).

        A runner that commits less than its walk on some row (a prefix of the device's commit: a length or stop cut,
        which vllm-tt-plugin leaves to the scheduler) gets the base propose instead, as does a propose that decides
        differently from the verify's plan (another table, other declined rows, positions that do not follow the
        verify). The base propose's drafter rewrites every drafter KV slot the device's wrote up to the runner's
        commit plus K - 1; the slots past it are written again before any read. Any other commit raises ValueError: it
        is not a prefix of the device's greedy walk over the same ids, so the device acceptance is wrong.

        After a verify a chained verify continued (``successor``), the drafts already ran and the chained verify's
        inputs followed the device's count: every difference raises, and the propose fills the chained verify's
        block and positions for its own propose."""
        step = self._pending.popleft() if self._pending else None
        args = (num_drafts, committed_tokens, committed_positions, accepted_counts, page_table, hidden)
        if step is not None and step.materialized is not None:
            return self._propose_materialized(step, args)
        if step is None or step.plan is None:
            if step is not None and step.successor is not None:
                raise RuntimeError("a chained verify continued a verify that ran no device drafter")
            return super().propose_draft_tokens(*args)
        step.wait()
        chained = step.successor is not None
        # A verify continued by a chained one decides with its own rows and withheld slots: the chained verify ran
        # after it and noted its own.
        q = self._propose_request(*args, step=step if chained else None)
        plan, B, K = step.plan, self.B, self.K
        moved = [u for u in range(B) if not q.pad[u] and int(q.cpos[u, 0]) != plan.p0[u] + 1]
        other_table = page_table is not None and not torch.equal(self._fit(page_table), step.pt)
        if other_table or moved or q.decline != plan.declined or q.keep != plan.keep:
            if chained:
                raise ValueError(
                    f"the propose after a verify a chained verify continued decides differently (table changed "
                    f"{other_table}, rows off their verify {moved}, declined {q.decline} vs {plan.declined})"
                )
            logger.warning(
                f"[spec_engine] propose after a one-enqueue verify decides differently (table changed {other_table}, "
                f"rows off their verify {moved}, declined {q.decline} vs {plan.declined}); drafting on the host path"
            )
            self.one_enqueue_host_proposes += 1
            return super().propose_draft_tokens(*args)
        m = (q.acc - 1).tolist()
        m_dev = (step.pack[draft_anchor(K), :B] - torch.arange(B) * self.T).tolist()
        live = [u for u in range(B) if not q.pad[u]]
        cut = [u for u in live if m[u] < m_dev[u]]
        prefix = all(m[u] <= m_dev[u] and torch.equal(q.ctok[u, : m[u] + 1], step.ids[u, : m[u] + 1]) for u in live)
        if cut and prefix and not chained:
            self.one_enqueue_host_proposes += 1
            return super().propose_draft_tokens(*args)
        base = self._rope_base(plan.p0, step.rope_deltas)
        want = drafter_pack(q.ctok, m, plan.p0, plan.declined, plan.keep, base, K, self.model.rope_table_rows()).to(
            torch.int64
        )
        if (cut and chained) or not torch.equal(want, step.pack[: want.shape[0], : want.shape[1]]):
            raise ValueError(
                f"the runner committed {[mu + 1 for mu in m]} tokens per row, the device acceptance "
                f"{[mu + 1 for mu in m_dev]}, and the runner's commit is "
                + (
                    "not the device's: the chained verify and the one-enqueue drafts continue the device's commit"
                    if chained
                    else "not a prefix of the device's: the one-enqueue drafts continue a wrong commit"
                )
            )
        if any(m):
            self.mtp_extra_steps += 1
        drafts = step.drafts.clone()
        drafts[torch.tensor(plan.declined)] = 0
        if chained:
            self._fill_successor(step.successor, q.ctok, m, plan.p0, drafts)
        return self._draft_result(drafts, plan.declined, offer=not chained)

    def _fill_successor(self, succ, ctok, m, p0, drafts):
        """The block a chained verify ran, once its predecessor's count is known: each live row's last committed token,
        then its valid drafts (a padded column repeats column 0, as sanitize_verify_block pads), at the column-0
        position p0 + m + 1. Its own propose compares against these."""
        B, T = self.B, self.T
        tokens = torch.zeros(B, T, dtype=torch.int64)
        start = torch.zeros(B, dtype=torch.int64)
        for u in range(B):
            if not bool(succ.active[u]):
                continue
            n = int(succ.num_valid[u])
            tokens[u] = int(ctok[u, m[u]])
            tokens[u, 1 : n + 1] = drafts[u, :n].to(torch.int64)
            start[u] = int(p0[u]) + m[u] + 1
        succ.tokens, succ.p0 = tokens, start
        succ.plan.p0 = start.tolist()

    def _propose_materialized(self, step, args):
        """The propose after a verify whose chained successor the engine built from its own read of it (_materialize):
        the drafts it computed then, once the runner's commit is the one the engine walked."""
        ctok, acc, result = step.materialized
        q = self._propose_request(*args, step=step)
        for u in range(self.B):
            if q.pad[u]:
                continue
            c = int(acc[u])
            if int(q.acc[u]) != c or not torch.equal(q.ctok[u, :c], ctok[u, :c]):
                raise ValueError(
                    f"row {u}: the runner committed {q.ctok[u, : int(q.acc[u])].tolist()}, the engine's walk of the "
                    f"verify a chained verify continued {ctok[u, :c].tolist()}"
                )
        return result

    # --------------------------------------------------------------------- #
    # Chained verify (vllm-tt-plugin supports_chained_spec_verify)
    # --------------------------------------------------------------------- #
    def _decode_chained(self, predecessor, start_pos, num_valid_drafts, page_table, slot_remap, mamba_state, sampling):
        """A verify that continues the unread ``predecessor`` row for row: each live row of the block (placeholders
        from the runner) is the predecessor's last committed token at the device's count, then up to
        ``num_valid_drafts`` of the drafts its drafter made, from the start state of that count
        (``mamba_state.groups[g].start_blocks_by_count``). Built on the device by the chain program when the
        predecessor ran the device acceptance and drafter and this verify's propose is decided for every count; else
        from the engine's own read and walk of the predecessor (_materialize). Returns VerifyResult with a
        PendingVerify."""
        B, K, T = self.B, self.K, self.T
        if not self._pending or predecessor is not self._pending[-1] or predecessor.successor is not None:
            raise ValueError("spec_predecessor is not the engine's last verify still to propose")
        if predecessor.p0 is None:
            raise ValueError("the predecessor's block is unknown: the verify before it was never proposed")
        start_pos = self._ints(start_pos, (B, T))
        hint = _int_vec(num_valid_drafts, B, "num_valid_drafts")
        active = start_pos[:, 0] >= 0
        if bool((active & ~predecessor.active).any()):
            raise ValueError(
                f"chained rows {active.tolist()} are not rows of the predecessor {predecessor.active.tolist()}"
            )
        if slot_remap is not None and [int(x) for x in slot_remap] != list(range(B)):
            raise ValueError(f"a chained verify moves no row, got slot_remap {list(slot_remap)}")
        if sampling is not None:
            plan = sampling_plan(sampling, torch.where(active, start_pos[:, 0], -1).tolist(), T)
            if plan.any_sampled:
                raise ValueError("a chained verify serves greedy rows only")
        if mamba_state is None:
            raise ValueError("a verify on GDN state blocks needs mamba_state")
        reason = self._chain_refusal(predecessor, active, hint, start_pos, page_table)
        if reason is None:
            return VerifyResult(
                "argmax_ids", self._verify_chained(predecessor, active, hint, start_pos, page_table, mamba_state)
            )
        logger.debug(f"[spec_engine] chained verify built from a read predecessor: {reason}")
        self.materialized_steps += 1
        return self._materialize(predecessor, active, hint, page_table, mamba_state, sampling)

    def _chain_nv(self, predecessor, active, hint):
        """Valid drafts of each chained row: the runner's cap, 0 where the predecessor's propose declines the row."""
        return [
            0 if not bool(active[u]) or predecessor.plan.declined[u] else min(int(hint[u]), self.K)
            for u in range(self.B)
        ]

    def _chain_refusal(self, predecessor, active, hint, start_pos, page_table):
        """Why the chain program cannot build this chained verify, or None. It needs its trace, a predecessor that ran
        the device acceptance and drafter with no draws, no prompt to finish, a propose decided alike for every count
        the predecessor can accept, and a grouped KV write free of collisions at every such count. Touches no engine
        state."""
        B, T = self.B, self.T
        if self._chain_trace is None or not self._one_enqueue_ready():
            return "no chain trace"
        if predecessor.plan is None or predecessor.sample_plan is not None:
            return "the predecessor ran no device acceptance and drafter of greedy ids"
        if any(bool(active[u]) and u in self._fresh for u in range(B)):
            return "a row's prompt is not finished"
        offered = {u for u in range(B) if not predecessor.plan.declined[u]}
        nv = self._chain_nv(predecessor, active, hint)
        no_draft = note_withheld_drafts(
            offered, self._no_draft, set(self._speculated), dict(self._withheld_streak), nv, self.no_draft_streak
        )
        saved = self._last_cap_blocks, self._last_pt
        self._last_cap_blocks = int(torch.as_tensor(page_table).shape[-1])
        self._last_pt = self._fit(page_table)
        try:
            plans = []
            for c in range(1, T + 1):
                p0 = torch.tensor([int(predecessor.p0[u]) + c if bool(active[u]) else 0 for u in range(B)])
                plans.append(self._one_enqueue_plan(p0, active, nv, None, no_draft=no_draft))
        finally:
            self._last_cap_blocks, self._last_pt = saved
        if any(p is None for p in plans) or any(
            (p.declined, p.keep) != (plans[0].declined, plans[0].keep) for p in plans
        ):
            return "the propose depends on the count"
        if not self._chain_kv_safe(self._fit(page_table), predecessor, active):
            return "the grouped KV write collides at some count"
        return None

    def _chain_kv_safe(self, pt, predecessor, active):
        """Whether no two users write one (block, 32-row tile) in one grouped KV write call at any count each can take
        (spec_kv_write_conflicts over the union of the counts)."""
        if not getattr(self.model, "_vfy_kv_grouped", False):
            return True
        bs, T = self.block_size, self.T
        for j in range(T):
            owner = {}
            for u in range(self.B):
                if not bool(active[u]):
                    continue
                for c in range(1, T + 1):
                    pos = int(predecessor.p0[u]) + c + j
                    blk = int(pt[u, pos // bs]) if pos // bs < pt.shape[1] else 0
                    if blk and owner.setdefault((blk, (pos % bs) // ttnn.TILE_SIZE), u) != u:
                        return False
        return True

    def _verify_chained(self, predecessor, active, hint, start_pos, page_table, mamba_state):
        """Stage the host rows and the ctrl candidates, replay the chain trace, then the verify, step and drafter
        traces, and enqueue the reads. The predecessor's propose fills the block's tokens and positions."""
        B, T, model = self.B, self.T, self.model
        if self.verify_steps % ONE_ENQUEUE_LOG_EVERY == 0:
            self._log_one_enqueue_counts()
        self.verify_steps += 1
        nv = self._chain_nv(predecessor, active, hint)
        self._offered = {u for u in range(B) if not predecessor.plan.declined[u]}
        self._no_draft = note_withheld_drafts(
            self._offered, self._no_draft, self._speculated, self._withheld_streak, nv, self.no_draft_streak
        )
        self._last_cap_blocks = int(torch.as_tensor(page_table).shape[-1])
        pt = self._fit(page_table).clone()
        pt[~active] = 0
        self._row_active, self._last_pt = active.clone(), pt
        upper = torch.where(active, start_pos[:, 0], 0)
        vpt = verify_write_table(pt, upper, nv, self.block_size)
        if int(upper.max()) + T > table_capacity(self._last_cap_blocks, self.block_size):
            raise ValueError(f"chained verify window from {upper.tolist()} past the paged KV capacity")
        p_any = torch.tensor([int(predecessor.p0[u]) + T if bool(active[u]) else 0 for u in range(B)])
        plan = self._one_enqueue_plan(p_any, active, nv, None)
        plan = SimpleNamespace(rows=None, p0=None, declined=plan.declined, keep=plan.keep)
        blocks = model.gdn_blocks
        blocks.set_verify_blocks(mamba_state, stage_ctrl=False)
        p0_prev = [int(predecessor.p0[u]) if bool(active[u]) else 0 for u in range(B)]
        overlay = chain_overlay(p0_prev, active.tolist(), list(self._rope_deltas), nv, plan.declined, plan.keep)
        self._chain.stage(overlay, blocks.ctrl_candidates(mamba_state))
        self._stage_table(self._draft_table(pt, plan.declined), "_mtp_pt", "_mtp_pt_staged")
        pt_rows = pt.repeat_interleave(T, dim=0)
        if self._acc_pt_staged is None or not torch.equal(self._acc_pt_staged, pt_rows):
            self._accept.stage_pt_rows(pt_rows)
            self._acc_pt_staged = pt_rows
        ttnn.execute_trace(self.mesh, self._chain_trace, cq_id=0, blocking=False)
        blocks.ctrl_written()
        model.verify_enqueue(
            [[0] * T] * B,
            [0] * B,
            [0] * B,
            page_tables=vpt,
            inactive=(~active).tolist(),
            rope_deltas=list(self._rope_deltas),
            step_staged=True,
        )
        step = PendingVerify(
            B, self.K, nv, active, pt, self._last_cap_blocks, self._no_draft, plan=plan, rope_deltas=self._rope_deltas
        )
        predecessor.successor = step
        self._pending.append(step)
        self._enqueue_step_and_drafter(step)
        self.chained_steps += 1
        return step

    def _materialize(self, predecessor, active, hint, page_table, mamba_state, sampling):
        """Build a chained verify from the engine's own read of its predecessor: wait for it, walk its ids, take the
        device drafts (checked against the drafter pack) or run the base propose, then verify the block on the host
        path. The runner's propose of the predecessor returns those drafts once its commit is this walk."""
        B, K, T = self.B, self.K, self.T
        pred = predecessor
        ids = pred.wait()
        if pred.tokens is None:
            raise RuntimeError("the predecessor's block is unknown: its own predecessor never proposed")
        m = greedy_counts(ids, pred.tokens, pred.num_valid)
        m[~pred.active] = 0
        acc = (m + 1).to(torch.int64)
        ctok = torch.full((B, T), PLACEHOLDER_TOKEN_ID, dtype=torch.int64)
        cpos = torch.full((B, T), -1, dtype=torch.int64)
        for u in range(B):
            c = int(acc[u])
            ctok[u, :c] = ids[u, :c]
            if bool(pred.active[u]):
                cpos[u, :c] = int(pred.p0[u]) + 1 + torch.arange(c)
        if pred.plan is not None:
            plan = pred.plan
            want = drafter_pack(
                ctok, m.tolist(), plan.p0, plan.declined, plan.keep, self._rope_base(plan.p0, pred.rope_deltas), K,
                self.model.rope_table_rows(),
            ).to(torch.int64)  # fmt: skip
            if not torch.equal(want, pred.pack[: want.shape[0], : want.shape[1]]):
                raise ValueError("the device acceptance of a chained verify's predecessor differs from its ids' walk")
            drafts = pred.drafts.clone()
            drafts[torch.tensor(plan.declined)] = 0
            result = self._draft_result(drafts, plan.declined)
        else:
            result = MTPSpecEngine.propose_draft_tokens(self, K, ctok.to(torch.int32), cpos.to(torch.int32), acc)
        pred.materialized = (ctok, acc, result)
        tokens = torch.zeros(B, T, dtype=torch.int64)
        positions = torch.full((B, T), -1, dtype=torch.int64)
        nv = torch.zeros(B, dtype=torch.int32)
        for u in range(B):
            if not bool(active[u]):
                continue
            c, n = int(acc[u]), min(int(hint[u]), int(result.num_valid[u]))
            tokens[u, 0] = ctok[u, c - 1]
            tokens[u, 1 : n + 1] = result.draft_token_ids[u, :n].to(torch.int64)
            positions[u, : n + 1] = int(pred.p0[u]) + c + torch.arange(n + 1)
            nv[u] = n
        state = _with_start_blocks(mamba_state, torch.where(active, m, 0))
        return self.decode_forward(
            tokens,
            positions,
            num_valid_drafts=nv,
            accepted_counts=torch.where(active, acc, 1).to(torch.int32),
            page_table=page_table,
            spec_mode="argmax_ids",
            mamba_state=state,
            sampling=sampling,
            defer=True,
            keep_pending=True,
        )


def _with_start_blocks(mamba_state, m):
    """``mamba_state`` with each group's start_blocks taken from start_blocks_by_count at column m[r] (rows past m keep
    block 0)."""
    groups = []
    for g in mamba_state.groups:
        by_count = torch.as_tensor(g.start_blocks_by_count)
        starts = torch.zeros(by_count.shape[0], dtype=torch.int32)
        n = min(int(m.numel()), by_count.shape[0])
        starts[:n] = by_count[torch.arange(n), m[:n].to(torch.int64)].to(torch.int32)
        if dataclasses.is_dataclass(g):
            groups.append(dataclasses.replace(g, start_blocks=starts, start_blocks_by_count=None))
        else:
            groups.append(SimpleNamespace(**{**vars(g), "start_blocks": starts, "start_blocks_by_count": None}))
    if dataclasses.is_dataclass(mamba_state):
        return dataclasses.replace(mamba_state, groups=tuple(groups))
    return SimpleNamespace(**{**vars(mamba_state), "groups": tuple(groups)})


def get_spec_engine(model, batch_size, draft_len, engine_cls=MTPSpecEngine):
    """The engine cached on ``model._spec_engine``, keyed (class, B, K); a different key shuts the old one down
    first."""
    key = (int(batch_size), int(draft_len))
    eng = getattr(model, "_spec_engine", None)
    if eng is not None and type(eng) is engine_cls and (eng.B, eng.K) == key:
        return eng
    if eng is not None:
        eng.shutdown()
    eng = engine_cls(model, *key)
    model._spec_engine = eng
    return eng
