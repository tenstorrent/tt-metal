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
import math
import os
import time
from dataclasses import dataclass

import torch
from loguru import logger
from ttnn.tools import trace_allocation_tracker

import ttnn

# Prompt length above which the reseed goes back to the per-slot loop (B == 1 only).
EAGER_RESEED_PROMPT_LEN = 131072
# Verify rows B*(K+1) must fit one decode tile (QKV heads, matmul configs, GDN core budget).
MAX_SPEC_ROWS = ttnn.TILE_SIZE
# Prompt lengths warmed in prepare: exact bucket lengths take the no-mask prefill path, the others the masked one.
_WARM_PROMPT_LENS = (100, 128, 200, 256, 400, 512, 900, 1024, 1900, 2048, 2148, 2248, 2448, 2948, 3948, 4096)
# Runner contract: token id for unused tail columns of a returned block.
PLACEHOLDER_TOKEN_ID = -1
SPEC_MODES = ("argmax_ids", "logits")


@dataclass
class VerifyResult:
    """decode_forward result: argmax_ids int32 [B, 1+K] (unused tail columns = -1), logits [B, 1+K, V] or None."""

    spec_mode: str
    argmax_ids: torch.Tensor
    logits: torch.Tensor | None = None


@dataclass
class DraftResult:
    """propose_draft_tokens result: draft_token_ids int32 [B, K], num_valid int32 [B] in [0, K]."""

    draft_token_ids: torch.Tensor
    num_valid: torch.Tensor


# --------------------------------------------------------------------------------------------- #
# Pure host helpers (no device needed)
# --------------------------------------------------------------------------------------------- #
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
    """Per row: may it draft? Its K drafter writes (p..p+K-1) and next verify window (p+1..p+1+K) must stay in capacity."""
    return [int(p) + 1 + K < capacity for p in anchor_pos]


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
    block_bytes = 2 * n_local_kv * int(block_size) * head_dim * int(kv_dtype_bytes)
    buckets = _MASK_BUCKETS_DEFAULT if mask_buckets is None else mask_buckets
    engine = (
        sum(_tile_bytes([1, 1, 1, b], bf) for b in buckets)  # _last_sel
        + _rm_bytes([B, nb], i32)
        + B * _rm_bytes([1, nb], i32)  # _mtp_pt, _mtp_pt_rows
        + _tile_bytes([1, 1, B, dim_frac], bf)  # _hp_buf
        + _tile_bytes([1, 1, B, rows], bf)  # _anchor_sel
        + _tile_bytes([1, 1, 1, rows], bf)  # _eager_sel
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


class MTPSpecEngine:
    """MTP speculative decode engine over B users (see module docstring)."""

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
        # Verify runs the fused GDN op; plain decode must use the same math or greedy near-ties flip.
        assert model.gdn_fused_decode, (
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
        # QWEN36_SPEC_CHECK_ANCHOR=1: assert the anchor matmul is bit-equal to the rows it names.
        self._check_anchor = bool(int(os.environ.get("QWEN36_SPEC_CHECK_ANCHOR", "0")))
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
        self.eager_reseed_len = EAGER_RESEED_PROMPT_LEN  # B == 1 anchors past this reseed eagerly
        self._vfeed = None  # the verify trace's own output rows; valid until the next replay
        # A/B + safety switch: 1 = run the drafter eagerly even when its trace is recorded
        self.recorded_drafter = os.environ.get("QWEN36_SPEC_EAGER_DRAFTER", "0") != "1"
        self._dr_bufs = []  # persistent drafter-trace inputs (all of the _dr_* tensors below)
        self._dr_tok0 = None  # [B,1] uint32 pending tokens
        self._dr_pos, self._dr_cos, self._dr_sin = [], [], []  # per draft step k: [B] int32, [1,B,1,rd] bf16 x2
        self._dr_rs_tok = (
            self._dr_rs_pos
        ) = self._dr_rs_pt = self._dr_rs_cos = self._dr_rs_sin = None  # reseed rows (B*T)
        self._dr_trace = None  # trace id: anchor gather + batched reseed + K draft steps
        self._dr_outs = []  # K persistent [1,1,B] uint32 ids outputs of that trace
        self._last_pt = None  # fitted host page table of the most recent verify
        self._fresh = {}  # slot -> prompt length, prefilled and awaiting the seed verify
        self._active = set()  # slots holding a live request
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
        self._stage(pt, self._mtp_pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        for u in range(self.B):
            self._stage(pt[u : u + 1].contiguous(), self._mtp_pt_rows[u], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)

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
        self._last_sel = {
            b: self._alloc_host_replicated(
                torch.zeros(1, 1, 1, b, dtype=torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT
            )
            for b in model._PREFILL_MASK_BUCKETS
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
        t0 = time.perf_counter()
        model.prepare_verify_trace(self._prep_pt, T, warm_positions=[0] * B, decode_cfg=True)

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
        finally:
            self.mesh.set_program_cache_misses_allowed(True)
        self.captured = True

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
        self._dr_tok0 = self._dr_rs_tok = self._dr_rs_pos = self._dr_rs_pt = self._dr_rs_cos = self._dr_rs_sin = None
        self._dr_pos, self._dr_cos, self._dr_sin = [], [], []
        bufs = [
            *self._last_hidden,
            *self._last_sel.values(),
            self._mtp_pt,
            *self._mtp_pt_rows,
            self._hp_buf,
            self._anchor_sel,
            self._eager_sel,
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
                )
            ),
        }
        per_token = {"mtp_kv": kv_bytes // (n_blocks * int(att.paged_k.shape[-2]))}
        for d in (per_seq, fixed, per_token):
            d["total"] = sum(d.values())
        return {"per_seq": per_seq, "fixed": fixed, "per_token": per_token}

    def release(self, slot):
        """Mark a slot empty (no device work)."""
        self._active.discard(slot)
        self._fresh.pop(slot, None)
        reset_draft_skip(self, slot)

    # --------------------------------------------------------------------- #
    # Prefill
    # --------------------------------------------------------------------- #
    def prefill(self, slot, prompt_ids, page_table_row, _cap_blocks=None):
        """Eager prefill into state slot ``slot``, warming its MTP KV; returns host float logits [vocab].
        Replays the recorded chunk trace when B == 1 and record_prefill ran, else the eager path.
        The GDN seed and the MTP KV of slot Tp-1 are written by the first ``verify``."""
        reset_draft_skip(self, slot)
        model = self.model
        assert 0 <= slot < self.B, f"slot {slot} out of range [0,{self.B})"
        assert self._last_hidden, "prefill before prepare"
        prompt_list = [int(t) for t in torch.as_tensor(prompt_ids).reshape(-1)]
        Tp = len(prompt_list)
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
    def verify(self, tokens, start_pos, accepted_counts, page_table, read_logits=False):
        """Replay the verify trace over [B, T] tokens at start_pos[u] + j. The first verify after prefill is
        the seed: row 0 holds the first generated token, other rows are filler, accepted_counts = 1.
        Returns (ids int32 [B, T], logits [B, T, vocab] or None)."""
        assert self.captured, "verify before capture"
        B, T = self.B, self.T
        tokens = self._ints(tokens, (B, T))
        start_pos = self._ints(start_pos, (B,))
        mi_prev = self._ints(accepted_counts, (B,)) - 1
        assert bool(((mi_prev >= 0) & (mi_prev < T)).all()), f"accepted_counts must be in [1, {T}]"
        self._last_cap_blocks = int(torch.as_tensor(page_table).shape[-1])
        pt = self._fit(page_table)
        self._last_pt = pt
        assert int(start_pos.max()) + T <= table_capacity(
            self._last_cap_blocks, self.block_size
        ), "verify window past the paged KV capacity"

        if self._fresh:
            if set(self._active) != set(self._fresh):
                raise NotImplementedError("joining a running batch needs per-slot GDN seeding")
            # Seed ring slot 0 + E_prev from the post-prefill state, then write each fresh user's slot Tp-1 MTP KV.
            for dn in self._gdn:
                dn.seed_spec_state()
            self._stage_mtp_tables(pt)
            for u, Tp in sorted(self._fresh.items()):
                assert int(start_pos[u]) == Tp, f"slot {u}: start_pos {int(start_pos[u])} != prompt length {Tp}"
                self._warm_mtp_last(self._last_hidden[u], int(tokens[u, 0]), Tp - 1, self._mtp_pt_rows[u])
            self._fresh.clear()

        args = (tokens.tolist(), start_pos.tolist(), mi_prev.tolist())
        if self._timing:
            ids, vfeed, lt = self._verify_split(*args, read_logits=read_logits, page_tables=pt)
        else:
            ids, vfeed, lt = self.model.verify_traced(*args, read_logits=read_logits, page_tables=pt)
        self._vfeed = vfeed
        ids = torch.tensor(ids, dtype=torch.int32).reshape(B, T)
        return ids, (None if lt is None else lt.reshape(B, T, -1))

    def decode_forward(
        self, tokens, start_pos, *, num_valid_drafts, accepted_counts, page_table, spec_mode="argmax_ids"
    ):
        """Runner verify contract over a [B, 1+K] block (see sanitize_verify_block); returns VerifyResult."""
        if spec_mode not in SPEC_MODES:
            raise ValueError(f"spec_mode {spec_mode!r} not in {SPEC_MODES}")
        clean, start0, active = sanitize_verify_block(tokens, start_pos, num_valid_drafts, self.K)
        self._no_draft = note_withheld_drafts(
            self._offered,
            self._no_draft,
            self._speculated,
            self._withheld_streak,
            num_valid_drafts,
            self.no_draft_streak,
        )
        if clean.shape[0] != self.B:
            raise ValueError(f"block has {clean.shape[0]} rows, engine has B={self.B}")
        if not bool(active.any()):
            raise ValueError("no active row in the verify block")
        if self.B > 1 and not bool(active.all()):
            raise NotImplementedError("inactive rows at B>1 need scratch-redirected KV writes")
        ids, logits = self.verify(clean, start0, accepted_counts, page_table, read_logits=spec_mode == "logits")
        return VerifyResult(spec_mode, mask_unused_columns(ids, num_valid_drafts), logits)

    def _verify_split(self, tokens, positions, mi_prev, **kw):
        """verify_traced with the readback timed separately (hooks ttnn.get_device_tensors); sets ``last_verify_timing``."""
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
        self, committed_tokens, committed_positions, accepted_counts, page_table, decline=None, cap_blocks=None
    ):
        """Refresh the anchor, reseed the MTP KV of the committed slots, then draft K tokens per user.
        ``decline`` (list[bool] [B]) rows draft nothing: their drafter writes go to scratch and their ids are 0;
        without it every row must fit. ``cap_blocks``: the caller's real table width (default: page_table's).
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
        pt = self._fit(page_table)
        self._check_scratch(pt)
        eager = B == 1 and p[0] > self.eager_reseed_len
        if self.recorded_drafter and self.captured and self._dr_trace is not None and not any(decline) and not eager:
            self._stage(pt, self._mtp_pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            return self._propose_recorded(ctok, cpos, m, pending, p, pt)
        self._stage_mtp_tables(pt)

        self._set_anchor(self._vfeed, m)
        if eager:
            self._reseed_mtp(ctok, cpos, m)
        else:
            self._reseed_mtp_batched(ctok, cpos, m, pt)
        if all(decline):
            return torch.zeros(B, K, dtype=torch.int32)
        if any(decline):  # declined rows draft at position 0 over an all-scratch table; their ids are discarded
            dpt = pt.clone()
            dpt[torch.tensor(decline)] = self._scratch_block
            self._stage(dpt, self._mtp_pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
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
        B, T, K = self.B, self.T, self.K
        if int(num_drafts) != K:
            raise ValueError(f"num_drafts {num_drafts} != engine K {K}")
        if page_table is None:
            if self._last_pt is None:
                raise ValueError("page_table is None and no verify has run")
            page_table, cap_blocks = self._last_pt, self._last_cap_blocks
        else:
            cap_blocks = int(torch.as_tensor(page_table).shape[-1])
        if hidden is not None:
            raise ValueError("hidden must be None (the drafter's hidden handoff is on device)")
        ctok = _int_block(committed_tokens, B, T, "committed_tokens")
        cpos = _int_block(committed_positions, B, T, "committed_positions")
        acc = _int_vec(accepted_counts, B, "accepted_counts")
        if bool(((acc < 1) | (acc > T)).any()):
            raise ValueError(f"accepted_counts must be in [1, {T}], got {acc.tolist()}")
        for u in range(B):
            c = int(acc[u])
            if bool((ctok[u, :c] < 0).any()):
                raise ValueError(f"row {u}: negative committed token in columns 0..{c - 1}")
            if not torch.equal(cpos[u, :c], cpos[u, 0] + torch.arange(c)) or int(cpos[u, 0]) < 1:
                raise ValueError(f"row {u}: committed positions {cpos[u, :c].tolist()} are not consecutive from >= 1")
        _, _, _, p = accepted_plan(ctok, cpos, acc, B, T)
        decline = [not f for f in draft_rows_fit(p, K, table_capacity(cap_blocks, self.block_size))]
        skip, skip_all = propose_skip_plan(self._no_draft, B)
        if skip_all:  # the runner withholds every row's drafts: no anchor/reseed/drafter work
            self.skipped_propose_calls += 1
            self._offered.clear()
            return DraftResult(torch.zeros(B, K, dtype=torch.int32), torch.zeros(B, dtype=torch.int32))
        decline = [d or s for d, s in zip(decline, skip)]
        drafts = self.propose(ctok, cpos, acc, page_table, decline=decline, cap_blocks=cap_blocks)
        num_valid = torch.tensor([0 if d else K for d in decline], dtype=torch.int32)
        self._offered = {u for u in range(B) if decline[u] is False}
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
            logits, h_next = self.model.ttnn_mtp_decode_forward(h, tok_tt, [pu + k for pu in p], self._mtp_pt)
            idx = self._argmax_last(logits)  # [1,1,B] uint32 ROW_MAJOR
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

    def _drafter_body(self, feed):
        """Anchor gather from ``feed`` [1,1,B*T,dim/tp], batched reseed, K chained draft steps; every input is a
        persistent buffer (device-input twin of _set_anchor + _reseed_mtp_batched + _draft). Returns the K ids
        tensors [1,1,B] uint32 (step k's drafts)."""
        B, mtp = self.B, self.mtp
        Hp = ttnn.matmul(self._anchor_sel, feed, compute_kernel_config=self._anchor_cc)
        ttnn.copy(Hp, self._hp_buf)
        ttnn.deallocate(Hp)
        _, hn = mtp.forward_decode(
            feed,
            self._dr_rs_tok,
            self._dr_rs_pos,
            self._dr_rs_cos,
            self._dr_rs_sin,
            self._dr_rs_pt,
            need_logits=False,
            alias_kv_write=True,
            spec_n_users=B,
        )
        ttnn.deallocate(hn)
        h, tok, outs = self._hp_buf, self._dr_tok0, []
        for k in range(self.K):
            logits, hn = mtp.forward_decode(h, tok, self._dr_pos[k], self._dr_cos[k], self._dr_sin[k], self._mtp_pt)
            idx = self._argmax_last(logits)
            ttnn.deallocate(logits)
            tok = ttnn.reshape(idx, (B, 1))
            outs.append(idx)
            if h is not self._hp_buf:
                ttnn.deallocate(h)
            h = hn
        ttnn.deallocate(h)
        return outs

    def _propose_recorded(self, ctok, cpos, m, pending, p, pt):
        """Stage the drafter trace's inputs (host->device copies only; _mtp_pt already staged), replay, read ids.
        The reseed rows always run (padding-only when every m == 0)."""
        B, T, K, model = self.B, self.T, self.K, self.model
        stage, bf, rm, tile = self._stage, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT
        stage(anchor_selector(m, B, T), self._anchor_sel, bf, tile)
        plan = reseed_row_plan(ctok, cpos, m, B, T)
        rtok, rpos, rpt = reseed_rows_host(plan, pt, T, self.nb, self._scratch_block)
        rcos, rsin = model._rope_tp_cos_sin_decode_torch(rpos)
        stage(rtok, self._dr_rs_tok, ttnn.uint32, rm)
        stage(rpos, self._dr_rs_pos, ttnn.int32, rm)
        stage(rpt, self._dr_rs_pt, ttnn.int32, rm)
        stage(rcos, self._dr_rs_cos, bf, tile)
        stage(rsin, self._dr_rs_sin, bf, tile)
        tok0, poss = draft_step_plan(pending, p, K)
        stage(tok0, self._dr_tok0, ttnn.uint32, rm)
        for k, pv in enumerate(poss):
            c, s = model._rope_tp_cos_sin_decode_torch(pv)
            stage(pv, self._dr_pos[k], ttnn.int32, rm)
            stage(c, self._dr_cos[k], bf, tile)
            stage(s, self._dr_sin[k], bf, tile)
        ttnn.execute_trace(self.mesh, self._dr_trace, cq_id=0, blocking=False)
        if any(m):
            self.mtp_extra_steps += 1
        steps = [self._id_to_host(t) for t in self._dr_outs]  # steps[k][u]
        return torch.tensor([[steps[k][u] for k in range(K)] for u in range(B)], dtype=torch.int32)

    def _draft_warmup(self, pending, Hp, p):
        """One need_logits=True draft step before capture (compiles head norm, LM head, argmax, mesh_partition)."""
        logits, h = self.model.ttnn_mtp_decode_forward(Hp, [int(t) for t in pending], list(p), self._mtp_pt)
        idx = self._argmax_last(logits)
        for t in (logits, idx, h):
            ttnn.deallocate(t)
        ttnn.synchronize_device(self.mesh)

    def _argmax_last(self, logits):
        """argmax over vocab for B rows -> [1,1,B] uint32. Untilize first: argmax on TILE input is single-core and slow."""
        u = ttnn.untilize(logits, use_multicore=True)
        out = ttnn.argmax(u, dim=-1, keepdim=False)
        ttnn.deallocate(u)
        return out

    def _id_to_host(self, id_tt):
        """[*, B] uint32 device ids -> B ints, read from the device-0 replica."""
        t = ttnn.to_torch(ttnn.get_device_tensors(id_tt)[0])
        return [int(v) for v in t.reshape(-1)[: self.B]]

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

    def _warm_mtp_last(self, last_hidden, first_tok, slot, page_table_tt):
        """Write the MTP KV of a user's slot Tp-1 from (its feed row, the first generated token); the
        first draft attends to it."""
        _, h_next = self.model.ttnn_mtp_decode_forward(
            last_hidden, int(first_tok), slot, page_table_tt, need_logits=False
        )
        ttnn.deallocate(h_next)

    def _reseed_mtp(self, committed_tokens, committed_positions, m):
        """Per-slot reseed (B == 1 past EAGER_RESEED_PROMPT_LEN): one 1-row drafter decode per committed slot."""
        T = self.T
        for u, (tok_row, pos_row) in enumerate(zip(committed_tokens.tolist(), committed_positions.tolist())):
            for i in range(m[u]):
                r = u * T + i
                row = self._eager_row(self._vfeed, r)
                _, h_next = self.model.ttnn_mtp_decode_forward(
                    row, tok_row[i], pos_row[i] - 1, self._mtp_pt_rows[u], need_logits=False
                )
                ttnn.deallocate(row)
                ttnn.deallocate(h_next)
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
        cos_t, sin_t = self.model._rope_tp_cos_sin_decode_torch(pos)
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
        for t in (tok_tt, pos_tt, pt_tt, cos, sin, h_next):
            ttnn.deallocate(t)
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
        if self._check_anchor:
            self._check_anchor_rows(Hp, feed_rows, m)
        ttnn.deallocate(Hp)

    def _check_anchor_rows(self, Hp, feed_rows, m):
        """QWEN36_SPEC_CHECK_ANCHOR=1: assert the one-hot matmul is bit-identical to the rows it names (debug only)."""
        T = self.T
        got = ttnn.to_torch(ttnn.get_device_tensors(Hp)[0]).reshape(-1, Hp.shape[-1])
        ref = ttnn.to_torch(ttnn.get_device_tensors(feed_rows)[0]).reshape(-1, feed_rows.shape[-1])
        for u in range(self.B):
            r = u * T + int(m[u])
            assert torch.equal(got[u], ref[r]), (
                f"anchor matmul mismatch for user {u} (m={int(m[u])}, feed row {r}): "
                f"max |delta| {float((got[u].float() - ref[r].float()).abs().max()):.3e}"
            )


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
