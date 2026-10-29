# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host side of the vLLM adapter's speculative decode (``tt/generator_vllm.py``, the vllm-tt-plugin contract of
``docs/SPEC_DECODE_CONTRACT.md``: ``spec_plan`` / ``decode_forward(spec_mode=...)`` / ``propose_draft_tokens``):
the enable switch and the draft-length policy, the config-time decision the plugin admits against, the candidate
block translation (the plugin's ``[B, 1+K]`` rows with placeholder columns -> the verify's real-token rows), the
per-step records the adapter keeps between a verify and its commit, the resident-bytes estimate of the drafter,
and a faithful host copy of the plugin's greedy accept walk (tests, self-checks). Pure Python + torch (no ttnn, no
device, no vLLM import at module scope), so the policy is testable on a CPU (``tests/test_serving_spec_host.py``).

The served loop (the harness loop of ``tt/spec_decode.py`` spread over the plugin's calls):

* an ordinary decode step of a speculating launch = graph P (the plain step + MTP level 1 on every user slot);
* a verify step = graph V on the ``[anchor, d_1 .. d_K]`` rows of every slot (``VerifyOutput(argmax_ids)``);
* ``propose_draft_tokens`` after a verify = graph C (the KDA / index commit with the runner's accepted counts) +
  graph D (levels ``2 .. K`` from the accepted row); after an ordinary step = graph D0 (levels ``2 .. K`` from
  P's level-1 output). Every proposal is ``K`` drafts or none per row (``DraftOutput.num_valid``).

Knobs: ``GLM53_VLLM_SPEC=1`` enables the path (the MTP drafter is built, the P / V / C / D / D0 graphs are warmed and
captured at boot; off = today's plain adapter, bit for bit); ``GLM53_VLLM_SPEC_K`` (default 3) the draft length
ceiling, capped by ``B (K + 1) <= 32`` per mesh row (``spec_decode.spec_draft_len``); ``GLM53_SPEC_COMMIT`` the
KDA commit mode (the adapter's default is ``fused``, the measured 8x4 point; the model's own default stays
``blend``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Sequence

import torch

SPEC_ENV = "GLM53_VLLM_SPEC"
SPEC_K_ENV = "GLM53_VLLM_SPEC_K"
# The draft length by the batch (the user's direction, 2026-10-29; doc/mesh/spec_k_by_batch.md): the harness sweep on the
# 8x4 at 45 layers (fused commit, the H1/H2 defaults, tests/test_mtp_spec.py) finds the drafter's acceptance at a
# plateau of ~2.5 drafts whatever K, so the shortest K that keeps it wins and a longer one only adds verify rows:
# 4 users per row K=3 30.1 tok/s per user; 1 per row K=7 25.5, K=15 16.9 (the same 2.47 accepted). The table maps
# users per row -> K; a launch uses the entry at the smallest listed batch >= its own; the catalog's
# num_speculative_tokens is the ceiling. GLM53_SPEC_K_BY_ROWS="B:K,B:K" replaces the table, "off" leaves the catalog's K.
K_BY_ROWS_ENV = "GLM53_SPEC_K_BY_ROWS"
DEFAULT_K_BY_ROWS: tuple[tuple[int, int], ...] = ((1, 3), (2, 3), (4, 3), (8, 3), (16, 1))
DEFAULT_SPEC_K = 3  # the measured sweet spot on the 8x4 (24.0 tok/s per user at 32 users, K=3, fused commit)
SPEC_COMMIT_ENV = "GLM53_SPEC_COMMIT"
DEFAULT_SERVING_COMMIT = "fused"
SPEC_COMMIT_MODES = ("blend", "gated", "fused")  # = spec_decode.SPEC_COMMIT_MODES (ttnn-free copy)
MAX_VERIFY_ROWS = 32  # = spec_decode.MAX_VERIFY_ROWS: B (K + 1) candidate rows fit one tile row per mesh row
ACCEPT_MODE_ARGMAX_IDS = "argmax_ids"
PLACEHOLDER_TOKEN_ID = -1  # the plugin's padded-column / padded-tail marker (vllm_tt_plugin.spec_decode)
MTP_LAYER = 45
MTP_CACHE_MARGIN = 32  # MTP positions beyond max_seq_len (the draft chain rows: tt/mtp.py, test_mtp_spec.py)
DRAFTER_STATE_INTERNAL = "internal"
HIDDEN_HANDOFF_ON_DEVICE = "on_device"
SPEC_REQUIREMENT_DEVICE_PROPOSE = "device_propose"
# GLM53_VLLM_SPEC_ALLOC_CHECK=1 (debug, default off): after every speculative step / proposal / prompt fill the
# adapter compares the DRAM allocator's footprint with the one taken right after the captures and refuses a growth
# (a device tensor allocated under live traces survives into the next replay, which clobbers it: the discipline
# GLMMesh.allocations_frozen states). One host-side allocator query per device per step.
ALLOC_CHECK_ENV = "GLM53_VLLM_SPEC_ALLOC_CHECK"


def alloc_check_enabled(env=None) -> bool:
    env = os.environ if env is None else env
    return _flag(env, ALLOC_CHECK_ENV, "0")


# ------------------------------------------------------------------------------------------------- knobs
def _flag(env, name: str, default: str) -> bool:
    return str(env.get(name, default)).strip().lower() in ("1", "true", "yes", "on")


def spec_enabled(env=None) -> bool:
    """``GLM53_VLLM_SPEC=1``: the adapter builds the MTP drafter and serves the plugin's speculative contract."""
    env = os.environ if env is None else env
    return _flag(env, SPEC_ENV, "0")


def draft_len_cap(users_per_row: int, k_ctx: int, max_rows: int = MAX_VERIFY_ROWS) -> int:
    """``min(k_ctx, max_rows // B - 1)`` (0 = plain decode): ``spec_decode.spec_draft_len`` without ttnn."""
    if int(users_per_row) <= 0:
        raise ValueError(f"users_per_row must be positive, got {users_per_row}")
    return max(0, min(int(k_ctx), max_rows // int(users_per_row) - 1))


def k_by_rows(env=None) -> tuple[tuple[int, int], ...] | None:
    """The (users per row, K) table in force: the default, the ``GLM53_SPEC_K_BY_ROWS`` list, or None for ``off``."""
    env = os.environ if env is None else env
    raw = str(env.get(K_BY_ROWS_ENV, "")).strip().lower()
    if raw in ("off", "none", "0"):
        return None
    if not raw:
        return DEFAULT_K_BY_ROWS
    table = []
    for item in raw.split(","):
        if not item.strip():
            continue
        b, _, k = item.partition(":")
        if not k:
            raise ValueError(f"{K_BY_ROWS_ENV}={raw!r}: entries are B:K")
        table.append((int(b), int(k)))
    return tuple(sorted(table))


def best_k(users_per_row: int, table=None) -> int | None:
    """The table's K for ``users_per_row``: the entry at the smallest listed batch >= it, None when none (or no
    table): the cap and the catalog's K then decide."""
    if table is None:
        return None
    fits = [k for b, k in sorted(table) if b >= int(users_per_row)]
    return fits[0] if fits else None


def spec_k_setting(env=None) -> int:
    """``GLM53_VLLM_SPEC_K``: the draft-length ceiling the adapter serves (default 3)."""
    env = os.environ if env is None else env
    raw = str(env.get(SPEC_K_ENV, str(DEFAULT_SPEC_K))).strip()
    k = int(raw)
    if k < 1:
        raise ValueError(f"{SPEC_K_ENV}={raw!r}: the draft length must be >= 1")
    return k


def serving_commit_mode(env=None) -> str:
    """``GLM53_SPEC_COMMIT`` for the served loop: ``fused`` unless set (``blend`` / ``gated`` / ``fused``)."""
    env = os.environ if env is None else env
    mode = str(env.get(SPEC_COMMIT_ENV, DEFAULT_SERVING_COMMIT)).strip().lower()
    if mode not in SPEC_COMMIT_MODES:
        raise ValueError(f"{SPEC_COMMIT_ENV}={mode!r}: expected one of {SPEC_COMMIT_MODES}")
    return mode


# ---------------------------------------------------------------------------------------- config-time decision
@dataclass(frozen=True)
class SpecDecision:
    """What ``GLM53FlashForCausalLM.spec_plan`` decides for one ``(max_num_seqs, requested_k)`` point: ``admitted``
    with the draft length ``effective_k`` (``<= requested_k``, ``<= k_cap``), or a refusal with its ``reason`` and
    the draft lengths the same concurrency would accept (``supported_k``)."""

    admitted: bool
    effective_k: int = 0
    k_cap: int = 0
    users_per_row: int = 0
    commit_mode: str = DEFAULT_SERVING_COMMIT
    reason: str = ""
    supported_k: tuple[int, ...] = ()


def resolve_spec_decision(
    env, *, max_num_seqs: int, requested_k: int, rows: int, cols: int, max_model_len: int | None = None
) -> SpecDecision:
    """The pure policy behind ``spec_plan`` (never raises on a bad launch: the refusal is the return value).

    Refused: the switch off; per-row prefill ownership (``GLM53_VLLM_ROWS_INDEPENDENT=1``: the MTP prompt fill is
    not wired for waves); the user-partitioned cache (``GLM53_PARTITION_USERS`` = 1, the serving default on a
    multi-column mesh: ``DsaSpecState`` is not wired for it, so a speculating launch sets it to 0 explicitly); a
    geometry whose verify would exceed one tile row (``B (K + 1) > 32`` for every K >= 1, e.g. 32 users per row);
    a requested draft length below 1. ``max_model_len`` is informational: beyond the dense range (positions +
    K <= 2050, ``dsa.dense_exact_max_pos``) a row simply gets no drafts (``ServingSpecPolicy``). The effective K is
    the batch's measured best (``k_by_rows``) under the catalog's ceiling and the geometry's cap."""
    env = os.environ if env is None else env
    if not spec_enabled(env):
        return SpecDecision(False, reason=f"speculative decoding is off: set {SPEC_ENV}=1 for the GLM53 adapter")
    if _flag(env, "GLM53_VLLM_ROWS_INDEPENDENT", "0"):
        return SpecDecision(
            False, reason="GLM53_VLLM_ROWS_INDEPENDENT=1: the MTP prompt fill is not wired for the wave prefill"
        )
    partitioned = _flag(env, "GLM53_PARTITION_USERS", "1") and int(cols) > 1
    if partitioned:
        return SpecDecision(
            False,
            reason=(
                "the speculative verify is not wired for the user-partitioned DSA cache (GLM53_PARTITION_USERS=1, "
                "the serving default): set GLM53_PARTITION_USERS=0 (replicated caches) for a speculating launch"
            ),
        )
    users_per_row = int(max_num_seqs)
    if users_per_row < 1:
        return SpecDecision(False, reason=f"max_num_seqs {max_num_seqs} < 1")
    try:
        k_setting = spec_k_setting(env)
        commit = serving_commit_mode(env)
    except ValueError as exc:
        return SpecDecision(False, reason=str(exc))
    k_cap = draft_len_cap(users_per_row, k_setting)
    k_any = draft_len_cap(users_per_row, MAX_VERIFY_ROWS)
    supported = tuple(range(1, k_cap + 1))
    if k_cap < 1:
        return SpecDecision(
            False,
            users_per_row=users_per_row,
            reason=(
                f"{users_per_row} users per row leave no room for drafts: B (K + 1) <= {MAX_VERIFY_ROWS} candidate "
                f"rows per mesh row (at most {MAX_VERIFY_ROWS // 2} users per row speculate; {SPEC_K_ENV}={k_setting}, "
                f"the geometry allows K <= {k_any})"
            ),
        )
    if int(requested_k) < 1:
        return SpecDecision(
            False,
            users_per_row=users_per_row,
            reason=f"num_speculative_tokens {requested_k} < 1",
            supported_k=supported,
        )
    effective_k = min(int(requested_k), k_cap)
    try:
        table_k = best_k(users_per_row, k_by_rows(env))
    except ValueError as exc:
        return SpecDecision(False, users_per_row=users_per_row, reason=str(exc), supported_k=supported)
    if table_k is not None:
        effective_k = max(1, min(effective_k, table_k))  # the batch's measured best, under the catalog's ceiling
    return SpecDecision(
        True,
        effective_k=effective_k,
        k_cap=k_cap,
        users_per_row=users_per_row,
        commit_mode=commit,
        supported_k=supported,
    )


# ---------------------------------------------------------------------------------------- the plugin's types
@dataclass(frozen=True)
class _LocalVerifyOutput:
    """``vllm_tt_plugin.spec_decode.VerifyOutput``'s fields for a host without the plugin (tests)."""

    spec_mode: str
    argmax_ids: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    accepted_token_ids: torch.Tensor | None = None
    accepted_counts: torch.Tensor | None = None
    hidden: object = None


@dataclass(frozen=True)
class _LocalDraftOutput:
    draft_token_ids: torch.Tensor
    num_valid: torch.Tensor | None = None
    draft_scores: torch.Tensor | None = None


@dataclass(frozen=True)
class _LocalSpecPlan:
    effective_k: int
    lanes_per_request: int
    extra_bytes_per_seq: int
    extra_bytes_per_token: int
    accept_modes: tuple[str, ...]
    drafter_state: str
    drafter_target_cache_requires: tuple[str, ...] = ()
    supports_narrow_decode: bool = False
    k_by_rows: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class _LocalSpecReject:
    reason: str
    supported_k: tuple[int, ...] = ()


def contract_types() -> dict[str, type]:
    """The plugin's contract types when the plugin is importable (the served process), else the local stand-ins
    with the same fields (host tests): ``{"VerifyOutput", "DraftOutput", "SpecPlan", "SpecReject"}``."""
    try:
        from vllm_tt_plugin.spec_decode import DraftOutput, SpecPlan, SpecReject, VerifyOutput
    except ImportError:
        return {
            "VerifyOutput": _LocalVerifyOutput,
            "DraftOutput": _LocalDraftOutput,
            "SpecPlan": _LocalSpecPlan,
            "SpecReject": _LocalSpecReject,
        }
    return {"VerifyOutput": VerifyOutput, "DraftOutput": DraftOutput, "SpecPlan": SpecPlan, "SpecReject": SpecReject}


def make_spec_plan(decision: SpecDecision, *, extra_bytes_per_seq: int = 0):
    """The plugin's ``SpecPlan`` / ``SpecReject`` of a decision. One decode row per request (the rows are the
    plugin's wire rows; the ``B (K + 1)`` verify rows stay the model's), no per-token drafter bytes (the MTP's
    cache is allocated at build), ``argmax_ids`` only (greedy; the plain decode serves sampled rows through
    ``supports_narrow_decode``), the drafter's state on device (``on_device`` hidden handoff)."""
    types = contract_types()
    if not decision.admitted:
        return types["SpecReject"](reason=decision.reason, supported_k=tuple(decision.supported_k))
    return types["SpecPlan"](
        effective_k=int(decision.effective_k),
        lanes_per_request=1,
        extra_bytes_per_seq=int(extra_bytes_per_seq),
        extra_bytes_per_token=0,
        accept_modes=(ACCEPT_MODE_ARGMAX_IDS,),
        drafter_state=DRAFTER_STATE_INTERNAL,
        supports_narrow_decode=True,
    )


# ------------------------------------------------------------------------------------------------ bytes
def spec_resident_bytes(cfg, rows: int, cols: int, users_per_row: int, max_seq_len: int, expert_bytes: float) -> float:
    """Per-chip bytes the speculative path adds beside the model: the MTP block (layer 45's routed experts in the
    hybrid layout = one MoE layer's share, its dense weights, ``eh_proj``), its own latent cache + pooled index
    for every user slot (replicated caches: all ``users_per_row`` slots per chip, ``max_seq_len + 32`` positions)
    and the verify's persistent buffers (projection caches per KDA layer, candidate rows, outputs)."""
    from models.demos.glm53_flash.tt.memory_plan import BFP8_BYTES, cache_bytes_per_token, expert_bytes_per_chip

    n_moe = max(1, cfg.num_hidden_layers - 3)
    n_dsa = sum(1 for t in cfg.layer_types if t != "linear_attention") if cfg.layer_types else 11
    experts = expert_bytes_per_chip(cfg, rows, cols, expert_bytes) / n_moe
    dense = 2 * cfg.hidden_size * cfg.hidden_size * BFP8_BYTES + 160 * 2**20  # eh_proj + the attention / norms
    cache = users_per_row * (max_seq_len + MTP_CACHE_MARGIN) * cache_bytes_per_token(cfg) / max(1, n_dsa)
    buffers = 64 * 2**20  # qkv_all / g2_all per KDA layer (~1 MB each at R = 32), k_all / g_all, out_* rows
    return experts + dense + cache + buffers


# ------------------------------------------------------------------------------------- the MTP prompt fill
TILE = 32
PREFILL_BUCKETS_ENV = "GLM53_PREFILL_BUCKETS"  # = mesh_model.PREFILL_BUCKETS_ENV (ttnn-free copies, pinned by a test)
DEFAULT_PREFILL_BUCKETS = (256, 512, 1024, 2048, 16384)


def prefill_buckets(env=None) -> tuple[int, ...]:
    """``mesh_model.prefill_buckets`` without ttnn: the eager prefill's padded chunk lengths."""
    env = os.environ if env is None else env
    raw = str(env.get(PREFILL_BUCKETS_ENV, ""))
    if raw.strip() == "0":
        return ()
    if not raw.strip():
        return DEFAULT_PREFILL_BUCKETS
    return tuple(sorted(int(v) for v in raw.split(",") if v.strip()))


def eager_pad_length(seq_len: int, unit: int, buckets: tuple[int, ...] | None = None) -> int:
    """``mesh_model.eager_pad_length`` without ttnn: the row unit's ceiling raised to the smallest bucket."""
    T = -(-int(seq_len) // unit) * unit
    for b in prefill_buckets() if buckets is None else buckets:
        if b % unit:
            raise ValueError(f"prefill bucket {b} is not a multiple of the row unit {unit}")
        if T <= b:
            return b
    return T


def pad_row_mask_host(T: int, seq_len: int) -> torch.Tensor:
    """``moe.pad_row_mask_host`` without ttnn: ``[1, 1, T, 32]`` bf16, column 0 = 1.0 on rows < ``seq_len``."""
    if not 0 <= seq_len <= T:
        raise ValueError(f"seq_len {seq_len} outside [0, {T}]")
    m = torch.zeros(1, 1, T, TILE, dtype=torch.bfloat16)
    m[0, 0, :seq_len, 0] = 1.0
    return m


def pad_mask_ones_host(W: int) -> torch.Tensor:
    """``moe.pad_mask_ones_host`` without ttnn: ``[1, 1, 32, W]`` bf16, row 0 all ones."""
    o = torch.zeros(1, 1, TILE, W, dtype=torch.bfloat16)
    o[0, 0, 0, :] = 1.0
    return o


@dataclass(frozen=True)
class MtpFillPlan:
    """The host side of one MTP prompt fill (``GLM53SpecEngine.prefill_fill``): one prefill chunk of ``seq_len``
    tokens at ``start`` whose streams the main model left as ``rows`` slabs of ``t_loc`` rows (mesh row r holds chunk
    rows ``[r t_loc, (r+1) t_loc)``; ``t_pad = rows * t_loc``, the eager ``eager_pad_length`` or the traced length).

    * ``next_ids`` ``[1, 1, 1, t_pad]`` int32: the MTP's token of chunk row i = the chunk's token i + 1 (rows
      ``0 .. seq_len - 2``), the chunk-end token ``first`` at row ``seq_len - 1`` (the prompt's first generated token
      on the last chunk, the model's own guess of the next chunk's first token on an intermediate one), the pad id
      past it; uploaded row-sharded on its last dim so mesh row r reads ``slab(r)`` next to its streams slab;
    * ``row_mask`` ``[1, 1, t_pad, 32]`` bf16 (``pad_row_mask_host``; None when the chunk fills every row): the rows
      past ``seq_len`` of the main model's streams are unspecified by contract (``TtDSA.prefill_forward``), so the
      hidden handed to the drafter is zeroed there (an exact 0/1 matmul + ``ttnn.where``: NaN-safe, bit-identical
      on the real rows) before the MTP's attention reads them as keys and its router routes them."""

    rows: int
    t_loc: int
    seq_len: int
    start: int
    next_ids: torch.Tensor
    row_mask: torch.Tensor | None

    @property
    def t_pad(self) -> int:
        return self.rows * self.t_loc

    @property
    def masked(self) -> bool:
        return self.row_mask is not None

    @property
    def owner_row(self) -> int:
        """The mesh row holding the chunk's last real row."""
        return (self.seq_len - 1) // self.t_loc

    @property
    def last_local_row(self) -> int:
        return (self.seq_len - 1) % self.t_loc

    def slab(self, row: int) -> torch.Tensor:
        """Mesh row ``row``'s ``[t_loc]`` ids (what its chip receives)."""
        return self.next_ids.reshape(-1)[row * self.t_loc : (row + 1) * self.t_loc]

    def mask_slab(self, row: int) -> torch.Tensor | None:
        return None if self.row_mask is None else self.row_mask[0, 0, row * self.t_loc : (row + 1) * self.t_loc, 0]


def mtp_fill_plan(
    chunk_ids: torch.Tensor, first: int, *, rows: int, t_loc: int, seq_len: int, start: int, pad_token_id: int
) -> MtpFillPlan:
    """The plan of one chunk (checks: ``chunk_ids`` holds exactly ``seq_len`` tokens, ``0 < seq_len <= t_pad``,
    ``t_loc % 32 == 0``, ``start % 32 == 0``)."""
    rows, t_loc, seq_len, start = int(rows), int(t_loc), int(seq_len), int(start)
    ids = torch.as_tensor(chunk_ids).reshape(-1).long()
    t_pad = rows * t_loc
    if rows < 1 or t_loc < TILE or t_loc % TILE:
        raise ValueError(f"a slab of {t_loc} rows on {rows} mesh rows: t_loc must be a positive multiple of {TILE}")
    if not 0 < seq_len <= t_pad:
        raise ValueError(f"seq_len {seq_len} outside (0, {t_pad}] (the chunk's padded rows)")
    if int(ids.numel()) != seq_len:
        raise ValueError(f"the chunk holds {ids.numel()} tokens for seq_len {seq_len}")
    if start < 0 or start % TILE:
        raise ValueError(f"chunk start {start} must be a non-negative multiple of {TILE}")
    if int(first) < 0:
        raise ValueError(f"chunk-end token {first} is not a token id")
    nxt = torch.full((t_pad,), int(pad_token_id), dtype=torch.int32)
    nxt[: seq_len - 1] = ids[1:].to(torch.int32)
    nxt[seq_len - 1] = int(first)
    mask = pad_row_mask_host(t_pad, seq_len) if seq_len < t_pad else None
    return MtpFillPlan(rows, t_loc, seq_len, start, nxt.reshape(1, 1, 1, t_pad), mask)


def mtp_fill_lengths(
    traced_lengths: Sequence[int], *, rows: int, device_prefill_chunk: int, max_seq_len: int, buckets=None
) -> list[int]:
    """Every padded chunk length ``t_pad`` the served prefill can hand the MTP fill: the traced lengths (the replay's
    streams are ``T`` rows) and the eager buckets up to the device chunk (``eager_pad_length`` of any eager chunk),
    each a multiple of the row unit and within the caches. The adapter runs the fill at each of them BEFORE any
    trace capture: the MTP's prefill collectives (``TtMTPBlock._prefill_rows_impl``: the row gather / reduce-scatter
    of its attention) exist at no other shape, and a collective first compiled after a capture corrupts the
    replays (the 2026-10-27 block run: the first traced prompt's fill at T=128 was the first MTP prefill at that
    shape and its routing came back garbage on every row)."""
    unit = TILE * int(rows)
    max_rows = -(-int(max_seq_len) // unit) * unit
    chunk = int(device_prefill_chunk)
    out = {int(T) for T in traced_lengths}
    bks = prefill_buckets() if buckets is None else tuple(buckets)
    for b in bks:
        if b <= chunk:
            out.add(int(b))
    out.add(eager_pad_length(chunk, unit, bks))
    return sorted(T for T in out if T % unit == 0 and T <= max_rows)


# ---------------------------------------------------------------------------------------- per-step policy
@dataclass(frozen=True)
class ServingSpecPolicy:
    """Which rows may draft and whether a step's verify can run: the verify writes candidate rows ``p .. p + K`` of
    every live slot through the dense flash-MLA, exact up to ``max_dense_pos`` (``dsa.dense_exact_max_pos`` = 2050,
    clipped to the model's ``max_seq_len - 1``; ``SpecDecoder.vin.max_dense_pos``). A live slot whose anchor sits
    beyond ``max_dense_pos - K`` cannot be verified (the sparse-range verify is not wired), and the step is
    mesh-wide, so one such slot makes the whole step a plain one: nobody drafts while it is live, and a verify
    the plugin still sends (its drafts proposed before that slot arrived) is answered one column wide."""

    draft_len: int  # K of the captured graphs
    max_dense_pos: int
    max_seq_len: int

    def anchor_ok(self, position: int) -> bool:
        """The verify rows ``position .. position + K`` are inside the dense range and the slot."""
        p = int(position)
        return 0 <= p and p + self.draft_len <= self.max_dense_pos and p + self.draft_len < self.max_seq_len

    def step_can_verify(self, anchor_positions: Sequence[int]) -> bool:
        """Every live anchor of the step passes ``anchor_ok`` (an empty step verifies trivially)."""
        return all(self.anchor_ok(p) for p in anchor_positions)

    def row_drafts(self, next_anchor: int, all_rows_ok: bool, k_requested: int) -> int:
        """``num_valid`` for a row whose next anchor sits at ``next_anchor``: ``min(K, k_requested)`` when the
        row and every other live row can be verified next step, else 0."""
        if not all_rows_ok or not self.anchor_ok(next_anchor):
            return 0
        return max(0, min(self.draft_len, int(k_requested)))


# -------------------------------------------------------------------------------------------- blocks / rows
def candidate_block(row: Sequence[int], valid: int, width: int) -> list[int]:
    """One plugin row ``[anchor, d_1 .. d_K']`` (``K' + 1 = len(row)``, columns past ``valid`` carrying the
    placeholder) -> the ``width`` real token ids the verify embeds: the anchor, the ``valid`` real drafts, then the
    anchor repeated for every padded column (a tentative row whose output the walk never reads; its latent row is
    overwritten by the next committed token). ``width = K + 1`` of the captured graphs may exceed the plugin's
    block width (a launch whose ``num_speculative_tokens`` is below the adapter's K)."""
    row = [int(v) for v in row]
    valid = int(valid)
    if not row:
        raise ValueError("an empty candidate row")
    if valid < 0 or valid > len(row) - 1 or valid > width - 1:
        raise ValueError(f"num_valid_drafts {valid} outside [0, {min(len(row), width) - 1}] for a row of {len(row)}")
    real = row[: 1 + valid]
    if any(t < 0 for t in real):
        raise ValueError(f"candidate row {row} carries a placeholder inside its {valid} valid drafts")
    anchor = real[0]
    return real + [anchor] * (width - len(real))


def wire_rows(values: torch.Tensor, read_index: Sequence[int], fill: int = 0) -> torch.Tensor:
    """Flat per-slot values ``[N, ...]`` -> the plugin's wire rows ``[len(read_index), ...]`` (row ``i`` reads flat
    ``read_index[i]``; a padding row (``-1``) is ``fill``)."""
    out = torch.full((len(read_index),) + tuple(values.shape[1:]), int(fill), dtype=values.dtype)
    for i, idx in enumerate(read_index):
        if int(idx) >= 0:
            out[i] = values[int(idx)]
    return out


def accept_greedy_walk(
    argmax_ids: torch.Tensor, draft_token_ids: torch.Tensor, num_valid_drafts: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``vllm_tt_plugin.spec_decode.accept_greedy_drafts`` on the host (the runner's walk, kept in step with it by
    ``tests/test_serving_spec_host.py``): ``argmax_ids [B, 1+K']`` (return column ``j`` = the model's choice at
    candidate position ``j``), ``draft_token_ids [B, K']``, ``num_valid_drafts [B]`` -> ``(committed [B, 1+K']
    int32 with the placeholder past each row's count, counts [B] int32 in [1, 1+K'])``. A one-column return
    (``K' = 0``: the adapter's plain-step answer to a verify it could not run) commits column 0 everywhere."""
    ids = torch.as_tensor(argmax_ids).to(torch.int32)
    rows, width = ids.shape
    num_drafts = width - 1
    drafts = torch.as_tensor(draft_token_ids).to(torch.int32)[:, :num_drafts]
    valid_n = torch.as_tensor(num_valid_drafts).to(torch.int64).reshape(rows)
    if num_drafts == 0:
        return ids, torch.ones(rows, dtype=torch.int32)
    matched = ids[:, :num_drafts] == drafts
    valid = torch.arange(num_drafts).unsqueeze(0) < valid_n.unsqueeze(1)
    rejected = valid & ~matched
    first_rejection = rejected.to(torch.int8).argmax(dim=1)
    counts = torch.where(rejected.any(dim=1), first_rejection + 1, valid_n.clamp(max=num_drafts) + 1).to(torch.int32)
    columns = torch.arange(width).unsqueeze(0).expand(rows, width)
    committed = torch.where(
        columns < counts.to(torch.int64).unsqueeze(1), ids, torch.full_like(ids, PLACEHOLDER_TOKEN_ID)
    )
    return committed, counts


# ---------------------------------------------------------------------------------------- step records
@dataclass
class VerifyRecord:
    """What a verify step leaves for its commit (``propose_draft_tokens``): the flat anchor positions the graphs
    ran at, the wire row -> flat index map, the live wire rows, the verify's ``ids`` / level-1 ``d1`` (``[N, K+1]``
    user-major) and the width the plugin's block had."""

    positions: list[int]
    read_index: list[int]
    live: list[int]
    ids: torch.Tensor
    d1: torch.Tensor
    width: int
    valid: dict[int, int] = field(default_factory=dict)  # wire row -> num_valid_drafts


@dataclass
class PlainRecord:
    """What an ordinary (P) step of a speculating launch leaves for the next proposal."""

    positions: list[int]
    read_index: list[int]
    live: list[int]


@dataclass
class SpecServingStats:
    """The served loop's counters for the perf log line."""

    verifies: int = 0
    plain_steps: int = 0
    fallbacks: int = 0  # verifies answered one column wide (the step could not run V)
    committed: int = 0  # tokens committed by verifies over the live rows
    accepted: int = 0  # drafts accepted by verifies over the live rows
    verify_s: float = 0.0
    propose_s: float = 0.0

    def summary(self) -> str:
        per = self.committed / self.verifies if self.verifies else 0.0
        acc = self.accepted / self.verifies if self.verifies else 0.0
        ms_v = 1000 * self.verify_s / self.verifies if self.verifies else float("nan")
        ms_p = 1000 * self.propose_s / max(1, self.verifies + self.plain_steps)
        return (
            f"{self.verifies} verifies ({self.fallbacks} one-column), {self.plain_steps} plain steps, "
            f"{per:.2f} tokens / {acc:.2f} accepted drafts per verify over the live rows, verify {ms_v:.1f} ms, "
            f"commit + draft {ms_p:.1f} ms per step (host wall)"
        )
