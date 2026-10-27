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
    K <= 2050, ``dsa.dense_exact_max_pos``) a row simply gets no drafts (``ServingSpecPolicy``)."""
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
