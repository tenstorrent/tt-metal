# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared Gemma4 prefill trace policy for standalone and vLLM generators."""

import os

import torch
from loguru import logger

from models.tt_transformers.tt.generator import (
    MAX_BATCHED_PREFILL_SEQ_LEN,
    SUPPORTED_PREFILL_BATCH_SIZES,
    batched_prefill_padded_batch,
    max_prefill_chunk_size_cutoff,
)

# Kernel sequence lengths that may capture/replay prefill device traces (MoE only).
#
# Each bucket is captured as a *separate resident prefill trace* at warmup, and
# the high buckets are large for big models (e.g. the 4096 bucket on
# Gemma4-26B-A4B is ~0.5 GB on its own). ``GEMMA4_TRACE_PREFILL_SEQ_LENS`` lets a
# deployment trim the set to the prompt lengths it actually serves — e.g. a
# throughput benchmark with short prompts only needs the smallest bucket — which
# directly shrinks the required ``trace_region_size``. The override drives both
# warmup capture (``patch_gemma4_trace_model_args``) and runtime eligibility
# (``can_gemma4_enable_prefill_trace``) so they stay consistent.
# Include 4096 so cold single-chunk / first scheduler grant ≤4k can replay a
# prefill device trace (TTFT). Continuations (``num_cached_tokens>0`` — APC hits
# and vLLM chunked-prefill remnants) always stay eager: Gemma4 sliding layers
# carry a mutable in-memory tail that diverges compile vs capture (TT_FATAL
# unwarmed ``ttnn.concat`` / ``ttnn.copy``). ``GEMMA4_CHUNKED_PREFILL_TRACE=1``
# only enables the *generator* long-ISL multi-chunk replay path (seeded
# ``sp0_mc``/``sp1_mc``), not JIT ``sp1`` via ``can_enable_trace``. Sliding-tail
# stash + persistent rebind in ``attention/prefill.py`` keep eager remnant
# chunks coherent across vLLM chunk boundaries (#51186). Trim via
# ``GEMMA4_TRACE_PREFILL_SEQ_LENS``.
_DEFAULT_TRACE_PREFILL_SEQ_LENS = [128, 512, 1024, 2048, 4096]


def _resolve_trace_prefill_seq_lens() -> list[int]:
    override = os.environ.get("GEMMA4_TRACE_PREFILL_SEQ_LENS")
    if override is None:
        return list(_DEFAULT_TRACE_PREFILL_SEQ_LENS)
    return [int(x) for x in override.split(",") if x.strip()]


GEMMA4_TRACE_PREFILL_SEQ_LENS = _resolve_trace_prefill_seq_lens()

# Prefill trace is disabled above 4k ISL (no perf gain, OOM risk) and at or above 32k
# batched virtual tokens (batch_size × padded prefill length). Prefills above the
# cap are instead kept safe by generator-level chunking (see
# resolve_gemma4_prefill_chunk_size): an 8192 prompt runs as 4096-token chunks
# rather than one full-length op, so it neither wedges the fetch queue (#49083)
# nor OOMs a whole-length trace.
GEMMA4_MAX_TRACE_PREFILL_SEQ_LEN = 4096


def _resolve_max_trace_batched_prefill_tokens() -> int:
    """Virtual-token ceiling above which batched prefill drops to eager.

    Aligned with the shared planner's ``_MAX_BATCHED_PREFILL_TOKENS`` (128Ki in
    models/common/llm_runtime/prefill/plan.py). Gemma4 previously used 32Ki --
    4x tighter than the shared value, and a batch x seq gate no peer model has
    at all (tt_transformers ``can_enable_trace`` gates on seq_len only, per
    (model, device), via ``trace_prefill_supported_seq_lens``).

    That divergence was costly, not protective. Measured on P150x8 / 12B at
    conc=32, isl=2048: crossing 32Ki sent batch-32 prefill down the eager path
    at ~322 tok/s versus ~5817 tok/s traced -- an 18x regression, not the
    "no perf gain" the original comment assumed. Eager wide prefill also mixes
    with traced batch-1 at the same seq_len, which precedes the P150x8 wedge:
    at budget 65536 the device stalled at 0 tok/s, while the same width kept
    traced ran clean (TTFT 29278 -> 8555 ms, TPOT 281.3 -> 34.8 ms, 240k ISL
    unchanged).

    The trace region holds command buffers; the wide prefill's activation
    (~63 MB/chip for 32x2048 on 12B) lives in DRAM (3.98 GiB/bank x 8 on
    Blackhole), so the original OOM rationale did not apply to this region.

    Wormhole stays protected by the seq_len gate rather than this cap: its
    ``GEMMA4_TRACE_PREFILL_SEQ_LENS`` is pinned to [128] on the T3K nightly,
    so 2048-wide prefills are never trace-eligible there regardless of batch.
    """
    raw = os.environ.get("GEMMA4_MAX_TRACE_BATCHED_PREFILL_TOKENS")
    if raw is None:
        return 128 * 1024
    val = int(raw)
    return val if val > 0 else 10**9


GEMMA4_MAX_TRACE_BATCHED_PREFILL_TOKENS = _resolve_max_trace_batched_prefill_tokens()


def chunked_prefill_trace_enabled() -> bool:
    """True when long-ISL *generator* multi-chunk should replay 4k prefill traces.

    Set ``GEMMA4_CHUNKED_PREFILL_TRACE=1`` to measure / enable. Each generator
    chunk (default 4096) replays the matching ``sp0_mc``/``sp1_mc`` prefill
    trace instead of an eager ``ttnn_prefill_forward``. Does **not** authorize
    vLLM APC / remnant JIT ``sp1`` captures — those stay eager via
    :func:`can_gemma4_enable_prefill_trace`.
    """
    return os.environ.get("GEMMA4_CHUNKED_PREFILL_TRACE", "0").lower() in ("1", "true", "yes")


# Default generator-level prefill chunk when GEMMA4_GEN_PREFILL_CHUNK is unset,
# applied on QB2 ONLY (P150x4 or P300x2 = 1x4 Blackhole / 2x P300, the board
# this was validated on).
#
# Every other model bounds its prefill chunk: the shared tt_transformers/Qwen
# generator defaults max_prefill_chunk_size to a per-(model,device) value (4096
# for combos without a table entry, which is Gemma4's case), DeepSeek uses a
# tiered chunk table, and Qwen3.6 chunk-traces at 2048. Gemma4 alone used to
# default to a SINGLE full-length chunk (max_seq_len in the vLLM generator, a
# power-of-2 in the demo generator) — divergent from each other and from the
# reference, and exactly the unbounded full-sequence prefill op that wedges the
# fetch queue at ISL>=8192 (#49083) or OOMs a whole-length trace.
#
# 4096 is the largest chunk validated on QB2/P150x8 to both trace safely
# (<= GEMMA4_MAX_TRACE_PREFILL_SEQ_LEN) and clear the #49083 wedge
# (repro_prefill_hang.py, REPRO_ISLS=8192,8192 -> ALL_DONE, no wedge/OOM).
# WH T3K uses a tighter per-board ``prefill_chunk`` (2048) — see policy table.
GEMMA4_DEFAULT_PREFILL_CHUNK = 4096

# Per-(model, device) long-context policy.
#
# Keys: model_key from ``normalize_gemma4_model_key`` × ``determine_device_name()``.
#
#   unbounded_isl_max:        largest max_seq_len that fits with FULL/unbounded
#                             sliding KV (DRAM). Informational + sweep target.
#   bounded_isl_min:          max_seq_len >= this → auto-enable bounded sliding.
#                             Set above ``unbounded_isl_max`` when the model still
#                             fits unbounded at that ISL (e.g. 12B @ 128k).
#   chunked_bounded_isl_min:  bounded AND max_seq_len >= this → auto multi-chunk
#                             (single-chunk scratch OOM). Between bounded_isl_min
#                             and this: bounded + single-chunk.
#   prefill_chunk:            default generator / vLLM scheduler chunk size.
#   prefill_chunk_by_isl:     optional list of {isl_min, chunk, require_bounded?}
#                             overrides. Highest matching isl_min wins when
#                             max_seq_len >= isl_min (and bounded_sliding if
#                             require_bounded). Demo + vLLM share this table so
#                             server specs can keep max_num_batched_tokens /
#                             long_prefill_token_threshold aligned with the
#                             resolved chunk for that max_context.
#   source:                   "measured" | "inferred" | "placeholder"
#
# Env overrides still win: GEMMA4_BOUNDED_SLIDING, GEMMA4_GEN_PREFILL_CHUNK,
# GEMMA4_DEMO_SINGLE_CHUNK.
#
# QB2 (P150x4 / P300x2) sweeps: ``isl_sweep_logs/model_policy_sweep_v2`` (64k/128k)
# and follow-on 256k runs via ``run_long_context_sweeps.sh``.
# ``determine_device_name`` returns P150x4 for 4 BH dies; MESH_DEVICE / docs also
# use P300x2 (2x P300 = 4 dies). Treat both as the same QB2 policy entry.
_QB2 = "P150x4"
_QB2_ALIASES = frozenset({"P150x4", "P300x2", "P300X2"})
_CHUNK = GEMMA4_DEFAULT_PREFILL_CHUNK

# SKU names ``determine_device_name`` returns, split by architecture. A policy
# entry measured on one arch must never be reused on the other: Wormhole carries
# 12 GB per ASIC against Blackhole's 32 GB, so a Blackhole entry's DRAM headroom
# (e.g. QB2 "unbounded KV through 128k") silently OOMs a WH board. See
# ``get_gemma4_long_context_policy`` for how the fallback is gated.
_WH_DEVICES = frozenset({"N150", "N300", "N150x4", "T3K", "TG"})
_BH_DEVICES = frozenset({"P100", "P150", "P300", "P150x4", "P150x8", "P300x2", "P300X2", "BHGLX"})

GEMMA4_LONG_CONTEXT_POLICY = {
    # Dense 31B — measured on QB2 (isl_sweep_logs + llm.yaml max_context=49152 serve).
    "31B": {
        _QB2: {
            "unbounded_isl_max": 65536,  # demo ran ~65k unbounded; vLLM serve ~49k
            "bounded_isl_min": 65536,  # auto bounded at 64k+
            "chunked_bounded_isl_min": 262144,  # single-chunk ~5.6GB OOM at 256k
            # Default 4096; at ISL>=128k + bounded, 2048 (4096 → token-0 garbage).
            "prefill_chunk": _CHUNK,
            "prefill_chunk_by_isl": [
                {"isl_min": 131072, "chunk": 2048, "require_bounded": True},
            ],
            "source": "measured",
        },
        # P150x8: unbounded 128k multi-chunk (4096) still collapses to
        # "lapped lapped…" (full_matrix LB + 2026-07-27 repro). QB2 128k with
        # bounded + chunk=2048 is coherent — mirror that cutover here.
        # 256k still needs bounded for DRAM. Override: GEMMA4_BOUNDED_SLIDING=0/1.
        "P150x8": {
            "unbounded_isl_max": 65536,
            "bounded_isl_min": 131072,  # auto bounded at 128k+ for coherence
            "chunked_bounded_isl_min": 262144,  # DRAM: multi-chunk required at 256k
            "prefill_chunk": _CHUNK,
            "prefill_chunk_by_isl": [
                # Same as QB2: bounded @ ≥128k with chunk=4096 → token-0 garbage.
                {"isl_min": 131072, "chunk": 2048, "require_bounded": True},
            ],
            "source": "measured",
        },
        # WH LoudBox / QuietBox (T3K): 12 GB GDDR6 per ASIC (n150/n300) vs
        # BH P150 32 GB; T3K mesh 1×8 ≈ 96 GB total vs BH LB 256 GB / QB2 128 GB
        # (docs.tenstorrent.com wormhole + t3000 / quietbox specs). Hybrid-OFF
        # full-length KV + full-ISL prefill scratch OOM at max_model_len=32768
        # (vLLM nightly run 30291376571: banks ~full, chunk=32768). Keep multi-
        # chunk=2048 and auto-bound earlier than BH; serve ≤16k until hybrid KV.
        # Validated on a real WH T3K on this branch: 4k (TTFT ~8.5 s, 15.7 tok/s),
        # 32k bounded+chunk2048 (TTFT ~55 s, 14.8 tok/s) and 128k (TTFT ~188 s,
        # 12.9 tok/s) all PASS. The inferred cutovers held, so this is promoted
        # from "inferred" to "measured".
        "T3K": {
            # Unbounded measured through 32768. bounded_isl_min sits above it so
            # concurrent serving never enters the bounded sliding remap, which is
            # broken for >1 request (see the 12B/T3K entry for the mechanism).
            "unbounded_isl_max": 32768,
            "bounded_isl_min": 65536,
            "chunked_bounded_isl_min": 65536,
            "prefill_chunk": 2048,
            "prefill_chunk_by_isl": [],
            "source": "measured",
        },
    },
    # Dense 12B — HF max_pos=256k. QB2: unbounded 64k+128k PASSED; unbounded 256k OOM.
    # P150x8: unbounded 64k+128k+256k PASSED (isl_sweep_logs/p150x8_bg_lb).
    # Single P150: unbounded 32k OK; 64k+ unbounded KV OOM (~22GB sliding pool) —
    # auto-bound + multi-chunk (4096) through full HF 256k (measured PASS + coherent
    # gen in isl_sweep_logs/full_matrix N150/P150 64k/128k/256k reruns).
    "12B": {
        _QB2: {
            "unbounded_isl_max": 131072,
            "bounded_isl_min": 262144,  # auto bounded at 256k (unbounded OOM)
            "chunked_bounded_isl_min": 262144,
            "prefill_chunk": _CHUNK,
            "source": "measured",
        },
        # Single P150: full ISL 256k via bounded sliding + chunked prefill.
        "P150": {
            "unbounded_isl_max": 32768,
            "bounded_isl_min": 65536,
            "chunked_bounded_isl_min": 65536,
            "prefill_chunk": _CHUNK,
            "source": "measured",
        },
        "P150x8": {
            "unbounded_isl_max": 262144,
            "bounded_isl_min": 524288,  # beyond measured 256k
            "chunked_bounded_isl_min": 524288,
            "prefill_chunk": _CHUNK,
            "source": "measured",
        },
        # WH T3K (1x8, ~96 GB). Measured unbounded through 131072 (32k / 64k /
        # 128k all PASS on this branch); bounded + multi-chunk 2048 additionally
        # reaches the full 256k ISL at batch-1 (TTFT ~468 s, 14.3 tok/s).
        #
        # bounded_isl_min is deliberately set ABOVE the measured unbounded
        # ceiling so normal serving never enters bounded mode. Bounded sliding
        # remaps sliding page tables to dense per-row block IDs
        # (``_pad_sliding_page_tables_for_bounded``) keyed on the row index of
        # the current page-table tensor rather than the request's persistent KV
        # slot, so with more than one concurrent request a request's sliding
        # blocks move between steps and it reads another user's KV — measured as
        # nondeterministic garbage from concurrency 2 upward. Unbounded at the
        # same context is clean at concurrency 32 (32/32 correct). Lower this
        # again once that remap is keyed on a stable slot.
        "T3K": {
            "unbounded_isl_max": 131072,
            "bounded_isl_min": 262144,
            "chunked_bounded_isl_min": 262144,
            "prefill_chunk": 2048,
            "source": "measured",
        },
        # WH N300 (1x2, 24 GB): the only Gemma4 variant that fits a single WH
        # card. Measured — 4k unbounded PASS; 32k unbounded OOMs (DRAM); bounded
        # + chunk 2048 PASSES 32k / 64k / 128k; 256k OOMs on KV allocation.
        # Serve at most 128k here.
        "N300": {
            "unbounded_isl_max": 8192,
            "bounded_isl_min": 16384,
            "chunked_bounded_isl_min": 16384,
            "prefill_chunk": 2048,
            "source": "measured",
        },
    },
    # MoE 26B-A4B — HF max_pos=256k. QB2: unbounded 64k PASSED (after instruct-clip
    # trim fix); 128k allocated/ran (no OOM, prior 1800s timeout); unbounded 256k OOM.
    # Bounded+chunked 256k PASSED (~72m prefill / TTFT~4345s at chunk=4096; needs
    # TIMEOUT_256K>=7200 — 3600s timed out mid-prefill).
    # P150x8: unbounded 64k+128k PASSED (~53m at 128k); unbounded 256k bus-error;
    # bounded single-chunk 256k PASSED (~131m TTFT).
    "26B-A4B": {
        _QB2: {
            "unbounded_isl_max": 131072,
            "bounded_isl_min": 262144,  # auto bounded at 256k (unbounded OOM)
            "chunked_bounded_isl_min": 262144,
            "prefill_chunk": _CHUNK,
            "source": "measured",
        },
        # P150x8: same coherency cutover as 31B (unbounded 128k → garbage).
        "P150x8": {
            "unbounded_isl_max": 65536,
            "bounded_isl_min": 131072,
            "chunked_bounded_isl_min": 262144,
            "prefill_chunk": _CHUNK,
            "prefill_chunk_by_isl": [
                {"isl_min": 131072, "chunk": 2048, "require_bounded": True},
            ],
            "source": "measured",
        },
        # WH T3K (1x8): measured — bounded + chunk 2048 PASSES 4k / 32k / 128k
        # and stays coherent. Functional ceiling is 128k, but MoE prefill on
        # Wormhole is far slower than the dense 31B (TTFT ~449 s @32k and
        # ~1506 s @128k vs 31B's ~55 s @32k), so serve specs should stay at
        # 32k — this table only bounds what fits, not what is fast.
        "T3K": {
            # Unbounded measured through 32768; bounded kept out of the serving
            # range (multi-request sliding remap bug — see 12B/T3K).
            "unbounded_isl_max": 32768,
            "bounded_isl_min": 65536,
            "chunked_bounded_isl_min": 65536,
            "prefill_chunk": 2048,
            "source": "measured",
        },
    },
    # MatFormer E4B — HF max_pos=128k native; demo can force higher. QB2: unbounded
    # 64k+128k+256k PASSED. P150x8: same (isl_sweep_logs/p150x8_bg_lb).
    # Single P150: unbounded through 256k measured (full_matrix N150 alias).
    "E4B": {
        _QB2: {
            "unbounded_isl_max": 262144,
            "bounded_isl_min": 524288,  # beyond measured 256k
            "chunked_bounded_isl_min": 524288,
            "prefill_chunk": _CHUNK,
            "source": "measured",
        },
        "P150": {
            "unbounded_isl_max": 262144,
            "bounded_isl_min": 524288,
            "chunked_bounded_isl_min": 524288,
            "prefill_chunk": _CHUNK,
            "source": "measured",
        },
        "P150x8": {
            "unbounded_isl_max": 262144,
            "bounded_isl_min": 524288,
            "chunked_bounded_isl_min": 524288,
            "prefill_chunk": _CHUNK,
            "source": "measured",
        },
    },
    # MatFormer E2B — HF max_pos=128k native; demo can force higher. QB2: unbounded
    # 64k+128k+256k PASSED. P150x8: same (isl_sweep_logs/p150x8_bg_lb).
    # Also use_double_wide_mlp on KV-shared layers (2× intermediate).
    # Prefer multi-chunk (4096): single-chunk 64k+ warmup can hang on P150x8.
    # Single P150: unbounded through 256k measured (full_matrix N150 alias).
    "E2B": {
        _QB2: {
            "unbounded_isl_max": 262144,
            "bounded_isl_min": 524288,  # beyond measured 256k
            "chunked_bounded_isl_min": 524288,
            "prefill_chunk": _CHUNK,
            "source": "measured",
        },
        "P150": {
            "unbounded_isl_max": 262144,
            "bounded_isl_min": 524288,
            "chunked_bounded_isl_min": 524288,
            "prefill_chunk": _CHUNK,
            "source": "measured",
        },
        "P150x8": {
            "unbounded_isl_max": 262144,
            "bounded_isl_min": 524288,
            "chunked_bounded_isl_min": 524288,
            "prefill_chunk": _CHUNK,
            "source": "measured",
        },
    },
}

# Wormhole board with no measured (model, device) entry. Deliberately
# conservative: bound the sliding KV early and keep the prefill chunk small,
# because WH carries 12 GB per ASIC (N300 1x2 = 24 GB, T3K 1x8 = 96 GB) against
# Blackhole QB2's 128 GB / LoudBox's 256 GB. ``source`` starts with "inferred"
# so ``resolve_gemma4_prefill_chunk_size`` still honours ``prefill_chunk``
# instead of degrading to a single full-length chunk.
_WH_DEFAULT_LONG_CONTEXT_POLICY = {
    "unbounded_isl_max": 8192,
    "bounded_isl_min": 16384,
    "chunked_bounded_isl_min": 16384,
    "prefill_chunk": 2048,
    "source": "inferred_wormhole_default",
}

# Unknown model: do not force 31B's aggressive bounded cutover.
_DEFAULT_LONG_CONTEXT_POLICY = {
    "unbounded_isl_max": 131072,
    "bounded_isl_min": 262144,
    "chunked_bounded_isl_min": 262144,
    "prefill_chunk": _CHUNK,
    "source": "default_unknown_model",
}


def normalize_gemma4_model_key(model_name_or_path) -> str:
    """Map HF id / path / base name → policy key (31B, 12B, 26B-A4B, E4B, E2B)."""
    name = str(model_name_or_path or "").lower().replace("_", "-")
    # vLLM often passes the resolved HF snapshot dir
    # (.../models--google--gemma-4-31b-it/snapshots/<hash>); the last path
    # component is then the hash, so search the full string.
    if "31b" in name:
        return "31B"
    if "12b" in name:
        return "12B"
    if "26b" in name or "a4b" in name:
        return "26B-A4B"
    if "e4b" in name:
        return "E4B"
    if "e2b" in name:
        return "E2B"
    return "unknown"


def _device_name(mesh_device) -> str | None:
    if mesh_device is None:
        return None
    try:
        from models.tt_transformers.tt.model_config import determine_device_name

        return determine_device_name(mesh_device)
    except Exception:
        return None


def _host_is_wormhole() -> bool:
    """True when the running host is Wormhole. Best-effort / never raises.

    Used only to disambiguate the historical ``MESH_DEVICE=N150`` tag, which
    older Blackhole sweeps used for a single P150. Unknown arch behaves as
    before (Blackhole), so this cannot change Blackhole resolution.
    """
    try:
        import ttnn

        return "wormhole" in str(ttnn.get_arch_name()).lower()
    except Exception:
        return False


def _device_arch_family(device: str | None) -> str | None:
    """``"wh"`` / ``"bh"`` for a known SKU name, else ``None``."""
    if device is None:
        return None
    if device in _WH_DEVICES:
        return "wh"
    if device in _BH_DEVICES:
        return "bh"
    return None


def _canonical_device_name(device: str | None) -> str | None:
    """Map device aliases onto canonical policy keys (QB2 / single P150)."""
    if device is None:
        return None
    if device in _QB2_ALIASES:
        return _QB2
    # Historical Blackhole sweeps tagged the single P150 as "N150" via
    # MESH_DEVICE. Only honour that alias on a Blackhole host — on Wormhole,
    # "N150" is a real 1x1 WH board (determine_device_name returns it) and must
    # not inherit Blackhole DRAM headroom.
    if device in ("N150", "n150") and not _host_is_wormhole():
        return "P150"
    return device


def get_gemma4_long_context_policy(mesh_device=None, model_name_or_path=None) -> dict:
    """Return long-context policy for ``(model_key, device)``."""
    model_key = normalize_gemma4_model_key(model_name_or_path)
    # Prefer live mesh; fall back to MESH_DEVICE so server/vLLM config-time
    # resolution (before mesh open) still picks the right board entry.
    device = _canonical_device_name(_device_name(mesh_device) or os.environ.get("MESH_DEVICE")) or _QB2
    by_model = GEMMA4_LONG_CONTEXT_POLICY.get(model_key)
    family = _device_arch_family(device)
    if by_model is not None:
        if device in by_model:
            return dict(by_model[device])
        # Fall back to the QB2 entry for this model, but only on Blackhole.
        # Reusing a Blackhole entry on Wormhole hands a 24 GB N300 / 96 GB T3K
        # the 128 GB QB2 headroom (unbounded KV through 128k) and OOMs on
        # allocation, so WH takes the conservative WH default instead.
        if _QB2 in by_model and family != "wh":
            policy = dict(by_model[_QB2])
            policy["source"] = f"{policy.get('source', 'inferred')}_device_fallback"
            return policy
    if family == "wh":
        policy = dict(_WH_DEFAULT_LONG_CONTEXT_POLICY)
        logger.warning(
            f"No measured Gemma4 long-context policy for model={model_key} on {device}; "
            f"using conservative Wormhole defaults (bounded_isl_min={policy['bounded_isl_min']}, "
            f"prefill_chunk={policy['prefill_chunk']}). Set GEMMA4_BOUNDED_SLIDING / "
            f"GEMMA4_GEN_PREFILL_CHUNK to override."
        )
        return policy
    policy = dict(_DEFAULT_LONG_CONTEXT_POLICY)
    if model_key == "unknown":
        logger.warning(
            f"No Gemma4 long-context policy for model={model_name_or_path!r}; "
            f"using unbounded-friendly defaults (bounded_isl_min={policy['bounded_isl_min']})."
        )
    return policy


def should_auto_enable_bounded_sliding(max_seq_len: int, mesh_device=None, model_name_or_path=None) -> bool:
    """True when demo/generator should auto-enable bounded sliding for this ISL."""
    policy = get_gemma4_long_context_policy(mesh_device, model_name_or_path)
    return max_seq_len >= int(policy["bounded_isl_min"])


def resolve_gemma4_bounded_sliding(
    max_seq_len: int,
    mesh_device=None,
    model_name_or_path=None,
    *,
    paged_attention: bool = True,
) -> bool:
    """Demo/vLLM shared bounded-sliding resolution (policy + env override).

    ``GEMMA4_BOUNDED_SLIDING`` unset → ``should_auto_enable_bounded_sliding``.
    Set to 1/true/yes → force on; any other value → force off.
    Always requires paged attention.
    """
    _bs_env = os.environ.get("GEMMA4_BOUNDED_SLIDING")
    if _bs_env is None:
        bounded = should_auto_enable_bounded_sliding(max_seq_len, mesh_device, model_name_or_path)
    else:
        bounded = _bs_env.lower() in ("1", "true", "yes")
    return bool(bounded and paged_attention)


def should_auto_enable_chunked_bounded(
    max_seq_len: int, mesh_device=None, model_name_or_path=None, *, bounded_sliding: bool = False
) -> bool:
    """True when bounded long-context should auto multi-chunk for DRAM fit."""
    if not bounded_sliding:
        return False
    policy = get_gemma4_long_context_policy(mesh_device, model_name_or_path)
    return max_seq_len >= int(policy["chunked_bounded_isl_min"])


def resolve_gemma4_demo_long_context(
    max_seq_len: int,
    mesh_device=None,
    model_name_or_path=None,
    *,
    paged_attention: bool = True,
    non_qb2_default=None,
) -> dict:
    """Resolve bounded + prefill chunk for demos (MESH_DEVICE / HF model aware).

    Returns keys: bounded_sliding, prefill_chunk, needs_chunked_bounded, policy_source.
    Both ``text_demo`` and ``text_demo_v2`` should use this so default commands
    pick the same coherency/perf cutovers without extra env knobs.
    """
    if non_qb2_default is None:
        non_qb2_default = GEMMA4_DEFAULT_PREFILL_CHUNK
    bounded = resolve_gemma4_bounded_sliding(
        max_seq_len, mesh_device, model_name_or_path, paged_attention=paged_attention
    )
    chunk = resolve_gemma4_prefill_chunk_size(
        max_seq_len,
        mesh_device=mesh_device,
        non_qb2_default=non_qb2_default,
        model_name_or_path=model_name_or_path,
        bounded_sliding=bounded,
    )
    policy = get_gemma4_long_context_policy(mesh_device, model_name_or_path)
    return {
        "bounded_sliding": bounded,
        "prefill_chunk": int(chunk),
        "needs_chunked_bounded": should_auto_enable_chunked_bounded(
            max_seq_len, mesh_device, model_name_or_path, bounded_sliding=bounded
        ),
        "policy_source": str(policy.get("source", "")),
    }


def _is_qb2(mesh_device) -> bool:
    """True only for the QB2 board (P150x4 or P300x2, 1x4 Blackhole)."""
    name = _device_name(mesh_device) or os.environ.get("MESH_DEVICE")
    return name in _QB2_ALIASES or _canonical_device_name(name) == _QB2


def _prefill_chunk_isl_tiers(policy: dict) -> list[dict]:
    """Normalize ``prefill_chunk_by_isl`` (plus legacy bounded_* keys)."""
    tiers = policy.get("prefill_chunk_by_isl")
    if tiers:
        return [dict(t) for t in tiers]
    # Legacy keys from earlier QB2 128k coherency wiring.
    bchunk = policy.get("bounded_prefill_chunk")
    bmin = policy.get("bounded_prefill_chunk_isl_min")
    if bchunk is not None and bmin is not None:
        return [{"isl_min": int(bmin), "chunk": int(bchunk), "require_bounded": True}]
    return []


def resolve_gemma4_prefill_chunk_size(
    max_seq_len: int,
    mesh_device=None,
    non_qb2_default=None,
    model_name_or_path=None,
    *,
    bounded_sliding: bool = False,
) -> int:
    """Generator-level prefill chunk size for demo + vLLM serving.

    ``GEMMA4_GEN_PREFILL_CHUNK`` (a 2048-multiple) overrides on any board.
    Otherwise the per-(model, device) ``prefill_chunk`` default applies when the
    policy is measured (incl. P150x8) or the board is QB2, then
    ``prefill_chunk_by_isl`` may select a smaller chunk for high ISL (e.g. QB2
    31B bounded @ ≥128k → 2048 for coherency). Other boards keep
    ``non_qb2_default`` (often ``max_seq_len``) so unvalidated configs stay
    single-chunk unless the caller passes 4096.

    Server specs should set vLLM ``max_num_batched_tokens`` /
    ``long_prefill_token_threshold`` to this same resolved value for the
    configured ``max_context``.

    P150x8 / 31B: unbounded chunk=4096 is fast (~31s TTFT) but quality collapses
    at 128k; bounded + chunk=2048 (same tier as QB2) is the coherency path.
    """
    override = int(os.environ.get("GEMMA4_GEN_PREFILL_CHUNK", "0"))
    if override > 0:
        return override
    policy = get_gemma4_long_context_policy(mesh_device, model_name_or_path)
    source = str(policy.get("source", ""))
    # measured / placeholder / inferred entries (and any board with ISL tiers)
    # share the policy chunk table so demo defaults stay coherent without env.
    use_policy_chunk = (
        _is_qb2(mesh_device)
        or source.startswith(("measured", "placeholder", "inferred"))
        or bool(_prefill_chunk_isl_tiers(policy))
    )
    if use_policy_chunk:
        chunk = int(policy["prefill_chunk"])
        for tier in sorted(
            _prefill_chunk_isl_tiers(policy),
            key=lambda t: int(t["isl_min"]),
            reverse=True,
        ):
            if max_seq_len < int(tier["isl_min"]):
                continue
            if tier.get("require_bounded", False) and not bounded_sliding:
                continue
            chunk = int(tier["chunk"])
            break
        return min(chunk, max_seq_len)
    return non_qb2_default if non_qb2_default is not None else max_seq_len


def model_uses_pli(model) -> bool:
    """True for E2B/E4B-style models with per-layer-input embeddings."""
    return bool(getattr(model, "hidden_size_per_layer_input", 0))


def can_gemma4_enable_prefill_trace(
    prefill_seq_len: int,
    *,
    batch_size: int = 1,
    num_cached_tokens: int = 0,
    uses_pli: bool = False,
) -> bool:
    """Return True when Gemma4 prefill device trace may be captured or replayed.

    Cold single-chunk / first scheduler grant only (``num_cached_tokens == 0``).
    APC hits and vLLM chunked-prefill remnants (``num_cached_tokens > 0``) stay
    eager — same posture as tt_transformers #32056 and required for Gemma4
    hybrid sliding tails (compile/capture graph must not depend on mutable
    ``sliding_tail_in`` / persistent ring state). Long-ISL traced multi-chunk
    bypasses this gate via :func:`chunked_prefill_trace_enabled` inside the
    generator chunk loop.
    """
    if uses_pli:
        return False
    # Never JIT-capture / replay sp1 through the shared can_enable_trace path.
    # Remnant / APC continuations are eager; generator multi-chunk uses its own
    # seeded sp0_mc/sp1_mc path when GEMMA4_CHUNKED_PREFILL_TRACE=1.
    if num_cached_tokens != 0:
        return False
    if prefill_seq_len > GEMMA4_MAX_TRACE_PREFILL_SEQ_LEN:
        return False
    if prefill_seq_len not in GEMMA4_TRACE_PREFILL_SEQ_LENS:
        return False
    if batch_size * prefill_seq_len >= GEMMA4_MAX_TRACE_BATCHED_PREFILL_TOKENS:
        return False
    return True


def apply_gemma4_prefill_trace_policy(
    enable_trace: bool,
    prefill_seq_len: int,
    batch_size: int,
    model,
) -> bool:
    """Apply Gemma4 prefill trace limits; log and return False when trace is disabled."""
    if not enable_trace:
        return False
    if can_gemma4_enable_prefill_trace(
        prefill_seq_len,
        batch_size=batch_size,
        uses_pli=model_uses_pli(model),
    ):
        return True
    if prefill_seq_len > GEMMA4_MAX_TRACE_PREFILL_SEQ_LEN:
        logger.info(
            "Disabling prefill trace for seq_len={}: above {} ISL (no perf gain, OOM risk)",
            prefill_seq_len,
            GEMMA4_MAX_TRACE_PREFILL_SEQ_LEN,
        )
    elif batch_size * prefill_seq_len >= GEMMA4_MAX_TRACE_BATCHED_PREFILL_TOKENS:
        logger.info(
            "Disabling prefill trace for batch_size={} seq_len={}: "
            "{}+ batched virtual tokens (no perf gain, OOM risk)",
            batch_size,
            prefill_seq_len,
            GEMMA4_MAX_TRACE_BATCHED_PREFILL_TOKENS,
        )
    else:
        logger.info(
            "Disabling prefill trace for batch_size={} seq_len={}: not eligible for capture",
            batch_size,
            prefill_seq_len,
        )
    return False


def resolve_gemma4_prefill_trace_enable(
    enable_trace: bool,
    model,
    model_args,
    *,
    batch_size: int,
    prefill_seq_lens: list[int],
    can_batch_prefill: bool,
    empty_slots=None,
) -> bool:
    """Resolve whether prefill trace stays enabled for this batch/prefill shape."""
    # Bounded sliding: TRACE capture uses get_last_token=-1 so attention takes the
    # mid-forward ``paged_fill_cache`` branch (valid_seq_len is None). That fill
    # before lm_head corrupts token-0 on TP — the same hazard documented on the
    # eager deferred-fill path (flush only after lm_head). Disable TRACE for all
    # bounded prefills (B=1 short nightly prompts included); eager deferred-fill
    # remains. Batched+bounded also cannot refresh the scalar fill-cap tensor.
    if getattr(model, "bounded_sliding_kv_cache", False) and (
        os.environ.get("GEMMA4_ALLOW_BOUNDED_PREFILL_TRACE", "0").lower() not in ("1", "true", "yes")
    ):
        if enable_trace:
            logger.info(
                "Disabling prefill trace for bounded sliding "
                "(TRACE mid-forward paged_fill_cache corrupts token-0 on TP; "
                "eager deferred-fill path remains)."
            )
        return False
    trace_batch_size = batch_size
    if can_batch_prefill:
        # Must match the padded_batch the batched prefill will actually run, which
        # spans the physical slots rather than just the request count.
        trace_batch_size = batched_prefill_padded_batch(batch_size, empty_slots, model_args.max_batch_size)
    return apply_gemma4_prefill_trace_policy(
        enable_trace,
        prefill_seq_lens[0],
        trace_batch_size,
        model,
    )


def patch_gemma4_trace_model_args(model_args, *, prefill_trace_enabled: bool = True) -> None:
    """Configure trace_prefill_supported_seq_lens and can_enable_trace on model_args."""
    if prefill_trace_enabled:
        model_args.trace_prefill_supported_seq_lens = [
            length for length in GEMMA4_TRACE_PREFILL_SEQ_LENS if length <= GEMMA4_MAX_TRACE_PREFILL_SEQ_LEN
        ]
        uses_pli = bool(getattr(model_args, "hidden_size_per_layer_input", 0))

        def _can_enable_trace(prefill_seq_len, num_cached_tokens=0, batch_size=1):
            return can_gemma4_enable_prefill_trace(
                prefill_seq_len,
                batch_size=batch_size,
                num_cached_tokens=num_cached_tokens,
                uses_pli=uses_pli,
            )

        model_args.can_enable_trace = _can_enable_trace
    else:
        model_args.trace_prefill_supported_seq_lens = []
        model_args.can_enable_trace = lambda prefill_seq_len, num_cached_tokens=0, batch_size=1: False


def maybe_disable_pli_prefill_trace(enable_trace: bool, model, batch_size: int = 1) -> bool:
    """Return False when PLI prefill must not use trace capture.

    PLI prefill uploads per-layer inputs via ttnn.from_torch inside forward, which
    triggers TT_FATAL during trace capture. Decode trace is unaffected.
    """
    if enable_trace and model_uses_pli(model):
        logger.info(
            "Disabling prefill trace on PLI model (batch_size={}): "
            "in-forward ttnn.from_torch PLI upload is incompatible with trace capture",
            batch_size,
        )
        return False
    return enable_trace


def skip_gemma4_full_prefill_warmup(generator) -> None:
    """Skip the full batch×ISL prefill warmup sweep on the next ``prefill_forward_text`` call."""
    generator.already_warmed_up_prefill = True


def warmup_gemma4_prefill_bucket(
    generator,
    kv_cache,
    *,
    enable_trace: bool,
    **prefill_kwargs,
) -> None:
    """Compile or capture prefill trace for one bucket only (no full warmup matrix).

    Used by parity/perf tests that exercise a single ``(batch_size, prefill_seq_len)``
    combination. Production/demo startup should still call
    :func:`warmup_gemma4_model_prefill` for the full sweep.
    """
    skip_gemma4_full_prefill_warmup(generator)
    generator.prefill_forward_text(
        **prefill_kwargs,
        kv_cache=kv_cache,
        enable_trace=maybe_disable_pli_prefill_trace(enable_trace, generator.model[0]),
        warmup_prefill=False,
    )


def warmup_gemma4_batched_prefill_traces(
    generator,
    kv_cache,
    *,
    enable_trace: bool,
    can_sample_on_device,
    greedy_only: bool = False,
    prefill_forward_fn=None,
) -> None:
    """Capture prefill traces for MoE models across batch sizes and trace ISLs.

    Matches tt_transformers ``Generator.warmup_model_prefill``: warm **batch=1**
    only for each trace-eligible ISL ≤ chunk (≤4096). Runtime may still pack
    multiple users into one prefill step; that path does not need a dedicated
    B>1 warmup capture. Longer prompts are covered by generator / vLLM
    chunked prefill (same chunk size), not by warming every batch×ISL combo.

    Override with ``GEMMA4_WARMUP_PREFILL_BATCHES=1,2,4`` only if a demo needs
    explicit B>1 trace capture (not the server product path).

    ``prefill_forward_fn`` selects the entry point used for each capture. It
    defaults to ``generator.prefill_forward_text`` (demo / uniform-page-table
    path). The vLLM hybrid bridge passes ``generator.prefill_forward`` so the
    capture runs *through* the per-layer page-table routing and binds the
    traced paged ops to the persistent per-layer buffers — exactly how decode
    warmup binds via ``decode_forward``. Pre-capturing the prefill buckets at
    warmup (before any traced decode) keeps runtime prefills to trace *replay*,
    avoiding the #49083 cold-eager-capture fetch-queue wedge.
    """
    if generator.already_warmed_up_prefill:
        return
    generator.already_warmed_up_prefill = True

    prefill_forward = prefill_forward_fn if prefill_forward_fn is not None else generator.prefill_forward_text

    model_args = generator.model_args[0]
    sequence_lengths_to_warmup = model_args.get_warmup_prefill_supported_seq_lens()
    trace_isls = set(model_args.trace_prefill_supported_seq_lens)
    max_batch_size = model_args.max_batch_size
    # Cap at the B≤4 hang ceiling (see GEMMA4_MAX_BATCHED_PREFILL_USERS).
    from models.demos.gemma4.tt.generator import max_batched_prefill_users

    user_cap = max_batched_prefill_users()
    override = os.environ.get("GEMMA4_WARMUP_PREFILL_BATCHES")
    if override:
        warmup_batch_sizes = tuple(
            b
            for b in sorted({int(x) for x in override.split(",") if x.strip()})
            if 1 <= b <= max_batch_size and b <= user_cap and b in SUPPORTED_PREFILL_BATCH_SIZES
        )
        if not warmup_batch_sizes:
            warmup_batch_sizes = (1,)
    else:
        # Same as tt_transformers: batch-1-only traced prefill warmup.
        warmup_batch_sizes = (1,)

    if warmup_batch_sizes == (1,):
        logger.info(
            "Using batch-1-only traced prefill warmup; runtime batched prefill "
            "remains enabled. Trace ISLs={} (user_cap={})",
            sorted(trace_isls),
            user_cap,
        )
    else:
        logger.info(
            "Gemma4 traced prefill warmup (GEMMA4_WARMUP_PREFILL_BATCHES override): "
            "batches={} x trace ISLs {} (user_cap={})",
            warmup_batch_sizes,
            sorted(trace_isls),
            user_cap,
        )

    skip_sequence_lengths = False
    sampling_parameters_sweeped = False

    for model_id in range(generator.data_parallel):
        for supported_length in sequence_lengths_to_warmup:
            if supported_length not in trace_isls:
                continue
            if model_id != 0 and (supported_length not in trace_isls or not enable_trace):
                continue

            for batch_size in warmup_batch_sizes:
                if batch_size * supported_length >= MAX_BATCHED_PREFILL_SEQ_LEN:
                    logger.info(
                        "Skipping batched prefill trace warmup for batch_size={}, seq_len={}: "
                        "exceeds {} token limit",
                        batch_size,
                        supported_length,
                        MAX_BATCHED_PREFILL_SEQ_LEN,
                    )
                    continue

                warmup_args = generator._mock_tokens(batch_size, supported_length, kv_cache, model_id)

                if warmup_args["page_table"] is None and max_prefill_chunk_size_cutoff(
                    supported_length, model_args.max_prefill_chunk_size
                ):
                    logger.warning(
                        "Skipping warmup for sequence lengths after: {} because they are greater than "
                        "the max prefill chunk size and paged attention is disabled",
                        supported_length,
                    )
                    skip_sequence_lengths = True
                    break

                if not sampling_parameters_sweeped:
                    sampling_params = generator._create_sampling_params(
                        can_sample_on_device=can_sample_on_device,
                        greedy_only=greedy_only,
                        batch_size=batch_size,
                    )
                else:
                    sampling_params = [None]

                capture_trace = apply_gemma4_prefill_trace_policy(
                    enable_trace,
                    supported_length,
                    batch_size,
                    generator.model[model_id],
                )

                for param in sampling_params:
                    if capture_trace:
                        logger.info(
                            "Warming up prefill trace for sequence length: {} batch size: {} "
                            "with sampling params: {}",
                            supported_length,
                            batch_size,
                            param,
                        )
                    else:
                        logger.info(
                            "Warming up prefill (trace off) for sequence length: {} batch size: {} "
                            "with sampling params: {}",
                            supported_length,
                            batch_size,
                            param,
                        )
                    prefill_forward(
                        **warmup_args,
                        kv_cache=kv_cache,
                        enable_trace=capture_trace,
                        model_id_warmup=model_id,
                        sampling_params=param,
                    )

                sampling_parameters_sweeped = True

            if skip_sequence_lengths:
                break

        if skip_sequence_lengths:
            break

    if getattr(model_args, "is_multimodal", False):
        vision_chunk_size = getattr(model_args, "vision_chunk_size", 896)
        vision_channels = getattr(model_args, "vision_in_channels", 3)
        model_id = 0
        warmup_pixel_values = [torch.zeros((1, vision_channels, vision_chunk_size, vision_chunk_size))]
        prefill_forward_args = generator._mock_tokens(1, 128, kv_cache, model_id)

        logger.info("Warming up vision encoder with image size {}x{}", vision_chunk_size, vision_chunk_size)
        prefill_forward(
            **prefill_forward_args,
            kv_cache=kv_cache,
            enable_trace=False,
            model_id_warmup=model_id,
            sampling_params=None,
            pixel_values=warmup_pixel_values,
            image_sizes=[(vision_chunk_size, vision_chunk_size)],
        )
        logger.info("Vision encoder warmup completed")


def warmup_gemma4_model_prefill(
    generator,
    kv_cache,
    *,
    enable_trace,
    can_sample_on_device,
    greedy_only: bool = False,
    prefill_forward_fn=None,
) -> None:
    """Shared prefill warmup for standalone and vLLM Gemma4 generators.

    ``prefill_forward_fn`` (vLLM hybrid bridge only) routes the trace capture
    through the per-layer page-table path; see
    :func:`warmup_gemma4_batched_prefill_traces`.

    When ``GEMMA4_CHUNKED_PREFILL_TRACE`` is on, also warms an 2×chunk multi-chunk
    prefill so the ``sp1`` (middle-chunk) 4k trace is captured before the first
    long-ISL request.
    """
    enable_trace = maybe_disable_pli_prefill_trace(enable_trace, generator.model[0])
    if enable_trace:
        warmup_gemma4_batched_prefill_traces(
            generator,
            kv_cache,
            enable_trace=enable_trace,
            can_sample_on_device=can_sample_on_device,
            greedy_only=greedy_only,
            prefill_forward_fn=prefill_forward_fn,
        )
        # Once-only: tt_transformers calls warmup_model_prefill on *every*
        # prefill (warmup_prefill=True). The batched helper early-returns via
        # already_warmed_up_prefill, but this 8192 sp1 capture used to re-run
        # and add ~1.4s to every request TTFT.
        if chunked_prefill_trace_enabled() and not getattr(generator, "_warmed_chunked_prefill_sp1", False):
            chunk = int(getattr(generator.model_args[0], "max_prefill_chunk_size", GEMMA4_DEFAULT_PREFILL_CHUNK))
            chunk = min(chunk, GEMMA4_MAX_TRACE_PREFILL_SEQ_LEN)
            if chunk > 0:
                # Two chunks → captures/replays sp0 then captures sp1 at ``chunk``.
                multi_len = chunk * 2
                logger.info(
                    "Warming up traced multi-chunk prefill (sp1): {} tokens in {}-token chunks",
                    multi_len,
                    chunk,
                )
                prefill_forward = (
                    prefill_forward_fn if prefill_forward_fn is not None else generator.prefill_forward_text
                )
                warmup_args = generator._mock_tokens(1, multi_len, kv_cache, 0)
                prefill_forward(
                    **warmup_args,
                    kv_cache=kv_cache,
                    enable_trace=True,
                    model_id_warmup=0,
                    sampling_params=None,
                    warmup_prefill=False,
                )
            generator._warmed_chunked_prefill_sp1 = True
        return

    # Eager (non-traced) warmup for long-ISL demos (prefill trace gated off).
    # Skip the stock 32/128/512/1024/2048/4096 sweep — it only matters for
    # trace capture. Warm a short length (+ chunk size) once.
    #
    # Important: do NOT run the chunk-sized prefill with on-device SamplingParams.
    # Stock Generator only compiles sampling on the first short bucket, then uses
    # sampling_params=None for longer lengths. Pairing SamplingParams with the
    # 4096 eager warmup hung indefinitely on 31B/P150x8 (256k bounded).
    #
    # Optional: warm max_batch×128 when GEMMA4_WARMUP_PREFILL_BATCHES lists B>1
    # (demo-only; product/server uses batch-1 like tt_transformers).
    if getattr(generator, "already_warmed_up_prefill", False):
        return
    generator.already_warmed_up_prefill = True

    chunk = int(getattr(generator.model_args[0], "max_prefill_chunk_size", GEMMA4_DEFAULT_PREFILL_CHUNK))
    max_seq = int(getattr(generator.model_args[0], "max_seq_len", chunk) or chunk)
    # Never warm a length whose padded prefill bucket exceeds max_seq_len.
    # e.g. chunk=49152 → get_padded_prefill_len=65536 > pool → RoPE slice FATAL.
    from models.tt_transformers.tt.common import get_padded_prefill_len

    # Cap eager compile lengths at the policy chunk (same as traced path /
    # tt_transformers capped_warmup_seq_len). Longer ISL is chunked at runtime.
    chunk = min(chunk, max_seq, GEMMA4_MAX_TRACE_PREFILL_SEQ_LEN)
    if chunk > 0 and get_padded_prefill_len(chunk) > max_seq:
        chunk = 1 << max(max_seq.bit_length() - 1, 11)
        chunk = min(chunk, max_seq, GEMMA4_MAX_TRACE_PREFILL_SEQ_LEN)
    # GEMMA4_TRACE_PREFILL_SEQ_LENS historically only trimmed the *traced* bucket
    # set. PLI / full-ISL single-chunk boots take this eager path instead, and
    # would otherwise warm max_prefill_chunk_size (== max_seq_len, e.g. 131072)
    # which exceeds practical server boot timeouts. Honor the same override here
    # so nightly's GEMMA4_TRACE_PREFILL_SEQ_LENS=128 actually shortens boot.
    override = os.environ.get("GEMMA4_TRACE_PREFILL_SEQ_LENS")
    if override is not None:
        lengths = []
        for raw in override.split(","):
            raw = raw.strip()
            if not raw:
                continue
            length = int(raw)
            if length > 0 and length <= max_seq and length not in lengths:
                lengths.append(length)
        if not lengths:
            lengths = [min(128, max_seq)]
    else:
        lengths = []
        for length in (128, chunk):
            if length > 0 and length <= max_seq and length not in lengths:
                lengths.append(length)

    sampling_params_short = None
    if can_sample_on_device:
        params = generator._create_sampling_params(
            can_sample_on_device=True,
            batch_size=1,
            greedy_only=greedy_only,
        )
        sampling_params_short = params[0] if params else None

    prefill_forward = prefill_forward_fn if prefill_forward_fn is not None else generator.prefill_forward_text
    logger.info(
        "Eager prefill warmup (no trace, batch=1): lengths={} sampling_on_short={}",
        lengths,
        sampling_params_short is not None,
    )
    for i, length in enumerate(lengths):
        # Match stock: sampling compile on the first/short bucket only.
        sampling_params = sampling_params_short if i == 0 else None
        logger.info(
            "Warming up eager prefill seq_len={} sampling={}",
            length,
            sampling_params is not None,
        )
        warmup_args = generator._mock_tokens(1, length, kv_cache, 0)
        prefill_forward(
            **warmup_args,
            kv_cache=kv_cache,
            enable_trace=False,
            model_id_warmup=0,
            sampling_params=sampling_params,
            warmup_prefill=False,
        )
        logger.info("Finished eager prefill warmup seq_len={}", length)

    # Demo-only opt-in: compile B>1 CCL once. Server / chunked-prefill product
    # path stays batch-1 (tt_transformers pattern).
    batch_override = os.environ.get("GEMMA4_WARMUP_PREFILL_BATCHES")
    if batch_override:
        from models.demos.gemma4.tt.generator import max_batched_prefill_users

        max_batch = int(getattr(generator.model_args[0], "max_batch_size", 1) or 1)
        warm_batch = min(max_batch, max_batched_prefill_users())
        requested = [int(x) for x in batch_override.split(",") if x.strip()]
        warm_batch = max((b for b in requested if 1 < b <= warm_batch), default=1)
        warm_batch = max((b for b in SUPPORTED_PREFILL_BATCH_SIZES if b <= warm_batch), default=1)
        if warm_batch > 1 and 128 * warm_batch < MAX_BATCHED_PREFILL_SEQ_LEN:
            logger.info(
                "Warming up eager batched prefill batch_size={} seq_len=128 (no sampling)",
                warm_batch,
            )
            warmup_args = generator._mock_tokens(warm_batch, 128, kv_cache, 0)
            prefill_forward(
                **warmup_args,
                kv_cache=kv_cache,
                enable_trace=False,
                model_id_warmup=0,
                sampling_params=None,
                warmup_prefill=False,
            )
            logger.info("Finished eager batched prefill warmup batch_size={}", warm_batch)
