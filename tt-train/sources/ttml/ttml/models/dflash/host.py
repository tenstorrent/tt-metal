# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Host-side (numpy only) pieces of DFlash drafter training.

Everything with data-dependent shapes stays here, on the CPU: anchor sampling,
the square attention masks, RoPE tables, the noise-block token ids, teacher row
indices and the per-slot loss weights. The device only sees fixed-shape tensors.

Reference semantics: speculators ``models/dflash/{core,utils,attention}.py`` and
``models/metrics.py``. Nothing here imports ttnn/ttml, so it can be tested on any
machine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

TILE = 32
FULL_ATTENTION = "full_attention"
SLIDING_ATTENTION = "sliding_attention"
LOSS_EPS = 1e-5  # speculators.models.metrics._EPS


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DFlashDrafterConfig:
    """Shape and algorithm settings of a DFlash drafter (tt-blaze compatible)."""

    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    num_aux_layers: int
    layer_types: tuple[str, ...]
    block_size: int
    mask_token_id: int
    rms_norm_eps: float = 1e-5
    sliding_window: int | None = None
    sliding_window_non_causal: bool = False
    sliding_window_per_query: bool = False
    rope_parameters: Mapping[str, Any] = field(default_factory=lambda: {"rope_type": "default", "rope_theta": 10000.0})

    def __post_init__(self) -> None:
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be a multiple of num_kv_heads")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")
        bad = set(self.layer_types) - {FULL_ATTENTION, SLIDING_ATTENTION}
        if bad:
            raise ValueError(f"unknown layer types {sorted(bad)}")
        if SLIDING_ATTENTION in self.layer_types and not (self.sliding_window and self.sliding_window >= 1):
            raise ValueError("sliding_attention layers need sliding_window >= 1")

    @property
    def num_layers(self) -> int:
        return len(self.layer_types)

    @classmethod
    def from_golden_meta(cls, meta: Mapping[str, Any]) -> "DFlashDrafterConfig":
        p = meta["preset_fields"]
        return cls(
            hidden_size=p["hidden_size"],
            num_heads=p["num_heads"],
            num_kv_heads=p["num_kv_heads"],
            head_dim=p["head_dim"],
            intermediate_size=p["intermediate_size"],
            vocab_size=p["vocab_size"],
            num_aux_layers=p["num_aux_layers"],
            layer_types=tuple(meta["layer_types"]),
            block_size=meta["block_size"],
            mask_token_id=meta["mask_token_id"],
            rms_norm_eps=meta["rms_norm_eps"],
            sliding_window=meta["sliding_window"],
            sliding_window_non_causal=meta["sliding_window_non_causal"],
            sliding_window_per_query=meta["sliding_window_per_query"],
            rope_parameters=dict(meta["rope_parameters"]),
        )

    @classmethod
    def from_speculators_config(cls, cfg: Mapping[str, Any]) -> "DFlashDrafterConfig":
        """From a speculators DFlash ``config.json`` (``DFlashSpeculatorConfig``)."""
        tl = cfg["transformer_layer_config"]
        if cfg.get("use_hidden_size_projections"):
            raise NotImplementedError("input_proj/output_proj are not ported (tt-blaze has no slot for them)")
        if cfg.get("draft_vocab_size", tl["vocab_size"]) != tl["vocab_size"]:
            raise NotImplementedError("reduced draft vocabularies (t2d/d2t) are not ported")
        rope = tl.get("rope_parameters") or tl.get("rope_scaling") or {}
        rope = dict(rope)
        rope.setdefault("rope_theta", tl.get("rope_theta", 10000.0))
        rope.setdefault("rope_type", rope.get("type", "default"))
        return cls(
            hidden_size=tl["hidden_size"],
            num_heads=tl["num_attention_heads"],
            num_kv_heads=tl["num_key_value_heads"],
            head_dim=tl.get("head_dim") or tl["hidden_size"] // tl["num_attention_heads"],
            intermediate_size=tl["intermediate_size"],
            vocab_size=tl["vocab_size"],
            num_aux_layers=len(cfg["aux_hidden_state_layer_ids"]),
            layer_types=tuple(tl.get("layer_types") or [FULL_ATTENTION] * tl["num_hidden_layers"]),
            block_size=cfg["block_size"],
            mask_token_id=cfg["mask_token_id"],
            rms_norm_eps=tl.get("rms_norm_eps", 1e-6),
            sliding_window=tl.get("sliding_window"),
            sliding_window_non_causal=cfg.get("sliding_window_non_causal", False),
            sliding_window_per_query=cfg.get("sliding_window_per_query", False),
            rope_parameters=rope,
        )


# ---------------------------------------------------------------------------
# RoPE (HF half-rotation layout, YaRN / DeepSeek-YaRN frequencies)
# ---------------------------------------------------------------------------


def _deepseek_attention_factor(params: Mapping[str, Any]) -> float:
    """speculators create_dflash_rotary_embedding: vLLM DeepseekScalingRotaryEmbedding mscale."""
    factor = params["factor"]
    log_scale = 0.1 * math.log(factor) if factor > 1 else 0.0
    return (
        (1.0 + params.get("mscale", 1.0) * log_scale)
        / (1.0 + params.get("mscale_all_dim", 0.0) * log_scale)
        * params.get("attn_factor", 1.0)
    )


def rope_inv_freq(head_dim: int, rope: Mapping[str, Any]) -> tuple[np.ndarray, float]:
    """``(inv_freq[head_dim // 2] float32, attention_scaling)``.

    Mirrors transformers ``_compute_default_rope_parameters`` / ``_compute_yarn_parameters``
    in float32 (as torch computes them), with the DeepSeek-YaRN attention factor that
    speculators injects for ``rope_type == "deepseek_yarn"``.
    """
    rope_type = rope.get("rope_type", rope.get("type", "default"))
    base = float(rope["rope_theta"])
    dim = head_dim
    exponent = np.arange(0, dim, 2, dtype=np.int64).astype(np.float32) / np.float32(dim)
    pos_freqs = np.power(np.float32(base), exponent, dtype=np.float32)
    if rope_type == "default":
        return (np.float32(1.0) / pos_freqs).astype(np.float32), 1.0
    if rope_type not in ("yarn", "deepseek_yarn"):
        raise NotImplementedError(f"rope_type {rope_type!r}")
    if rope.get("extrapolation_factor", 1.0) != 1.0:
        raise ValueError("DFlash DeepSeek YaRN requires extrapolation_factor=1")

    factor = float(rope["factor"])
    if rope_type == "deepseek_yarn":
        attention_factor = _deepseek_attention_factor(rope)
    else:
        attention_factor = rope.get("attention_factor")
        if attention_factor is None:

            def get_mscale(scale, m=1.0):
                return 1.0 if scale <= 1 else 0.1 * m * math.log(scale) + 1.0

            mscale, mscale_all_dim = rope.get("mscale"), rope.get("mscale_all_dim")
            if mscale and mscale_all_dim:
                attention_factor = float(get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim))
            else:
                attention_factor = get_mscale(factor)
    beta_fast = rope.get("beta_fast") or 32
    beta_slow = rope.get("beta_slow") or 1
    original = rope["original_max_position_embeddings"]

    def correction_dim(num_rotations):
        return (dim * math.log(original / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    low, high = correction_dim(beta_fast), correction_dim(beta_slow)
    if rope.get("truncate", True):
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, dim - 1)
    if low == high:
        high += 0.001
    ramp = np.clip((np.arange(dim // 2, dtype=np.float32) - np.float32(low)) / np.float32(high - low), 0, 1)
    extrapolation_factor = (np.float32(1) - ramp).astype(np.float32)
    inv_extra = np.float32(1.0) / pos_freqs
    inv_inter = np.float32(1.0) / (np.float32(factor) * pos_freqs)
    inv_freq = inv_inter * (np.float32(1) - extrapolation_factor) + inv_extra * extrapolation_factor
    return inv_freq.astype(np.float32), float(attention_factor)


def rope_cos_sin(positions: np.ndarray, head_dim: int, rope: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """``cos, sin`` of shape ``[len(positions), head_dim]`` (HF ``cat(freqs, freqs)`` layout)."""
    inv_freq, scale = rope_inv_freq(head_dim, rope)
    freqs = positions.astype(np.float32)[:, None] * inv_freq[None, :]
    emb = np.concatenate([freqs, freqs], axis=-1)
    return (np.cos(emb) * np.float32(scale)).astype(np.float32), (np.sin(emb) * np.float32(scale)).astype(np.float32)


def signed_sin(sin: np.ndarray) -> np.ndarray:
    """``rotate_half(x) * sin == concat(x2, x1) * signed_sin(sin)``: negate the first half."""
    half = sin.shape[-1] // 2
    out = sin.copy()
    out[..., :half] *= -1
    return out


# ---------------------------------------------------------------------------
# Anchors
# ---------------------------------------------------------------------------


def anchor_candidates(
    loss_mask: np.ndarray, document_ids: np.ndarray, block_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Eligible anchors and their run ids (speculators ``get_anchor_candidates``).

    An anchor needs a nonzero loss mask, a non-padding document, and its whole block
    inside one contiguous run of the same document id.
    """
    loss_mask = np.asarray(loss_mask).reshape(-1)
    docs = np.asarray(document_ids, dtype=np.int64).reshape(-1)
    starts = np.ones_like(docs, dtype=bool)
    starts[1:] = docs[1:] != docs[:-1]
    run_ids = np.cumsum(starts) - 1
    n = max(loss_mask.size - block_size + 1, 0)
    ok = (loss_mask[:n] != 0) & (docs[:n] >= 0) & (run_ids[:n] == run_ids[block_size - 1 : block_size - 1 + n])
    positions = np.nonzero(ok)[0]
    return positions.astype(np.int64), run_ids[positions]


def sample_anchors(
    rng: np.random.Generator,
    loss_mask: np.ndarray,
    document_ids: np.ndarray,
    block_size: int,
    max_anchors_per_seq: int,
    anchor_budget: int,
    overflow: str = "raise",
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-shape version of speculators ``select_anchors``.

    Draws up to ``max_anchors_per_seq`` distinct anchors per packed sequence
    (uniformly, without replacement), then pads to ``anchor_budget`` slots with
    ``anchor_valid = False``. The budget is a shape constant so the device graph is
    reused across steps; size it for the worst case (``max_anchors_per_seq`` times
    the most sequences a row can hold), keeping ``anchor_budget * block_size`` a
    multiple of 32.

    speculators has no row-level cap. If a row needs more than ``anchor_budget`` anchors,
    ``overflow="raise"`` errors and ``overflow="subsample"`` keeps a uniform random
    ``anchor_budget``-subset (a deviation from the reference; count it via
    ``required_anchor_count``).

    Returns ``(anchor_positions[anchor_budget] int64, anchor_valid[anchor_budget] bool)``.
    """
    positions, run_ids = anchor_candidates(loss_mask, document_ids, block_size)
    chosen = []
    for run in np.unique(run_ids):
        cand = positions[run_ids == run]
        chosen.append(rng.choice(cand, size=min(max_anchors_per_seq, cand.size), replace=False))
    anchors = np.sort(np.concatenate(chosen)) if chosen else np.zeros(0, np.int64)
    if anchors.size == 0:
        raise ValueError("no eligible DFlash anchors in this packed row")
    if anchors.size > anchor_budget:
        if overflow == "subsample":
            anchors = np.sort(rng.choice(anchors, size=anchor_budget, replace=False))
        else:
            raise ValueError(
                f"{anchors.size} anchors exceed the fixed budget {anchor_budget}; raise anchor_budget, "
                "lower max_anchors_per_seq, or pass overflow='subsample'"
            )
    pad = anchor_budget - anchors.size
    return (
        np.concatenate([anchors, np.full(pad, anchors[0])]).astype(np.int64),
        np.concatenate([np.ones(anchors.size, bool), np.zeros(pad, bool)]),
    )


def required_anchor_count(loss_mask, document_ids, block_size: int, max_anchors_per_seq: int) -> int:
    """Anchors speculators would draw for this row (``count_anchors``)."""
    _, run_ids = anchor_candidates(loss_mask, document_ids, block_size)
    _, counts = np.unique(run_ids, return_counts=True)
    return int(np.minimum(counts, max_anchors_per_seq).sum())


# ---------------------------------------------------------------------------
# Square attention mask (D1a)
# ---------------------------------------------------------------------------


def build_dflash_square_mask(
    document_ids: np.ndarray,
    anchor_positions: np.ndarray,
    anchor_valid: np.ndarray,
    block_size: int,
    sliding_window: int | None = None,
    sliding_window_non_causal: bool = False,
    sliding_window_per_query: bool = False,
) -> np.ndarray:
    """Build the bool ``[T + n*B, T + n*B]`` mask for the square fused SDPA.

    Args:
        document_ids: ``[T]`` document index per packed position, ``-1`` for padding.
        anchor_positions: ``[n]`` anchor positions into the packed sequence.
        anchor_valid: ``[n]`` bool; padded anchors attend only to themselves.
        block_size: slots per anchor block.
        sliding_window, sliding_window_non_causal, sliding_window_per_query: same
            meaning as in speculators ``create_anchor_block_mask_mod``.

    Returns:
        ``mask[q, kv] == True`` where query row ``q`` of ``X = [context ; blocks]`` may
        attend to key ``kv``. Context rows and padded-anchor rows see only themselves.
    """
    document_ids = np.asarray(document_ids, dtype=np.int64)
    anchor_positions = np.asarray(anchor_positions, dtype=np.int64)
    anchor_valid = np.asarray(anchor_valid, dtype=bool)
    if document_ids.ndim != 1 or anchor_positions.ndim != 1 or anchor_valid.shape != anchor_positions.shape:
        raise ValueError("document_ids, anchor_positions and anchor_valid must be 1D (anchors of equal length)")
    if sliding_window is not None and sliding_window < 1:
        raise ValueError(f"sliding_window must be >= 1, got {sliding_window}")

    T = document_ids.size
    nB = anchor_positions.size * block_size
    if T % TILE or nB % TILE:
        raise ValueError(
            f"context length ({T}) and anchor slots n*B ({nB}) must be multiples of {TILE}; "
            "pad the anchor budget with anchor_valid=0"
        )
    if anchor_positions.size and (anchor_positions.min() < 0 or anchor_positions.max() + block_size > T):
        raise ValueError("every anchor block must lie inside the packed row")
    non_causal = sliding_window is None or sliding_window_non_causal

    S = T + nB
    mask = np.eye(S, dtype=bool)

    slot = np.arange(nB)
    slot_offset = slot % block_size  # [nB]
    slot_anchor = np.repeat(anchor_positions, block_size)  # [nB]
    slot_valid = np.repeat(anchor_valid, block_size)  # [nB]

    # Context keys: same document, before the anchor, inside the window.
    ctx_pos = np.arange(T)[None, :]  # [1, T]
    q_anchor = slot_anchor[:, None]  # [nB, 1]
    q_doc = document_ids[slot_anchor][:, None]
    ctx_allowed = (q_doc == document_ids[None, :]) & (q_doc != -1) & (ctx_pos < q_anchor)
    if sliding_window is not None:
        if sliding_window_per_query:
            query_position = q_anchor + slot_offset[:, None]
            ctx_allowed &= ctx_pos >= query_position - sliding_window + 1
        else:
            ctx_allowed &= ctx_pos >= q_anchor - sliding_window

    # Block keys: own block only (causal within the block unless non-causal).
    q_slot = slot[:, None]
    kv_slot = slot[None, :]
    blk_allowed = (q_slot // block_size) == (kv_slot // block_size)
    if not non_causal:
        blk_allowed &= kv_slot <= q_slot
    if sliding_window_per_query and sliding_window is not None:
        blk_allowed &= kv_slot >= q_slot - sliding_window + 1

    rows = T + slot[slot_valid]
    mask[rows, :T] = ctx_allowed[slot_valid]
    mask[rows, T:] = blk_allowed[slot_valid]
    return mask


# ---------------------------------------------------------------------------
# Loss weights
# ---------------------------------------------------------------------------


def dflash_decay(slot_offset: np.ndarray, gamma: float) -> np.ndarray:
    """speculators ``dflash_loss_decay``: 0 at the anchor, then ``exp(-(k - 1) / gamma)``."""
    k = slot_offset.astype(np.float32)
    return (np.exp(-np.clip(k - 1, 0, None) / np.float32(gamma)) * (k != 0)).astype(np.float32)


# ---------------------------------------------------------------------------
# One packed row -> fixed-shape host batch
# ---------------------------------------------------------------------------


@dataclass
class DFlashHostBatch:
    """Fixed-shape inputs for one DFlash training step on one packed row.

    ``S = T + n*B``. Rows of every ``[S, ...]`` tensor are ``[context ; block slots]``.
    """

    hidden_states: np.ndarray  # [T, num_aux * H] float32: fc input
    teacher_hidden: np.ndarray  # [n*B, H] float32: final target state graded at each slot
    noise_token_ids: np.ndarray  # [n*B] uint32: anchor token at slot 0, mask token elsewhere
    positions: np.ndarray  # [S] int64: RoPE positions
    cos: np.ndarray  # [S, D] float32
    sin_signed: np.ndarray  # [S, D] float32 (see signed_sin)
    masks: dict[str, np.ndarray]  # layer type -> bool [S, S]
    aligned_loss_mask: np.ndarray  # [n*B] float32 (anchor slots and padded anchors are 0)
    slot_weights: np.ndarray  # [n*B] float32: loss = sum_slots slot_weights * KL
    anchor_positions: np.ndarray  # [n] int64
    anchor_valid: np.ndarray  # [n] bool
    block_indices: np.ndarray  # [n*B] int64: packed position predicted by each slot

    @property
    def context_len(self) -> int:
        return self.hidden_states.shape[0]

    @property
    def num_slots(self) -> int:
        return self.noise_token_ids.shape[0]


def build_host_batch(
    cfg: DFlashDrafterConfig,
    *,
    hidden_states: np.ndarray,
    verifier_last_hidden_states: np.ndarray,
    input_ids: np.ndarray,
    loss_mask: np.ndarray,
    document_ids: np.ndarray,
    position_ids: np.ndarray,
    anchor_positions: np.ndarray,
    anchor_valid: np.ndarray,
    gamma: float,
) -> DFlashHostBatch:
    """Everything DFlashDraftModel.forward derives from the batch, as fixed-shape arrays.

    Inputs are one packed row (leading batch dim of 1 is accepted and dropped).
    """
    hidden_states = np.asarray(hidden_states, dtype=np.float32)
    if hidden_states.ndim == 3:
        hidden_states = hidden_states[0]
    final = np.asarray(verifier_last_hidden_states, dtype=np.float32)
    if final.ndim == 3:
        final = final[0]
    input_ids = np.asarray(input_ids).reshape(-1)
    loss_mask = np.asarray(loss_mask, dtype=np.float32).reshape(-1)
    document_ids = np.asarray(document_ids, dtype=np.int64).reshape(-1)
    position_ids = np.asarray(position_ids, dtype=np.int64).reshape(-1)
    anchor_positions = np.asarray(anchor_positions, dtype=np.int64).reshape(-1)
    anchor_valid = np.asarray(anchor_valid, dtype=bool).reshape(-1)

    T = input_ids.size
    B = cfg.block_size
    nB = anchor_positions.size * B
    expected_fc_in = cfg.num_aux_layers * cfg.hidden_size
    if hidden_states.shape != (T, expected_fc_in):
        raise ValueError(f"hidden_states {hidden_states.shape} != ({T}, {expected_fc_in})")
    if final.shape != (T, cfg.hidden_size):
        raise ValueError(f"verifier_last_hidden_states {final.shape} != ({T}, {cfg.hidden_size})")
    for name, arr in (("loss_mask", loss_mask), ("document_ids", document_ids), ("position_ids", position_ids)):
        if arr.size != T:
            raise ValueError(f"{name} has {arr.size} entries, expected {T}")

    offsets = np.arange(B, dtype=np.int64)
    block_indices = (anchor_positions[:, None] + offsets[None, :]).reshape(-1)  # [nB]
    if block_indices.size and (block_indices.min() < 0 or block_indices.max() >= T):
        raise ValueError("every anchor block must lie inside the packed row")

    noise = np.full(nB, cfg.mask_token_id, dtype=np.int64)
    noise[::B] = input_ids[anchor_positions]

    # Teacher: verifier_logits rolled right by one over the packed row, then gathered at
    # the block indices, i.e. the final state at (index - 1) mod T.
    teacher_rows = (block_indices - 1) % T

    block_positions = (position_ids[anchor_positions][:, None] + offsets[None, :]).reshape(-1)
    positions = np.concatenate([position_ids, block_positions])
    cos, sin = rope_cos_sin(positions, cfg.head_dim, cfg.rope_parameters)

    masks: dict[str, np.ndarray] = {}
    for layer_type in sorted(set(cfg.layer_types)):
        sliding = layer_type == SLIDING_ATTENTION
        masks[layer_type] = build_dflash_square_mask(
            document_ids,
            anchor_positions,
            anchor_valid,
            B,
            sliding_window=cfg.sliding_window if sliding else None,
            sliding_window_non_causal=cfg.sliding_window_non_causal if sliding else False,
            sliding_window_per_query=cfg.sliding_window_per_query,
        )

    aligned = loss_mask[block_indices] * np.repeat(anchor_valid, B).astype(np.float32)
    aligned[::B] = 0.0
    decay = dflash_decay(np.arange(nB) % B, gamma)
    slot_weights = (aligned * decay / (aligned.sum() + LOSS_EPS)).astype(np.float32)

    return DFlashHostBatch(
        hidden_states=hidden_states,
        teacher_hidden=final[teacher_rows],
        noise_token_ids=noise.astype(np.uint32),
        positions=positions,
        cos=cos,
        sin_signed=signed_sin(sin),
        masks=masks,
        aligned_loss_mask=aligned.astype(np.float32),
        slot_weights=slot_weights,
        anchor_positions=anchor_positions,
        anchor_valid=anchor_valid,
        block_indices=block_indices,
    )
