# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Experimental TT core for the published Laguna DFlash draft checkpoints.

The draft is the one published for the selected target (``TT_LAGUNA_MODEL``,
``tt/model_spec.py`` ``DFLASH_MODELS``): ``poolside/Laguna-XS-2.1-DFlash`` (five layers,
hidden 2048, served on p150x2) or ``poolside/Laguna-S-2.1-DFlash`` (six layers, hidden
3072, served on p150x4).  Every size below comes from the checkpoint config.

This module owns the isolated model core and request-scoped draft state.  The separate
``dflash_serving`` controller schedules verification/acceptance, and the vLLM bridge
registers that controller only when ``TT_LAGUNA_DFLASH=1``.  Core construction and
execution remain explicitly disabled unless ``enable_experimental=True`` is supplied.

The draft layers reuse :class:`MultichipDecoder` after strict checkpoint mapping:
the published fused QKV rows are split into Q/K/V projections and a corrected HF-like
configuration describes dense causal sliding-attention layers.  In particular,
the draft uses full 128-channel NeoX RoPE with theta 500,000; it must never inherit the
target Laguna sliding-layer theta of 10,000.

Mesh placement on a 1×D mesh (D=2 for XS, D=4 for S) is the target's own layout:
each draft layer is tensor-parallel exactly like a dense target layer (query/KV heads,
gate and dense FFN split D ways, two all-reduces per layer, replicated BF16 residual);
the auxiliary norms, fusion ``fc``, ``hidden_norm`` and final ``norm`` are replicated;
the target's auxiliary capture is a replicated ``[1, rows, num_aux * hidden]`` tensor
(the target residual is replicated); and the sampled rows are projected by the target's
vocab-sharded LM head, so logits come back as D vocab shards.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch

import ttnn

from .dflash_reference import (
    DEFAULT_DFLASH_SNAPSHOT,
    DFlashProposalBlock,
    DFlashTargetAuxCapture,
    LagunaDFlashCheckpoint,
    LagunaDFlashConfig,
    build_proposal_block,
    expected_checkpoint_shapes,
)
from .model_spec import MODEL_ENV, MODEL_ID, dflash_spec
from .multichip_decoder import MultichipDecoder, _cache_layer_identity
from .ring_write import ring_write
from .optimized_decoder import TILE, PrecisionPolicy, _cached_device_tensor, _dram_weight_memcfg, weight_cache_key

DFLASH_CACHE_NAMESPACE = "dflash"


@dataclass(frozen=True)
class DFlashDecoderConfig:
    """The subset of an HF config consumed by ``LayerConfig``/``MultichipDecoder``."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int
    rms_norm_eps: float
    sliding_window: int
    rope_theta: float
    layer_types: tuple[str, ...]
    rope_parameters: dict[str, dict[str, float | str]]
    swa_rope_parameters: dict[str, float | str]
    partial_rotary_factor: float
    mlp_only_layers: tuple[int, ...]
    num_experts: int
    decoder_sparse_step: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    norm_topk_prob: bool
    num_attention_heads_per_layer: None
    hidden_act: str
    attention_bias: bool
    _name_or_path: str


def build_dflash_decoder_config(config: LagunaDFlashConfig) -> DFlashDecoderConfig:
    """Build the dense/SWA config expected by the existing TT decoder.

    Both RoPE branches are populated defensively because the inherited helper selects a
    branch by attention kind.  The draft's attention remains sliding, but its rotary
    factor is 1.0 and theta is 500,000 in that branch too.
    """

    config.validate()
    rope = {
        "rope_type": "default",
        "rope_theta": float(config.rope_theta),
        "partial_rotary_factor": 1.0,
    }
    return DFlashDecoderConfig(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        vocab_size=config.vocab_size,
        max_position_embeddings=config.max_position_embeddings,
        rms_norm_eps=config.rms_norm_eps,
        sliding_window=config.sliding_window,
        rope_theta=config.rope_theta,
        layer_types=("sliding_attention",) * config.num_hidden_layers,
        rope_parameters={"full_attention": dict(rope), "sliding_attention": dict(rope)},
        swa_rope_parameters=dict(rope),
        partial_rotary_factor=1.0,
        # ``LayerConfig.from_hf`` treats these as dense-only layers before consulting
        # any MoE geometry.  Supplying every field avoids truthiness/version drift.
        mlp_only_layers=tuple(range(config.num_hidden_layers)),
        num_experts=0,
        decoder_sparse_step=1,
        num_experts_per_tok=0,
        moe_intermediate_size=0,
        shared_expert_intermediate_size=0,
        norm_topk_prob=False,
        num_attention_heads_per_layer=None,
        hidden_act=config.hidden_act,
        attention_bias=config.attention_bias,
        _name_or_path=dflash_spec(config.target_model_id).repo_id,
    )


def build_dflash_rope_tables(
    config: LagunaDFlashConfig,
    max_seq_len: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return full-dimension NeoX cosine/sine tables as ``[position, head_dim]``."""

    max_seq_len = int(max_seq_len)
    if not 1 <= max_seq_len <= config.max_position_embeddings:
        raise ValueError(f"max_seq_len must be in [1, {config.max_position_embeddings}], got {max_seq_len}")
    half = config.head_dim // 2
    inv_freq = 1.0 / (config.rope_theta ** (torch.arange(half, dtype=torch.float32) * 2.0 / config.head_dim))
    phase = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), inv_freq)
    phase = torch.cat((phase, phase), dim=-1)
    return phase.cos().to(dtype=dtype), phase.sin().to(dtype=dtype)


def _validate_layer_index(config: LagunaDFlashConfig, layer_idx: int) -> int:
    layer_idx = int(layer_idx)
    if not 0 <= layer_idx < config.num_hidden_layers:
        raise ValueError(f"draft layer index outside [0, {config.num_hidden_layers}): {layer_idx}")
    return layer_idx


def dflash_layer_checkpoint_names(config: LagunaDFlashConfig, layer_idx: int) -> tuple[str, ...]:
    """Exact published checkpoint names for one draft layer."""

    layer_idx = _validate_layer_index(config, layer_idx)
    prefix = f"layers.{layer_idx}."
    return tuple(name for name in expected_checkpoint_shapes(config) if name.startswith(prefix))


def dflash_shared_checkpoint_names(config: LagunaDFlashConfig) -> tuple[str, ...]:
    """Exact draft-owned weights outside the draft decoder layers."""

    expected = expected_checkpoint_shapes(config)
    return tuple(name for name in expected if not name.startswith("layers."))


def _validate_checkpoint_subset(
    state_dict: Mapping[str, torch.Tensor],
    expected_names: Sequence[str],
    expected_shapes: Mapping[str, tuple[int, ...]],
    *,
    scope_names: Sequence[str],
    scope: str,
) -> None:
    present = set(state_dict)
    expected = set(expected_names)
    scoped = set(scope_names)
    missing = sorted(expected - present)
    unexpected = sorted(scoped - expected)
    shape_mismatches = {
        name: (tuple(state_dict[name].shape), expected_shapes[name])
        for name in sorted(expected & present)
        if tuple(state_dict[name].shape) != expected_shapes[name]
    }
    wrong_dtypes = {
        name: str(state_dict[name].dtype)
        for name in sorted(expected & present)
        if state_dict[name].dtype != torch.bfloat16
    }
    if missing or unexpected or shape_mismatches or wrong_dtypes:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        if shape_mismatches:
            details.append(f"shape_mismatch={shape_mismatches}")
        if wrong_dtypes:
            details.append(f"non_bf16={wrong_dtypes}")
        raise ValueError(f"invalid {scope} checkpoint mapping: " + "; ".join(details))


def map_dflash_layer_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    config: LagunaDFlashConfig,
    layer_idx: int,
) -> dict[str, torch.Tensor]:
    """Strictly map one published draft layer to ``MultichipDecoder`` names.

    Unrelated layers/shared tensors may be present, but the selected ``layers.N``
    namespace must contain exactly the ten published tensors with BF16 dtype and exact
    shapes.  The fused QKV projection is row-concatenated ``[Q, K, V]``.
    """

    layer_idx = _validate_layer_index(config, layer_idx)
    prefix = f"layers.{layer_idx}."
    names = dflash_layer_checkpoint_names(config, layer_idx)
    shapes = expected_checkpoint_shapes(config)
    scoped = [name for name in state_dict if name.startswith(prefix)]
    _validate_checkpoint_subset(
        state_dict,
        names,
        shapes,
        scope_names=scoped,
        scope=f"DFlash layer {layer_idx}",
    )

    def get(suffix: str) -> torch.Tensor:
        return state_dict[prefix + suffix]

    fused = get("self_attn.qkv_proj.weight")
    q_proj, k_proj, v_proj = fused.split((config.q_size, config.kv_size, config.kv_size), dim=0)
    return {
        "input_layernorm.weight": get("input_layernorm.weight"),
        "self_attn.q_proj.weight": q_proj,
        "self_attn.k_proj.weight": k_proj,
        "self_attn.v_proj.weight": v_proj,
        "self_attn.q_norm.weight": get("self_attn.q_norm.weight"),
        "self_attn.k_norm.weight": get("self_attn.k_norm.weight"),
        "self_attn.g_proj.weight": get("self_attn.g_proj.weight"),
        "self_attn.o_proj.weight": get("self_attn.o_proj.weight"),
        "post_attention_layernorm.weight": get("post_attention_layernorm.weight"),
        "mlp.gate_proj.weight": get("mlp.gate_proj.weight"),
        "mlp.up_proj.weight": get("mlp.up_proj.weight"),
        "mlp.down_proj.weight": get("mlp.down_proj.weight"),
    }


def map_dflash_shared_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    config: LagunaDFlashConfig,
) -> dict[str, torch.Tensor]:
    """Validate and return aux norms, fusion FC, hidden norm, and final norm."""

    names = dflash_shared_checkpoint_names(config)
    shapes = expected_checkpoint_shapes(config)
    # A complete checkpoint may contain all layer keys.  Any other top-level tensor
    # is an ownership/layout error (not, for example, a silently accepted LM head).
    scoped = [name for name in state_dict if not name.startswith("layers.")]
    _validate_checkpoint_subset(
        state_dict,
        names,
        shapes,
        scope_names=scoped,
        scope="DFlash shared weights",
    )
    return {name: state_dict[name] for name in names}


def dflash_bf16_policy() -> PrecisionPolicy:
    """Accuracy-first BF16 policy for this unoptimized core proof. TT_LAGUNA_DFLASH_DRAFT_WDT=bf8 stores the
    attention and MLP projection weights as bfloat8_b (half the bytes a 32-row draft layer reads; the default:
    teacher-forced AIME24 acceptance 2.667 vs 2.677 with bf16, draft 5.7-6.0 -> 5.2-5.3 ms per round); bf16 keeps BF16."""

    wdt = {"bf16": ttnn.bfloat16, "bf8": ttnn.bfloat8_b}[os.environ.get("TT_LAGUNA_DFLASH_DRAFT_WDT", "bf8")]
    return PrecisionPolicy(
        attn_qkv=wdt,
        attn_o=wdt,
        attn_gate=ttnn.bfloat16,
        dense_ff13=wdt,
        dense_ff2=wdt,
        moe_ff13=ttnn.bfloat16,
        moe_ff2=ttnn.bfloat16,
        shared_ff13=ttnn.bfloat16,
        shared_ff2=ttnn.bfloat16,
        router=ttnn.bfloat16,
        qk_norm=ttnn.bfloat16,
        lm_head=ttnn.bfloat16,
        kv_cache=ttnn.bfloat16,
        ccl=ttnn.bfloat16,
        activation=ttnn.bfloat16,
        logits=ttnn.bfloat16,
        # Five (XS) or six (S) serial draft layers amplify projection error enough to change
        # proposal top-1 under HiFi2.  These matrices are BF16 and the proposal
        # block is only 16 rows, so use the accurate fp32-destination kernels.
        fid_attn_qkv="HiFi4",
        fid_attn_o="HiFi4",
        fid_attn_gate="HiFi4",
        fid_dense="HiFi4",
        fid_shared="HiFi4",
        fid_router="HiFi4",
        fid_moe="HiFi4",
    )


@dataclass
class DFlashTTSharedWeights:
    aux_hidden_norms: tuple[object, ...]
    fc: object
    hidden_norm: object
    final_norm: object


@dataclass(frozen=True)
class DFlashTTProposalRound:
    """Device result and exact host geometry for one anchor+15 proposal."""

    block: DFlashProposalBlock
    logits_shards: object
    sampled_hidden_states: object


def _deallocate_owned(tensor) -> None:
    """Best-effort TT tensor release used only for explicitly owned cache state."""

    if tensor is None:
        return
    deallocate = getattr(tensor, "deallocate", None)
    if callable(deallocate):
        deallocate(True)


class DFlashTTProposalCache:
    """Bounded request-scoped draft KV and rolling target auxiliary window.

    The draft never needs target history older than 511 rows.  This object owns
    one tiny local KV pair per draft layer and any rolling concat/slice tensor it creates, but
    never deallocates capture tensors supplied by the target model.  A request
    must be explicitly begun and ended; use-after-end and use-after-close fail
    before launching a device operation.
    """

    def __init__(self, core: "DFlashTTCore", *, block_size: int = 32):
        if tuple(core.layers) != tuple(range(core.config.num_hidden_layers)):
            raise RuntimeError(
                f"a DFlash proposal cache requires all {core.config.num_hidden_layers} draft layers in "
                f"checkpoint order; got {tuple(core.layers)}"
            )
        block_size = int(block_size)
        if block_size != 32:
            raise ValueError(f"DFlash TT proposal cache requires tile/block size 32, got {block_size}")
        self.core = core
        self.block_size = block_size
        # target rows the draft sees (its window: sliding_window - 1 = 511); TT_LAGUNA_DFLASH_CONTEXT_ROWS keeps fewer,
        # which shrinks every proposal (draft layers run over context + 16 query rows)
        self.max_context_rows = min(
            core.config.sliding_window - 1,
            int(os.environ.get("TT_LAGUNA_DFLASH_CONTEXT_ROWS", core.config.sliding_window - 1)),
        )
        self.query_rows = core.config.block_size
        self.capacity = math.ceil((self.max_context_rows + self.query_rows) / block_size) * block_size
        self.kv_cache = {
            index: layer.alloc_kv_cache(
                max_users=1,
                max_seq_len=self.capacity,
                block_size=block_size,
                dtype=ttnn.bfloat16,
            )
            for index, layer in core.layers.items()
        }
        self.page_tables = {
            index: core.layers[index].make_page_table(1, kv["blocks_per_user"]) for index, kv in self.kv_cache.items()
        }
        self._request_id = None
        self._context = None
        self._context_owned = False
        self._context_start = None
        self._context_rows = 0
        self._closed = False
        # Fixed-address context (enable_fixed_context): the COMBINED target context (combine_aux_hidden_states
        # output, [1, rows, hidden]) in one buffer allocated before any trace is captured. Each update combines
        # only the new rows, copies into the buffer and frees its temporaries, so no long-lived buffer is
        # allocated while a trace is resident (a later replay may overwrite such a buffer), and the combine
        # kernels see at most 16 new rows per round instead of every retained row.
        self._fixed = None
        # Cached context keys/values (enable_kv_ring): per draft layer a [1, nkv, RING, hd] K and V ring holding
        # dflash_kv_rows of the last RING target positions (slot of position p: (p - _ring_base) % RING); the draft
        # then runs only its 32 query rows (MultichipDecoder.dflash_query_forward_cached) and a round's committed
        # rows are the only context rows projected.
        self._ring = None
        self._slot_pos = None
        self._ring_base = 0
        self._fixed_stale = False

    @property
    def fixed_combined(self) -> bool:
        return getattr(self, "_fixed", None) is not None

    def enable_fixed_context(self) -> None:
        """Allocate the fixed combined-context buffer. Call before capturing any trace."""

        self._require_open()
        if self._fixed is not None:
            return
        if self._request_id is not None:
            raise RuntimeError("enable_fixed_context must run before any DFlash request")
        # Full proposal capacity (context + query, tile-aligned): rows after the context stay zero, so a draft
        # round can read any padded_total-row prefix (see DFlashTTCore._proposal_round_fixed).
        rows = self.capacity
        self._fixed = ttnn.from_torch(
            torch.zeros((1, rows, self.core.config.hidden_size), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.core.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.core.mesh_device),
        )

    # >= max_context_rows (511) + the uncommitted verify rows a traced update also writes (it writes every verify
    # row, so its inputs are known before the verify ends; a row written past the commit lands in a slot whose old
    # position is outside the window, and the next round rewrites it)
    RING = 544
    # write a round's rows into the rings with one multi-core program (ring_write; ~6 us for the 12 rings) instead of
    # ring = ring * keep + place @ new (four ops per ring, ~236 us)
    RING_WRITE = os.environ.get("TT_LAGUNA_DFLASH_RING_WRITE", "1") == "1"

    @property
    def kv_ring(self) -> bool:
        return getattr(self, "_ring", None) is not None

    def enable_kv_ring(self) -> None:
        """Allocate the per-layer context K/V rings (fixed mode only). Call before capturing any trace."""

        self._require_open()
        if not self.fixed_combined:
            raise RuntimeError("enable_kv_ring requires enable_fixed_context")
        if self._ring is not None:
            return
        cfg = next(iter(self.core.layers.values())).cfg
        zeros = torch.zeros((1, cfg.num_kv_heads, self.RING, cfg.head_dim), dtype=torch.bfloat16)

        def dev():
            return ttnn.from_torch(zeros, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.core.mesh_device,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                   mesh_mapper=ttnn.ReplicateTensorToMesh(self.core.mesh_device))  # fmt: skip

        self._ring = {index: (dev(), dev()) for index in self.core.layers}
        self._slot_pos = [-1] * self.RING

    def ring_kv(self, layer_idx: int):
        return self._ring[layer_idx]

    def _ring_fill(self) -> None:
        """Rebuild every ring from the fixed combined context (after a replace): slot j = position start + j."""

        core = self.core
        start, rows = int(self._context_start), int(self._context_rows)
        n = min(self.RING, int(self._fixed.shape[1]))
        src = self.buffer_rows(n)  # context rows, then zeros
        if n < self.RING:
            src = ttnn.pad(src, [(0, 0), (0, self.RING - n), (0, 0)], value=0.0)
        cos, sin = core.rope_window(start, self.RING)
        for index, layer in core.layers.items():
            k, v = layer.dflash_kv_rows(src, (cos, sin))
            ttnn.copy(k, self._ring[index][0])
            ttnn.copy(v, self._ring[index][1])
            _deallocate_owned(k)
            _deallocate_owned(v)
        for tensor in (src, cos, sin):
            if tensor is not self._fixed:
                _deallocate_owned(tensor)
        self._ring_base = start
        self._slot_pos = [start + j if j < rows else -1 for j in range(self.RING)]
        self._fixed_stale = False

    def ring_append_inputs(self, start_pos: int, count: int):
        """Host tensors writing ``count`` rows at positions start_pos.. into the rings: place [1, nkv, RING, 32]
        (place[s_i, i] = 1 for row i's slot s_i), keep [1, 1, RING, 1] (0 at those slots, else 1) and the rows'
        RoPE (cos, sin) [1, 1, 32, head_dim]. ring = ring * keep + place @ kv_rows(new rows)."""

        start_pos, count = int(start_pos), int(count)
        if not 1 <= count <= 32:
            raise ValueError(f"a DFlash ring append takes 1..32 rows, got {count}")
        cfg = next(iter(self.core.layers.values())).cfg
        place = torch.zeros((self.RING, 32), dtype=torch.bfloat16)
        keep = torch.ones((self.RING, 1), dtype=torch.bfloat16)
        for i in range(count):
            slot = (start_pos + i - self._ring_base) % self.RING
            place[slot, i] = 1.0
            keep[slot, 0] = 0.0
        phase = torch.outer(torch.arange(start_pos, start_pos + 32, dtype=torch.float32), self.core._rope_inv_freq)
        phase = torch.cat((phase, phase), dim=-1)
        hd = int(self.core.config.head_dim)
        return (
            place.reshape(1, 1, self.RING, 32).expand(1, cfg.num_kv_heads, self.RING, 32).contiguous(),
            keep.reshape(1, 1, self.RING, 1),
            phase.cos().to(torch.bfloat16).reshape(1, 1, 32, hd),
            phase.sin().to(torch.bfloat16).reshape(1, 1, 32, hd),
        )

    def ring_slots(self, start_pos: int, count: int):
        """Host uint32 [1, 32]: the ring slot of each of ``count`` rows at positions start_pos.. (ring_write's slots)."""

        slots = torch.zeros((1, 32), dtype=torch.int32)
        for i in range(int(count)):
            slots[0, i] = (int(start_pos) + i - self._ring_base) % self.RING
        return slots

    def ring_write_rows(self, kvs, slots, count: int) -> None:
        """Write rows 0..count-1 of each layer's new (k, v) [1, nkv, 32, hd] into its rings at ``slots`` (a uint32
        [1, 32] row-major device tensor)."""

        rings, srcs = [], []
        for index, (k, v) in kvs.items():
            rings += list(self._ring[index])
            srcs += [k, v]
        ring_write(rings, srcs, slots, count)

    def ring_update(self, index: int, k, v, place, keep) -> None:
        """ring = ring * keep + place @ (k | v) for draft layer ``index`` (k, v: [1, nkv, 32, hd] new rows)."""

        ck = self.core.layers[index]._ck_hifi4  # 0/1 placement: exact with fp32 accumulation
        for ring, new in zip(self._ring[index], (k, v)):
            updated = ttnn.add(ttnn.mul(ring, keep), ttnn.matmul(place, new, compute_kernel_config=ck))
            ttnn.copy(updated, ring)
            _deallocate_owned(updated)

    def ring_commit(self, start_pos: int, count: int, written: int | None = None) -> None:
        """Bookkeeping after rows at positions start_pos.. were written into the rings: the first ``count`` are
        committed context; rows ``count``..``written`` - 1 (an update that wrote every verify row) are not."""

        start_pos, count = int(start_pos), int(count)
        written = count if written is None else int(written)
        expected = int(self._context_start) + int(self._context_rows)
        if start_pos != expected:
            raise ValueError(f"DFlash target capture is not adjacent: expected start {expected}, got {start_pos}")
        if written - count > self.RING - int(self.max_context_rows):
            raise ValueError(f"{written - count} uncommitted ring rows would overwrite retained context")
        for i in range(written):
            self._slot_pos[(start_pos + i - self._ring_base) % self.RING] = start_pos + i if i < count else -1
        rows = min(int(self.max_context_rows), int(self._context_rows) + count)
        self._context_start = start_pos + count - rows
        self._context_rows = rows
        self._fixed_stale = True  # the combined buffer is no longer appended to

    def ring_query_inputs_after(self, start_pos: int, count: int, written: int, query_rows: int):
        """ring_query_inputs as it will be after ring_commit(start_pos, count, written) (state unchanged)."""

        saved = (list(self._slot_pos), self._context_start, self._context_rows, self._fixed_stale)
        try:
            self.ring_commit(start_pos, count, written=written)
            return self.ring_query_inputs(query_rows)
        finally:
            self._slot_pos, self._context_start, self._context_rows, self._fixed_stale = saved

    def _ring_append_capture(self, capture: DFlashTargetAuxCapture) -> None:
        count = int(capture.row_count)
        new_rows = self.core.combine_aux_hidden_states(capture.hidden_states)
        padded = ttnn.pad(new_rows, [(0, 0), (0, 32 - count), (0, 0)], value=0.0) if count < 32 else new_rows
        host = self.ring_append_inputs(capture.start_position, count)
        place, keep, cos, sin = (
            ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.core.mesh_device,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            mesh_mapper=ttnn.ReplicateTensorToMesh(self.core.mesh_device))
            for t in host
        )  # fmt: skip
        slots = None
        if self.RING_WRITE:
            slots = ttnn.from_torch(self.ring_slots(capture.start_position, count), dtype=ttnn.uint32,
                                    layout=ttnn.ROW_MAJOR_LAYOUT, device=self.core.mesh_device,
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.core.mesh_device))  # fmt: skip
        kvs = {}
        for index, layer in self.core.layers.items():
            k, v = layer.dflash_kv_rows(padded, (cos, sin))
            if slots is not None:
                kvs[index] = (k, v)
                continue
            self.ring_update(index, k, v, place, keep)
            _deallocate_owned(k)
            _deallocate_owned(v)
        if slots is not None:
            self.ring_write_rows(kvs, slots, count)
            for k, v in kvs.values():
                _deallocate_owned(k)
                _deallocate_owned(v)
        for tensor in {id(t): t for t in (new_rows, padded, place, keep, cos, sin, slots)}.values():
            _deallocate_owned(tensor)
        self.ring_commit(capture.start_position, count)

    def ring_query_inputs(self, query_rows: int):
        """Host (cos, sin) [1, 1, 32, head_dim] at the query positions P0 + i (P0 = the position after the context)
        and the mask [1, 1, 32, RING + 32] over the ring slots then the 32 query rows: query row i (position
        P0 + i) sees a slot holding position p when start <= p, P0 + i - (sliding_window - 1) <= p < P0 + i, and query
        rows i' <= i -- the causal sliding window of prefill_forward over context + query. Rows past the query copy
        the last query row's mask."""

        start, rows = self.context_bounds()
        p0 = start + rows
        phase = torch.outer(torch.arange(p0, p0 + 32, dtype=torch.float32), self.core._rope_inv_freq)
        phase = torch.cat((phase, phase), dim=-1)
        hd = int(self.core.config.head_dim)
        qi = torch.arange(32).clamp(max=int(query_rows) - 1)
        qpos = p0 + qi
        pos = torch.tensor(self._slot_pos, dtype=torch.int64)
        window = int(self.core.config.sliding_window) - 1
        ctx_ok = (pos[None, :] >= start) & (pos[None, :] < qpos[:, None]) & (pos[None, :] >= qpos[:, None] - window)
        q_ok = torch.arange(32)[None, :] <= qi[:, None]
        allowed = torch.cat((ctx_ok, q_ok), dim=1)
        mask = torch.where(allowed, 0.0, -1e9).reshape(1, 1, 32, self.RING + 32)
        return (
            phase.cos().to(torch.bfloat16).reshape(1, 1, 32, hd),
            phase.sin().to(torch.bfloat16).reshape(1, 1, 32, hd),
            mask.to(torch.bfloat16),
        )

    def _fixed_rows(self, rows: int):
        return ttnn.slice(self._fixed, [0, 0, 0], [1, int(rows), self.core.config.hidden_size])

    def buffer_rows(self, rows: int):
        """First ``rows`` rows of the fixed buffer (context, then zeros); fixed mode only."""

        if not self.fixed_combined:
            raise RuntimeError("buffer_rows requires enable_fixed_context")
        rows = int(rows)
        if rows == int(self._fixed.shape[1]):
            return self._fixed
        return self._fixed_rows(rows)

    def context_bounds(self) -> tuple[int, int]:
        """(start position, row count) of the retained target context."""

        self._require_open()
        has_context = self._context_rows > 0 if self.fixed_combined else self._context is not None
        if self._request_id is None or not has_context:
            raise RuntimeError("DFlash proposal cache has no active target context")
        return int(self._context_start), int(self._context_rows)

    def combined_context(self):
        """[1, rows, hidden] combined target context (fixed mode only); a temporary slice of the buffer."""

        if not self.fixed_combined:
            raise RuntimeError("combined_context requires enable_fixed_context")
        _, rows = self.context_bounds()
        return self._fixed_rows(rows)

    def _append_fixed(self, capture: DFlashTargetAuxCapture) -> None:
        """Append 1..32 committed rows with device programs whose shapes do not depend on the context length.

        The buffer keeps rows ``[0, rows)`` and zeros after them. With ``drop`` = rows that fall out of the
        window, the new buffer is ``S @ buffer + P @ new`` where ``S`` moves rows ``drop..rows-1`` to ``0..`` and
        ``P`` places the new rows right after them; both are 0/1 matrices built on the host. Concatenating and
        slicing at the current row count instead compiled new programs in every round while the context grew."""

        core = self.core
        count = int(capture.row_count)
        expected_start = int(self._context_start) + int(self._context_rows)
        if int(capture.start_position) != expected_start:
            raise ValueError(
                f"DFlash target capture is not adjacent: expected start {expected_start}, got {capture.start_position}"
            )
        total_rows = int(self._fixed.shape[1])
        new_rows = core.combine_aux_hidden_states(capture.hidden_states)
        temporaries = [new_rows]
        if count < 32:
            new_rows = ttnn.pad(new_rows, [(0, 0), (0, 32 - count), (0, 0)], value=0.0)
            temporaries.append(new_rows)
        drop = max(0, int(self._context_rows) + count - int(self.max_context_rows))
        kept = int(self._context_rows) - drop
        if drop:
            shift = torch.zeros((total_rows, total_rows), dtype=torch.bfloat16)
            index = torch.arange(kept)
            shift[index, index + drop] = 1.0
            shift_tt = ttnn.from_torch(
                shift.unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=core.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(core.mesh_device),
            )
            base = core.exact_matmul(shift_tt, self._fixed)
            temporaries += [shift_tt, base]
        else:
            base = self._fixed
        place = core.placement(total_rows, kept, count)
        placed = core.exact_matmul(place, new_rows)
        updated = ttnn.add(base, placed)
        temporaries += [place, placed, updated]
        ttnn.copy(updated, self._fixed)
        released = set()
        for tensor in temporaries:
            if id(tensor) not in released:
                released.add(id(tensor))
                _deallocate_owned(tensor)
        self._context_start = int(self._context_start) + drop
        self._context_rows = kept + count

    def _update_fixed(self, capture: DFlashTargetAuxCapture, replace: bool) -> None:
        if self.kv_ring and not replace and self._context_rows > 0 and int(capture.row_count) <= 32:
            self._ring_append_capture(capture)
            return
        if not replace and self._context_rows > 0 and int(capture.row_count) <= 32:
            self._append_fixed(capture)
            return
        if self.kv_ring and not replace and self._context_rows > 0 and self._fixed_stale:
            raise RuntimeError("a long DFlash append after ring-only appends: the combined buffer is stale")
        width = self.core.config.hidden_size
        new_rows = self.core.combine_aux_hidden_states(capture.hidden_states)
        temporaries = [new_rows]
        if replace or self._context_rows == 0:
            source = new_rows
            start = int(capture.start_position)
            rows = int(capture.row_count)
        else:
            expected_start = int(self._context_start) + int(self._context_rows)
            if int(capture.start_position) != expected_start:
                raise ValueError(
                    f"DFlash target capture is not adjacent: expected start {expected_start}, "
                    f"got {capture.start_position}"
                )
            current = self._fixed_rows(self._context_rows)
            source = ttnn.concat((current, new_rows), dim=1)
            temporaries += [current, source]
            start = int(self._context_start)
            rows = int(self._context_rows) + int(capture.row_count)
        if rows > self.max_context_rows:
            drop = rows - self.max_context_rows
            source = ttnn.slice(source, [0, drop, 0], [1, rows, width])
            temporaries.append(source)
            start += drop
            rows = self.max_context_rows
        if source.layout != ttnn.TILE_LAYOUT:
            source = ttnn.to_layout(source, ttnn.TILE_LAYOUT)
            temporaries.append(source)
        if source.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            source = ttnn.to_memory_config(source, ttnn.DRAM_MEMORY_CONFIG)
            temporaries.append(source)
        if source.dtype != ttnn.bfloat16:
            source = ttnn.typecast(source, ttnn.bfloat16)
            temporaries.append(source)
        padded_rows = int(self._fixed.shape[1])
        if rows < padded_rows:
            source = ttnn.pad(source, [(0, 0), (0, padded_rows - rows), (0, 0)], value=0.0)
            temporaries.append(source)
        ttnn.copy(source, self._fixed)
        released = set()
        for tensor in temporaries:
            if id(tensor) not in released:
                released.add(id(tensor))
                _deallocate_owned(tensor)
        self._context_start = start
        self._context_rows = rows
        if self.kv_ring:
            self._ring_fill()

    @property
    def active_request_id(self):
        return self._request_id

    @property
    def closed(self) -> bool:
        return self._closed

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("DFlash proposal cache is closed")

    def _release_context(self) -> None:
        if getattr(self, "_fixed", None) is not None:
            self._context_start = None
            self._context_rows = 0
            if self.kv_ring:
                self._slot_pos = [-1] * self.RING
                self._fixed_stale = False
            return
        if self._context_owned:
            _deallocate_owned(self._context)
        self._context = None
        self._context_owned = False
        self._context_start = None
        self._context_rows = 0

    def begin_request(self, request_id) -> None:
        self._require_open()
        if request_id is None or request_id == "":
            raise ValueError("DFlash request_id must be non-empty")
        if self._request_id is not None:
            raise RuntimeError(f"DFlash proposal cache already owns request {self._request_id!r}")
        self._request_id = request_id

    def update_target_capture(self, capture: DFlashTargetAuxCapture, *, replace: bool = False) -> None:
        """Replace a prefill window or append an adjacent decode capture."""

        self._require_open()
        if self._request_id is None:
            raise RuntimeError("begin_request must be called before supplying DFlash target state")
        if not isinstance(capture, DFlashTargetAuxCapture):
            raise TypeError("DFlash target state must be a DFlashTargetAuxCapture")
        capture.validate(self.core.config)
        if getattr(self, "_fixed", None) is not None:
            self._update_fixed(capture, replace)
            return
        if replace:
            self._release_context()
        if self._context is None:
            self._context = capture.hidden_states
            self._context_owned = False
            self._context_start = int(capture.start_position)
            self._context_rows = int(capture.row_count)
            return
        expected_start = int(self._context_start) + int(self._context_rows)
        if int(capture.start_position) != expected_start:
            raise ValueError(
                f"DFlash target capture is not adjacent: expected start {expected_start}, "
                f"got {capture.start_position}"
            )
        joined = ttnn.concat((self._context, capture.hidden_states), dim=1)
        total = self._context_rows + int(capture.row_count)
        old = self._context
        old_owned = self._context_owned
        if total > self.max_context_rows:
            drop = total - self.max_context_rows
            retained = ttnn.slice(
                joined,
                [0, drop, 0],
                [1, total, self.core.config.num_aux_hidden_states * self.core.config.hidden_size],
            )
            _deallocate_owned(joined)
            self._context = retained
            self._context_start = int(self._context_start) + drop
            self._context_rows = self.max_context_rows
        else:
            self._context = joined
            self._context_rows = total
        self._context_owned = True
        if old_owned:
            _deallocate_owned(old)

    def target_capture(self) -> DFlashTargetAuxCapture:
        self._require_open()
        if self.fixed_combined:
            raise RuntimeError(
                "a fixed-context DFlash cache keeps combined states only; use context_bounds()/combined_context()"
            )
        if self._request_id is None or self._context is None:
            raise RuntimeError("DFlash proposal cache has no active target context")
        capture = DFlashTargetAuxCapture(
            hidden_states=self._context,
            start_position=int(self._context_start),
            row_count=int(self._context_rows),
        )
        capture.validate(self.core.config)
        return capture

    def end_request(self, request_id=None) -> None:
        self._require_open()
        if self._request_id is None:
            raise RuntimeError("DFlash proposal cache has no active request")
        if request_id is not None and request_id != self._request_id:
            raise RuntimeError(f"cannot end DFlash request {request_id!r}; cache owns {self._request_id!r}")
        self._release_context()
        self._request_id = None

    def close(self) -> None:
        if self._closed:
            return
        self._release_context()
        _deallocate_owned(getattr(self, "_fixed", None))
        self._fixed = None
        for pair in (getattr(self, "_ring", None) or {}).values():
            for tensor in pair:
                _deallocate_owned(tensor)
        self._ring = None
        for kv in self.kv_cache.values():
            _deallocate_owned(kv["k"])
            _deallocate_owned(kv["v"])
        for page_table in self.page_tables.values():
            _deallocate_owned(page_table)
        self.kv_cache.clear()
        self.page_tables.clear()
        self._request_id = None
        self._closed = True

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def load_dflash_shared_weights(
    state_dict: Mapping[str, torch.Tensor],
    config: LagunaDFlashConfig,
    mesh_device,
    *,
    cache_namespace: str = DFLASH_CACHE_NAMESPACE,
) -> DFlashTTSharedWeights:
    """Load all shared draft-owned BF16 weights, replicated on a 1×D mesh."""

    shared = map_dflash_shared_state_dict(state_dict, config)
    _cache_layer_identity(0, cache_namespace)  # validates the namespace contract
    devices = mesh_device.get_num_devices()
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)

    def cached(name: str, build, dtype=ttnn.bfloat16, shard_dim=None):
        return _cached_device_tensor(
            build,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate if shard_dim is None else ttnn.ShardTensorToMesh(mesh_device, dim=shard_dim),
            cache_key=weight_cache_key(
                f"{cache_namespace}_{name}", "shared", f"rep_d{devices}" if shard_dim is None else f"sh{shard_dim}_d{devices}"
            ),
        )

    h = config.hidden_size
    aux = tuple(
        cached(
            f"aux_hidden_norm_{index}",
            lambda index=index: shared[f"aux_hidden_norms.{index}.weight"].float().reshape(1, 1, 1, h),
        )
        for index in range(config.num_aux_hidden_states)
    )
    # the [5H, H] fusion of the target's auxiliary rows (94 MB in BF16, read every round for the committed rows);
    # TT_LAGUNA_DFLASH_FC_WDT=bf8 stores it as bfloat8_b (teacher-forced AIME24 acceptance 2.677 vs 2.667; the round
    # time did not change: the update is not on the critical path)
    fc_dtype = {"bf16": ttnn.bfloat16, "bf8": ttnn.bfloat8_b}[os.environ.get("TT_LAGUNA_DFLASH_FC_WDT", "bf16")]
    # TT_LAGUNA_DFLASH_FC_TP (default on): each chip holds 1/D of the output columns (a quarter of the bytes read)
    # and combine_aux_hidden_states all-gathers the D column blocks
    fc_tp = devices > 1 and os.environ.get("TT_LAGUNA_DFLASH_FC_TP", "1") == "1"
    fc = cached("fc", lambda: shared["fc.weight"].float().t().contiguous(), fc_dtype, shard_dim=1 if fc_tp else None)
    hidden_norm = cached("hidden_norm", lambda: shared["hidden_norm.weight"].float().reshape(1, 1, 1, h))
    final_norm = cached("norm", lambda: shared["norm.weight"].float().reshape(1, 1, 1, h))
    return DFlashTTSharedWeights(
        aux_hidden_norms=aux,
        fc=fc,
        hidden_norm=hidden_norm,
        final_norm=final_norm,
    )


def build_dflash_draft_layer(
    state_dict: Mapping[str, torch.Tensor],
    config: LagunaDFlashConfig,
    *,
    layer_idx: int,
    mesh_device,
    max_seq_len: int,
    rope_tables: dict[str, tuple[object, object]],
    policy: PrecisionPolicy | None = None,
    cache_namespace: str = DFLASH_CACHE_NAMESPACE,
) -> MultichipDecoder:
    """Construct one namespaced dense draft layer on the existing TP decoder."""

    mapped = map_dflash_layer_state_dict(state_dict, config, layer_idx)
    decoder_config = build_dflash_decoder_config(config)
    return MultichipDecoder.from_state_dict(
        mapped,
        hf_config=decoder_config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        max_seq_len=max_seq_len,
        policy=policy or dflash_bf16_policy(),
        rope_tables=rope_tables,
        cache_namespace=cache_namespace,
    )


class DFlashTTCore:
    """Default-off TT-owned DFlash weights and one-round proposal driver.

    The target continues to own token embeddings and the column-sharded LM head.
    This core owns only the published draft checkpoint and accepts the
    target's explicit auxiliary capture.  Acceptance/verification and scheduler
    integration remain outside this isolated one-round primitive.
    """

    def __init__(
        self,
        config: LagunaDFlashConfig,
        decoder_config: DFlashDecoderConfig,
        shared: DFlashTTSharedWeights,
        layers: Mapping[int, MultichipDecoder],
        *,
        mesh_device,
        max_seq_len: int,
        rope_tables: dict[str, tuple[object, object]],
        cache_namespace: str,
    ):
        self.config = config
        self.decoder_config = decoder_config
        self.shared = shared
        self.layers = dict(layers)
        # the draft's all-reduces cover only the proposal block's rows (MultichipDecoder._dflash_query_tail)
        if os.environ.get("TT_LAGUNA_DFLASH_AR_ROWS", "1") == "1":
            for layer in self.layers.values():
                layer._dflash_ar_rows = int(config.block_size)
        self.mesh_device = mesh_device
        self.max_seq_len = max_seq_len
        self.rope_tables = rope_tables
        self.cache_namespace = cache_namespace
        half = config.head_dim // 2
        self._rope_inv_freq = 1.0 / (
            config.rope_theta ** (torch.arange(half, dtype=torch.float32) * 2.0 / config.head_dim)
        )

    def rope_window(self, start: int, rows: int):
        """(cos, sin) TILE ``[1, 1, rows, head_dim]`` for absolute positions ``[start, start + rows)``.

        Bit-identical to slicing the build_dflash_rope_tables tables, but computed on the host and uploaded:
        a device slice of the table at a new start offset builds a new device program, and the draft's start
        advances every round, so slicing compiled a program per round while serving (~50 ms each)."""

        start, rows = int(start), int(rows)
        if start < 0 or start + rows > int(self.max_seq_len):
            raise ValueError(f"DFlash RoPE window [{start}, {start + rows}) is outside the horizon {self.max_seq_len}")
        phase = torch.outer(torch.arange(start, start + rows, dtype=torch.float32), self._rope_inv_freq)
        phase = torch.cat((phase, phase), dim=-1)
        replicate = ttnn.ReplicateTensorToMesh(self.mesh_device)
        return tuple(
            ttnn.from_torch(
                table.to(torch.bfloat16).reshape(1, 1, rows, self.config.head_dim),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=replicate,
            )
            for table in (phase.cos(), phase.sin())
        )

    @classmethod
    def from_checkpoint(
        cls,
        mesh_device,
        *,
        snapshot: str | Path = DEFAULT_DFLASH_SNAPSHOT,
        layer_indices: Sequence[int] | None = None,
        max_seq_len: int | None = None,
        policy: PrecisionPolicy | None = None,
        cache_namespace: str = DFLASH_CACHE_NAMESPACE,
        enable_experimental: bool = False,
    ) -> "DFlashTTCore":
        if not enable_experimental:
            raise RuntimeError(
                "the TT DFlash core and one-round proposal path are experimental; "
                "pass enable_experimental=True for isolated qualification"
            )
        if max_seq_len is None:
            raise ValueError("max_seq_len must be supplied explicitly for bounded DFlash allocation")

        checkpoint = LagunaDFlashCheckpoint(snapshot)
        checkpoint.validate_layout()
        config = checkpoint.config
        # The draft reads the target's hidden states, embedding and LM head, so it must be the
        # draft published for the selected target. Fail before any device allocation.
        if config.target_model_id != MODEL_ID:
            raise ValueError(
                f"DFlash snapshot {snapshot} is the draft for {config.target_model_id}, but "
                f"{MODEL_ENV} selects {MODEL_ID}"
            )
        max_seq_len = int(max_seq_len)
        # Build on CPU first so invalid bounds fail before any device allocation.
        cos, sin = build_dflash_rope_tables(config, max_seq_len, dtype=torch.bfloat16)
        _cache_layer_identity(0, cache_namespace)

        if layer_indices is None:
            layer_indices = tuple(range(config.num_hidden_layers))
        layer_indices = tuple(_validate_layer_index(config, index) for index in layer_indices)
        if not layer_indices or tuple(sorted(set(layer_indices))) != layer_indices:
            raise ValueError("layer_indices must be non-empty, unique, and increasing")

        replicate = ttnn.ReplicateTensorToMesh(mesh_device)
        cos_tt = ttnn.from_torch(
            cos,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
        )
        sin_tt = ttnn.from_torch(
            sin,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
        )
        rope_tables = {"sliding_attention": (cos_tt, sin_tt)}

        shared_names = dflash_shared_checkpoint_names(config)
        shared_state = checkpoint.load_tensors(shared_names)
        shared = load_dflash_shared_weights(
            shared_state,
            config,
            mesh_device,
            cache_namespace=cache_namespace,
        )
        bf16_policy = policy or dflash_bf16_policy()
        layers: dict[int, MultichipDecoder] = {}
        for layer_idx in layer_indices:
            names = dflash_layer_checkpoint_names(config, layer_idx)
            layer_state = checkpoint.load_tensors(names)
            layers[layer_idx] = build_dflash_draft_layer(
                layer_state,
                config,
                layer_idx=layer_idx,
                mesh_device=mesh_device,
                max_seq_len=max_seq_len,
                rope_tables=rope_tables,
                policy=bf16_policy,
                cache_namespace=cache_namespace,
            )
        return cls(
            config,
            build_dflash_decoder_config(config),
            shared,
            layers,
            mesh_device=mesh_device,
            max_seq_len=max_seq_len,
            rope_tables=rope_tables,
            cache_namespace=cache_namespace,
        )

    def combine_aux_hidden_states(self, hidden_states):
        """Normalize the flattened target slices, concatenate, project, and norm.

        ``hidden_states`` must be replicated TILE ``[1, tokens, 5*hidden]``.  A
        flattened contract keeps every slice boundary tile-aligned and avoids reshaping
        through a physically padded num_aux-wide dimension.
        """

        expected_width = self.config.num_aux_hidden_states * self.config.hidden_size
        if len(hidden_states.shape) != 3 or hidden_states.shape[0] != 1 or hidden_states.shape[-1] != expected_width:
            raise ValueError(
                f"TT aux hidden states must have shape [1, tokens, {expected_width}], "
                f"got {tuple(hidden_states.shape)}"
            )
        tokens = hidden_states.shape[-2]
        flat = ttnn.reshape(hidden_states, (1, 1, tokens, expected_width))
        h = self.config.hidden_size
        # Auxiliary fusion feeds every draft layer, so a small error here is
        # amplified once per draft layer.  Reuse the draft layer's qualified HiFi4,
        # fp32-destination kernel explicitly instead of relying on TTNN's
        # operation default, whose destination-accumulation policy is not part
        # of this module's accuracy contract.
        layer0 = next(iter(self.layers.values()))
        precision_ck = layer0._ck_hifi4
        # few rows (a round's committed rows): the norms run width-sharded over 32 cores through the draft layer's
        # _rms (same HiFi4 / fp32 kernel config and epsilon; the interleaved norm used one core per 32-row tile row,
        # ~65 us each); more rows (a prefill window) keep the interleaved norm
        if tokens <= TILE and float(layer0.cfg.eps) == float(self.config.rms_norm_eps):
            def norm(x, weight):
                return layer0._rms(x, weight)
        else:
            def norm(x, weight):
                return ttnn.rms_norm(x, weight=weight, epsilon=self.config.rms_norm_eps, compute_kernel_config=precision_ck)
        normalized = []
        for index, weight in enumerate(self.shared.aux_hidden_norms):
            part = ttnn.slice(flat, [0, 0, 0, index * h], [1, 1, tokens, (index + 1) * h])
            normalized.append(norm(part, weight))
        combined = ttnn.concat(normalized, dim=-1)
        if tokens <= TILE and os.environ.get("TT_LAGUNA_DFLASH_FC_DS", "1") == "1":
            # a round's few rows: DRAM-sharded matmul over a width-sharded copy of fc (made at the first -- eager --
            # call; ~116 vs ~249 us for [6, 18432] @ [18432, 768] per chip, same error)
            fc = self.shared.fc
            k, n = int(fc.shape[-2]), int(fc.shape[-1])
            if getattr(self, "_fc_ds", None) is None:
                self._fc_ds = ttnn.to_memory_config(fc, _dram_weight_memcfg(k, n, self.mesh_device.dram_grid_size().x))
            combined = ttnn.sharded_to_interleaved(
                layer0._dram_mm(combined, fc, self._fc_ds, k, n, precision_ck), ttnn.L1_MEMORY_CONFIG
            )
        else:
            combined = ttnn.linear(combined, self.shared.fc, compute_kernel_config=precision_ck)
        if combined.shape[-1] != h:  # column-sharded fc: gather the D column blocks
            combined = ttnn.all_gather(
                combined, dim=3, cluster_axis=layer0.tp_axis, topology=layer0.ccl_topology, num_links=layer0.num_links
            )
        combined = norm(combined, self.shared.hidden_norm)
        return ttnn.reshape(combined, (1, tokens, h))

    def apply_final_norm(self, hidden_states):
        """Apply the draft checkpoint's final RMSNorm to a TT hidden tensor."""

        if hidden_states.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"DFlash final norm expects hidden width {self.config.hidden_size}, " f"got {hidden_states.shape[-1]}"
            )
        layer0 = next(iter(self.layers.values()))
        if hidden_states.shape[-2] <= TILE and float(layer0.cfg.eps) == float(self.config.rms_norm_eps):
            # query rows: width-sharded over 32 cores (the draft layers' _rms; the interleaved norm runs on one core,
            # ~65 us)
            return layer0._rms(hidden_states, self.shared.final_norm)
        return ttnn.rms_norm(
            hidden_states,
            weight=self.shared.final_norm,
            epsilon=self.config.rms_norm_eps,
            compute_kernel_config=layer0._ck_hifi4,
        )

    def allocate_proposal_cache(
        self,
        *,
        block_size: int = 32,
        enable_experimental: bool = False,
    ) -> DFlashTTProposalCache:
        """Allocate bounded per-draft-layer request state after an explicit opt-in."""

        if not bool(enable_experimental):
            raise RuntimeError(
                "DFlash proposal-cache allocation is experimental and default-off; " "pass enable_experimental=True"
            )
        return DFlashTTProposalCache(self, block_size=block_size)

    def capture_prefix(
        self,
        capture: DFlashTargetAuxCapture,
        row_count: int,
    ) -> DFlashTargetAuxCapture:
        """Retain committed verify rows and discard speculative look-ahead.

        Target verify writes all anchor+draft rows to KV, but only the known
        bonus plus the accepted draft prefix is part of the committed auxiliary
        history.  Future target verify rounds overwrite rejected KV positions.
        """

        if not isinstance(capture, DFlashTargetAuxCapture):
            raise TypeError("DFlash verify state must be a DFlashTargetAuxCapture")
        capture.validate(self.config)
        row_count = int(row_count)
        if not 1 <= row_count <= int(capture.row_count):
            raise ValueError(f"DFlash committed capture rows must be in [1, {capture.row_count}], got {row_count}")
        if row_count == int(capture.row_count):
            return capture
        hidden = ttnn.slice(
            capture.hidden_states,
            [0, 0, 0],
            [1, row_count, self.config.num_aux_hidden_states * self.config.hidden_size],
        )
        return DFlashTargetAuxCapture(
            hidden_states=hidden,
            start_position=int(capture.start_position),
            row_count=row_count,
            layer_ids=tuple(capture.layer_ids),
        )

    def _validate_target_owner(self, target_model) -> None:
        required = ("embed_prefill", "lm_head_shards_dflash", "cfg", "device")
        missing = tuple(name for name in required if not hasattr(target_model, name))
        if missing:
            raise TypeError(f"DFlash target owner is missing required attributes {missing}")
        if target_model.device is not self.mesh_device:
            raise ValueError("DFlash draft and target owner must use the identical mesh object")
        if int(target_model.cfg.hidden) != self.config.hidden_size:
            raise ValueError(f"DFlash target hidden width {target_model.cfg.hidden} != {self.config.hidden_size}")
        if int(target_model.cfg.vocab) != self.config.vocab_size:
            raise ValueError(f"DFlash target vocabulary {target_model.cfg.vocab} != {self.config.vocab_size}")

    def proposal_round(
        self,
        cache: DFlashTTProposalCache,
        *,
        target_model,
        bonus_token_id: int,
        num_speculative_tokens: int = 15,
        enable_experimental: bool = False,
    ) -> DFlashTTProposalRound:
        """Run one exact all-draft-layer anchor+mask proposal on the TT mesh.

        Each layer receives the *same* fused target context prefix while only
        the query suffix is carried from the preceding draft layer.  Consequently
        every layer independently applies its own input norm and K/V projection
        to target context, matching the official Laguna DFlash architecture.
        Context and query are locally rebased to cache position zero, while RoPE
        matrices are sliced at their absolute target positions.
        """

        if not bool(enable_experimental):
            raise RuntimeError(
                "DFlash proposal execution is experimental and default-off; " "pass enable_experimental=True"
            )
        if not isinstance(cache, DFlashTTProposalCache) or cache.core is not self:
            raise TypeError("DFlash proposal cache must be allocated by this exact core")
        cache._require_open()
        if cache.active_request_id is None:
            raise RuntimeError("begin_request must be called before a DFlash proposal")
        if tuple(self.layers) != tuple(range(self.config.num_hidden_layers)):
            raise RuntimeError(
                f"DFlash proposal execution requires all {self.config.num_hidden_layers} draft layers in order; "
                f"got {tuple(self.layers)}"
            )
        self._validate_target_owner(target_model)
        if cache.fixed_combined:
            context_start, context_rows = cache.context_bounds()
        else:
            capture = cache.target_capture()
            context_start, context_rows = int(capture.start_position), int(capture.row_count)
        last_valid_position = context_start + context_rows - 1
        block = build_proposal_block(
            self.config,
            bonus_token_id=int(bonus_token_id),
            last_valid_position=last_valid_position,
            num_speculative_tokens=int(num_speculative_tokens),
        )
        if int(num_speculative_tokens) != self.config.max_speculative_tokens:
            raise ValueError(
                "the qualified TT DFlash round requires exactly 15 sampled mask rows; " f"got {num_speculative_tokens}"
            )

        logical_query_rows = int(block.input_ids.numel())
        padded_total = math.ceil((context_rows + logical_query_rows) / cache.block_size) * cache.block_size
        if padded_total > cache.capacity:
            raise RuntimeError(
                f"DFlash proposal needs {padded_total} local rows but cache capacity is {cache.capacity}"
            )
        absolute_end = context_start + padded_total
        if absolute_end > self.max_seq_len:
            raise ValueError(
                f"DFlash proposal RoPE interval [{context_start}, {absolute_end}) exceeds "
                f"the core horizon {self.max_seq_len}"
            )

        if cache.kv_ring:
            return self._proposal_round_ring(cache, target_model, block, logical_query_rows)
        if cache.fixed_combined:
            return self._proposal_round_fixed(
                cache, target_model, block, context_start, context_rows, logical_query_rows, padded_total
            )

        # Extend the 16 semantic query rows only at the end.  These later rows
        # are causally invisible to the anchor/mask block and exist solely to
        # make the complete context+query sequence tile-aligned.
        padded_query_rows = padded_total - context_rows
        token_ids = torch.full(
            (1, padded_query_rows),
            self.config.mask_token_id,
            dtype=torch.int32,
        )
        token_ids[0, :logical_query_rows] = block.input_ids.to(dtype=torch.int32)
        token_ids_tt = ttnn.from_torch(
            token_ids,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        query_hidden = target_model.embed_prefill(token_ids_tt)
        if cache.fixed_combined:
            context_hidden = cache.combined_context()
        else:
            context_hidden = self.combine_aux_hidden_states(capture.hidden_states)

        # All draft layers share the published theta/dimension and the same
        # absolute interval, so a single pair of RoPE tensors is exact.
        rope_mats = self.rope_window(context_start, padded_total)
        for layer_idx in range(self.config.num_hidden_layers):
            # Reset context to the fused target representation at every layer;
            # carry only query hidden state through the draft stack.
            layer_input = ttnn.concat((context_hidden, query_hidden), dim=1)
            layer_output = self.layers[layer_idx].prefill_forward(
                layer_input,
                cache.kv_cache[layer_idx],
                cache.page_tables[layer_idx],
                user_id=0,
                start_pos=0,
                rope_mats=rope_mats,
            )
            query_hidden = ttnn.slice(
                layer_output,
                [0, context_rows, 0],
                [1, padded_total, self.config.hidden_size],
            )

        # Apply only the draft checkpoint's norm.  The target-owned projection
        # below is intentionally raw and must not apply target final norm.
        query_hidden = self.apply_final_norm(query_hidden)
        sampled_hidden = ttnn.slice(
            query_hidden,
            [0, 1, 0],
            [1, 1 + self.config.max_speculative_tokens, self.config.hidden_size],
        )
        logits_shards = target_model.lm_head_shards_dflash(
            sampled_hidden,
            enable_experimental=True,
        )
        return DFlashTTProposalRound(
            block=block,
            logits_shards=logits_shards,
            sampled_hidden_states=sampled_hidden,
        )

    def placement(self, rows: int, offset: int, count: int, *, transpose: bool = False):
        """0/1 matrix ``[1, rows, 32]`` with ``M[offset + i, i] = 1`` for ``i < count`` (``[1, 32, rows]`` when
        ``transpose``). ``M @ X`` puts the first ``count`` rows of a 32-row ``X`` at rows ``offset..``; the transpose
        reads them back. Built on the host as data, so a new offset runs the same device program."""

        rows, offset, count = int(rows), int(offset), int(count)
        if not (0 <= offset and 0 <= count <= 32 and offset + count <= rows):
            raise ValueError(f"DFlash placement of {count} rows at {offset} does not fit {rows} rows")
        matrix = torch.zeros((rows, 32), dtype=torch.bfloat16)
        index = torch.arange(count)
        matrix[offset + index, index] = 1.0
        if transpose:
            matrix = matrix.transpose(0, 1).contiguous()
        return ttnn.from_torch(
            matrix.unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def query_inputs(self, start: int, rows: int, query_rows: int, padded: int):
        """Host (cos, sin) ``[1, 1, 32, head_dim]`` at the query positions ``start + rows + i`` and the attention
        mask ``[1, 1, 32, padded]`` for MultichipDecoder.dflash_query_forward: query row i (local row rows + i) sees
        keys j with rows + i - (sliding_window - 1) <= j <= rows + i, as the causal sliding-window prefill does;
        rows past the query copy the last query row's mask."""

        start, rows, query_rows, padded = int(start), int(rows), int(query_rows), int(padded)
        phase = torch.outer(torch.arange(start + rows, start + rows + 32, dtype=torch.float32), self._rope_inv_freq)
        phase = torch.cat((phase, phase), dim=-1)
        hd = int(self.config.head_dim)
        local = rows + torch.arange(32).clamp(max=query_rows - 1)
        keys = torch.arange(padded)
        allowed = (keys[None, :] <= local[:, None]) & (keys[None, :] >= local[:, None] - (self.config.sliding_window - 1))
        mask = torch.where(allowed, 0.0, -1e9).reshape(1, 1, 32, padded)
        return (
            phase.cos().to(torch.bfloat16).reshape(1, 1, 32, hd),
            phase.sin().to(torch.bfloat16).reshape(1, 1, 32, hd),
            mask.to(torch.bfloat16),
        )

    def exact_matmul(self, a, b):
        """Row-selection matmul (one 1.0 per output row): HiFi4 with fp32 accumulation reproduces bf16 exactly."""

        return ttnn.matmul(a, b, compute_kernel_config=next(iter(self.layers.values()))._ck_hifi4)

    def _proposal_round_ring(self, cache, target_model, block, query_rows):
        """proposal_round over the cached context K/V rings: only the 32 query rows run through the draft layers."""

        width = self.config.hidden_size
        token_ids = torch.full((1, 32), self.config.mask_token_id, dtype=torch.int32)
        token_ids[0, :query_rows] = block.input_ids.to(dtype=torch.int32)
        replicate = ttnn.ReplicateTensorToMesh(self.mesh_device)
        token_ids_tt = ttnn.from_torch(token_ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                       device=self.mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                       mesh_mapper=replicate)  # fmt: skip
        cos, sin, mask = (
            ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=replicate)
            for t in cache.ring_query_inputs(query_rows)
        )  # fmt: skip
        query_hidden = target_model.embed_prefill(token_ids_tt)  # [1, 32, width]
        for layer_idx in range(self.config.num_hidden_layers):
            query_hidden = self.layers[layer_idx].dflash_query_forward_cached(
                query_hidden, *cache.ring_kv(layer_idx), (cos, sin), mask
            )
        query_hidden = self.apply_final_norm(query_hidden)
        sampled_hidden = ttnn.slice(query_hidden, [0, 1, 0], [1, 1 + self.config.max_speculative_tokens, width])
        logits_shards = target_model.lm_head_shards_dflash(sampled_hidden, enable_experimental=True)
        return DFlashTTProposalRound(block=block, logits_shards=logits_shards, sampled_hidden_states=sampled_hidden)

    def _proposal_round_fixed(self, cache, target_model, block, context_start, context_rows, query_rows, padded_total):
        """proposal_round over the fixed context buffer with shapes that depend only on ``padded_total``.

        The buffer holds the context in rows ``[0, context_rows)`` and zeros after it. Each layer's input is the
        first ``padded_total`` buffer rows plus the 16 query rows placed at ``context_rows`` by a 0/1 matmul, and the
        query rows are read back with the transposed matrix. The pad rows after the query are zero instead of
        carried mask-token states; they are causally invisible to the 16 query rows, so the draft output is
        unchanged. Slicing and concatenating at ``context_rows`` instead built new device programs every round
        while the context grew (one per row count)."""

        width = self.config.hidden_size
        token_ids = torch.full((1, 32), self.config.mask_token_id, dtype=torch.int32)
        token_ids[0, :query_rows] = block.input_ids.to(dtype=torch.int32)
        token_ids_tt = ttnn.from_torch(
            token_ids,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        query_hidden = target_model.embed_prefill(token_ids_tt)  # [1, 32, width]
        context_hidden = cache.buffer_rows(padded_total)  # [1, padded_total, width]; zeros from context_rows on
        place = self.placement(padded_total, context_rows, query_rows)
        take = self.placement(padded_total, context_rows, query_rows, transpose=True)
        rope_mats = self.rope_window(context_start, padded_total)
        query_only = os.environ.get("TT_LAGUNA_DFLASH_DRAFT_QUERY", "1") == "1"  # see dflash_query_forward
        if query_only:
            rope_q = tuple(
                ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device))
                for t in self.query_inputs(context_start, context_rows, query_rows, padded_total)
            )  # fmt: skip
        for layer_idx in range(self.config.num_hidden_layers):
            layer_input = ttnn.add(context_hidden, self.exact_matmul(place, query_hidden))
            if query_only:
                query_hidden = self.layers[layer_idx].dflash_query_forward(
                    layer_input, query_hidden, take, rope_mats, rope_q[:2], rope_q[2]
                )
                continue
            layer_output = self.layers[layer_idx].prefill_forward(
                layer_input,
                cache.kv_cache[layer_idx],
                cache.page_tables[layer_idx],
                user_id=0,
                start_pos=0,
                rope_mats=rope_mats,
            )
            query_hidden = self.exact_matmul(take, layer_output)  # [1, 32, width]: 16 query rows, then zeros
        query_hidden = self.apply_final_norm(query_hidden)
        sampled_hidden = ttnn.slice(query_hidden, [0, 1, 0], [1, 1 + self.config.max_speculative_tokens, width])
        logits_shards = target_model.lm_head_shards_dflash(sampled_hidden, enable_experimental=True)
        return DFlashTTProposalRound(block=block, logits_shards=logits_shards, sampled_hidden_states=sampled_hidden)


__all__ = [
    "DFLASH_CACHE_NAMESPACE",
    "DFlashDecoderConfig",
    "DFlashTTCore",
    "DFlashTTProposalCache",
    "DFlashTTProposalRound",
    "DFlashTTSharedWeights",
    "build_dflash_decoder_config",
    "build_dflash_draft_layer",
    "build_dflash_rope_tables",
    "dflash_bf16_policy",
    "dflash_layer_checkpoint_names",
    "dflash_shared_checkpoint_names",
    "load_dflash_shared_weights",
    "map_dflash_layer_state_dict",
    "map_dflash_shared_state_dict",
]
