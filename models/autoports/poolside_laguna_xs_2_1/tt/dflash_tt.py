# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Experimental TT core for the published Laguna-XS-2.1 DFlash checkpoint.

This module owns the isolated model core and request-scoped draft state.  The separate
``dflash_serving`` controller schedules verification/acceptance, and the vLLM bridge
registers that controller only when ``TT_LAGUNA_DFLASH=1``.  Core construction and
execution remain explicitly disabled unless ``enable_experimental=True`` is supplied.

The five draft layers reuse :class:`MultichipDecoder` after strict checkpoint mapping:
the published fused QKV rows are split into Q/K/V projections and a corrected HF-like
configuration describes five dense causal sliding-attention layers.  In particular,
the draft uses full 128-channel NeoX RoPE with theta 500,000; it must never inherit the
target Laguna sliding-layer theta of 10,000.
"""

from __future__ import annotations

import math
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
from .multichip_decoder import MultichipDecoder, _cache_layer_identity
from .optimized_decoder import PrecisionPolicy, _cached_device_tensor, weight_cache_key

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
        _name_or_path="poolside/Laguna-XS-2.1-DFlash",
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
    """Exact draft-owned weights outside the five decoder layers."""

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
    """Accuracy-first BF16 policy for this unoptimized core proof."""

    return PrecisionPolicy(
        attn_qkv=ttnn.bfloat16,
        attn_o=ttnn.bfloat16,
        attn_gate=ttnn.bfloat16,
        dense_ff13=ttnn.bfloat16,
        dense_ff2=ttnn.bfloat16,
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
        # Five serial draft layers amplify projection error enough to change
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
    five tiny local KV pairs and any rolling concat/slice tensor it creates, but
    never deallocates capture tensors supplied by the target model.  A request
    must be explicitly begun and ended; use-after-end and use-after-close fail
    before launching a device operation.
    """

    def __init__(self, core: "DFlashTTCore", *, block_size: int = 32):
        if tuple(core.layers) != tuple(range(core.config.num_hidden_layers)):
            raise RuntimeError(
                "a DFlash proposal cache requires all five draft layers in checkpoint order; "
                f"got {tuple(core.layers)}"
            )
        block_size = int(block_size)
        if block_size != 32:
            raise ValueError(f"DFlash TT proposal cache requires tile/block size 32, got {block_size}")
        self.core = core
        self.block_size = block_size
        self.max_context_rows = core.config.sliding_window - 1
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

    def cached(name: str, build):
        return _cached_device_tensor(
            build,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
            cache_key=weight_cache_key(f"{cache_namespace}_{name}", "shared", f"rep_d{devices}"),
        )

    h = config.hidden_size
    aux = tuple(
        cached(
            f"aux_hidden_norm_{index}",
            lambda index=index: shared[f"aux_hidden_norms.{index}.weight"].float().reshape(1, 1, 1, h),
        )
        for index in range(config.num_aux_hidden_states)
    )
    fc = cached("fc", lambda: shared["fc.weight"].float().t().contiguous())
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
    This core owns only the published five-layer draft checkpoint and accepts the
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
        self.mesh_device = mesh_device
        self.max_seq_len = max_seq_len
        self.rope_tables = rope_tables
        self.cache_namespace = cache_namespace

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
        """Normalize five flattened target slices, concatenate, project, and norm.

        ``hidden_states`` must be replicated TILE ``[1, tokens, 5*hidden]``.  A
        flattened contract keeps every slice boundary tile-aligned and avoids reshaping
        through a physically padded five-wide dimension.
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
        # amplified five times.  Reuse the draft layer's qualified HiFi4,
        # fp32-destination kernel explicitly instead of relying on TTNN's
        # operation default, whose destination-accumulation policy is not part
        # of this module's accuracy contract.
        precision_ck = next(iter(self.layers.values()))._ck_hifi4
        normalized = []
        for index, weight in enumerate(self.shared.aux_hidden_norms):
            part = ttnn.slice(flat, [0, 0, 0, index * h], [1, 1, tokens, (index + 1) * h])
            normalized.append(
                ttnn.rms_norm(
                    part,
                    weight=weight,
                    epsilon=self.config.rms_norm_eps,
                    compute_kernel_config=precision_ck,
                )
            )
        combined = ttnn.concat(normalized, dim=-1)
        combined = ttnn.linear(combined, self.shared.fc, compute_kernel_config=precision_ck)
        combined = ttnn.rms_norm(
            combined,
            weight=self.shared.hidden_norm,
            epsilon=self.config.rms_norm_eps,
            compute_kernel_config=precision_ck,
        )
        return ttnn.reshape(combined, (1, tokens, h))

    def apply_final_norm(self, hidden_states):
        """Apply the draft checkpoint's final RMSNorm to a TT hidden tensor."""

        if hidden_states.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"DFlash final norm expects hidden width {self.config.hidden_size}, " f"got {hidden_states.shape[-1]}"
            )
        return ttnn.rms_norm(
            hidden_states,
            weight=self.shared.final_norm,
            epsilon=self.config.rms_norm_eps,
            compute_kernel_config=next(iter(self.layers.values()))._ck_hifi4,
        )

    def allocate_proposal_cache(
        self,
        *,
        block_size: int = 32,
        enable_experimental: bool = False,
    ) -> DFlashTTProposalCache:
        """Allocate bounded five-layer request state after an explicit opt-in."""

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
        """Run one exact five-layer anchor+mask proposal on the TT mesh.

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
                "DFlash proposal execution requires all five draft layers in order; " f"got {tuple(self.layers)}"
            )
        self._validate_target_owner(target_model)
        capture = cache.target_capture()
        last_valid_position = capture.end_position
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

        context_rows = int(capture.row_count)
        logical_query_rows = int(block.input_ids.numel())
        padded_total = math.ceil((context_rows + logical_query_rows) / cache.block_size) * cache.block_size
        if padded_total > cache.capacity:
            raise RuntimeError(
                f"DFlash proposal needs {padded_total} local rows but cache capacity is {cache.capacity}"
            )
        absolute_end = int(capture.start_position) + padded_total
        if absolute_end > self.max_seq_len:
            raise ValueError(
                f"DFlash proposal RoPE interval [{capture.start_position}, {absolute_end}) exceeds "
                f"the core horizon {self.max_seq_len}"
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
        context_hidden = self.combine_aux_hidden_states(capture.hidden_states)

        # All five layers share the published theta/dimension and the same
        # absolute interval, so a single pair of RoPE tensors is exact.
        first_layer = self.layers[0]
        rope_mats = (
            first_layer._rope_prefill(int(capture.start_position), padded_total),
            first_layer._rope_prefill(int(capture.start_position), padded_total, sin=True),
        )
        for layer_idx in range(self.config.num_hidden_layers):
            # Reset context to the fused target representation at every layer;
            # carry only query hidden state through the five-layer draft stack.
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
