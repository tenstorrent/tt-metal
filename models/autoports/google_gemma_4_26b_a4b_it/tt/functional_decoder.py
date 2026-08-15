# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Functional TTNN decoder layer for ``google/gemma-4-26B-A4B-it``.

Contract
========

``FunctionalDecoder.from_state_dict`` accepts canonical HuggingFace checkpoint
keys for one text decoder layer, for example
``model.language_model.layers.0.self_attn.q_proj.weight``. The method performs
all host-side tensor conversion and uploads TTNN weights. The hot
``prefill_forward`` and ``decode_forward`` paths use TTNN tensors only and do
not call torch, ``ttnn.from_torch``, or ``ttnn.to_torch``.

This module targets a single 1x1 device mesh. It supports both meaningful
Gemma4 text layer kinds:

* ``sliding_attention``: Q head dim 256, 8 KV heads, 64-token cache blocks,
  sliding window 1024.
* ``full_attention``: Q head dim 512, 2 KV heads, 128-token cache block view,
  K projection reused as V before the scale-free V RMSNorm.

Tensor contracts:

* Prefill input and output: ``[batch, 1, seq_len, hidden_size]`` TILE TTNN
  tensor. The implementation serializes users through the single-user paged
  attention kernel, selecting each user's page-table row and physical pages.
  Any positive logical ``seq_len`` is accepted; the implementation pads to a
  tile internally and slices the result back to the logical length.
  ``position_cos`` and ``position_sin`` are HF-format RoPE tensors shaped
  ``[1, 1, seq_len, head_dim]`` for this layer kind. ``page_table`` is the
  paged-cache table for ``user_id``. When ``cache_position_modulo`` is set,
  non-aligned prompts write the aligned prefix with paged fill and the exact
  1..31-token logical tail with serialized paged updates; padding never becomes
  live cache state.
  Prefill above 32,768 tokens uses paged chunked SDPA for full-attention layers
  and overlapping square sliding-window SDPA slices for sliding layers.
* Decode input and output: ``[1, 1, batch, hidden_size]`` TILE TTNN tensor.
  ``current_pos`` is a device tensor shaped ``[batch]`` with one current
  position per batch slot. Decode is trace-safe when callers update stable
  input/current-position/page-table tensors before ``ttnn.execute_trace``.
* ``kv_cache`` is a ``(key_cache, value_cache)`` tuple. Sliding layers normally
  use physical shape ``[blocks, 8, 64, 256]``. Full layers may use their natural
  physical shape ``[blocks, 2, 128, 512]`` or the shared Gemma4 HMA physical
  shape ``[blocks, 8, 64, 256]``; in the latter case this decoder passes the
  required ``block_size=128`` and ``num_kv_heads=2`` view overrides.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.demos.gemma4.tt.experts import Gemma4ExpertConfig
from models.demos.gemma4.tt.experts.decode import _build_sparse_matmul_config
from models.demos.gemma4.tt.experts.operations import apply_geglu
from models.demos.gemma4.tt.experts.prefill import prefill_forward as sparse_expert_prefill
from models.demos.gemma4.tt.experts.weights import ExpertWeights

TILE_SIZE = 32
MODEL_ID = "google/gemma-4-26B-A4B-it"
HIDDEN_SIZE = 2816
MLP_INTERMEDIATE_SIZE = 2112
NUM_Q_HEADS = 16
SLIDING_NUM_KV_HEADS = 8
FULL_NUM_KV_HEADS = 2
SLIDING_HEAD_DIM = 256
FULL_HEAD_DIM = 512
NUM_EXPERTS = 128
TOP_K_EXPERTS = 8
MOE_INTERMEDIATE_SIZE = 704
PREFILL_MOE_CHUNK_SIZE = 1024
SLIDING_BLOCK_SIZE = 64
FULL_BLOCK_SIZE = 128
SLIDING_WINDOW = 1024
# The non-chunked prefill SDPA kernel silently returns incorrect results when
# the query sequence exceeds 2**15. Keep this correctness boundary aligned with
# the canonical Gemma4 attention implementation.
PREFILL_SDPA_MAX_SEQ = 32768
PREFILL_FULL_CHUNK_SIZE = 8192
PREFILL_SLIDING_CHUNK_SIZE = 30720


def _prefill_attention_path(
    seq_len: int,
    *,
    is_sliding: bool,
    has_paged_cache: bool,
    max_non_chunked_seq: int = PREFILL_SDPA_MAX_SEQ,
) -> str:
    """Return the correctness-preserving prefill attention implementation."""
    if seq_len <= max_non_chunked_seq:
        return "non_chunked"
    if is_sliding:
        return "sliding_chunked"
    if has_paged_cache:
        return "full_chunked"
    raise ValueError("long full-attention prefill requires a populated paged cache")


def _bounded_cache_fill_plan(logical_seq_len: int) -> tuple[int, tuple[int, ...]]:
    """Split a bounded-cache fill into a tile prefix and exact logical tail."""
    aligned_prefix = (logical_seq_len // TILE_SIZE) * TILE_SIZE
    return aligned_prefix, tuple(range(aligned_prefix, logical_seq_len))


@dataclass(frozen=True)
class _LayerKind:
    name: str
    num_kv_heads: int
    head_dim: int
    block_size: int
    sliding_window: int | None
    uses_k_as_v: bool

    @property
    def q_width(self) -> int:
        return NUM_Q_HEADS * self.head_dim

    @property
    def kv_width(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def qkv_width(self) -> int:
        return self.q_width + 2 * self.kv_width


SLIDING_KIND = _LayerKind(
    name="sliding_attention",
    num_kv_heads=SLIDING_NUM_KV_HEADS,
    head_dim=SLIDING_HEAD_DIM,
    block_size=SLIDING_BLOCK_SIZE,
    sliding_window=SLIDING_WINDOW,
    uses_k_as_v=False,
)
FULL_KIND = _LayerKind(
    name="full_attention",
    num_kv_heads=FULL_NUM_KV_HEADS,
    head_dim=FULL_HEAD_DIM,
    block_size=FULL_BLOCK_SIZE,
    sliding_window=None,
    uses_k_as_v=True,
)


@dataclass(frozen=True)
class _DecoderWeights:
    layer_scalar: ttnn.Tensor
    input_ln: ttnn.Tensor
    post_attn_ln: ttnn.Tensor
    pre_ff_ln: ttnn.Tensor
    post_ff_ln: ttnn.Tensor
    post_ff_ln_1: ttnn.Tensor
    post_ff_ln_2: ttnn.Tensor
    pre_ff_ln_2: ttnn.Tensor
    q_norm: ttnn.Tensor
    k_norm: ttnn.Tensor
    qkv: ttnn.Tensor
    o_proj: ttnn.Tensor
    mlp_gate: ttnn.Tensor
    mlp_up: ttnn.Tensor
    mlp_down: ttnn.Tensor
    router_scale: ttnn.Tensor
    router_proj: ttnn.Tensor
    router_per_expert_scale: ttnn.Tensor
    expert_gate: ttnn.Tensor
    expert_up: ttnn.Tensor
    expert_down: ttnn.Tensor


class FunctionalDecoder(LightweightModule):
    """Single-layer Gemma4 text decoder with paged prefill and traced decode support."""

    def __init__(
        self,
        *,
        hf_config: Any,
        layer_idx: int,
        layer_kind: _LayerKind,
        mesh_device: Any,
        weights: _DecoderWeights,
        expert_prefill_sparsity: ttnn.Tensor,
        activation_dtype: ttnn.DataType,
        eps: float,
    ) -> None:
        self.hf_config = hf_config
        self.layer_idx = layer_idx
        self.layer_kind = layer_kind
        self.mesh_device = mesh_device
        self.weights = weights
        self.expert_config = Gemma4ExpertConfig(hf_config)
        self.expert_weights = ExpertWeights(
            gate_proj=weights.expert_gate,
            up_proj=weights.expert_up,
            down_proj=weights.expert_down,
            intermediate_size_per_device=MOE_INTERMEDIATE_SIZE,
        )
        self.expert_prefill_sparsity = expert_prefill_sparsity
        self.activation_dtype = activation_dtype
        self.eps = eps
        self.router_hidden_scale = HIDDEN_SIZE**-0.5
        self.sdpa_program_config = _make_sdpa_program_config(mesh_device)
        # Real-weight A/Bs localized the acceptance-edge losses to RMS
        # normalization and sparse expert gate projection. Every other
        # operation uses the framework-default compute kernel configuration.
        self.correctness_compute_config = _make_correctness_compute_config(mesh_device)

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, Any],
        *,
        hf_config: Any,
        layer_idx: int,
        mesh_device: Any,
        weight_dtype: ttnn.DataType = ttnn.bfloat16,
        expert_weight_dtype: ttnn.DataType = ttnn.bfloat16,
        activation_dtype: ttnn.DataType = ttnn.bfloat16,
        tensor_cache_path: str | Path | None = None,
        **_: Any,
    ) -> "FunctionalDecoder":
        """Build a TTNN Gemma4 decoder layer from HuggingFace-format weights.

        Host-side torch manipulation is intentionally confined to this setup
        method. Forward calls operate only on TTNN tensors.
        """
        import torch

        text_config = _text_config(hf_config)
        _validate_text_config(text_config)
        layer_kind = _layer_kind(text_config.layer_types[layer_idx])
        prefix = _detect_layer_prefix(state_dict, layer_idx)
        cache_root = Path(tensor_cache_path) if tensor_cache_path is not None else None

        def get(name: str):
            return state_dict[f"{prefix}.{name}"]

        def as_tt(
            name: str,
            source: Any,
            *,
            dtype: ttnn.DataType = weight_dtype,
            layout: ttnn.Layout = ttnn.TILE_LAYOUT,
            memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
        ) -> ttnn.Tensor:
            kwargs = {
                "device": mesh_device,
                "layout": layout,
                "dtype": dtype,
                "memory_config": memory_config,
            }
            mapper = _replicate_mapper(mesh_device)
            if mapper is not None:
                kwargs["mesh_mapper"] = mapper
            if cache_root is not None:
                kwargs["cache_file_name"] = str(cache_root / f"layer_{layer_idx}" / name)
            return ttnn.as_tensor(source, **kwargs)

        q = get("self_attn.q_proj.weight").transpose(-2, -1).contiguous()
        k = get("self_attn.k_proj.weight").transpose(-2, -1).contiguous()
        if layer_kind.uses_k_as_v:
            v = k
        else:
            v = get("self_attn.v_proj.weight").transpose(-2, -1).contiguous()
        qkv = torch.cat([q, k, v], dim=-1).unsqueeze(0).unsqueeze(0)

        mlp_gate = get("mlp.gate_proj.weight").transpose(-2, -1).contiguous().unsqueeze(0).unsqueeze(0)
        mlp_up = get("mlp.up_proj.weight").transpose(-2, -1).contiguous().unsqueeze(0).unsqueeze(0)
        mlp_down = get("mlp.down_proj.weight").transpose(-2, -1).contiguous().unsqueeze(0).unsqueeze(0)

        gate_up = get("experts.gate_up_proj")
        expert_gate = gate_up[:, :MOE_INTERMEDIATE_SIZE, :].transpose(-2, -1).contiguous().unsqueeze(0)
        expert_up = gate_up[:, MOE_INTERMEDIATE_SIZE:, :].transpose(-2, -1).contiguous().unsqueeze(0)
        expert_down = get("experts.down_proj").transpose(-2, -1).contiguous().unsqueeze(0)
        expert_prefill_sparsity = as_tt(
            "expert_prefill_sparsity",
            torch.ones(1, 1, 1, NUM_EXPERTS, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        weights = _DecoderWeights(
            layer_scalar=as_tt("layer_scalar", get("layer_scalar").reshape(1, 1, 1, 1), dtype=ttnn.bfloat16),
            input_ln=as_tt("input_ln", get("input_layernorm.weight").reshape(1, 1, 1, HIDDEN_SIZE)),
            post_attn_ln=as_tt("post_attn_ln", get("post_attention_layernorm.weight").reshape(1, 1, 1, HIDDEN_SIZE)),
            pre_ff_ln=as_tt("pre_ff_ln", get("pre_feedforward_layernorm.weight").reshape(1, 1, 1, HIDDEN_SIZE)),
            post_ff_ln=as_tt("post_ff_ln", get("post_feedforward_layernorm.weight").reshape(1, 1, 1, HIDDEN_SIZE)),
            post_ff_ln_1=as_tt(
                "post_ff_ln_1", get("post_feedforward_layernorm_1.weight").reshape(1, 1, 1, HIDDEN_SIZE)
            ),
            post_ff_ln_2=as_tt(
                "post_ff_ln_2", get("post_feedforward_layernorm_2.weight").reshape(1, 1, 1, HIDDEN_SIZE)
            ),
            pre_ff_ln_2=as_tt("pre_ff_ln_2", get("pre_feedforward_layernorm_2.weight").reshape(1, 1, 1, HIDDEN_SIZE)),
            q_norm=as_tt("q_norm", get("self_attn.q_norm.weight").reshape(1, 1, 1, layer_kind.head_dim)),
            k_norm=as_tt("k_norm", get("self_attn.k_norm.weight").reshape(1, 1, 1, layer_kind.head_dim)),
            qkv=as_tt("qkv", qkv),
            o_proj=as_tt(
                "o_proj", get("self_attn.o_proj.weight").transpose(-2, -1).contiguous().unsqueeze(0).unsqueeze(0)
            ),
            mlp_gate=as_tt("mlp_gate", mlp_gate),
            mlp_up=as_tt("mlp_up", mlp_up),
            mlp_down=as_tt("mlp_down", mlp_down),
            router_scale=as_tt(
                "router_scale",
                get("router.scale").reshape(1, 1, 1, HIDDEN_SIZE),
                dtype=ttnn.float32,
            ),
            router_proj=as_tt(
                "router_proj",
                get("router.proj.weight").transpose(-2, -1).contiguous().unsqueeze(0).unsqueeze(0),
                dtype=ttnn.float32,
            ),
            router_per_expert_scale=as_tt(
                "router_per_expert_scale",
                get("router.per_expert_scale").reshape(1, NUM_EXPERTS),
                dtype=ttnn.float32,
            ),
            expert_gate=as_tt("expert_gate", expert_gate, dtype=expert_weight_dtype),
            expert_up=as_tt("expert_up", expert_up, dtype=expert_weight_dtype),
            expert_down=as_tt("expert_down", expert_down, dtype=expert_weight_dtype),
        )

        return cls(
            hf_config=text_config,
            layer_idx=layer_idx,
            layer_kind=layer_kind,
            mesh_device=mesh_device,
            weights=weights,
            expert_prefill_sparsity=expert_prefill_sparsity,
            activation_dtype=activation_dtype,
            eps=text_config.rms_norm_eps,
        )

    def prefill_forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        position_cos: ttnn.Tensor,
        position_sin: ttnn.Tensor,
        page_table: ttnn.Tensor,
        kv_cache: tuple[ttnn.Tensor, ttnn.Tensor],
        user_id: int = 0,
        chunk_page_table: ttnn.Tensor | None = None,
        cache_position_modulo: int | None = None,
    ) -> ttnn.Tensor:
        """Run paged prefill for one or more users at a common sequence length.

        TTNN's paged prefill kernels consume one user at a time. For batch > 1
        this wrapper slices only device tensors, selects the corresponding
        page-table row, runs the same single-user implementation, and
        concatenates the outputs on device. The hot path has no host fallback.
        """
        batch = hidden_states.shape[0]
        if batch < 1:
            raise ValueError("prefill requires at least one batch row")
        if batch == 1:
            return self._prefill_forward_single_user(
                hidden_states,
                position_cos=position_cos,
                position_sin=position_sin,
                page_table=page_table,
                kv_cache=kv_cache,
                user_id=user_id,
                chunk_page_table=chunk_page_table,
                cache_position_modulo=cache_position_modulo,
            )
        if user_id + batch > page_table.shape[0]:
            raise ValueError(
                f"prefill batch rows [{user_id}, {user_id + batch}) exceed " f"page table batch {page_table.shape[0]}"
            )

        outputs = []
        for batch_index in range(batch):
            table_index = user_id + batch_index
            hidden_row = ttnn.slice(
                hidden_states,
                [batch_index, 0, 0, 0],
                [batch_index + 1, 1, hidden_states.shape[2], hidden_states.shape[3]],
            )
            cos_row = ttnn.slice(
                position_cos,
                [batch_index, 0, 0, 0],
                [batch_index + 1, 1, position_cos.shape[2], position_cos.shape[3]],
            )
            sin_row = ttnn.slice(
                position_sin,
                [batch_index, 0, 0, 0],
                [batch_index + 1, 1, position_sin.shape[2], position_sin.shape[3]],
            )
            page_table_row = ttnn.slice(
                page_table,
                [table_index, 0],
                [table_index + 1, page_table.shape[1]],
            )
            chunk_page_table_row = None
            if chunk_page_table is not None:
                chunk_page_table_row = ttnn.slice(
                    chunk_page_table,
                    [table_index, 0],
                    [table_index + 1, chunk_page_table.shape[1]],
                )
            outputs.append(
                self._prefill_forward_single_user(
                    hidden_row,
                    position_cos=cos_row,
                    position_sin=sin_row,
                    page_table=page_table_row,
                    kv_cache=kv_cache,
                    user_id=0,
                    chunk_page_table=chunk_page_table_row,
                    cache_position_modulo=cache_position_modulo,
                )
            )
        return ttnn.concat(outputs, dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _prefill_forward_single_user(
        self,
        hidden_states: ttnn.Tensor,
        *,
        position_cos: ttnn.Tensor,
        position_sin: ttnn.Tensor,
        page_table: ttnn.Tensor,
        kv_cache: tuple[ttnn.Tensor, ttnn.Tensor],
        user_id: int,
        chunk_page_table: ttnn.Tensor | None,
        cache_position_modulo: int | None,
    ) -> ttnn.Tensor:
        """Run paged prefill for one user and any positive logical length.

        TTNN kernels operate on a tile-aligned physical sequence. Padding is
        internal and the returned tensor is sliced back to the logical length.
        """
        logical_seq_len = hidden_states.shape[-2]
        if logical_seq_len < 1:
            raise ValueError("prefill requires at least one logical token")
        padded_seq_len = ((logical_seq_len + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
        if padded_seq_len != logical_seq_len:
            pad = [(0, 0), (0, 0), (0, padded_seq_len - logical_seq_len), (0, 0)]
            hidden_states = ttnn.pad(hidden_states, pad, 0.0)
            position_cos = ttnn.pad(position_cos, pad, 0.0)
            position_sin = ttnn.pad(position_sin, pad, 0.0)

        residual = hidden_states
        attn_in = self._rms_norm(hidden_states, self.weights.input_ln)
        attn_out = self._attention_prefill(
            attn_in,
            position_cos=position_cos,
            position_sin=position_sin,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            kv_cache=kv_cache,
            user_id=user_id,
            cache_position_modulo=cache_position_modulo,
            logical_seq_len=logical_seq_len,
        )
        attn_out = self._rms_norm(attn_out, self.weights.post_attn_ln)
        hidden_states = ttnn.add(residual, attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        residual = hidden_states
        mlp_in = self._rms_norm(hidden_states, self.weights.pre_ff_ln)
        mlp_out = self._dense_mlp(mlp_in)
        hidden_1 = self._rms_norm(mlp_out, self.weights.post_ff_ln_1)

        router_weights = self._router_weights(residual)
        moe_in = self._rms_norm(residual, self.weights.pre_ff_ln_2)
        hidden_2 = self._moe_prefill(moe_in, router_weights)
        hidden_2 = self._rms_norm(hidden_2, self.weights.post_ff_ln_2)

        hidden_states = ttnn.add(hidden_1, hidden_2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        hidden_states = self._rms_norm(hidden_states, self.weights.post_ff_ln)
        hidden_states = ttnn.add(residual, hidden_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        hidden_states = self._apply_layer_scalar(hidden_states)
        if padded_seq_len != logical_seq_len:
            hidden_states = ttnn.slice(
                hidden_states,
                starts=[0, 0, 0, 0],
                ends=[1, 1, logical_seq_len, HIDDEN_SIZE],
                steps=[1, 1, 1, 1],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return hidden_states

    def decode_forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        position_cos: ttnn.Tensor,
        position_sin: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        page_table: ttnn.Tensor,
        kv_cache: tuple[ttnn.Tensor, ttnn.Tensor],
        cache_position_modulo: int | None = None,
    ) -> ttnn.Tensor:
        """Run paged decode for ``[1, 1, batch, hidden]`` input.

        This method is trace-capture safe when all input tensors, ``current_pos``,
        ``page_table``, and ``kv_cache`` buffers are stable allocations.
        """
        if hidden_states.shape[-2] < 1:
            raise ValueError("decode requires at least one batch row")

        residual = hidden_states
        attn_in = self._rms_norm(hidden_states, self.weights.input_ln)
        attn_out = self._attention_decode(
            attn_in,
            position_cos=position_cos,
            position_sin=position_sin,
            current_pos=current_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            cache_position_modulo=cache_position_modulo,
        )
        attn_out = self._rms_norm(attn_out, self.weights.post_attn_ln)
        hidden_states = ttnn.add(residual, attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        residual = hidden_states
        mlp_in = self._rms_norm(hidden_states, self.weights.pre_ff_ln)
        mlp_out = self._dense_mlp(mlp_in)
        hidden_1 = self._rms_norm(mlp_out, self.weights.post_ff_ln_1)

        router_weights = self._router_weights(residual)
        moe_in = self._rms_norm(residual, self.weights.pre_ff_ln_2)
        hidden_2 = self._moe_decode(moe_in, router_weights)
        hidden_2 = self._rms_norm(hidden_2, self.weights.post_ff_ln_2)

        hidden_states = ttnn.add(hidden_1, hidden_2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        hidden_states = self._rms_norm(hidden_states, self.weights.post_ff_ln)
        hidden_states = ttnn.add(residual, hidden_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return self._apply_layer_scalar(hidden_states)

    def forward(self, hidden_states: ttnn.Tensor, *, mode: str, **kwargs: Any) -> ttnn.Tensor:
        if mode == "prefill":
            return self.prefill_forward(hidden_states, **kwargs)
        if mode == "decode":
            return self.decode_forward(hidden_states, **kwargs)
        raise ValueError(f"mode must be 'prefill' or 'decode', got {mode!r}")

    def _rms_norm(
        self,
        x: ttnn.Tensor,
        weight: ttnn.Tensor | None,
    ) -> ttnn.Tensor:
        return ttnn.rms_norm(
            x,
            epsilon=self.eps,
            weight=weight,
            compute_kernel_config=self.correctness_compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _apply_layer_scalar(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.mul(hidden_states, self.weights.layer_scalar, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _attention_prefill(
        self,
        x: ttnn.Tensor,
        *,
        position_cos: ttnn.Tensor,
        position_sin: ttnn.Tensor,
        page_table: ttnn.Tensor,
        chunk_page_table: ttnn.Tensor | None,
        kv_cache: tuple[ttnn.Tensor, ttnn.Tensor],
        user_id: int,
        cache_position_modulo: int | None,
        logical_seq_len: int,
    ) -> ttnn.Tensor:
        kind = self.layer_kind
        seq_len = x.shape[-2]
        xqkv = ttnn.linear(x, self.weights.qkv, dtype=self.activation_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=NUM_Q_HEADS,
            num_kv_heads=kind.num_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        q_heads = self._rms_norm(q_heads, self.weights.q_norm)
        k_heads = self._rms_norm(k_heads, self.weights.k_norm)
        v_heads = self._rms_norm(v_heads, None)

        q_heads = ttnn.experimental.rotary_embedding_hf(q_heads, position_cos, position_sin, is_decode_mode=False)
        k_heads = ttnn.experimental.rotary_embedding_hf(k_heads, position_cos, position_sin, is_decode_mode=False)

        key_cache, value_cache = kv_cache
        fill_table = chunk_page_table if chunk_page_table is not None else page_table
        fill_kwargs = self._cache_view_kwargs(prefill=True)
        self._fill_prefill_cache(
            key_cache,
            value_cache,
            k_heads,
            v_heads,
            fill_table,
            user_id=user_id,
            logical_seq_len=logical_seq_len,
            cache_position_modulo=cache_position_modulo,
            fill_kwargs=fill_kwargs,
        )

        attention_path = _prefill_attention_path(
            seq_len,
            is_sliding=kind.sliding_window is not None,
            has_paged_cache=fill_table is not None,
        )
        if attention_path == "sliding_chunked":
            attn_out = self._sliding_chunked_prefill_attention(q_heads, k_heads, v_heads)
        elif attention_path == "full_chunked":
            attn_out = self._full_chunked_prefill_attention(
                q_heads,
                key_cache,
                value_cache,
                fill_table,
                user_id=user_id,
            )
        else:
            attn_out = ttnn.transformer.scaled_dot_product_attention(
                q_heads,
                k_heads,
                v_heads,
                is_causal=True,
                sliding_window_size=kind.sliding_window,
                scale=1.0,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        attn_out = ttnn.reshape(attn_out, [1, NUM_Q_HEADS, seq_len, kind.head_dim])
        attn_out = ttnn.experimental.nlp_concat_heads(attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.linear(
            attn_out,
            self.weights.o_proj,
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _fill_prefill_cache(
        self,
        key_cache: ttnn.Tensor,
        value_cache: ttnn.Tensor,
        k_heads: ttnn.Tensor,
        v_heads: ttnn.Tensor,
        page_table: ttnn.Tensor,
        *,
        user_id: int,
        logical_seq_len: int,
        cache_position_modulo: int | None,
        fill_kwargs: dict[str, int],
    ) -> None:
        """Fill paged K/V without allowing padded rows to wrap over live data."""
        if cache_position_modulo is None or logical_seq_len % TILE_SIZE == 0:
            modulo_kwargs = (
                {"cache_position_modulo": cache_position_modulo} if cache_position_modulo is not None else {}
            )
            ttnn.experimental.paged_fill_cache(
                key_cache, k_heads, page_table, batch_idx=user_id, **fill_kwargs, **modulo_kwargs
            )
            ttnn.experimental.paged_fill_cache(
                value_cache, v_heads, page_table, batch_idx=user_id, **fill_kwargs, **modulo_kwargs
            )
            return

        aligned_prefix, tail_positions = _bounded_cache_fill_plan(logical_seq_len)
        if aligned_prefix:
            k_prefix = ttnn.slice(
                k_heads,
                [0, 0, 0, 0],
                [k_heads.shape[0], k_heads.shape[1], aligned_prefix, k_heads.shape[3]],
            )
            v_prefix = ttnn.slice(
                v_heads,
                [0, 0, 0, 0],
                [v_heads.shape[0], v_heads.shape[1], aligned_prefix, v_heads.shape[3]],
            )
            ttnn.experimental.paged_fill_cache(
                key_cache,
                k_prefix,
                page_table,
                batch_idx=user_id,
                cache_position_modulo=cache_position_modulo,
                **fill_kwargs,
            )
            ttnn.experimental.paged_fill_cache(
                value_cache,
                v_prefix,
                page_table,
                batch_idx=user_id,
                cache_position_modulo=cache_position_modulo,
                **fill_kwargs,
            )
            k_prefix.deallocate(True)
            v_prefix.deallocate(True)

        # paged_fill_cache writes complete tiles. Write the 1..31 real tail
        # rows individually so physical padding can never wrap over live cache
        # rows. A one-user height shard is the minimal layout required by
        # paged_update_cache; serial calls also avoid same-tile RMW races.
        page_table_row = page_table
        owns_page_table_row = False
        if page_table.shape[0] > 1:
            page_table_row = ttnn.slice(
                page_table,
                [user_id, 0],
                [user_id + 1, page_table.shape[1]],
            )
            owns_page_table_row = True
        update_mem_config = _make_single_user_cache_update_memory_config(self.mesh_device, self.layer_kind.head_dim)
        update_kwargs = self._cache_view_kwargs(prefill=False)
        update_kwargs["cache_position_modulo"] = cache_position_modulo
        for position in tail_positions:
            k_token = ttnn.slice(
                k_heads,
                [0, 0, position, 0],
                [1, self.layer_kind.num_kv_heads, position + 1, self.layer_kind.head_dim],
            )
            v_token = ttnn.slice(
                v_heads,
                [0, 0, position, 0],
                [1, self.layer_kind.num_kv_heads, position + 1, self.layer_kind.head_dim],
            )
            k_token = ttnn.transpose(k_token, 1, 2)
            v_token = ttnn.transpose(v_token, 1, 2)
            k_token = ttnn.to_memory_config(k_token, update_mem_config, dtype=k_token.dtype)
            v_token = ttnn.to_memory_config(v_token, update_mem_config, dtype=v_token.dtype)
            position_tensor = ttnn.full(
                (1,),
                position,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.experimental.paged_update_cache(
                key_cache,
                k_token,
                update_idxs_tensor=position_tensor,
                page_table=page_table_row,
                **update_kwargs,
            )
            ttnn.experimental.paged_update_cache(
                value_cache,
                v_token,
                update_idxs_tensor=position_tensor,
                page_table=page_table_row,
                **update_kwargs,
            )
            k_token.deallocate(True)
            v_token.deallocate(True)
            position_tensor.deallocate(True)
        if owns_page_table_row:
            page_table_row.deallocate(True)

    def _full_chunked_prefill_attention(
        self,
        q_heads: ttnn.Tensor,
        key_cache: ttnn.Tensor,
        value_cache: ttnn.Tensor,
        page_table: ttnn.Tensor,
        *,
        user_id: int,
    ) -> ttnn.Tensor:
        """Run long causal prefill from the populated paged cache."""
        num_pages = page_table.shape[-1]
        user_page_table = page_table
        owns_user_page_table = False
        if page_table.shape[0] > 1:
            user_page_table = ttnn.slice(page_table, [user_id, 0], [user_id + 1, num_pages])
            owns_user_page_table = True

        outputs = []
        seq_len = q_heads.shape[-2]
        start = 0
        while start < seq_len:
            chunk_len = min(PREFILL_FULL_CHUNK_SIZE, seq_len - start)
            q_chunk = ttnn.slice(
                q_heads,
                [0, 0, start, 0],
                [1, NUM_Q_HEADS, start + chunk_len, self.layer_kind.head_dim],
            )
            output = ttnn.transformer.chunked_scaled_dot_product_attention(
                q_chunk,
                key_cache,
                value_cache,
                user_page_table,
                chunk_start_idx=start,
                scale=1.0,
                compute_kernel_config=self.correctness_compute_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            q_chunk.deallocate(True)
            outputs.append(output)
            start += chunk_len
        if owns_user_page_table:
            user_page_table.deallocate(True)
        if len(outputs) == 1:
            return outputs[0]
        result = ttnn.concat(outputs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for output in outputs:
            output.deallocate(True)
        return result

    def _sliding_chunked_prefill_attention(
        self,
        q_heads: ttnn.Tensor,
        k_heads: ttnn.Tensor,
        v_heads: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Run long sliding prefill as overlapping square SDPA windows."""
        seq_len = q_heads.shape[-2]
        history = ((self.layer_kind.sliding_window + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
        outputs = []
        start = 0
        while start < seq_len:
            output_len = min(PREFILL_SLIDING_CHUNK_SIZE, seq_len - start)
            slice_start = max(0, start - history)
            slice_end = start + output_len
            q_slice = ttnn.slice(
                q_heads,
                [0, 0, slice_start, 0],
                [1, NUM_Q_HEADS, slice_end, self.layer_kind.head_dim],
            )
            k_slice = ttnn.slice(
                k_heads,
                [0, 0, slice_start, 0],
                [1, self.layer_kind.num_kv_heads, slice_end, self.layer_kind.head_dim],
            )
            v_slice = ttnn.slice(
                v_heads,
                [0, 0, slice_start, 0],
                [1, self.layer_kind.num_kv_heads, slice_end, self.layer_kind.head_dim],
            )
            output = ttnn.transformer.scaled_dot_product_attention(
                q_slice,
                k_slice,
                v_slice,
                is_causal=True,
                sliding_window_size=self.layer_kind.sliding_window,
                scale=1.0,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            q_slice.deallocate(True)
            k_slice.deallocate(True)
            v_slice.deallocate(True)
            drop = start - slice_start
            if drop:
                full_output = output
                output = ttnn.slice(
                    full_output,
                    [0, 0, drop, 0],
                    [1, NUM_Q_HEADS, slice_end - slice_start, self.layer_kind.head_dim],
                )
                full_output.deallocate(True)
            outputs.append(output)
            start += output_len
        if len(outputs) == 1:
            return outputs[0]
        result = ttnn.concat(outputs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for output in outputs:
            output.deallocate(True)
        return result

    def _attention_decode(
        self,
        x: ttnn.Tensor,
        *,
        position_cos: ttnn.Tensor,
        position_sin: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        page_table: ttnn.Tensor,
        kv_cache: tuple[ttnn.Tensor, ttnn.Tensor],
        cache_position_modulo: int | None,
    ) -> ttnn.Tensor:
        kind = self.layer_kind
        batch = x.shape[-2]
        xqkv = ttnn.linear(x, self.weights.qkv, dtype=self.activation_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        qkv_head_mem_config = _make_decode_height_sharded_memory_config(
            self.mesh_device,
            batch,
            kind.head_dim,
        )
        q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv,
            num_heads=NUM_Q_HEADS,
            num_kv_heads=kind.num_kv_heads,
            memory_config=qkv_head_mem_config,
        )
        q_mem_config = q_heads.memory_config()
        k_mem_config = k_heads.memory_config()
        v_mem_config = v_heads.memory_config()
        q_heads = ttnn.to_memory_config(q_heads, ttnn.L1_MEMORY_CONFIG, dtype=q_heads.dtype)
        k_heads = ttnn.to_memory_config(k_heads, ttnn.L1_MEMORY_CONFIG, dtype=k_heads.dtype)
        v_heads = ttnn.to_memory_config(v_heads, ttnn.L1_MEMORY_CONFIG, dtype=v_heads.dtype)
        q_heads = self._rms_norm(q_heads, self.weights.q_norm)
        k_heads = self._rms_norm(k_heads, self.weights.k_norm)
        v_heads = self._rms_norm(v_heads, None)
        if kind.name == "full_attention":
            q_heads = ttnn.transpose(q_heads, 1, 2)
            k_heads = ttnn.transpose(k_heads, 1, 2)
            q_heads = ttnn.experimental.rotary_embedding_hf(q_heads, position_cos, position_sin, is_decode_mode=False)
            k_heads = ttnn.experimental.rotary_embedding_hf(k_heads, position_cos, position_sin, is_decode_mode=False)
            q_heads = ttnn.transpose(q_heads, 1, 2)
            k_heads = ttnn.transpose(k_heads, 1, 2)
            q_heads = ttnn.to_memory_config(q_heads, q_mem_config, dtype=q_heads.dtype)
            k_heads = ttnn.to_memory_config(k_heads, k_mem_config, dtype=k_heads.dtype)
            v_heads = ttnn.to_memory_config(v_heads, v_mem_config, dtype=v_heads.dtype)
        else:
            q_heads = ttnn.to_memory_config(q_heads, q_mem_config, dtype=q_heads.dtype)
            k_heads = ttnn.to_memory_config(k_heads, k_mem_config, dtype=k_heads.dtype)
            v_heads = ttnn.to_memory_config(v_heads, v_mem_config, dtype=v_heads.dtype)
            rope_mem_config = _make_decode_rope_memory_config(self.mesh_device, batch, kind.head_dim)
            position_cos = ttnn.interleaved_to_sharded(position_cos, rope_mem_config)
            position_sin = ttnn.interleaved_to_sharded(position_sin, rope_mem_config)
            q_heads = ttnn.experimental.rotary_embedding_hf(q_heads, position_cos, position_sin, is_decode_mode=True)
            k_heads = ttnn.experimental.rotary_embedding_hf(k_heads, position_cos, position_sin, is_decode_mode=True)

        key_cache, value_cache = kv_cache
        update_kwargs = self._cache_view_kwargs(prefill=False)
        if cache_position_modulo is not None:
            update_kwargs["cache_position_modulo"] = cache_position_modulo
        ttnn.experimental.paged_update_cache(
            key_cache,
            k_heads,
            update_idxs_tensor=current_pos,
            page_table=page_table,
            **update_kwargs,
        )
        ttnn.experimental.paged_update_cache(
            value_cache,
            v_heads,
            update_idxs_tensor=current_pos,
            page_table=page_table,
            **update_kwargs,
        )

        sdpa_kwargs = self._cache_view_kwargs(prefill=False)
        attn_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_heads,
            key_cache,
            value_cache,
            page_table_tensor=page_table,
            cur_pos_tensor=current_pos,
            scale=1.0,
            sliding_window_size=kind.sliding_window,
            program_config=self.sdpa_program_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **sdpa_kwargs,
        )
        concat_mem_config = _make_decode_height_sharded_memory_config(
            self.mesh_device,
            batch,
            kind.head_dim,
        )
        attn_out = ttnn.to_memory_config(attn_out, concat_mem_config, dtype=attn_out.dtype)
        attn_out = ttnn.experimental.nlp_concat_heads_decode(attn_out, num_heads=NUM_Q_HEADS)
        attn_out = ttnn.sharded_to_interleaved(attn_out, ttnn.DRAM_MEMORY_CONFIG)
        attn_out = ttnn.linear(
            attn_out,
            self.weights.o_proj,
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if attn_out.shape[-2] != batch:
            attn_out = ttnn.slice(
                attn_out,
                starts=[0, 0, 0, 0],
                ends=[1, 1, batch, HIDDEN_SIZE],
                steps=[1, 1, 1, 1],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return attn_out

    def _cache_view_kwargs(self, *, prefill: bool) -> dict[str, int]:
        if self.layer_kind.name != "full_attention":
            return {}
        kwargs = {"block_size": self.layer_kind.block_size}
        if not prefill:
            kwargs["num_kv_heads"] = self.layer_kind.num_kv_heads
        return kwargs

    def _dense_mlp(self, x: ttnn.Tensor) -> ttnn.Tensor:
        gate = ttnn.linear(
            x,
            self.weights.mlp_gate,
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        up = ttnn.linear(
            x,
            self.weights.mlp_up,
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gate = ttnn.gelu(gate, fast_and_approximate_mode=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        hidden = ttnn.mul(gate, up, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.linear(
            hidden,
            self.weights.mlp_down,
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _router_weights(self, residual: ttnn.Tensor) -> ttnn.Tensor:
        tokens = residual.shape[-2]
        router_in = self._rms_norm(residual, None)
        router_in = ttnn.mul(router_in, self.weights.router_scale, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        router_in = ttnn.mul(router_in, self.router_hidden_scale, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        router_in = ttnn.reshape(router_in, [tokens, HIDDEN_SIZE])
        router_in = ttnn.typecast(router_in, ttnn.float32)
        logits = ttnn.linear(
            router_in,
            self.weights.router_proj,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        logits = ttnn.typecast(logits, ttnn.bfloat16)
        top_values, top_indices = ttnn.topk(logits, k=TOP_K_EXPERTS, dim=-1, sorted=True)
        top_values = ttnn.softmax(
            top_values,
            dim=-1,
            numeric_stable=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        routing = ttnn.scatter(ttnn.zeros_like(logits), dim=-1, index=top_indices, src=top_values)
        routing = ttnn.mul(routing, self.weights.router_per_expert_scale, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        routing = ttnn.typecast(routing, ttnn.bfloat16)
        return ttnn.reshape(routing, [1, 1, tokens, NUM_EXPERTS])

    def _moe_decode(self, hidden_states: ttnn.Tensor, routing_weights: ttnn.Tensor) -> ttnn.Tensor:
        """Run sparse experts for each independent decode routing mask.

        sparse_matmul's expert-weight batch is 128, so one invocation accepts
        one 128-entry sparsity row. Decode batches are serialized as TTNN
        slices and concatenated on device; all calls remain trace-capturable.
        """
        batch = hidden_states.shape[2]
        if batch == 1:
            return self._moe_decode_single_user(hidden_states, routing_weights)

        outputs = []
        for batch_index in range(batch):
            hidden_row = ttnn.slice(
                hidden_states,
                [0, 0, batch_index, 0],
                [1, 1, batch_index + 1, HIDDEN_SIZE],
            )
            routing_row = ttnn.slice(
                routing_weights,
                [0, 0, batch_index, 0],
                [1, 1, batch_index + 1, NUM_EXPERTS],
            )
            outputs.append(self._moe_decode_single_user(hidden_row, routing_row))
        return ttnn.concat(outputs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _moe_decode_single_user(
        self,
        hidden_states: ttnn.Tensor,
        routing_weights: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Run top-k sparse experts, with high accumulation only on gate-proj.

        A real-weight sparse-vs-dense A/B localized the acceptance-edge PCC
        loss to sparse expert accumulation. A down-only high-accumulation
        control worsened PCC, so up/down retain framework defaults; the gate
        projection alone uses the decoder's recorded correctness config.
        """
        batch = hidden_states.shape[2]
        sparsity = ttnn.to_layout(routing_weights, ttnn.ROW_MAJOR_LAYOUT)
        output_tile = ttnn.Tile([TILE_SIZE, TILE_SIZE])
        gate_up_config = _build_sparse_matmul_config(batch, MOE_INTERMEDIATE_SIZE)
        down_config = _build_sparse_matmul_config(batch, HIDDEN_SIZE)

        gate = ttnn.sparse_matmul(
            hidden_states,
            self.weights.expert_gate,
            sparsity=sparsity,
            nnz=TOP_K_EXPERTS,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=gate_up_config,
            dtype=self.activation_dtype,
            compute_kernel_config=self.correctness_compute_config,
        )
        sparse_intermediate = gate.shape[-1]
        gate = ttnn.reshape(gate, (batch, NUM_EXPERTS, 1, sparse_intermediate))
        gate = ttnn.transpose(gate, 1, 2)
        gate = ttnn.reshape(gate, (batch, NUM_EXPERTS, sparse_intermediate))

        up = ttnn.sparse_matmul(
            hidden_states,
            self.weights.expert_up,
            sparsity=sparsity,
            nnz=TOP_K_EXPERTS,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=gate_up_config,
            dtype=self.activation_dtype,
        )
        up = ttnn.reshape(up, (batch, NUM_EXPERTS, 1, sparse_intermediate))
        up = ttnn.transpose(up, 1, 2)
        up = ttnn.reshape(up, (batch, NUM_EXPERTS, sparse_intermediate))
        down_input = apply_geglu(gate, up)
        down_input = ttnn.transpose(down_input, 1, 0)
        down_input = ttnn.reshape(down_input, (1, NUM_EXPERTS, batch, sparse_intermediate))

        down = ttnn.sparse_matmul(
            down_input,
            self.weights.expert_down,
            sparsity=sparsity,
            nnz=TOP_K_EXPERTS,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=down_config,
            is_input_a_sparse=True,
            dtype=self.activation_dtype,
        )
        next_states = ttnn.permute(down, (0, 2, 1, 3))
        next_states = ttnn.reshape(next_states, (batch, NUM_EXPERTS, HIDDEN_SIZE))
        routing_3d = ttnn.reshape(routing_weights, (batch, NUM_EXPERTS, 1))
        next_states = ttnn.mul(next_states, routing_3d)
        next_states = ttnn.sum(next_states, dim=1)
        next_states = ttnn.unsqueeze_to_4D(next_states)
        return ttnn.reshape(
            next_states,
            (1, 1, batch, HIDDEN_SIZE),
            (1, 1, max(TILE_SIZE, batch), HIDDEN_SIZE),
        )

    def _moe_prefill(self, hidden_states: ttnn.Tensor, routing_weights: ttnn.Tensor) -> ttnn.Tensor:
        seq_len = hidden_states.shape[-2]
        if seq_len <= PREFILL_MOE_CHUNK_SIZE:
            return self._moe_prefill_chunk(hidden_states, routing_weights)

        # The functional all-expert formulation repeats each token 128 times.
        # At 16K tokens a monolithic repeat requests an 11.81 GB allocation,
        # larger than one contiguous free region on P300. Keep the exact math
        # and framework-default matmuls, but bound the live working set.
        chunks = []
        for start in range(0, seq_len, PREFILL_MOE_CHUNK_SIZE):
            end = min(start + PREFILL_MOE_CHUNK_SIZE, seq_len)
            hidden_chunk = ttnn.slice(
                hidden_states,
                starts=[0, 0, start, 0],
                ends=[1, 1, end, HIDDEN_SIZE],
                steps=[1, 1, 1, 1],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            routing_chunk = ttnn.slice(
                routing_weights,
                starts=[0, 0, start, 0],
                ends=[1, 1, end, NUM_EXPERTS],
                steps=[1, 1, 1, 1],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            chunks.append(self._moe_prefill_chunk(hidden_chunk, routing_chunk))
        return ttnn.concat(chunks, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _moe_prefill_chunk(self, hidden_states: ttnn.Tensor, routing_weights: ttnn.Tensor) -> ttnn.Tensor:
        return sparse_expert_prefill(
            hidden_states=hidden_states,
            routing_weights=routing_weights,
            weights=self.expert_weights,
            config=self.expert_config,
            prefill_sparsity=self.expert_prefill_sparsity,
        )


def _text_config(hf_config: Any) -> Any:
    return getattr(hf_config, "text_config", hf_config)


def _validate_text_config(config: Any) -> None:
    expected = {
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": MLP_INTERMEDIATE_SIZE,
        "num_attention_heads": NUM_Q_HEADS,
        "num_key_value_heads": SLIDING_NUM_KV_HEADS,
        "num_global_key_value_heads": FULL_NUM_KV_HEADS,
        "head_dim": SLIDING_HEAD_DIM,
        "global_head_dim": FULL_HEAD_DIM,
        "num_experts": NUM_EXPERTS,
        "top_k_experts": TOP_K_EXPERTS,
        "moe_intermediate_size": MOE_INTERMEDIATE_SIZE,
    }
    for name, value in expected.items():
        if getattr(config, name) != value:
            raise ValueError(f"{MODEL_ID} expected config.{name}={value}, got {getattr(config, name)!r}")
    if not config.enable_moe_block:
        raise ValueError(f"{MODEL_ID} requires enable_moe_block=True")
    if config.hidden_size_per_layer_input != 0:
        raise ValueError("Gemma4 26B-A4B functional decoder expects hidden_size_per_layer_input=0")
    if config.hidden_activation != "gelu_pytorch_tanh":
        raise ValueError(f"expected gelu_pytorch_tanh activation, got {config.hidden_activation!r}")
    if not config.attention_k_eq_v:
        raise ValueError("Gemma4 26B-A4B full attention expects attention_k_eq_v=True")


def _layer_kind(layer_type: str) -> _LayerKind:
    if layer_type == "sliding_attention":
        return SLIDING_KIND
    if layer_type == "full_attention":
        return FULL_KIND
    raise ValueError(f"unsupported Gemma4 layer type {layer_type!r}")


def _detect_layer_prefix(state_dict: dict[str, Any], layer_idx: int) -> str:
    candidates = [
        f"model.language_model.layers.{layer_idx}",
        f"language_model.layers.{layer_idx}",
        f"layers.{layer_idx}",
    ]
    for prefix in candidates:
        if f"{prefix}.self_attn.q_proj.weight" in state_dict:
            return prefix
    raise KeyError(f"could not find layer {layer_idx} q_proj in state_dict")


def _replicate_mapper(device: Any) -> Any | None:
    if isinstance(device, ttnn.MeshDevice):
        return ttnn.ReplicateTensorToMesh(device)
    return None


def _make_correctness_compute_config(device: Any) -> ttnn.DeviceComputeKernelConfig:
    arch = device.arch() if hasattr(device, "arch") else ttnn.device.GetDefaultDevice().arch()
    return ttnn.init_device_compute_kernel_config(
        arch,
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


def _make_sdpa_program_config(device: Any) -> ttnn.SDPAProgramConfig:
    """Return Gemma-4 decode SDPA's minimal correctness-derived grid.

    The default Blackhole grid exceeds the decode kernel's 64-core-per-head
    reduction limit for this model's low KV-head counts and corrupts batch-32
    output. A 32-core cap passes both head-dim 256 and 512 layer contracts.
    """
    device_grid = device.compute_with_storage_grid_size()
    grid = ttnn.CoreCoord(min(8, device_grid.x), min(4, device_grid.y))
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid,
        q_chunk_size=32,
        k_chunk_size=64,
        exp_approx_mode=False,
    )


def _make_decode_rope_memory_config(device: Any, batch: int, head_dim: int) -> ttnn.MemoryConfig:
    return _make_decode_height_sharded_memory_config(device, batch, head_dim)


def _make_single_user_cache_update_memory_config(device: Any, head_dim: int) -> ttnn.MemoryConfig:
    del device
    one_core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    return ttnn.create_sharded_memory_config(
        shape=(TILE_SIZE, head_dim),
        core_grid=one_core,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _make_decode_height_sharded_memory_config(device: Any, batch: int, width: int) -> ttnn.MemoryConfig:
    try:
        grid = device.compute_with_storage_grid_size()
        max_grid_x, max_grid_y = grid.x, grid.y
    except Exception:
        max_grid_x, max_grid_y = 8, 8
    num_cores = max(1, batch)
    grid_x = min(num_cores, max_grid_x)
    while num_cores % grid_x != 0 or num_cores // grid_x > max_grid_y:
        grid_x -= 1
    grid_y = num_cores // grid_x
    batch_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))])
    return ttnn.create_sharded_memory_config(
        shape=(TILE_SIZE, width),
        core_grid=batch_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
