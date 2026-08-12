# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Functional TTNN decoder layer for ``meta-models/Muse-Glimmer-30B``.

This module implements one HuggingFace ``MuseGlimmerTextDecoderLayer`` on a
single ``1x1`` ``ttnn.MeshDevice``.  Setup-time helpers consume PyTorch tensors
from a HuggingFace state dict, but ``prefill_forward`` and ``decode_forward``
are TTNN-only hot paths: no ``torch``, no ``ttnn.from_torch``, no
``ttnn.to_torch`` and no host fallback.

Architecture summary (from ``transformers.models.muse_glimmer``)
---------------------------------------------------------------

Two decoder-layer *kinds* are selected by ``config.layer_types[layer_idx]`` and
``config.layer_rope_theta[layer_idx]``; every other computation is identical:

===================  ==================  ==========================
layer kind           ``layer_types``     ``layer_rope_theta`` (gate)
===================  ==================  ==========================
``sliding``          sliding_attention   500000.0  (RoPE applied)
``full``             full_attention      0         (NoPE, no rotary)
===================  ==================  ==========================

``layer_rope_theta[i]`` is only a *gate* in HF: the rotary base itself is the
model-level ``rope_parameters["rope_theta"]`` (see ``_rope_theta``).

Per layer::

    residual = x
    h = input_layernorm(x)                    # centered RMSNorm, eps=rms_norm_eps
    h = self_attn(h)
    h = post_attention_layernorm(h)           # centered RMSNorm, eps=post_norm_eps
    x = residual + h

    residual = x
    h = pre_feedforward_layernorm(x)          # centered RMSNorm, eps=rms_norm_eps
    h = down_proj(silu(gate_proj(h)) * up_proj(h))
    h = post_feedforward_layernorm(h)         # centered RMSNorm, eps=post_norm_eps
    x = residual + h

"Centered" RMSNorm is ``rms_norm(x) * (1 + w)``; the ``1 +`` is folded into the
device weight at setup time.

Attention (``MuseGlimmerTextAttention``)::

    q = q_proj(h) -> [b, 32, s, 128]
    k = k_proj(h) -> [b,  2, s, 128]
    v = v_proj(h) -> [b,  2, s, 128]
    q = rmsnorm_no_scale(q) * qk_scale_factor      # eps = rms_norm_eps
    k = rmsnorm_no_scale(k)                        # v is NOT normed
    q, k = rope(q, k)                              # sliding layers only
    o = sdpa(q, k, v, scale=1/sqrt(head_dim), window=2048 on sliding layers)
    o = concat_heads(o) * sigmoid(attn_gate_proj(h))
    o = o_proj(o)

``qk_scale_factor`` multiplies Q before RoPE.  RoPE is a rotation and the only
consumer of Q is the ``q @ k^T`` product, so the constant is folded into the
SDPA ``scale`` (``qk_scale_factor / sqrt(head_dim)``) — algebraically identical
and one device op cheaper.

Forward contract
----------------

``prefill_forward(hidden_states, *, page_table, user_id=0, start_pos=0,
sliding_kv_tail=None, return_sliding_kv_tail=False)``
    ``hidden_states``: TTNN tile tensor ``[1, 1, seq_len, hidden_size]``.
    ``seq_len`` is the *logical* prompt length and may be any value in
    ``[1, max_seq_len - start_pos]``; it does not have to be a multiple of the
    tile height, the page block size or the internal prefill chunk size.  The
    layer pads, masks and slices internally.
    ``page_table``: ``int32`` ``[max_batch_size, blocks_per_seq]`` row-major
    tensor mapping virtual KV blocks to physical blocks.
    ``user_id``: page-table row / KV-cache batch slot for this prompt.
    ``start_pos``: absolute position of the first token (``0`` for a fresh
    prompt; ``>0`` for generator-level chunked prefill, in which case the paged
    cache must already contain positions ``[0, start_pos)`` for this user and
    ``start_pos`` must be a multiple of the page block size).  The caller is
    responsible for ``start_pos`` matching the *logical* length already
    prefilled: a previous call of logical length ``L`` must be continued with
    ``start_pos == L``, otherwise the continuation attends this layer's
    tile-padding K/V.
    ``sliding_kv_tail``: on ``sliding`` layers only, and *required* whenever
    ``start_pos > 0``, the previous call's last ``sliding_kv_tail_len(start_pos)``
    K/V rows as ``(k, v)``, each ``[1, num_key_value_heads, tail_len, head_dim]``.
    The paged chunked-SDPA op has no sliding-window mask, so the window cannot
    be read back out of the paged cache; the tail is handed over explicitly (the
    same contract ``models/demos/gemma4`` uses at generator level).  The tensors
    are **consumed** (deallocated) by this call.
    ``return_sliding_kv_tail``: return ``(output, tail)`` instead of ``output``,
    where ``tail`` is the ``(k, v)`` pair to feed the next continuation call
    (``None`` on ``full`` layers).  The caller then owns those tensors.
    Returns a TTNN tile tensor ``[1, 1, seq_len, hidden_size]``.

``decode_forward(hidden_states, *, current_pos, page_table, rope_pos_ids=None)``
    ``hidden_states``: TTNN tile tensor ``[1, 1, batch, hidden_size]``.
    ``current_pos``: ``int32`` device tensor ``[batch]`` with the absolute
    position each user is decoding (also the KV-cache write index).
    ``rope_pos_ids``: ``uint32`` device tensor ``[1, batch]`` holding the same
    positions, used for the on-device cos/sin gather.  Required on ``sliding``
    (RoPE) layers, ignored on ``full`` (NoPE) layers.
    ``page_table``: same mapping used by prefill.
    Returns a TTNN tile tensor ``[1, 1, batch, hidden_size]``.

Both entry points are trace-safe: every runtime input is a device tensor whose
*contents* the caller refreshes outside the captured region.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Any

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.model_config import num_to_corerange

MODEL_ID = "meta-models/Muse-Glimmer-30B"
MODEL_DIR_NAME = "meta_models_muse_glimmer_30b"

TILE_SIZE = 32

#: Sequence chunk used by ``prefill_forward`` to bound peak DRAM.  The MLP
#: intermediate is ``19968`` wide, so a full 128k prompt materialised in one
#: shot would need >10 GB of activations; chunking the *whole* layer keeps the
#: transient working set proportional to the chunk instead of the prompt.
DEFAULT_PREFILL_CHUNK_SIZE = 8192

#: ``ttnn.transformer.scaled_dot_product_attention`` is only used on slices no
#: longer than this.  The non-chunked prefill SDPA has a documented correctness
#: cliff at ``seq_len >= 32768`` (see ``models/demos/gemma4/tt/attention``);
#: every path here stays well below it because of the prefill chunking above.
PREFILL_SDPA_MAX_SEQ = 32768

#: Q/K flash-attention chunk used by the prefill SDPA calls.  Must satisfy
#: ``q_chunk_size == k_chunk_size`` — see ``_prefill_program_config``.
PREFILL_SDPA_CHUNK = 128

LAYER_KIND_SLIDING = "sliding"
LAYER_KIND_FULL = "full"


def _as_float32(value: float) -> float:
    """Round a Python double to the nearest ``float32``-representable double."""
    return struct.unpack("f", struct.pack("f", float(value)))[0]


@dataclass(frozen=True)
class PagedAttentionConfig:
    """Paged KV-cache geometry consumed by the TTNN SDPA / cache kernels."""

    block_size: int = 64
    max_num_blocks: int = 2048


@dataclass(frozen=True)
class MuseGlimmerLayerConfig:
    """Static per-layer configuration resolved from the HF text config."""

    layer_idx: int
    layer_kind: str
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    post_norm_eps: float
    qk_scale_factor: float
    sliding_window: int | None
    rope_theta: float | None
    max_seq_len: int
    max_batch_size: int
    paged_attention_config: PagedAttentionConfig
    prefill_chunk_size: int

    @property
    def is_sliding(self) -> bool:
        return self.layer_kind == LAYER_KIND_SLIDING

    @property
    def uses_rope(self) -> bool:
        return self.rope_theta is not None

    @property
    def sdpa_scale(self) -> float:
        """``qk_scale_factor`` folded into the standard ``1/sqrt(head_dim)``.

        Rounded to the nearest ``float32``: the TTNN SDPA bindings declare
        ``scale`` as ``std::optional<float>`` with ``nb::arg(...).noconvert()``,
        so a Python double that is not exactly representable as a ``float`` is
        rejected with a signature-mismatch ``TypeError``.  Rounding here keeps
        the call site readable and costs ~1e-8 relative error.
        """
        return _as_float32(self.qk_scale_factor / math.sqrt(self.head_dim))


def _text_config(hf_config: Any) -> Any:
    """Accept either ``MuseGlimmerConfig`` or its ``text_config`` sub-config."""
    text_config = getattr(hf_config, "text_config", None)
    return hf_config if text_config is None else text_config


def _require_muse_glimmer_text_config(text_config: Any) -> None:
    expected = {
        "model_type": "muse_glimmer_text",
        # The RoPE base lives here, once for the whole model.  ``layer_rope_theta``
        # is only HF's NoPE gate (see ``_rope_theta``), so both are pinned.
        "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
        "hidden_size": 6656,
        "intermediate_size": 19968,
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "hidden_activation": "silu",
        "attention_bias": False,
        "sliding_window": 2048,
        "rms_norm_eps": 1e-5,
        "post_norm_eps": 1e-8,
        "qk_scale_factor": 3.87,
        "max_position_embeddings": 131072,
        "num_hidden_layers": 52,
    }
    for name, expected_value in expected.items():
        actual = getattr(text_config, name, None)
        if actual != expected_value:
            raise ValueError(f"{MODEL_ID} functional decoder expects {name}={expected_value!r}, got {actual!r}")


def _rope_theta(text_config: Any) -> float:
    """The model-level RoPE base, the way HF's rotary module reads it.

    ``MuseGlimmerTextRotaryEmbedding`` takes its base from
    ``config.rope_parameters["rope_theta"]`` — a single value for the whole
    model.  The per-layer ``layer_rope_theta`` list is used by
    ``MuseGlimmerTextModel.forward`` only as a *boolean* NoPE gate
    (``position_embeddings if config.layer_rope_theta[i] else None``), so it must
    not be read as the base even though this checkpoint happens to store the same
    number in both places.  ``_require_muse_glimmer_text_config`` pins both.
    """
    return float(text_config.rope_parameters["rope_theta"])


def resolve_layer_kind(hf_config: Any, layer_idx: int) -> str:
    """Map an HF layer index to this module's layer kind.

    ``layer_rope_theta[layer_idx]`` is HF's NoPE gate, so only its truthiness is
    used here; the RoPE base comes from ``_rope_theta``.
    """
    text_config = _text_config(hf_config)
    layer_type = text_config.layer_types[layer_idx]
    rope_theta = text_config.layer_rope_theta[layer_idx]
    if layer_type == "sliding_attention" and rope_theta:
        return LAYER_KIND_SLIDING
    if layer_type == "full_attention" and not rope_theta:
        return LAYER_KIND_FULL
    raise ValueError(
        f"{MODEL_ID} layer {layer_idx} has an unsupported (layer_type, rope_theta) "
        f"combination: ({layer_type!r}, {rope_theta!r}). The released checkpoint only "
        "pairs sliding_attention with RoPE and full_attention with NoPE."
    )


def reference_layer_indices(hf_config: Any) -> dict[str, int]:
    """First HF layer index of each layer kind — the canonical test targets."""
    text_config = _text_config(hf_config)
    found: dict[str, int] = {}
    for layer_idx in range(text_config.num_hidden_layers):
        kind = resolve_layer_kind(hf_config, layer_idx)
        found.setdefault(kind, layer_idx)
    return found


def _layer_prefix(layer_idx: int) -> str:
    return f"model.language_model.layers.{layer_idx}"


def _get_layer_tensor(state_dict: dict[str, torch.Tensor], layer_idx: int, suffix: str) -> torch.Tensor:
    canonical = f"{_layer_prefix(layer_idx)}.{suffix}"
    if canonical in state_dict:
        return state_dict[canonical]
    if suffix in state_dict:
        return state_dict[suffix]
    raise KeyError(f"Missing HF decoder tensor {canonical!r} or layer-local key {suffix!r}")


def _to_device(
    tensor: torch.Tensor,
    *,
    mesh_device: ttnn.MeshDevice,
    dtype: ttnn.DataType,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor,
        device=mesh_device,
        layout=layout,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _rope_cos_sin(max_seq_len: int, head_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """HF ``default`` RoPE cos/sin tables, ``[max_seq_len, head_dim]``.

    Mirrors ``MuseGlimmerTextRotaryEmbedding``: ``inv_freq`` over half the head
    dim, ``emb = cat(freqs, freqs)``, ``attention_scaling == 1.0``.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


class _MuseGlimmerNorm(LightweightModule):
    """Centered RMSNorm: ``rms_norm(x) * (1 + w)`` with ``1 + w`` pre-folded."""

    def __init__(self, weight: ttnn.Tensor, eps: float) -> None:
        super().__init__()
        self.weight = weight
        self.eps = eps

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.rms_norm(x, weight=self.weight, epsilon=self.eps, memory_config=ttnn.DRAM_MEMORY_CONFIG)


class _MuseGlimmerMLP(LightweightModule):
    """SwiGLU MLP: ``down(silu(gate(x)) * up(x))``."""

    def __init__(self, gate: ttnn.Tensor, up: ttnn.Tensor, down: ttnn.Tensor, activation_dtype: ttnn.DataType) -> None:
        super().__init__()
        self.gate = gate
        self.up = up
        self.down = down
        self.activation_dtype = activation_dtype

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        gate = ttnn.linear(x, self.gate, dtype=self.activation_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        up = ttnn.linear(x, self.up, dtype=self.activation_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        activated = ttnn.silu(gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gate)
        hidden = ttnn.mul(activated, up, dtype=self.activation_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(activated)
        ttnn.deallocate(up)
        out = ttnn.linear(hidden, self.down, dtype=self.activation_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(hidden)
        return out


class FunctionalDecoder(LightweightModule):
    """Single-layer TTNN implementation of ``MuseGlimmerTextDecoderLayer``."""

    def __init__(
        self,
        *,
        config: MuseGlimmerLayerConfig,
        mesh_device: ttnn.MeshDevice,
        input_layernorm: _MuseGlimmerNorm,
        post_attention_layernorm: _MuseGlimmerNorm,
        pre_feedforward_layernorm: _MuseGlimmerNorm,
        post_feedforward_layernorm: _MuseGlimmerNorm,
        mlp: _MuseGlimmerMLP,
        wqkv: ttnn.Tensor,
        w_attn_gate: ttnn.Tensor,
        wo: ttnn.Tensor,
        k_cache: ttnn.Tensor,
        v_cache: ttnn.Tensor,
        cos_cache: ttnn.Tensor | None,
        sin_cache: ttnn.Tensor | None,
        activation_dtype: ttnn.DataType,
        kv_cache_dtype: ttnn.DataType,
    ) -> None:
        super().__init__()
        self.config = config
        self.mesh_device = mesh_device
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm
        self.pre_feedforward_layernorm = pre_feedforward_layernorm
        self.post_feedforward_layernorm = post_feedforward_layernorm
        self.mlp = mlp
        self.wqkv = wqkv
        self.w_attn_gate = w_attn_gate
        self.wo = wo
        self.k_cache = k_cache
        self.v_cache = v_cache
        self.cos_cache = cos_cache
        self.sin_cache = sin_cache
        self.activation_dtype = activation_dtype
        self.kv_cache_dtype = kv_cache_dtype

        self.sdpa_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        grid = mesh_device.compute_with_storage_grid_size()
        self.decode_sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
            q_chunk_size=32,
            k_chunk_size=64,
            exp_approx_mode=False,
        )
        self.prefill_sdpa_grid = ttnn.CoreCoord(grid.x, grid.y)

    def _prefill_program_config(self, seq_len: int) -> ttnn.SDPAProgramConfig:
        """Prefill SDPA chunking, clamped to the (tile-padded) slice length.

        ``q_chunk_size`` and ``k_chunk_size`` are deliberately kept equal.  With
        ``q_chunk_size == 2 * k_chunk_size`` the sliding-window prefill SDPA
        returns wrong results for sequence lengths a little past the window
        (measured PCC ~0.977 at ``seq=2080/4128/8224``, window 2048, both for
        256/128 and 128/64), while every ``q == k`` pairing stays at ~0.9998.
        See ``doc/functional_decoder/sdpa_sliding_window_chunk_repro.py``.
        """
        padded = ((seq_len + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
        chunk = max(TILE_SIZE, min(PREFILL_SDPA_CHUNK, padded))
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.prefill_sdpa_grid,
            q_chunk_size=chunk,
            k_chunk_size=chunk,
            exp_approx_mode=False,
        )

    # ------------------------------------------------------------------ setup

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        *,
        hf_config: Any,
        layer_idx: int,
        mesh_device: ttnn.MeshDevice,
        max_batch_size: int = 1,
        max_seq_len: int | None = None,
        page_block_size: int = 64,
        max_num_blocks: int | None = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat16,
        activation_dtype: ttnn.DataType = ttnn.bfloat16,
        kv_cache_dtype: ttnn.DataType = ttnn.bfloat16,
        prefill_chunk_size: int = DEFAULT_PREFILL_CHUNK_SIZE,
        **kwargs,
    ) -> "FunctionalDecoder":
        """Build a decoder layer from canonical HF decoder-layer weights.

        ``state_dict`` may be a full HuggingFace state dict or a layer-local
        dict whose keys omit ``model.language_model.layers.<layer_idx>.``.  All
        ``torch`` and ``ttnn.from_torch`` work happens here, never at runtime.
        """
        if kwargs:
            raise TypeError(f"Unexpected FunctionalDecoder.from_state_dict kwargs: {sorted(kwargs)}")
        if mesh_device.get_num_devices() != 1:
            raise ValueError("FunctionalDecoder is the single-chip stage; use a 1x1 MeshDevice.")

        text_config = _text_config(hf_config)
        _require_muse_glimmer_text_config(text_config)
        layer_kind = resolve_layer_kind(hf_config, layer_idx)

        max_seq_len = int(max_seq_len or text_config.max_position_embeddings)
        if max_seq_len > text_config.max_position_embeddings:
            raise ValueError(
                f"max_seq_len={max_seq_len} exceeds the HF-advertised context " f"{text_config.max_position_embeddings}"
            )
        if page_block_size % TILE_SIZE != 0:
            raise ValueError(f"page_block_size must be a multiple of {TILE_SIZE}, got {page_block_size}")
        if max_seq_len % TILE_SIZE != 0:
            # prefill rounds the logical length up to a tile multiple, so a
            # non-tile-aligned capacity would let a legal prompt overrun the
            # RoPE tables and the page table.
            raise ValueError(f"max_seq_len must be a multiple of {TILE_SIZE}, got {max_seq_len}")
        blocks_per_seq = (max_seq_len + page_block_size - 1) // page_block_size
        if max_num_blocks is None:
            max_num_blocks = max_batch_size * blocks_per_seq
        if max_num_blocks < max_batch_size * blocks_per_seq:
            raise ValueError(
                f"max_num_blocks={max_num_blocks} cannot hold max_batch_size={max_batch_size} x "
                f"{blocks_per_seq} blocks of {page_block_size} tokens"
            )
        if prefill_chunk_size % page_block_size or prefill_chunk_size % TILE_SIZE:
            raise ValueError(
                f"prefill_chunk_size={prefill_chunk_size} must be a multiple of the page block size "
                f"({page_block_size}) and the tile height ({TILE_SIZE})"
            )
        if prefill_chunk_size > PREFILL_SDPA_MAX_SEQ // 2:
            raise ValueError(
                f"prefill_chunk_size={prefill_chunk_size} is too large: the sliding-window prefill "
                f"slice (chunk + window) must stay below the {PREFILL_SDPA_MAX_SEQ}-token SDPA limit"
            )

        config = MuseGlimmerLayerConfig(
            layer_idx=layer_idx,
            layer_kind=layer_kind,
            hidden_size=text_config.hidden_size,
            intermediate_size=text_config.intermediate_size,
            num_attention_heads=text_config.num_attention_heads,
            num_key_value_heads=text_config.num_key_value_heads,
            head_dim=text_config.head_dim,
            rms_norm_eps=text_config.rms_norm_eps,
            post_norm_eps=text_config.post_norm_eps,
            qk_scale_factor=text_config.qk_scale_factor,
            sliding_window=text_config.sliding_window if layer_kind == LAYER_KIND_SLIDING else None,
            rope_theta=(_rope_theta(text_config) if layer_kind == LAYER_KIND_SLIDING else None),
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            paged_attention_config=PagedAttentionConfig(
                block_size=page_block_size,
                max_num_blocks=max_num_blocks,
            ),
            prefill_chunk_size=prefill_chunk_size,
        )

        def norm(name: str, eps: float) -> _MuseGlimmerNorm:
            weight = _get_layer_tensor(state_dict, layer_idx, f"{name}.weight").to(torch.float32)
            # Centered RMSNorm multiplies by (1 + w); fold the +1 in at setup.
            folded = (1.0 + weight).reshape(1, 1, 1, config.hidden_size)
            return _MuseGlimmerNorm(
                _to_device(folded.to(torch.bfloat16), mesh_device=mesh_device, dtype=ttnn.bfloat16),
                eps,
            )

        def linear_weight(suffix: str) -> torch.Tensor:
            # HF stores nn.Linear weights as [out, in]; ttnn.linear wants [in, out].
            return _get_layer_tensor(state_dict, layer_idx, suffix).to(torch.float32).transpose(-2, -1).contiguous()

        wq = linear_weight("self_attn.q_proj.weight")
        wk = linear_weight("self_attn.k_proj.weight")
        wv = linear_weight("self_attn.v_proj.weight")
        wqkv = torch.cat([wq, wk, wv], dim=-1).unsqueeze(0).unsqueeze(0)
        w_attn_gate = linear_weight("self_attn.gate_proj.weight").unsqueeze(0).unsqueeze(0)
        wo = linear_weight("self_attn.o_proj.weight").unsqueeze(0).unsqueeze(0)

        mlp = _MuseGlimmerMLP(
            gate=_to_device(
                linear_weight("mlp.gate_proj.weight").unsqueeze(0).unsqueeze(0),
                mesh_device=mesh_device,
                dtype=weight_dtype,
            ),
            up=_to_device(
                linear_weight("mlp.up_proj.weight").unsqueeze(0).unsqueeze(0),
                mesh_device=mesh_device,
                dtype=weight_dtype,
            ),
            down=_to_device(
                linear_weight("mlp.down_proj.weight").unsqueeze(0).unsqueeze(0),
                mesh_device=mesh_device,
                dtype=weight_dtype,
            ),
            activation_dtype=activation_dtype,
        )

        cache_shape = (max_num_blocks, config.num_key_value_heads, page_block_size, config.head_dim)
        k_cache = _to_device(torch.zeros(cache_shape), mesh_device=mesh_device, dtype=kv_cache_dtype)
        v_cache = _to_device(torch.zeros(cache_shape), mesh_device=mesh_device, dtype=kv_cache_dtype)

        cos_cache = sin_cache = None
        if config.uses_rope:
            cos, sin = _rope_cos_sin(max_seq_len, config.head_dim, config.rope_theta)
            # 2D [max_seq_len, head_dim] tables so decode can gather per-user
            # rows on device with ttnn.embedding (trace-safe).
            cos_cache = _to_device(
                cos.to(torch.bfloat16), mesh_device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            sin_cache = _to_device(
                sin.to(torch.bfloat16), mesh_device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
            )

        return cls(
            config=config,
            mesh_device=mesh_device,
            input_layernorm=norm("input_layernorm", config.rms_norm_eps),
            post_attention_layernorm=norm("post_attention_layernorm", config.post_norm_eps),
            pre_feedforward_layernorm=norm("pre_feedforward_layernorm", config.rms_norm_eps),
            post_feedforward_layernorm=norm("post_feedforward_layernorm", config.post_norm_eps),
            mlp=mlp,
            wqkv=_to_device(wqkv, mesh_device=mesh_device, dtype=weight_dtype),
            w_attn_gate=_to_device(w_attn_gate, mesh_device=mesh_device, dtype=weight_dtype),
            wo=_to_device(wo, mesh_device=mesh_device, dtype=weight_dtype),
            k_cache=k_cache,
            v_cache=v_cache,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            activation_dtype=activation_dtype,
            kv_cache_dtype=kv_cache_dtype,
        )

    # -------------------------------------------------------------- utilities

    @property
    def kv_cache(self) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        return self.k_cache, self.v_cache

    def _per_head_rmsnorm(self, tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Scale-less RMSNorm over ``head_dim`` (``MuseGlimmerRMSNorm``)."""
        shape = tensor.shape
        flat = ttnn.reshape(tensor, (1, 1, shape[0] * shape[1] * shape[2], shape[3]))
        normed = ttnn.rms_norm(flat, epsilon=self.config.rms_norm_eps, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(flat)
        return ttnn.reshape(normed, shape)

    @staticmethod
    def _rotate_half(x: ttnn.Tensor) -> ttnn.Tensor:
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        negated = ttnn.neg(x2)
        ttnn.deallocate(x2)
        out = ttnn.concat([negated, x1], dim=-1)
        ttnn.deallocate(negated)
        ttnn.deallocate(x1)
        return out

    def _apply_rope(self, tensor: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
        """``x * cos + rotate_half(x) * sin`` (HF NeoX convention)."""
        rotated = self._rotate_half(tensor)
        out = ttnn.add(
            ttnn.mul(tensor, cos, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            ttnn.mul(rotated, sin, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(rotated)
        return out

    def _blocks_per_seq(self) -> int:
        block_size = self.config.paged_attention_config.block_size
        return (self.config.max_seq_len + block_size - 1) // block_size

    # --------------------------------------------------------------- prefill

    def _prefill_rope_tables(self, start_pos: int, length: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """cos/sin for the contiguous positions ``[start_pos, start_pos+length)``.

        ``ttnn.slice`` on the setup-time table is a device op; ``start_pos`` and
        ``length`` are Python ints only because prefill shapes are static per
        call, exactly like the SDPA program config.

        A full-range ``ttnn.slice`` returns the *input tensor itself*, so the
        intermediate is only deallocated when this call actually owns it —
        otherwise a ``max_seq_len``-long single-chunk prefill would free the
        layer's persistent RoPE tables.
        """
        head_dim = self.config.head_dim
        full_range = start_pos == 0 and length == self.cos_cache.shape[0]
        cos = (
            self.cos_cache if full_range else ttnn.slice(self.cos_cache, [start_pos, 0], [start_pos + length, head_dim])
        )
        sin = (
            self.sin_cache if full_range else ttnn.slice(self.sin_cache, [start_pos, 0], [start_pos + length, head_dim])
        )
        cos_t = ttnn.reshape(ttnn.to_layout(cos, ttnn.TILE_LAYOUT), (1, 1, length, head_dim))
        sin_t = ttnn.reshape(ttnn.to_layout(sin, ttnn.TILE_LAYOUT), (1, 1, length, head_dim))
        if not full_range:
            ttnn.deallocate(cos)
            ttnn.deallocate(sin)
        return cos_t, sin_t

    def _prefill_attention(
        self,
        normed: ttnn.Tensor,
        *,
        page_table: ttnn.Tensor,
        user_id: int,
        start_pos: int,
        sliding_tail: tuple[ttnn.Tensor, ttnn.Tensor] | None,
        need_tail: bool,
    ) -> tuple[ttnn.Tensor, tuple[ttnn.Tensor, ttnn.Tensor] | None]:
        cfg = self.config
        seq_len = normed.shape[-2]
        head_dim = cfg.head_dim
        n_heads = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads

        xqkv = ttnn.linear(normed, self.wqkv, dtype=self.activation_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=n_heads,
            num_kv_heads=n_kv,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(xqkv)

        q_normed = self._per_head_rmsnorm(q)
        ttnn.deallocate(q)
        k_normed = self._per_head_rmsnorm(k)
        ttnn.deallocate(k)
        q, k = q_normed, k_normed

        if cfg.uses_rope:
            cos, sin = self._prefill_rope_tables(start_pos, seq_len)
            q_rot = self._apply_rope(q, cos, sin)
            ttnn.deallocate(q)
            k_rot = self._apply_rope(k, cos, sin)
            ttnn.deallocate(k)
            ttnn.deallocate(cos)
            ttnn.deallocate(sin)
            q, k = q_rot, k_rot

        # Paged KV fill.  ``paged_fill_cache`` does no dtype conversion, so cast
        # to the cache dtype first (decode's update op owns its own repack).
        block_size = cfg.paged_attention_config.block_size
        k_fill = k if k.dtype == self.kv_cache_dtype else ttnn.typecast(k, self.kv_cache_dtype)
        v_fill = v if v.dtype == self.kv_cache_dtype else ttnn.typecast(v, self.kv_cache_dtype)
        chunk_page_table, owns_chunk_pt = self._chunk_page_table(page_table, user_id, start_pos, seq_len)
        ttnn.experimental.paged_fill_cache(self.k_cache, k_fill, chunk_page_table, batch_idx=0, block_size=block_size)
        ttnn.experimental.paged_fill_cache(self.v_cache, v_fill, chunk_page_table, batch_idx=0, block_size=block_size)
        if owns_chunk_pt:
            ttnn.deallocate(chunk_page_table)
        if k_fill is not k:
            ttnn.deallocate(k_fill)
        if v_fill is not v:
            ttnn.deallocate(v_fill)

        next_tail: tuple[ttnn.Tensor, ttnn.Tensor] | None = None
        if cfg.is_sliding:
            attn, next_tail = self._prefill_sdpa_sliding(q, k, v, sliding_tail, need_tail)
        else:
            attn = self._prefill_sdpa_full(q, k, v, page_table, user_id, start_pos)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        out = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn)
        gate = ttnn.linear(normed, self.w_attn_gate, dtype=self.activation_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gate = ttnn.sigmoid(gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gated = ttnn.mul(out, gate, dtype=self.activation_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out)
        ttnn.deallocate(gate)
        projected = ttnn.linear(gated, self.wo, dtype=self.activation_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gated)
        return projected, next_tail

    def _chunk_page_table(
        self, page_table: ttnn.Tensor, user_id: int, start_pos: int, seq_len: int
    ) -> tuple[ttnn.Tensor, bool]:
        """Single-row page table covering the blocks this chunk writes.

        ``paged_fill_cache`` always writes from virtual block 0 of the table it
        is given, so a chunk starting at ``start_pos`` needs the user's row
        shifted by ``start_pos / block_size`` blocks.
        """
        block_size = self.config.paged_attention_config.block_size
        if start_pos % block_size:
            raise ValueError(
                f"chunked prefill start_pos={start_pos} must be a multiple of the page block size {block_size}"
            )
        first_block = start_pos // block_size
        num_blocks = (seq_len + block_size - 1) // block_size
        last_block = first_block + num_blocks
        if last_block > page_table.shape[-1]:
            raise ValueError(
                f"page table has {page_table.shape[-1]} blocks per sequence but this prefill chunk needs "
                f"{last_block} (start_pos={start_pos}, seq_len={seq_len})"
            )
        return self._page_table_row(page_table, user_id, first_block, last_block)

    @staticmethod
    def _page_table_row(
        page_table: ttnn.Tensor, user_id: int, first_block: int, last_block: int
    ) -> tuple[ttnn.Tensor, bool]:
        """``(row_view, owned)`` for ``page_table[user_id, first:last]``.

        ``ttnn.slice`` returns the *input tensor itself* when the requested
        range covers the whole tensor, so deallocating the result blind would
        free the caller's page table.  ``owned`` says whether the caller may
        deallocate.
        """
        rows, cols = page_table.shape[0], page_table.shape[-1]
        if user_id == 0 and rows == 1 and first_block == 0 and last_block == cols:
            return page_table, False
        return ttnn.slice(page_table, [user_id, first_block], [user_id + 1, last_block]), True

    def _prefill_sdpa_full(
        self,
        q: ttnn.Tensor,
        k: ttnn.Tensor,
        v: ttnn.Tensor,
        page_table: ttnn.Tensor,
        user_id: int,
        start_pos: int,
    ) -> ttnn.Tensor:
        cfg = self.config
        if start_pos == 0:
            return ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                scale=cfg.sdpa_scale,
                program_config=self._prefill_program_config(q.shape[-2]),
                compute_kernel_config=self.sdpa_compute_kernel_config,
            )
        # Later chunks attend the whole prefix, which now lives in the paged
        # cache (this chunk's K/V was just filled).
        seq_len = q.shape[-2]
        n_heads = cfg.num_attention_heads
        head_dim = cfg.head_dim
        # ``chunked_scaled_dot_product_attention`` requires ``chunk_start_idx``
        # to be a multiple of both q_chunk_size and k_chunk_size, so shrink the
        # chunk until it divides this call's absolute offset.  ``start_pos`` is
        # a multiple of the page block size (itself a tile multiple), so this
        # terminates at TILE_SIZE at worst.
        chunked_q = 128
        while start_pos % chunked_q and chunked_q > TILE_SIZE:
            chunked_q //= 2
        if start_pos % chunked_q:
            raise ValueError(
                f"continuation prefill start_pos={start_pos} is not a multiple of the minimum SDPA "
                f"chunk size {TILE_SIZE}"
            )
        pad = (-seq_len) % chunked_q
        q_in = q
        if pad:
            q_in = ttnn.pad(q, [(0, 0), (0, 0), (0, pad), (0, 0)], value=0.0)
        user_pt, owns_user_pt = self._page_table_row(page_table, user_id, 0, page_table.shape[-1])
        out = ttnn.transformer.chunked_scaled_dot_product_attention(
            q_in,
            self.k_cache,
            self.v_cache,
            user_pt,
            start_pos,  # chunk_start_idx is positional-only in the ttnn binding
            scale=cfg.sdpa_scale,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=self.prefill_sdpa_grid,
                q_chunk_size=chunked_q,
                k_chunk_size=chunked_q,
                exp_approx_mode=False,
            ),
            compute_kernel_config=self.sdpa_compute_kernel_config,
        )
        if owns_user_pt:
            ttnn.deallocate(user_pt)
        if pad:
            ttnn.deallocate(q_in)
            trimmed = ttnn.slice(out, [0, 0, 0, 0], [1, n_heads, seq_len, head_dim])
            ttnn.deallocate(out)
            out = trimmed
        return out

    def _prefill_sdpa_sliding(
        self,
        q: ttnn.Tensor,
        k: ttnn.Tensor,
        v: ttnn.Tensor,
        sliding_tail: tuple[ttnn.Tensor, ttnn.Tensor] | None,
        need_tail: bool,
    ) -> tuple[ttnn.Tensor, tuple[ttnn.Tensor, ttnn.Tensor] | None]:
        """Windowed prefill SDPA plus, if ``need_tail``, the K/V tail the next
        chunk (or the next continuation call) needs.

        ``chunked_scaled_dot_product_attention`` (the paged reader) has no
        sliding-window mask, so sliding layers attend a *square*
        ``[previous-window tail | this chunk]`` slice instead: the previous
        chunk's last ``window`` K/V rows are prepended and Q is zero-padded in
        front by the same amount (those rows are causal-only filler and their
        outputs are dropped).
        """
        cfg = self.config
        window = cfg.sliding_window
        n_heads = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads
        head_dim = cfg.head_dim
        seq_len = q.shape[-2]

        if sliding_tail is None:
            attn = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                scale=cfg.sdpa_scale,
                sliding_window_size=window,
                program_config=self._prefill_program_config(seq_len),
                compute_kernel_config=self.sdpa_compute_kernel_config,
            )
            tail_source = (k, v)
        else:
            k_tail, v_tail = sliding_tail
            tail_len = k_tail.shape[-2]
            # ttnn.pad cannot front-pad a tiled tensor, so build the filler Q
            # rows explicitly. They are zeros: causal softmax over >= 1 key is
            # well defined and their outputs are sliced away below.
            q_filler = ttnn.zeros(
                [1, n_heads, tail_len, head_dim],
                dtype=q.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            q_padded = ttnn.concat([q_filler, q], dim=2)
            ttnn.deallocate(q_filler)
            k_cat = ttnn.concat([k_tail, k], dim=2)
            v_cat = ttnn.concat([v_tail, v], dim=2)
            ttnn.deallocate(k_tail)
            ttnn.deallocate(v_tail)
            full = ttnn.transformer.scaled_dot_product_attention(
                q_padded,
                k_cat,
                v_cat,
                is_causal=True,
                scale=cfg.sdpa_scale,
                sliding_window_size=window,
                program_config=self._prefill_program_config(tail_len + seq_len),
                compute_kernel_config=self.sdpa_compute_kernel_config,
            )
            ttnn.deallocate(q_padded)
            attn = ttnn.slice(full, [0, 0, tail_len, 0], [1, n_heads, tail_len + seq_len, head_dim])
            ttnn.deallocate(full)
            tail_source = (k_cat, v_cat)

        # Carry the last ``window`` K/V rows into the next chunk. They are taken
        # from ``[previous tail | this chunk]`` so a chunk shorter than the
        # window still hands over the full history the next chunk needs.
        source_k, source_v = tail_source
        next_tail = None
        if need_tail:
            source_len = source_k.shape[-2]
            tail_start = max(0, source_len - window)
            next_tail = (
                ttnn.clone(
                    ttnn.slice(source_k, [0, 0, tail_start, 0], [1, n_kv, source_len, head_dim]),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                ),
                ttnn.clone(
                    ttnn.slice(source_v, [0, 0, tail_start, 0], [1, n_kv, source_len, head_dim]),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                ),
            )
        if source_k is not k:
            ttnn.deallocate(source_k)
            ttnn.deallocate(source_v)
        return attn, next_tail

    def _prefill_chunk(
        self,
        hidden_states: ttnn.Tensor,
        *,
        page_table: ttnn.Tensor,
        user_id: int,
        start_pos: int,
        sliding_tail: tuple[ttnn.Tensor, ttnn.Tensor] | None,
        need_tail: bool,
    ) -> tuple[ttnn.Tensor, tuple[ttnn.Tensor, ttnn.Tensor] | None]:
        residual = hidden_states
        normed = self.input_layernorm(residual)
        attn, next_tail = self._prefill_attention(
            normed,
            page_table=page_table,
            user_id=user_id,
            start_pos=start_pos,
            sliding_tail=sliding_tail,
            need_tail=need_tail,
        )
        ttnn.deallocate(normed)
        attn = self.post_attention_layernorm(attn)
        hidden = ttnn.add(residual, attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn)

        mlp_in = self.pre_feedforward_layernorm(hidden)
        mlp_out = self.mlp(mlp_in)
        ttnn.deallocate(mlp_in)
        mlp_out = self.post_feedforward_layernorm(mlp_out)
        out = ttnn.add(hidden, mlp_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(hidden)
        ttnn.deallocate(mlp_out)
        return out, next_tail

    def sliding_kv_tail_len(self, start_pos: int) -> int:
        """Number of K/V rows a ``start_pos`` continuation prefill must be handed."""
        if not self.config.is_sliding:
            return 0
        return min(self.config.sliding_window, start_pos)

    def prefill_forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        page_table: ttnn.Tensor,
        user_id: int = 0,
        start_pos: int = 0,
        sliding_kv_tail: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
        return_sliding_kv_tail: bool = False,
    ) -> ttnn.Tensor | tuple[ttnn.Tensor, tuple[ttnn.Tensor, ttnn.Tensor] | None]:
        """Paged prefill for one user; see the module docstring for the contract."""
        cfg = self.config
        seq_len = int(hidden_states.shape[-2])
        if hidden_states.shape[-1] != cfg.hidden_size:
            raise ValueError(f"prefill expects hidden size {cfg.hidden_size}, got {hidden_states.shape[-1]}")
        if seq_len < 1:
            raise ValueError("prefill needs at least one token")
        if start_pos + seq_len > cfg.max_seq_len:
            raise ValueError(
                f"prefill range [{start_pos}, {start_pos + seq_len}) exceeds max_seq_len={cfg.max_seq_len}"
            )
        if user_id >= cfg.max_batch_size:
            raise ValueError(f"user_id={user_id} outside max_batch_size={cfg.max_batch_size}")

        # Continuation prefill on a sliding layer needs the previous call's K/V
        # window explicitly: the paged chunked SDPA op has no sliding-window
        # mask, so the prefix cannot be read back from the paged cache the way
        # full-attention layers read it.  This mirrors the generator-level
        # sliding-tail hand-off in models/demos/gemma4/tt/attention/prefill.py.
        required_tail = self.sliding_kv_tail_len(start_pos)
        if sliding_kv_tail is not None and not cfg.is_sliding:
            raise ValueError("sliding_kv_tail is only meaningful on sliding-window layers")
        if cfg.is_sliding and start_pos > 0 and sliding_kv_tail is None:
            raise ValueError(
                f"continuation prefill at start_pos={start_pos} on a sliding-window layer needs "
                f"sliding_kv_tail: the previous call's last {required_tail} K/V rows. Get it by "
                "passing return_sliding_kv_tail=True to the previous prefill_forward call."
            )
        if sliding_kv_tail is not None:
            for name, tensor in zip(("k", "v"), sliding_kv_tail):
                expected = (1, cfg.num_key_value_heads, required_tail, cfg.head_dim)
                if tuple(tensor.shape) != expected:
                    raise ValueError(
                        f"sliding_kv_tail {name} must be shaped {expected} for start_pos={start_pos}, "
                        f"got {tuple(tensor.shape)}"
                    )

        # Physical padding to the tile height. The padded tail attends causally
        # to real tokens only, its outputs are dropped below, and the junk K/V
        # it writes into the cache sits at positions >= seq_len which decode
        # never reads (cur_pos starts at the logical length).
        padded_len = ((seq_len + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
        padded_input = hidden_states
        if padded_len != seq_len:
            padded_input = ttnn.pad(hidden_states, [(0, 0), (0, 0), (0, padded_len - seq_len), (0, 0)], value=0.0)

        chunk = cfg.prefill_chunk_size
        outputs: list[ttnn.Tensor] = []
        sliding_tail: tuple[ttnn.Tensor, ttnn.Tensor] | None = sliding_kv_tail
        offset = 0
        while offset < padded_len:
            length = min(chunk, padded_len - offset)
            piece = (
                padded_input
                if length == padded_len
                else ttnn.slice(padded_input, [0, 0, offset, 0], [1, 1, offset + length, cfg.hidden_size])
            )
            is_last_chunk = offset + length >= padded_len
            out, sliding_tail = self._prefill_chunk(
                piece,
                page_table=page_table,
                user_id=user_id,
                start_pos=start_pos + offset,
                sliding_tail=sliding_tail,
                # The tail is only worth building for the next internal chunk or
                # for the caller's next continuation call.
                need_tail=return_sliding_kv_tail or not is_last_chunk,
            )
            if piece is not padded_input:
                ttnn.deallocate(piece)
            outputs.append(out)
            offset += length

        if sliding_tail is not None and not return_sliding_kv_tail:
            ttnn.deallocate(sliding_tail[0])
            ttnn.deallocate(sliding_tail[1])
            sliding_tail = None
        if padded_input is not hidden_states:
            ttnn.deallocate(padded_input)

        if len(outputs) == 1:
            result = outputs[0]
        else:
            result = ttnn.concat(outputs, dim=2)
            for piece in outputs:
                ttnn.deallocate(piece)
        if padded_len != seq_len:
            trimmed = ttnn.slice(result, [0, 0, 0, 0], [1, 1, seq_len, cfg.hidden_size])
            ttnn.deallocate(result)
            result = trimmed
        if return_sliding_kv_tail:
            # Caller owns the returned tail and must deallocate it (or hand it
            # straight back to the next continuation prefill).
            return result, sliding_tail
        return result

    # ---------------------------------------------------------------- decode

    def _decode_rope_tables(self, rope_pos_ids: ttnn.Tensor, batch: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Per-user cos/sin rows gathered on device (trace-safe)."""
        cos = ttnn.unsqueeze_to_4D(ttnn.embedding(rope_pos_ids, self.cos_cache, layout=ttnn.TILE_LAYOUT))
        sin = ttnn.unsqueeze_to_4D(ttnn.embedding(rope_pos_ids, self.sin_cache, layout=ttnn.TILE_LAYOUT))
        # [1, 1, batch, head_dim] -> [1, batch, 1, head_dim] to line up with the
        # decode head layout [1, batch, heads, head_dim].
        cos_b = ttnn.transpose(cos, 1, 2)
        sin_b = ttnn.transpose(sin, 1, 2)
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        if cos_b.shape[1] != batch:
            cos_b = cos_b[:, :batch, :, :]
            sin_b = sin_b[:, :batch, :, :]
        heads = self.config.num_attention_heads
        # binary_ng cannot broadcast over the head (height) dim inside a tile.
        cos_full = ttnn.repeat(cos_b, ttnn.Shape([1, 1, heads, 1]))
        sin_full = ttnn.repeat(sin_b, ttnn.Shape([1, 1, heads, 1]))
        cos_kv = ttnn.repeat(cos_b, ttnn.Shape([1, 1, self.config.num_key_value_heads, 1]))
        sin_kv = ttnn.repeat(sin_b, ttnn.Shape([1, 1, self.config.num_key_value_heads, 1]))
        ttnn.deallocate(cos_b)
        ttnn.deallocate(sin_b)
        return (cos_full, sin_full), (cos_kv, sin_kv)

    def decode_forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        current_pos: ttnn.Tensor,
        page_table: ttnn.Tensor,
        rope_pos_ids: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """Single-token paged decode; see the module docstring for the contract."""
        cfg = self.config
        batch = int(hidden_states.shape[-2])
        if hidden_states.shape[-1] != cfg.hidden_size:
            raise ValueError(f"decode expects hidden size {cfg.hidden_size}, got {hidden_states.shape[-1]}")
        if cfg.uses_rope and rope_pos_ids is None:
            raise ValueError("sliding (RoPE) layers require rope_pos_ids for the on-device cos/sin gather")

        residual = hidden_states
        normed = self.input_layernorm(residual)

        xqkv = ttnn.linear(normed, self.wqkv, dtype=self.activation_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # nlp_create_qkv_heads_decode's interleaved DRAM reader zeroes odd Q rows
        # on Blackhole (tt-metal #16667); stage the fused QKV in L1 first.
        xqkv_l1 = ttnn.to_memory_config(xqkv, ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(xqkv)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv_l1,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )
        ttnn.deallocate(xqkv_l1)
        sharded_memcfg = q.memory_config()

        q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG)
        v = ttnn.to_memory_config(v, ttnn.DRAM_MEMORY_CONFIG)

        q_normed = self._per_head_rmsnorm(q)
        ttnn.deallocate(q)
        k_normed = self._per_head_rmsnorm(k)
        ttnn.deallocate(k)
        q, k = q_normed, k_normed

        if cfg.uses_rope:
            (cos_q, sin_q), (cos_k, sin_k) = self._decode_rope_tables(rope_pos_ids, batch)
            q_rot = self._apply_rope(q, cos_q, sin_q)
            ttnn.deallocate(q)
            k_rot = self._apply_rope(k, cos_k, sin_k)
            ttnn.deallocate(k)
            for tensor in (cos_q, sin_q, cos_k, sin_k):
                ttnn.deallocate(tensor)
            q, k = q_rot, k_rot

        block_size = cfg.paged_attention_config.block_size
        # paged_update_cache needs a height-sharded (one user per core) update
        # tensor and owns the repack into the cache dtype itself, so K/V stay
        # BF16 here even when the cache is lower precision.
        k_sharded = ttnn.to_memory_config(k, sharded_memcfg)
        v_sharded = ttnn.to_memory_config(v, sharded_memcfg)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        ttnn.experimental.paged_update_cache(
            self.k_cache,
            k_sharded,
            update_idxs_tensor=current_pos,
            page_table=page_table,
            block_size=block_size,
            num_kv_heads=cfg.num_key_value_heads,
        )
        ttnn.experimental.paged_update_cache(
            self.v_cache,
            v_sharded,
            update_idxs_tensor=current_pos,
            page_table=page_table,
            block_size=block_size,
            num_kv_heads=cfg.num_key_value_heads,
        )
        ttnn.deallocate(k_sharded)
        ttnn.deallocate(v_sharded)

        attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            self.k_cache,
            self.v_cache,
            cur_pos_tensor=current_pos,
            page_table_tensor=page_table,
            scale=cfg.sdpa_scale,
            sliding_window_size=cfg.sliding_window,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.decode_sdpa_program_config,
        )
        ttnn.deallocate(q)

        out = self._concat_heads_decode(attn, batch)
        ttnn.deallocate(attn)

        gate = ttnn.linear(normed, self.w_attn_gate, dtype=self.activation_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(normed)
        gate = ttnn.sigmoid(gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gated = ttnn.mul(out, gate, dtype=self.activation_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out)
        ttnn.deallocate(gate)
        attn_out = ttnn.linear(gated, self.wo, dtype=self.activation_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gated)

        attn_out = self.post_attention_layernorm(attn_out)
        hidden = ttnn.add(residual, attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn_out)

        mlp_in = self.pre_feedforward_layernorm(hidden)
        mlp_out = self.mlp(mlp_in)
        ttnn.deallocate(mlp_in)
        mlp_out = self.post_feedforward_layernorm(mlp_out)
        out = ttnn.add(hidden, mlp_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(hidden)
        ttnn.deallocate(mlp_out)
        return out

    def _decode_concat_grid_width(self, batch: int) -> int | None:
        """Width of a ``batch``-core rectangle that fits the device grid.

        ``nlp_concat_heads_decode`` needs a height-sharded input with exactly
        one core per user, and ``create_sharded_memory_config`` only accepts a
        *rectangular* core range for that layout.  ``batch`` values with no
        divisor pair ``(w <= grid.x, h <= grid.y)`` — every prime above
        ``grid.x``, e.g. 13/17/19/23 on an 11x10 Blackhole grid — have no such
        rectangle; ``None`` means "use the generic concat path instead".
        """
        grid = self.mesh_device.compute_with_storage_grid_size()
        widths = [w for w in range(min(batch, grid.x), 0, -1) if batch % w == 0 and batch // w <= grid.y]
        return widths[0] if widths else None

    def _concat_heads_decode(self, attn: ttnn.Tensor, batch: int) -> ttnn.Tensor:
        """``[1, batch, heads, head_dim]`` -> ``[1, 1, batch, heads*head_dim]``."""
        cfg = self.config
        grid_x = self._decode_concat_grid_width(batch)
        if grid_x is None:
            # Generic path: transpose to [1, heads, batch, head_dim] and use the
            # interleaved concat. Slower than the sharded decode op but shape
            # agnostic, so an awkward batch degrades in speed rather than
            # failing.
            transposed = ttnn.transpose(attn, 1, 2)
            out = ttnn.experimental.nlp_concat_heads(transposed, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(transposed)
            return out

        grid = self.mesh_device.compute_with_storage_grid_size()
        core_grid = ttnn.CoreRangeSet({num_to_corerange(batch, grid_x=grid_x, grid_y=grid.y)})
        shard_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, cfg.head_dim),
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        sharded = ttnn.to_memory_config(attn, shard_config)
        concatenated = ttnn.experimental.nlp_concat_heads_decode(sharded, num_heads=cfg.num_attention_heads)
        ttnn.deallocate(sharded)
        out = ttnn.sharded_to_interleaved(concatenated, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(concatenated)
        if out.shape[2] != batch:
            trimmed = out[:, :, :batch, :]
            ttnn.deallocate(out)
            out = trimmed
        return out

    # ------------------------------------------------------------------ misc

    def forward(self, hidden_states: ttnn.Tensor, *, mode: str, **kwargs) -> ttnn.Tensor:
        if mode == "prefill":
            return self.prefill_forward(hidden_states, **kwargs)
        if mode == "decode":
            return self.decode_forward(hidden_states, **kwargs)
        raise ValueError(f"Unknown decoder mode {mode!r}; expected 'prefill' or 'decode'.")
