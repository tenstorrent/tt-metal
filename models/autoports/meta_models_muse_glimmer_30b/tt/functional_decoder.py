# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Functional TTNN decoder layer for ``meta-models/Muse-Glimmer-30B`` (text decoder).

Scope
-----
This module implements ``MuseGlimmerTextDecoderLayer`` (the ``text_config`` /
``model.language_model.layers.*`` path of the ``MuseGlimmerForConditionalGeneration``
vision-language checkpoint). The vision tower, multimodal projector and image/video
inputs are a separate bringup stage and are intentionally not modelled here.
``final_logit_softcapping`` / ``output_multiplier`` live on the LM head
(``MuseGlimmerForConditionalGeneration.forward``), not on the decoder layer, so they
are also out of scope for this file.

Layer kinds
-----------
``text_config`` interleaves two decoder-layer kinds with a period of 4
(``[sliding, sliding, sliding, full]``):

* ``sliding_attention`` + RoPE (``layer_rope_theta[i] == 500000.0``,
  ``sliding_window == 2048``);
* ``full_attention`` + **NoPE** (``layer_rope_theta[i] == 0`` means the HF model
  passes ``position_embeddings=None`` to that layer, i.e. no rotary embedding at all).

Both kinds share one implementation; the layer kind only selects the SDPA window and
whether RoPE is applied. ``FunctionalDecoder`` is therefore parameterised by
``MuseGlimmerDecoderConfig.from_hf_config(hf_config, layer_idx)``.

Layer math (HF ``MuseGlimmerTextDecoderLayer.forward``)
------------------------------------------------------
Sandwich norms; every norm is a *centered* RMSNorm (``normed * (1 + w)``)::

    xn      = input_layernorm(x)                       # eps = rms_norm_eps  (1e-5)
    a       = self_attn(xn)                            # see below
    a       = post_attention_layernorm(a)              # eps = post_norm_eps (1e-8)
    h       = x + a
    y       = pre_feedforward_layernorm(h)             # eps = rms_norm_eps
    y       = down(silu(gate(y)) * up(y))              # SwiGLU
    y       = post_feedforward_layernorm(y)            # eps = post_norm_eps
    out     = h + y

Attention (GQA 32 Q heads / 2 KV heads, ``head_dim`` 128, no biases)::

    q,k,v   = q_proj(xn), k_proj(xn), v_proj(xn)
    q       = rmsnorm(q, over head_dim, no scale) * qk_scale_factor   # 3.87
    k       = rmsnorm(k, over head_dim, no scale)
    q,k     = rope(q,k)                                # sliding layers only
    o       = softmax(q k^T / sqrt(head_dim) + mask) v  # window 2048 on sliding layers
    o       = o * sigmoid(gate_proj(xn))               # gated attention
    o       = o_proj(o)

``qk_scale_factor`` is applied to Q after the QK norm and before RoPE, exactly where HF
applies it, and SDPA runs with HF's own ``scaling = head_dim ** -0.5``. (Folding the
factor into the SDPA scale would be mathematically identical, but
``chunked_scaled_dot_product_attention`` cannot accept a Python ``scale`` at all — see
:attr:`MuseGlimmerDecoderConfig.sdpa_scale`.)

Public contract
---------------
``FunctionalDecoder.from_state_dict(state_dict, hf_config=..., layer_idx=...,
mesh_device=...)`` performs every host-side conversion (weight transposes, QKV fusion,
``1 + w`` norm folding, RoPE cos/sin caches). After construction, ``prefill_forward``
and ``decode_forward`` run entirely on device: no ``torch``, no ``ttnn.from_torch`` /
``ttnn.to_torch``, no host fallback.

``prefill_forward(hidden_states, kv_cache=..., page_table=..., user_ids=...)``
    ``hidden_states``: ``[batch, 1, seq_len, hidden_size]``, TILE, bf16.
    ``seq_len`` may be **any** value in ``[1, max_position_embeddings]``; the layer
    owns tile padding, chunking and output slicing. Returns
    ``[batch, 1, seq_len, hidden_size]`` and fills the paged K/V cache for
    positions ``[start_pos, start_pos + seq_len)`` of each ``user_ids[b]`` slot.

``decode_forward(hidden_states, kv_cache=..., page_table=..., current_pos=...,
rope_idxs=...)``
    ``hidden_states``: ``[1, 1, batch, hidden_size]``, TILE, bf16 (one token per user).
    ``current_pos``: int32 device tensor ``[batch]`` (per-user absolute position, also
    the paged-cache write index). ``rope_idxs``: uint32 device tensor ``[1, batch]``
    with the same positions, used for the on-device cos/sin gather (required for RoPE
    layers, ignored by NoPE layers). Both are plain device tensors so decode is
    trace-safe: capture once, then only their contents are updated per step.
    Returns ``[1, 1, batch, hidden_size]``.

Prefill is internally chunked over the sequence (``PREFILL_CHUNK_SIZE``) so activation
memory stays bounded at the advertised 131072-token context, and so the non-chunked
SDPA correctness cliff at 32768 Q tokens is never reached:

* ``full_attention``: ``ttnn.transformer.chunked_scaled_dot_product_attention`` reads
  the whole causal prefix out of the paged cache (filled chunk-by-chunk before the
  SDPA call).
* ``sliding_attention``: the chunked SDPA op has no window support, so each chunk runs
  the ordinary causal+windowed SDPA over an *overlapping* slice that also covers the
  preceding ``sliding_window`` positions, and keeps only the chunk's own rows. The
  slice length stays ``<= PREFILL_CHUNK_SIZE + sliding_window <= 32768``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.model_config import num_to_corerange

TILE = 32

# Non-chunked prefill SDPA is only trusted up to 32768 Q tokens (see
# models/demos/gemma4/tt/attention/operations.py: beyond that the op silently returns
# wrong results). Every prefill path here stays at or below this bound by chunking.
SDPA_MAX_SEQ = 32768

# Sequence chunk for prefill. Bounds activation memory (the SwiGLU intermediate is
# chunk * 19968 elements) and keeps every SDPA call inside SDPA_MAX_SEQ. Must be a
# multiple of the paged-cache block size, of the chunked-SDPA q_chunk_size, and small
# enough that chunk + sliding_window <= SDPA_MAX_SEQ.
PREFILL_CHUNK_SIZE = 8192

# SDPA prefill chunk sizes. Measured on this Blackhole grid by
# scripts/sweep_core_grids.py under this layer's own compute-kernel config
# (doc/functional_decoder/perf/core_grid_sweep.md): q=256 / k=256 is the fastest *legal*
# combination for both prefill call sites at 8192 tokens — 7.46-7.55 ms, against
# 8.44-8.59 ms at k=128, 10.3 ms at q=512/k=128 and 12.0-12.2 ms at q=128/k=128. Larger is
# not better and not always legal: q=1024 (any k) and q=512/k=256 exceed L1 outright
# ("Statically allocated circular buffers ... grow to 2294528 B which is beyond max L1 size
# of 1572864 B"), which is why the candidate list is capped rather than maximised.
# chunked_scaled_dot_product_attention additionally requires chunk_start_idx to be a
# multiple of both chunk sizes.
PREFILL_SDPA_Q_CHUNK = 256
PREFILL_SDPA_K_CHUNK = 256
# Candidate SDPA q/k chunk sizes, largest first; a chunk size must divide the Q length.
SDPA_CHUNK_CANDIDATES = (512, 256, 128, 64, 32)

# Decode SDPA grid and K chunk, also from the sweep (batch 1 and batch 32 both measured):
# at batch 1 an 8x4 sub-grid beats the full 11x10 grid by ~10-25% (flash-decode splits the K
# sequence across cores, and with only 2 KV heads the wider grid adds reduction-tree depth
# without adding useful parallelism); at batch 32 the 8x4 grid is not legal at all (see
# _decode_sdpa_program_config) and the full grid is within 2% of the best legal grid (11x8).
# k_chunk 64 is the measured best at batch 32 for both layer kinds and is within a few
# microseconds of the best at batch 1, so one value is used everywhere.
DECODE_SDPA_GRID = (8, 4)
DECODE_SDPA_K_CHUNK = 64

DEFAULT_BLOCK_SIZE = 64


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _lcm(first: int, second: int) -> int:
    return first * second // math.gcd(first, second)


@dataclass(frozen=True)
class MuseGlimmerDecoderConfig:
    """Everything ``FunctionalDecoder`` needs from ``MuseGlimmerTextConfig``."""

    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    post_norm_eps: float
    qk_scale_factor: float
    layer_type: str
    sliding_window: int | None
    rope_theta: float | None
    max_position_embeddings: int
    layer_idx: int

    @classmethod
    def from_hf_config(cls, hf_config, layer_idx: int) -> "MuseGlimmerDecoderConfig":
        """Build the layer config from an HF ``MuseGlimmer*Config``.

        Accepts either the top-level ``MuseGlimmerConfig`` (uses ``.text_config``) or a
        ``MuseGlimmerTextConfig`` directly.
        """
        text_config = getattr(hf_config, "text_config", hf_config)
        num_layers = text_config.num_hidden_layers
        if not 0 <= layer_idx < num_layers:
            raise ValueError(f"layer_idx {layer_idx} out of range for {num_layers} layers")

        layer_type = text_config.layer_types[layer_idx]
        if layer_type not in ("sliding_attention", "full_attention"):
            raise ValueError(f"Unsupported layer_types[{layer_idx}] = {layer_type!r}")

        rope_theta = text_config.layer_rope_theta[layer_idx]
        # layer_rope_theta[i] == 0 is the HF model's NoPE marker: MuseGlimmerTextModel
        # passes position_embeddings=None to that layer.
        rope_theta = float(rope_theta) if rope_theta else None
        if rope_theta is not None:
            # HF builds ONE rotary module, from rope_parameters["rope_theta"], and uses
            # layer_rope_theta only as the on/off marker. Reading the per-layer value as the
            # base is therefore only correct while the two agree — which they do for this
            # checkpoint. Fail loudly rather than silently rotate at the wrong frequency if a
            # future checkpoint makes them differ.
            shared_theta = float(text_config.rope_parameters["rope_theta"])
            if rope_theta != shared_theta:
                raise ValueError(
                    f"layer_rope_theta[{layer_idx}] = {rope_theta} disagrees with "
                    f"rope_parameters['rope_theta'] = {shared_theta}. HF applies the shared "
                    "value to every RoPE layer, so a genuinely per-layer theta needs the "
                    "rotary cache to be built per layer before this can be supported."
                )

        sliding_window = text_config.sliding_window if layer_type == "sliding_attention" else None
        if layer_type == "sliding_attention":
            if not sliding_window or sliding_window <= 0:
                raise ValueError(f"sliding_attention layer {layer_idx} needs a positive sliding_window")
            if sliding_window % TILE != 0:
                raise ValueError(f"sliding_window {sliding_window} must be a multiple of the tile height {TILE}")

        if text_config.hidden_activation != "silu":
            raise ValueError(f"Unsupported hidden_activation {text_config.hidden_activation!r} (expected 'silu')")
        if text_config.attention_bias:
            raise ValueError("attention_bias=True is not implemented for this decoder")
        if text_config.num_attention_heads % text_config.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be a multiple of num_key_value_heads")

        return cls(
            hidden_size=text_config.hidden_size,
            intermediate_size=text_config.intermediate_size,
            num_attention_heads=text_config.num_attention_heads,
            num_key_value_heads=text_config.num_key_value_heads,
            head_dim=text_config.head_dim,
            rms_norm_eps=text_config.rms_norm_eps,
            post_norm_eps=text_config.post_norm_eps,
            qk_scale_factor=text_config.qk_scale_factor,
            layer_type=layer_type,
            sliding_window=sliding_window,
            rope_theta=rope_theta,
            max_position_embeddings=text_config.max_position_embeddings,
            layer_idx=layer_idx,
        )

    @property
    def is_sliding(self) -> bool:
        return self.layer_type == "sliding_attention"

    @property
    def uses_rope(self) -> bool:
        return self.rope_theta is not None

    @property
    def sdpa_scale(self) -> float:
        """HF ``MuseGlimmerTextAttention.scaling`` (``head_dim ** -0.5``).

        ``qk_scale_factor`` is applied to Q separately, exactly as HF does, rather than
        folded in here: ``ttnn.transformer.chunked_scaled_dot_product_attention``'s
        nanobind signature marks ``scale`` ``.noconvert()`` on a ``std::optional<float>``,
        so a Python float cannot be passed to that op at all and it must fall back to its
        default ``1 / sqrt(head_dim)`` — which is this value.
        """
        return 1.0 / math.sqrt(self.head_dim)

    @property
    def qkv_width(self) -> int:
        return (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim

    @property
    def kind_id(self) -> str:
        """Short identifier for the layer kind, used in artifact paths."""
        return "sliding_rope" if self.is_sliding else "full_nope"


class FunctionalDecoder(LightweightModule):
    """One Muse-Glimmer text decoder layer on a TTNN mesh device."""

    def __init__(
        self,
        *,
        config: MuseGlimmerDecoderConfig,
        mesh_device,
        weights: dict,
        rope_cache: dict | None,
        cache_dtype=ttnn.bfloat16,
        block_size: int = DEFAULT_BLOCK_SIZE,
        prefill_chunk_size: int = PREFILL_CHUNK_SIZE,
        sdpa_core_grid: tuple[int, int] | None = None,
        prefill_sdpa_q_chunk: int = PREFILL_SDPA_Q_CHUNK,
        prefill_sdpa_k_chunk: int = PREFILL_SDPA_K_CHUNK,
        decode_sdpa_core_grid: tuple[int, int] | None = None,
        decode_sdpa_k_chunk: int | None = None,
    ):
        super().__init__()
        self.config = config
        self.mesh_device = mesh_device
        self.cache_dtype = cache_dtype
        self.block_size = block_size
        self.prefill_chunk_size = prefill_chunk_size

        if block_size % TILE != 0:
            raise ValueError(f"block_size {block_size} must be a multiple of {TILE}")
        if prefill_chunk_size % block_size != 0:
            raise ValueError(f"prefill_chunk_size {prefill_chunk_size} must be a multiple of block_size {block_size}")
        if prefill_chunk_size % prefill_sdpa_q_chunk != 0 or prefill_chunk_size % prefill_sdpa_k_chunk != 0:
            raise ValueError(
                f"prefill_chunk_size {prefill_chunk_size} must be a multiple of the SDPA chunk sizes "
                f"({prefill_sdpa_q_chunk}, {prefill_sdpa_k_chunk}): chunked-SDPA chunk_start_idx is a "
                f"multiple of prefill_chunk_size and must also be a multiple of both chunk sizes"
            )
        window = config.sliding_window or 0
        if prefill_chunk_size + window > SDPA_MAX_SEQ:
            raise ValueError(
                f"prefill_chunk_size {prefill_chunk_size} + sliding_window {window} exceeds the "
                f"non-chunked SDPA correctness bound {SDPA_MAX_SEQ}"
            )

        # Weight tensors (already device-resident, produced by from_state_dict).
        self.wqkv = weights["wqkv"]
        self.w_attn_gate = weights["w_attn_gate"]
        self.wo = weights["wo"]
        self.w_mlp_gate = weights["w_mlp_gate"]
        self.w_mlp_up = weights["w_mlp_up"]
        self.w_mlp_down = weights["w_mlp_down"]
        self.input_norm_w = weights["input_norm_w"]
        self.post_attn_norm_w = weights["post_attn_norm_w"]
        self.pre_ff_norm_w = weights["pre_ff_norm_w"]
        self.post_ff_norm_w = weights["post_ff_norm_w"]
        self.rope_cache = rope_cache

        grid = mesh_device.compute_with_storage_grid_size()
        self.device_grid = (grid.x, grid.y)
        # Default to the whole Blackhole compute grid; scripts/sweep_core_grids.py measures
        # the legal alternatives and doc/functional_decoder/perf/core_grid_sweep.md records
        # the result that picked this default.
        self.sdpa_core_grid = sdpa_core_grid or self.device_grid
        self.prefill_sdpa_q_chunk = prefill_sdpa_q_chunk
        self.prefill_sdpa_k_chunk = prefill_sdpa_k_chunk
        self.decode_sdpa_core_grid = decode_sdpa_core_grid or (
            min(DECODE_SDPA_GRID[0], grid.x),
            min(DECODE_SDPA_GRID[1], grid.y),
        )
        self.decode_sdpa_k_chunk = decode_sdpa_k_chunk or DECODE_SDPA_K_CHUNK

        arch = mesh_device.arch()
        # Architecture-appropriate compute-kernel configs (Blackhole here): built from
        # the live device arch rather than a hard-coded arch class.
        self.hifi4_kernel_config = ttnn.init_device_compute_kernel_config(
            arch,
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.hifi2_kernel_config = ttnn.init_device_compute_kernel_config(
            arch,
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.norm_kernel_config = self.hifi4_kernel_config
        self.sdpa_kernel_config = self.hifi4_kernel_config

    # ------------------------------------------------------------------ setup

    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx: int,
        mesh_device,
        state_dict_prefix: str | None = None,
        weight_dtype=ttnn.bfloat16,
        cache_dtype=ttnn.bfloat16,
        rope_dtype=ttnn.bfloat16,
        block_size: int = DEFAULT_BLOCK_SIZE,
        prefill_chunk_size: int = PREFILL_CHUNK_SIZE,
        rope_max_seq_len: int | None = None,
        sdpa_core_grid: tuple[int, int] | None = None,
        prefill_sdpa_q_chunk: int = PREFILL_SDPA_Q_CHUNK,
        prefill_sdpa_k_chunk: int = PREFILL_SDPA_K_CHUNK,
        decode_sdpa_core_grid: tuple[int, int] | None = None,
        decode_sdpa_k_chunk: int | None = None,
    ) -> "FunctionalDecoder":
        """Build the layer from a (real or synthetic) HF state dict.

        ``state_dict`` keys are the HF names, either layer-local
        (``self_attn.q_proj.weight``) or prefixed
        (``model.language_model.layers.<i>.self_attn.q_proj.weight``). Pass
        ``state_dict_prefix`` to strip an explicit prefix; otherwise the default HF
        checkpoint prefix for ``layer_idx`` is tried, then the layer-local names.

        All host work (torch transposes, QKV fusion, ``1 + w`` norm folding, RoPE
        cos/sin caches) happens here so that prefill/decode are pure device paths.
        """
        import torch  # setup-time only: weight conversion never happens at runtime

        config = MuseGlimmerDecoderConfig.from_hf_config(hf_config, layer_idx)

        if state_dict_prefix is None:
            default_prefix = f"model.language_model.layers.{layer_idx}."
            state_dict_prefix = default_prefix if any(k.startswith(default_prefix) for k in state_dict) else ""

        def get(name: str) -> "torch.Tensor":
            key = f"{state_dict_prefix}{name}"
            if key not in state_dict:
                raise KeyError(f"missing weight {key!r} in state dict")
            return state_dict[key].to(torch.float32)

        hidden = config.hidden_size
        n_heads = config.num_attention_heads
        n_kv = config.num_key_value_heads
        head_dim = config.head_dim

        q_proj = get("self_attn.q_proj.weight")
        k_proj = get("self_attn.k_proj.weight")
        v_proj = get("self_attn.v_proj.weight")
        o_proj = get("self_attn.o_proj.weight")
        attn_gate = get("self_attn.gate_proj.weight")
        mlp_gate = get("mlp.gate_proj.weight")
        mlp_up = get("mlp.up_proj.weight")
        mlp_down = get("mlp.down_proj.weight")

        expected = {
            "self_attn.q_proj.weight": (n_heads * head_dim, hidden),
            "self_attn.k_proj.weight": (n_kv * head_dim, hidden),
            "self_attn.v_proj.weight": (n_kv * head_dim, hidden),
            "self_attn.o_proj.weight": (hidden, n_heads * head_dim),
            "self_attn.gate_proj.weight": (n_heads * head_dim, hidden),
            "mlp.gate_proj.weight": (config.intermediate_size, hidden),
            "mlp.up_proj.weight": (config.intermediate_size, hidden),
            "mlp.down_proj.weight": (hidden, config.intermediate_size),
        }
        actual = {
            "self_attn.q_proj.weight": q_proj,
            "self_attn.k_proj.weight": k_proj,
            "self_attn.v_proj.weight": v_proj,
            "self_attn.o_proj.weight": o_proj,
            "self_attn.gate_proj.weight": attn_gate,
            "mlp.gate_proj.weight": mlp_gate,
            "mlp.up_proj.weight": mlp_up,
            "mlp.down_proj.weight": mlp_down,
        }
        for name, shape in expected.items():
            if tuple(actual[name].shape) != shape:
                raise ValueError(f"{name} has shape {tuple(actual[name].shape)}, expected {shape}")

        def as_device_tensor(torch_tensor, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT):
            return ttnn.as_tensor(
                torch_tensor,
                dtype=dtype,
                layout=layout,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )

        # Fused QKV: ttnn.experimental.nlp_create_qkv_heads* expects the fused width
        # ordered [q_heads * head_dim, k_heads * head_dim, v_heads * head_dim].
        wqkv = torch.cat([q_proj, k_proj, v_proj], dim=0).transpose(0, 1).contiguous()

        def norm_weight(name: str):
            # HF MuseGlimmerTextCenteredRMSNorm: normed * (1 + weight). ttnn.rms_norm
            # computes normed * weight, so fold the unit offset in at setup time.
            w = get(name)
            if tuple(w.shape) != (hidden,):
                raise ValueError(f"{name} has shape {tuple(w.shape)}, expected {(hidden,)}")
            return as_device_tensor((1.0 + w).reshape(1, 1, 1, hidden))

        weights = {
            "wqkv": as_device_tensor(wqkv),
            "w_attn_gate": as_device_tensor(attn_gate.transpose(0, 1).contiguous()),
            "wo": as_device_tensor(o_proj.transpose(0, 1).contiguous()),
            "w_mlp_gate": as_device_tensor(mlp_gate.transpose(0, 1).contiguous()),
            "w_mlp_up": as_device_tensor(mlp_up.transpose(0, 1).contiguous()),
            "w_mlp_down": as_device_tensor(mlp_down.transpose(0, 1).contiguous()),
            "input_norm_w": norm_weight("input_layernorm.weight"),
            "post_attn_norm_w": norm_weight("post_attention_layernorm.weight"),
            "pre_ff_norm_w": norm_weight("pre_feedforward_layernorm.weight"),
            "post_ff_norm_w": norm_weight("post_feedforward_layernorm.weight"),
        }

        rope_cache = None
        if config.uses_rope:
            max_pos = rope_max_seq_len or config.max_position_embeddings
            # HF MuseGlimmerTextRotaryEmbedding: inv_freq = 1/theta^(arange(0,d,2)/d),
            # emb = cat(freqs, freqs), cos/sin = emb.cos()/emb.sin() (NeoX rotate_half).
            inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
            positions = torch.arange(max_pos, dtype=torch.float32)
            freqs = torch.outer(positions, inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            cos = emb.cos()
            sin = emb.sin()
            rope_cache = {
                "cos_prefill": as_device_tensor(cos.reshape(1, 1, max_pos, head_dim), dtype=rope_dtype),
                "sin_prefill": as_device_tensor(sin.reshape(1, 1, max_pos, head_dim), dtype=rope_dtype),
                # Row-major 2D caches for the trace-safe ttnn.embedding position gather.
                "cos_decode": as_device_tensor(cos, dtype=rope_dtype, layout=ttnn.ROW_MAJOR_LAYOUT),
                "sin_decode": as_device_tensor(sin, dtype=rope_dtype, layout=ttnn.ROW_MAJOR_LAYOUT),
                "max_seq_len": max_pos,
            }

        return cls(
            config=config,
            mesh_device=mesh_device,
            weights=weights,
            rope_cache=rope_cache,
            cache_dtype=cache_dtype,
            block_size=block_size,
            prefill_chunk_size=prefill_chunk_size,
            sdpa_core_grid=sdpa_core_grid,
            prefill_sdpa_q_chunk=prefill_sdpa_q_chunk,
            prefill_sdpa_k_chunk=prefill_sdpa_k_chunk,
            decode_sdpa_core_grid=decode_sdpa_core_grid,
            decode_sdpa_k_chunk=decode_sdpa_k_chunk,
        )

    # ----------------------------------------------------------- kv cache api

    def allocate_kv_cache(self, *, batch_size: int, max_seq_len: int, num_blocks: int | None = None):
        """Allocate an empty paged K/V cache pair for this layer.

        Returns ``(k_cache, v_cache)`` shaped
        ``[num_blocks, num_key_value_heads, block_size, head_dim]``. ``num_blocks``
        defaults to exactly enough blocks for ``batch_size`` sequences of
        ``max_seq_len`` tokens; pass a larger value to model a shared block pool.
        """
        import torch  # setup-time only

        blocks_per_seq = self.blocks_per_seq(max_seq_len)
        total_blocks = num_blocks if num_blocks is not None else blocks_per_seq * batch_size
        if total_blocks < blocks_per_seq * batch_size:
            raise ValueError(
                f"num_blocks {total_blocks} cannot hold {batch_size} sequences of {max_seq_len} tokens "
                f"({blocks_per_seq} blocks each)"
            )
        shape = (total_blocks, self.config.num_key_value_heads, self.block_size, self.config.head_dim)
        caches = []
        for _ in range(2):
            caches.append(
                ttnn.as_tensor(
                    torch.zeros(shape, dtype=torch.float32),
                    dtype=self.cache_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                )
            )
        return caches[0], caches[1]

    # ------------------------------------------------------------- primitives

    @staticmethod
    def _slice_view(tensor, begins, ends):
        """``ttnn.slice`` that reports whether a new buffer was produced.

        For a full-range slice ``ttnn.slice`` returns the input itself
        (``slice.cpp``: ``no_step && starts_zero && ends_max``), and the returned handle
        shares the input's buffer — deallocating it would free the caller's tensor (a
        weight, the RoPE cache, or the page table). Detect that case up front and return
        ``owned=False`` so callers only free real copies.
        """
        shape = list(tensor.shape)
        if all(b == 0 for b in begins) and all(e == s for e, s in zip(ends, shape)):
            return tensor, False
        return ttnn.slice(tensor, begins, ends), True

    def _norm(self, x, weight, eps):
        return ttnn.rms_norm(
            x,
            weight=weight,
            epsilon=eps,
            compute_kernel_config=self.norm_kernel_config,
        )

    def _qk_norm(self, x):
        """RMSNorm over ``head_dim`` with no scale (HF ``MuseGlimmerRMSNorm(with_scale=False)``)."""
        return ttnn.rms_norm(
            x,
            epsilon=self.config.rms_norm_eps,
            compute_kernel_config=self.norm_kernel_config,
        )

    def _mlp(self, x):
        cfg = self.config
        gate = ttnn.linear(x, self.w_mlp_gate, compute_kernel_config=self.hifi2_kernel_config)
        up = ttnn.linear(x, self.w_mlp_up, compute_kernel_config=self.hifi2_kernel_config)
        activated = ttnn.multiply(ttnn.silu(gate), up)
        gate.deallocate(True)
        up.deallocate(True)
        out = ttnn.linear(activated, self.w_mlp_down, compute_kernel_config=self.hifi2_kernel_config)
        activated.deallocate(True)
        return out

    def _sdpa_program_config(self, *, q_chunk: int, k_chunk: int, grid: tuple[int, int] | None = None):
        grid = grid or self.sdpa_core_grid
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(grid[0], grid[1]),
            q_chunk_size=q_chunk,
            k_chunk_size=k_chunk,
            exp_approx_mode=False,
        )

    # ---------------------------------------------------------------- prefill

    def prefill_forward(
        self,
        hidden_states,
        *,
        kv_cache,
        page_table,
        user_ids=None,
        seq_len: int | None = None,
        start_pos: int = 0,
    ):
        """Multi-token prefill for one or more users. See the module docstring.

        Args:
            hidden_states: ``[batch, 1, seq_len, hidden_size]`` device tensor (TILE).
            kv_cache: ``(k_cache, v_cache)`` from :meth:`allocate_kv_cache`.
            page_table: int32 ``[rows, blocks_per_seq]`` device tensor. Row ``r`` maps the
                logical blocks of cache slot ``r`` to physical cache blocks. ``rows`` may
                exceed the input batch (a shared slot pool).
            user_ids: host ``list[int]`` of length ``batch``: the page-table row (cache
                slot) each input row belongs to. Defaults to ``range(batch)``. These are
                host-side scheduling metadata, not a device tensor, so prefill stays free
                of runtime host transfers.
            seq_len: logical sequence length, defaults to ``hidden_states.shape[-2]``.
            start_pos: absolute position of the first token (multiple of the tile height and
                of the block size). Non-zero values continue an existing cache and are
                supported on ``full_attention`` layers only — a ``sliding_attention`` layer
                raises, because its window prefix comes from the input tensor rather than from
                the cache (see the check below). The default 0 is a fresh prompt; a single call
                already handles any length up to the full context by chunking internally.
        """
        cfg = self.config
        batch = hidden_states.shape[0]
        rows = list(range(batch)) if user_ids is None else [int(u) for u in user_ids]
        if len(rows) != batch:
            raise ValueError(f"user_ids has {len(rows)} entries for a batch of {batch}")
        if len(set(rows)) != len(rows):
            raise ValueError(f"user_ids must be distinct cache slots, got {rows}")
        if max(rows) >= page_table.shape[0]:
            raise ValueError(f"user_ids {rows} exceed the {page_table.shape[0]} page-table rows")
        logical_seq = seq_len if seq_len is not None else hidden_states.shape[-2]
        if logical_seq <= 0:
            raise ValueError(f"seq_len must be positive, got {logical_seq}")
        if hidden_states.shape[-2] < logical_seq:
            raise ValueError(f"hidden_states has {hidden_states.shape[-2]} rows, need at least {logical_seq}")
        if start_pos % TILE != 0 or start_pos % self.block_size != 0:
            raise ValueError(f"start_pos {start_pos} must be a multiple of {TILE} and block_size {self.block_size}")
        if start_pos and cfg.is_sliding:
            # The windowed prefill SDPA takes its window prefix from the *input tensor*
            # (`ext_start = chunk_start - sliding_window`), and the chunked SDPA op that could
            # read the prefix out of the paged cache has no sliding-window mode. A continued
            # segment would therefore attend to a truncated window for its first
            # `sliding_window` positions — a silent wrong answer. Fail instead.
            raise ValueError(
                f"start_pos={start_pos} (continued prefill) is not supported on a "
                f"sliding_attention layer: the first {cfg.sliding_window} positions of the new "
                "segment would attend to a truncated window. Prefill the whole prompt in one "
                "call (this layer already chunks internally to bound memory), or extend the "
                "sliding path to read its window prefix from the paged cache."
            )
        if start_pos + logical_seq > cfg.max_position_embeddings:
            raise ValueError(
                f"start_pos + seq_len = {start_pos + logical_seq} exceeds max_position_embeddings "
                f"{cfg.max_position_embeddings}"
            )

        # Tile-align the working length. The pad rows carry K/V derived from zero
        # activations; causal masking means real query rows never read them, and decode
        # only reads cache positions <= current_pos, so they are inert.
        padded_seq = _round_up(logical_seq, TILE)
        x = hidden_states
        x_owned = False
        if x.shape[-2] < padded_seq:
            x = ttnn.pad(x, [(0, 0), (0, 0), (0, padded_seq - x.shape[-2]), (0, 0)], value=0.0)
            x_owned = True
        elif x.shape[-2] > padded_seq:
            x, x_owned = self._slice_view(x, [0, 0, 0, 0], [batch, x.shape[1], padded_seq, cfg.hidden_size])

        hist = cfg.sliding_window if cfg.is_sliding else 0
        chunk = self.prefill_chunk_size
        outputs = []
        chunk_start = 0
        while chunk_start < padded_seq:
            chunk_len = min(chunk, padded_seq - chunk_start)
            ext_start = max(0, chunk_start - hist) if cfg.is_sliding else chunk_start
            outputs.append(
                self._prefill_chunk(
                    x,
                    kv_cache=kv_cache,
                    page_table=page_table,
                    rows=rows,
                    ext_start=ext_start,
                    chunk_start=chunk_start,
                    chunk_len=chunk_len,
                    start_pos=start_pos,
                )
            )
            chunk_start += chunk_len

        if x_owned:
            x.deallocate(True)

        out = outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=2)
        if len(outputs) > 1:
            for chunk_out in outputs:
                chunk_out.deallocate(True)
        if out.shape[-2] != logical_seq:
            sliced, owned = self._slice_view(out, [0, 0, 0, 0], [batch, 1, logical_seq, cfg.hidden_size])
            if owned:
                out.deallocate(True)
            out = sliced
        return out

    def _prefill_chunk(
        self,
        x,
        *,
        kv_cache,
        page_table,
        rows,
        ext_start: int,
        chunk_start: int,
        chunk_len: int,
        start_pos: int,
    ):
        cfg = self.config
        batch = x.shape[0]
        hidden = cfg.hidden_size
        ext_len = chunk_start + chunk_len - ext_start
        keep_from = chunk_start - ext_start

        x_ext = (
            x
            if (ext_start == 0 and ext_len == x.shape[-2])
            else ttnn.slice(x, [0, 0, ext_start, 0], [batch, 1, ext_start + ext_len, hidden])
        )
        xn_ext = self._norm(x_ext, self.input_norm_w, cfg.rms_norm_eps)

        xqkv = ttnn.linear(xn_ext, self.wqkv, compute_kernel_config=self.hifi2_kernel_config)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        xqkv.deallocate(True)

        q = self._qk_norm(q)
        k = self._qk_norm(k)
        # HF scales Q by qk_scale_factor after the QK norm and before RoPE.
        q = ttnn.multiply(q, cfg.qk_scale_factor)

        if cfg.uses_rope:
            q, k = self._rope_prefill(q, k, position_start=start_pos + ext_start, length=ext_len)

        # Fill only this chunk's new positions; ext_start < chunk_start rows were
        # already written by the previous chunk.
        k_new = k if keep_from == 0 else ttnn.slice(k, [0, 0, keep_from, 0], [batch, k.shape[1], ext_len, cfg.head_dim])
        v_new = v if keep_from == 0 else ttnn.slice(v, [0, 0, keep_from, 0], [batch, v.shape[1], ext_len, cfg.head_dim])
        self._fill_paged_cache(
            kv_cache,
            k_new,
            v_new,
            page_table=page_table,
            rows=rows,
            absolute_start=start_pos + chunk_start,
        )
        if k_new is not k:
            k_new.deallocate(True)
        if v_new is not v:
            v_new.deallocate(True)

        if cfg.is_sliding:
            attn = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                scale=cfg.sdpa_scale,
                sliding_window_size=cfg.sliding_window,
                program_config=self._sdpa_program_config(
                    # Both chunk sizes must divide the slice length (see
                    # _sdpa_chunk_for_length: a non-dividing q_chunk_size hangs the op).
                    q_chunk=self._sdpa_chunk_for_length(ext_len, self.prefill_sdpa_q_chunk),
                    k_chunk=self._sdpa_chunk_for_length(ext_len, self.prefill_sdpa_k_chunk),
                ),
                compute_kernel_config=self.sdpa_kernel_config,
            )
            q.deallocate(True)
            k.deallocate(True)
            v.deallocate(True)
            if keep_from:
                trimmed = ttnn.slice(
                    attn, [0, 0, keep_from, 0], [batch, cfg.num_attention_heads, ext_len, cfg.head_dim]
                )
                attn.deallocate(True)
                attn = trimmed
        else:
            k.deallocate(True)
            v.deallocate(True)
            attn = self._chunked_prefill_sdpa(
                q,
                kv_cache=kv_cache,
                page_table=page_table,
                rows=rows,
                chunk_start_idx=start_pos + chunk_start,
                chunk_len=chunk_len,
            )
            q.deallocate(True)

        attn_cat = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn.deallocate(True)

        xn = xn_ext if keep_from == 0 else ttnn.slice(xn_ext, [0, 0, keep_from, 0], [batch, 1, ext_len, hidden])
        gate = ttnn.linear(xn, self.w_attn_gate, compute_kernel_config=self.hifi2_kernel_config)
        gate_sigmoid = ttnn.sigmoid(gate)
        gate.deallocate(True)
        attn_cat = ttnn.multiply(attn_cat, gate_sigmoid)
        gate_sigmoid.deallocate(True)
        if xn is not xn_ext:
            xn.deallocate(True)
        xn_ext.deallocate(True)

        attn_out = ttnn.linear(attn_cat, self.wo, compute_kernel_config=self.hifi2_kernel_config)
        attn_cat.deallocate(True)
        attn_out = self._norm(attn_out, self.post_attn_norm_w, cfg.post_norm_eps)

        x_chunk = x_ext if keep_from == 0 else ttnn.slice(x_ext, [0, 0, keep_from, 0], [batch, 1, ext_len, hidden])
        h = ttnn.add(x_chunk, attn_out)
        attn_out.deallocate(True)
        if x_chunk is not x_ext:
            x_chunk.deallocate(True)
        if x_ext is not x:
            x_ext.deallocate(True)

        y = self._norm(h, self.pre_ff_norm_w, cfg.rms_norm_eps)
        y = self._mlp(y)
        y = self._norm(y, self.post_ff_norm_w, cfg.post_norm_eps)
        out = ttnn.add(h, y)
        h.deallocate(True)
        y.deallocate(True)
        return out

    def _rope_prefill(self, q, k, *, position_start: int, length: int):
        cfg = self.config
        cache = self.rope_cache
        if position_start % TILE != 0:
            raise ValueError(f"rope position_start {position_start} must be a multiple of {TILE}")
        begins = [0, 0, position_start, 0]
        ends = [1, 1, position_start + length, cfg.head_dim]
        cos, cos_owned = self._slice_view(cache["cos_prefill"], begins, ends)
        sin, sin_owned = self._slice_view(cache["sin_prefill"], begins, ends)
        q_out = ttnn.experimental.rotary_embedding_hf(q, cos, sin, is_decode_mode=False)
        k_out = ttnn.experimental.rotary_embedding_hf(k, cos, sin, is_decode_mode=False)
        q.deallocate(True)
        k.deallocate(True)
        if cos_owned:
            cos.deallocate(True)
        if sin_owned:
            sin.deallocate(True)
        return q_out, k_out

    def _fill_paged_cache(self, kv_cache, k, v, *, page_table, rows, absolute_start: int):
        """Write ``k``/``v`` into the paged cache at absolute positions ``[absolute_start, ...)``.

        ``paged_fill_cache`` always maps input tile 0 to the *first block of the page table
        it is given*, so a chunk that starts at ``absolute_start`` is written through a
        page-table slice that starts at that chunk's first block. The same slice also
        selects the user's row, which keeps the op on its single-row path
        (``batch_idx = 0``) and needs no host-built ``batch_idx_tensor``.
        """
        k_cache, v_cache = kv_cache
        batch = k.shape[0]
        first_block = absolute_start // self.block_size
        blocks_needed = _round_up(k.shape[-2], self.block_size) // self.block_size
        if first_block + blocks_needed > page_table.shape[1]:
            raise ValueError(
                f"page table has {page_table.shape[1]} blocks per sequence, need "
                f"{first_block + blocks_needed} for positions [{absolute_start}, "
                f"{absolute_start + k.shape[-2]})"
            )

        k_fill = k if k.dtype == self.cache_dtype else ttnn.typecast(k, self.cache_dtype)
        v_fill = v if v.dtype == self.cache_dtype else ttnn.typecast(v, self.cache_dtype)
        for index, row in enumerate(rows):
            row_page_table, pt_owned = self._slice_view(
                page_table, [row, first_block], [row + 1, first_block + blocks_needed]
            )
            if batch == 1:
                k_user, v_user, user_owned = k_fill, v_fill, False
            else:
                k_user, user_owned = self._slice_view(
                    k_fill, [index, 0, 0, 0], [index + 1, k.shape[1], k.shape[2], k.shape[3]]
                )
                v_user, _ = self._slice_view(v_fill, [index, 0, 0, 0], [index + 1, v.shape[1], v.shape[2], v.shape[3]])
            ttnn.experimental.paged_fill_cache(k_cache, k_user, row_page_table, block_size=self.block_size)
            ttnn.experimental.paged_fill_cache(v_cache, v_user, row_page_table, block_size=self.block_size)
            if pt_owned:
                row_page_table.deallocate(True)
            if user_owned:
                k_user.deallocate(True)
                v_user.deallocate(True)
        if k_fill is not k:
            k_fill.deallocate(True)
        if v_fill is not v:
            v_fill.deallocate(True)

    @staticmethod
    def _sdpa_chunk_for_length(length: int, cap: int) -> int:
        """Largest candidate chunk size that is ``<= cap`` and *divides* ``length``.

        Divisibility is not optional. ``ttnn.transformer.scaled_dot_product_attention``
        **hangs** (host spins in the command queue; ``tools/tt-triage.py`` shows
        ``SDPAOperation`` stuck on 32 cores) when ``q_chunk_size`` does not divide the Q
        sequence length — reproduced with Q ``[1, 32, 2080, 128]`` and ``q_chunk_size=512``
        (2080 = 4*512 + 32). Because ``length`` is always tile-aligned, 32 is always a
        valid fallback.
        """
        if length % TILE != 0:
            raise ValueError(f"SDPA sequence length {length} must be tile-aligned")
        for candidate in SDPA_CHUNK_CANDIDATES:
            if candidate <= cap and candidate <= length and length % candidate == 0:
                return candidate
        return TILE

    def _chunked_sdpa_chunk_sizes(self, chunk_len: int, chunk_start_idx: int, kv_length: int) -> tuple[int, int]:
        """Pick ``(q_chunk_size, k_chunk_size)`` for ``chunked_scaled_dot_product_attention``.

        The op's constraints (``sdpa_device_operation.cpp``) are:

        * both chunk sizes are multiples of the tile height;
        * ``chunk_start_idx`` is a multiple of both (the second is the workaround for
          tt-metal issue 35225);
        * ``kv_length >= q_len + chunk_start_idx``, where ``kv_length`` is
          ``page_table.shape[1] * block_size`` — the *page-table capacity*, not the
          filled length.

        The last one is why the Q chunk must not be zero-padded up to a fixed
        ``q_chunk_size``: padding a tile-aligned tail chunk (e.g. 2080 -> 2176) makes the
        op believe the prefix runs past the end of the page table and it fails. Instead
        pick the largest chunk size that divides the actual chunk length, so Q is passed
        at its real (tile-aligned) length.

        Both candidates are capped by ``prefill_sdpa_q_chunk`` / ``prefill_sdpa_k_chunk``, so
        this call site runs the same measured-optimal geometry as the sliding one rather than
        the largest size that happens to divide.

        The K chunk carries one extra condition the op does *not* validate: the whole
        ``k_chunk``-padded prefix must still be addressable by the page table. The op only
        checks ``kv_length >= q_len + chunk_start_idx`` (``sdpa_device_operation.cpp``), but
        its program factory rounds the K extent up to the chunk size
        (``sdpa_program_factory.cpp``: ``padded_Sk = ceil(Sk / k_chunk_size) * k_chunk_size``,
        ``k_num_chunks = padded_Sk / k_chunk_size``) and the reader walks a block id per
        ``block_size`` of ``padded_Sk``. With ``Sk = 8256``, ``k_chunk_size = 256`` and a
        65-block page table (8320 tokens) it reads block ids 65 and 66 — past the row's 65
        valid entries, inside the page-table stick's 32-byte alignment padding. Those bytes
        are zero when the page table came from ``ttnn.from_torch`` (which writes the whole
        aligned page), so the overrun is invisible in isolation; when the page table is a
        device tensor produced by ``ttnn.slice`` + ``ttnn.concat`` — the ``_rows_page_table``
        gather any non-identity slot order takes — the padding holds stale DRAM, the block
        ids are arbitrary, and the op reads K/V from outside the cache buffer. Measured on
        the tail chunk of a batched 8256-token prefill: layer-output PCC 0.7345 instead of
        0.9998. Keeping ``padded_Sk`` inside the page table avoids the overrun entirely; 32
        always qualifies because ``chunk_start_idx + chunk_len`` is tile-aligned and the op
        already requires it to fit in ``kv_length``.
        """
        q_candidates = [
            c
            for c in SDPA_CHUNK_CANDIDATES
            if c <= self.prefill_sdpa_q_chunk and chunk_len % c == 0 and chunk_start_idx % c == 0
        ]
        k_candidates = [
            c
            for c in SDPA_CHUNK_CANDIDATES
            if c <= self.prefill_sdpa_k_chunk
            and chunk_start_idx % c == 0
            and _round_up(chunk_start_idx + chunk_len, c) <= kv_length
        ]
        if not q_candidates or not k_candidates:
            raise ValueError(
                f"cannot pick SDPA chunk sizes for chunk_len={chunk_len}, "
                f"chunk_start_idx={chunk_start_idx}, kv_length={kv_length}"
            )
        return max(q_candidates), max(k_candidates)

    def _rows_page_table(self, page_table, rows):
        """Page table restricted to ``rows``, in input-batch order.

        The chunked and decode SDPA ops require ``page_table.shape[0] == q.shape[0]``, so a
        shared slot pool (more rows than the batch) or a non-identity slot order has to be
        gathered first. Returns ``(tensor, owned)``; ``owned`` tells the caller to free it.
        """
        if rows == list(range(page_table.shape[0])):
            return page_table, False
        blocks = page_table.shape[1]
        row_slices = [self._slice_view(page_table, [row, 0], [row + 1, blocks]) for row in rows]
        if len(row_slices) == 1:
            return row_slices[0]
        gathered = ttnn.concat([tensor for tensor, _ in row_slices], dim=0)
        for tensor, owned in row_slices:
            if owned:
                tensor.deallocate(True)
        return gathered, True

    def _chunked_prefill_sdpa(self, q, *, kv_cache, page_table, rows, chunk_start_idx: int, chunk_len: int):
        """Causal SDPA for one Q chunk against the full paged prefix (full-attention layers)."""
        k_cache, v_cache = kv_cache
        kv_length = page_table.shape[1] * self.block_size
        q_chunk, k_chunk = self._chunked_sdpa_chunk_sizes(chunk_len, chunk_start_idx, kv_length)
        program_config = self._sdpa_program_config(q_chunk=q_chunk, k_chunk=k_chunk)
        sdpa_page_table, owned = self._rows_page_table(page_table, rows)

        # No `scale=`: the op's nanobind binding marks scale `.noconvert()` on a
        # std::optional<float>, so a Python float raises "incompatible function
        # arguments". Its default is 1 / sqrt(head_dim) == cfg.sdpa_scale, and Q already
        # carries qk_scale_factor, so the omission is exact rather than a compromise.
        out = ttnn.transformer.chunked_scaled_dot_product_attention(
            q,
            k_cache,
            v_cache,
            sdpa_page_table,
            chunk_start_idx,
            program_config=program_config,
            compute_kernel_config=self.sdpa_kernel_config,
        )
        if owned:
            sdpa_page_table.deallocate(True)
        return out

    # ----------------------------------------------------------------- decode

    def decode_forward(self, hidden_states, *, kv_cache, page_table, current_pos, rope_idxs=None):
        """Single-token decode for ``batch`` users. See the module docstring.

        ``hidden_states``: ``[1, 1, batch, hidden_size]``. ``page_table``: int32
        ``[batch, blocks_per_seq]`` — unlike prefill, decode requires exactly one row per
        active user, in batch order (both ``paged_update_cache`` and
        ``paged_scaled_dot_product_attention_decode`` index it by batch row).
        ``current_pos``: int32 ``[batch]`` device tensor of absolute positions.
        ``rope_idxs``: uint32 ``[1, batch]`` device tensor with the same positions, used
        for the on-device cos/sin gather (required for RoPE layers).
        """
        cfg = self.config
        batch = hidden_states.shape[-2]
        k_cache, v_cache = kv_cache
        if page_table.shape[0] != batch:
            raise ValueError(
                f"decode needs one page-table row per user: got {page_table.shape[0]} rows for batch {batch}"
            )
        if current_pos.shape[-1] != batch:
            raise ValueError(f"current_pos has {current_pos.shape[-1]} entries for batch {batch}")
        self._check_decode_page_table_capacity(page_table.shape[1] * self.block_size)

        xn = self._norm(hidden_states, self.input_norm_w, cfg.rms_norm_eps)
        xqkv = ttnn.linear(xn, self.wqkv, compute_kernel_config=self.hifi2_kernel_config)

        # nlp_create_qkv_heads_decode's interleaved DRAM reader zeroes odd Q rows on
        # Blackhole (tt-metal #16667); the L1 path is unaffected.
        if xqkv.memory_config().buffer_type == ttnn.BufferType.DRAM:
            xqkv_l1 = ttnn.to_memory_config(xqkv, ttnn.L1_MEMORY_CONFIG)
            xqkv.deallocate(True)
            xqkv = xqkv_l1
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )
        xqkv.deallocate(True)

        q = self._norm_sharded(q, self._qk_norm)
        k = self._norm_sharded(k, self._qk_norm)
        # HF scales Q by qk_scale_factor after the QK norm and before RoPE.
        q_scaled = ttnn.multiply(q, cfg.qk_scale_factor, memory_config=q.memory_config())
        q.deallocate(True)
        q = q_scaled

        if cfg.uses_rope:
            if rope_idxs is None:
                raise ValueError("rope_idxs is required for a RoPE (sliding_attention) layer")
            cos, sin = self._rope_decode_mats(rope_idxs, batch=batch, like=q)
            q_rot = ttnn.experimental.rotary_embedding_hf(q, cos, sin, is_decode_mode=True)
            k_rot = ttnn.experimental.rotary_embedding_hf(k, cos, sin, is_decode_mode=True)
            q.deallocate(True)
            k.deallocate(True)
            cos.deallocate(True)
            sin.deallocate(True)
            q, k = q_rot, k_rot

        ttnn.experimental.paged_update_cache(
            k_cache,
            k,
            update_idxs_tensor=current_pos,
            page_table=page_table,
            block_size=self.block_size,
            num_kv_heads=cfg.num_key_value_heads,
        )
        ttnn.experimental.paged_update_cache(
            v_cache,
            v,
            update_idxs_tensor=current_pos,
            page_table=page_table,
            block_size=self.block_size,
            num_kv_heads=cfg.num_key_value_heads,
        )
        k.deallocate(True)
        v.deallocate(True)

        attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            k_cache,
            v_cache,
            cur_pos_tensor=current_pos,
            page_table_tensor=page_table,
            scale=cfg.sdpa_scale,
            sliding_window_size=cfg.sliding_window,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self._decode_sdpa_program_config(batch),
            compute_kernel_config=self.sdpa_kernel_config,
            block_size=self.block_size,
            num_kv_heads=cfg.num_key_value_heads,
        )
        q.deallocate(True)

        attn_cat = self._concat_heads_decode(attn, batch)
        attn.deallocate(True)

        gate = ttnn.linear(xn, self.w_attn_gate, compute_kernel_config=self.hifi2_kernel_config)
        gate_sigmoid = ttnn.sigmoid(gate)
        gate.deallocate(True)
        attn_cat = ttnn.multiply(attn_cat, gate_sigmoid)
        gate_sigmoid.deallocate(True)
        xn.deallocate(True)

        attn_out = ttnn.linear(attn_cat, self.wo, compute_kernel_config=self.hifi2_kernel_config)
        attn_cat.deallocate(True)
        attn_out = self._norm(attn_out, self.post_attn_norm_w, cfg.post_norm_eps)
        h = ttnn.add(hidden_states, attn_out)
        attn_out.deallocate(True)

        y = self._norm(h, self.pre_ff_norm_w, cfg.rms_norm_eps)
        y = self._mlp(y)
        y = self._norm(y, self.post_ff_norm_w, cfg.post_norm_eps)
        out = ttnn.add(h, y)
        h.deallocate(True)
        y.deallocate(True)
        return out

    def _norm_sharded(self, x, norm_fn):
        """Run a norm on a height-sharded decode tensor, restoring the shard config."""
        mem_config = x.memory_config()
        x_int = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        x.deallocate(True)
        normed = norm_fn(x_int)
        x_int.deallocate(True)
        out = ttnn.to_memory_config(normed, mem_config)
        normed.deallocate(True)
        return out

    def _rope_decode_mats(self, rope_idxs, *, batch: int, like):
        """Gather per-user cos/sin rows on device and shard them exactly like ``like``.

        The sharded ``rotary_embedding_hf`` decode kernel pairs shard *i* of the cos/sin
        caches with shard *i* of Q, so the two must use the *same* core layout.
        ``nlp_create_qkv_heads_decode`` picks its own grid, and for batch 32 on this
        Blackhole grid that is the non-rectangular ``11 + 11 + 10`` row-major set — laying
        cos/sin out on any other grid (e.g. a tidy 8x4 rectangle) silently rotates each
        user by another user's position and costs ~0.09 PCC. So derive the layout from Q
        instead of constructing one.
        """
        cfg = self.config
        cache = self.rope_cache
        cos = ttnn.embedding(rope_idxs, cache["cos_decode"], layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(rope_idxs, cache["sin_decode"], layout=ttnn.TILE_LAYOUT)
        cos = ttnn.transpose(ttnn.unsqueeze_to_4D(cos), 1, 2)  # [1, batch, 1(->32), head_dim]
        sin = ttnn.transpose(ttnn.unsqueeze_to_4D(sin), 1, 2)
        if cos.shape[1] != batch:
            cos = cos[:, :batch, :, :]
            sin = sin[:, :batch, :, :]
        shard_spec = like.memory_config().shard_spec
        shard_cfg = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, cfg.head_dim),
            core_grid=shard_spec.grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=shard_spec.orientation,
            use_height_and_width_as_shard_shape=True,
        )
        cos_sh = ttnn.to_memory_config(cos, shard_cfg)
        sin_sh = ttnn.to_memory_config(sin, shard_cfg)
        cos.deallocate(True)
        sin.deallocate(True)
        return cos_sh, sin_sh

    def _decode_core_range_set(self, batch: int):
        """One core per user, arranged as a single rectangle.

        ``ttnn.num_cores_to_corerangeset`` fills row-major and produces a *non-rectangular*
        set as soon as ``batch`` is not a multiple of the grid width — on this Blackhole
        grid (11x10) batch 32 becomes 11+11+10, and ``nlp_concat_heads_decode`` then fails
        with "bad optional access". Pick a rectangle whose width divides ``batch`` instead
        (batch 32 -> 8x4), the same shape the Gemma4 decode concat uses.
        """
        grid_x, grid_y = self.device_grid
        if batch > grid_x * grid_y:
            raise ValueError(f"batch {batch} exceeds the {grid_x}x{grid_y} core grid")
        width = min(batch, grid_x)
        if batch % width != 0 or batch // width > grid_y:
            candidates = [c for c in range(width, 0, -1) if batch % c == 0 and batch // c <= grid_y]
            if not candidates:
                raise ValueError(f"cannot fit batch {batch} in a rectangle on a {grid_x}x{grid_y} grid")
            width = candidates[0]
        return ttnn.CoreRangeSet({num_to_corerange(batch, grid_x=width, grid_y=grid_y)})

    def _decode_batch_shard_config(self, batch: int, width: int):
        """Height-sharded config with one user per core, matching the Q/K decode shards."""
        return ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, width),
            core_grid=self._decode_core_range_set(batch),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def blocks_per_seq(self, max_seq_len: int) -> int:
        """Page-table width (blocks per sequence) that ``max_seq_len`` tokens need.

        Rounded up so the resulting capacity is a whole number of decode K chunks — see
        :meth:`_check_decode_page_table_capacity` for why that matters. Callers that build
        their own page tables should size them with this.
        """
        blocks = _round_up(max_seq_len, self.block_size) // self.block_size
        capacity_step = _lcm(self.block_size, self.decode_sdpa_k_chunk)
        return _round_up(blocks * self.block_size, capacity_step) // self.block_size

    def _check_decode_page_table_capacity(self, kv_length: int) -> None:
        """The decode SDPA rounds its K extent up; the page table must cover the rounded value.

        ``paged_scaled_dot_product_attention_decode`` has the same unvalidated rounding as the
        chunked prefill op (see :meth:`_chunked_sdpa_chunk_sizes`), on a different axis:
        ``rt_args_common.hpp`` computes ``valid_seq_len = nearest_n(cur_pos + 1,
        k_chunk_size)`` (and ``window_end_aligned`` likewise for the sliding branch), and
        ``reader_decode_all.cpp`` walks that *rounded* extent, resolving every tile row through
        the unbounded ``page_table_ptr[virtual_block]`` in ``dataflow_common.hpp``. ``cur_pos``
        is a device tensor, so nothing host-side bounds it.

        A page-table capacity that is a whole number of K chunks makes the rounding safe for
        *every* addressable position: ``nearest_n(cur_pos + 1, k) <= capacity`` for all
        ``cur_pos < capacity``. Without it, e.g. a 25-block page table at ``block_size=32``
        (capacity 800) decoding at ``cur_pos=777`` with ``k_chunk=64`` rounds to 832 and reads
        block-id entry 25 of a 25-entry row — the page-table stick's alignment padding — then
        reads K/V from wherever that integer points. Those tiles are past ``cur_pos`` so the
        mask discards their *values*; the problem is the out-of-bounds read itself, which on
        the prefill side was measured both as garbage output and as a device hang.

        :meth:`blocks_per_seq` sizes page tables so this holds; the check is here because the
        page table is the caller's tensor.
        """
        if kv_length % self.decode_sdpa_k_chunk == 0:
            return
        needed = _round_up(kv_length, _lcm(self.block_size, self.decode_sdpa_k_chunk))
        raise ValueError(
            f"decode page-table capacity {kv_length} tokens "
            f"({kv_length // self.block_size} blocks x block_size {self.block_size}) is not a "
            f"whole number of decode K chunks ({self.decode_sdpa_k_chunk}); the SDPA decode op "
            "rounds its K extent up to the K chunk and reads the page table past its last "
            f"entry. Allocate {needed // self.block_size} blocks per sequence instead "
            "(FunctionalDecoder.blocks_per_seq does this)."
        )

    def _decode_sdpa_program_config(self, batch: int):
        """Decode SDPA grid, which depends on the batch.

        An explicit program config is mandatory on Blackhole: with ``program_config=None``
        the op spreads over the whole >=110-core grid, and with only 2 KV heads that
        exceeds the 64-cores-per-head flash-decode reduction-tree limit
        (MAX_TREE_REDUCTION_ROUNDS = 6). The struct's ``max_cores_per_head_batch``
        default (16) caps the tree.

        The grid must also give **every (user, KV head) pair its own core**. In
        ``sdpa_decode_program_factory.cpp`` a grid with
        ``cores < batch * num_kv_heads`` still passes validation
        (only ``cores >= batch`` is checked) and silently folds both KV heads onto one core
        (``num_heads_per_core = 2``), which returns wrong results: batch 32 on the
        sweep-optimal 8x4 grid measured PCC **0.7176** instead of 0.9999. So the measured
        grid (`doc/functional_decoder/perf/core_grid_sweep.md`, fastest at batch 1) is used
        only while it has enough cores, and the full compute grid is used beyond that.
        """
        cfg = self.config
        min_cores = batch * cfg.num_key_value_heads
        grid_x, grid_y = self.device_grid
        grid = self.decode_sdpa_core_grid
        if grid[0] * grid[1] < min_cores:
            grid = (grid_x, grid_y)
        if grid[0] * grid[1] < min_cores:
            raise ValueError(
                f"decode batch {batch} needs {min_cores} cores (batch * num_key_value_heads) "
                f"but the compute grid only has {grid_x * grid_y}"
            )
        return self._sdpa_program_config(q_chunk=TILE, k_chunk=self.decode_sdpa_k_chunk, grid=grid)

    def _concat_heads_decode(self, attn, batch: int):
        cfg = self.config
        shard_cfg = self._decode_batch_shard_config(batch, cfg.head_dim)
        attn_sh = ttnn.to_memory_config(attn, shard_cfg)
        out_sh = ttnn.experimental.nlp_concat_heads_decode(attn_sh, num_heads=cfg.num_attention_heads)
        attn_sh.deallocate(True)
        out = ttnn.sharded_to_interleaved(out_sh, ttnn.DRAM_MEMORY_CONFIG)
        out_sh.deallocate(True)
        if out.shape[2] != batch:  # the op pads the batch dim up to a tile
            trimmed = out[:, :, :batch, :]
            out.deallocate(True)
            out = trimmed
        return out

    # ------------------------------------------------------------------ hosts

    def forward(self, *args, mode: str = "decode", **kwargs):
        """Dispatch to :meth:`prefill_forward` / :meth:`decode_forward`."""
        if mode == "prefill":
            return self.prefill_forward(*args, **kwargs)
        if mode == "decode":
            return self.decode_forward(*args, **kwargs)
        raise ValueError(f"unknown mode {mode!r}")
