# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN [`SeamlessM4Tv2TextToUnitForConditionalGeneration`]: encoder + decoder + ``lm_head``.

**Parity target (Hugging Face):** ``SeamlessM4Tv2TextToUnitForConditionalGeneration`` — same dataflow as
``modeling_seamless_m4t_v2``: encoder on ``inputs_embeds``; decoder char upsample, duration predictor,
unit upsample, self-attn + conv decoder stack; ``lm_head`` logits in the field HF names
``last_hidden_state``.

**Implementation policy:** All math in this file runs through **ttnn** (no ``torch`` / no ``numpy`` /
no Transformers helpers). Host-side control flow uses Python ``int`` / ``Sequence[int]`` for repeat
counts and sequence lengths. Small host readbacks use ``ttnn.from_device`` plus ``ttnn.to_torch`` as a
host transport only — torch is not used for any math — to unpack float32 values into integer repeat
counts for ``ttnn.repeat_interleave``, matching HF ``round(expm1(...))`` with clamp. I/O tensors stay
on device except for that readback.

**Whole-model compatibility:** Callers should pass ``char_count_per_id`` as a length-``enc_seq`` list of
non-negative integers (batch size 1), matching HF ``char_count_per_id.sum(-1)`` semantics; build the
encoder 4D additive mask with the same helpers used for the text encoder PCC tests.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence, Tuple

import torch
import ttnn

from models.common.utility_functions import nearest_32

# HF ``torch.finfo(torch.bfloat16).min`` additive padding mask floor (approx.).
_BF16_MASK_FLOOR = -3.3895313892565356e38


def _row_major_host_f32_flat(host_tensor: ttnn.Tensor, *, num_floats: int) -> list[float]:
    """Flat list of float32 values from a host-side ttnn tensor.

    Uses ``ttnn.to_torch`` for the host readback because the ``HostBuffer`` iterator yields
    ``std::byte``, which pybind does not auto-convert (``bytes(shard)`` raises TypeError). The
    decoded values feed integer repeat counts for ``ttnn.repeat_interleave`` — torch is only used
    as a host transport, no math runs through it.
    """
    flat = ttnn.to_torch(host_tensor).to(torch.float32).reshape(-1)
    n = int(num_floats)
    if int(flat.numel()) < n:
        raise RuntimeError(f"Host tensor has {int(flat.numel())} floats; need {n}.")
    return flat[:n].tolist()


def _core_grid(device: ttnn.Device) -> ttnn.CoreGrid:
    grid = device.compute_with_storage_grid_size()
    return ttnn.CoreGrid(y=grid.y, x=grid.x)


def _mask_row_valid_prefix(device: ttnn.Device, width: int, valid_len: int) -> ttnn.Tensor:
    """``[1, width]`` bfloat16 tile: 1 where column index < ``valid_len``, else 0 (batch 1)."""
    idx_rm = ttnn.arange(
        0,
        width,
        step=1,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    idx = ttnn.reshape(idx_rm, (1, width))
    idx_t = ttnn.to_layout(idx, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    lim = ttnn.full(
        (1, width),
        float(valid_len),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ones = ttnn.full(
        (1, width),
        1.0,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    zeros = ttnn.full(
        (1, width),
        0.0,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out = ttnn.where(ttnn.lt(idx_t, lim), ones, zeros, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return out


def _expand_4d_padding_additive_b1(
    device: ttnn.Device, mask_2d_tile: ttnn.Tensor, seq_len: int, width: int
) -> ttnn.Tensor:
    """
    HF ``AttentionMaskConverter._expand_mask`` for a 2D padding mask (1 = keep), batch 1.

    Returns additive mask ``[1, 1, seq_len, width]`` (tile bf16): 0 keep, ``_BF16_MASK_FLOOR`` masked.
    """
    m = ttnn.to_layout(mask_2d_tile, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    row = ttnn.reshape(m, (1, 1, 1, width))
    expanded = ttnn.repeat_interleave(row, seq_len, dim=2)
    ones = ttnn.full(
        (1, 1, seq_len, width),
        1.0,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    inverted = ttnn.add(
        ones,
        ttnn.multiply(expanded, -1.0, memory_config=ttnn.DRAM_MEMORY_CONFIG),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    floor = ttnn.full(
        (1, 1, seq_len, width),
        _BF16_MASK_FLOOR,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    zeros = ttnn.full(
        (1, 1, seq_len, width),
        0.0,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out = ttnn.where(ttnn.gt(inverted, 0.5), floor, zeros, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return out


def _hard_upsample_nlc(
    enc: ttnn.Tensor, repeats: Sequence[int], *, device: ttnn.Device, hidden_size: int
) -> ttnn.Tensor:
    """HF ``_hard_upsample`` for batch 1: ``enc`` is ``[1, T, H]`` tile; ``repeats`` length ``T``."""
    parts: list[ttnn.Tensor] = []
    enc_seq = len(repeats)
    for t, r in enumerate(repeats):
        r = int(r)
        if r <= 0:
            continue
        row = ttnn.slice(enc, [0, t, 0], [1, t + 1, hidden_size])
        row = ttnn.reshape(row, (1, 1, hidden_size))
        reped = ttnn.repeat_interleave(row, r, dim=1)
        parts.append(reped)
    if not parts:
        raise ValueError("_hard_upsample_nlc: empty output (all repeat counts zero).")
    out = parts[0]
    for p in parts[1:]:
        out = ttnn.concat([out, p], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return out


def _discrete_duration_counts(log_dur: ttnn.Tensor, *, batch: int, seq: int) -> list[int]:
    """HF-exact ``clamp(round(expm1(log_dur)), min=1).long()`` per position.

    Caller (``_duration_predictor``) produces ``log_dur`` already in float32 — no bf16→fp32 typecast
    is needed, and the bf16 write-out that previously perturbed ``log_dur`` is gone.

    Round mode: ``ttnn.round`` uses the SFPU ``_round_even_`` op (round-half-to-even, banker's
    rounding) — bit-identical to ``torch.round`` semantics, so the on-device pipeline below is
    HF-equivalent ``expm1`` → banker's-round → clamp. The host loop only converts fp32 → Python int
    and applies a defensive ``max(1, ...)`` (no further rounding); we use ``int(round(...))`` rather
    than ``int(...)`` because ``int()`` truncates toward zero and could surprise on negative values
    (which clamp rules out anyway, but the explicit ``round`` is cheap insurance).
    """
    ld = ttnn.reshape(log_dur, (int(batch), int(seq)))
    if ld.dtype != ttnn.float32:
        ld = ttnn.typecast(ld, ttnn.float32)
    x = ttnn.expm1(ld, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x = ttnn.round(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x = ttnn.clamp(x, min=1.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    host_x = ttnn.from_device(x_rm)
    ttnn.deallocate(x_rm)
    ttnn.deallocate(x)
    x_h = _row_major_host_f32_flat(host_x, num_floats=int(batch) * int(seq))
    return [max(1, int(round(v))) for v in x_h]


def _conv1d_same(
    device: ttnn.Device,
    x_tile: ttnn.Tensor,
    *,
    sequence_length: int,
    weight_rm: ttnn.Tensor,
    bias_rm: Optional[ttnn.Tensor],
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    padding: int,
    compute_kernel_config: ttnn.DeviceComputeKernelConfig,
    activation: Optional[str] = None,
) -> ttnn.Tensor:
    """Same-padding Conv1d stride 1 via ``ttnn.conv1d`` (activations ``[B,S,C]`` NLC).

    When ``activation`` is provided (e.g. ``"relu"``) it is fused into the
    conv kernel via ``Conv1dConfig.activation``; the SFPU applies it at the
    writeback stage of each output tile so no separate UnaryDeviceOperation
    is dispatched.  Numerically equivalent to a post-conv ``ttnn.relu``.
    """
    batch = int(x_tile.shape[0])
    seq = int(sequence_length)
    x_rm = ttnn.to_layout(x_tile, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    w_dev = ttnn.to_device(weight_rm, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    bias_dev = ttnn.to_device(bias_rm, device, memory_config=ttnn.DRAM_MEMORY_CONFIG) if bias_rm is not None else None

    conv_kwargs = dict(
        weights_dtype=ttnn.bfloat16,
        shard_layout=None,
        deallocate_activation=True,
    )
    if activation:
        # ``Conv1dConfig.activation`` is bound as ``UnaryWithParam`` in Python;
        # the friendly ``"relu"`` string is mapped here so call sites remain
        # compact.  Extend the mapping if/when other fused activations are needed.
        _ACTIVATION_OP_TYPES = {
            "relu": ttnn.UnaryOpType.RELU,
            "silu": ttnn.UnaryOpType.SILU,
            "gelu": ttnn.UnaryOpType.GELU,
        }
        op_type = _ACTIVATION_OP_TYPES.get(activation.lower())
        if op_type is None:
            raise ValueError(f"_conv1d_same: unsupported activation {activation!r}")
        conv_kwargs["activation"] = ttnn.UnaryWithParam(op_type)
    if seq > 64 or in_channels >= 512:
        conv_kwargs["act_block_h_override"] = 32
    conv_config = ttnn.Conv1dConfig(**conv_kwargs)
    out_tt, _out_len = ttnn.conv1d(
        input_tensor=x_rm,
        weight_tensor=w_dev,
        in_channels=in_channels,
        out_channels=out_channels,
        device=device,
        bias_tensor=bias_dev,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
        batch_size=batch,
        input_length=seq,
        conv_config=conv_config,
        compute_config=compute_kernel_config,
        groups=1,
        dtype=ttnn.bfloat16,
        return_output_dim=True,
    )
    out_tile = ttnn.to_layout(out_tt, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return out_tile


class TTSeamlessM4Tv2TextToUnitEncoder:
    """
    Encoder stack inside HF ``SeamlessM4Tv2TextToUnitForConditionalGeneration.model`` —
    ``SeamlessM4Tv2Encoder(..., is_t2u_encoder=True)``: ``inputs_embeds`` only, then transformer + ``layer_norm``.
    """

    def __init__(
        self,
        device: ttnn.Device,
        parameters,
        *,
        layer_norm_eps: float,
        num_hidden_layers: int,
        num_attention_heads: int,
        hidden_size: int,
    ):
        self.device = device
        self.parameters = parameters
        self.layer_norm_eps = layer_norm_eps
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self._sdpa_compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self._linear_ln_compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _sdpa_program_config(self, seq_q: int, seq_k: int) -> ttnn.SDPAProgramConfig:
        q_chunk = max(64, min(256, nearest_32(seq_q)))
        k_chunk = max(64, min(256, nearest_32(seq_k)))
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
            q_chunk_size=q_chunk,
            k_chunk_size=k_chunk,
            exp_approx_mode=False,
        )

    def _linear(
        self,
        x: ttnn.Tensor,
        weight: ttnn.Tensor,
        bias: ttnn.Tensor,
        *,
        activation: str | None = None,
    ) -> ttnn.Tensor:
        return ttnn.linear(
            x,
            weight,
            bias=bias,
            activation=activation,
            core_grid=_core_grid(self.device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._linear_ln_compute_cfg,
        )

    def _layer_norm(self, x: ttnn.Tensor, *, weight: ttnn.Tensor, bias: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.layer_norm(
            x,
            weight=weight,
            bias=bias,
            epsilon=self.layer_norm_eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._linear_ln_compute_cfg,
        )

    @staticmethod
    def _heads(x: ttnn.Tensor, batch: int, seq: int, num_heads: int, head_dim: int) -> ttnn.Tensor:
        x = ttnn.reshape(x, (batch, seq, num_heads, head_dim))
        return ttnn.permute(x, (0, 2, 1, 3))

    @staticmethod
    def _merge_heads(
        x: ttnn.Tensor, batch: int, seq: int, num_heads: int, head_dim: int, hidden_size: int
    ) -> ttnn.Tensor:
        x = ttnn.permute(x, (0, 2, 1, 3))
        return ttnn.reshape(x, (batch, seq, hidden_size))

    def _self_attention(
        self,
        hidden_states: ttnn.Tensor,
        attn_module,
        attn_mask: ttnn.Tensor,
        *,
        batch: int,
        seq: int,
        num_heads: int,
        head_dim: int,
        hidden_size: int,
        sdpa_cfg: ttnn.SDPAProgramConfig,
    ) -> ttnn.Tensor:
        q = self._linear(hidden_states, attn_module.q_proj.weight, attn_module.q_proj.bias)
        k = self._linear(hidden_states, attn_module.k_proj.weight, attn_module.k_proj.bias)
        v = self._linear(hidden_states, attn_module.v_proj.weight, attn_module.v_proj.bias)

        qh = self._heads(q, batch, seq, num_heads, head_dim)
        kh = self._heads(k, batch, seq, num_heads, head_dim)
        vh = self._heads(v, batch, seq, num_heads, head_dim)

        qh = ttnn.to_memory_config(qh, ttnn.DRAM_MEMORY_CONFIG)
        kh = ttnn.to_memory_config(kh, ttnn.DRAM_MEMORY_CONFIG)
        vh = ttnn.to_memory_config(vh, ttnn.DRAM_MEMORY_CONFIG)

        qh = ttnn.multiply(qh, 1.0 / math.sqrt(head_dim), memory_config=ttnn.DRAM_MEMORY_CONFIG)

        attn_out = ttnn.transformer.scaled_dot_product_attention(
            qh,
            kh,
            vh,
            attn_mask=attn_mask,
            is_causal=False,
            scale=1.0,
            program_config=sdpa_cfg,
            compute_kernel_config=self._sdpa_compute_cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(qh)
        ttnn.deallocate(kh)
        ttnn.deallocate(vh)

        attn_out = ttnn.slice(
            attn_out,
            [0, 0, 0, 0],
            [batch, num_heads, seq, head_dim],
            [1, 1, 1, 1],
        )

        merged = self._merge_heads(attn_out, batch, seq, num_heads, head_dim, hidden_size)
        proj = self._linear(merged, attn_module.out_proj.weight, attn_module.out_proj.bias)
        ttnn.deallocate(merged)
        ttnn.deallocate(attn_out)
        return proj

    def forward(self, inputs_embeds: ttnn.Tensor, attention_mask_4d: ttnn.Tensor) -> ttnn.Tensor:
        parameters = self.parameters
        num_heads = self.num_attention_heads
        hidden_size = self.hidden_size
        head_dim = hidden_size // num_heads
        num_layers = self.num_hidden_layers

        batch = int(inputs_embeds.shape[0])
        seq = int(inputs_embeds.shape[1])

        hidden = inputs_embeds
        sdpa_cfg = self._sdpa_program_config(seq, seq)

        for i in range(num_layers):
            layer = parameters.layers[i]

            normed = self._layer_norm(
                hidden,
                weight=layer.self_attn_layer_norm.weight,
                bias=layer.self_attn_layer_norm.bias,
            )
            attn_out = self._self_attention(
                normed,
                layer.self_attn,
                attention_mask_4d,
                batch=batch,
                seq=seq,
                num_heads=num_heads,
                head_dim=head_dim,
                hidden_size=hidden_size,
                sdpa_cfg=sdpa_cfg,
            )
            ttnn.deallocate(normed)
            hidden = ttnn.add(hidden, attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(attn_out)

            normed = self._layer_norm(
                hidden,
                weight=layer.ffn_layer_norm.weight,
                bias=layer.ffn_layer_norm.bias,
            )
            ff = self._linear(
                normed,
                layer.ffn.fc1.weight,
                layer.ffn.fc1.bias,
                activation="relu",
            )
            ttnn.deallocate(normed)
            ff = self._linear(ff, layer.ffn.fc2.weight, layer.ffn.fc2.bias)
            hidden = ttnn.add(hidden, ff, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(ff)

        hidden = self._layer_norm(
            hidden,
            weight=parameters.layer_norm.weight,
            bias=parameters.layer_norm.bias,
        )
        return hidden


class TTSeamlessM4Tv2TextToUnitForConditionalGeneration:
    """
    TTNN port of HF ``SeamlessM4Tv2TextToUnitForConditionalGeneration`` (encoder + decoder + ``lm_head``).

    Conv1d blocks use ``ttnn.conv1d`` (same padding, stride 1). If a deployment hits L1 limits, tune
    ``Conv1dConfig.act_block_h_override`` inside ``_conv1d_same``.
    """

    def __init__(
        self,
        device: ttnn.Device,
        parameters: Any,
        *,
        layer_norm_eps: float,
        encoder_layers: int,
        encoder_attention_heads: int,
        decoder_layers: int,
        decoder_attention_heads: int,
        hidden_size: int,
        pad_token_id: int,
        variance_predictor_embed_dim: int,
        variance_predictor_hidden_dim: int,
        variance_predictor_kernel_size: int,
    ):
        self.device = device
        self.parameters = parameters
        self.layer_norm_eps = layer_norm_eps
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.hidden_size = hidden_size
        self.pad_token_id = pad_token_id
        self.variance_predictor_embed_dim = variance_predictor_embed_dim
        self.variance_predictor_hidden_dim = variance_predictor_hidden_dim
        self.variance_predictor_kernel_size = variance_predictor_kernel_size

        self.encoder = TTSeamlessM4Tv2TextToUnitEncoder(
            device,
            parameters.encoder,
            layer_norm_eps=layer_norm_eps,
            num_hidden_layers=encoder_layers,
            num_attention_heads=encoder_attention_heads,
            hidden_size=hidden_size,
        )
        self._sdpa_compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self._linear_ln_compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self._conv_compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # Pre-cast the duration-predictor's final projection (proj) weight/bias to float32.
        # The proj produces the scalar ``log_dur`` that feeds into ``clamp(round(expm1(...)), min=1)``;
        # the rounding boundary at .5 (banker's-rounded, matching HF) means any bf16-level noise in
        # ``log_dur`` flips ``t_audio`` for some characters, which shifts the entire downstream
        # waveform. Running the proj in fp32 (fp32 weights, fp32 dest accumulator, fp32 output) keeps
        # ``log_dur`` at fp32 precision so the round() decision matches HF for ~all characters.
        # The cast is done once at init — the per-call cost is zero, only memory is +4× for this tiny
        # weight (hidden_dim → 1, ~256 floats).
        p_dp = parameters.decoder.duration_predictor
        self._dp_proj_w_fp32 = ttnn.typecast(p_dp.proj.weight, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self._dp_proj_b_fp32 = (
            ttnn.typecast(p_dp.proj.bias, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if getattr(p_dp.proj, "bias", None) is not None
            else None
        )

    def _sdpa_program_config(self, seq_q: int, seq_k: int) -> ttnn.SDPAProgramConfig:
        m = max(seq_q, seq_k)
        if m > 96:
            cap = 32
        elif m > 64:
            cap = 64
        elif m > 32:
            cap = 128
        else:
            cap = 256
        q_chunk = max(32, min(cap, nearest_32(seq_q)))
        k_chunk = max(32, min(cap, nearest_32(seq_k)))
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
            q_chunk_size=q_chunk,
            k_chunk_size=k_chunk,
            exp_approx_mode=False,
        )

    def _linear(
        self,
        x: ttnn.Tensor,
        weight: ttnn.Tensor,
        bias: ttnn.Tensor,
        *,
        activation: str | None = None,
    ) -> ttnn.Tensor:
        return ttnn.linear(
            x,
            weight,
            bias=bias,
            activation=activation,
            core_grid=_core_grid(self.device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._linear_ln_compute_cfg,
        )

    def _layer_norm(self, x: ttnn.Tensor, *, weight: ttnn.Tensor, bias: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.layer_norm(
            x,
            weight=weight,
            bias=bias,
            epsilon=self.layer_norm_eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._linear_ln_compute_cfg,
        )

    @staticmethod
    def _heads(x: ttnn.Tensor, batch: int, seq: int, num_heads: int, head_dim: int) -> ttnn.Tensor:
        x = ttnn.reshape(x, (batch, seq, num_heads, head_dim))
        return ttnn.permute(x, (0, 2, 1, 3))

    @staticmethod
    def _merge_heads(
        x: ttnn.Tensor, batch: int, seq: int, num_heads: int, head_dim: int, hidden_size: int
    ) -> ttnn.Tensor:
        x = ttnn.permute(x, (0, 2, 1, 3))
        return ttnn.reshape(x, (batch, seq, hidden_size))

    def _decoder_self_attention(
        self,
        hidden_states: ttnn.Tensor,
        attn_module: Any,
        attn_mask: ttnn.Tensor,
        *,
        batch: int,
        seq: int,
        num_heads: int,
        head_dim: int,
        hidden_size: int,
        sdpa_cfg: ttnn.SDPAProgramConfig,
    ) -> ttnn.Tensor:
        q = self._linear(hidden_states, attn_module.q_proj.weight, attn_module.q_proj.bias)
        k = self._linear(hidden_states, attn_module.k_proj.weight, attn_module.k_proj.bias)
        v = self._linear(hidden_states, attn_module.v_proj.weight, attn_module.v_proj.bias)

        qh = self._heads(q, batch, seq, num_heads, head_dim)
        kh = self._heads(k, batch, seq, num_heads, head_dim)
        vh = self._heads(v, batch, seq, num_heads, head_dim)

        qh = ttnn.to_memory_config(qh, ttnn.DRAM_MEMORY_CONFIG)
        kh = ttnn.to_memory_config(kh, ttnn.DRAM_MEMORY_CONFIG)
        vh = ttnn.to_memory_config(vh, ttnn.DRAM_MEMORY_CONFIG)

        qh = ttnn.multiply(qh, 1.0 / math.sqrt(head_dim), memory_config=ttnn.DRAM_MEMORY_CONFIG)

        attn_out = ttnn.transformer.scaled_dot_product_attention(
            qh,
            kh,
            vh,
            attn_mask=attn_mask,
            is_causal=False,
            scale=1.0,
            program_config=sdpa_cfg,
            compute_kernel_config=self._sdpa_compute_cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(qh)
        ttnn.deallocate(kh)
        ttnn.deallocate(vh)

        attn_out = ttnn.slice(
            attn_out,
            [0, 0, 0, 0],
            [batch, num_heads, seq, head_dim],
            [1, 1, 1, 1],
        )

        merged = self._merge_heads(attn_out, batch, seq, num_heads, head_dim, hidden_size)
        proj = self._linear(merged, attn_module.out_proj.weight, attn_module.out_proj.bias)
        ttnn.deallocate(merged)
        ttnn.deallocate(attn_out)
        return proj

    def _duration_predictor(
        self, char_hidden: ttnn.Tensor, char_padding_mask_tt: ttnn.Tensor, *, seq: int
    ) -> ttnn.Tensor:
        p = self.parameters.decoder.duration_predictor
        batch = int(char_hidden.shape[0])
        mask_bc = ttnn.reshape(char_padding_mask_tt, (batch, seq, 1))
        h = ttnn.multiply(char_hidden, mask_bc, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        k = self.variance_predictor_kernel_size
        pad = k // 2
        h = _conv1d_same(
            self.device,
            h,
            sequence_length=seq,
            weight_rm=p.conv1.weight,
            bias_rm=p.conv1.bias,
            in_channels=self.variance_predictor_embed_dim,
            out_channels=self.variance_predictor_hidden_dim,
            kernel_size=k,
            padding=pad,
            compute_kernel_config=self._conv_compute_cfg,
            activation="relu",
        )
        h = self._layer_norm(h, weight=p.ln1.weight, bias=p.ln1.bias)
        h = ttnn.multiply(h, mask_bc, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        h = _conv1d_same(
            self.device,
            h,
            sequence_length=seq,
            weight_rm=p.conv2.weight,
            bias_rm=p.conv2.bias,
            in_channels=self.variance_predictor_hidden_dim,
            out_channels=self.variance_predictor_hidden_dim,
            kernel_size=k,
            padding=pad,
            compute_kernel_config=self._conv_compute_cfg,
            activation="relu",
        )
        h = self._layer_norm(h, weight=p.ln2.weight, bias=p.ln2.bias)

        # Final projection in float32: cast h (bf16) → fp32 and use the fp32-cached proj weights so
        # that ``log_dur`` keeps the fp32-accumulator precision all the way through to the
        # downstream ``expm1`` + banker's-round. Without this, the bf16 write-out of proj loses
        # ~1% precision, which flips the rounding decision near .5 boundaries and propagates a
        # one-frame ``t_audio`` shift through every downstream conv in the vocoder.
        h_fp32 = ttnn.typecast(h, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(h)
        log_dur = ttnn.linear(
            h_fp32,
            self._dp_proj_w_fp32,
            bias=self._dp_proj_b_fp32,
            core_grid=_core_grid(self.device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._linear_ln_compute_cfg,
            dtype=ttnn.float32,
        )
        ttnn.deallocate(h_fp32)
        return log_dur

    def _decoder_layer(
        self,
        hidden: ttnn.Tensor,
        attention_mask_4d: ttnn.Tensor,
        padding_mask_1d: ttnn.Tensor,
        layer: Any,
        *,
        batch: int,
        seq: int,
        num_heads: int,
        head_dim: int,
        hidden_size: int,
        sdpa_cfg: ttnn.SDPAProgramConfig,
    ) -> ttnn.Tensor:
        residual = hidden
        attn_out = self._decoder_self_attention(
            hidden,
            layer.self_attn,
            attention_mask_4d,
            batch=batch,
            seq=seq,
            num_heads=num_heads,
            head_dim=head_dim,
            hidden_size=hidden_size,
            sdpa_cfg=sdpa_cfg,
        )
        hidden = ttnn.add(residual, attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn_out)
        hidden = self._layer_norm(
            hidden,
            weight=layer.self_attn_layer_norm.weight,
            bias=layer.self_attn_layer_norm.bias,
        )

        residual = hidden
        mask_bc = ttnn.reshape(padding_mask_1d, (batch, seq, 1))
        hidden = ttnn.multiply(hidden, mask_bc, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        hidden = _conv1d_same(
            self.device,
            hidden,
            sequence_length=seq,
            weight_rm=layer.conv1.weight,
            bias_rm=layer.conv1.bias,
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=7,
            padding=3,
            compute_kernel_config=self._conv_compute_cfg,
            activation="relu",
        )
        hidden = ttnn.multiply(hidden, mask_bc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        hidden = _conv1d_same(
            self.device,
            hidden,
            sequence_length=seq,
            weight_rm=layer.conv2.weight,
            bias_rm=layer.conv2.bias,
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=7,
            padding=3,
            compute_kernel_config=self._conv_compute_cfg,
        )
        hidden = ttnn.add(residual, hidden, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        hidden = self._layer_norm(
            hidden,
            weight=layer.conv_layer_norm.weight,
            bias=layer.conv_layer_norm.bias,
        )
        return hidden

    def forward(
        self,
        inputs_embeds: ttnn.Tensor,
        encoder_attention_mask_4d: ttnn.Tensor,
        char_input_ids: ttnn.Tensor,
        char_count_per_id: Sequence[int],
        *,
        reference_discrete_durations: Optional[Sequence[int]] = None,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Args:
            inputs_embeds: ``[1, enc_seq, hidden]`` tile bf16 on device.
            encoder_attention_mask_4d: encoder self-attention additive mask ``[1, 1, enc_seq, enc_seq]``.
            char_input_ids: ``uint32`` ``[1, char_len]`` on device (padded).
            char_count_per_id: length-``enc_seq`` sequence of non-negative ints (sum = ``char_len``), batch 1.
            reference_discrete_durations: optional per-character integer durations (length ``char_len``),
                e.g. from [`hf_discrete_duration_counts_batch1`], to match HF unit length in PCC tests while
                the TTNN duration predictor is converged. When ``None``, durations come from the TT predictor.

        Returns:
            ``(lm_logits, padding_mask)`` both tile bf16 on device (``padding_mask`` is ``1`` = valid),
            matching HF logits and ``padding_mask`` semantics.
        """
        dec = self.parameters.decoder
        batch = int(inputs_embeds.shape[0])
        enc_seq = int(inputs_embeds.shape[1])
        if batch != 1:
            raise NotImplementedError("batch > 1 not supported for TT text-to-unit.")

        cc_list = [int(x) for x in char_count_per_id]
        if len(cc_list) != enc_seq:
            raise ValueError(f"char_count_per_id length {len(cc_list)} must equal enc_seq {enc_seq}.")

        enc_out = self.encoder.forward(inputs_embeds, encoder_attention_mask_4d)

        char_w = int(char_input_ids.shape[1])
        char_seq_total = int(sum(cc_list))
        char_pad = _mask_row_valid_prefix(self.device, char_w, char_seq_total)
        if char_seq_total < char_w:
            char_pad = ttnn.slice(char_pad, [0, 0], [1, char_seq_total], [1, 1])

        up1 = _hard_upsample_nlc(enc_out, cc_list, device=self.device, hidden_size=self.hidden_size)
        # Upsampled character length equals ``sum(char_count_per_id)`` (batch 1); do not rely on
        # ``up1.shape[1]`` alone — tile layout can report incorrect logical widths to Python.
        char_len = char_seq_total
        if int(up1.shape[1]) != char_len:
            raise RuntimeError(
                f"char upsample width mismatch: up1.shape[1]={int(up1.shape[1])} vs sum(char_count)={char_len}."
            )
        if char_len < char_w:
            char_pad = ttnn.slice(char_pad, [0, 0], [1, char_len], [1, 1])
        elif char_len > char_w:
            raise ValueError(f"Upsampled char length {char_len} exceeds char_input_ids width {char_w}; pad HF inputs.")

        # HF-style character padding: prefix ones for real characters (matches ``_mask_row_valid_prefix``
        # + slices above when ``char_len == char_seq_total``). Avoids reading ``char_pad`` after reshape
        # views may share storage with ``char_pad_tt``.
        char_pad_valid_host = [1.0] * char_len

        pos_ids = ttnn.reshape(
            ttnn.arange(
                self.pad_token_id + 1,
                self.pad_token_id + 1 + char_len,
                step=1,
                dtype=ttnn.uint32,
                device=self.device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            (1, char_len),
        )
        pos_emb_tt = ttnn.embedding(
            pos_ids,
            weight=dec.embed_char_positions.weight,
            layout=ttnn.TILE_LAYOUT,
        )
        ttnn.deallocate(pos_ids)

        char_ids_slice = char_input_ids
        if char_w > char_len:
            char_ids_slice = ttnn.slice(char_input_ids, [0, 0], [batch, char_len], [1, 1])
        char_emb_tt = ttnn.embedding(char_ids_slice, weight=dec.embed_char.weight, layout=ttnn.TILE_LAYOUT)

        char_h = ttnn.add(
            ttnn.multiply(dec.pos_emb_alpha_char, pos_emb_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            char_emb_tt,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(pos_emb_tt)
        ttnn.deallocate(char_emb_tt)
        char_h = ttnn.add(char_h, up1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(up1)

        char_pad_tt = ttnn.reshape(char_pad, (batch, char_len, 1))

        if reference_discrete_durations is None:
            log_dur = self._duration_predictor(char_h, char_pad_tt, seq=char_len)
            dur_list = _discrete_duration_counts(log_dur, batch=batch, seq=char_len)
            ttnn.deallocate(log_dur)
        else:
            dur_list = [int(x) for x in reference_discrete_durations]
            if len(dur_list) != char_len:
                raise ValueError(f"reference_discrete_durations length {len(dur_list)} must equal char_len {char_len}.")

        for j in range(char_len):
            if j < len(char_pad_valid_host) and char_pad_valid_host[j] < 0.5:
                dur_list[j] = 0
        ttnn.deallocate(char_pad)

        # Drain the device profiler buffer at the natural encoder/decoder boundary.  The
        # duration readback above already syncs host<->device, so this adds no extra wait
        # in profiler builds and compiles to a no-op in normal builds.  Without it, the
        # 12000-marker on-device buffer overflows on a full forward (~1300 ops) and
        # ``python -m tracy`` fails post-run with "Op N not present in cpp_device_perf_report".
        ttnn.ReadDeviceProfiler(self.device)

        up2 = _hard_upsample_nlc(char_h, dur_list, device=self.device, hidden_size=self.hidden_size)
        ttnn.deallocate(char_h)
        unit_seq = int(sum(dur_list))
        if int(up2.shape[1]) != unit_seq:
            raise RuntimeError(
                f"unit upsample width mismatch: up2.shape[1]={int(up2.shape[1])} vs sum(dur_list)={unit_seq}."
            )

        pos_ids2 = ttnn.reshape(
            ttnn.arange(
                self.pad_token_id + 1,
                self.pad_token_id + 1 + unit_seq,
                step=1,
                dtype=ttnn.uint32,
                device=self.device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            (1, unit_seq),
        )
        pos2_tt = ttnn.embedding(
            pos_ids2,
            weight=dec.embed_positions.weight,
            layout=ttnn.TILE_LAYOUT,
        )
        ttnn.deallocate(pos_ids2)

        hidden = ttnn.add(
            up2,
            ttnn.multiply(dec.pos_emb_alpha, pos2_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(pos2_tt)
        ttnn.deallocate(up2)

        # Pad the unit sequence to a tile-aligned length so the decoder SDPA does not score real
        # queries against tile-padded garbage keys. ``pad_unit`` carries the valid-prefix length so
        # the additive mask drives padded keys to -inf and the post-SDPA gating zeros padded queries.
        dur_sum = int(sum(dur_list))
        assert dur_sum == unit_seq
        padded_unit_seq = ((unit_seq + 31) // 32) * 32
        if padded_unit_seq > unit_seq:
            hidden = ttnn.pad(hidden, [(0, 0), (0, padded_unit_seq - unit_seq), (0, 0)], value=0.0)
        pad_unit = _mask_row_valid_prefix(self.device, padded_unit_seq, dur_sum)
        attn_4d_tt = _expand_4d_padding_additive_b1(self.device, pad_unit, padded_unit_seq, padded_unit_seq)
        pad_unit_tt = ttnn.reshape(pad_unit, (1, padded_unit_seq, 1))

        num_heads = self.decoder_attention_heads
        head_dim = self.hidden_size // num_heads
        sdpa_cfg = self._sdpa_program_config(padded_unit_seq, padded_unit_seq)

        for i in range(self.decoder_layers):
            hidden = self._decoder_layer(
                hidden,
                attn_4d_tt,
                pad_unit_tt,
                dec.layers[i],
                batch=batch,
                seq=padded_unit_seq,
                num_heads=num_heads,
                head_dim=head_dim,
                hidden_size=self.hidden_size,
                sdpa_cfg=sdpa_cfg,
            )

        hidden = self._layer_norm(
            hidden,
            weight=dec.layer_norm.weight,
            bias=dec.layer_norm.bias,
        )
        logits = ttnn.linear(
            hidden,
            self.parameters.lm_head.weight,
            bias=None,
            core_grid=_core_grid(self.device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._linear_ln_compute_cfg,
        )
        ttnn.deallocate(hidden)
        ttnn.deallocate(attn_4d_tt)
        ttnn.deallocate(pad_unit_tt)
        # Slice back to the logical unit-seq so callers see ``[..., unit_seq, vocab]``. ``ttnn.slice``
        # requires the begins/ends/strides to match the input rank, which can be 3D or 4D depending
        # on the linear output layout.
        if padded_unit_seq > unit_seq:
            logits_shape = tuple(logits.shape)
            vocab = int(logits_shape[-1])
            rank = len(logits_shape)
            begins = [0] * rank
            ends = list(logits_shape)
            ends[-2] = unit_seq
            ends[-1] = vocab
            strides = [1] * rank
            logits = ttnn.slice(logits, begins, ends, strides)

        pad_out = pad_unit
        return logits, pad_out
