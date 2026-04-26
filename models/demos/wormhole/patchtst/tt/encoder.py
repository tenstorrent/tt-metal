# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Source lineage: HuggingFace PatchTST and PatchTST paper implementation details
# - https://huggingface.co/docs/transformers/en/model_doc/patchtst
# - https://github.com/huggingface/transformers/tree/main/src/transformers/models/patchtst
# - https://arxiv.org/abs/2211.14730

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

import ttnn
from models.demos.wormhole.patchtst.reference.hf_reference import ReferenceArtifacts
from models.demos.wormhole.patchtst.tt.attention import PatchTSTAttention, build_attention
from models.demos.wormhole.patchtst.tt.common import (
    MEMORY_CONFIG_BY_TIER,
    PatchTSTRuntimePolicy,
    TTLinear,
    build_linear_from_state,
    maybe_height_shard_3d_tensor,
    sharded_layout_roundtrip_3d_tensor,
    to_tt_param,
)


@dataclass
class PatchTSTEncoderLayer:
    attention: PatchTSTAttention
    norm1_kind: str
    norm1_weight: ttnn.Tensor
    norm1_bias: ttnn.Tensor
    norm2_kind: str | None
    norm2_weight: ttnn.Tensor | None
    norm2_bias: ttnn.Tensor | None
    norm3_kind: str
    norm3_weight: ttnn.Tensor
    norm3_bias: ttnn.Tensor
    ff1: TTLinear
    ff2: TTLinear
    channel_attention_enabled: bool

    def release(self) -> None:
        self.attention.release()
        ttnn.deallocate(self.norm1_weight)
        ttnn.deallocate(self.norm1_bias)
        if self.norm2_weight is not None:
            ttnn.deallocate(self.norm2_weight)
        if self.norm2_bias is not None:
            ttnn.deallocate(self.norm2_bias)
        ttnn.deallocate(self.norm3_weight)
        ttnn.deallocate(self.norm3_bias)
        self.ff1.release()
        self.ff2.release()


def _apply_norm_tt(
    kind: str, weight: ttnn.Tensor, bias: ttnn.Tensor, hidden_state: ttnn.Tensor, mem_cfg
) -> ttnn.Tensor:
    if kind == "layer_norm":
        return ttnn.layer_norm(hidden_state, weight=weight, bias=bias, epsilon=1e-5, memory_config=mem_cfg)
    if kind == "batch_norm_affine":
        return ttnn.add(ttnn.multiply(hidden_state, weight, memory_config=mem_cfg), bias, memory_config=mem_cfg)
    raise ValueError(f"Unsupported norm kind: {kind}")


def _feed_forward_tt(
    hidden_state: ttnn.Tensor, layer: PatchTSTEncoderLayer, mem_cfg, dtype, use_fused_ffn
) -> ttnn.Tensor:
    if use_fused_ffn:
        out = ttnn.linear(
            hidden_state, layer.ff1.weight, bias=layer.ff1.bias, memory_config=mem_cfg, dtype=dtype, activation="gelu"
        )
    else:
        out = ttnn.linear(hidden_state, layer.ff1.weight, bias=layer.ff1.bias, memory_config=mem_cfg, dtype=dtype)
        out = ttnn.gelu(out)
    return ttnn.linear(out, layer.ff2.weight, bias=layer.ff2.bias, memory_config=mem_cfg, dtype=dtype)


def encode_hidden_state(
    hidden_state: ttnn.Tensor,
    encoder_layers: list[PatchTSTEncoderLayer],
    pre_norm: bool,
    enable_channel_attention: bool,
    runtime: PatchTSTRuntimePolicy,
    device,
    dtype: ttnn.DataType = ttnn.bfloat16,
) -> ttnn.Tensor:
    mem_cfg = MEMORY_CONFIG_BY_TIER[runtime.activation_memory_tier]

    def run_attention(attention_input: ttnn.Tensor, attention: PatchTSTAttention) -> ttnn.Tensor:
        attention_input_for_compute = attention_input
        used_compute_sharding = False
        trim_rows_after_attention = 0
        trim_seq_len = 0
        trim_hidden_size = 0
        auxiliary_tensors: list[ttnn.Tensor] = []

        def track(tensor: ttnn.Tensor) -> None:
            if tensor is not attention_input:
                auxiliary_tensors.append(tensor)

        try:
            if len(attention_input_for_compute.shape) == 4:
                flattened = ttnn.reshape(
                    attention_input_for_compute,
                    (
                        int(attention_input_for_compute.shape[0]) * int(attention_input_for_compute.shape[1]),
                        int(attention_input_for_compute.shape[2]),
                        int(attention_input_for_compute.shape[3]),
                    ),
                )
                if flattened is not attention_input_for_compute:
                    track(flattened)
                attention_input_for_compute = flattened
            if len(attention_input_for_compute.shape) != 3:
                raise ValueError(
                    "Self-attention expects a rank-3 [B, S, D] tensor after canonicalization; "
                    f"received rank={len(attention_input_for_compute.shape)}."
                )

            if runtime.use_sharded_attention_inputs:
                canonical_input = attention_input_for_compute
                sharding_input = canonical_input
                input_rows = int(canonical_input.shape[0])
                input_seq_len = int(canonical_input.shape[1])
                input_hidden_size = int(canonical_input.shape[2])
                if input_seq_len > 0 and (input_rows * input_seq_len) % ttnn.TILE_SIZE != 0:
                    step = ttnn.TILE_SIZE // math.gcd(input_seq_len, ttnn.TILE_SIZE)
                    aligned_rows = ((input_rows + step - 1) // step) * step
                    pad_rows = aligned_rows - input_rows
                    if pad_rows > 0:
                        sharding_input = ttnn.pad(canonical_input, padding=((0, pad_rows), (0, 0), (0, 0)), value=0.0)
                        track(sharding_input)
                        trim_rows_after_attention = input_rows
                        trim_seq_len = input_seq_len
                        trim_hidden_size = input_hidden_size

                sharded_input, used_compute_sharding = maybe_height_shard_3d_tensor(
                    sharding_input, device=device, enable=True, allow_fallback=runtime.allow_shard_fallback
                )
                if used_compute_sharding:
                    attention_input_for_compute = sharded_input
                    if sharded_input is not sharding_input:
                        track(sharded_input)
                else:
                    if sharded_input is not sharding_input:
                        track(sharded_input)
                    roundtripped_input, _ = sharded_layout_roundtrip_3d_tensor(
                        canonical_input,
                        device=device,
                        enable=True,
                        interleaved_memory_config=mem_cfg,
                        max_rows_per_chunk=runtime.sharding_max_rows_per_chunk,
                        allow_fallback=runtime.allow_shard_fallback,
                    )
                    if roundtripped_input is not canonical_input:
                        attention_input_for_compute = roundtripped_input
                        track(roundtripped_input)

            attention_output = attention(attention_input_for_compute, runtime, device, dtype)
            if used_compute_sharding and trim_rows_after_attention > 0:
                trimmed_output = ttnn.slice(
                    attention_output, (0, 0, 0), (trim_rows_after_attention, trim_seq_len, trim_hidden_size)
                )
                ttnn.deallocate(attention_output)
                attention_output = trimmed_output
            return attention_output
        finally:
            seen_ids: set[int] = set()
            for tensor in auxiliary_tensors:
                tensor_id = id(tensor)
                if tensor_id in seen_ids:
                    continue
                seen_ids.add(tensor_id)
                ttnn.deallocate(tensor)

    for layer_idx, layer in enumerate(encoder_layers):
        batch_size, num_channels, sequence_length, d_model = hidden_state.shape
        x = ttnn.reshape(hidden_state, (batch_size * num_channels, sequence_length, d_model))
        if pre_norm:
            attn_out = run_attention(
                _apply_norm_tt(layer.norm1_kind, layer.norm1_weight, layer.norm1_bias, x, mem_cfg), layer.attention
            )
            x = ttnn.add(x, attn_out, memory_config=mem_cfg)
        else:
            attn_out = run_attention(x, layer.attention)
            x = _apply_norm_tt(
                layer.norm1_kind,
                layer.norm1_weight,
                layer.norm1_bias,
                ttnn.add(x, attn_out, memory_config=mem_cfg),
                mem_cfg,
            )
        hidden_state = ttnn.reshape(x, (batch_size, num_channels, sequence_length, d_model))
        ttnn.deallocate(attn_out)

        if enable_channel_attention:
            if (
                not layer.channel_attention_enabled
                or layer.norm2_kind is None
                or layer.norm2_weight is None
                or layer.norm2_bias is None
            ):
                raise ValueError(
                    "channel_mode='attention' requested but reference checkpoint has channel_attention=False. "
                    f"Layer {layer_idx} does not expose channel-attention parameters."
                )
            ch = ttnn.reshape(
                ttnn.permute(hidden_state, (0, 2, 1, 3)), (batch_size * sequence_length, num_channels, d_model)
            )
            if pre_norm:
                ch_attn_out = run_attention(
                    _apply_norm_tt(layer.norm2_kind, layer.norm2_weight, layer.norm2_bias, ch, mem_cfg), layer.attention
                )
                ch = ttnn.add(ch, ch_attn_out, memory_config=mem_cfg)
                ttnn.deallocate(ch_attn_out)
            else:
                ch_attn_out = run_attention(ch, layer.attention)
                ch = _apply_norm_tt(
                    layer.norm2_kind,
                    layer.norm2_weight,
                    layer.norm2_bias,
                    ttnn.add(ch, ch_attn_out, memory_config=mem_cfg),
                    mem_cfg,
                )
                ttnn.deallocate(ch_attn_out)
            hidden_state = ttnn.permute(
                ttnn.reshape(ch, (batch_size, sequence_length, num_channels, d_model)), (0, 2, 1, 3)
            )

        x = ttnn.reshape(hidden_state, (batch_size * num_channels, sequence_length, d_model))
        if pre_norm:
            ff_out = _feed_forward_tt(
                _apply_norm_tt(layer.norm3_kind, layer.norm3_weight, layer.norm3_bias, x, mem_cfg),
                layer,
                mem_cfg,
                dtype,
                runtime.use_fused_ffn,
            )
            x = ttnn.add(x, ff_out, memory_config=mem_cfg)
            ttnn.deallocate(ff_out)
        else:
            ff_out = _feed_forward_tt(x, layer, mem_cfg, dtype, runtime.use_fused_ffn)
            x = _apply_norm_tt(
                layer.norm3_kind,
                layer.norm3_weight,
                layer.norm3_bias,
                ttnn.add(x, ff_out, memory_config=mem_cfg),
                mem_cfg,
            )
            ttnn.deallocate(ff_out)
        hidden_state = ttnn.reshape(x, (batch_size, num_channels, sequence_length, d_model))

    return hidden_state


def _build_norm(
    state: dict[str, torch.Tensor], layer_module: torch.nn.Module, prefix: str, *, device, dtype, memory_config
):
    if isinstance(layer_module, torch.nn.LayerNorm):
        return (
            "layer_norm",
            to_tt_param(state[f"{prefix}.weight"], device=device, dtype=dtype, memory_config=memory_config),
            to_tt_param(state[f"{prefix}.bias"], device=device, dtype=dtype, memory_config=memory_config),
        )
    if hasattr(layer_module, "batchnorm"):
        batch_norm = layer_module.batchnorm
        weight = state[f"{prefix}.batchnorm.weight"].detach().to(torch.float32)
        bias = state[f"{prefix}.batchnorm.bias"].detach().to(torch.float32)
        running_mean = state[f"{prefix}.batchnorm.running_mean"].detach().to(torch.float32)
        running_var = state[f"{prefix}.batchnorm.running_var"].detach().to(torch.float32)
        scale = weight / torch.sqrt(running_var + float(batch_norm.eps))
        shift = bias - running_mean * scale
        return (
            "batch_norm_affine",
            to_tt_param(scale, device=device, dtype=dtype, memory_config=memory_config),
            to_tt_param(shift, device=device, dtype=dtype, memory_config=memory_config),
        )
    raise ValueError(f"Unsupported norm module at {prefix}: {type(layer_module).__name__}")


def build_encoder_layers(
    reference: ReferenceArtifacts,
    *,
    device,
    dtype: ttnn.DataType = ttnn.bfloat16,
    memory_tier: str = "dram",
) -> tuple[list[PatchTSTEncoderLayer], bool]:
    state = reference.model.state_dict()
    memory_config = MEMORY_CONFIG_BY_TIER[memory_tier]
    encoder = reference.model.model.encoder
    num_heads = int(reference.config.num_attention_heads)
    hidden_size = int(reference.config.d_model)
    padded_head_dim = ((hidden_size // num_heads + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
    layers = []
    for layer_idx, layer_module in enumerate(encoder.layers):
        layer_prefix = f"model.encoder.layers.{layer_idx}"
        norm1_kind, norm1_weight, norm1_bias = _build_norm(
            state,
            getattr(layer_module, "norm_sublayer1"),
            f"{layer_prefix}.norm_sublayer1",
            device=device,
            dtype=dtype,
            memory_config=memory_config,
        )
        if getattr(layer_module, "channel_attention", False):
            norm2_kind, norm2_weight, norm2_bias = _build_norm(
                state,
                getattr(layer_module, "norm_sublayer2"),
                f"{layer_prefix}.norm_sublayer2",
                device=device,
                dtype=dtype,
                memory_config=memory_config,
            )
        else:
            norm2_kind, norm2_weight, norm2_bias = None, None, None
        norm3_kind, norm3_weight, norm3_bias = _build_norm(
            state,
            getattr(layer_module, "norm_sublayer3"),
            f"{layer_prefix}.norm_sublayer3",
            device=device,
            dtype=dtype,
            memory_config=memory_config,
        )
        layers.append(
            PatchTSTEncoderLayer(
                attention=build_attention(
                    state,
                    layer_prefix,
                    num_heads,
                    padded_head_dim,
                    device=device,
                    dtype=dtype,
                    memory_config=memory_config,
                ),
                norm1_kind=norm1_kind,
                norm1_weight=norm1_weight,
                norm1_bias=norm1_bias,
                norm2_kind=norm2_kind,
                norm2_weight=norm2_weight,
                norm2_bias=norm2_bias,
                norm3_kind=norm3_kind,
                norm3_weight=norm3_weight,
                norm3_bias=norm3_bias,
                ff1=build_linear_from_state(
                    state, f"{layer_prefix}.ff.0", device=device, dtype=dtype, memory_config=memory_config
                ),
                ff2=build_linear_from_state(
                    state, f"{layer_prefix}.ff.3", device=device, dtype=dtype, memory_config=memory_config
                ),
                channel_attention_enabled=bool(getattr(layer_module, "channel_attention", False)),
            )
        )
    return layers, bool(reference.config.pre_norm)
