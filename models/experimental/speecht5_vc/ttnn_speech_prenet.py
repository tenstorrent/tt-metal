# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN SpeechT5 voice-conversion speech-encoder prenet.

This module ports the SpeechT5 speech-prenet path to TTNN:
1. FeatureEncoder (conv stack)
2. FeatureProjection (layer norm + linear projection)
3. PositionalConvEmbedding
4. Sinusoidal positional embedding addition

The implementation mirrors HuggingFace SpeechT5SpeechEncoderPrenet for inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import ttnn


def _to_ttnn_conv_weight(tensor: torch.Tensor, device) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor.contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _to_ttnn_tensor(tensor: torch.Tensor, device, *, layout=ttnn.TILE_LAYOUT) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor.contiguous(),
        dtype=ttnn.bfloat16,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _apply_activation(x: ttnn.Tensor, activation_name: str) -> ttnn.Tensor:
    act = activation_name.lower()
    if act in {"gelu", "gelu_new"}:
        return ttnn.gelu(x)
    if act == "relu":
        return ttnn.relu(x)
    if act == "silu":
        return ttnn.silu(x)
    if act == "elu":
        return ttnn.elu(x)
    raise ValueError(f"Unsupported SpeechT5 activation for TTNN prenet: {activation_name}")


@dataclass
class TTNNSpeechEncoderPrenetConfig:
    conv_dim: Tuple[int, ...]
    conv_stride: Tuple[int, ...]
    conv_kernel: Tuple[int, ...]
    conv_bias: bool
    feat_extract_norm: str
    feat_extract_activation: str
    hidden_size: int
    layer_norm_eps: float
    num_conv_pos_embeddings: int
    num_conv_pos_embedding_groups: int
    max_speech_positions: int
    pad_token_id: int

    @classmethod
    def from_hf_config(cls, cfg) -> "TTNNSpeechEncoderPrenetConfig":
        return cls(
            conv_dim=tuple(cfg.conv_dim),
            conv_stride=tuple(cfg.conv_stride),
            conv_kernel=tuple(cfg.conv_kernel),
            conv_bias=bool(cfg.conv_bias),
            feat_extract_norm=str(cfg.feat_extract_norm),
            feat_extract_activation=str(cfg.feat_extract_activation),
            hidden_size=int(cfg.hidden_size),
            layer_norm_eps=float(cfg.layer_norm_eps),
            num_conv_pos_embeddings=int(cfg.num_conv_pos_embeddings),
            num_conv_pos_embedding_groups=int(cfg.num_conv_pos_embedding_groups),
            max_speech_positions=int(cfg.max_speech_positions),
            pad_token_id=int(cfg.pad_token_id),
        )


def _prepare_group_norm_params(weight: torch.Tensor, bias: torch.Tensor, num_channels: int, device) -> Dict[str, ttnn.Tensor]:
    # Single-core-across-channel keeps configuration portable across WH/BH.
    num_cores_across_channel = 1
    try:
        input_mask = ttnn.create_group_norm_input_mask(num_channels, num_channels, num_cores_across_channel, ttnn.bfloat16)
    except TypeError:
        input_mask = ttnn.create_group_norm_input_mask(num_channels, num_channels, num_cores_across_channel)
    input_mask = ttnn.to_device(input_mask, device)

    weight_rm = ttnn.create_group_norm_weight_bias_rm(weight, num_channels, num_cores_across_channel)
    bias_rm = ttnn.create_group_norm_weight_bias_rm(bias, num_channels, num_cores_across_channel)

    return {
        "input_mask": input_mask,
        "weight": _to_ttnn_tensor(weight_rm, device, layout=ttnn.ROW_MAJOR_LAYOUT),
        "bias": _to_ttnn_tensor(bias_rm, device, layout=ttnn.ROW_MAJOR_LAYOUT),
    }


def preprocess_speech_encoder_prenet_parameters(
    torch_prenet,
    config: TTNNSpeechEncoderPrenetConfig,
    device,
) -> Dict:
    parameters: Dict[str, object] = {}

    conv_layers: List[Dict[str, object]] = []
    for conv_layer in torch_prenet.feature_encoder.conv_layers:
        conv = conv_layer.conv
        conv_params: Dict[str, object] = {
            "in_channels": int(conv.in_channels),
            "out_channels": int(conv.out_channels),
            "kernel_size": int(conv.kernel_size[0]),
            "stride": int(conv.stride[0]),
            "padding": int(conv.padding[0]),
            "groups": int(conv.groups),
            "weight": _to_ttnn_conv_weight(conv.weight.data, device),
            "bias": _to_ttnn_tensor(conv.bias.data.view(1, 1, 1, -1), device, layout=ttnn.ROW_MAJOR_LAYOUT)
            if conv.bias is not None
            else None,
            "norm_type": "none",
        }

        if hasattr(conv_layer, "layer_norm"):
            if isinstance(conv_layer.layer_norm, torch.nn.GroupNorm):
                conv_params["norm_type"] = "group"
                conv_params["group_norm"] = _prepare_group_norm_params(
                    conv_layer.layer_norm.weight.data.float(),
                    conv_layer.layer_norm.bias.data.float(),
                    int(conv.out_channels),
                    device,
                )
            elif isinstance(conv_layer.layer_norm, torch.nn.LayerNorm):
                conv_params["norm_type"] = "layer"
                conv_params["layer_norm"] = {
                    "weight": _to_ttnn_tensor(conv_layer.layer_norm.weight.data, device),
                    "bias": _to_ttnn_tensor(conv_layer.layer_norm.bias.data, device),
                }

        conv_layers.append(conv_params)

    parameters["feature_encoder"] = {"conv_layers": conv_layers}

    feature_projection = torch_prenet.feature_projection
    parameters["feature_projection"] = {
        "layer_norm": {
            "weight": _to_ttnn_tensor(feature_projection.layer_norm.weight.data, device),
            "bias": _to_ttnn_tensor(feature_projection.layer_norm.bias.data, device),
        },
        "projection": {
            "weight": _to_ttnn_tensor(
                feature_projection.projection.weight.data.transpose(0, 1),
                device,
            ),
            "bias": _to_ttnn_tensor(feature_projection.projection.bias.data, device),
        },
    }

    pos_conv = torch_prenet.pos_conv_embed.conv
    parameters["pos_conv_embed"] = {
        "in_channels": int(pos_conv.in_channels),
        "out_channels": int(pos_conv.out_channels),
        "kernel_size": int(pos_conv.kernel_size[0]),
        "stride": int(pos_conv.stride[0]),
        "padding": int(pos_conv.padding[0]),
        "groups": int(pos_conv.groups),
        # `pos_conv_embed.conv.weight` already resolves weight-norm reparameterization.
        "weight": _to_ttnn_conv_weight(pos_conv.weight.data, device),
        "bias": _to_ttnn_tensor(pos_conv.bias.data.view(1, 1, 1, -1), device, layout=ttnn.ROW_MAJOR_LAYOUT)
        if pos_conv.bias is not None
        else None,
        "num_pad_remove": 1 if config.num_conv_pos_embeddings % 2 == 0 else 0,
    }

    # Store torch weights for position-id based lookup; cache TT tensors by seq len at runtime.
    parameters["pos_sinusoidal_weights_torch"] = torch_prenet.pos_sinusoidal_embed.weights.detach().cpu().float()

    return parameters


class TTNNSpeechEncoderPrenet:
    """
    TTNN SpeechT5 speech-prenet:
    raw waveform -> conv stack -> projection -> positional embeddings.
    """

    def __init__(self, device, parameters: Dict, config: TTNNSpeechEncoderPrenetConfig):
        self.device = device
        self.parameters = parameters
        self.config = config

        self.conv_config = ttnn.Conv1dConfig(
            weights_dtype=ttnn.bfloat16,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        )
        self.conv_compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        self._sinusoidal_cache: Dict[Tuple[int, int], ttnn.Tensor] = {}

    def _run_conv1d(
        self,
        x: ttnn.Tensor,
        params: Dict[str, object],
        *,
        input_length: int,
    ) -> Tuple[ttnn.Tensor, int]:
        result = ttnn.conv1d(
            input_tensor=x,
            weight_tensor=params["weight"],
            in_channels=params["in_channels"],
            out_channels=params["out_channels"],
            device=self.device,
            bias_tensor=params["bias"],
            kernel_size=params["kernel_size"],
            stride=params["stride"],
            padding=params["padding"],
            dilation=1,
            groups=params["groups"],
            batch_size=x.shape[0],
            input_length=input_length,
            dtype=ttnn.bfloat16,
            conv_config=self.conv_config,
            compute_config=self.conv_compute_config,
            return_output_dim=True,
        )

        if isinstance(result, tuple):
            if len(result) == 3:
                out_tensor, out_length, _ = result
            else:
                out_tensor, out_length = result
        else:
            out_tensor = result
            out_length = out_tensor.shape[2]

        if out_tensor.memory_config().is_sharded():
            out_tensor = ttnn.to_memory_config(out_tensor, ttnn.DRAM_MEMORY_CONFIG)

        return out_tensor, int(out_length)

    def _apply_group_norm(self, x: ttnn.Tensor, norm_params: Dict[str, ttnn.Tensor], num_groups: int) -> ttnn.Tensor:
        try:
            return ttnn.group_norm(
                x,
                num_groups=num_groups,
                input_mask=norm_params["input_mask"],
                weight=norm_params["weight"],
                bias=norm_params["bias"],
                epsilon=self.config.layer_norm_eps,
                core_grid=ttnn.CoreGrid(y=1, x=1),
                dtype=ttnn.bfloat16,
                inplace=False,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        except TypeError:
            return ttnn.group_norm(
                x,
                num_groups=num_groups,
                input_mask=norm_params["input_mask"],
                weight=norm_params["weight"],
                bias=norm_params["bias"],
                epsilon=self.config.layer_norm_eps,
            )

    def _apply_conv_layer_norm(self, x: ttnn.Tensor, layer_norm_params: Dict[str, ttnn.Tensor]) -> ttnn.Tensor:
        return ttnn.layer_norm(
            x,
            weight=layer_norm_params["weight"],
            bias=layer_norm_params["bias"],
            epsilon=self.config.layer_norm_eps,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    def _feature_encoder(self, input_values: torch.Tensor) -> ttnn.Tensor:
        # [B, T] -> [B, 1, T, 1]
        hidden_states = ttnn.from_torch(
            input_values.unsqueeze(1).unsqueeze(-1),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        input_length = int(input_values.shape[1])
        for conv_layer in self.parameters["feature_encoder"]["conv_layers"]:
            hidden_states, input_length = self._run_conv1d(hidden_states, conv_layer, input_length=input_length)

            norm_type = conv_layer.get("norm_type", "none")
            if norm_type == "group":
                hidden_states = self._apply_group_norm(
                    hidden_states,
                    conv_layer["group_norm"],
                    num_groups=conv_layer["out_channels"],
                )
            elif norm_type == "layer":
                hidden_states = self._apply_conv_layer_norm(hidden_states, conv_layer["layer_norm"])

            hidden_states = _apply_activation(hidden_states, self.config.feat_extract_activation)

        # [B, 1, T', C] -> [B, T', C]
        return ttnn.to_layout(ttnn.squeeze(hidden_states, dim=1), ttnn.TILE_LAYOUT)

    @staticmethod
    def _get_feat_extract_output_lengths(
        input_lengths: torch.Tensor,
        conv_kernel: Tuple[int, ...],
        conv_stride: Tuple[int, ...],
    ) -> torch.Tensor:
        output_lengths = input_lengths.to(torch.long)
        for kernel_size, stride in zip(conv_kernel, conv_stride):
            output_lengths = torch.div(output_lengths - kernel_size, stride, rounding_mode="floor") + 1
        return output_lengths

    def _get_feature_vector_attention_mask(
        self,
        feature_vector_length: int,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        non_padded_lengths = attention_mask.long().cumsum(dim=-1)[:, -1]
        output_lengths = self._get_feat_extract_output_lengths(
            non_padded_lengths,
            self.config.conv_kernel,
            self.config.conv_stride,
        ).to(torch.long)

        batch_size = attention_mask.shape[0]
        reduced_attention_mask = torch.zeros(
            (batch_size, feature_vector_length),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        reduced_attention_mask[
            (torch.arange(batch_size, device=attention_mask.device), torch.clamp(output_lengths - 1, min=0))
        ] = 1
        reduced_attention_mask = reduced_attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return reduced_attention_mask

    def _feature_projection(self, extract_features: ttnn.Tensor) -> ttnn.Tensor:
        projection_params = self.parameters["feature_projection"]
        hidden_states = ttnn.layer_norm(
            extract_features,
            weight=projection_params["layer_norm"]["weight"],
            bias=projection_params["layer_norm"]["bias"],
            epsilon=self.config.layer_norm_eps,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        hidden_states = ttnn.linear(
            hidden_states,
            projection_params["projection"]["weight"],
            bias=projection_params["projection"]["bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        return hidden_states

    def _positional_conv_embedding(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        pos_conv_params = self.parameters["pos_conv_embed"]
        conv_input = ttnn.to_layout(ttnn.unsqueeze(hidden_states, dim=1), ttnn.ROW_MAJOR_LAYOUT)  # [B, 1, T, C]
        conv_output, out_length = self._run_conv1d(
            conv_input,
            pos_conv_params,
            input_length=int(hidden_states.shape[1]),
        )

        if pos_conv_params["num_pad_remove"] > 0 and out_length > 0:
            conv_output = ttnn.slice(
                conv_output,
                [0, 0, 0, 0],
                [conv_output.shape[0], conv_output.shape[1], out_length - pos_conv_params["num_pad_remove"], conv_output.shape[3]],
            )
            out_length = out_length - pos_conv_params["num_pad_remove"]

        conv_output = _apply_activation(conv_output, self.config.feat_extract_activation)
        conv_output = ttnn.to_layout(ttnn.squeeze(conv_output, dim=1), ttnn.TILE_LAYOUT)  # [B, T, C]

        if out_length > hidden_states.shape[1]:
            conv_output = ttnn.slice(
                conv_output,
                [0, 0, 0],
                [conv_output.shape[0], hidden_states.shape[1], conv_output.shape[2]],
            )
        elif out_length < hidden_states.shape[1]:
            hidden_states = ttnn.slice(
                hidden_states,
                [0, 0, 0],
                [hidden_states.shape[0], out_length, hidden_states.shape[2]],
            )

        return ttnn.add(hidden_states, conv_output, memory_config=ttnn.L1_MEMORY_CONFIG)

    def _build_position_ids(self, padding_mask: torch.Tensor) -> torch.Tensor:
        # Matches SpeechT5SinusoidalPositionalEmbedding.create_position_ids_from_input_ids
        pad_idx = self.config.pad_token_id
        mask = padding_mask.ne(pad_idx).int()
        incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
        return incremental_indices.long() + pad_idx

    def _sinusoidal_embedding(self, batch_size: int, seq_len: int, reduced_attention_mask: Optional[torch.Tensor]) -> ttnn.Tensor:
        if reduced_attention_mask is None:
            cache_key = (batch_size, seq_len)
            if cache_key not in self._sinusoidal_cache:
                start = self.config.pad_token_id + 1
                pos = self.parameters["pos_sinusoidal_weights_torch"][start : start + seq_len].unsqueeze(0)
                pos = pos.expand(batch_size, -1, -1).contiguous()
                self._sinusoidal_cache[cache_key] = _to_ttnn_tensor(pos, self.device)
            return self._sinusoidal_cache[cache_key]

        padding_mask = reduced_attention_mask.ne(1).long()
        position_ids = self._build_position_ids(padding_mask)
        flat_ids = position_ids.reshape(-1)
        pos = self.parameters["pos_sinusoidal_weights_torch"].index_select(0, flat_ids).view(batch_size, seq_len, -1)
        return _to_ttnn_tensor(pos, self.device)

    def __call__(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[ttnn.Tensor, Optional[torch.Tensor]]:
        extract_features = self._feature_encoder(input_values)

        reduced_attention_mask: Optional[torch.Tensor] = None
        if attention_mask is not None:
            reduced_attention_mask = self._get_feature_vector_attention_mask(
                int(extract_features.shape[1]),
                attention_mask,
            )

        hidden_states = self._feature_projection(extract_features)
        hidden_states = self._positional_conv_embedding(hidden_states)
        hidden_states = ttnn.add(
            hidden_states,
            self._sinusoidal_embedding(
                batch_size=int(input_values.shape[0]),
                seq_len=int(hidden_states.shape[1]),
                reduced_attention_mask=reduced_attention_mask,
            ),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        return hidden_states, reduced_attention_mask
