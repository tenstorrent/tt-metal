# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from typing import Optional, Tuple
from .configuration_tinytimemixer import TinyTimeMixerConfig


class TinyTimeMixerPatchify:
    def __init__(self, config: TinyTimeMixerConfig):
        self.patch_length = config.patch_length
        self.patch_stride = config.patch_stride
        self.num_patches = config.num_patches
        self.context_length = config.context_length

    def __call__(self, past_values: torch.Tensor) -> torch.Tensor:
        # past_values: [batch_size, context_length, num_channels]
        # Output: [batch_size, num_channels, num_patches, patch_length]
        batch_size, seq_len, num_channels = past_values.shape
        # Use unfold
        patches = past_values.unfold(dimension=1, size=self.patch_length, step=self.patch_stride)
        # patches: [batch_size, num_patches, num_channels, patch_length]
        patches = patches.transpose(1, 2)  # [batch_size, num_channels, num_patches, patch_length]
        return patches


class TinyTimeMixerStdScaler:
    def __init__(self, config: TinyTimeMixerConfig):
        self.dim = 1
        self.keepdim = True
        self.minimum_scale = 1e-5

    def __call__(self, data: torch.Tensor, observed_indicator: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        denominator = observed_indicator.sum(self.dim, keepdim=self.keepdim)
        denominator = denominator.clamp_min(1.0)
        loc = (data * observed_indicator).sum(self.dim, keepdim=self.keepdim) / denominator
        variance = (((data - loc) * observed_indicator) ** 2).sum(self.dim, keepdim=self.keepdim) / denominator
        scale = torch.sqrt(variance + self.minimum_scale)
        return (data - loc) / scale, loc, scale


class TinyTimeMixerNormLayer:
    def __init__(self, config: TinyTimeMixerConfig):
        self.norm_mlp = config.norm_mlp
        self.d_model = config.d_model
        self.norm_eps = config.norm_eps
        if "batch" in config.norm_mlp.lower():
            # For simplicity, use LayerNorm for now
            pass
        else:
            self.norm = torch.nn.LayerNorm(self.d_model, eps=self.norm_eps)

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        # Assume inputs: [batch_size, num_channels, num_patches, d_model]
        # For LayerNorm, normalize over d_model
        return self.norm(inputs)


class TinyTimeMixerMLP:
    def __init__(self, in_features, out_features, config):
        self.in_features = in_features
        self.out_features = out_features
        num_hidden = in_features * config.expansion_factor
        self.fc1 = torch.nn.Linear(in_features, num_hidden)
        self.fc2 = torch.nn.Linear(num_hidden, out_features)
        self.dropout = torch.nn.Dropout(config.dropout)

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.dropout(torch.nn.functional.gelu(self.fc1(inputs)))
        inputs = self.dropout(self.fc2(inputs))
        return inputs


class TinyTimeMixerGatedAttention:
    def __init__(self, in_size: int, out_size: int):
        self.attn_layer = torch.nn.Linear(in_size, out_size)
        self.attn_softmax = torch.nn.Softmax(dim=-1)

    def __call__(self, inputs):
        attn_weight = self.attn_softmax(self.attn_layer(inputs))
        inputs = inputs * attn_weight
        return inputs


class PatchMixerBlock:
    def __init__(self, config: TinyTimeMixerConfig):
        self.norm = TinyTimeMixerNormLayer(config)
        self.self_attn = config.self_attn
        self.gated_attn = config.gated_attn
        self.mlp = TinyTimeMixerMLP(
            in_features=config.num_patches,
            out_features=config.num_patches,
            config=config,
        )
        if config.gated_attn:
            self.gating_block = TinyTimeMixerGatedAttention(config.num_patches, config.num_patches)
        if config.self_attn:
            # For simplicity, skip self-attention for now
            pass

    def __call__(self, hidden_state: torch.Tensor) -> torch.Tensor:
        residual = hidden_state
        hidden_state = self.norm(hidden_state)
        # Transpose for patch mixing
        hidden_state = hidden_state.transpose(2, 3)  # [bs, n_vars, d_model, num_patches]
        hidden_state = self.mlp(hidden_state)
        if self.gated_attn:
            hidden_state = self.gating_block(hidden_state)
        hidden_state = hidden_state.transpose(2, 3)  # back
        out = hidden_state + residual
        return out


class FeatureMixerBlock:
    def __init__(self, config: TinyTimeMixerConfig):
        self.norm = TinyTimeMixerNormLayer(config)
        self.gated_attn = config.gated_attn
        self.mlp = TinyTimeMixerMLP(
            in_features=config.d_model,
            out_features=config.d_model,
            config=config,
        )
        if config.gated_attn:
            self.gating_block = TinyTimeMixerGatedAttention(config.d_model, config.d_model)

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        residual = hidden
        hidden = self.norm(hidden)
        hidden = self.mlp(hidden)
        if self.gated_attn:
            hidden = self.gating_block(hidden)
        out = hidden + residual
        return out


class TinyTimeMixerLayer:
    def __init__(self, config: TinyTimeMixerConfig):
        self.num_patches = config.num_patches
        if config.num_patches > 1:
            self.patch_mixer = PatchMixerBlock(config=config)
        self.feature_mixer = FeatureMixerBlock(config=config)
        self.mode = config.mode
        if config.mode == "mix_channel":
            # For simplicity, skip channel mixing
            pass

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.mode == "mix_channel":
            pass  # skip
        if self.num_patches > 1:
            hidden = self.patch_mixer(hidden)
        hidden = self.feature_mixer(hidden)
        return hidden


class TinyTimeMixerBlock:
    def __init__(self, config: TinyTimeMixerConfig):
        self.mixers = torch.nn.ModuleList([TinyTimeMixerLayer(config=config) for _ in range(config.num_layers)])
        self.adaptive_patching_levels = config.adaptive_patching_levels
        # For simplicity, set to 0

    def __call__(self, hidden_state: torch.Tensor, output_hidden_states: bool = False) -> Tuple[torch.Tensor, Optional[list]]:
        all_hidden_states = []
        embedding = hidden_state
        for mod in self.mixers:
            embedding = mod(embedding)
            if output_hidden_states:
                all_hidden_states.append(embedding)
        if output_hidden_states:
            return embedding, all_hidden_states
        else:
            return embedding, None


class TinyTimeMixerEncoder:
    def __init__(self, config: TinyTimeMixerConfig):
        self.patcher = torch.nn.Linear(config.patch_length, config.d_model)
        self.positional_encoder = None  # Skip for now
        self.mlp_mixer_encoder = TinyTimeMixerBlock(config=config)

    def __call__(self, past_values: torch.Tensor, output_hidden_states: bool = False) -> Tuple[torch.Tensor, Optional[list]]:
        # past_values: [bs, n_vars, num_patches, patch_length]
        patches = self.patcher(past_values)  # [bs, n_vars, num_patches, d_model]
        if self.positional_encoder is not None:
            patches = self.positional_encoder(patches)
        last_hidden_state, hidden_states = self.mlp_mixer_encoder(patches, output_hidden_states=output_hidden_states)
        return last_hidden_state, hidden_states


class TinyTimeMixerModel:
    def __init__(self, config: TinyTimeMixerConfig):
        self.encoder = TinyTimeMixerEncoder(config)
        self.patching = TinyTimeMixerPatchify(config)
        self.scaler = TinyTimeMixerStdScaler(config)

    def __call__(self, past_values: torch.Tensor, past_observed_mask: Optional[torch.Tensor] = None, output_hidden_states: bool = False):
        if past_observed_mask is None:
            past_observed_mask = torch.ones_like(past_values)
        scaled_past_values, loc, scale = self.scaler(past_values, past_observed_mask)
        patched_x = self.patching(scaled_past_values)
        encoder_output, hidden_states = self.encoder(patched_x, output_hidden_states=output_hidden_states)
        return encoder_output, hidden_states, patched_x, loc, scale


class TinyTimeMixerForPredictionHead:
    def __init__(self, config: TinyTimeMixerConfig):
        self.prediction_channel_indices = config.prediction_channel_indices
        self.prediction_length = config.prediction_length
        self.dropout_layer = torch.nn.Dropout(config.head_dropout)
        self.base_forecast_block = torch.nn.Linear(config.num_patches * config.d_model, config.prediction_length)
        self.flatten = torch.nn.Flatten(start_dim=-2)

    def __call__(self, hidden_features: torch.Tensor, past_values: torch.Tensor, future_values=None):
        # hidden_features: [bs, n_vars, num_patches * d_model]
        hidden_features = self.dropout_layer(hidden_features)
        forecast = self.base_forecast_block(hidden_features)  # [bs, n_vars, prediction_length]
        forecast = forecast.transpose(-1, -2)  # [bs, prediction_length, n_vars]
        return forecast


class TinyTimeMixerForPrediction:
    def __init__(self, config: TinyTimeMixerConfig):
        self.config = config
        self.backbone = TinyTimeMixerModel(config)
        self.head = TinyTimeMixerForPredictionHead(config)

    def __call__(self, past_values: torch.Tensor, future_values: Optional[torch.Tensor] = None, past_observed_mask: Optional[torch.Tensor] = None):
        model_output, hidden_states, patched_x, loc, scale = self.backbone(past_values, past_observed_mask)
        hidden_features = self.flatten(model_output)  # [bs, n_vars, num_patches * d_model]
        y_hat = self.head(hidden_features, past_values, future_values)
        y_hat = y_hat * scale + loc
        return y_hat</content>
<parameter name="filePath">/home/mahmudsudo/tt-metal/models/tt_tinytimemixer/modeling_tinytimemixer.py