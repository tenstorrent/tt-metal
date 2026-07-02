# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from typing import Optional, Tuple
from .configuration_tinytimemixer import TinyTimeMixerConfig


class TinyTimeMixerNormLayer:
    def __init__(self, config: TinyTimeMixerConfig, device, state_dict, layer_name):
        self.weight = ttnn.from_torch(state_dict[f"{layer_name}.norm.weight"], device=device)
        self.bias = ttnn.from_torch(state_dict[f"{layer_name}.norm.bias"], device=device)
        self.norm_eps = config.norm_eps

    def __call__(self, inputs: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.layer_norm(inputs, epsilon=self.norm_eps, weight=self.weight, bias=self.bias)

class TinyTimeMixerMLP:
    def __init__(self, in_features, out_features, config, device, state_dict, layer_name):
        num_hidden = in_features * config.expansion_factor
        self.fc1_weight = ttnn.from_torch(state_dict[f"{layer_name}.fc1.weight"], device=device)
        self.fc1_bias = ttnn.from_torch(state_dict[f"{layer_name}.fc1.bias"], device=device)
        self.fc2_weight = ttnn.from_torch(state_dict[f"{layer_name}.fc2.weight"], device=device)
        self.fc2_bias = ttnn.from_torch(state_dict[f"{layer_name}.fc2.bias"], device=device)
        self.device = device

    def __call__(self, inputs: ttnn.Tensor) -> ttnn.Tensor:
        inputs = ttnn.linear(inputs, self.fc1_weight, bias=self.fc1_bias)
        inputs = ttnn.gelu(inputs)
        inputs = ttnn.linear(inputs, self.fc2_weight, bias=self.fc2_bias)
        return inputs


class TinyTimeMixerGatedAttention:
    def __init__(self, in_size: int, out_size: int, device, state_dict, layer_name):
        self.attn_weight = ttnn.from_torch(state_dict[f"{layer_name}.attn_layer.weight"], device=device)
        self.attn_bias = ttnn.from_torch(state_dict[f"{layer_name}.attn_layer.bias"], device=device)
        self.device = device

    def __call__(self, inputs: ttnn.Tensor) -> ttnn.Tensor:
        attn_weight = ttnn.linear(inputs, self.attn_weight, bias=self.attn_bias)
        attn_weight = ttnn.softmax(attn_weight, dim=-1)
        inputs = inputs * attn_weight
        return inputs


class PatchMixerBlock:
    def __init__(self, config: TinyTimeMixerConfig, device, state_dict, layer_name):
        self.norm = TinyTimeMixerNormLayer(config, device, state_dict, layer_name)
        self.mlp = TinyTimeMixerMLP(
            in_features=config.num_patches,
            out_features=config.num_patches,
            config=config,
            device=device,
            state_dict=state_dict,
            layer_name=f"{layer_name}.mlp"
        )
        if config.gated_attn:
            self.gating_block = TinyTimeMixerGatedAttention(config.num_patches, config.num_patches, device, state_dict, f"{layer_name}.gating_block")

    def __call__(self, hidden: ttnn.Tensor) -> ttnn.Tensor:
        # hidden: [batch, n_vars, num_patches, d_model]
        residual = hidden
        hidden = self.norm(hidden)
        # Transpose for patch mixing: [batch, n_vars, d_model, num_patches]
        hidden = ttnn.permute(hidden, (0, 1, 3, 2))
        hidden = self.mlp(hidden)
        if hasattr(self, 'gating_block'):
            hidden = self.gating_block(hidden)
        # Transpose back: [batch, n_vars, num_patches, d_model]
        hidden = ttnn.permute(hidden, (0, 1, 3, 2))
        return hidden + residual

class FeatureMixerBlock:
    def __init__(self, config: TinyTimeMixerConfig, device, state_dict, layer_name):
        self.norm = TinyTimeMixerNormLayer(config, device, state_dict, layer_name)
        self.mlp = TinyTimeMixerMLP(
            in_features=config.d_model,
            out_features=config.d_model,
            config=config,
            device=device,
            state_dict=state_dict,
            layer_name=f"{layer_name}.mlp"
        )
        if config.gated_attn:
            self.gating_block = TinyTimeMixerGatedAttention(config.d_model, config.d_model, device, state_dict, f"{layer_name}.gating_block")

    def __call__(self, hidden: ttnn.Tensor) -> ttnn.Tensor:
        residual = hidden
        hidden = self.norm(hidden)
        hidden = self.mlp(hidden)
        if hasattr(self, 'gating_block'):
            hidden = self.gating_block(hidden)
        out = hidden + residual
        return out


class TinyTimeMixerLayer:
    def __init__(self, config: TinyTimeMixerConfig, device, state_dict, layer_name):
        self.num_patches = config.num_patches
        if self.num_patches > 1:
            self.patch_mixer = PatchMixerBlock(config, device, state_dict, f"{layer_name}.patch_mixer")
        self.feature_mixer = FeatureMixerBlock(config, device, state_dict, f"{layer_name}.feature_mixer")

    def __call__(self, hidden: ttnn.Tensor) -> ttnn.Tensor:
        if self.num_patches > 1:
            hidden = self.patch_mixer(hidden)
        hidden = self.feature_mixer(hidden)
        return hidden


class TinyTimeMixerBlock:
    def __init__(self, config: TinyTimeMixerConfig, device, state_dict, layer_name):
        self.mixers = [TinyTimeMixerLayer(config, device, state_dict, f"{layer_name}.mixers.{i}") for i in range(config.num_layers)]

    def __call__(self, hidden_state: ttnn.Tensor) -> ttnn.Tensor:
        embedding = hidden_state
        for mod in self.mixers:
            embedding = mod(embedding)
        return embedding


class TinyTimeMixerStdScalerTTNN:
    def __init__(self, config: TinyTimeMixerConfig, device):
        self.device = device
        self.dim = 1
        self.keepdim = True
        self.minimum_scale = 1e-5

    def __call__(self, data: ttnn.Tensor, observed_indicator: ttnn.Tensor) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        denominator = ttnn.sum(observed_indicator, self.dim, keepdim=self.keepdim)
        denominator = ttnn.maximum(denominator, ttnn.from_torch(torch.tensor(1.0), device=self.device)) # clamp_min

        loc = ttnn.sum(ttnn.multiply(data, observed_indicator), self.dim, keepdim=self.keepdim)
        loc = ttnn.divide(loc, denominator)

        variance = ttnn.sum(ttnn.square(ttnn.multiply(ttnn.subtract(data, loc), observed_indicator)), self.dim, keepdim=self.keepdim)
        variance = ttnn.divide(variance, denominator)
        
        scale = ttnn.sqrt(ttnn.add(variance, ttnn.from_torch(torch.tensor(self.minimum_scale), device=self.device)))
        
        return ttnn.divide(ttnn.subtract(data, loc), scale), loc, scale


class TinyTimeMixerPatchifyTTNN:
    def __init__(self, config: TinyTimeMixerConfig, device):
        self.device = device
        self.patch_length = config.patch_length
        self.patch_stride = config.patch_stride

    def __call__(self, past_values: ttnn.Tensor) -> ttnn.Tensor:
        # past_values: [batch_size, context_length, num_channels]
        # We need to slide over the context_length dimension (dim 1)
        # First, permute to put the sliding dimension last, as required by sliding_window
        # Shape: [batch_size, num_channels, context_length]
        permuted_values = ttnn.permute(past_values, (0, 2, 1))

        # Use sliding_window to create patches
        # output shape: [batch_size, num_channels, num_patches, patch_length]
        patches = ttnn.experimental.tensor.sliding_window(
            permuted_values,
            window_size=self.patch_length,
            stride=self.patch_stride,
            dim=2
        )
        return patches


class TinyTimeMixerForPredictionTTNN:
    def __init__(self, config: TinyTimeMixerConfig, device, state_dict):
        self.config = config
        self.device = device
        
        # Scaler and Patcher
        self.scaler = TinyTimeMixerStdScalerTTNN(config, device)
        self.patcher = TinyTimeMixerPatchifyTTNN(config, device)

        # Load weights
        # Note: The original HF model has 'backbone.patcher' but it seems to be for adaptive patching.
        # The linear projection after fixed patching is 'backbone.encoder.patcher'.
        self.patch_projection_weight = ttnn.from_torch(state_dict["backbone.encoder.patcher.weight"], device=device)
        self.patch_projection_bias = ttnn.from_torch(state_dict["backbone.encoder.patcher.bias"], device=device)
        self.mlp_mixer_encoder = TinyTimeMixerBlock(config, device, state_dict, "backbone.encoder.mlp_mixer_encoder")
        self.head_weight = ttnn.from_torch(state_dict["head.base_forecast_block.weight"], device=device)
        self.head_bias = ttnn.from_torch(state_dict["head.base_forecast_block.bias"], device=device)

    def __call__(self, past_values: torch.Tensor) -> torch.Tensor:
        # 0. To TTNN
        past_values_tt = ttnn.from_torch(past_values, device=self.device, layout=ttnn.TILE_LAYOUT)
        
        # 1. Scaling
        observed_indicator = ttnn.ones_like(past_values_tt)
        scaled_past_values, loc, scale = self.scaler(past_values_tt, observed_indicator)

        # 2. Patchify
        patched_x = self.patcher(scaled_past_values)
        
        # 3. Patch Projection
        patches = ttnn.linear(patched_x, self.patch_projection_weight, bias=self.patch_projection_bias)
        
        # 4. Mixer
        embedding = self.mlp_mixer_encoder(patches)
        
        # 5. Flatten
        hidden_features = ttnn.reshape(embedding, (embedding.shape[0], embedding.shape[1], embedding.shape[2] * embedding.shape[3]))
        
        # 6. Head
        forecast = ttnn.linear(hidden_features, self.head_weight, bias=self.head_bias)
        forecast = ttnn.permute(forecast, (0, 2, 1))
        
        # 7. Reverse Scaling
        forecast = ttnn.add(ttnn.multiply(forecast, scale), loc)
        
        # To torch
        return ttnn.to_torch(forecast)