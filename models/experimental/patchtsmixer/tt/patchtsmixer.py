# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from typing import Optional, Tuple


class TtPatchTSMixer:
    def __init__(
        self,
        device,
        input_length: int,
        patch_length: int,
        stride: int,
        num_input_channels: int,
        d_model: int,
        num_blocks: int,
        expansion_factor: float,
        dropout: float,
        num_output_steps: int,
        norm_type: str = "batch",
        channel_mixing: bool = True,
        use_gated_attention: bool = False,
        enable_reconciliation: bool = False,
    ):
        self.device = device
        self.input_length = input_length
        self.patch_length = patch_length
        self.stride = stride
        self.num_input_channels = num_input_channels
        self.d_model = d_model
        self.num_blocks = num_blocks
        self.expansion_factor = expansion_factor
        self.dropout = dropout
        self.num_output_steps = num_output_steps
        self.norm_type = norm_type
        self.channel_mixing = channel_mixing
        self.use_gated_attention = use_gated_attention
        self.enable_reconciliation = enable_reconciliation

        # Calculate number of patches
        self.num_patches = (input_length - patch_length) // stride + 1

        # Patch projection layer
        self.patch_proj_weight = ttnn.create_parameter(
            (num_input_channels * patch_length, d_model),
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            initializer=ttnn.init_methods.kaiming_uniform,
            dtype=ttnn.bfloat16,
        )
        self.patch_proj_bias = ttnn.create_parameter(
            (1, d_model),
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            initializer=ttnn.init_methods.zeros,
            dtype=ttnn.bfloat16,
        )

        # Time mixing MLPs
        self.time_mixing_mlps = []
        for _ in range(num_blocks):
            mlp = {
                "norm": self._create_norm_layer(d_model),
                "fc1_weight": ttnn.create_parameter(
                    (d_model, int(d_model * expansion_factor)),
                    device=self.device,
                    layout=ttnn.TILE_LAYOUT,
                    initializer=ttnn.init_methods.kaiming_uniform,
                    dtype=ttnn.bfloat16,
                ),
                "fc1_bias": ttnn.create_parameter(
                    (1, int(d_model * expansion_factor)),
                    device=self.device,
                    layout=ttnn.TILE_LAYOUT,
                    initializer=ttnn.init_methods.zeros,
                    dtype=ttnn.bfloat16,
                ),
                "fc2_weight": ttnn.create_parameter(
                    (int(d_model * expansion_factor), d_model),
                    device=self.device,
                    layout=ttnn.TILE_LAYOUT,
                    initializer=ttnn.init_methods.kaiming_uniform,
                    dtype=ttnn.bfloat16,
                ),
                "fc2_bias": ttnn.create_parameter(
                    (1, d_model),
                    device=self.device,
                    layout=ttnn.TILE_LAYOUT,
                    initializer=ttnn.init_methods.zeros,
                    dtype=ttnn.bfloat16,
                ),
            }
            self.time_mixing_mlps.append(mlp)

        # Channel mixing MLPs (if enabled)
        self.channel_mixing_mlps = []
        if self.channel_mixing:
            for _ in range(num_blocks):
                mlp = {
                    "norm": self._create_norm_layer(d_model),
                    "fc1_weight": ttnn.create_parameter(
                        (d_model, int(d_model * expansion_factor)),
                        device=self.device,
                        layout=ttnn.TILE_LAYOUT,
                        initializer=ttnn.init_methods.kaiming_uniform,
                        dtype=ttnn.bfloat16,
                    ),
                    "fc1_bias": ttnn.create_parameter(
                        (1, int(d_model * expansion_factor)),
                        device=self.device,
                        layout=ttnn.TILE_LAYOUT,
                        initializer=ttnn.init_methods.zeros,
                        dtype=ttnn.bfloat16,
                    ),
                    "fc2_weight": ttnn.create_parameter(
                        (int(d_model * expansion_factor), d_model),
                        device=self.device,
                        layout=ttnn.TILE_LAYOUT,
                        initializer=ttnn.init_methods.kaiming_uniform,
                        dtype=ttnn.bfloat16,
                    ),
                    "fc2_bias": ttnn.create_parameter(
                        (1, d_model),
                        device=self.device,
                        layout=ttnn.TILE_LAYOUT,
                        initializer=ttnn.init_methods.zeros,
                        dtype=ttnn.bfloat16,
                    ),
                }
                self.channel_mixing_mlps.append(mlp)

        # Gated attention (if enabled)
        if self.use_gated_attention:
            self.gate_weights = ttnn.create_parameter(
                (d_model, d_model),
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
                initializer=ttnn.init_methods.kaiming_uniform,
                dtype=ttnn.bfloat16,
            )
            self.gate_bias = ttnn.create_parameter(
                (1, d_model),
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
                initializer=ttnn.init_methods.zeros,
                dtype=ttnn.bfloat16,
            )

        # Forecasting head
        self.head_weight = ttnn.create_parameter(
            (d_model * self.num_patches, num_output_steps),
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            initializer=ttnn.init_methods.kaiming_uniform,
            dtype=ttnn.bfloat16,
        )
        self.head_bias = ttnn.create_parameter(
            (1, num_output_steps),
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            initializer=ttnn.init_methods.zeros,
            dtype=ttnn.bfloat16,
        )

        # Reconciliation head (if enabled)
        if self.enable_reconciliation:
            self.recon_weight = ttnn.create_parameter(
                (num_output_steps, num_output_steps),
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
                initializer=ttnn.init_methods.kaiming_uniform,
                dtype=ttnn.bfloat16,
            )
            self.recon_bias = ttnn.create_parameter(
                (1, num_output_steps),
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
                initializer=ttnn.init_methods.zeros,
                dtype=ttnn.bfloat16,
            )

    def _create_norm_layer(self, normalized_shape: int):
        if self.norm_type == "batch":
            return ttnn.BatchNorm1d(
                num_features=normalized_shape,
                eps=1e-5,
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
        elif self.norm_type == "instance":
            return ttnn.InstanceNorm1d(
                num_features=normalized_shape,
                eps=1e-5,
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
        else:
            raise ValueError(f"Unsupported norm_type: {self.norm_type}")

    def _unfold_time_series(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # x shape: (batch, num_input_channels, input_length)
        batch_size = x.shape[0]
        seq_len = x.shape[2]

        # Extract patches using sliding window
        patches = []
        for i in range(0, seq_len - self.patch_length + 1, self.stride):
            patch = ttnn.slice(
                x,
                (0, 0, i),
                (batch_size - 1, self.num_input_channels - 1, i + self.patch_length - 1),
                (1, 1, 1),
            )
            patches.append(patch)

        # Concatenate all patches along sequence dimension
        patches_tensor = ttnn.concat(patches, dim=2)  # (batch, channels, total_patched_len)

        # Reshape to separate patches
        new_shape = (
            batch_size,
            self.num_input_channels,
            self.num_patches,
            self.patch_length,
        )
        patches_reshaped = ttnn.reshape(patches_tensor, new_shape)

        # Transpose to (batch, num_patches, channels, patch_length)
        patches_transposed = ttnn.permute(patches_reshaped, (0, 2, 1, 3))

        # Reshape to (batch, num_patches, channels * patch_length)
        patches_flattened = ttnn.reshape(
            patches_transposed,
            (batch_size, self.num_patches, self.num_input_channels * self.patch_length),
        )

        return patches_flattened

    def _apply_linear_projection(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # x shape: (batch, num_patches, input_dim)
        # Project to d_model
        x = ttnn.linear(
            x,
            self.patch_proj_weight,
            bias=self.patch_proj_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            core_grid=None,
        )
        return x

    def _apply_time_mixing(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # x shape: (batch, num_patches, d_model)
        for mlp in self.time_mixing_mlps:
            residual = x
            x = mlp["norm"](x)
            x = ttnn.linear(
                x,
                mlp["fc1_weight"],
                bias=mlp["fc1_bias"],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                core_grid=None,
            )
            x = ttnn.gelu(x)
            x = ttnn.dropout(x, p=self.dropout)
            x = ttnn.linear(
                x,
                mlp["fc2_weight"],
                bias=mlp["fc2_bias"],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                core_grid=None,
            )
            x = ttnn.dropout(x, p=self.dropout)
            x = x + residual
        return x

    def _apply_channel_mixing(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # x shape: (batch, num_patches, d_model)
        if not self.channel_mixing:
            return x

        for mlp in self.channel_mixing_mlps:
            residual = x
            x = mlp["norm"](x)
            x = ttnn.linear(
                x,
                mlp["fc1_weight"],
                bias=mlp["fc1_bias"],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                core_grid=None,
            )
            x = ttnn.gelu(x)
            x = ttnn.dropout(x, p=self.dropout)
            x = ttnn.linear(
                x,
                mlp["fc2_weight"],
                bias=mlp["fc2_bias"],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                core_grid=None,
            )
            x = ttnn.dropout(x, p=self.dropout)
            x = x + residual
        return x

    def _apply_gated_attention(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if not self.use_gated_attention:
            return x

        # Simple gating mechanism
        gate = ttnn.linear(
            x,
            self.gate_weights,
            bias=self.gate_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            core_grid=None,
        )
        gate = ttnn.sigmoid(gate)
        return ttnn.mul(x, gate)

    def _apply_forecasting_head(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # x shape: (batch, num_patches, d_model)
        # Flatten across patches
        batch_size = x.shape[0]
        x = ttnn.reshape(x, (batch_size, self.num_patches * self.d_model))
        # Apply linear head
        x = ttnn.linear(
            x,
            self.head_weight,
            bias=self.head_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            core_grid=None,
        )
        return x

    def _apply_reconciliation_head(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if not self.enable_reconciliation:
            return x

        # Apply reconciliation transformation
        x = ttnn.linear(
            x,
            self.recon_weight,
            bias=self.recon_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            core_grid=None,
        )
        return x

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # Input shape: (batch, num_input_channels, input_length)
        x = self._unfold_time_series(x)
        x = self._apply_linear_projection(x)
        x = self._apply_time_mixing(x)
        x = self._apply_channel_mixing(x)
        x = self._apply_gated_attention(x)
        x = self._apply_forecasting_head(x)
        x = self._apply_reconciliation_head(x)
        return x