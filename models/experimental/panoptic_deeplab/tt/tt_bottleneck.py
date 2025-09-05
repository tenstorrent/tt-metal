# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import torch
import ttnn

from .tt_conv2d_wrapper import (
    TtConv2d,
    TtConv2dParameters,
    SliceConfig,
    SliceMode,
)


class TtBottleneck(nn.Module):
    """
    TTNN implementation of BottleneckBlock.

    Based on the model structure, BottleneckBlock contains:
    - conv1: 1x1 conv (channel reduction)
    - conv2: 3x3 conv (spatial convolution, potentially with stride/dilation)
    - conv3: 1x1 conv (channel expansion)
    - shortcut: optional 1x1 conv for residual connection when input/output dimensions differ
    Each with SyncBatchNorm and ReLU activation, except the final output uses residual addition + ReLU.
    """

    def __init__(
        self,
        device: ttnn.MeshDevice,
        state_dict: dict[str, torch.Tensor],
        dtype: ttnn.DataType = ttnn.bfloat16,
        has_shortcut: bool = False,
        stride: int = 1,
        dilation: int = 1,
        shortcut_stride: int = 1,
        block_id: str = "unknown",
    ):
        super().__init__()
        self.device = device
        self.has_shortcut = has_shortcut
        self.block_id = block_id

        # Extract weights for conv1, conv2, conv3
        conv1_state = {k.replace("conv1.", ""): v for k, v in state_dict.items() if k.startswith("conv1.")}
        conv2_state = {k.replace("conv2.", ""): v for k, v in state_dict.items() if k.startswith("conv2.")}
        conv3_state = {k.replace("conv3.", ""): v for k, v in state_dict.items() if k.startswith("conv3.")}

        # Use passed stride and dilation parameters
        conv2_stride = (stride, stride)
        conv2_dilation = (dilation, dilation)
        conv2_padding = (dilation, dilation)  # Padding should match dilation for 3x3 conv

        # Initialize conv layers
        self.conv1 = TtConv2d(
            TtConv2dParameters.from_torch(conv1_state, device=device, dtype=dtype), stride=(1, 1), padding=(0, 0)
        )

        # For conv2, use architecture parameters
        conv2_params = TtConv2dParameters.from_torch(conv2_state, device=device, dtype=dtype)
        # Update conv2_params with correct dilation
        conv2_params.dilation = conv2_dilation
        self.conv2 = TtConv2d(conv2_params, stride=conv2_stride, padding=conv2_padding)

        self.conv3 = TtConv2d(
            TtConv2dParameters.from_torch(conv3_state, device=device, dtype=dtype), stride=(1, 1), padding=(0, 0)
        )

        # Initialize shortcut if needed
        if has_shortcut:
            shortcut_state = {k.replace("shortcut.", ""): v for k, v in state_dict.items() if k.startswith("shortcut.")}
            shortcut_stride_tuple = (shortcut_stride, shortcut_stride)

            # Apply width slicing for res3 blocks
            shortcut_slice_config = None
            if block_id.startswith("res3"):
                shortcut_slice_config = SliceConfig(mode=SliceMode.WIDTH, num_slices=2)

            self.shortcut = TtConv2d(
                TtConv2dParameters.from_torch(
                    shortcut_state, device=device, dtype=dtype, slice_config=shortcut_slice_config
                ),
                stride=shortcut_stride_tuple,
                padding=(0, 0),
            )

        # Extract normalization parameters
        conv1_norm_state = {
            k.replace("conv1.norm.", ""): v for k, v in state_dict.items() if k.startswith("conv1.norm.")
        }
        conv2_norm_state = {
            k.replace("conv2.norm.", ""): v for k, v in state_dict.items() if k.startswith("conv2.norm.")
        }
        conv3_norm_state = {
            k.replace("conv3.norm.", ""): v for k, v in state_dict.items() if k.startswith("conv3.norm.")
        }

        # Convert normalization parameters to TTNN tensors
        mesh_mapper = ttnn.ReplicateTensorToMesh(device) if isinstance(device, ttnn.MeshDevice) else None

        # Conv1 normalization
        self.conv1_norm_weight = ttnn.from_torch(
            conv1_norm_state["weight"].view(1, -1, 1, 1),
            dtype=dtype,
            device=device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
        )
        self.conv1_norm_bias = ttnn.from_torch(
            conv1_norm_state["bias"].view(1, -1, 1, 1),
            dtype=dtype,
            device=device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
        )
        self.conv1_norm_running_mean = ttnn.from_torch(
            conv1_norm_state["running_mean"].view(1, -1, 1, 1),
            dtype=dtype,
            device=device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
        )
        self.conv1_norm_running_var = ttnn.from_torch(
            conv1_norm_state["running_var"].view(1, -1, 1, 1),
            dtype=dtype,
            device=device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
        )

        # Conv2 normalization
        self.conv2_norm_weight = ttnn.from_torch(
            conv2_norm_state["weight"].view(1, -1, 1, 1),
            dtype=dtype,
            device=device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
        )
        self.conv2_norm_bias = ttnn.from_torch(
            conv2_norm_state["bias"].view(1, -1, 1, 1),
            dtype=dtype,
            device=device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
        )
        self.conv2_norm_running_mean = ttnn.from_torch(
            conv2_norm_state["running_mean"].view(1, -1, 1, 1),
            dtype=dtype,
            device=device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
        )
        self.conv2_norm_running_var = ttnn.from_torch(
            conv2_norm_state["running_var"].view(1, -1, 1, 1),
            dtype=dtype,
            device=device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
        )

        # Conv3 normalization
        self.conv3_norm_weight = ttnn.from_torch(
            conv3_norm_state["weight"].view(1, -1, 1, 1),
            dtype=dtype,
            device=device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
        )
        self.conv3_norm_bias = ttnn.from_torch(
            conv3_norm_state["bias"].view(1, -1, 1, 1),
            dtype=dtype,
            device=device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
        )
        self.conv3_norm_running_mean = ttnn.from_torch(
            conv3_norm_state["running_mean"].view(1, -1, 1, 1),
            dtype=dtype,
            device=device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
        )
        self.conv3_norm_running_var = ttnn.from_torch(
            conv3_norm_state["running_var"].view(1, -1, 1, 1),
            dtype=dtype,
            device=device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
        )

        # Shortcut normalization (if exists)
        if has_shortcut:
            shortcut_norm_state = {
                k.replace("shortcut.norm.", ""): v for k, v in state_dict.items() if k.startswith("shortcut.norm.")
            }
            self.shortcut_norm_weight = ttnn.from_torch(
                shortcut_norm_state["weight"].view(1, -1, 1, 1),
                dtype=dtype,
                device=device,
                mesh_mapper=mesh_mapper,
                layout=ttnn.TILE_LAYOUT,
            )
            self.shortcut_norm_bias = ttnn.from_torch(
                shortcut_norm_state["bias"].view(1, -1, 1, 1),
                dtype=dtype,
                device=device,
                mesh_mapper=mesh_mapper,
                layout=ttnn.TILE_LAYOUT,
            )
            self.shortcut_norm_running_mean = ttnn.from_torch(
                shortcut_norm_state["running_mean"].view(1, -1, 1, 1),
                dtype=dtype,
                device=device,
                mesh_mapper=mesh_mapper,
                layout=ttnn.TILE_LAYOUT,
            )
            self.shortcut_norm_running_var = ttnn.from_torch(
                shortcut_norm_state["running_var"].view(1, -1, 1, 1),
                dtype=dtype,
                device=device,
                mesh_mapper=mesh_mapper,
                layout=ttnn.TILE_LAYOUT,
            )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        print(f"[BOTTLENECK {self.block_id}] Starting forward pass, input shape: {x.shape}")

        # Store input for residual connection
        identity = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        # workaround for conv tilize issue with non-height shard
        if identity.spec.layout != ttnn.TILE_LAYOUT:
            identity = ttnn.tilize(identity)
        # ttnn.deallocate(x)

        # Process shortcut if needed
        if self.has_shortcut:
            print(f"[BOTTLENECK {self.block_id}] Processing shortcut convolution...")
            identity = self.shortcut(identity)
            identity = ttnn.to_memory_config(identity, ttnn.DRAM_MEMORY_CONFIG)
            # Convert NHWC to NCHW for batch_norm
            identity = ttnn.permute(identity, (0, 3, 1, 2))
            print(f"[BOTTLENECK {self.block_id}] Applying shortcut batch norm...")
            identity = ttnn.batch_norm(
                identity,
                running_mean=self.shortcut_norm_running_mean,
                running_var=self.shortcut_norm_running_var,
                weight=self.shortcut_norm_weight,
                bias=self.shortcut_norm_bias,
                eps=1e-05,
                training=False,
            )
            # Convert back to NHWC
            identity = ttnn.permute(identity, (0, 2, 3, 1))
            print(f"[BOTTLENECK {self.block_id}] Shortcut processing complete")

        # Main path: Conv1 + BatchNorm + ReLU
        print(f"[BOTTLENECK {self.block_id}] Processing conv1 (1x1 reduction)...")
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        out = self.conv1(x)
        # Convert NHWC to NCHW for batch_norm
        out = ttnn.permute(out, (0, 3, 1, 2))
        print(f"[BOTTLENECK {self.block_id}] Applying conv1 batch norm...")
        out = ttnn.batch_norm(
            out,
            running_mean=self.conv1_norm_running_mean,
            running_var=self.conv1_norm_running_var,
            weight=self.conv1_norm_weight,
            bias=self.conv1_norm_bias,
            eps=1e-05,
            training=False,
        )
        # Convert back to NHWC
        out = ttnn.permute(out, (0, 2, 3, 1))
        out = ttnn.relu(out)
        out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
        print(f"[BOTTLENECK {self.block_id}] Conv1 processing complete, output shape: {out.shape}")

        # Conv2 + BatchNorm + ReLU
        print(f"[BOTTLENECK {self.block_id}] Processing conv2 (3x3 spatial)...")
        out = self.conv2(out)
        # Convert NHWC to NCHW for batch_norm
        # out = ttnn.to_memory_config(out, ttnn.L1_MEMORY_CONFIG)
        out = ttnn.permute(out, (0, 3, 1, 2))
        print(f"[BOTTLENECK {self.block_id}] Applying conv2 batch norm...")
        out = ttnn.batch_norm(
            out,
            running_mean=self.conv2_norm_running_mean,
            running_var=self.conv2_norm_running_var,
            weight=self.conv2_norm_weight,
            bias=self.conv2_norm_bias,
            eps=1e-05,
            training=False,
        )
        # Convert back to NHWC
        out = ttnn.permute(out, (0, 2, 3, 1))
        out = ttnn.relu(out)
        out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
        print(f"[BOTTLENECK {self.block_id}] Conv2 processing complete, output shape: {out.shape}")

        # Conv3 + BatchNorm (no ReLU yet)
        print(f"[BOTTLENECK {self.block_id}] Processing conv3 (1x1 expansion) - THIS IS WHERE PCC MIGHT FAIL...")
        out = self.conv3(out)
        print(f"[BOTTLENECK {self.block_id}] Conv3 convolution complete, output shape: {out.shape}")
        # Convert NHWC to NCHW for batch_norm
        out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.permute(out, (0, 3, 1, 2))
        print(f"[BOTTLENECK {self.block_id}] Applying conv3 batch norm...")
        out = ttnn.batch_norm(
            out,
            running_mean=self.conv3_norm_running_mean,
            running_var=self.conv3_norm_running_var,
            weight=self.conv3_norm_weight,
            bias=self.conv3_norm_bias,
            eps=1e-05,
            training=False,
        )
        # Convert back to NHWC
        out = ttnn.permute(out, (0, 2, 3, 1))
        print(f"[BOTTLENECK {self.block_id}] Conv3 processing complete, output shape: {out.shape}")

        # Residual connection + ReLU
        print(f"[BOTTLENECK {self.block_id}] Adding residual connection and applying final ReLU...")
        if self.has_shortcut or identity.shape == out.shape:
            out = ttnn.add(out, identity)
        out = ttnn.relu(out)
        out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)

        print(f"[BOTTLENECK {self.block_id}] Forward pass complete, final output shape: {out.shape}")
        return out
