# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import math

from torch import nn

import ttnn
from models.demos.gpt_oss.config import MeshConfig, ModeConfig
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name


class RMSNorm(nn.Module):
    """
    RMSNorm with support for distributed computation on 2D mesh.

    Uses true distributed RMS norm when:
    1. is_distributed=True
    2. Input is sharded (per-device width)

    For non-tile-aligned per-device sizes (e.g., 360 with TP=8), the input is
    padded to tile alignment and num_elements_per_device is passed to
    rms_norm_post_all_gather to compute correct normalization statistics.

    For non-sharded (full-width) input, uses regular RMS norm with replicated weights.
    """

    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        tensor_cache_path=None,
        mesh_config=None,
        is_distributed=False,
        ccl_manager=None,
    ):
        super().__init__()
        torch_weight = state_dict["weight"]
        # Keep a host copy for debug comparisons.
        self.torch_weight = torch_weight.detach().clone()

        # Use MeshConfig for clean parallelization
        self.mesh_config = mesh_config or MeshConfig(mesh_device.shape, decode=ModeConfig(tp=mesh_device.shape[1]))
        self.mesh_shape = mesh_device.shape  # Store mesh shape for distributed forward
        self.is_distributed = is_distributed  # Enable distributed RMS norm when activations are TP sharded
        self.ccl_manager = ccl_manager  # CCL manager for distributed all_gather

        hidden_size = hf_config.hidden_size
        tp = self.mesh_config.tp

        self.eps = hf_config.rms_norm_eps
        self.mesh_device = mesh_device
        self.hidden_size = hidden_size

        # Check if hidden_size_per_device is tile-aligned
        hidden_size_per_device = hidden_size // tp if tp > 1 else hidden_size
        tiles_per_device = math.ceil(hidden_size_per_device / ttnn.TILE_SIZE)
        padded_hidden_per_device = tiles_per_device * ttnn.TILE_SIZE
        self.is_tile_aligned = hidden_size_per_device % ttnn.TILE_SIZE == 0

        # Store dimensions for forward pass
        self.hidden_size_per_device = hidden_size_per_device
        self.padded_hidden_per_device = padded_hidden_per_device
        self.needs_padding = not self.is_tile_aligned

        # Track whether output is gathered (for callers to know)
        # This is set dynamically in forward based on which path is used
        self._last_output_gathered = False

        # Create full-width replicated weight for full-width input case
        self.tt_weight_full = ttnn.as_tensor(
            torch_weight.reshape((1, 1, -1, ttnn.TILE_SIZE)),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            cache_file_name=get_cache_file_name(tensor_cache_path, "weight"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        # Create sharded weight for per-device cases
        self.tt_weight_sharded = None
        self.tt_weight_distributed = None

        if self.is_distributed and tp > 1:
            import torch.nn.functional as F

            weight_mesh_mapper = self.mesh_config.column_parallel(mesh_device)

            if self.is_tile_aligned:
                # Tile-aligned case - create distributed weight for distributed norm
                tiles_per_device_actual = hidden_size_per_device // ttnn.TILE_SIZE

                # Reshape weight for column_parallel sharding
                weight_per_device = torch_weight.reshape(tp, hidden_size_per_device)
                weight_tiled = weight_per_device.reshape(tp, tiles_per_device_actual, ttnn.TILE_SIZE)
                weight_transposed = weight_tiled.permute(1, 0, 2)
                weight_for_tt = weight_transposed.reshape(1, 1, tiles_per_device_actual, tp * ttnn.TILE_SIZE)

                self.tt_weight_distributed = ttnn.as_tensor(
                    weight_for_tt,
                    device=mesh_device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    cache_file_name=get_cache_file_name(
                        tensor_cache_path, f"weight_distributed_{tiles_per_device_actual}"
                    ),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=weight_mesh_mapper,
                )
            else:
                # Non-tile-aligned case - create PADDED distributed weight for distributed norm
                # This allows us to use true distributed norm with num_elements_per_device
                # to compute correct statistics even with padded tensors

                # Split weights per device and pad each shard
                weight_per_device = torch_weight.reshape(tp, hidden_size_per_device)
                # Pad each shard from 360 to 384 (pad with ones since these are gamma weights)
                padding_size = padded_hidden_per_device - hidden_size_per_device
                # For RMSNorm, gamma=1 is identity, so pad with 1s
                weight_per_device_padded = F.pad(weight_per_device, (0, padding_size), value=1.0)

                # Reshape for distributed norm: [1, 1, tiles, tp * 32]
                weight_tiled = weight_per_device_padded.reshape(tp, tiles_per_device, ttnn.TILE_SIZE)
                weight_transposed = weight_tiled.permute(1, 0, 2)
                weight_for_tt = weight_transposed.reshape(1, 1, tiles_per_device, tp * ttnn.TILE_SIZE)

                self.tt_weight_distributed = ttnn.as_tensor(
                    weight_for_tt,
                    device=mesh_device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    cache_file_name=get_cache_file_name(
                        tensor_cache_path, f"weight_distributed_padded_{tiles_per_device}"
                    ),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=weight_mesh_mapper,
                )

    @property
    def output_is_gathered(self):
        """Return whether the last forward pass produced gathered (full-width) output."""
        return self._last_output_gathered

    def _is_input_sharded(self, x):
        """Check if input tensor is sharded (per-device width) vs full width."""
        # Get the input's last dimension (width)
        input_width = x.shape[-1]
        # If input width is close to padded per-device size, it's sharded
        # If input width is close to full hidden size, it's full width
        return input_width <= self.padded_hidden_per_device

    def forward(self, x):
        # Check if input is actually sharded (per-device width) or full width
        input_is_sharded = self._is_input_sharded(x)

        # Use distributed norm when:
        # 1. is_distributed=True
        # 2. Input is sharded (per-device width)
        # 3. We have the distributed weight tensor
        # Note: For non-tile-aligned sizes, we pad input and use num_elements_per_device
        use_distributed = (
            self.is_distributed
            and self.mesh_config.tp > 1
            and input_is_sharded
            and self.tt_weight_distributed is not None
        )

        if use_distributed:
            # True distributed norm path - input is sharded
            # For non-tile-aligned sizes, we need to pad input first
            x_for_norm = x
            if self.needs_padding:
                # Pad input from hidden_size_per_device (360) to padded_hidden_per_device (384)
                # Use ttnn.pad to add zeros at the end of the last dimension
                input_shape = x.shape
                padded_shape = list(input_shape)
                padded_shape[-1] = self.padded_hidden_per_device
                x_for_norm = ttnn.pad(x, padded_shape, [0, 0, 0, 0], 0.0)

            # Run distributed rmsnorm part 1: compute local stats
            # For non-tile-aligned widths, use an interleaved view for stats to
            # match the distributed pre_all_gather behavior.
            x_for_stats = x_for_norm
            x_for_norm_interleaved = None
            if self.needs_padding:
                x_for_norm_interleaved = ttnn.to_memory_config(x_for_norm, ttnn.DRAM_MEMORY_CONFIG)
                x_for_stats = x_for_norm_interleaved
            tt_stats = ttnn.rms_norm_pre_all_gather(x_for_stats, dtype=ttnn.bfloat16)

            # Reshape stats for all_gather (match tt_transformers pattern)
            padded_shape = (1, 1, x_for_norm.shape[-2], 32)
            tt_stats = ttnn.reshape(tt_stats, ttnn.Shape(padded_shape))

            # AllGather stats across columns (cluster_axis=1)
            tt_gathered_stats = ttnn.experimental.all_gather_async(
                tt_stats,
                dim=3,
                cluster_axis=1,
                mesh_device=self.mesh_device,
                topology=ttnn.Topology.Ring,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(),
                num_links=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                barrier_semaphore=self.ccl_manager.get_barrier_semaphore(),
            )
            ttnn.deallocate(tt_stats)

            # Run distributed rmsnorm part 2: apply normalization with gathered stats
            # For non-tile-aligned sizes, pass num_elements_per_device to compute correct stats
            if self.needs_padding:
                tt_output = ttnn.rms_norm_post_all_gather(
                    x_for_norm_interleaved,
                    epsilon=self.eps,
                    weight=self.tt_weight_distributed,
                    stats=tt_gathered_stats,
                    num_elements_per_device=self.hidden_size_per_device,
                )
                ttnn.deallocate(x_for_norm_interleaved)
                ttnn.deallocate(x_for_norm)
            else:
                tt_output = ttnn.rms_norm_post_all_gather(
                    x_for_norm,
                    epsilon=self.eps,
                    weight=self.tt_weight_distributed,
                    stats=tt_gathered_stats,
                )
            ttnn.deallocate(tt_gathered_stats)

            # For non-tile-aligned, slice output back to original size
            if self.needs_padding:
                output_shape = list(tt_output.shape)
                output_shape[-1] = self.hidden_size_per_device
                tt_output = ttnn.slice(tt_output, [0, 0, 0, 0], output_shape)
                # Return sharded output to keep residual stream sharding.
                tt_output = ttnn.to_memory_config(tt_output, x.memory_config())

            # Output is sharded (per-device width), not gathered
            self._last_output_gathered = False
            return tt_output
        else:
            # Input is full width - use regular rms_norm with full weight
            tt_output = ttnn.rms_norm(
                x,
                weight=self.tt_weight_full,
                epsilon=self.eps,
            )
            # Output is full width
            self._last_output_gathered = True
            return tt_output
