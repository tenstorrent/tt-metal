"""
Layer norm for the Janus-Pro-7B vision tower.

The shard is sized from the sequence rather than pinned to one tile-row, so its core grid is
whatever the shape allows -- which is what lets the projection downstream read the result in
place. See `ModelArgs.vision_norm_shard_configs`.
"""

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtJanusProLayerNorm(LightweightModule):
    def __init__(
        self,
        device,
        dim,
        state_dict,
        state_dict_prefix,
        configuration,
        weight_cache_path=None,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        weight_dtype=ttnn.bfloat8_b,
        eps: float = 1e-05,
    ):
        super().__init__()
        self.configuration = configuration
        self.eps = eps

        # Broadcast to a full tile row: ttnn.layer_norm reads weight and bias as tiles, so a
        # single row of `dim` values would occupy 1/32 of the tile the kernel loads.
        torch_weight = state_dict[f"{state_dict_prefix}weight"].unsqueeze(0).view(1, 1, dim).expand([1, 32, dim])
        torch_bias = state_dict[f"{state_dict_prefix}bias"].unsqueeze(0).view(1, 1, dim).expand([1, 32, dim])
        if weight_cache_path is None:
            cache_name = lambda *_: None
        else:
            cache_name = lambda suffix: weight_cache_path / (state_dict_prefix + f"{suffix}")

        is_mesh_device = device.__class__.__name__ == "MeshDevice"
        self.weight = ttnn.as_tensor(
            torch_weight,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
            cache_file_name=cache_name("weight"),
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
        )

        self.bias = ttnn.as_tensor(
            torch_bias,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
            cache_file_name=cache_name("bias"),
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
        )

        # The shard is sized on the first forward, not here: its grid follows the sequence
        # length, which __init__ does not see. A one-tile-row shard would be 18x too small
        # for 576 tokens.
        self.shard_shape = None
        self.shard_config = self.shard_program_config = None

    def forward(self, x: ttnn.Tensor, out_sharded=False) -> ttnn.Tensor:
        if (x.shape[-2], x.shape[-1]) != self.shard_shape:
            self.shard_shape = (x.shape[-2], x.shape[-1])
            self.shard_config, self.shard_program_config = self.configuration.vision_norm_shard_configs(
                *self.shard_shape
            )

        if self.shard_program_config is None:
            assert not out_sharded, f"no sharded layer-norm config for shape {self.shard_shape}"
            return ttnn.layer_norm(
                x,
                weight=self.weight,
                bias=self.bias,
                epsilon=self.eps,
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    math_approx_mode=False,
                    fp32_dest_acc_en=False,
                    packer_l1_acc=False,
                ),
            )

        # to_memory_config hands back the input unchanged when the config already matches
        # (to_memory_config_op.cpp:258), so only a shard this call created may be freed.
        already_sharded = x.memory_config() == self.shard_config
        x_sharded = ttnn.to_memory_config(x, self.shard_config)
        normed = ttnn.layer_norm(
            x_sharded,
            epsilon=self.eps,
            weight=self.weight,
            bias=self.bias,
            program_config=self.shard_program_config,
            memory_config=self.shard_config,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4),
        )
        if not already_sharded:
            ttnn.deallocate(x_sharded)
        if out_sharded:
            return normed

        normed_interleaved = ttnn.sharded_to_interleaved(normed)
        normed.deallocate(True)
        return normed_interleaved
