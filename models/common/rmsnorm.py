# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import ttnn
from models.common.modules import LightweightModule, WeightSetting


TILE = 32
SHARD_HEIGHT = TILE  # Current ttnn.rms_norm implementation requires shard height to be a single tile


class RMSNorm(LightweightModule):
    """
    RMSNorm supporting replication over a DeviceMesh and sharding within devices.

    This class implements a Root Mean Square Normalization (RMSNorm) that can be
    distributed across multiple devices and cores. If the `device` parameter is a
    DeviceMesh, the weights and computations are replicated across all devices in
    the mesh. Expects an interleaved input tensor, can optionally output a sharded tensor.

    Args:
        device: The device or DeviceMesh on which to perform the computations.
        dim: Input dimension (e.g. model hidden dimension size).
        state_dict: The state dictionary containing the model parameters.
        state_dict_prefix: Prefix to the
        weight_cache_path: Optional path for caching the tilized weights.
        is_sharded: Sharded version is faster for some models but doesn't support all batch sizes.
        eps (float): Small value to avoid division by zero in normalization, default is 1e-05.
    """

    def __init__(
        self,
        device,
        dim,
        state_dict,
        state_dict_prefix,
        weight_cache_path=None,
        is_sharded=False,
        eps=1e-05,
    ):
        super().__init__(device)
        self.eps = eps
        self.is_sharded = is_sharded
        self.weight_settings = {
            "weight": WeightSetting(
                state_dict_prefix + ".weight",
                ttnn.bfloat8_b,
                conversion_fn=lambda t: t.unsqueeze(0).view(1, 1, dim).expand([1, SHARD_HEIGHT, dim]),
                mapper=ttnn.ReplicateTensorToMesh(device) if self.is_device_mesh else None,
            ),
        }

        self.load_weights(state_dict, weight_cache_path)

        if is_sharded:
            assert (
                dim % SHARD_HEIGHT == 0
            ), f"Input dimension dim ({dim}) must be a multiple of SHARD_HEIGHT ({SHARD_HEIGHT})"
            shard_width_hidden_dim_across_32_cores = dim // SHARD_HEIGHT
            core_grid = ttnn.CoreGrid(x=8, y=SHARD_HEIGHT // 8)
            self.input_config = ttnn.create_sharded_memory_config(
                shape=(SHARD_HEIGHT, shard_width_hidden_dim_across_32_cores),
                core_grid=core_grid,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=[core_grid.x, core_grid.y],
                subblock_w=shard_width_hidden_dim_across_32_cores // TILE,
                block_h=SHARD_HEIGHT // TILE,
                block_w=shard_width_hidden_dim_across_32_cores // TILE,
                inplace=False,
            )
            self.output_config = self.input_config

    def forward(self, x: ttnn.Tensor, out_sharded=False) -> ttnn.Tensor:
        if self.is_sharded:  # sharded version converts from interleaved inputs and optionally back
            x = ttnn.experimental.tensor.interleaved_to_sharded(
                x,
                sharded_mem_config=self.input_config,
            )
            x = ttnn.rms_norm(
                x,
                epsilon=self.eps,
                weight=self.weight,
                program_config=self.program_config,
                memory_config=self.output_config,
            )
            if out_sharded:
                return x
            x_interleaved = ttnn.experimental.tensor.sharded_to_interleaved(x)
            x.deallocate(True)
            return x_interleaved
        else:  # Interleaved rmsnorm does not need program or memory configs
            assert not out_sharded, "Non-sharded version of RMSNorm cannot output a sharded tensor"
            x = ttnn.rms_norm(x, weight=self.weight, epsilon=self.eps)
            return x
