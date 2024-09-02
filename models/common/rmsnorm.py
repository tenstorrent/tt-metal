# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import ttnn
from models.common.lightweightmodule import LightweightModule


TILE = 32
SHARD_HEIGHT = TILE  # Current ttnn.rms_norm implementation requires shard height to be a single tile


class RMSNorm(LightweightModule):
    """
    RMSNorm supporting replication over a MeshDevice and sharding within devices.

    This class implements a Root Mean Square Normalization (RMSNorm) that can be
    distributed across multiple devices and cores. If the `device` parameter is a
    MeshDevice, the weights and computations are replicated across all devices in
    the mesh. Expects an interleaved input tensor, can optionally output a sharded tensor.

    Args:
        device: The device or MeshDevice on which to perform the computations.
        state_dict: The state dictionary containing the model parameters.
        dim: Input dimension (e.g. model hidden dimension size).
        layer_num: The layer number to determine the weight key in the state dictionary.
        weight_key: The key for retrieving the weight from the state dictionary.
        weight_cache_path: Optional path for caching the tilized weights.
        weight_memory_config: Configuration for the weight memory, default is DRAM_MEMORY_CONFIG.
        weight_dtype: The data type for the tensors, bfp8_b hits >0.999 PCC in the models we tested.
        model_config: Optional configuration dictionary for the model.
        is_sharded: Sharded version is faster for some models but doesn't support all batch sizes.
        eps (float): Small value to avoid division by zero in normalization, default is 1e-05.

    If model_config is provided, it must specify SHARDED_NORM_INPUT_MEMCFG, SHARDED_NORM_PRGM_CFG
    and SHARDED_NORM_OUTPUT_MEMCFG. If not provided, default configurations will be generated.
    """

    def __init__(
        self,
        device,
        dim,
        state_dict,
        weight_key,
        layer_num=None,
        weight_cache_path=None,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        weight_dtype=ttnn.bfloat8_b,
        model_config=None,
        is_sharded=False,
        eps: float = 1e-05,
    ):
        super().__init__()
        self.eps = eps
        self.is_sharded = is_sharded

        if layer_num is None:
            weight_name = f"{weight_key}.weight"
        else:
            weight_name = f"layers.{layer_num}.{weight_key}.weight"

        torch_weight = state_dict[weight_name].unsqueeze(0).view(1, 1, dim).expand([1, SHARD_HEIGHT, dim])
        cache_name = None if weight_cache_path is None else weight_cache_path / weight_name

        is_mesh_device = device.__class__.__name__ == "MeshDevice"
        self.weight = ttnn.as_tensor(
            torch_weight,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
            cache_file_name=cache_name,
            mesh_mapper=ttnn.ShardTensorToMesh(device, dim=2) if is_mesh_device else None,
        )

        if is_sharded:
            if model_config:
                self.input_config = model_config["SHARDED_NORM_INPUT_MEMCFG"]
                self.program_config = model_config["SHARDED_NORM_PRGM_CFG"]
                self.output_config = model_config["SHARDED_NORM_OUTPUT_MEMCFG"]
            else:
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
            x = ttnn.interleaved_to_sharded(
                x,
                self.input_config,
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
            x_interleaved = ttnn.sharded_to_interleaved(x)
            x.deallocate(True)
            return x_interleaved
        else:  # Interleaved rmsnorm does not need program or memory configs
            assert not out_sharded, "Non-sharded version of RMSNorm cannot output a sharded tensor"
            x = ttnn.rms_norm(x, weight=self.weight, epsilon=self.eps)
            return x
