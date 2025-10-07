# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from pathlib import Path
from typing import Any, final

import torch
import ttnn.experimental
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.tt.ccl_1d import CCL1D
from models.demos.deepseek_v3.utils.abstract_module import AbstractModule
from models.demos.deepseek_v3.utils.config_dataclass import (
    AllGatherAsyncConfig,
    FromWeightConfig,
    LinearConfig,
    MeshDeviceStub,
    MulConfig,
    OpConfigBase,
    ReduceScatterAsyncConfig,
    ReshardConfig,
)
from models.demos.deepseek_v3.utils.config_helpers import (
    COMPUTE_KERNEL_CONFIG_LOFI,
    MAX_BATCH_SIZE,
    SEQ_LEN_CHUNK_SIZE,
    dram_sharded_weight_config,
    even_int_div,
    find_largest_divisor,
    get_activation_sharding_core_counts_for_dram_matmul,
    get_dram_sharded_matmul_config,
    get_state_dicts,
    save_and_get_path,
)
from models.demos.deepseek_v3.utils.run_config import (
    MESH_DEVICE_STATE_DICT_KEY,
    ModelDecodeConfig,
    ModelPrefillConfig,
    ModelState,
    RunDecodeConfig,
    RunPrefillConfig,
    WeightConfig,
)


class MLP1D(AbstractModule):
    """MLP module with 1D tensor parallelism based on TTT code.
    See the `AbstractModule` docstring for usage info.
    NOTE: This is not the MLP we will use for DeepSeek-R1, but we do use it as a base class for the other MLPs.
    """

    WEIGHT_TORCH_DTYPE = torch.bfloat16
    WEIGHT_DTYPE = ttnn.bfloat4_b

    @dataclass
    class ProgramConfigData(OpConfigBase):
        """Data class for the data for generating the PC for ttnn.linear."""

        dim: int
        hidden_dim: int
        num_devices: int
        core_grid_size: ttnn.CoreCoord

    @classmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dicts: tuple[dict[str, torch.Tensor] | None, ...],
        output_path: Path,
        mesh_device: ttnn.Device,
    ) -> WeightConfig:
        return {
            models_name: {
                "input_tensor_b": save_and_get_path(
                    output_path / f"{models_name}.input_tensor_b",
                    cls.convert_metaweight(
                        get_state_dicts(
                            state_dicts,
                            f"{hf_name}.weight",
                            shape=(out_features, in_features),
                            dtype=cls.WEIGHT_TORCH_DTYPE,
                        ),
                        mesh_device,
                        is_w2,
                    ),
                )
            }
            for hf_name, models_name, is_w2 in [
                ("gate_proj", "w1", False),
                ("down_proj", "w2", True),
                ("up_proj", "w3", False),
            ]
            for in_features, out_features in [cls.get_weight_shape(hf_config, is_w2)]
        }

    @final
    @classmethod
    def convert_metaweight(
        cls,
        torch_metaweight_tensor: torch.Tensor,
        mesh_device: ttnn.Device,
        is_w2: bool,
    ) -> ttnn.Tensor:
        """
        Convert a normal (non-quantized) weight tensor to a format suitable for TTNN.

        Args:
            metaweight_tensor: The weight tensor.
            mesh_device: The mesh device to use for the conversion.

        Returns:
            The converted TTNN tensor.
        """
        torch_metaweight_tensor = torch_metaweight_tensor.transpose(
            2, 1
        )  # In torch the weights are in (out_features, in_features) format

        # Calculate the expected weight dimensions
        num_shards, per_device_in_features, per_device_out_features = torch_metaweight_tensor.shape
        mp, tp = mesh_device.shape
        assert num_shards == mp, "Number of mesh rows does not match weight shards"

        if is_w2:
            per_device_in_features = even_int_div(per_device_in_features, tp)
            mesh_sharded_dim = 1
        else:
            per_device_out_features = even_int_div(per_device_out_features, tp)
            mesh_sharded_dim = 2

        # Convert the torch tensor to a TTNN tensor
        metaweight_tensor = ttnn.from_torch(
            torch_metaweight_tensor,
            dtype=cls.WEIGHT_DTYPE,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=dram_sharded_weight_config(
                per_device_in_features,
                per_device_out_features,
                mesh_device.dram_grid_size(),
            ),
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, (0, mesh_sharded_dim)),
        )
        return ttnn.squeeze(metaweight_tensor, 0)  # Remove the row shard dimension

    @final
    @classmethod
    def get_weight_shape(cls, hf_config: PretrainedConfig, is_w2: bool) -> tuple[int, int]:
        """Get the shape of the weight tensor for the MLP."""
        dim, hidden_dim = cls._get_model_dims_from_cfg(hf_config)
        if is_w2:
            return hidden_dim, dim
        else:
            return dim, hidden_dim

    @classmethod
    def prefill_model_config(cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device) -> ModelPrefillConfig:
        """Generate prefill configuration for this module.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device the model will be placed later on

        Returns:
            ModelPrefillConfig containing operator configurations for prefill mode
        """
        matmul_core_grid_size = ttnn.CoreCoord(
            mesh_device.core_grid.x,
            mesh_device.core_grid.y,
        )  # NOTE: we might modify this later during optimization stage

        # Calculate device metrics
        _, mesh_width = mesh_device.shape

        # Extract dimensions from HF config
        dim, hidden_dim = cls._get_model_dims_from_cfg(hf_config)

        # Compute the program config for the linear layers
        linear_op_config = LinearConfig(
            input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
        )

        # Construct the config
        return {
            "all_gather": AllGatherAsyncConfig(
                mesh_device=MeshDeviceStub(mesh_device.shape),
                cluster_axis=1,
                dim=-1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=ttnn.Topology.Linear,  # One row of Galaxy does not form a ring
            ),
            "max_rows": SEQ_LEN_CHUNK_SIZE,  # NOTE: should be 512 for blackhole (in case of future bring-up)
            "linear_pc_gen": MLP1D.ProgramConfigData(
                dim=dim, hidden_dim=hidden_dim, num_devices=mesh_width, core_grid_size=matmul_core_grid_size
            ),
            "w1": linear_op_config,
            "w2": linear_op_config,
            "w3": linear_op_config,
            "mul": MulConfig(
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            ),
            "reduce_scatter_async": ReduceScatterAsyncConfig(
                dim=-1,  # We are scattering across the feature dimension (last one)
                cluster_axis=1,  # Reduce-scatter across the mesh rows
                mesh_device=MeshDeviceStub(mesh_device.shape),
                math_op=ttnn.ReduceType.Sum,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=ttnn.Topology.Linear,  # One row of Galaxy does not form a ring
            ),
            "output_memory_config": ttnn.DRAM_MEMORY_CONFIG,
        }

    @classmethod
    def decode_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
        input_num_cores: int | None = None,
        output_num_cores: int | None = None,
    ) -> ModelDecodeConfig:
        """Generate decode configuration for this module.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device the model will be placed later on
            input_num_cores (optional): Number of cores the input tensor is sharded on.
                Must be a divisor of the tile width of the input tensor (i.e. sharding cannot be padded)
            output_num_cores (optional): Number of cores the output tensor is sharded on.
                Must be a divisor of the tile width of the output tensor (i.e. sharding cannot be padded)

        Returns:
            ModelDecodeConfig containing operator configurations for decode mode
        """
        # Extract dimensions from HF config
        dim, hidden_dim = cls._get_model_dims_from_cfg(hf_config)

        # Calculate device metrics
        _, mesh_width = mesh_device.shape
        max_num_cores = mesh_device.core_grid.x * mesh_device.core_grid.y
        input_num_cores = input_num_cores or max(
            get_activation_sharding_core_counts_for_dram_matmul(dim, max_num_cores)
        )
        inner_num_cores = max(
            get_activation_sharding_core_counts_for_dram_matmul(even_int_div(hidden_dim, mesh_width), max_num_cores)
        )
        output_num_cores = output_num_cores or max(
            get_activation_sharding_core_counts_for_dram_matmul(even_int_div(dim, mesh_width), max_num_cores)
        )
        assert (
            input_num_cores <= max_num_cores
        ), "input_num_cores must be less than or equal to the maximum number of cores"
        assert (
            output_num_cores <= max_num_cores
        ), "output_num_cores must be less than or equal to the maximum number of cores"
        assert dim % input_num_cores == 0, "input_num_cores must divide the input tensor width evenly"
        assert (
            even_int_div(dim, mesh_width) % output_num_cores == 0
        ), "output_num_cores must divide the output tensor width evenly"

        # Calculate input and output memory configurations
        output_memory_config = cls._get_decode_activation_memory_config(
            even_int_div(dim, mesh_width), output_num_cores, mesh_device
        )

        # Construct the config
        return {
            "all_gather": AllGatherAsyncConfig(
                mesh_device=MeshDeviceStub(mesh_device.shape),
                cluster_axis=1,
                dim=-1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=ttnn.Topology.Linear,  # One row of Galaxy does not form a ring
            ),
            "all_gather_reshard": ReshardConfig(
                memory_config=cls._get_decode_activation_memory_config(dim, input_num_cores, mesh_device)
            ),
            "w1": LinearConfig(
                input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                program_config=get_dram_sharded_matmul_config(
                    MAX_BATCH_SIZE, dim, even_int_div(hidden_dim, mesh_width), input_num_cores, inner_num_cores
                ),
                compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
            ),
            "w2": LinearConfig(
                input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                program_config=get_dram_sharded_matmul_config(
                    MAX_BATCH_SIZE,
                    even_int_div(hidden_dim, mesh_width),
                    dim,
                    inner_num_cores,
                    output_num_cores,
                ),
                compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
            ),
            "w3": LinearConfig(
                input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                program_config=get_dram_sharded_matmul_config(
                    MAX_BATCH_SIZE, dim, even_int_div(hidden_dim, mesh_width), input_num_cores, inner_num_cores
                ),
                compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
            ),
            "mul": MulConfig(
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            ),
            "reduce_scatter_async": ReduceScatterAsyncConfig(
                mesh_device=MeshDeviceStub(mesh_device.shape),
                cluster_axis=1,  # Reduce-scatter across the mesh rows
                dim=-1,  # We are scattering across the feature dimension (last one)
                math_op=ttnn.ReduceType.Sum,
                topology=ttnn.Topology.Linear,  # One row of Galaxy does not form a ring
                memory_config=output_memory_config,
            ),
            "output_memory_config": output_memory_config,  # For asserting the output of the MLP
        }

    @classmethod
    def _get_model_dims_from_cfg(cls, hf_config: PretrainedConfig) -> tuple[int, int]:
        """Get the dimensions of the model from the HuggingFace config.

        Args:
            hf_config: HuggingFace model configuration object.

        Returns:
            Tuple containing the input dimension and hidden dimension of the MLP.
        """
        dim = hf_config.hidden_size
        hidden_dim = hf_config.intermediate_size
        return dim, hidden_dim

    @final
    @classmethod
    def _get_decode_activation_memory_config(
        cls, per_device_width: int, activation_sharding_num_cores: int, mesh_device: ttnn.Device
    ) -> ttnn.MemoryConfig:
        """Get the memory config for an activation tensor in decode mode."""
        return ttnn.create_sharded_memory_config_(
            shape=(
                ttnn.core.roundup(MAX_BATCH_SIZE, ttnn.TILE_SIZE),
                ttnn.core.roundup(even_int_div(per_device_width, activation_sharding_num_cores), ttnn.TILE_SIZE),
            ),
            core_grid=ttnn.num_cores_to_corerangeset(
                activation_sharding_num_cores,
                ttnn.CoreCoord(mesh_device.core_grid.x, mesh_device.core_grid.y),
                row_wise=True,
            ),
            strategy=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            tile_layout=True,
            use_height_and_width_as_shard_shape=True,
        )

    @classmethod
    def create_state(cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device, ccl: CCL1D) -> ModelState:
        """Create the model state for this module.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device the model will be placed later on
            ccl: CCL1D instance for async CCLs

        Returns:
            ModelState containing the state information for this module
        """
        return {
            MESH_DEVICE_STATE_DICT_KEY: mesh_device,
            "all_gather": {
                "multi_device_global_semaphore": ccl.get_gather_sem(1),
                "num_links": ccl.get_max_links(1),
            },
            "reduce_scatter_async": {
                "from_remote_multi_device_global_semaphore": ccl.get_from_sem(1),
                "to_remote_multi_device_global_semaphore": ccl.get_to_sem(1),
                "num_links": ccl.get_max_links(1),
            },
        }

    @classmethod
    def _get_prefill_pc(
        cls, seq_len: int, dim: int, hidden_dim: int, num_devices: int, core_grid_size: ttnn.CoreCoord, is_w2: bool
    ) -> Any:
        """Get the program config for linear layers in prefill mode based on sequence length."""
        if is_w2:
            per_device_in_features, per_device_out_features = even_int_div(hidden_dim, num_devices), dim
        else:
            per_device_in_features, per_device_out_features = dim, even_int_div(hidden_dim, num_devices)

        per_core_M_tiles = ttnn.core.divup(seq_len, ttnn.TILE_SIZE * core_grid_size.y)
        K_tiles = ttnn.core.divup(per_device_in_features, ttnn.TILE_SIZE)
        per_core_N_tiles = ttnn.core.divup(per_device_out_features, ttnn.TILE_SIZE * core_grid_size.x)

        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=core_grid_size,
            in0_block_w=find_largest_divisor(K_tiles),
            out_subblock_h=1,
            out_subblock_w=find_largest_divisor(
                per_core_N_tiles,
                4,  # out_subblock_h * out_subblock_w <= 4
            ),
            per_core_M=per_core_M_tiles,
            per_core_N=per_core_N_tiles,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )

    @classmethod
    def _silu_workaround(cls, x: ttnn.Tensor) -> ttnn.Tensor:  # TODO: remove once ttnn.silu PCC is fixed
        """Workaround for the silu PCC issue in ttnn."""
        # -x
        x1 = ttnn.neg(x)

        # 1
        x2 = ttnn.ones_like(x)

        # exp(-x)
        x3 = ttnn.exp(x1)
        ttnn.deallocate(x1)

        # 1 + exp(-x)
        x4 = ttnn.add(x3, 1)
        ttnn.deallocate(x3)

        # 1 / (1 + exp(-x))
        x5 = ttnn.div(x2, x4)
        ttnn.deallocate(x2)
        ttnn.deallocate(x4)

        # x * (1 / (1 + exp(-x)))
        x6 = ttnn.mul(x, x5)
        ttnn.deallocate(x5)

        return x6

    @classmethod
    def forward_prefill(cls, x: ttnn.Tensor, cfg: RunPrefillConfig) -> ttnn.Tensor:
        num_layers, _, seq_len, _ = x.shape

        # All gather for efficient matmuls
        x = ttnn.experimental.all_gather_async(x, **cfg["all_gather"])

        # Chunk the input if needed
        if seq_len > cfg["max_rows"]:  # For large sequence lengths, process the input in chunks
            x = ttnn.reshape(x, [num_layers, even_int_div(seq_len, cfg["max_rows"]), cfg["max_rows"], -1])
            seq_len = cfg["max_rows"]

        # Gate and up projections with dynamic program configs
        w1_out = ttnn.linear(
            x, program_config=cls._get_prefill_pc(seq_len=seq_len, is_w2=False, **cfg["linear_pc_gen"]), **cfg["w1"]
        )
        w3_out = ttnn.linear(
            x, program_config=cls._get_prefill_pc(seq_len=seq_len, is_w2=False, **cfg["linear_pc_gen"]), **cfg["w3"]
        )
        ttnn.deallocate(x)

        # Apply silu
        # w1_out_activated = cls._silu_workaround(w1_out)
        # ttnn.deallocate(w1_out)

        # Apply activation and multiply
        activated = ttnn.mul(w1_out, w3_out, **cfg["mul"])
        ttnn.deallocate(w1_out)
        ttnn.deallocate(w3_out)

        # Down projection with dynamic program configs, no need to reshard as we are using dram activations
        output = ttnn.linear(
            activated,
            program_config=cls._get_prefill_pc(seq_len=seq_len, is_w2=True, **cfg["linear_pc_gen"]),
            **cfg["w2"],
        )
        ttnn.deallocate(activated)

        # Reduce-scatter across devices to sum partial results
        output = ttnn.experimental.reduce_scatter_async(output, **cfg["reduce_scatter_async"])

        # De-chunk the output if the input was chunked
        _, num_chunks, _, output_dim = output.shape
        if num_chunks > 1:
            output = ttnn.reshape(output, [num_layers, 1, -1, output_dim])

        assert output.memory_config() == cfg["output_memory_config"]
        return output

    @classmethod
    def forward_decode(cls, x: ttnn.Tensor, cfg: RunDecodeConfig) -> ttnn.Tensor:
        # All gather
        x = ttnn.experimental.all_gather_async(x, **cfg["all_gather"])

        # TODO: File issue on AG not being able to do this internally (Issue #26672)
        x = ttnn.to_memory_config(x, **cfg["all_gather_reshard"])

        # Gate and up projections
        w1_out = ttnn.linear(x, **cfg["w1"])
        w3_out = ttnn.linear(x, **cfg["w3"])

        # Apply silu
        w1_out_activated = cls._silu_workaround(w1_out)
        ttnn.deallocate(w1_out)

        # Apply activation and multiply
        activated = ttnn.mul(w1_out_activated, w3_out, **cfg["mul"])
        ttnn.deallocate(w1_out_activated)
        ttnn.deallocate(w3_out)

        # Down projection
        w2_out = ttnn.linear(activated, **cfg["w2"])
        ttnn.deallocate(activated)

        # Add reduce-scatter
        output = ttnn.experimental.reduce_scatter_async(w2_out, **cfg["reduce_scatter_async"])
        ttnn.deallocate(w2_out)

        assert output.memory_config() == cfg["output_memory_config"]
        return output
