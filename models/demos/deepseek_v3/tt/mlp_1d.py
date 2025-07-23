# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from pathlib import Path
from typing import Any, final

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.utils.abstract_module import AbstractModule
from models.demos.deepseek_v3.utils.config_dataclass import (
    FromWeightConfig,
    LinearConfig,
    MeshDeviceStub,
    MulConfig,
    OpConfigBase,
    ReduceScatterConfig,
)
from models.demos.deepseek_v3.utils.config_helpers import (
    COMPUTE_KERNEL_CONFIG_LOFI,
    COMPUTE_KERNEL_CONFIG_SDPA,
    MAX_BATCH_SIZE,
    SEQ_LEN_CHUNK_SIZE,
    dram_sharded_weight_config,
    even_int_div,
    find_largest_divisor,
    get_activation_sharding_core_counts_for_dram_matmul,
    get_dram_sharded_matmul_config,
    save_and_get_path,
)
from models.demos.deepseek_v3.utils.run_config import (
    ModelDecodeConfig,
    ModelPrefillConfig,
    RunDecodeConfig,
    RunPrefillConfig,
    WeightConfig,
)


class MLP1D(AbstractModule):
    """MLP module with 1D tensor parallelism based on TTT code.
    See the `AbstractModule` docstring for usage info.
    NOTE: This is not the MLP we will use for DeepSeek-R1, but we do use it as a base class for the other MLPs.
    """

    @dataclass
    class MLPProgramConfigData(OpConfigBase):
        """Data class for the data for generating the PC for ttnn.linear."""

        dim: int
        hidden_dim: int
        num_devices: int
        core_grid_size: ttnn.CoreCoord

    DRAM_SHARD_GRID_WIDTH = 8
    PREFILL_ROWS = 8

    @classmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dict: dict[str, torch.Tensor],
        output_path: Path,
        mesh_device: ttnn.Device,
    ) -> WeightConfig:
        assert cls.is_device_supported(mesh_device)
        return {
            models_name: {
                "input_tensor_b": save_and_get_path(
                    output_path / f"{models_name}.input_tensor_b",
                    cls.convert_weight(
                        hf_config,
                        state_dict[f"{hf_name}.weight"],
                        mesh_device,
                        is_w2=(models_name == "w2"),
                    ),
                )
            }
            for hf_name, models_name in [
                ("gate_proj", "w1"),
                ("down_proj", "w2"),
                ("up_proj", "w3"),
            ]
        }

    @final
    @classmethod
    def convert_weight(
        cls,
        hf_config: PretrainedConfig,
        weight_tensor: torch.Tensor,
        mesh_device: ttnn.Device,
        is_w2: bool,
    ) -> ttnn.Tensor:
        """
        Convert a normal (non-quantized) weight tensor to a format suitable for TTNN.

        Args:
            weight_tensor: The weight tensor.
            mesh_device: The mesh device to use for the conversion.

        Returns:
            The converted TTNN tensor.
        """
        dim, hidden_dim = cls._get_model_dims_from_cfg(hf_config)

        torch_weight_tensor = weight_tensor.permute(
            1, 0
        )  # In torch the weights are in (out_features, in_features) format

        if is_w2:
            assert torch_weight_tensor.shape == (hidden_dim, dim)
            per_device_in_features, per_device_out_features = (
                even_int_div(hidden_dim, mesh_device.get_num_devices()),
                dim,
            )
            mesh_sharded_dim = 0
        else:
            assert torch_weight_tensor.shape == (dim, hidden_dim)
            per_device_in_features, per_device_out_features = dim, even_int_div(
                hidden_dim, mesh_device.get_num_devices()
            )
            mesh_sharded_dim = 1

        return ttnn.from_torch(
            torch_weight_tensor,
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=dram_sharded_weight_config(
                per_device_in_features,
                per_device_out_features,
                mesh_device.dram_grid_size(),
            ),
            mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(mesh_device, mesh_sharded_dim),
        )

    @final
    @classmethod
    def is_device_supported(cls, mesh_device: ttnn.Device) -> bool:
        """
        As we only support 1D tensor parallelism, we only support 1D mesh devices.

        Args:
            mesh_device: The mesh device to check.

        Returns:
            True if the device is supported, False otherwise.
        """
        return tuple(mesh_device.shape)[0] == 1

    @classmethod
    def prefill_model_config(cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device) -> ModelPrefillConfig:
        """Generate prefill configuration for this module.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device the model will be placed later on

        Returns:
            ModelPrefillConfig containing operator configurations for prefill mode
        """
        matmul_core_grid_size = ttnn.CoreCoord(  # Matmul expects the core grid size in (y, x) format
            mesh_device.core_grid.x,
            mesh_device.core_grid.y,
        )  # NOTE: we might modify this later during optimization stage

        # Calculate device metrics
        num_devices = mesh_device.get_num_devices()

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
            "max_rows": SEQ_LEN_CHUNK_SIZE,  # NOTE: should be 512 for blackhole (in case of future bring-up)
            "linear_pc_gen": MLP1D.MLPProgramConfigData(
                dim=dim, hidden_dim=hidden_dim, num_devices=num_devices, core_grid_size=matmul_core_grid_size
            ),
            "w1": linear_op_config,
            "w2": linear_op_config,
            "w3": linear_op_config,
            "mul": MulConfig(
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            ),
            "reduce_scatter": ReduceScatterConfig(
                dim=-1,  # We are scattering across the feature dimension (last one)
                math_op=ttnn.ReduceType.Sum,
                topology=ttnn.Topology.Linear,  # One row of Galaxy does not form a ring
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            "input_memory_config": ttnn.DRAM_MEMORY_CONFIG,  # RMSNorm must provide this shard spec as its output
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
        num_devices = mesh_device.get_num_devices()
        max_num_cores = mesh_device.core_grid.x * mesh_device.core_grid.y
        input_num_cores = input_num_cores or max(
            get_activation_sharding_core_counts_for_dram_matmul(dim, max_num_cores)
        )
        inner_num_cores = max(
            get_activation_sharding_core_counts_for_dram_matmul(even_int_div(hidden_dim, num_devices), max_num_cores)
        )
        output_num_cores = output_num_cores or max(
            get_activation_sharding_core_counts_for_dram_matmul(even_int_div(dim, num_devices), max_num_cores)
        )
        assert (
            input_num_cores <= max_num_cores
        ), "input_num_cores must be less than or equal to the maximum number of cores"
        assert (
            output_num_cores <= max_num_cores
        ), "output_num_cores must be less than or equal to the maximum number of cores"
        assert dim % input_num_cores == 0, "input_num_cores must divide the input tensor width evenly"
        assert (
            even_int_div(dim, num_devices) % output_num_cores == 0
        ), "output_num_cores must divide the output tensor width evenly"

        # Calculate input and output memory configurations
        input_memory_config = cls._get_decode_activation_memory_config(dim, input_num_cores, mesh_device)
        output_memory_config = cls._get_decode_activation_memory_config(
            even_int_div(dim, num_devices), output_num_cores, mesh_device
        )

        # Construct the config
        return {
            "w1": LinearConfig(
                input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                program_config=get_dram_sharded_matmul_config(
                    MAX_BATCH_SIZE, dim, even_int_div(hidden_dim, num_devices), input_num_cores, inner_num_cores
                ),
                compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
            ),
            "w2": LinearConfig(
                input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                program_config=get_dram_sharded_matmul_config(
                    MAX_BATCH_SIZE,
                    even_int_div(hidden_dim, num_devices),
                    dim,
                    inner_num_cores,
                    output_num_cores,
                ),
                compute_kernel_config=COMPUTE_KERNEL_CONFIG_SDPA,
            ),
            "w3": LinearConfig(
                input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                program_config=get_dram_sharded_matmul_config(
                    MAX_BATCH_SIZE, dim, even_int_div(hidden_dim, num_devices), input_num_cores, inner_num_cores
                ),
                compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
            ),
            "mul": MulConfig(
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            ),
            "reduce_scatter": ReduceScatterConfig(
                dim=-1,  # We are scattering across the feature dimension (last one)
                math_op=ttnn.ReduceType.Sum,
                memory_config=output_memory_config,
                topology=ttnn.Topology.Linear,  # One row of Galaxy does not form a ring
            ),
            "input_memory_config": input_memory_config,  # For asserting the input to the MLP
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
                even_int_div(ttnn.core.roundup(per_device_width, ttnn.TILE_SIZE), activation_sharding_num_cores),
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
    def forward_prefill(cls, x: ttnn.Tensor, cfg: RunPrefillConfig) -> ttnn.Tensor:
        _, _, seq_len, _ = x.shape

        if seq_len > cfg["max_rows"]:  # For large sequence lengths, process the input in chunks
            x = ttnn.reshape(x, [1, even_int_div(seq_len, cfg["max_rows"]), cfg["max_rows"], -1])
            seq_len = cfg["max_rows"]

        # Gate and up projections with dynamic program configs
        w1_out = ttnn.linear(
            x, program_config=cls._get_prefill_pc(seq_len=seq_len, is_w2=False, **cfg["linear_pc_gen"]), **cfg["w1"]
        )
        w3_out = ttnn.linear(
            x, program_config=cls._get_prefill_pc(seq_len=seq_len, is_w2=False, **cfg["linear_pc_gen"]), **cfg["w3"]
        )
        ttnn.deallocate(x)

        # add reduce-scatter here to gather intermediates

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
        output = ttnn.reduce_scatter(output, **cfg["reduce_scatter"])

        # De-chunk the output if the input was chunked
        _, num_chunks, _, output_dim = output.shape
        if num_chunks > 1:
            output = ttnn.reshape(output, [1, 1, -1, output_dim])

        assert output.memory_config() == cfg["output_memory_config"]
        return output

    @classmethod
    def forward_decode(cls, x: ttnn.Tensor, cfg: RunDecodeConfig) -> ttnn.Tensor:
        assert x.memory_config() == cfg["input_memory_config"]

        # Gate and up projections
        w1_out = ttnn.linear(x, **cfg["w1"])
        w3_out = ttnn.linear(x, **cfg["w3"])

        # Apply activation and multiply
        activated = ttnn.mul(w1_out, w3_out, **cfg["mul"])

        # Down projection
        w2_out = ttnn.linear(activated, **cfg["w2"])
        # ttnn.deallocate(activated)

        # Add reduce-scatter
        output = ttnn.reduce_scatter(w2_out, **cfg["reduce_scatter"])
        # ttnn.deallocate(w2_out)

        assert output.memory_config() == cfg["output_memory_config"]
        return output
