# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path
from typing import final

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.tt.mlp_1d import MLP1D
from models.demos.deepseek_v3.utils.config_dataclass import FromWeightConfig, LinearConfig, MeshDeviceStub, MulConfig
from models.demos.deepseek_v3.utils.config_helpers import (
    COMPUTE_KERNEL_CONFIG_LOFI,
    COMPUTE_KERNEL_CONFIG_SDPA,
    MAX_BATCH_SIZE,
    SEQ_LEN_CHUNK_SIZE,
    dram_sharded_weight_config,
    even_int_div,
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


class Expert(MLP1D):  # The only difference with the regular Dequantized MLP is the intermediate layer size
    """Expert layer for Mixture-of-Experts (MoE) models."""

    @classmethod
    def _get_model_dims_from_cfg(cls, hf_config: PretrainedConfig) -> tuple[int, int]:
        dim = hf_config.hidden_size
        hidden_dim = hf_config.moe_intermediate_size
        return dim, hidden_dim

    @classmethod
    def convert_weights_moe(
        cls,
        hf_config: PretrainedConfig,
        state_dict: dict[str, torch.Tensor],
        output_path: Path,
        mesh_device: ttnn.Device,
    ) -> WeightConfig:
        # assert cls.is_device_supported(mesh_device)

        # breakpoint()  # Debugging point to inspect the state_dict structure
        num_experts_per_device = hf_config.n_routed_experts // mesh_device.get_num_devices()
        num_groups = hf_config.n_routed_experts // num_experts_per_device

        w1_per_device_state_dict_group = []
        w2_per_device_state_dict_group = []
        w3_per_device_state_dict_group = []

        # Group experts by device
        for device_idx in range(mesh_device.get_num_devices()):
            w1_sub_group = []
            w2_sub_group = []
            w3_sub_group = []

            # Add experts for this device
            start_expert = device_idx * num_experts_per_device
            end_expert = start_expert + num_experts_per_device

            for expert_idx in range(start_expert, end_expert):
                w1_sub_group.append(state_dict[f"experts.{expert_idx}.gate_proj.weight"])
                w2_sub_group.append(state_dict[f"experts.{expert_idx}.down_proj.weight"])
                w3_sub_group.append(state_dict[f"experts.{expert_idx}.up_proj.weight"])

            w1_per_device_state_dict_group.append(torch.stack(w1_sub_group, dim=0).permute(0, 2, 1))
            w2_per_device_state_dict_group.append(torch.stack(w2_sub_group, dim=0).permute(0, 2, 1))
            w3_per_device_state_dict_group.append(torch.stack(w3_sub_group, dim=0).permute(0, 2, 1))

        # Convert weights for each expert group
        return {
            "w1_experts": {
                "input_tensor_b": save_and_get_path(
                    output_path / f"w1_experts.input_tensor_b",
                    cls._convert_weight_moe(hf_config, w1_per_device_state_dict_group, mesh_device, is_w2=False),
                )
            },
            "w2_experts": {
                "input_tensor_b": save_and_get_path(
                    output_path / f"w2_experts.input_tensor_b",
                    cls._convert_weight_moe(hf_config, w2_per_device_state_dict_group, mesh_device, is_w2=True),
                )
            },
            "w3_experts": {
                "input_tensor_b": save_and_get_path(
                    output_path / f"w3_experts.input_tensor_b",
                    cls._convert_weight_moe(hf_config, w3_per_device_state_dict_group, mesh_device, is_w2=False),
                )
            },
        }

    @final
    @classmethod
    def _convert_weight_moe(
        cls,
        hf_config: PretrainedConfig,
        wx_per_device_state_dict_group: list[torch.Tensor],
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

        per_device_in_features, per_device_out_features = dim, hidden_dim
        if is_w2:
            per_device_in_features, per_device_out_features = hidden_dim, dim

        multi_dev_host_weights = ttnn.from_host_shards(
            [ttnn.from_torch(e, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT) for e in wx_per_device_state_dict_group],
            mesh_device.shape,
        )

        multi_dev_weights = ttnn.to_device(
            multi_dev_host_weights,
            mesh_device,
            memory_config=dram_sharded_weight_config(
                per_device_in_features * 8, per_device_out_features, mesh_device.dram_grid_size()
            ),
        )

        return multi_dev_weights

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
        return tuple(mesh_device.shape)[0] >= 1

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

        num_experts_per_device = hf_config.n_routed_experts // mesh_device.get_num_devices()

        num_devices = 1  # Weights are not TP'd

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
            "linear_pc_gen": cls.MLPProgramConfigData(
                dim=dim, hidden_dim=hidden_dim, num_devices=num_devices, core_grid_size=matmul_core_grid_size
            ),
            "w1_experts": linear_op_config,
            "w2_experts": linear_op_config,
            "w3_experts": linear_op_config,
            "mul_experts": MulConfig(
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            ),
            "input_memory_config": ttnn.DRAM_MEMORY_CONFIG,  # RMSNorm must provide this shard spec as its output
            "output_memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "num_experts_per_device": num_experts_per_device,
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

        num_experts_per_device = hf_config.n_routed_experts // mesh_device.get_num_devices()

        # Calculate device metrics
        num_devices = 1  # Weights are not TP'd
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
        input_memory_config = ttnn.L1_MEMORY_CONFIG
        output_memory_config = ttnn.L1_MEMORY_CONFIG

        # Construct the config
        return {
            "w1_experts": LinearConfig(
                input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                program_config=get_dram_sharded_matmul_config(
                    MAX_BATCH_SIZE, dim, even_int_div(hidden_dim, num_devices), input_num_cores, inner_num_cores
                ),
                compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
            ),
            "w2_experts": LinearConfig(
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
            "w3_experts": LinearConfig(
                input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                program_config=get_dram_sharded_matmul_config(
                    MAX_BATCH_SIZE, dim, even_int_div(hidden_dim, num_devices), input_num_cores, inner_num_cores
                ),
                compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
            ),
            "mul_experts": MulConfig(
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            ),
            "input_memory_config": input_memory_config,  # For asserting the input to the MLP
            "output_memory_config": output_memory_config,  # For asserting the output of the MLP
            "num_experts_per_device": num_experts_per_device,
        }

    @classmethod
    def forward_prefill(cls, x: ttnn.Tensor, cfg: RunPrefillConfig) -> ttnn.Tensor:
        # Repeat the activations for each expert as we don't support activation broadcasting inside the matmul
        x = ttnn.repeat(x, ttnn.Shape((1, cfg["num_experts_per_device"], 1, 1)))
        _, experts_per_device, seq_len, _ = x.shape

        if seq_len > cfg["max_rows"]:  # For large sequence lengths, process the input in chunks
            x = ttnn.reshape(x, [1, even_int_div(seq_len, cfg["max_rows"]) * experts_per_device, cfg["max_rows"], -1])
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
        breakpoint()

        # Repeat the activations for each expert as we don't support activation broadcasting inside the matmul
        x = ttnn.repeat(x, ttnn.Shape((1, cfg["num_experts_per_device"], 1, 1)))

        _, experts_per_device, batch_size, _ = x.shape

        # Gate and up projections
        w1_out = ttnn.linear(x, **cfg["w1_experts"])
        w3_out = ttnn.linear(x, **cfg["w3_experts"])

        # Apply activation and multiply
        activated = ttnn.mul(w1_out, w3_out, **cfg["mul_experts"])

        # Down projection
        output = ttnn.linear(activated, **cfg["w2_experts"])
        ttnn.deallocate(activated)

        assert output.memory_config() == cfg["output_memory_config"]
        return output
