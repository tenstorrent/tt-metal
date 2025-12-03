# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import math
from pathlib import Path

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.utils.abstract_module import AbstractModule
from models.demos.deepseek_v3.utils.config_dataclass import (
    FromWeightConfig,
    MeshDeviceStub,
    MulConfig,
    SparseMatmulConfig,
)
from models.demos.deepseek_v3.utils.config_helpers import (
    COMPUTE_KERNEL_CONFIG_LOFI,
    SPARSITY_BLOCK_SIZE,
    dequantize,
    even_int_div,
    shard_and_save,
)
from models.demos.deepseek_v3.utils.run_config import (
    ModelDecodeConfig,
    ModelPrefillConfig,
    RunDecodeConfig,
    RunPrefillConfig,
    WeightConfig,
)


class Experts(AbstractModule):
    """Experts layer for Mixture-of-Experts (MoE) module."""

    @classmethod
    def _get_num_experts_per_device(cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device) -> int:
        """Calculate the number of experts per device based on the total number of experts and the device shape."""
        return even_int_div(hf_config.n_routed_experts, mesh_device.get_num_devices())

    @classmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dicts: tuple[dict[str, torch.Tensor] | None, ...],
        output_path: Path,
        mesh_device: ttnn.Device,
    ) -> WeightConfig:
        assert hf_config.n_routed_experts % mesh_device.get_num_devices() == 0, (
            f"Number of experts ({hf_config.n_routed_experts}) must be divisible by the number of devices "
            f"({mesh_device.get_num_devices()})"
        )
        (state_dict,) = state_dicts
        assert state_dict is not None

        return {
            ttnn_name: {
                "input_tensor_b": shard_and_save(
                    output_path / f"{ttnn_name}.input_tensor_b",
                    dequantize(
                        torch.stack(
                            [
                                state_dict[f"experts.{expert_id}.{hf_name}.weight"]
                                for expert_id in range(hf_config.n_routed_experts)
                            ]
                        ),
                        torch.stack(
                            [
                                state_dict[f"experts.{expert_id}.{hf_name}.weight_scale_inv"]
                                for expert_id in range(hf_config.n_routed_experts)
                            ]
                        ),
                        (1, *hf_config.quantization_config["weight_block_size"]),
                    )
                    .unsqueeze(0)
                    .transpose(-1, -2),
                    shard_dims=(1, 1),
                    mesh_device=mesh_device,
                    dtype=ttnn.bfloat4_b,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            }
            for hf_name, ttnn_name in [
                ("gate_proj", "w1_experts"),
                ("down_proj", "w2_experts"),
                ("up_proj", "w3_experts"),
            ]
        }

    @classmethod
    def is_device_supported(cls, mesh_device: ttnn.Device) -> bool:
        """
        As we only support 1D tensor parallelism, we only support 1D mesh devices.

        Args:
            mesh_device: The mesh device to check.

        Returns:
            True if the device is supported, False otherwise.
        """
        return mesh_device.shape[1] == 8

    @classmethod
    def _get_sparse_pc(cls, core_x: int, core_y: int, n: int) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
        """Generate program config for sparse matmul.

        Args:
            core_x: Grid x dimension (from config)
            core_y: Grid y dimension (from config)
            n: Output feature dimension (from config)

        Returns:
            MatmulMultiCoreReuseMultiCast1DProgramConfig for sparse_matmul
        """
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=int(math.ceil(n / ttnn.TILE_SIZE)) // (core_x * core_y),
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )

    @classmethod
    def _create_model_config(
        cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device, mode: str
    ) -> ModelPrefillConfig | ModelDecodeConfig:
        num_experts_per_device = cls._get_num_experts_per_device(hf_config, mesh_device)

        # Calculate dimensions
        hidden_size = hf_config.hidden_size
        moe_intermediate_size = hf_config.moe_intermediate_size

        # Calculate input and output memory configurations
        if mode == "decode":
            input_memory_config = ttnn.L1_MEMORY_CONFIG
            output_memory_config = ttnn.L1_MEMORY_CONFIG
        else:
            input_memory_config = ttnn.DRAM_MEMORY_CONFIG
            output_memory_config = ttnn.DRAM_MEMORY_CONFIG

        # Construct the config
        return {
            "w1_experts": SparseMatmulConfig(
                input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                memory_config=output_memory_config,
                compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
                program_config=cls._get_sparse_pc(core_x=8, core_y=8, n=moe_intermediate_size),
                is_input_a_sparse=False,
                is_input_b_sparse=True,
                output_tile=ttnn.Tile([SPARSITY_BLOCK_SIZE, ttnn.TILE_SIZE]),
            ),
            "w2_experts": SparseMatmulConfig(
                input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                memory_config=output_memory_config,
                compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
                program_config=cls._get_sparse_pc(core_x=8, core_y=7, n=hidden_size),
                is_input_a_sparse=True,
                is_input_b_sparse=False,
                output_tile=ttnn.Tile([SPARSITY_BLOCK_SIZE, ttnn.TILE_SIZE]),
            ),
            "w3_experts": SparseMatmulConfig(
                input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                memory_config=output_memory_config,
                compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
                program_config=cls._get_sparse_pc(core_x=8, core_y=8, n=moe_intermediate_size),
                is_input_a_sparse=False,
                is_input_b_sparse=True,
                output_tile=ttnn.Tile([SPARSITY_BLOCK_SIZE, ttnn.TILE_SIZE]),
            ),
            "mul_experts": MulConfig(
                memory_config=output_memory_config,
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            ),
            "input_memory_config": input_memory_config,
            "output_memory_config": output_memory_config,
            "num_experts_per_device": num_experts_per_device,
        }

    @classmethod
    def decode_model_config(cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device) -> ModelDecodeConfig:
        """Generate decode configuration for this module.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device the model will be placed later on

        Returns:
            ModelPrefillConfig containing operator configurations for prefill mode
        """
        return cls._create_model_config(hf_config, mesh_device, "decode")

    @classmethod
    def prefill_model_config(cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device) -> ModelPrefillConfig:
        """Generate prefill configuration for this module.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device the model will be placed later on

        Returns:
            ModelPrefillConfig containing operator configurations for prefill mode
        """
        return cls._create_model_config(hf_config, mesh_device, "prefill")

    @classmethod
    def _forward(cls, x: ttnn.Tensor, sparsity: ttnn.Tensor, cfg: RunDecodeConfig) -> ttnn.Tensor:
        assert x.memory_config() == cfg["input_memory_config"], f"{x.memory_config()} != {cfg['input_memory_config']}"

        _, _, num_tokens, hidden_size = x.shape
        num_sparse_blocks = num_tokens // SPARSITY_BLOCK_SIZE
        x = ttnn.reshape(x, shape=(1, num_sparse_blocks, SPARSITY_BLOCK_SIZE, hidden_size))

        # Gate and up projections
        w1_out = ttnn.sparse_matmul(x, sparsity=sparsity, **cfg["w1_experts"])
        w3_out = ttnn.sparse_matmul(x, sparsity=sparsity, **cfg["w3_experts"])

        # Apply activation and multiply
        activated = ttnn.mul(w1_out, w3_out, **cfg["mul_experts"])
        ttnn.deallocate(w1_out)
        ttnn.deallocate(w3_out)

        # Reshape for down projection
        # activated.shape = Shape([1, 4, 1, 8, 32, 2048])
        activated = ttnn.squeeze(activated, 0)
        activated = ttnn.squeeze(activated, 1)

        # Down projection
        output = ttnn.sparse_matmul(activated, sparsity=sparsity, **cfg["w2_experts"])
        ttnn.deallocate(activated)

        # Reshape for output
        output = ttnn.permute(output, (1, 0, 2, 3))
        output = ttnn.reshape(output, shape=(1, cfg["num_experts_per_device"], num_tokens, hidden_size))

        assert output.memory_config() == cfg["output_memory_config"]
        return output

    @classmethod
    def forward_decode(cls, x: ttnn.Tensor, sparsity: ttnn.Tensor, cfg: RunDecodeConfig) -> ttnn.Tensor:
        return cls._forward(x, sparsity, cfg)

    @classmethod
    def forward_prefill(cls, x: ttnn.Tensor, sparsity: ttnn.Tensor, cfg: RunPrefillConfig) -> ttnn.Tensor:
        return cls._forward(x, sparsity, cfg)
