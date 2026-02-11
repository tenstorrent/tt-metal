# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.tt.ccl import CCL
from models.demos.deepseek_v3.utils.abstract_module import AbstractModule
from models.demos.deepseek_v3.utils.composite_ops import mesh_scatter
from models.demos.deepseek_v3.utils.config_dataclass import (
    AllGatherAsyncConfig,
    FromWeightConfig,
    LinearConfig,
    MeshDeviceStub,
    OpConfigBase,
)
from models.demos.deepseek_v3.utils.config_helpers import (
    COMPUTE_KERNEL_CONFIG_LOFI,
    SEQ_LEN_CHUNK_SIZE,
    USERS_PER_ROW,
    dram_sharded_weight_config,
    even_int_div,
    find_largest_divisor,
    get_activation_sharding_core_counts_for_dram_matmul,
    get_dram_sharded_matmul_config,
    shard_and_save,
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


class LMHead(AbstractModule):
    """TT implementation of Language model head for Deepseek V3."""

    @dataclass
    class ProgramConfigData(OpConfigBase):
        """Data class for the data for generating the PC for ttnn.linear."""

        hidden_dim: int
        vocab_size: int
        num_devices: int
        core_grid_size: ttnn.CoreCoord

    @classmethod
    def _get_model_dims_from_cfg(cls, hf_config: PretrainedConfig) -> tuple[int, int]:
        """Get the dimensions of the model from the HuggingFace config.

        Args:
            hf_config: HuggingFace model configuration object.

        Returns:
            Tuple containing the hidden dimension and vocab_size of the LMHHead.
        """

        return hf_config.hidden_size, hf_config.vocab_size

    @classmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dicts: tuple[dict[str, torch.Tensor], ...],
        output_path: Path,
        mesh_device: ttnn.Device,
    ) -> WeightConfig:
        assert len(state_dicts) == 1, "Only one non-padding state dict is expected for LMHead conversion"
        (state_dict,) = state_dicts

        hidden_dim, vocab_size = cls._get_model_dims_from_cfg(hf_config)

        weight_tensor = state_dict["weight"].permute(
            1, 0
        )  # In torch the weights are in (out_features, in_features) format
        assert weight_tensor.shape == (hidden_dim, vocab_size)

        return {
            "linear": {
                "input_tensor_b": shard_and_save(
                    output_path / "linear.input_tensor_b",
                    weight_tensor,
                    shard_dims=(-1, -1),
                    mesh_device=mesh_device,
                    dtype=ttnn.bfloat4_b,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=dram_sharded_weight_config(
                        hidden_dim,
                        even_int_div(vocab_size, mesh_device.get_num_devices()),
                        mesh_device.dram_grid_size(),
                    ),
                )
            }
        }

    @classmethod
    def _get_decode_activation_memory_config(
        cls, per_device_width: int, activation_sharding_num_cores: int, mesh_device: ttnn.Device
    ) -> ttnn.MemoryConfig:
        """Get the memory config for an activation tensor in decode mode."""
        return ttnn.create_sharded_memory_config_(
            shape=(
                ttnn.core.roundup(USERS_PER_ROW, ttnn.TILE_SIZE),
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
    def decode_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
        input_row_idx: int,
        input_num_cores: int | None = None,
        output_num_cores: int | None = None,
    ) -> ModelDecodeConfig:
        """Generate decode configuration for this module.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device the model will be placed later on

        Returns:
            ModelDecodeConfig containing operator configurations for decode mode
        """

        hidden_dim, vocab_size = cls._get_model_dims_from_cfg(hf_config)

        # Calculate device metrics
        num_devices = mesh_device.get_num_devices()
        max_num_cores = mesh_device.core_grid.x * mesh_device.core_grid.y
        input_num_cores = input_num_cores or max(
            get_activation_sharding_core_counts_for_dram_matmul(hidden_dim, max_num_cores)
        )

        output_num_cores = output_num_cores or max(
            get_activation_sharding_core_counts_for_dram_matmul(even_int_div(vocab_size, num_devices), max_num_cores)
        )
        assert (
            input_num_cores <= max_num_cores
        ), "input_num_cores must be less than or equal to the maximum number of cores"
        assert (
            output_num_cores <= max_num_cores
        ), "output_num_cores must be less than or equal to the maximum number of cores"
        assert hidden_dim % input_num_cores == 0, "input_num_cores must divide the input tensor width evenly"
        assert (
            even_int_div(hidden_dim, num_devices) % output_num_cores == 0
        ), "output_num_cores must divide the output tensor width evenly"

        # Calculate input and output memory configurations

        input_memory_config = cls._get_decode_activation_memory_config(hidden_dim, input_num_cores, mesh_device)
        output_memory_config = cls._get_decode_activation_memory_config(
            even_int_div(vocab_size, num_devices), output_num_cores, mesh_device
        )

        # Construct the config
        return {
            "mesh_scatter": {
                "mesh_shape": tuple(mesh_device.shape),
                "scatter_idx": (input_row_idx, None),
            },
            "linear": LinearConfig(
                input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                program_config=get_dram_sharded_matmul_config(
                    USERS_PER_ROW, hidden_dim, even_int_div(vocab_size, num_devices), input_num_cores, output_num_cores
                ),
                compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
            ),
            "all_gather": AllGatherAsyncConfig(
                mesh_device=mesh_device,
                cluster_axis=1,
                dim=-1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=ttnn.Topology.Linear,
            ),
            "input_memory_config": input_memory_config,
            "output_memory_config": output_memory_config,
        }

    @classmethod
    def prefill_model_config(
        cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device, input_row_idx: int
    ) -> ModelPrefillConfig:
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
        num_devices = mesh_device.get_num_devices()

        # Extract dimensions from HF config
        hidden_dim, vocab_size = cls._get_model_dims_from_cfg(hf_config)

        # Construct the config
        return {
            "max_rows": SEQ_LEN_CHUNK_SIZE,
            "mesh_scatter": {
                "mesh_shape": tuple(mesh_device.shape),
                "scatter_idx": (input_row_idx, None),
            },
            "linear_pc_gen": LMHead.ProgramConfigData(
                hidden_dim=hidden_dim,
                vocab_size=vocab_size,
                num_devices=num_devices,
                core_grid_size=matmul_core_grid_size,
            ),
            "linear": LinearConfig(
                input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
            ),
            "all_gather": AllGatherAsyncConfig(
                mesh_device=mesh_device,
                cluster_axis=1,
                dim=-1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=ttnn.Topology.Linear,
            ),
            "input_memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "output_memory_config": ttnn.DRAM_MEMORY_CONFIG,
        }

    @classmethod
    def create_state(cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device, ccl: CCL) -> ModelState:
        # Store CCL object for runtime semaphore initialization
        return {
            MESH_DEVICE_STATE_DICT_KEY: mesh_device,
            "ccl": ccl,
        }

    @classmethod
    def forward_decode(cls, x: ttnn.Tensor, cfg: RunDecodeConfig) -> ttnn.Tensor:
        assert x.memory_config() == cfg["input_memory_config"], f"{x.memory_config()} != {cfg['input_memory_config']}"

        mesh_scatter(x, **cfg["mesh_scatter"])
        output = ttnn.linear(x, **cfg["linear"])

        assert output.memory_config() == cfg["output_memory_config"]

        return output

    @classmethod
    def _get_prefill_pc(
        cls, seq_len: int, hidden_dim: int, vocab_size: int, num_devices: int, core_grid_size: ttnn.CoreCoord
    ) -> Any:
        """Get the program config for linear layers in prefill mode based on sequence length."""

        per_device_in_features, per_device_out_features = hidden_dim, even_int_div(vocab_size, num_devices)

        per_core_M_tiles = ttnn.core.divup(seq_len, ttnn.TILE_SIZE * core_grid_size.y)
        K_tiles = ttnn.core.divup(per_device_in_features, ttnn.TILE_SIZE)
        per_core_N_tiles = ttnn.core.divup(per_device_out_features, ttnn.TILE_SIZE * core_grid_size.x)

        out_subblock_h = 1
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=core_grid_size,
            in0_block_w=find_largest_divisor(K_tiles),
            out_subblock_h=out_subblock_h,
            out_subblock_w=find_largest_divisor(
                per_core_N_tiles,
                out_subblock_h,
            ),
            per_core_M=per_core_M_tiles,
            per_core_N=per_core_N_tiles,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )

    @classmethod
    def forward_prefill(cls, x: ttnn.Tensor, cfg: RunPrefillConfig) -> ttnn.Tensor:
        assert x.memory_config() == cfg["input_memory_config"], f"{x.memory_config()} != {cfg['input_memory_config']}"

        mesh_scatter(x, **cfg["mesh_scatter"])

        _, _, seq_len, _ = x.shape

        # Use effective sequence length (chunk size) for program config to avoid L1 overflow
        effective_seq_len = min(seq_len, cfg["max_rows"])

        if seq_len > cfg["max_rows"]:  # For large sequence lengths, process the input in chunks
            x = ttnn.reshape(x, [1, even_int_div(seq_len, cfg["max_rows"]), cfg["max_rows"], -1])

        output = ttnn.linear(
            x, program_config=cls._get_prefill_pc(seq_len=effective_seq_len, **cfg["linear_pc_gen"]), **cfg["linear"]
        )
        ttnn.deallocate(x)

        # De-chunk the output if the input was chunked
        _, num_chunks, _, output_dim = output.shape
        if num_chunks > 1:
            output = ttnn.reshape(output, [1, 1, -1, output_dim])

        assert output.memory_config() == cfg["output_memory_config"]
        return output
