# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import torch
import ttnn.experimental
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.tt.ccl_1d import CCL1D
from models.demos.deepseek_v3.tt.rms_norm.rms_norm_base import RMSNormBase
from models.demos.deepseek_v3.utils.config_dataclass import (
    AllGatherAsyncConfig,
    FromWeightConfig,
    MeshDeviceStub,
    OpConfigBase,
    RMSNormPostAllGatherConfig,
    RMSNormPreAllGatherConfig,
)
from models.demos.deepseek_v3.utils.config_helpers import (
    MAX_BATCH_SIZE,
    even_int_div,
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


class DistributedRMSNorm(RMSNormBase):
    @classmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dicts: tuple[dict[str, torch.Tensor] | None, ...],
        output_path: Path,
        mesh_device: ttnn.Device,
    ) -> WeightConfig:
        torch_metaweight = get_state_dicts(state_dicts, "weight", shape=(hf_config.hidden_size,), dtype=torch.bfloat16)
        num_shards = torch_metaweight.shape[0]
        assert num_shards == mesh_device.shape[0], "Number of state dicts does not match the number of rows."

        tt_weight = ttnn.as_tensor(
            torch_metaweight.reshape(
                (num_shards, 1, -1, ttnn.TILE_SIZE)
            ),  # Reshape to tile width sticks for optimal performance
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=(0, -2)),
        )

        # Save to disk with standard naming - "rmsnorm" must match the op name used in the model config
        # so that RunConfig can populate it with the actual weight tensors at runtime
        return {
            "rms_norm_post_all_gather": {
                "weight": save_and_get_path(output_path / "rmsnorm.weight", tt_weight),
            }
        }

    @classmethod
    def prefill_model_config(cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device) -> ModelPrefillConfig:
        """Generate prefill configuration for this module.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device the model will be placed later on

        Returns:
            ModelPrefillConfig containing operator configurations for prefill mode
        """
        return cls._model_config(
            hf_config=hf_config,
            mesh_device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            rms_norm_stats_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # type: ignore

    @classmethod
    def decode_model_config(cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device) -> ModelDecodeConfig:
        """Generate decode configuration for this module.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device the model will be placed later on

        Returns:
            ModelDecodeConfig containing operator configurations for decode mode
        """
        shard_core_grid = ttnn.CoreGrid(x=4, y=7)
        memory_config = ttnn.create_sharded_memory_config(
            shape=(
                MAX_BATCH_SIZE,
                ttnn.core.roundup(
                    even_int_div(hf_config.hidden_size, shard_core_grid.num_cores * mesh_device.shape[1]),
                    ttnn.TILE_SIZE,
                ),
            ),
            core_grid=shard_core_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        return cls._model_config(
            hf_config=hf_config,
            mesh_device=mesh_device,
            memory_config=memory_config,
            rms_norm_stats_memory_config=ttnn.create_sharded_memory_config(
                shape=[1, 1, ttnn.TILE_SIZE, ttnn.TILE_SIZE * mesh_device.shape[1]],
                core_grid=ttnn.CoreGrid(y=1, x=1),
                strategy=ttnn.ShardStrategy.WIDTH,
            ),
        )  # type: ignore

    @classmethod
    def _model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
        memory_config: ttnn.MemoryConfig,
        rms_norm_stats_memory_config: ttnn.MemoryConfig,
    ) -> dict[str, OpConfigBase]:
        """Generate model configuration for RMSNorm."""
        return {
            "input_memory_config": memory_config,
            "rms_norm_pre_all_gather": RMSNormPreAllGatherConfig(
                dtype=ttnn.bfloat16,
            ),
            "all_gather": AllGatherAsyncConfig(
                dim=3,
                cluster_axis=1,
                mesh_device=MeshDeviceStub(mesh_device.shape),
                memory_config=rms_norm_stats_memory_config,
                topology=ttnn.Topology.Linear,
            ),
            "rms_norm_post_all_gather": RMSNormPostAllGatherConfig(
                epsilon=hf_config.rms_norm_eps,
                weight=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                dtype=ttnn.bfloat16,
            ),
        }

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
            },
        }

    @classmethod
    def _rmsnorm_forward(cls, x: ttnn.Tensor, cfg: RunPrefillConfig | RunDecodeConfig) -> ttnn.Tensor:
        """Forward pass of the embedding.

        Args:
            x: Input tensor (token indices)
            cfg: RunConfig containing weights and op configurations

        Returns:
            Output tensor after embedding lookup
        """

        program_config = cls._get_pc(x.memory_config())
        # Run distributed rmsnorm part 1
        tt_stats = ttnn.rms_norm_pre_all_gather(x, program_config=program_config, **cfg["rms_norm_pre_all_gather"])

        # AllGather stats
        tt_gathered_stats = ttnn.experimental.all_gather_async(tt_stats, **cfg["all_gather"])
        ttnn.deallocate(tt_stats)

        # Run distributed rmsnorm part 2
        tt_out = ttnn.rms_norm_post_all_gather(
            x,
            tt_gathered_stats,
            program_config=program_config,
            **cfg["rms_norm_post_all_gather"],
        )
        ttnn.deallocate(tt_gathered_stats)

        return tt_out
