# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from loguru import logger
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.tt.ccl import CCL
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
    COMPUTE_KERNEL_CONFIG_HIFI4_NOFP32_ACC,
    USERS_PER_ROW,
    even_int_div,
    get_state_dicts,
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
from models.demos.deepseek_v3.utils.shared_state_addon import SharedStateAddOn


def _has_distinct_buffer(a: ttnn.Tensor, b: ttnn.Tensor) -> bool:
    try:
        return a.buffer_address() != b.buffer_address()
    except Exception:
        return a is not b


class DistributedRMSNorm(SharedStateAddOn, RMSNormBase):
    @classmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dicts: tuple[dict[str, torch.Tensor] | None, ...],
        output_path: Path,
        mesh_device: ttnn.Device,
    ) -> WeightConfig:
        torch_metaweight = get_state_dicts(state_dicts, "weight", dtype=torch.bfloat16)
        num_shards = torch_metaweight.shape[0]
        assert num_shards == mesh_device.shape[0], "Number of state dicts does not match the number of rows."

        # Save to disk with standard naming - "rmsnorm" must match the op name used in the model config
        # so that RunConfig can populate it with the actual weight tensors at runtime
        return {
            "rms_norm_post_all_gather": {
                "weight": shard_and_save(
                    output_path / "rmsnorm.weight",
                    torch_metaweight.reshape(
                        (num_shards, 1, -1, ttnn.TILE_SIZE)
                    ),  # Reshape to tile width sticks for optimal performance
                    shard_dims=(0, -2),
                    mesh_device=mesh_device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
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
                ttnn.core.roundup(USERS_PER_ROW, ttnn.TILE_SIZE),
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
                compute_kernel_config=COMPUTE_KERNEL_CONFIG_HIFI4_NOFP32_ACC,
            ),
            "all_gather": AllGatherAsyncConfig(
                dim=3,
                cluster_axis=1,
                mesh_device=MeshDeviceStub(mesh_device.shape),
                memory_config=rms_norm_stats_memory_config,
            ),
            "rms_norm_post_all_gather": RMSNormPostAllGatherConfig(
                epsilon=hf_config.rms_norm_eps,
                weight=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                dtype=ttnn.bfloat16,
                compute_kernel_config=COMPUTE_KERNEL_CONFIG_HIFI4_NOFP32_ACC,
            ),
        }

    @classmethod
    def create_state(cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device, ccl: CCL) -> ModelState:
        """Create the model state for this module.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device the model will be placed later on
            ccl: CCL instance for async CCLs

        Returns:
            ModelState containing the state information for this module
        """
        return {
            MESH_DEVICE_STATE_DICT_KEY: mesh_device,
            "ccl": ccl,
        }

    @classmethod
    def create_shared_state(cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device) -> ModelState:
        """Create shared model state containing tensors that are constant across all instances.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device the model will be placed later on

        Returns:
            ModelState containing shared tensors
        """
        logger.info("Creating RMSNorm shared state: 32x32 tiled tensor and semaphore...")
        tensor_start = perf_counter()

        # Create a 32x32 sharded tensor on core (0,0) as required by fused_rms_minimal
        # Shape per device (32,32) and sharded with shard shape (32,32) on core (0,0)
        shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
            (32, 32),
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        sharded_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

        persistent_tensor = ttnn.from_torch(
            torch.zeros((32, 32), dtype=torch.bfloat16),
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.bfloat16,
            memory_config=sharded_mem_config,
            layout=ttnn.TILE_LAYOUT,
        )

        # Create semaphore for fused_rms_minimal
        grid = mesh_device.compute_with_storage_grid_size()
        num_cores = grid.x * grid.y
        core_range_set = ttnn.num_cores_to_corerangeset(num_cores, grid, row_wise=True)
        semaphore = ttnn.create_global_semaphore(mesh_device, core_range_set, 0)

        # Synchronize to ensure semaphore is created
        ttnn.synchronize_device(mesh_device)

        logger.info(f"Created RMSNorm persistent tensor and semaphore in {perf_counter() - tensor_start:.2f}s")

        return {
            "persistent_tensor": persistent_tensor,
            "semaphore": semaphore,
        }

    @staticmethod
    def _fwd_rms_norm_pre_all_gather(x: ttnn.Tensor, cfg: dict, program_config: Any) -> ttnn.Tensor:
        """Wrapper for distributed RMS norm part 1: compute local statistics.

        Args:
            x: Input tensor
            cfg: Config for rms_norm_pre_all_gather (cfg["rms_norm_pre_all_gather"])
            program_config: Program config (computed externally via _get_pc)

        Returns:
            Local statistics tensor
        """
        return ttnn.rms_norm_pre_all_gather(x, program_config=program_config, **cfg["rms_norm_pre_all_gather"])

    @staticmethod
    def _fwd_all_gather_stats(stats: ttnn.Tensor, cfg: dict, ccl) -> ttnn.Tensor:
        """Wrapper for all-gather statistics.

        Args:
            stats: Local statistics tensor
            cfg: Config containing all_gather settings (cfg["all_gather"])
            ccl: CCL runtime object

        Returns:
            Gathered statistics tensor
        """
        return ttnn.experimental.all_gather_async(stats, **ccl.populate_all_gather_runtime_args(cfg["all_gather"]))

    @staticmethod
    def _fwd_rms_norm_post_all_gather(
        x: ttnn.Tensor, stats: ttnn.Tensor, cfg: dict, program_config: Any
    ) -> ttnn.Tensor:
        """Wrapper for distributed RMS norm part 2: apply normalization with gathered stats.

        Args:
            x: Input tensor (same as input to pre_all_gather)
            stats: Gathered statistics tensor
            cfg: Config for rms_norm_post_all_gather (cfg["rms_norm_post_all_gather"])
            program_config: Program config (computed externally via _get_pc)

        Returns:
            Normalized output tensor
        """
        return ttnn.rms_norm_post_all_gather(x, stats, program_config=program_config, **cfg["rms_norm_post_all_gather"])

    @classmethod
    def _rmsnorm_forward_decode(
        cls,
        x: ttnn.Tensor,
        cfg: RunDecodeConfig,
        memory_config: ttnn.MemoryConfig,
        output_memory_config: ttnn.MemoryConfig,
        residual: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """Forward pass of the RMSNorm for decode mode.

        Args:
            x: Input tensor
            cfg: RunDecodeConfig containing weights and op configurations
            memory_config: Memory configuration for the input tensor
            output_memory_config: Memory configuration for the output tensor
            residual: Optional residual tensor to add

        Returns:
            Output tensor after RMSNorm computation
        """
        tensor_in = ttnn.to_memory_config(x, memory_config)

        program_config = cls._get_pc(memory_config)
        # Get mesh device from the all_gather config (it's available there)
        tt_out = ttnn.fused_rms_minimal(
            tensor_in,
            program_config,
            1,
            cfg["all_gather"]["mesh_device"],
            cfg["semaphore"],
            topology=ttnn.Topology.Linear,
            residual_input_tensor=residual,
            num_links=1,
            epsilon=cfg["rms_norm_post_all_gather"]["epsilon"],
            weight=cfg["rms_norm_post_all_gather"]["weight"],
            stats=cfg["persistent_tensor"],
            memory_config=output_memory_config,
            use_noc1_only=False,
        )

        if _has_distinct_buffer(x, tensor_in):
            ttnn.deallocate(tensor_in)

        return tt_out, residual

    @classmethod
    def _rmsnorm_forward_prefill(cls, x: ttnn.Tensor, cfg: RunPrefillConfig) -> ttnn.Tensor:
        """Forward pass of the RMSNorm for prefill mode.

        Args:
            x: Input tensor
            cfg: RunPrefillConfig containing weights and op configurations

        Returns:
            Output tensor after RMSNorm computation
        """

        program_config = cls._get_pc(x.memory_config())
        # Run distributed rmsnorm part 1
        tt_stats = cls._fwd_rms_norm_pre_all_gather(x, cfg, program_config=program_config)

        # AllGather stats
        ccl = cfg["ccl"]
        tt_gathered_stats = cls._fwd_all_gather_stats(tt_stats, cfg, ccl)
        ttnn.deallocate(tt_stats)

        # Run distributed rmsnorm part 2
        tt_out = cls._fwd_rms_norm_post_all_gather(x, tt_gathered_stats, cfg, program_config=program_config)
        ttnn.deallocate(tt_gathered_stats)

        return tt_out
