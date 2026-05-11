# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.mistral_small_4_119B.tt.rms_norm.rms_norm_base import RMSNormBase
from models.demos.mistral_small_4_119B.tt_utils.ccl import CCL
from models.demos.mistral_small_4_119B.tt_utils.config_dataclass import (
    AllGatherAsyncConfig,
    FromWeightConfig,
    MeshDeviceStub,
    OpConfigBase,
    RMSNormPostAllGatherConfig,
    RMSNormPreAllGatherConfig,
)
from models.demos.mistral_small_4_119B.tt_utils.config_helpers import (
    COMPUTE_KERNEL_CONFIG_HIFI4_NOFP32_ACC,
    all_gather_mesh_extent_on_cluster_axis,
    get_state_dicts,
    shard_and_save,
)
from models.demos.mistral_small_4_119B.tt_utils.run_config import (
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
        del hf_config
        torch_metaweight = get_state_dicts(state_dicts, "weight", dtype=torch.bfloat16)
        num_shards = torch_metaweight.shape[0]
        assert num_shards == mesh_device.shape[0], "Number of state dicts does not match the number of rows."
        return {
            "rms_norm_post_all_gather": {
                "weight": shard_and_save(
                    output_path / "rmsnorm.weight",
                    torch_metaweight.reshape((num_shards, 1, -1, ttnn.TILE_SIZE)),
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
        return cls._model_config(
            hf_config=hf_config,
            mesh_device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            rms_norm_stats_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    @classmethod
    def decode_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
        batch_size_per_row: int,
    ) -> ModelDecodeConfig:
        del batch_size_per_row
        return cls._model_config(
            hf_config=hf_config,
            mesh_device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            rms_norm_stats_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    @classmethod
    def _model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
        memory_config: ttnn.MemoryConfig,
        rms_norm_stats_memory_config: ttnn.MemoryConfig,
    ) -> dict[str, OpConfigBase]:
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
        del hf_config
        return {MESH_DEVICE_STATE_DICT_KEY: mesh_device, "ccl": ccl}

    @staticmethod
    def _fwd_rms_norm_pre_all_gather(x: ttnn.Tensor, cfg: dict, program_config: Any) -> ttnn.Tensor:
        return ttnn.rms_norm_pre_all_gather(x, program_config=program_config, **cfg["rms_norm_pre_all_gather"])

    @staticmethod
    def _fwd_all_gather_stats(stats: ttnn.Tensor, cfg: dict, ccl: CCL | None) -> ttnn.Tensor:
        ag = cfg["all_gather"]
        if all_gather_mesh_extent_on_cluster_axis(ag) <= 1:
            return stats
        assert ccl is not None
        return ttnn.experimental.all_gather_async(stats, **ccl.populate_all_gather_runtime_args(ag))

    @staticmethod
    def _fwd_rms_norm_post_all_gather(
        x: ttnn.Tensor, stats: ttnn.Tensor, cfg: dict, program_config: Any
    ) -> ttnn.Tensor:
        return ttnn.rms_norm_post_all_gather(x, stats, program_config=program_config, **cfg["rms_norm_post_all_gather"])

    @classmethod
    def _rmsnorm_forward(cls, x: ttnn.Tensor, cfg: RunPrefillConfig | RunDecodeConfig) -> ttnn.Tensor:
        program_config = cls._get_pc(x.memory_config())
        tt_stats = cls._fwd_rms_norm_pre_all_gather(x, cfg, program_config=program_config)
        ccl = cfg["ccl"]
        tt_gathered_stats = cls._fwd_all_gather_stats(tt_stats, cfg, ccl)
        if tt_gathered_stats is not tt_stats:
            ttnn.deallocate(tt_stats)
        tt_out = cls._fwd_rms_norm_post_all_gather(x, tt_gathered_stats, cfg, program_config=program_config)
        ttnn.deallocate(tt_gathered_stats)
        return tt_out
