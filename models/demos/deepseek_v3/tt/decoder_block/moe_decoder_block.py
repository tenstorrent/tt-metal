# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from pathlib import Path

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.tt.ccl_1d import CCL1D
from models.demos.deepseek_v3.tt.decoder_block.decoder_block_base import DecoderBlockBase
from models.demos.deepseek_v3.tt.mlp.shared_expert import SharedExpert
from models.demos.deepseek_v3.tt.moe import MoE
from models.demos.deepseek_v3.utils.config_dataclass import AllGatherAsyncConfig, ConvertToDPConfig
from models.demos.deepseek_v3.utils.run_config import (
    ModelDecodeConfig,
    ModelPrefillConfig,
    ModelState,
    RunDecodeConfig,
    RunPrefillConfig,
    WeightConfig,
)


class MoEDecoderBlock(DecoderBlockBase):
    @classmethod
    @abstractmethod
    def convert_mlp_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dicts: tuple[dict[str, torch.Tensor] | None],
        output_path: Path,
        mesh_device: ttnn.MeshDevice,
    ) -> WeightConfig:
        assert mesh_device.shape[0] == len(
            state_dicts
        ), "Number of state dicts must match the number of mesh device rows"
        return {
            "shared_expert": SharedExpert.convert_weights(
                hf_config, state_dicts, output_path / "shared_expert", mesh_device
            ),
            "moe": [
                (
                    MoE.convert_weights(hf_config, state_dict, output_path / "moe", mesh_device)
                    if state_dict is not None
                    else None
                )
                for state_dict in state_dicts
            ],
        }

    @classmethod
    @abstractmethod
    def prefill_mlp_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
    ) -> ModelPrefillConfig:
        return {
            "moe": MoE.prefill_model_config(hf_config, mesh_device),
            "shared_expert": SharedExpert.prefill_model_config(hf_config, mesh_device),
            "apply_dp": ConvertToDPConfig(
                mesh_device=mesh_device,
                dim=0,  # Batch dimension
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cluster_axis=0,
            ),
            "revert_dp": AllGatherAsyncConfig(
                mesh_device=mesh_device,
                dim=0,  # Batch dimension
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cluster_axis=0,
                topology=ttnn.Topology.Linear,
                num_links=3,
            ),
        }

    @classmethod
    @abstractmethod
    def decode_mlp_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
    ) -> ModelDecodeConfig:
        return {
            "moe": MoE.decode_model_config(hf_config, mesh_device),
            "shared_expert": SharedExpert.decode_model_config(hf_config, mesh_device),
            "apply_dp": ConvertToDPConfig(
                mesh_device=mesh_device,
                dim=0,  # Batch dimension
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cluster_axis=0,
            ),
            "revert_dp": AllGatherAsyncConfig(
                mesh_device=mesh_device,
                dim=0,  # Batch dimension
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cluster_axis=0,
                topology=ttnn.Topology.Linear,
                num_links=3,
            ),
        }

    @classmethod
    @abstractmethod
    def create_mlp_state(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
        ccl: CCL1D,
    ) -> ModelState:
        return {
            "moe": MoE.create_state(hf_config, mesh_device, ccl),
            "shared_expert": SharedExpert.create_state(hf_config, mesh_device, ccl),
            "apply_dp": {
                "global_semaphore": ccl.get_semaphore(0),
            },
            "revert_dp": {
                "multi_device_global_semaphore": ccl.get_semaphore(0),
            },
        }

    @classmethod
    def apply_data_parallelism(
        cls,
        x: ttnn.Tensor,
        row_idx: int,
        mesh_device: ttnn.MeshDevice,
        dim: int,
        memory_config: ttnn.MemoryConfig,
        cluster_axis: int = 0,
        global_semaphore=None,
    ) -> ttnn.Tensor:
        """Apply data parallelism by broadcasting from source row and partitioning across mesh."""
        # First broadcast from row_idx to all rows in the mesh
        x_broadcasted = cls._broadcast_row_to_mesh(x, mesh_device, row_idx, global_semaphore)

        # Then partition the batch across the mesh
        x_partitioned = cls._partition_batch_on_mesh(x_broadcasted, dim, memory_config, cluster_axis)

        return x_partitioned

    @classmethod
    def _broadcast_row_to_mesh(
        cls, tt_input: ttnn.Tensor, mesh_device: ttnn.MeshDevice, src_row: int, semaphore
    ) -> ttnn.Tensor:
        """Broadcast data from a source row to all other rows in the mesh."""
        mesh_shape = tuple(mesh_device.shape)
        # Broadcast from src_row to mesh
        # Loop over rows, skip src_row
        for row_dim_idx in range(mesh_shape[0]):
            if row_dim_idx == src_row:
                continue

            # Do p2p transfer from nodes in src_row to other nodes in that column
            for col_dim_idx in range(mesh_shape[1]):
                source_coord = ttnn.MeshCoordinate(src_row, col_dim_idx)
                dest_coord = ttnn.MeshCoordinate(row_dim_idx, col_dim_idx)

                ttnn.point_to_point(
                    tt_input,
                    dest_coord,
                    source_coord,
                    ttnn.Topology.Linear,
                    semaphore,
                    optional_output_tensor=tt_input,
                )

        return tt_input

    @classmethod
    def _partition_batch_on_mesh(
        cls, tt_input: ttnn.Tensor, dim: int, memory_config: ttnn.MemoryConfig, cluster_axis: int = 0
    ) -> ttnn.Tensor:
        """Partition tensor along specified dimension across mesh cluster axis."""
        tt_out_tensor = ttnn.mesh_partition(
            tt_input,
            dim,
            cluster_axis=cluster_axis,
            memory_config=memory_config,
        )
        return tt_out_tensor

    @classmethod
    def revert_data_parallelism(cls, x: ttnn.Tensor, row_idx: int, **cfg) -> ttnn.Tensor:
        """Revert data parallelism by gathering partitioned tensor back to original form."""
        # Gather tensor along specified dimension across mesh cluster axis
        tt_out_tensor = ttnn.experimental.all_gather_async(x, **cfg)
        return tt_out_tensor

    @classmethod
    @abstractmethod
    def forward_mlp_prefill(cls, x: ttnn.Tensor, row_idx: int, cfg: RunPrefillConfig) -> ttnn.Tensor:
        num_tokens_per_row = x.shape[-3] * x.shape[-2]
        moe_input = x
        apply_dp = num_tokens_per_row > 32
        if apply_dp:
            moe_input = cls.apply_data_parallelism(x, row_idx, **cfg["apply_dp"])
        mlp_out = MoE.forward_prefill(moe_input, cfg["moe"][row_idx])
        if apply_dp:
            mlp_out = cls.revert_data_parallelism(mlp_out, row_idx, **cfg["revert_dp"])
        x = ttnn.to_memory_config(x, **cfg["mlp"]["input_reshard"])
        mlp_out += SharedExpert.forward_prefill(x, cfg["shared_expert"])
        return mlp_out

    @classmethod
    @abstractmethod
    def forward_mlp_decode(cls, x: ttnn.Tensor, row_idx: int, cfg: RunDecodeConfig) -> ttnn.Tensor:
        num_tokens_per_row = x.shape[-3] * x.shape[-2]
        moe_input = x
        apply_dp = num_tokens_per_row > 32
        if apply_dp:
            moe_input = cls.apply_data_parallelism(x, row_idx, **cfg["apply_dp"])
        mlp_out = MoE.forward_decode(moe_input, cfg["moe"][row_idx])
        if apply_dp:
            mlp_out = cls.revert_data_parallelism(mlp_out, row_idx, **cfg["revert_dp"])
        x = ttnn.to_memory_config(x, **cfg["mlp"]["input_reshard"])
        mlp_out += SharedExpert.forward_decode(x, cfg["shared_expert"])
        return mlp_out
