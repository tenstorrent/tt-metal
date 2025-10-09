# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0


from pathlib import Path

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.tt.ccl import CCL
from models.demos.deepseek_v3.tt.mla.mla1d import MLA1D
from models.demos.deepseek_v3.utils.config_dataclass import (
    AllGatherAsyncConfig,
    MeshDeviceStub,
    ReduceScatterAsyncMinimalConfig,
    SavedWeight,
)
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW
from models.demos.deepseek_v3.utils.run_config import (
    MESH_DEVICE_STATE_DICT_KEY,
    ModelDecodeConfig,
    ModelPrefillConfig,
    ModelState,
    RunDecodeConfig,
    RunPrefillConfig,
    WeightConfig,
)
from models.tt_transformers.tt.common import PagedAttentionConfig


class MLA2D(MLA1D):
    """
    Batch and Sequence-Parallel Multi-Latent Attention Module for 1D tensor parallelism.
    NOTE: This implementation is a temporary one until MLA1D is removed or ring-attention is incorporated, to avoid code duplication and for making improvements in a single place.
    """

    @classmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dicts: tuple[dict[str, torch.Tensor] | None, ...],
        output_path: Path,
        mesh_device: ttnn.Device,
    ) -> WeightConfig:
        (state_dict,) = state_dicts
        assert state_dict is not None, "State dict must be provided for weight conversion."
        return {
            "mla1d": super().convert_weights(hf_config, (state_dict,) * mesh_device.shape[0], output_path, mesh_device)
        }

    @classmethod
    def _convert_weight(
        cls,
        path: Path,
        torch_metaweight: torch.Tensor,
        dims: tuple[int | None, int | None],
        mesh_device: ttnn.MeshDevice,
    ) -> SavedWeight:
        if dims[0] is not None:
            slices = torch.split(torch_metaweight, 1, dim=dims[0])
            assert all(torch.allclose(s1, s2) for s1, s2 in zip(slices[:-1], slices[1:]))
            torch_metaweight = slices[0]
            dims = (None, dims[1])
        return super()._convert_weight(path, torch_metaweight, dims, mesh_device)

    @classmethod
    def prefill_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
    ) -> ModelPrefillConfig:
        super_cfg = super().prefill_model_config(hf_config, mesh_device)
        input_memory_config = super_cfg.pop("input_memory_config")
        return {
            "mla1d": super_cfg,
            "input_memory_config": input_memory_config,
            "seq_ag_prefill": AllGatherAsyncConfig(
                mesh_device=MeshDeviceStub(mesh_device.shape),
                cluster_axis=0,
                dim=2,
                topology=ttnn.Topology.Linear,
            ),
            "seq_rs_prefill": ReduceScatterAsyncMinimalConfig(
                cluster_axis=0,
                dim=2,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=ttnn.Topology.Linear,
            ),
        }

    @classmethod
    def decode_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
    ) -> ModelDecodeConfig:
        super_cfg = super().decode_model_config(hf_config, mesh_device)
        input_memory_config = super_cfg.pop("input_memory_config")
        return {
            "mla1d": super_cfg,
            "input_memory_config": input_memory_config,
        }

    @classmethod
    def create_page_table(
        cls,
        paged_config: PagedAttentionConfig,
        mesh_device: ttnn.MeshDevice,
        page_table: torch.Tensor | None = None,
        batch_size_per_row: int = USERS_PER_ROW,
    ) -> ttnn.Tensor:
        return super().create_page_table(
            paged_config=paged_config,
            mesh_device=mesh_device,
            page_table=page_table,
            batch_size=batch_size_per_row * mesh_device.shape[0],
        )

    @classmethod
    def create_state(
        cls,
        hf_config: PretrainedConfig,
        paged_config: PagedAttentionConfig,
        mesh_device: ttnn.MeshDevice,
        ccl: CCL,
        cache: torch.Tensor | None = None,
    ) -> ModelState:
        return {
            MESH_DEVICE_STATE_DICT_KEY: mesh_device,
            "mla1d": super().create_state(
                hf_config,
                paged_config,
                mesh_device,
                ccl,
                None if cache is None else cache.reshape(mesh_device.shape[0], -1, *cache.shape[1:]),
            ),
            "seq_ag_prefill": {
                "multi_device_global_semaphore": ccl.get_gather_sem(axis=0),
                "barrier_semaphore": ccl.get_barrier_sem(axis=0),
                "num_links": ccl.get_max_links(axis=0),
            },
            "seq_rs_prefill": {
                "multi_device_global_semaphore": ccl.get_reduce_scatter_sem(axis=0),
                "barrier_semaphore": ccl.get_barrier_sem(axis=0),
                "num_links": ccl.get_max_links(axis=0),
            },
        }

    @classmethod
    def forward_decode(
        cls,
        x: ttnn.Tensor,
        position_idxs: ttnn.Tensor,
        cfg: RunDecodeConfig,
        rope_tensors: dict,
        page_table: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Forward pass of MLA in decode mode.

        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            cfg: RunConfig containing weights and op configurations
            position_idxs: List of position indices for the current batch
            rope_tensors: Dictionary containing RoPE tensors
            page_table: Page table tensor for paged attention
        Returns:
            Output tensor after MLA computation

        """
        return super().forward_decode(
            x,
            position_idxs=position_idxs,
            row_idx=None,
            cfg=cfg["mla1d"],
            rope_tensors=rope_tensors,
            page_table=page_table,
        )

    @classmethod
    def forward_prefill(
        cls,
        x: ttnn.Tensor,
        batch_idx: int,
        cfg: RunPrefillConfig,
        rope_tensors: dict,
        page_table: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Forward pass of MLA in prefill mode.

        Args:
            x: Input tensor
            cfg: RunConfig containing weights and op configurations
            batch_idx: Batch index for cache updates (wrt to global batch size)
            rope_tensors: Dictionary containing RoPE tensors
            page_table: Page table tensor for paged attention

        Returns:
            Output tensor after MLP computation
        """

        scale = 1 / cfg["mla1d"]["mesh_shape"][0]

        x_next = ttnn.experimental.all_gather_async(x, **cfg["seq_ag_prefill"])
        x_out = super().forward_prefill(
            x_next,
            batch_idx=batch_idx % USERS_PER_ROW,
            row_idx=batch_idx // USERS_PER_ROW,
            cfg=cfg["mla1d"],
            rope_tensors=rope_tensors,
            page_table=page_table,
        )
        ttnn.deallocate(x_next)

        x_rs = ttnn.experimental.reduce_scatter_minimal_async(x_out, **cfg["seq_rs_prefill"]) * scale
        return x_rs
