# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import math
from pathlib import Path
from typing import Any

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.tt.ccl_1d import CCL1D
from models.demos.deepseek_v3.tt.decoder_block.decoder_block import DecoderBlock
from models.demos.deepseek_v3.tt.embedding_1d import Embedding1D
from models.demos.deepseek_v3.tt.lm_head import LMHead
from models.demos.deepseek_v3.tt.rms_norm.distributed_rms_norm import DistributedRMSNorm
from models.demos.deepseek_v3.utils.abstract_module import AbstractModule
from models.demos.deepseek_v3.utils.config_dataclass import PointToPointConfig, ReshardConfig
from models.demos.deepseek_v3.utils.config_helpers import get_mesh_coords, sub_state_dict
from models.demos.deepseek_v3.utils.run_config import (
    ModelDecodeConfig,
    ModelPrefillConfig,
    ModelState,
    RunDecodeConfig,
    RunPrefillConfig,
    WeightConfig,
)
from models.demos.deepseek_v3.utils.shared_state_addon import SharedStateAddOn
from models.tt_transformers.tt.common import PagedAttentionConfig


class Model1D(SharedStateAddOn, AbstractModule):
    NUM_MLP_META_LAYERS = 1
    NUM_MLP_ROWS = 3
    NUM_MOE_META_LAYERS = 15
    NUM_MOE_ROWS = 4

    @classmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dict: dict[str, torch.Tensor],
        output_path: Path,
        mesh_device: ttnn.MeshDevice,
        state_dict_prefix: str = "",
    ) -> WeightConfig:
        """Convert weights for the 1D model."""

        mesh_shape = list(mesh_device.shape)

        # Create the state dicts for the MLP decoder block
        mlp_meta_layer_mapping = cls.create_meta_layer_mapping(
            hf_config.first_k_dense_replace, mesh_shape[0]
        )  # [num_meta_layers, num_rows]
        assert len(mlp_meta_layer_mapping) == cls.NUM_MLP_META_LAYERS, "Unexpected number of meta layers for MLP."

        mlp_decoder_block_state_dicts = [
            [
                sub_state_dict(state_dict, state_dict_prefix + f"layers.{layer_idx}.")
                if layer_idx is not None
                else None
                for layer_idx in mapping
            ]
            for mapping in mlp_meta_layer_mapping
        ]

        return {
            "embedding": Embedding1D.convert_weights(
                hf_config,
                [sub_state_dict(state_dict, state_dict_prefix + "embed_tokens.")],
                output_path / "embedding",
                mesh_device,
            ),
            "mlp_decoder_block": [
                DecoderBlock.convert_weights(
                    hf_config, mlp_decoder_block_state_dicts[ml], output_path / "mlp_decoder_block", mesh_device
                )
                for ml in range(cls.NUM_MLP_META_LAYERS)
            ],
            "norm": DistributedRMSNorm.convert_weights(
                hf_config,
                [sub_state_dict(state_dict, state_dict_prefix + "norm.")] * mesh_shape[0],
                output_path / "norm",
                mesh_device,
            ),
            "lm_head": LMHead.convert_weights(
                hf_config, [sub_state_dict(state_dict, "lm_head.")], output_path / "lm_head", mesh_device
            ),
        }

    @classmethod
    def create_meta_layer_mapping(cls, num_layers: int, num_rows: int) -> list[list[int | None]]:
        """Distribute `num_layers` evenly across `num_rows`, returning a
        list of rows where each element is either a layer index or None
        (if padding is needed to fill the structure). The result is
        transposed such that each inner list corresponds to a layer
        position across all rows.

        Returns:
            A list of lists of shape [num_meta_layers][num_rows], where
            each element is an int (layer index) or None (padding).
        """

        if num_rows <= 0:
            raise ValueError("Number of rows must be greater than zero.")
        if num_layers < 0:
            raise ValueError("Number of layers cannot be negative.")

        num_meta_layers = math.ceil(num_layers / num_rows)
        total_slots = num_meta_layers * num_rows

        # Create a flat list of layers, padding with -1
        padded_layers = list(range(num_layers)) + [-1] * (total_slots - num_layers)

        # Reshape into [num_rows, num_meta_layers]
        mapping = torch.tensor(padded_layers).reshape(num_rows, num_meta_layers)

        # Transpose to [num_meta_layers, num_rows] and convert to list of lists
        transposed = mapping.T.tolist()

        # Replace -1s with None
        return [[layer if layer != -1 else None for layer in row] for row in transposed]

    @classmethod
    def prefill_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
    ) -> ModelPrefillConfig:
        """Create the model configuration for prefill mode."""

        mesh_shape = list(mesh_device.shape)

        return {
            "hf_config": hf_config,
            "mesh_shape": mesh_shape,
            "embedding": Embedding1D.prefill_model_config(hf_config, mesh_device),
            "mlp_decoder_block": [
                DecoderBlock.prefill_model_config(
                    hf_config,
                    mesh_device,
                    is_padding_layer=None,
                )
                for _ in range(cls.NUM_MLP_META_LAYERS)
            ],
            "transfer_row": PointToPointConfig(
                topology=ttnn.Topology.Linear,
            ),
            "norm": DistributedRMSNorm.prefill_model_config(hf_config, mesh_device),
        }

    @classmethod
    def decode_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
    ) -> ModelDecodeConfig:
        """Create the model configuration for decode mode."""

        mesh_shape = list(mesh_device.shape)

        norm_config = DistributedRMSNorm.decode_model_config(hf_config, mesh_device)

        return {
            "hf_config": hf_config,
            "mesh_shape": mesh_shape,
            "embedding": Embedding1D.decode_model_config(hf_config, mesh_device),
            "mlp_decoder_block": [
                DecoderBlock.decode_model_config(
                    hf_config,
                    mesh_device,
                    is_padding_layer=None,
                )
                for _ in range(cls.NUM_MLP_META_LAYERS)
            ],
            "transfer_row": PointToPointConfig(
                topology=ttnn.Topology.Linear,
            ),
            "norm_reshard": ReshardConfig(memory_config=norm_config["input_memory_config"]),
            "norm": norm_config,
            "lm_head": LMHead.decode_model_config(hf_config, mesh_device, 0),
        }

    @classmethod
    def create_shared_state(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
    ) -> ModelState:
        return {
            "mlp_decoder_block": [
                DecoderBlock.create_shared_state(hf_config, mesh_device, is_padding_layer=None)
                for _ in range(cls.NUM_MLP_META_LAYERS)
            ],
        }

    @classmethod
    def create_state(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
        paged_config: PagedAttentionConfig,
        ccl: CCL1D,
    ) -> Any:
        """Create the state for the 1D model."""

        return {
            "embedding": Embedding1D.create_state(hf_config, mesh_device, ccl),
            "mlp_decoder_block": [
                DecoderBlock.create_state(hf_config, mesh_device, paged_config, is_padding_layer=None, ccl=ccl)
                for _ in range(cls.NUM_MLP_META_LAYERS)
            ],
            "norm": DistributedRMSNorm.create_state(hf_config, mesh_device, ccl),
            "lm_head": LMHead.create_state(hf_config, mesh_device, ccl),
        }

    @classmethod
    def transfer_row(
        cls,
        x: ttnn.Tensor,
        src_row_idx: int,
        dst_row_ix: int,
        cfg: RunDecodeConfig | RunPrefillConfig,
    ) -> ttnn.Tensor:
        """Transfer a row of data from one row to another in the mesh.
        Args:
            x: Input tensor to transfer
            src_row_idx: Source row index
            dst_row_ix: Destination row index
            cfg: Run configuration containing mesh shape and other parameters
        Returns:
            The tensor after transferring the row
        """

        mesh_shape = cfg["mesh_shape"]

        src_row = get_mesh_coords(mesh_shape, src_row_idx)
        dst_row = get_mesh_coords(mesh_shape, dst_row_ix)

        for src_coord, dst_coord in zip(src_row, dst_row):
            ttnn.point_to_point(
                x,
                dst_coord,
                src_coord,
                optional_output_tensor=x,
                **cfg["transfer_row"],
            )

        return x

    @classmethod
    def forward_decode(
        cls,
        x: ttnn.Tensor,
        position_idxs: ttnn.Tensor,
        rope_tensors: dict,
        page_table: ttnn.Tensor,
        cfg: RunDecodeConfig,
    ) -> ttnn.Tensor:
        """Forward pass for decode mode."""

        x = Embedding1D.forward_decode(x, cfg["embedding"])

        # Stage 1: MLP Decoder Block
        for row_idx in range(cls.NUM_MLP_ROWS):
            for meta_layer_idx in range(cls.NUM_MLP_META_LAYERS):
                x = DecoderBlock.forward_decode(
                    x,
                    row_idx,
                    position_idxs,
                    rope_tensors,
                    page_table,
                    cfg["mlp_decoder_block"][meta_layer_idx],
                )

            # Transfer rows
            x = cls.transfer_row(x, row_idx, (row_idx + 1) % cls.NUM_MLP_ROWS, cfg)

        x = ttnn.to_memory_config(x, **cfg["norm_reshard"])
        x = DistributedRMSNorm.forward_decode(x, cfg["norm"])
        x = ttnn.experimental.all_gather_async(x, **cfg["lm_head"]["all_gather"])
        x = ttnn.to_memory_config(x, cfg["lm_head"]["input_memory_config"])
        x = LMHead.forward_decode(x, cfg["lm_head"])

        return x

    @classmethod
    def forward_prefill(
        cls,
        x: ttnn.Tensor,
        cfg: RunPrefillConfig,
    ) -> ttnn.Tensor:
        return x
