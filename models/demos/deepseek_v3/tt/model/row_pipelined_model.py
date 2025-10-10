# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.tt.ccl import CCL
from models.demos.deepseek_v3.tt.decoder_block.decoder_block_1d import DecoderBlock1D
from models.demos.deepseek_v3.tt.decoder_block.moe_decoder_block_1d import MoEDecoderBlock1D
from models.demos.deepseek_v3.tt.embedding.embedding1d import Embedding1D
from models.demos.deepseek_v3.tt.lm_head import LMHead
from models.demos.deepseek_v3.tt.rms_norm.distributed_rms_norm import DistributedRMSNorm
from models.demos.deepseek_v3.utils.abstract_module import AbstractModule
from models.demos.deepseek_v3.utils.config_dataclass import ReshardConfig
from models.demos.deepseek_v3.utils.config_helpers import get_mesh_coords, sub_state_dict
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
from models.tt_transformers.tt.common import PagedAttentionConfig


class RowPipelinedModel(SharedStateAddOn, AbstractModule):
    @classmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dicts: list[dict[str, torch.Tensor]],
        output_path: Path,
        mesh_device: ttnn.MeshDevice,
    ) -> WeightConfig:
        """Convert weights for the 1D model."""
        assert (
            hf_config.first_k_dense_replace <= hf_config.num_hidden_layers
        ), "Number of non-MoE blocks cannot be greater than the total number of blocks."
        (state_dict,) = state_dicts

        _, mlp_meta_layer_indices = cls.get_meta_layer_mapping(mesh_device.shape[0], hf_config.first_k_dense_replace)

        _, moe_meta_layer_indices = cls.get_meta_layer_mapping(
            mesh_device.shape[0], hf_config.first_k_dense_replace, hf_config.num_hidden_layers
        )

        decoder_block_state_dicts = np.array(
            [
                sub_state_dict(state_dict, f"model.layers.{layer_idx}.")
                for layer_idx in range(hf_config.num_hidden_layers)
            ]
            + [None]
        )

        return {
            "embedding": Embedding1D.convert_weights(
                hf_config, (sub_state_dict(state_dict, "model.embed_tokens."),), output_path / "embedding", mesh_device
            ),
            "mlp_decoder_block": [
                DecoderBlock1D.convert_weights(
                    hf_config,
                    decoder_block_state_dicts[layer_indices].tolist(),
                    output_path / f"mlp_decoder_block_{meta_layer_idx}",
                    mesh_device,
                )
                for meta_layer_idx, layer_indices in enumerate(mlp_meta_layer_indices)
            ],
            "moe_decoder_block": [
                MoEDecoderBlock1D.convert_weights(
                    hf_config,
                    decoder_block_state_dicts[layer_indices].tolist(),
                    output_path / f"moe_decoder_block_{meta_layer_idx}",
                    mesh_device,
                )
                for meta_layer_idx, layer_indices in enumerate(moe_meta_layer_indices)
            ],
            "norm": DistributedRMSNorm.convert_weights(
                hf_config,
                [sub_state_dict(state_dict, "model.norm.")] * mesh_device.shape[0],
                output_path / "norm",
                mesh_device,
            ),
            "lm_head": LMHead.convert_weights(
                hf_config, [sub_state_dict(state_dict, "lm_head.")], output_path / "lm_head", mesh_device
            ),
        }

    @classmethod
    def get_meta_layer_mapping(
        cls, num_mesh_rows: int, start_layer_idx: int, end_layer_idx: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Distribute `num_layers` evenly across `num_mesh_rows`, returning a
        list of rows where each element is either a layer index or -1
        (if padding is needed to fill the structure). The result is
        transposed such that each inner list corresponds to a layer
        position across all rows.

        Returns:
            A tuple (pad_map, mapping) where:
            - pad_map: A list of lists of shape (num_meta_layers, num_mesh_rows), where
            each element is a boolean indicating if that position is padding (True) or a valid layer index (False).
            - mapping: A list of lists of shape (num_meta_layers, num_mesh_rows), where
            each element is an int (layer index) or -1 for padding positions.
        """
        if end_layer_idx is None:
            end_layer_idx = start_layer_idx
            start_layer_idx = 0
        num_layers = end_layer_idx - start_layer_idx
        assert num_mesh_rows > 0 and num_layers >= 0

        mapping = (
            torch.arange(ttnn.core.roundup(num_layers, num_mesh_rows))
            .reshape(num_mesh_rows, ttnn.core.divup(num_layers, num_mesh_rows))
            .T
            + start_layer_idx
        )
        mask = mapping >= end_layer_idx
        mapping[mask] = -1

        return mask, mapping

    @classmethod
    def prefill_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
    ) -> ModelPrefillConfig:
        """Create the model configuration for prefill mode."""
        return {
            "embedding": Embedding1D.prefill_model_config(hf_config, mesh_device),
            "mlp_decoder_block": [
                DecoderBlock1D.prefill_model_config(
                    hf_config,
                    mesh_device,
                )
            ],
            "moe_decoder_block": [
                MoEDecoderBlock1D.prefill_model_config(
                    hf_config,
                    mesh_device,
                )
            ],
            "transfer_row": {"topology": ttnn.Topology.Linear},
            "norm": DistributedRMSNorm.prefill_model_config(hf_config, mesh_device),
            "lm_head": LMHead.prefill_model_config(hf_config, mesh_device, input_row_idx=0),
        }

    @classmethod
    def decode_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
    ) -> ModelDecodeConfig:
        """Create the model configuration for decode mode."""
        norm_config = DistributedRMSNorm.decode_model_config(hf_config, mesh_device)

        return {
            "embedding": Embedding1D.decode_model_config(hf_config, mesh_device),
            "mlp_decoder_block": [
                DecoderBlock1D.decode_model_config(
                    hf_config,
                    mesh_device,
                )
            ],
            "moe_decoder_block": [
                MoEDecoderBlock1D.decode_model_config(
                    hf_config,
                    mesh_device,
                )
            ],
            "transfer_row": {"topology": ttnn.Topology.Linear},
            "norm_reshard": ReshardConfig(memory_config=norm_config["input_memory_config"]),
            "norm": norm_config,
            "lm_head": LMHead.decode_model_config(hf_config, mesh_device, input_row_idx=0),
        }

    @classmethod
    def create_shared_state(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
    ) -> ModelState:
        mlp_is_padding_layer, _ = cls.get_meta_layer_mapping(mesh_device.shape[0], hf_config.first_k_dense_replace)
        moe_is_padding_layer, _ = cls.get_meta_layer_mapping(
            mesh_device.shape[0], hf_config.first_k_dense_replace, hf_config.num_hidden_layers
        )

        return {
            MESH_DEVICE_STATE_DICT_KEY: mesh_device,
            "num_rows": mesh_device.shape[0],
            "num_mlp_layers": hf_config.first_k_dense_replace,
            "num_moe_layers": hf_config.num_hidden_layers - hf_config.first_k_dense_replace,
            "mlp_decoder_block": [
                DecoderBlock1D.create_shared_state(
                    hf_config,
                    mesh_device,
                )
            ],
            "moe_decoder_block": [
                MoEDecoderBlock1D.create_shared_state(
                    hf_config,
                    mesh_device,
                )
            ],
        }

    @classmethod
    def create_state(
        cls,
        hf_config: PretrainedConfig,
        paged_config: PagedAttentionConfig,
        mesh_device: ttnn.MeshDevice,
        ccl: CCL,
        mla_caches: Sequence[torch.Tensor] | None = None,
    ) -> Any:
        """Create the state for the 1D model."""
        assert mla_caches is None or (
            len(mla_caches) >= 1
            and len(mla_caches) == hf_config.num_hidden_layers
            and all(mla_cache.shape == mla_caches[0].shape for mla_cache in mla_caches)
        )
        mlp_is_padding_layer, mlp_meta_layer_indices = cls.get_meta_layer_mapping(
            mesh_device.shape[0], hf_config.first_k_dense_replace
        )  # [num_meta_layers, num_rows]
        moe_is_padding_layer, moe_meta_layer_indices = cls.get_meta_layer_mapping(
            mesh_device.shape[0], hf_config.first_k_dense_replace, hf_config.num_hidden_layers
        )

        if mla_caches is not None:
            assert len(mla_caches) == hf_config.num_hidden_layers >= hf_config.first_k_dense_replace
            mla_caches = torch.stack(list(mla_caches) + [torch.zeros_like(mla_caches[0])], dim=0)

        return {
            "embedding": Embedding1D.create_state(hf_config, mesh_device, ccl),
            "mlp_decoder_block": [
                DecoderBlock1D.create_state(
                    hf_config,
                    paged_config,
                    mesh_device,
                    ccl,
                    tuple(is_padding_layer.tolist()),
                    mla_caches[layer_indices] if mla_caches is not None else None,
                )
                for is_padding_layer, layer_indices in zip(mlp_is_padding_layer, mlp_meta_layer_indices)
            ],
            "moe_decoder_block": [
                MoEDecoderBlock1D.create_state(
                    hf_config,
                    paged_config,
                    mesh_device,
                    ccl,
                    tuple(is_padding_layer.tolist()),
                    mla_caches[layer_indices] if mla_caches is not None else None,
                )
                for is_padding_layer, layer_indices in zip(moe_is_padding_layer, moe_meta_layer_indices)
            ],
            "norm": DistributedRMSNorm.create_state(hf_config, mesh_device, ccl),
            "lm_head": LMHead.create_state(hf_config, mesh_device, ccl),
            "transfer_row": {"mesh_shape": mesh_device.shape},
        }

    @classmethod
    def transfer_row(
        cls,
        x: ttnn.Tensor,
        src_row_idx: int,
        dst_row_idx: int,
        topology: ttnn.Topology,
        mesh_shape: ttnn.MeshShape,
    ) -> ttnn.Tensor:
        """Transfer a row of data from one row to another in the mesh.
        Args:
            x: Input tensor to transfer
            src_row_idx: Source row index
            dst_row_idx: Destination row index
            cfg: Run configuration containing mesh shape and other parameters
        Returns:
            The tensor after transferring the row
        """

        src_row = get_mesh_coords(mesh_shape, src_row_idx)
        dst_row = get_mesh_coords(mesh_shape, dst_row_idx)

        for src_coord, dst_coord in zip(src_row, dst_row):
            ttnn.point_to_point(
                x,
                dst_coord,
                src_coord,
                optional_output_tensor=x,
                topology=topology,
            )

        return x

    @classmethod
    def forward_decoder_blocks(
        cls,
        x: ttnn.Tensor,
        num_mesh_rows: int,
        start_layer_idx: int,
        end_layer_idx: int,
        block_configs: Sequence[RunDecodeConfig],
        page_tables: Sequence[ttnn.Tensor],
        transfer_row_cfg: RunDecodeConfig,
        block_forward_fn: Callable[[ttnn.Tensor, int, RunDecodeConfig, ttnn.Tensor], ttnn.Tensor],
    ) -> ttnn.Tensor:
        is_padding_layer, meta_layer_indices = cls.get_meta_layer_mapping(num_mesh_rows, start_layer_idx, end_layer_idx)
        for row_idx, (per_row_is_padding_layer, per_row_meta_layer_indices) in enumerate(
            zip(is_padding_layer.T, meta_layer_indices.T, strict=True)
        ):
            for meta_layer_idx, (is_padding_layer, layer_idx) in enumerate(
                zip(per_row_is_padding_layer, per_row_meta_layer_indices, strict=True)
            ):
                if is_padding_layer:
                    continue
                x_next = block_forward_fn(x, row_idx, block_configs[meta_layer_idx], page_tables[layer_idx])
                ttnn.deallocate(x)
                x = x_next

            # Transfer rows
            cls.transfer_row(x, row_idx, (row_idx + 1) % num_mesh_rows, **transfer_row_cfg)
        return x

    @classmethod
    def forward_decode(
        cls,
        x: ttnn.Tensor,
        position_idxs: ttnn.Tensor,
        cfg: RunDecodeConfig,
        rope_tensors: dict,
        page_tables: Sequence[ttnn.Tensor],
    ) -> ttnn.Tensor:
        """Forward pass for decode mode."""

        x = Embedding1D.forward_decode(x, cfg["embedding"])

        x = cls.forward_decoder_blocks(
            x,
            cfg["num_rows"],
            0,
            cfg["num_mlp_layers"],
            cfg["mlp_decoder_block"],
            page_tables,
            cfg["transfer_row"],
            lambda x_in, row_idx, block_cfg, page_table: DecoderBlock1D.forward_decode(
                x_in, position_idxs, row_idx, block_cfg, rope_tensors, page_table
            ),
        )

        x = cls.forward_decoder_blocks(
            x,
            cfg["num_rows"],
            cfg["num_mlp_layers"],
            cfg["num_mlp_layers"] + cfg["num_moe_layers"],
            cfg["moe_decoder_block"],
            page_tables,
            cfg["transfer_row"],
            lambda x_in, row_idx, block_cfg, page_table: MoEDecoderBlock1D.forward_decode(
                x_in, position_idxs, row_idx, block_cfg, rope_tensors, page_table
            ),
        )

        x_resharded = ttnn.to_memory_config(x, **cfg["norm_reshard"])
        ttnn.deallocate(x)

        x_norm = DistributedRMSNorm.forward_decode(x_resharded, cfg["norm"])
        ttnn.deallocate(x_resharded)

        # CCL runtime initialization in execution order
        ccl = cfg["lm_head"]["ccl"]

        x_ag = ttnn.experimental.all_gather_async(
            x_norm, **ccl.populate_all_gather_runtime_args(cfg["lm_head"]["all_gather"])
        )
        ttnn.deallocate(x_norm)

        x_resharded = ttnn.to_memory_config(x_ag, cfg["lm_head"]["input_memory_config"])
        ttnn.deallocate(x_ag)

        x_lmhead = LMHead.forward_decode(x_resharded, cfg["lm_head"])
        ttnn.deallocate(x_resharded)

        return x_lmhead

    @classmethod
    def forward_prefill(
        cls,
        x: ttnn.Tensor,
        user_id: int,
        cfg: RunPrefillConfig,
        rope_tensors: dict,
        page_tables: Sequence[ttnn.Tensor],
    ) -> ttnn.Tensor:
        """Forward pass for prefill mode."""

        # Embedding
        x = Embedding1D.forward_prefill(x, cfg["embedding"])

        x = cls.forward_decoder_blocks(
            x,
            cfg["num_rows"],
            0,
            cfg["num_mlp_layers"],
            cfg["mlp_decoder_block"],
            page_tables,
            cfg["transfer_row"],
            lambda x_in, row_idx, block_cfg, page_table: DecoderBlock1D.forward_prefill(
                x_in, user_id, row_idx, block_cfg, rope_tensors, page_table
            ),
        )

        x = cls.forward_decoder_blocks(
            x,
            cfg["num_rows"],
            cfg["num_mlp_layers"],
            cfg["num_mlp_layers"] + cfg["num_moe_layers"],
            cfg["moe_decoder_block"],
            page_tables,
            cfg["transfer_row"],
            lambda x_in, row_idx, block_cfg, page_table: MoEDecoderBlock1D.forward_prefill(
                x_in, user_id, row_idx, block_cfg, rope_tensors, page_table
            ),
        )

        # Norm (no resharding needed for prefill)
        x_norm = DistributedRMSNorm.forward_prefill(x, cfg["norm"])
        ttnn.deallocate(x)

        # All gather before LM Head (same as decode path)
        # CCL runtime initialization in execution order
        ccl = cfg["lm_head"]["ccl"]

        x_ag = ttnn.experimental.all_gather_async(
            x_norm, **ccl.populate_all_gather_runtime_args(cfg["lm_head"]["all_gather"])
        )
        ttnn.deallocate(x_norm)

        # LM Head
        x_lmhead = LMHead.forward_prefill(x_ag, cfg["lm_head"])
        ttnn.deallocate(x_ag)

        return x_lmhead
