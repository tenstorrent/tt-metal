# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import itertools
from pathlib import Path
from typing import Any, Sequence

import torch
from tqdm.auto import tqdm
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.tt.ccl import CCL
from models.demos.deepseek_v3.tt.decoder_block.decoder_block_2d import DecoderBlock2D
from models.demos.deepseek_v3.tt.decoder_block.moe_decoder_block_2d import MoEDecoderBlock2D
from models.demos.deepseek_v3.tt.embedding.embedding2d import Embedding2D
from models.demos.deepseek_v3.tt.lm_head1d import LMHead1D
from models.demos.deepseek_v3.tt.rms_norm.distributed_rms_norm import DistributedRMSNorm
from models.demos.deepseek_v3.utils.abstract_module import AbstractModule
from models.demos.deepseek_v3.utils.config_dataclass import KvCacheConfig, ReshardConfig
from models.demos.deepseek_v3.utils.config_helpers import sub_state_dict
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


class RowBatchedModel(SharedStateAddOn, AbstractModule):
    @classmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dicts: list[dict[str, torch.Tensor]],
        output_path: Path,
        mesh_device: ttnn.MeshDevice,
    ) -> WeightConfig:
        assert (
            hf_config.first_k_dense_replace <= hf_config.num_hidden_layers
        ), "Number of non-MoE blocks cannot be greater than the total number of blocks."
        (state_dict,) = state_dicts

        return {
            "embedding": Embedding2D.convert_weights(
                hf_config, (sub_state_dict(state_dict, "model.embed_tokens."),), output_path / "embedding", mesh_device
            ),
            "mlp_decoder_block": [
                DecoderBlock2D.convert_weights(
                    hf_config,
                    (sub_state_dict(state_dict, f"model.layers.{layer_idx}."),),
                    output_path / f"mlp_decoder_block_{layer_idx}",
                    mesh_device,
                )
                for layer_idx in tqdm(
                    range(hf_config.first_k_dense_replace),
                    desc="Converting MLP layers",
                )
            ],
            "moe_decoder_block": [
                MoEDecoderBlock2D.convert_weights(
                    hf_config,
                    (sub_state_dict(state_dict, f"model.layers.{layer_idx}."),),
                    output_path / f"moe_decoder_block_{layer_idx}",
                    mesh_device,
                )
                for layer_idx in tqdm(
                    range(hf_config.first_k_dense_replace, hf_config.num_hidden_layers),
                    desc="Converting MoE layers",
                )
            ],
            "norm": DistributedRMSNorm.convert_weights(
                hf_config,
                [sub_state_dict(state_dict, "model.norm.")] * mesh_device.shape[0],
                output_path / "norm",
                mesh_device,
            ),
            "lm_head": LMHead1D.convert_weights(
                hf_config, [sub_state_dict(state_dict, "lm_head.")], output_path / "lm_head", mesh_device
            ),
        }

    @classmethod
    def prefill_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
    ) -> ModelPrefillConfig:
        """Create the model configuration for prefill mode."""
        return {
            "embedding": Embedding2D.prefill_model_config(hf_config, mesh_device),
            "mlp_decoder_block": [
                DecoderBlock2D.prefill_model_config(
                    hf_config,
                    mesh_device,
                )
            ],
            "moe_decoder_block": [
                MoEDecoderBlock2D.prefill_model_config(
                    hf_config,
                    mesh_device,
                )
            ],
            "norm": DistributedRMSNorm.prefill_model_config(hf_config, mesh_device),
            "lm_head": LMHead1D.prefill_model_config(mesh_device),
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
            "embedding": Embedding2D.decode_model_config(hf_config, mesh_device),
            "mlp_decoder_block": [
                DecoderBlock2D.decode_model_config(
                    hf_config,
                    mesh_device,
                )
            ],
            "moe_decoder_block": [
                MoEDecoderBlock2D.decode_model_config(
                    hf_config,
                    mesh_device,
                )
            ],
            "norm_reshard": ReshardConfig(memory_config=norm_config["input_memory_config"]),
            "norm": norm_config,
            "lm_head": LMHead1D.decode_model_config(mesh_device),
        }

    @classmethod
    def create_shared_state(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
    ) -> ModelState:
        return {
            MESH_DEVICE_STATE_DICT_KEY: mesh_device,
            "num_rows": mesh_device.shape[0],
            "num_mlp_layers": hf_config.first_k_dense_replace,
            "num_moe_layers": hf_config.num_hidden_layers - hf_config.first_k_dense_replace,
            "mlp_decoder_block": [
                DecoderBlock2D.create_shared_state(
                    hf_config,
                    mesh_device,
                )
            ],
            "moe_decoder_block": [
                MoEDecoderBlock2D.create_shared_state(
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
        kv_cache_override: KvCacheConfig | None = None,
    ) -> Any:
        assert mla_caches is None or (
            len(mla_caches) >= 1
            and len(mla_caches) == hf_config.num_hidden_layers
            and all(mla_cache.shape == mla_caches[0].shape for mla_cache in mla_caches)
        )

        mlp_caches = (
            mla_caches[: hf_config.first_k_dense_replace]
            if mla_caches is not None
            else [None] * hf_config.first_k_dense_replace
        )
        moe_caches = (
            mla_caches[hf_config.first_k_dense_replace :]
            if mla_caches is not None
            else [None] * (hf_config.num_hidden_layers - hf_config.first_k_dense_replace)
        )

        return {
            "embedding": Embedding2D.create_state(hf_config, mesh_device, ccl),
            "mlp_decoder_block": [
                DecoderBlock2D.create_state(
                    hf_config,
                    paged_config,
                    mesh_device,
                    ccl,
                    mla_cache,
                    kv_cache_override,
                )
                for mla_cache in tqdm(mlp_caches, desc="Creating MLP layer states")
            ],
            "moe_decoder_block": [
                MoEDecoderBlock2D.create_state(hf_config, paged_config, mesh_device, ccl, mla_cache, kv_cache_override)
                for mla_cache in tqdm(moe_caches, desc="Creating MoE layer states")
            ],
            "norm": DistributedRMSNorm.create_state(hf_config, mesh_device, ccl),
            "lm_head": LMHead1D.create_state(mesh_device, ccl),
        }

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

        x = Embedding2D.forward_decode(x, cfg["embedding"])

        for (block_cfg, BlockClass), page_table in zip(
            itertools.chain(
                zip(cfg["mlp_decoder_block"], itertools.repeat(DecoderBlock2D)),
                zip(cfg["moe_decoder_block"], itertools.repeat(MoEDecoderBlock2D)),
            ),
            page_tables,
            strict=True,
        ):
            x = BlockClass.forward_decode(x, position_idxs, block_cfg, rope_tensors, page_table)

        x = ttnn.to_memory_config(x, **cfg["norm_reshard"])
        x = DistributedRMSNorm.forward_decode(x, cfg["norm"])

        ccl = cfg["lm_head"]["ccl"]

        x = ttnn.experimental.all_gather_async(x, **ccl.populate_all_gather_runtime_args(cfg["lm_head"]["all_gather"]))
        x = LMHead1D.forward_decode(x, cfg["lm_head"])
        return x

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

        x = Embedding2D.forward_prefill(x, cfg["embedding"])

        for (block_cfg, BlockClass), page_table in zip(
            itertools.chain(
                zip(cfg["mlp_decoder_block"], itertools.repeat(DecoderBlock2D)),
                zip(cfg["moe_decoder_block"], itertools.repeat(MoEDecoderBlock2D)),
            ),
            page_tables,
            strict=True,
        ):
            x = BlockClass.forward_prefill(x, user_id, block_cfg, rope_tensors, page_table)

        x = DistributedRMSNorm.forward_prefill(x, cfg["norm"])  # no resharding needed for prefill

        ccl = cfg["lm_head"]["ccl"]

        x = ttnn.experimental.all_gather_async(x, **ccl.populate_all_gather_runtime_args(cfg["lm_head"]["all_gather"]))
        x = LMHead1D.forward_prefill(x, cfg["lm_head"])
        return x
