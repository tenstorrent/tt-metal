# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import itertools
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import torch
from loguru import logger
from tqdm.auto import tqdm
from tracy import signpost
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.tt.ccl import CCL
from models.demos.deepseek_v3.tt.decoder_block.decoder_block_2d import DecoderBlock2D
from models.demos.deepseek_v3.tt.decoder_block.moe_decoder_block_2d import MoEDecoderBlock2D
from models.demos.deepseek_v3.tt.embedding.embedding2d import Embedding2D
from models.demos.deepseek_v3.tt.lm_head1d import LMHead1D
from models.demos.deepseek_v3.tt.mtp import MTP2D
from models.demos.deepseek_v3.tt.rms_norm.distributed_rms_norm import DistributedRMSNorm
from models.demos.deepseek_v3.utils.abstract_module import AbstractModule
from models.demos.deepseek_v3.utils.config_dataclass import KvCacheConfig, ReshardConfig
from models.demos.deepseek_v3.utils.config_helpers import get_fabric_config, sub_state_dict
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
    def _has_mtp_layer(cls, hf_config: PretrainedConfig) -> bool:
        return int(getattr(hf_config, "num_nextn_predict_layers", 0)) > 0

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

        weight_cfg: dict[str, WeightConfig] = {
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
        mtp_layer_idx = hf_config.num_hidden_layers
        mtp_layer_prefix = f"model.layers.{mtp_layer_idx}."
        if cls._has_mtp_layer(hf_config) and f"{mtp_layer_prefix}eh_proj.weight" in state_dict:
            weight_cfg["mtp"] = MTP2D.convert_weights(
                hf_config,
                (sub_state_dict(state_dict, mtp_layer_prefix),),
                output_path / "mtp",
                mesh_device,
            )
        return weight_cfg

    @classmethod
    def prefill_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
        batch_size_per_row: int,
    ) -> ModelPrefillConfig:
        """Create the model configuration for prefill mode."""
        model_cfg = {
            "embedding": Embedding2D.prefill_model_config(hf_config, mesh_device),
            "mlp_decoder_block": [
                DecoderBlock2D.prefill_model_config(
                    hf_config,
                    mesh_device,
                    get_fabric_config(),
                    batch_size_per_row=batch_size_per_row,
                )
            ],
            "moe_decoder_block": [
                MoEDecoderBlock2D.prefill_model_config(
                    hf_config,
                    mesh_device,
                    get_fabric_config(),
                    batch_size_per_row=batch_size_per_row,
                )
            ],
            "norm": DistributedRMSNorm.prefill_model_config(hf_config, mesh_device),
            "lm_head": LMHead1D.prefill_model_config(mesh_device),
        }
        if cls._has_mtp_layer(hf_config):
            model_cfg["mtp"] = MTP2D.prefill_model_config(hf_config, mesh_device, get_fabric_config())
        return model_cfg

    @classmethod
    def decode_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
        batch_size_per_row: int,
    ) -> ModelDecodeConfig:
        """Create the model configuration for decode mode."""
        norm_config = DistributedRMSNorm.decode_model_config(
            hf_config,
            mesh_device,
            batch_size_per_row=batch_size_per_row,
        )
        model_cfg = {
            "embedding": Embedding2D.decode_model_config(hf_config, mesh_device),
            "mlp_decoder_block": [
                DecoderBlock2D.decode_model_config(
                    hf_config,
                    mesh_device,
                    get_fabric_config(),
                    batch_size_per_row=batch_size_per_row,
                )
            ],
            "moe_decoder_block": [
                MoEDecoderBlock2D.decode_model_config(
                    hf_config,
                    mesh_device,
                    get_fabric_config(),
                    batch_size_per_row=batch_size_per_row,
                )
            ],
            "norm_reshard": ReshardConfig(memory_config=norm_config["input_memory_config"]),
            "norm": norm_config,
            "lm_head": LMHead1D.decode_model_config(mesh_device),
        }
        if cls._has_mtp_layer(hf_config):
            model_cfg["mtp"] = MTP2D.decode_model_config(
                hf_config, mesh_device, get_fabric_config(), batch_size_per_row=batch_size_per_row
            )
        return model_cfg

    @classmethod
    def create_shared_state(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
    ) -> ModelState:
        state: ModelState = {
            MESH_DEVICE_STATE_DICT_KEY: mesh_device,
            "num_rows": mesh_device.shape[0],
            "num_mlp_layers": hf_config.first_k_dense_replace,
            "num_moe_layers": hf_config.num_hidden_layers - hf_config.first_k_dense_replace,
        }

        shared_state_steps: list[tuple[str, str, Any]] = [
            (
                "MLP decoder block shared state",
                "mlp_decoder_block",
                lambda: [
                    DecoderBlock2D.create_shared_state(
                        hf_config,
                        mesh_device,
                    )
                ],
            ),
            (
                "MoE decoder block shared state",
                "moe_decoder_block",
                lambda: [
                    MoEDecoderBlock2D.create_shared_state(
                        hf_config,
                        mesh_device,
                    )
                ],
            ),
        ]
        if cls._has_mtp_layer(hf_config):
            shared_state_steps.append(
                ("MTP shared state", "mtp", lambda: MTP2D.create_shared_state(hf_config, mesh_device))
            )

        total_steps = len(shared_state_steps)
        overall_start = perf_counter()
        logger.info(f"Creating RowBatchedModel shared state ({total_steps} steps)...")
        for step_idx, (label, key, factory) in enumerate(shared_state_steps, start=1):
            step_start = perf_counter()
            logger.info(f"[{step_idx}/{total_steps}] Creating {label}...")
            state[key] = factory()
            logger.info(f"[{step_idx}/{total_steps}] Created {label} in {perf_counter() - step_start:.2f}s")
        logger.info(f"Created RowBatchedModel shared state in {perf_counter() - overall_start:.2f}s")
        return state

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

        state = {
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
        if cls._has_mtp_layer(hf_config):
            state["mtp"] = MTP2D.create_state(hf_config, paged_config, mesh_device, ccl)
        return state

    @classmethod
    def forward_decode(
        cls,
        x: ttnn.Tensor,
        position_idxs: ttnn.Tensor,
        cfg: RunDecodeConfig,
        rope_tensors: dict,
        page_tables: Sequence[ttnn.Tensor],
        profile_decode: bool = False,
        return_hidden: bool = False,
    ) -> ttnn.Tensor | tuple[ttnn.Tensor, ttnn.Tensor]:
        """Forward pass for decode mode.

        Args:
            x: Input tensor
            position_idxs: Position indices
            cfg: Run configuration
            rope_tensors: RoPE tensors
            page_tables: Page tables for each layer
            profile_decode: If True, only run first dense layer + first MoE layer for profiling
        """

        x = Embedding2D.forward_decode(x, cfg["embedding"])

        if profile_decode:
            # Profile mode: run only first dense layer + first MoE layer
            # First dense layer (MLP)
            if cfg["mlp_decoder_block"]:
                signpost(header="first_dense_layer")
                x = DecoderBlock2D.forward_decode(
                    x, position_idxs, cfg["mlp_decoder_block"][0], rope_tensors, page_tables[0]
                )
                signpost(header="first_dense_layer")
            # First MoE layer
            if cfg["moe_decoder_block"]:
                signpost(header="first_moe_layer")
                moe_page_table_idx = len(cfg["mlp_decoder_block"])
                x = MoEDecoderBlock2D.forward_decode(
                    x, position_idxs, cfg["moe_decoder_block"][0], rope_tensors, page_tables[moe_page_table_idx]
                )
                signpost(header="first_moe_layer")

        else:
            # Normal mode: run all layers
            for (block_cfg, BlockClass), page_table in zip(
                itertools.chain(
                    zip(cfg["mlp_decoder_block"], itertools.repeat(DecoderBlock2D)),
                    zip(cfg["moe_decoder_block"], itertools.repeat(MoEDecoderBlock2D)),
                ),
                page_tables,
                strict=True,
            ):
                x = BlockClass.forward_decode(x, position_idxs, block_cfg, rope_tensors, page_table)

        # Capture pre-norm hidden states for MTP; MTP applies its own hnorm.
        hidden_for_mtp = x if return_hidden else None

        x = ttnn.to_memory_config(x, **cfg["norm_reshard"])
        x = DistributedRMSNorm.forward_decode(x, cfg["norm"])

        ccl = cfg["lm_head"]["ccl"]

        x = ttnn.experimental.all_gather_async(x, **ccl.populate_all_gather_runtime_args(cfg["lm_head"]["all_gather"]))
        if return_hidden:
            lm_head_in = ttnn.clone(x)
            ttnn.deallocate(x)
            logits = LMHead1D.forward_decode(lm_head_in, cfg["lm_head"])
            return logits, hidden_for_mtp

        logits = LMHead1D.forward_decode(x, cfg["lm_head"])
        return logits

    @classmethod
    def forward_prefill(
        cls,
        x: ttnn.Tensor,
        user_id: int,
        cfg: RunPrefillConfig,
        rope_tensors: dict,
        page_tables: Sequence[ttnn.Tensor],
        return_hidden: bool = False,
    ) -> ttnn.Tensor | tuple[ttnn.Tensor, ttnn.Tensor]:
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

        # Capture pre-norm hidden states for MTP; MTP applies its own hnorm.
        hidden_for_mtp = x if return_hidden else None

        x = DistributedRMSNorm.forward_prefill(x, cfg["norm"])  # no resharding needed for prefill

        ccl = cfg["lm_head"]["ccl"]

        x = ttnn.experimental.all_gather_async(x, **ccl.populate_all_gather_runtime_args(cfg["lm_head"]["all_gather"]))
        if return_hidden:
            lm_head_in = ttnn.clone(x)
            ttnn.deallocate(x)
            logits = LMHead1D.forward_prefill(lm_head_in, cfg["lm_head"])
            return logits, hidden_for_mtp

        logits = LMHead1D.forward_prefill(x, cfg["lm_head"])
        return logits

    @classmethod
    def forward_mtp_decode(
        cls,
        hidden_states: ttnn.Tensor,
        token_ids: ttnn.Tensor,
        position_idxs: ttnn.Tensor,
        cfg: RunDecodeConfig,
        rope_tensors: dict,
        page_table: ttnn.Tensor,
    ) -> ttnn.Tensor:
        assert "mtp" in cfg, "MTP config is missing from decode run config"
        return MTP2D.forward_decode(
            hidden_states=hidden_states,
            token_ids=token_ids,
            position_idxs=position_idxs,
            cfg=cfg["mtp"],
            rope_tensors=rope_tensors,
            page_table=page_table,
        )

    @classmethod
    def forward_mtp_prefill(
        cls,
        hidden_states: ttnn.Tensor,
        token_ids: ttnn.Tensor,
        user_id: int,
        cfg: RunPrefillConfig,
        rope_tensors: dict,
        page_table: ttnn.Tensor,
    ) -> ttnn.Tensor:
        assert "mtp" in cfg, "MTP config is missing from prefill run config"
        return MTP2D.forward_prefill(
            hidden_states=hidden_states,
            token_ids=token_ids,
            user_id=user_id,
            cfg=cfg["mtp"],
            rope_tensors=rope_tensors,
            page_table=page_table,
        )
