# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import itertools
from pathlib import Path
from typing import Any, Sequence

import torch
from tqdm.auto import tqdm
from tracy import signpost
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
    def get_dtype_tag(cls, hf_config: PretrainedConfig) -> str:
        """Derive a cache tag from the weight dtypes used by RowBatchedModel.

        The tag is assembled from module-level dtype constants defined in the
        constituent weight-conversion modules.  Updating those constants
        automatically changes the tag, preventing stale-cache collisions.
        """
        from models.demos.deepseek_v3.tt.experts import EXPERT_DOWN_WEIGHT_DTYPE, EXPERT_UP_WEIGHT_DTYPE
        from models.demos.deepseek_v3.tt.lm_head1d import LM_HEAD_WEIGHT_DTYPE
        from models.demos.deepseek_v3.tt.mla.mla1d import WKV_B_WEIGHT_DTYPE

        _abbrev = {
            ttnn.bfloat16: "bf16",
            ttnn.bfloat8_b: "bf8b",
            ttnn.bfloat4_b: "bf4b",
            ttnn.float32: "fp32",
        }

        def abbrev(dtype: ttnn.DataType) -> str:
            return _abbrev.get(dtype, dtype.name.lower())

        return (
            f"wkv-{abbrev(WKV_B_WEIGHT_DTYPE)}"
            f"_exp-up-{abbrev(EXPERT_UP_WEIGHT_DTYPE)}"
            f"_exp-dn-{abbrev(EXPERT_DOWN_WEIGHT_DTYPE)}"
            f"_lmh-{abbrev(LM_HEAD_WEIGHT_DTYPE)}"
        )

    @classmethod
    def get_num_cached_layers(cls, weight_config: WeightConfig) -> int:
        """Return the number of transformer layers present in weight_config."""
        return len(weight_config.get("mlp_decoder_block", [])) + len(weight_config.get("moe_decoder_block", []))

    @classmethod
    def augment_weight_config(
        cls,
        hf_config: PretrainedConfig,
        state_dicts: list[dict[str, torch.Tensor]],
        existing_config: WeightConfig | None,
        output_path: Path,
        mesh_device: ttnn.MeshDevice,
    ) -> WeightConfig:
        """Produce a weight config for hf_config.num_hidden_layers layers.

        Reuses layers already present in existing_config and converts only the
        missing ones.  Shared weights (embedding, norm, lm_head) are reused if
        present; otherwise converted fresh.
        """
        assert (
            hf_config.first_k_dense_replace <= hf_config.num_hidden_layers
        ), "Number of non-MoE blocks cannot be greater than the total number of blocks."
        (state_dict,) = state_dicts

        n = hf_config.num_hidden_layers
        k = hf_config.first_k_dense_replace

        existing_mlp: list = existing_config.get("mlp_decoder_block", []) if existing_config else []
        existing_moe: list = existing_config.get("moe_decoder_block", []) if existing_config else []
        target_mlp = min(n, k)
        target_moe = max(0, n - k)

        # Convert only missing MLP layers.
        new_mlp = list(existing_mlp)
        for layer_idx in tqdm(
            range(len(existing_mlp), target_mlp),
            desc="Converting MLP layers",
        ):
            new_mlp.append(
                DecoderBlock2D.convert_weights(
                    hf_config,
                    (sub_state_dict(state_dict, f"model.layers.{layer_idx}."),),
                    output_path / f"mlp_decoder_block_{layer_idx}",
                    mesh_device,
                )
            )

        # Convert only missing MoE layers.
        new_moe = list(existing_moe)
        for i in tqdm(
            range(len(existing_moe), target_moe),
            desc="Converting MoE layers",
        ):
            layer_idx = k + i
            new_moe.append(
                MoEDecoderBlock2D.convert_weights(
                    hf_config,
                    (sub_state_dict(state_dict, f"model.layers.{layer_idx}."),),
                    output_path / f"moe_decoder_block_{layer_idx}",
                    mesh_device,
                )
            )

        # Reuse or convert shared weights.
        if existing_config and "embedding" in existing_config:
            embedding = existing_config["embedding"]
        else:
            embedding = Embedding2D.convert_weights(
                hf_config,
                (sub_state_dict(state_dict, "model.embed_tokens."),),
                output_path / "embedding",
                mesh_device,
            )

        if existing_config and "norm" in existing_config:
            norm = existing_config["norm"]
        else:
            norm = DistributedRMSNorm.convert_weights(
                hf_config,
                [sub_state_dict(state_dict, "model.norm.")] * mesh_device.shape[0],
                output_path / "norm",
                mesh_device,
            )

        if existing_config and "lm_head" in existing_config:
            lm_head = existing_config["lm_head"]
        else:
            lm_head = LMHead1D.convert_weights(
                hf_config,
                [sub_state_dict(state_dict, "lm_head.")],
                output_path / "lm_head",
                mesh_device,
            )

        return {
            "embedding": embedding,
            "mlp_decoder_block": new_mlp,
            "moe_decoder_block": new_moe,
            "norm": norm,
            "lm_head": lm_head,
        }

    @classmethod
    def slice_weight_config(cls, weight_config: WeightConfig, hf_config: PretrainedConfig) -> WeightConfig:
        """Slice a larger weight config down to hf_config.num_hidden_layers.

        Keeps the shared weights (embedding, norm, lm_head) unchanged and
        truncates the per-layer lists to the requested number of layers.
        ``_meta`` is intentionally excluded from the result; the caller is
        responsible for writing fresh metadata if needed.
        """
        n = hf_config.num_hidden_layers
        k = hf_config.first_k_dense_replace
        return {
            key: val
            for key, val in weight_config.items()
            if key not in ("_meta", "mlp_decoder_block", "moe_decoder_block")
        } | {
            "mlp_decoder_block": weight_config["mlp_decoder_block"][: min(n, k)],
            "moe_decoder_block": weight_config["moe_decoder_block"][: max(0, n - k)],
        }

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
        profile_decode: bool = False,
    ) -> ttnn.Tensor:
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
