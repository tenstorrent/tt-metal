# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from time import perf_counter

import torch
from loguru import logger
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.tt.ccl import CCL
from models.demos.deepseek_v3.tt.decoder_block.moe_decoder_block_2d import MoEDecoderBlock2D
from models.demos.deepseek_v3.tt.embedding.embedding2d import Embedding2D
from models.demos.deepseek_v3.tt.lm_head1d import LMHead1D
from models.demos.deepseek_v3.tt.rms_norm.distributed_rms_norm import DistributedRMSNorm
from models.demos.deepseek_v3.utils.abstract_module import AbstractModule
from models.demos.deepseek_v3.utils.config_dataclass import (
    AllGatherAsyncConfig,
    ConcatConfig,
    FromWeightConfig,
    LinearConfig,
    MeshDeviceStub,
    ReshardConfig,
)
from models.demos.deepseek_v3.utils.config_helpers import COMPUTE_KERNEL_CONFIG_LOFI, shard_and_save, sub_state_dict
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


def _has_distinct_buffer(a: ttnn.Tensor, b: ttnn.Tensor) -> bool:
    try:
        return a.buffer_address() != b.buffer_address()
    except Exception:
        return a is not b


class MTP2D(AbstractModule):
    """Second-token predictor for DeepSeek-R1 using the dedicated MTP layer weights."""

    WEIGHT_DTYPE = ttnn.bfloat8_b

    @classmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dicts: tuple[dict[str, torch.Tensor] | None, ...],
        output_path: Path,
        mesh_device: ttnn.MeshDevice,
        reuse_embedding_weight_cfg: WeightConfig | None = None,
        reuse_head_weight_cfg: WeightConfig | None = None,
    ) -> WeightConfig:
        assert len(state_dicts) == 1 and state_dicts[0] is not None
        (state_dict,) = state_dicts

        eh_proj_weight = state_dict["eh_proj.weight"].permute(1, 0)
        assert eh_proj_weight.shape[0] == 2 * hf_config.hidden_size
        assert eh_proj_weight.shape[1] == hf_config.hidden_size

        embedding_weight_cfg = (
            reuse_embedding_weight_cfg
            if reuse_embedding_weight_cfg is not None
            else Embedding2D.convert_weights(
                hf_config,
                (sub_state_dict(state_dict, "embed_tokens."),),
                output_path / "embedding",
                mesh_device,
            )
        )
        head_weight_cfg = (
            reuse_head_weight_cfg
            if reuse_head_weight_cfg is not None
            else LMHead1D.convert_weights(
                hf_config,
                (sub_state_dict(state_dict, "shared_head.head."),),
                output_path / "head",
                mesh_device,
            )
        )

        return {
            "embedding": embedding_weight_cfg,
            "hidden_norm": DistributedRMSNorm.convert_weights(
                hf_config,
                (sub_state_dict(state_dict, "hnorm."),) * mesh_device.shape[0],
                output_path / "hidden_norm",
                mesh_device,
            ),
            "token_norm": DistributedRMSNorm.convert_weights(
                hf_config,
                (sub_state_dict(state_dict, "enorm."),) * mesh_device.shape[0],
                output_path / "token_norm",
                mesh_device,
            ),
            "eh_proj": {
                "linear": {
                    "input_tensor_b": shard_and_save(
                        output_path / "eh_proj.linear.input_tensor_b",
                        eh_proj_weight,
                        # Shard output features across mesh columns to match decoder decode sharding.
                        shard_dims=(None, -1),
                        mesh_device=mesh_device,
                        dtype=cls.WEIGHT_DTYPE,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                }
            },
            "decoder_block": MoEDecoderBlock2D.convert_weights(
                hf_config, (state_dict,), output_path / "decoder_block", mesh_device
            ),
            "head_norm": DistributedRMSNorm.convert_weights(
                hf_config,
                (sub_state_dict(state_dict, "shared_head.norm."),) * mesh_device.shape[0],
                output_path / "head_norm",
                mesh_device,
            ),
            "head": head_weight_cfg,
        }

    @classmethod
    def prefill_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
        fabric_config: ttnn.FabricConfig,
    ) -> ModelPrefillConfig:
        hidden_norm_cfg = DistributedRMSNorm.prefill_model_config(hf_config, mesh_device)
        token_norm_cfg = DistributedRMSNorm.prefill_model_config(hf_config, mesh_device)
        decoder_block_cfg = MoEDecoderBlock2D.prefill_model_config(hf_config, mesh_device, fabric_config)
        head_norm_cfg = DistributedRMSNorm.prefill_model_config(hf_config, mesh_device)
        head_cfg = LMHead1D.prefill_model_config(mesh_device)
        return {
            "embedding": Embedding2D.prefill_model_config(hf_config, mesh_device),
            "hidden_norm_reshard": ReshardConfig(memory_config=hidden_norm_cfg["input_memory_config"]),
            "hidden_norm": hidden_norm_cfg,
            "token_norm_reshard": ReshardConfig(memory_config=token_norm_cfg["input_memory_config"]),
            "token_norm": token_norm_cfg,
            "norm_all_gather": AllGatherAsyncConfig(
                mesh_device=mesh_device,
                cluster_axis=1,
                dim=-1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            "concat": ConcatConfig(dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            "eh_proj": {
                "linear": LinearConfig(
                    input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
                ),
            },
            "decoder_input_reshard": ReshardConfig(
                memory_config=decoder_block_cfg["mla_norm_reshard"]["memory_config"]
            ),
            "decoder_block": decoder_block_cfg,
            "head_norm_reshard": ReshardConfig(memory_config=head_norm_cfg["input_memory_config"]),
            "head_norm": head_norm_cfg,
            "head_all_gather": AllGatherAsyncConfig(
                mesh_device=mesh_device,
                cluster_axis=1,
                dim=-1,
                memory_config=head_cfg["input_memory_config"],
            ),
            "head": head_cfg,
        }

    @classmethod
    def decode_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
        fabric_config: ttnn.FabricConfig,
    ) -> ModelDecodeConfig:
        hidden_norm_cfg = DistributedRMSNorm.decode_model_config(hf_config, mesh_device)
        token_norm_cfg = DistributedRMSNorm.decode_model_config(hf_config, mesh_device)
        decoder_block_cfg = MoEDecoderBlock2D.decode_model_config(hf_config, mesh_device, fabric_config)
        head_norm_cfg = DistributedRMSNorm.decode_model_config(hf_config, mesh_device)
        head_cfg = LMHead1D.decode_model_config(mesh_device)
        # Decode is single-token, so keep the MTP-specific intermediate tensors in L1.
        decode_memory_config = ttnn.L1_MEMORY_CONFIG
        return {
            "embedding": Embedding2D.decode_model_config(hf_config, mesh_device),
            "hidden_norm_reshard": ReshardConfig(memory_config=hidden_norm_cfg["input_memory_config"]),
            "hidden_norm": hidden_norm_cfg,
            "token_norm_reshard": ReshardConfig(memory_config=token_norm_cfg["input_memory_config"]),
            "token_norm": token_norm_cfg,
            "norm_all_gather": AllGatherAsyncConfig(
                mesh_device=mesh_device,
                cluster_axis=1,
                dim=-1,
                memory_config=decode_memory_config,
            ),
            "concat": ConcatConfig(dim=-1, memory_config=decode_memory_config),
            "eh_proj": {
                "linear": LinearConfig(
                    input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                    memory_config=decode_memory_config,
                    compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
                ),
            },
            "decoder_input_reshard": ReshardConfig(
                memory_config=decoder_block_cfg["mla_norm_reshard"]["memory_config"]
            ),
            "decoder_block": decoder_block_cfg,
            "head_norm_reshard": ReshardConfig(memory_config=head_norm_cfg["input_memory_config"]),
            "head_norm": head_norm_cfg,
            "head_all_gather": AllGatherAsyncConfig(
                mesh_device=mesh_device,
                cluster_axis=1,
                dim=-1,
                memory_config=head_cfg["input_memory_config"],
            ),
            "head": head_cfg,
        }

    @classmethod
    def create_state(
        cls,
        hf_config: PretrainedConfig,
        paged_config: PagedAttentionConfig,
        mesh_device: ttnn.MeshDevice,
        ccl: CCL,
    ) -> ModelState:
        return {
            MESH_DEVICE_STATE_DICT_KEY: mesh_device,
            "ccl": ccl,
            "embedding": Embedding2D.create_state(hf_config, mesh_device, ccl),
            "hidden_norm": DistributedRMSNorm.create_state(hf_config, mesh_device, ccl),
            "token_norm": DistributedRMSNorm.create_state(hf_config, mesh_device, ccl),
            "decoder_block": MoEDecoderBlock2D.create_state(hf_config, paged_config, mesh_device, ccl),
            "head_norm": DistributedRMSNorm.create_state(hf_config, mesh_device, ccl),
            "head": LMHead1D.create_state(mesh_device, ccl),
        }

    @classmethod
    def create_shared_state(
        cls, hf_config: PretrainedConfig, mesh_device: ttnn.MeshDevice, fabric_config: ttnn.FabricConfig
    ) -> ModelState:
        logger.info("Creating MTP shared state...")
        decoder_block_start = perf_counter()
        decoder_block_shared_state = MoEDecoderBlock2D.create_shared_state(hf_config, mesh_device, fabric_config)
        logger.info(f"Created MTP decoder block shared state in {perf_counter() - decoder_block_start:.2f}s")
        return {
            "decoder_block": decoder_block_shared_state,
        }

    @classmethod
    def forward_decode(
        cls,
        hidden_states: ttnn.Tensor,
        token_ids: ttnn.Tensor,
        position_idxs: ttnn.Tensor,
        cfg: RunDecodeConfig,
        rope_tensors: dict,
        page_table: ttnn.Tensor,
    ) -> ttnn.Tensor:
        ccl = cfg["ccl"]

        token_emb = Embedding2D.forward_decode(token_ids, cfg["embedding"])

        hidden_norm_in = ttnn.to_memory_config(hidden_states, **cfg["hidden_norm_reshard"])
        hidden_norm = DistributedRMSNorm.forward_decode(hidden_norm_in, cfg["hidden_norm"])
        if _has_distinct_buffer(hidden_norm_in, hidden_states):
            ttnn.deallocate(hidden_norm_in)

        token_norm_in = ttnn.to_memory_config(token_emb, **cfg["token_norm_reshard"])
        token_norm = DistributedRMSNorm.forward_decode(token_norm_in, cfg["token_norm"])
        if _has_distinct_buffer(token_norm_in, token_emb):
            ttnn.deallocate(token_norm_in)
        ttnn.deallocate(token_emb)

        hidden_full = ttnn.experimental.all_gather_async(
            hidden_norm, **ccl.populate_all_gather_runtime_args(cfg["norm_all_gather"])
        )
        ttnn.deallocate(hidden_norm)
        token_full = ttnn.experimental.all_gather_async(
            token_norm, **ccl.populate_all_gather_runtime_args(cfg["norm_all_gather"])
        )
        ttnn.deallocate(token_norm)

        orig_hidden_full = hidden_full
        orig_token_full = token_full
        assert (
            hidden_full.shape[2] == token_full.shape[2]
        ), f"MTP hidden/token length mismatch: hidden_full.shape={hidden_full.shape} token_full.shape={token_full.shape}"

        # Concatenate token embedding then hidden.
        concat_in = ttnn.concat([token_full, hidden_full], **cfg["concat"])
        ttnn.deallocate(hidden_full)
        ttnn.deallocate(token_full)
        if orig_hidden_full is not hidden_full:
            ttnn.deallocate(orig_hidden_full)
        if orig_token_full is not token_full:
            ttnn.deallocate(orig_token_full)

        eh_out = ttnn.linear(concat_in, **cfg["eh_proj"]["linear"])
        ttnn.deallocate(concat_in)

        decoder_in = ttnn.to_memory_config(eh_out, **cfg["decoder_input_reshard"])
        if _has_distinct_buffer(decoder_in, eh_out):
            ttnn.deallocate(eh_out)
        decoder_out = MoEDecoderBlock2D.forward_decode(
            decoder_in,
            position_idxs,
            cfg["decoder_block"],
            rope_tensors,
            page_table,
        )
        ttnn.deallocate(decoder_in)

        head_norm_in = ttnn.to_memory_config(decoder_out, **cfg["head_norm_reshard"])
        if _has_distinct_buffer(head_norm_in, decoder_out):
            ttnn.deallocate(decoder_out)
        head_norm_out = DistributedRMSNorm.forward_decode(head_norm_in, cfg["head_norm"])
        ttnn.deallocate(head_norm_in)

        head_full = ttnn.experimental.all_gather_async(
            head_norm_out, **ccl.populate_all_gather_runtime_args(cfg["head_all_gather"])
        )
        ttnn.deallocate(head_norm_out)
        return LMHead1D.forward_decode(head_full, cfg["head"])

    @classmethod
    def forward_prefill(
        cls,
        hidden_states: ttnn.Tensor,
        token_ids: ttnn.Tensor,
        user_id: int,
        cfg: RunPrefillConfig,
        rope_tensors: dict,
        page_table: ttnn.Tensor,
    ) -> ttnn.Tensor:
        ccl = cfg["ccl"]

        token_emb = Embedding2D.forward_prefill(token_ids, cfg["embedding"])

        hidden_norm_in = ttnn.to_memory_config(hidden_states, **cfg["hidden_norm_reshard"])
        hidden_norm = DistributedRMSNorm.forward_prefill(hidden_norm_in, cfg["hidden_norm"])
        if _has_distinct_buffer(hidden_norm_in, hidden_states):
            ttnn.deallocate(hidden_norm_in)

        token_norm_in = ttnn.to_memory_config(token_emb, **cfg["token_norm_reshard"])
        token_norm = DistributedRMSNorm.forward_prefill(token_norm_in, cfg["token_norm"])
        if _has_distinct_buffer(token_norm_in, token_emb):
            ttnn.deallocate(token_norm_in)
        ttnn.deallocate(token_emb)

        hidden_full = ttnn.experimental.all_gather_async(
            hidden_norm, **ccl.populate_all_gather_runtime_args(cfg["norm_all_gather"])
        )
        ttnn.deallocate(hidden_norm)
        token_full = ttnn.experimental.all_gather_async(
            token_norm, **ccl.populate_all_gather_runtime_args(cfg["norm_all_gather"])
        )
        ttnn.deallocate(token_norm)

        orig_hidden_full = hidden_full
        orig_token_full = token_full
        assert (
            hidden_full.shape[2] == token_full.shape[2]
        ), f"MTP hidden/token length mismatch: hidden_full.shape={hidden_full.shape} token_full.shape={token_full.shape}"

        # Concatenate token embedding then hidden.
        concat_in = ttnn.concat([token_full, hidden_full], **cfg["concat"])
        ttnn.deallocate(hidden_full)
        ttnn.deallocate(token_full)
        if orig_hidden_full is not hidden_full:
            ttnn.deallocate(orig_hidden_full)
        if orig_token_full is not token_full:
            ttnn.deallocate(orig_token_full)

        eh_out = ttnn.linear(concat_in, **cfg["eh_proj"]["linear"])
        ttnn.deallocate(concat_in)

        decoder_in = ttnn.to_memory_config(eh_out, **cfg["decoder_input_reshard"])
        if _has_distinct_buffer(decoder_in, eh_out):
            ttnn.deallocate(eh_out)
        decoder_out = MoEDecoderBlock2D.forward_prefill(
            decoder_in,
            user_id,
            cfg["decoder_block"],
            rope_tensors,
            page_table,
        )
        ttnn.deallocate(decoder_in)

        head_norm_in = ttnn.to_memory_config(decoder_out, **cfg["head_norm_reshard"])
        if _has_distinct_buffer(head_norm_in, decoder_out):
            ttnn.deallocate(decoder_out)
        head_norm_out = DistributedRMSNorm.forward_prefill(head_norm_in, cfg["head_norm"])
        ttnn.deallocate(head_norm_in)

        head_full = ttnn.experimental.all_gather_async(
            head_norm_out, **ccl.populate_all_gather_runtime_args(cfg["head_all_gather"])
        )
        ttnn.deallocate(head_norm_out)
        return LMHead1D.forward_prefill(head_full, cfg["head"])
