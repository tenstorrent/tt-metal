# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
# TT Ministral3DecoderLayer: pre-norm attn + MLP with residuals.

from __future__ import annotations

import os

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.devstral2_small.tt.tt_ministralattn import TtMinistralAttention
from models.experimental.devstral2_small.tt.tt_ministralmlp import TtMinistralMLP
from models.experimental.devstral2_small.tt.tt_ministralrmsnorm import (
    TtMinistralRMSNorm,
    ministral_prefill_block_shard_add_eligible,
    ministral_prefill_block_shard_mem_cfg,
    ministral_prefill_block_shard_norm_config,
    ministral_prefill_is_block_sharded_l1,
    ministral_prefill_prepare_block_shard_input,
)
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.model_config import (
    MathFidelitySetting,
    ModelOptimizations,
    OpGroup,
    PrecisionSetting,
    TensorGroup,
)


def ministral_text_reduced_precision_enabled() -> bool:
    return os.environ.get("TT_MINISTRAL3_TEXT_REDUCED_PRECISION", "1").strip().lower() not in ("0", "false", "no")


class TtMinistral3DecoderLayer(LightweightModule):
    """Pre-norm attention + MLP with residuals (HF Ministral3DecoderLayer order)."""

    def __init__(
        self,
        mesh_device,
        tt_ccl,
        model_args,
        meta_state_dict,
        weight_cache_path,
        layer_num: int,
        dtype,
        transformation_mats,
        configuration,
        llama_4_scaling_beta=None,
        original_max_position_embeddings=None,
        paged_attention_config=None,
    ):
        super().__init__()
        self.layer_num = layer_num
        self.tt_ccl = tt_ccl  # fabric num_links for all_gather
        use_reduced_precision = ministral_text_reduced_precision_enabled()
        if use_reduced_precision:
            model_args.decoders_optimizations.set_decoder_conf(
                layer_num,
                ModelOptimizations(
                    {
                        "TensorPrecision": {
                            TensorGroup.WQKV: PrecisionSetting.BFP8,
                            TensorGroup.WO: PrecisionSetting.BFP8,
                            TensorGroup.ACTIVATION: PrecisionSetting.BFP8,
                        },
                        "OpFidelity": {
                            OpGroup.LI_QKV_PREFILL: MathFidelitySetting.LOFI,
                            OpGroup.LI_O_PREFILL: MathFidelitySetting.HIFI2,
                        },
                    }
                ),
            )

        self.input_layernorm = TtMinistralRMSNorm(
            mesh_device,
            model_args,
            meta_state_dict,
            weight_cache_path,
            layer_num,
            tt_ccl,
            post_attention=False,
        )
        self.self_attn = TtMinistralAttention(
            mesh_device,
            tt_ccl,
            model_args,
            meta_state_dict,
            model_args.weight_cache_path(ttnn.bfloat8_b)
            if use_reduced_precision and weight_cache_path is not None
            else weight_cache_path,
            layer_num,
            dtype,
            transformation_mats,
            configuration=configuration,
            llama_4_scaling_beta=llama_4_scaling_beta,
            original_max_position_embeddings=original_max_position_embeddings,
            paged_attention_config=paged_attention_config,
        )
        self.post_attention_layernorm = TtMinistralRMSNorm(
            mesh_device,
            model_args,
            meta_state_dict,
            weight_cache_path,
            layer_num,
            tt_ccl,
            post_attention=True,
        )
        self.mlp = TtMinistralMLP(
            mesh_device,
            tt_ccl,
            model_args,
            meta_state_dict,
            weight_cache_path,
            layer_num,
            dtype,
            model_args.get_model_config(),
        )

    def forward_prefill(
        self,
        x_11SH: ttnn.Tensor,
        rot_mats,
        position_ids: ttnn.Tensor | None = None,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        user_id: int = 0,
    ) -> ttnn.Tensor:
        """``x_11SH``: ``[batch, 1, seq, hidden]`` (same layout as :meth:`TtMinistralAttention.forward_prefill`)."""
        args = self.self_attn.args
        seq_len = int(x_11SH.shape[-2])
        use_block_sharded_adds = ministral_prefill_block_shard_add_eligible(seq_len)
        block_shard_mem = ministral_prefill_block_shard_mem_cfg(args, seq_len) if use_block_sharded_adds else None

        residual = x_11SH
        residual_bs = None
        owned_residual_bs = False
        if use_block_sharded_adds:
            if ministral_prefill_is_block_sharded_l1(residual):
                residual_bs = residual
            else:
                residual_bs = ministral_prefill_prepare_block_shard_input(residual, args, seq_len)
                owned_residual_bs = residual_bs is not residual

        qkv_in_mem = self.self_attn.get_prefill_qkv_input_mem_config(seq_len)
        if use_block_sharded_adds:
            attn_norm_cfg = ministral_prefill_block_shard_norm_config(args, seq_len, output_mem_cfg=qkv_in_mem)
            h = self.input_layernorm(
                residual_bs,
                Mode.PREFILL,
                in_sharded=True,
                out_sharded=qkv_in_mem is not None,
                norm_config=attn_norm_cfg,
            )
        else:
            h = self.input_layernorm(
                x_11SH,
                Mode.PREFILL,
                out_sharded=qkv_in_mem is not None,
                norm_config={"sharded_output_config": qkv_in_mem} if qkv_in_mem is not None else None,
            )
        attn_out = self.self_attn.forward_prefill(
            h,
            rot_mats,
            user_id=user_id,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
            position_ids=position_ids,
        )
        if self.self_attn.num_devices > 1:  # gather attn output before residual add
            attn_out = ttnn.all_gather(
                attn_out,
                dim=3,
                num_links=self.tt_ccl.get_num_links(),
                topology=args.ccl_topology(),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        if use_block_sharded_adds:
            attn_bs = ministral_prefill_prepare_block_shard_input(attn_out, args, seq_len)
            if attn_bs is not attn_out:
                ttnn.deallocate(attn_out)
            h = ttnn.add(residual_bs, attn_bs, memory_config=block_shard_mem)
            if owned_residual_bs:
                ttnn.deallocate(residual_bs)
            ttnn.deallocate(attn_bs)
            residual_mlp = h
        else:
            skip_mem_cfg = residual.memory_config()
            attn_out = ttnn.to_memory_config(attn_out, skip_mem_cfg)
            h = ttnn.add(residual, attn_out, memory_config=skip_mem_cfg)
            residual_mlp = h

        ff1_input_mem_cfg = self.mlp.get_prefill_ff1_input_mem_config(seq_len)
        if use_block_sharded_adds:
            post_norm_cfg = ministral_prefill_block_shard_norm_config(args, seq_len, output_mem_cfg=ff1_input_mem_cfg)
            h = self.post_attention_layernorm(
                h,
                Mode.PREFILL,
                in_sharded=True,
                out_sharded=ff1_input_mem_cfg is not None,
                norm_config=post_norm_cfg,
            )
        else:
            h = self.post_attention_layernorm(
                h,
                Mode.PREFILL,
                out_sharded=ff1_input_mem_cfg is not None,
                norm_config={"sharded_output_config": ff1_input_mem_cfg} if ff1_input_mem_cfg is not None else None,
            )
        ff_out = self.mlp(h, Mode.PREFILL)
        if self.self_attn.num_devices > 1:
            ff_out = ttnn.all_gather(
                ff_out,
                dim=3,
                num_links=self.tt_ccl.get_num_links(),
                topology=args.ccl_topology(),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        if use_block_sharded_adds:
            ff_bs = ministral_prefill_prepare_block_shard_input(ff_out, args, seq_len)
            if ff_bs is not ff_out:
                ttnn.deallocate(ff_out)
            out = ttnn.add(residual_mlp, ff_bs, memory_config=block_shard_mem)
            ttnn.deallocate(ff_bs)
            return out
        ff_out = ttnn.to_memory_config(ff_out, residual_mlp.memory_config())
        return ttnn.add(residual_mlp, ff_out, memory_config=residual_mlp.memory_config())

    def forward_decode(
        self,
        x: ttnn.Tensor,
        current_pos,
        rot_mats,
        user_id: int = 0,
        page_table=None,
    ) -> ttnn.Tensor:
        """Decode one token (width-sharded ``x``, ``current_pos`` [1,batch], ``rot_mats`` from cache)."""
        args = self.self_attn.args
        num_devices = self.self_attn.num_devices
        residual_mem_cfg = args.get_residual_mem_config(Mode.DECODE, None)
        attn_input_mem_cfg = args.get_attn_input_mem_config(Mode.DECODE, None)
        mlp_input_mem_cfg = args.get_mlp_input_mem_config(Mode.DECODE, None)

        x = ttnn.to_memory_config(x, residual_mem_cfg)
        x_dram = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        h_normed = self.input_layernorm(x_dram, Mode.DECODE)
        ttnn.deallocate(x_dram)
        if num_devices > 1:
            h_normed = ttnn.all_gather(
                h_normed,
                dim=3,
                num_links=self.tt_ccl.get_num_links(),
                topology=args.ccl_topology(),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        h = ttnn.to_memory_config(h_normed, attn_input_mem_cfg)
        ttnn.deallocate(h_normed)

        attn_out = self.self_attn.forward(
            h,
            current_pos,
            rot_mats,
            user_id=user_id,
            mode=Mode.DECODE,
            page_table=page_table,
        )
        attn_out = ttnn.to_memory_config(attn_out, residual_mem_cfg)
        skip1 = ttnn.add(x, attn_out, memory_config=residual_mem_cfg)
        ttnn.deallocate(attn_out)

        skip1_dram = ttnn.to_memory_config(skip1, ttnn.DRAM_MEMORY_CONFIG)
        h2_normed = self.post_attention_layernorm(skip1_dram, Mode.DECODE)
        ttnn.deallocate(skip1_dram)
        if num_devices > 1:
            h2_normed = ttnn.all_gather(
                h2_normed,
                dim=3,
                num_links=self.tt_ccl.get_num_links(),
                topology=args.ccl_topology(),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        h2 = ttnn.to_memory_config(h2_normed, mlp_input_mem_cfg)
        ttnn.deallocate(h2_normed)

        ff_out = self.mlp(h2, Mode.DECODE)  # mlp deallocates h2 internally
        ff_out = ttnn.to_memory_config(ff_out, residual_mem_cfg)
        result = ttnn.add(skip1, ff_out, memory_config=residual_mem_cfg)
        ttnn.deallocate(ff_out)
        ttnn.deallocate(skip1)
        return result


__all__ = ["TtMinistral3DecoderLayer"]
