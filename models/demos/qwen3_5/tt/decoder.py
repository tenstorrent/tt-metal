# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 hybrid transformer block."""

from __future__ import annotations

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.demos.qwen3_5.tt.attention import Qwen3_5FullAttentionTT
from models.tt_transformers.tt.distributed_norm import DistributedNorm
from models.tt_transformers.tt.gated_delta_net import GatedDeltaNetTT
from models.tt_transformers.tt.mlp import MLP


class HybridTransformerBlock(LightweightModule):
    def __init__(
        self,
        args,
        mesh_device,
        tt_ccl,
        dtype,
        state_dict,
        layer_num,
        weight_cache_path,
        transformation_mats=None,
        paged_attention_config=None,
        use_paged_kv_cache=False,
        attention_class=None,
        prefetcher=None,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.args = args
        self.layer_num = layer_num
        self.model_config = args.get_model_config()
        self.prefetcher = prefetcher
        self.is_linear_attn = getattr(args, "linear_attention_pattern", [False] * (layer_num + 1))[layer_num]

        if self.is_linear_attn:
            self.attention = GatedDeltaNetTT(
                mesh_device=mesh_device,
                args=args,
                state_dict=state_dict,
                weight_cache_path=weight_cache_path,
                layer_num=layer_num,
                dtype=dtype,
            )
        else:
            self.attention = Qwen3_5FullAttentionTT(
                mesh_device=mesh_device,
                args=args,
                state_dict=state_dict,
                weight_cache_path=weight_cache_path,
                layer_num=layer_num,
                dtype=dtype,
            )

        self.feed_forward = MLP(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            args=args,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            model_config=self.model_config,
            prefetcher=prefetcher,
        )

        norm_kwargs = dict(
            dim=args.dim,
            eps=args.norm_eps,
            state_dict=state_dict,
            state_dict_prefix=args.get_state_dict_prefix("", layer_num),
            weight_cache_path=None if args.dummy_weights else weight_cache_path,
            weight_dtype=ttnn.bfloat16,
            is_distributed=args.is_distributed_norm,
            add_unit_offset=args.rms_norm_add_unit_offset,
            ccl_topology=args.ccl_topology(),
            tt_ccl=tt_ccl,
            fp32_dest_acc_en=False,  # Keeps L1 CB within 1.5MB for dim=5120
        )

        def make_norm(weight_key, ag_key):
            return DistributedNorm(
                RMSNorm(device=mesh_device, weight_key=weight_key, **norm_kwargs),
                args,
                tt_ccl=tt_ccl,
                prefetcher=prefetcher,
                TG=args.is_galaxy,
                ag_config_key=ag_key,
            )

        self.attention_norm = make_norm("attention_norm", "ATTN_LN_AG_CONFIG")
        self.ff_norm = make_norm("ffn_norm", "FFN_LN_AG_CONFIG")

    def forward(
        self,
        x,
        current_pos,
        rot_mats_global=None,
        rot_mats_local=None,
        user_id=0,
        mode="decode",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
        conv_state=None,
        recurrent_state=None,
    ):
        pass

        residual = x
        skip_mem_cfg = self.args.get_residual_mem_config(mode, None)

        attn_in = self.attention_norm(x, mode, norm_config=self.args.get_norm_config("attn", mode))

        if self.is_linear_attn:
            attn_out, conv_state_new, recurrent_state_new = self.attention.forward(
                attn_in,
                mode=mode,
                conv_state=conv_state,
                recurrent_state=recurrent_state,
            )
            new_kv = None
        else:
            attn_out, new_kv = self.attention.forward(
                attn_in,
                current_pos=current_pos,
                mode=mode,
                kv_cache=kv_cache,
            )
            conv_state_new = conv_state
            recurrent_state_new = recurrent_state

        ttnn.deallocate(attn_in)
        attn_out = ttnn.to_memory_config(attn_out, skip_mem_cfg)
        hidden_states = ttnn.add(residual, attn_out, memory_config=skip_mem_cfg)
        residual = hidden_states
        ttnn.deallocate(attn_out)

        hidden_states = self.ff_norm(hidden_states, mode, norm_config=self.args.get_norm_config("ff", mode))
        ff_out = self.feed_forward.forward(hidden_states, mode)
        output = ttnn.add(residual, ff_out, memory_config=skip_mem_cfg)
        ttnn.deallocate(ff_out)

        return output, new_kv, conv_state_new, recurrent_state_new
