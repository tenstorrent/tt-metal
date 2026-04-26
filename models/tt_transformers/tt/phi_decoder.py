# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.layernorm import LayerNorm
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.decoder import TransformerBlock
from models.tt_transformers.tt.distributed_norm import DistributedNorm
from models.tt_transformers.tt.model_config import TensorGroup
from models.tt_transformers.tt.phi_mlp import Phi1MLP


class Phi1TransformerBlock(TransformerBlock):
    def _create_feed_forward(
        self,
        args,
        mesh_device,
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
        prefetcher,
    ):
        return Phi1MLP(
            mesh_device=mesh_device,
            tt_ccl=self.tt_ccl,
            args=args,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            model_config=self.model_config,
            prefetcher=prefetcher,
        )

    def _init_norms(self, args, mesh_device, state_dict, weight_cache_path, layer_num):
        self.input_norm = DistributedNorm(
            LayerNorm(
                device=mesh_device,
                dim=args.dim,
                eps=args.norm_eps,
                state_dict=state_dict,
                state_dict_prefix=args.get_state_dict_prefix("", layer_num),
                weight_cache_path=None if args.dummy_weights else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                weight_key="attention_norm",
                is_distributed=self.args.is_distributed_norm,
            ),
            args,
            tt_ccl=self.tt_ccl,
            prefetcher=self.prefetcher,
            TG=False,
            ag_config_key="ATTN_LN_AG_CONFIG",
        )
        self.attention_norm = self.input_norm
        self.ff_norm = None
        self.pre_ff_norm = None
        self.post_ff_norm = None

    def forward(
        self,
        x: ttnn.Tensor,
        current_pos,
        rot_mats_global=None,
        rot_mats_local=None,
        user_id=0,
        mode="decode",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
        batch_size=1,
    ) -> ttnn.Tensor:
        del batch_size
        TG = self.args.is_galaxy
        is_decode = mode == Mode.DECODE or mode == "decode"
        residual = x

        skip_mem_cfg = self.args.get_residual_mem_config(mode, self.prefetcher)
        assert (
            x.memory_config() == skip_mem_cfg
        ), f"decoder input memcfg mismatch: {x.memory_config()} != {skip_mem_cfg}"

        rot_mats = (
            rot_mats_local if (hasattr(self.attention, "is_sliding") and self.attention.is_sliding) else rot_mats_global
        )

        attn_norm_config = self.args.get_norm_config("attn", mode, self.prefetcher)
        normed_input = self.input_norm(x, mode, norm_config=attn_norm_config)

        mlp_input = ttnn.clone(
            normed_input,
            dtype=normed_input.dtype,
            memory_config=normed_input.memory_config(),
        )

        attn_out = self.attention.forward(
            normed_input,
            current_pos,
            rot_mats,
            user_id,
            mode,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
            kv_cache=kv_cache,
        )
        attn_out = ttnn.to_memory_config(attn_out, skip_mem_cfg)

        if TG and is_decode:
            mlp_input = ttnn.to_memory_config(mlp_input, memory_config=self.args.get_mlp_act_mem_config(mode))
        mlp_out = self.feed_forward.forward(mlp_input, mode)

        activation_dtype = self.args.decoders_optimizations.get_tensor_dtype(
            decoder_id=self.layer_num, tensor=TensorGroup.ACTIVATION
        )

        out = ttnn.add(
            residual,
            attn_out,
            memory_config=skip_mem_cfg,
            dtype=self.args.ccl_dtype
            if TG and not self.args.is_distributed_norm(mode)
            else activation_dtype or ttnn.bfloat16,
        )
        ttnn.deallocate(attn_out)

        out = ttnn.add(
            out,
            mlp_out,
            memory_config=skip_mem_cfg,
            dtype=self.args.ccl_dtype
            if TG and not self.args.is_distributed_norm(mode)
            else activation_dtype or ttnn.bfloat16,
        )
        ttnn.deallocate(mlp_out)
        return out
