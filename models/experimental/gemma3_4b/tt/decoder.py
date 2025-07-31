"""
source: models/tt_transformers/tt/decoder.py

This is the Decoder block for the gemma 3-4b-it model
We couldn't use the existing implementation in TT-Transformers because the usage of submodules is different

In Gemma-3-4b-it, The decoder Block has Additional pre_feedforward_layernorm and post_feedforward_layernorm,
And the logic of implementation is different from the existing implementation in TT-Transformers.
"""

# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.distributed_norm import DistributedNorm
from models.experimental.gemma3_4b.tt.rmsnorm import RMSNorm

from models.experimental.gemma3_4b.tt.attention import Attention

from models.experimental.gemma3_4b.tt.mlp import MLP
from models.tt_transformers.tt.model_config import TensorGroup


class TransformerBlock(LightweightModule):
    def __init__(
        self,
        args,
        mesh_device,
        dtype,
        state_dict,
        layer_num,
        weight_cache_path,
        transformation_mats,
        paged_attention_config=None,
        use_paged_kv_cache=False,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device

        self.args = args
        self.hidden_size = args.dim
        self.n_heads = args.n_heads
        self.head_dim = self.hidden_size // self.n_heads
        self.max_seq_len = args.max_seq_len
        self.dim = args.dim
        self.max_batch_size = args.max_batch_size
        self.n_kv_heads = args.n_kv_heads
        self.current = 0
        self.model_config = args.get_model_config()

        self.layer_num = layer_num

        self.attention = Attention(
            mesh_device=mesh_device,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            transformation_mats=transformation_mats,
            configuration=args,
            paged_attention_config=paged_attention_config,
            use_paged_kv_cache=use_paged_kv_cache,
        )
        self.feed_forward = MLP(
            mesh_device=mesh_device,
            args=args,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            model_config=self.model_config,
        )

        self.attention_norm = DistributedNorm(  # input_layernorm
            RMSNorm(
                device=mesh_device,
                dim=args.dim,
                eps=args.norm_eps,
                state_dict=state_dict,
                state_dict_prefix=args.get_state_dict_prefix("", layer_num),
                weight_cache_path=None if args.dummy_weights else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                weight_key="attention_norm",
                is_distributed=self.args.is_distributed_norm,
                sharded_program_config=self.model_config["SHARDED_NORM_ATTN_PRGM_CFG"],
                sharded_output_config=self.model_config["SHARDED_ATTN_INPUT_MEMCFG"],
                ccl_topology=self.args.ccl_topology(),
            ),
            args,
            TG=args.is_galaxy,
        )

        self.ff_norm = DistributedNorm(  # post_attention_layernorm
            RMSNorm(
                device=mesh_device,
                dim=args.dim,
                eps=args.norm_eps,
                state_dict=state_dict,
                state_dict_prefix=args.get_state_dict_prefix("", layer_num),
                weight_cache_path=None if args.dummy_weights else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                weight_key="ffn_norm",
                is_distributed=self.args.is_distributed_norm,
                sharded_program_config=self.model_config["SHARDED_NORM_MLP_PRGM_CFG"],
                sharded_output_config=self.model_config["SHARDED_MLP_INPUT_MEMCFG"],
                ccl_topology=self.args.ccl_topology(),
            ),
            args,
            TG=args.is_galaxy,
        )

        self.pre_ff_norm = DistributedNorm(  # pre_feedforward_layernorm
            RMSNorm(
                device=mesh_device,
                dim=args.dim,
                eps=args.norm_eps,
                state_dict=state_dict,
                state_dict_prefix=args.get_state_dict_prefix("", layer_num),
                weight_cache_path=None if args.dummy_weights else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                weight_key="pre_feedforward_layernorm",
                is_distributed=self.args.is_distributed_norm,
                sharded_program_config=self.model_config["SHARDED_NORM_MLP_PRGM_CFG"],
                sharded_output_config=self.model_config["SHARDED_MLP_INPUT_MEMCFG"],
                ccl_topology=self.args.ccl_topology(),
            ),
            args,
            TG=args.is_galaxy,
        )

        self.post_ff_norm = DistributedNorm(  # post_feedforward_layernorm
            RMSNorm(
                device=mesh_device,
                dim=args.dim,
                eps=args.norm_eps,
                state_dict=state_dict,
                state_dict_prefix=args.get_state_dict_prefix("", layer_num),
                weight_cache_path=None if args.dummy_weights else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                weight_key="post_feedforward_layernorm",
                is_distributed=self.args.is_distributed_norm,
                sharded_program_config=self.model_config["SHARDED_NORM_MLP_PRGM_CFG"],
                sharded_output_config=self.model_config["SHARDED_MLP_INPUT_MEMCFG"],
                ccl_topology=self.args.ccl_topology(),
            ),
            args,
            TG=args.is_galaxy,
        )

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        current_pos,
        rot_mats=None,
        user_id=0,
        mode="decode",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
    ):
        TG = self.args.is_galaxy
        skip_mem_cfg = self.model_config["DECODE_RESIDUAL_MEMCFG"] if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG

        assert (
            hidden_states.memory_config() == skip_mem_cfg
        ), f"decoder input memcfg mismatch: {hidden_states.memory_config()} != {skip_mem_cfg}"
        residual = hidden_states

        attn_in = self.attention_norm(hidden_states, mode)

        if self.attention.is_sliding:
            position_embeddings = rot_mats[1]
        else:
            position_embeddings = rot_mats[0]

        attn_out = self.attention.forward(
            attn_in,
            current_pos,
            position_embeddings,
            user_id,
            mode,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
            kv_cache=kv_cache,
        )

        hidden_states = self.ff_norm(attn_out, mode)

        ttnn.deallocate(attn_out)
        ttnn.deallocate(attn_in)

        hidden_states = ttnn.add(hidden_states, residual, memory_config=skip_mem_cfg, dtype=ttnn.bfloat16)

        residual = hidden_states

        hidden_states = self.pre_ff_norm(hidden_states, mode)

        if TG and mode == "decode":
            hidden_states = ttnn.to_memory_config(hidden_states, memory_config=self.model_config["MLP_ACT_MEMCFG"])

        hidden_states = self.feed_forward.forward(hidden_states, mode)

        activation_dtype = self.model_config["DECODERS_OPTIMIZATIONS"].get_tensor_dtype(
            decoder_id=self.layer_num, tensor=TensorGroup.ACTIVATION
        )

        hidden_states = self.post_ff_norm(hidden_states, mode)

        hidden_states = ttnn.add(
            hidden_states,
            residual,
            memory_config=skip_mem_cfg,
            dtype=self.args.ccl_dtype
            if TG and not self.args.is_distributed_norm(mode)
            else activation_dtype or ttnn.bfloat16,
        )

        return hidden_states
