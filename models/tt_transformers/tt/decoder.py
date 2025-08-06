# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.tt_transformers.tt.attention import Attention as DefaultAttention
from models.tt_transformers.tt.distributed_norm import DistributedNorm
from models.tt_transformers.tt.mlp import MLP
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
        attention_class=None,
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

        ActualAttentionClass = attention_class if attention_class is not None else DefaultAttention

        self.attention = ActualAttentionClass(
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
        self.attention_norm = DistributedNorm(
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
                add_unit_offset=self.args.rms_norm_add_unit_offset,
                sharded_program_config=self.model_config["SHARDED_NORM_ATTN_PRGM_CFG"],
                sharded_output_config=self.model_config["SHARDED_ATTN_INPUT_MEMCFG"],
                ccl_topology=self.args.ccl_topology(),
            ),
            args,
            TG=args.is_galaxy,
        )

        if not self.args.use_pre_ffn:
            self.ff_norm = DistributedNorm(
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
                    add_unit_offset=self.args.rms_norm_add_unit_offset,
                    sharded_program_config=self.model_config["SHARDED_NORM_MLP_PRGM_CFG"],
                    sharded_output_config=self.model_config["SHARDED_MLP_INPUT_MEMCFG"],
                    ccl_topology=self.args.ccl_topology(),
                ),
                args,
                TG=args.is_galaxy,
            )

        if self.args.use_pre_ffn:
            self.ff_norm = DistributedNorm(
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
                    add_unit_offset=self.args.rms_norm_add_unit_offset,
                    sharded_program_config=self.model_config["SHARDED_NORM_ATTN_PRGM_CFG"],
                    sharded_output_config=self.model_config["SHARDED_ATTN_INPUT_MEMCFG"],
                    ccl_topology=self.args.ccl_topology(),
                ),
                args,
                TG=args.is_galaxy,
            )

            self.pre_ffn_norm = DistributedNorm(
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
                    add_unit_offset=self.args.rms_norm_add_unit_offset,
                    sharded_program_config=self.model_config["SHARDED_NORM_MLP_PRGM_CFG"],
                    sharded_output_config=self.model_config["SHARDED_MLP_INPUT_MEMCFG"],
                    ccl_topology=self.args.ccl_topology(),
                ),
                args,
                TG=args.is_galaxy,
            )
        else:
            self.pre_ffn_norm = None

        if self.args.use_post_ffn:
            self.post_ffn_norm = DistributedNorm(
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
                    add_unit_offset=self.args.rms_norm_add_unit_offset,
                    sharded_program_config=self.model_config["SHARDED_NORM_MLP_PRGM_CFG"],
                    sharded_output_config=self.model_config["SHARDED_MLP_INPUT_MEMCFG"],
                    ccl_topology=self.args.ccl_topology(),
                ),
                args,
                TG=args.is_galaxy,
            )
        else:
            self.post_ffn_norm = None

    def forward(
        self,
        x: ttnn.Tensor,
        current_pos,
        rot_mats=None,
        user_id=0,
        mode="decode",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
    ) -> ttnn.Tensor:
        TG = self.args.is_galaxy
        # x is fractured across devices and interleaved in DRAM (for prefill) and sharded in L1 (for decode)
        skip_mem_cfg = self.model_config["DECODE_RESIDUAL_MEMCFG"] if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG
        assert (
            x.memory_config() == skip_mem_cfg
        ), f"decoder input memcfg mismatch: {x.memory_config()} != {skip_mem_cfg}"
        # Norms take fractured inputs and output replicated across devices
        attn_in = self.attention_norm(x, mode)
        # Attention takes replicated inputs and produces fractured outputs
        attn_out = self.attention.forward(
            attn_in,
            current_pos,
            rot_mats,
            user_id,
            mode,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
            kv_cache=kv_cache,
        )
        attn_in.deallocate(True)

        if not (self.pre_ffn_norm and self.post_ffn_norm):
            """
            Llama like

            h = x + attn_out

            ff_in = post_attention_layernorm(h)
            ff_out = mlp(ff_in)
            out = h + ff_out
            """
            # Here x and attn_out are both fractured across devices
            h = ttnn.add(x, attn_out, memory_config=skip_mem_cfg, dtype=ttnn.bfloat16 if TG else None)
            ttnn.deallocate(attn_out)
            if mode == "prefill":
                x.deallocate(True)

            # Norms take fractured inputs and output replicated across devices
            ff_in = self.ff_norm(h, mode)
            if TG and mode == "decode":
                ff_in = ttnn.to_memory_config(ff_in, memory_config=self.model_config["MLP_ACT_MEMCFG"])

            ff_out = self.feed_forward.forward(ff_in, mode)

            # ff_out and h are both fractured across devices
            activation_dtype = self.model_config["DECODERS_OPTIMIZATIONS"].get_tensor_dtype(
                decoder_id=self.layer_num, tensor=TensorGroup.ACTIVATION
            )
            out = ttnn.add(
                h,
                ff_out,
                memory_config=skip_mem_cfg,
                dtype=self.args.ccl_dtype
                if TG and not self.args.is_distributed_norm(mode)
                else activation_dtype or ttnn.bfloat16,
            )
        else:
            """
            Gemma like

            post_attn_out = post_attention_layernorm(attn_out)
            add_out = x + post_attn_out

            pre_ffn_out = pre_feedforward_layernorm(add_out)
            ff_out = mlp(pre_ffn_out)
            post_ffn_out = post_feedforward_layernorm(ff_out)
            out = add_out + post_ffn_out
            """
            # Here x and attn_out are both fractured across devices
            post_attn_out = self.ff_norm(attn_out, mode)
            attn_out.deallocate(True)

            add_out = ttnn.add(
                x,
                post_attn_out,
                memory_config=skip_mem_cfg,
                dtype=self.args.ccl_dtype,
            )
            x.deallocate(True)
            post_attn_out.deallocate(True)

            # Norms take fractured inputs and output replicated across devices
            if TG and mode == "decode":
                add_out = ttnn.to_memory_config(add_out, memory_config=self.model_config["MLP_ACT_MEMCFG"])

            # MLP takes replicated inputs and produces fractured outputs
            pre_ffn_out = self.pre_ffn_norm.forward(add_out, mode)
            ff_out = self.feed_forward.forward(pre_ffn_out, mode)
            pre_ffn_out.deallocate(True)
            post_ffn_out = self.post_ffn_norm.forward(ff_out, mode)
            ff_out.deallocate(True)

            # ff_out and h are both fractured across devices
            activation_dtype = self.model_config["DECODERS_OPTIMIZATIONS"].get_tensor_dtype(
                decoder_id=self.layer_num, tensor=TensorGroup.ACTIVATION
            )
            out = ttnn.add(
                add_out,
                post_ffn_out,
                memory_config=skip_mem_cfg,
                dtype=self.args.ccl_dtype
                if TG and not self.args.is_distributed_norm(mode)
                else activation_dtype or ttnn.bfloat16,
            )
            add_out.deallocate(True)
            post_ffn_out.deallocate(True)
        return out  # fractured across devices
