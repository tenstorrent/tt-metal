# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import ttnn
from models.demos.llama3_70b_galaxy.tt.llama_attention import TtLlamaAttention
from models.demos.llama3_70b_galaxy.tt.llama_mlp import TtLlamaMLP
from models.common.rmsnorm import RMSNorm
from models.common.lightweightmodule import LightweightModule
from models.demos.llama3_70b_galaxy.tt.distributed_norm import DistributedNorm

# Llama 3.3 70B Model Architecture
# LlamaForCausalLM(
#   (model): LlamaModel(
#     (embed_tokens): Embedding(128256, 8192)
#     (layers): ModuleList(
#       (0-79): 80 x LlamaDecoderLayer(
#         (self_attn): LlamaAttention(
#           (q_proj): Linear(in_features=8192, out_features=8192, bias=False)
#           (k_proj): Linear(in_features=8192, out_features=1024, bias=False)
#           (v_proj): Linear(in_features=8192, out_features=1024, bias=False)
#           (o_proj): Linear(in_features=8192, out_features=8192, bias=False)
#           (rotary_emb): LlamaRotaryEmbedding_L31()
#         )
#         (mlp): LlamaMLP(
#           (gate_proj): Linear(in_features=8192, out_features=28672, bias=False)
#           (up_proj): Linear(in_features=8192, out_features=28672, bias=False)
#           (down_proj): Linear(in_features=28672, out_features=8192, bias=False)
#           (act_fn): SiLUActivation()
#         )
#         (input_layernorm): LlamaRMSNorm()
#         (post_attention_layernorm): LlamaRMSNorm()
#       )
#     )
#     (norm): LlamaRMSNorm()
#   )
#   (lm_head): Linear(in_features=8192, out_features=128256, bias=False)
# )


# Eagle 3 Model Architecture
# Model(
#   (embed_tokens): Embedding(128256, 6144, padding_idx=0)
#   (lm_head): Linear(in_features=6144, out_features=32000, bias=False)
#   (midlayer): LlamaDecoderLayeremb(
#     (self_attn): LlamaAttention(
#       (q_proj): Linear(in_features=12288, out_features=6144, bias=False)
#       (k_proj): Linear(in_features=12288, out_features=1024, bias=False)
#       (v_proj): Linear(in_features=12288, out_features=1024, bias=False)
#       (o_proj): Linear(in_features=6144, out_features=6144, bias=False)
#       (rotary_emb): LlamaRotaryEmbedding()
#     )
#     (mlp): LlamaMLP(
#       (gate_proj): Linear(in_features=6144, out_features=16384, bias=False)
#       (up_proj): Linear(in_features=6144, out_features=16384, bias=False)
#       (down_proj): Linear(in_features=16384, out_features=6144, bias=False)
#       (act_fn): SiLUActivation()
#     )
#     (hidden_norm): LlamaRMSNorm()
#     (input_layernorm): LlamaRMSNorm()
#     (post_attention_layernorm): LlamaRMSNorm()
#   )
#   (fc): Linear(in_features=24576, out_features=6144, bias=False)
#   (norm): LlamaRMSNorm()
#   (logsoftmax): LogSoftmax(dim=-1)
# )


class TtEagleDecoder(LightweightModule):
    def __init__(
        self,
        args,
        mesh_device,
        dtype,
        state_dict,
        layer_num,
        n_layers,
        weight_cache_path,
        transformation_mats,
        paged_attention_config=None,
        use_paged_kv_cache=False,
        prefetcher_setup=None,
        tt_ccl=None,
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
        self.weight_cache_path = weight_cache_path
        self.current = 0
        self.model_config = args.get_model_config()

        self.layer_num = layer_num
        self.n_layers = n_layers

        self.prefetcher_setup = prefetcher_setup
        self.tt_ccl = tt_ccl

        self.attention = TtLlamaAttention(
            mesh_device=mesh_device,
            state_dict=state_dict,
            state_dict_prefix="midlayer",
            weight_key="self_attn",
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            transformation_mats=transformation_mats,
            configuration=args,
            paged_attention_config=paged_attention_config,
            use_paged_kv_cache=use_paged_kv_cache,
            prefetcher_setup=prefetcher_setup,
            tt_ccl=tt_ccl,
        )
        self.feed_forward = TtLlamaMLP(
            mesh_device=mesh_device,
            args=args,
            state_dict=state_dict,
            state_dict_prefix="midlayer.mlp",
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            model_config=self.model_config,
            prefetcher_setup=prefetcher_setup,
            tt_ccl=tt_ccl,
        )
        self.input_norm = DistributedNorm(
            RMSNorm(
                device=mesh_device,
                dim=args.dim,
                state_dict=state_dict,
                state_dict_prefix="midlayer",
                weight_cache_path=None if args.dummy_weights else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                weight_key="input_layernorm",
                is_distributed=self.args.is_distributed_norm,
                sharded_program_config=self.model_config["SHARDED_NORM_ATTN_PRGM_CFG"],
                sharded_output_config=self.model_config["SHARDED_ATTN_INPUT_MEMCFG"],
                output_mem_config=self.model_config["SHARDED_ATTN_INPUT_RING_MEMCFG"],
            ),
            args,
            TG=args.is_galaxy,
            tt_ccl=tt_ccl,
            ccl_topology=self.model_config["CCL_TOPOLOGY"],
        )

        self.attention_norm = DistributedNorm(
            RMSNorm(
                device=mesh_device,
                dim=args.dim,
                state_dict=state_dict,
                state_dict_prefix="midlayer",
                weight_cache_path=None if args.dummy_weights else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                weight_key="hidden_norm",
                is_distributed=self.args.is_distributed_norm,
                sharded_program_config=self.model_config["SHARDED_NORM_ATTN_PRGM_CFG"],
                sharded_output_config=self.model_config["SHARDED_ATTN_INPUT_MEMCFG"],
                output_mem_config=self.model_config["SHARDED_ATTN_INPUT_RING_MEMCFG"],
            ),
            args,
            TG=args.is_galaxy,
            tt_ccl=tt_ccl,
            ccl_topology=self.model_config["CCL_TOPOLOGY"],
        )
        self.ff_norm = DistributedNorm(
            RMSNorm(
                device=mesh_device,
                dim=args.dim,
                state_dict=state_dict,
                state_dict_prefix="midlayer",
                weight_cache_path=None if args.dummy_weights else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                weight_key="post_attention_layernorm",
                is_distributed=self.args.is_distributed_norm,
                sharded_program_config=self.model_config["SHARDED_NORM_MLP_PRGM_CFG"],
                sharded_output_config=self.model_config["SHARDED_MLP_INPUT_MEMCFG"],
                output_mem_config=self.model_config["SHARDED_FF12_RING_MEMCFG"],
            ),
            args,
            TG=args.is_galaxy,
            tt_ccl=tt_ccl,
            ccl_topology=self.model_config["CCL_TOPOLOGY"],
        )

    def prefetch(self, prefetcher_setup, tt_ccl):
        self.prefetcher_setup = prefetcher_setup
        self.tt_ccl = tt_ccl
        self.attention.prefetch(prefetcher_setup, tt_ccl)
        self.feed_forward.prefetch(prefetcher_setup, tt_ccl)
        self.attention_norm.tt_ccl = tt_ccl
        self.ff_norm.tt_ccl = tt_ccl

    def forward(
        self,
        input_emb: ttnn.Tensor,
        hidden_states: ttnn.Tensor,
        current_pos,
        rot_mats=None,
        user_id=0,
        page_table=None,
        kv_cache=None,
        batch_size=1,
    ) -> ttnn.Tensor:
        skip_mem_cfg = self.model_config["DECODE_RESIDUAL_MEMCFG"]

        residual = hidden_states
        attn_input_sharded, _ = self.attention_norm(hidden_states, None, "decode")
        input_emb_sharded, _ = self.input_norm(input_emb, None, "decode")

        attn_input_sharded = ttnn.concat([input_emb_sharded, attn_input_sharded], dim=-1, memory_config=skip_mem_cfg)

        attn_output = self.attention.forward(
            attn_input_sharded,
            current_pos,
            rot_mats,
            user_id,
            mode="decode",
            page_table=page_table,
            kv_cache=kv_cache,
            batch_size=batch_size,
        )
        attn_input_sharded.deallocate(True)

        ff_input_sharded, residual = self.ff_norm(attn_output, residual, "decode")
        attn_output.deallocate(True)

        ff_out = self.feed_forward.forward(ff_input_sharded, "decode")

        out = ttnn.add(ff_out, residual, memory_config=skip_mem_cfg)

        return out
