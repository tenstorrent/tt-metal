# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.tt_transformers.tt.distributed_norm import DistributedNorm
from models.tt_transformers.tt.mlp import MLP
from models.tt_transformers.tt.multimodal.llama_cross_attention import TtLlamaCrossAttention


class TtLlamaCrossAttentionTransformerBlock(LightweightModule):
    def __init__(
        self,
        mesh_device,
        state_dict,
        state_dict_prefix,
        weight_cache_path,
        dtype,
        configuration,
        no_ffn=False,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.num_devices = configuration.num_devices
        self.n_heads = configuration.n_heads
        self.n_kv_heads = configuration.n_kv_heads
        self.hidden_size = configuration.dim
        self.head_dim = self.hidden_size // self.n_heads
        self.model_config = configuration.get_model_config()

        assert not no_ffn, "No FFN not supported"

        self.attention = TtLlamaCrossAttention(
            mesh_device,
            state_dict,
            state_dict_prefix=f"{state_dict_prefix}attention.",
            weight_cache_path=weight_cache_path,
            dtype=dtype,
            configuration=configuration,
            dim=self.hidden_size,
            head_dim=self.head_dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            norm_eps=configuration.norm_eps,
        )

        self.attention_norm = DistributedNorm(
            RMSNorm(
                device=mesh_device,
                dim=self.hidden_size,
                state_dict=state_dict,
                state_dict_prefix=state_dict_prefix,
                weight_cache_path=None if configuration.dummy_weights else weight_cache_path,
                weight_key="attention_norm",
                is_distributed=configuration.is_distributed_norm,
                sharded_program_config=self.model_config["SHARDED_NORM_ATTN_PRGM_CFG"],
                sharded_output_config=self.model_config["SHARDED_ATTN_INPUT_MEMCFG"],
            ),
            configuration,
        )

        self.gate_attn = ttnn.as_tensor(
            state_dict[f"{state_dict_prefix}gate_attn"].unsqueeze(0).expand(1, self.hidden_size),
            dtype=ttnn.bfloat16,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self.feed_forward = MLP(
            mesh_device=mesh_device,
            args=configuration,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=-1,
            dtype=dtype,
            model_config=self.model_config,
            state_dict_prefix=f"{state_dict_prefix}feed_forward",
        )

        self.ffn_norm = DistributedNorm(
            RMSNorm(
                device=mesh_device,
                dim=self.hidden_size,
                state_dict=state_dict,
                state_dict_prefix=state_dict_prefix,
                weight_cache_path=None if configuration.dummy_weights else weight_cache_path,
                weight_key="ffn_norm",
                is_distributed=configuration.is_distributed_norm,
                sharded_program_config=self.model_config["SHARDED_NORM_MLP_PRGM_CFG"],
                sharded_output_config=self.model_config["SHARDED_MLP_INPUT_MEMCFG"],
            ),
            configuration,
        )

        self.gate_ffwd = ttnn.as_tensor(
            state_dict[f"{state_dict_prefix}gate_ffwd"].unsqueeze(0).expand(1, self.hidden_size),
            dtype=ttnn.bfloat16,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward(
        self,
        x_11SH,
        xattn_mask,
        # Broadcast ops broken so pass in two copies of same mask, different shapes
        full_text_row_masked_out_mask_11SD,
        full_text_row_masked_out_mask_1NSH,
        xattn_cache,
        mode,
        user_id=0,
        vision_tokens=None,
        cross_page_table=None,
    ):
        skip_mem_cfg = self.model_config["DECODE_RESIDUAL_MEMCFG"] if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG
        assert (
            x_11SH.memory_config() == skip_mem_cfg
        ), f"decoder input memcfg mismatch: {x_11SH.memory_config()} != {skip_mem_cfg}"

        attn_out = self.attention(
            x_11SH=self.attention_norm(x_11SH, mode=mode),
            xattn_mask=xattn_mask,
            xattn_cache=xattn_cache,
            full_text_row_masked_out_mask_1NSH=full_text_row_masked_out_mask_1NSH,
            mode=mode,
            user_id=user_id,
            vision_tokens=vision_tokens,
            cross_page_table=cross_page_table,
        )
        # FIXME: DRAM workaround for No circular buffer with id error
        attn_out = ttnn.to_memory_config(attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn_out = ttnn.mul(attn_out, ttnn.tanh(self.gate_attn))

        res = ttnn.add(x_11SH, attn_out)
        mlp_out = self.feed_forward(self.ffn_norm(res, mode=mode), mode=mode)
        # FIXME: DRAM workaround for No circular buffer with id error
        mlp_out = ttnn.to_memory_config(mlp_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        mlp_out = ttnn.mul(mlp_out, full_text_row_masked_out_mask_11SD)
        mlp_out = ttnn.mul(mlp_out, ttnn.tanh(self.gate_ffwd))
        out = ttnn.add(res, mlp_out)
        return out
