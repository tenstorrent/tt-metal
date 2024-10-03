# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import ttnn
from typing import Optional
from models.demos.wormhole.llama31_8b_N300.tt.llama_attention import TtLlamaAttention
from models.demos.wormhole.llama31_8b_N300.tt.llama_mlp import TtLlamaMLP
from models.common.rmsnorm import RMSNorm


class TtTransformerBlock(torch.nn.Module):
    def __init__(self, args, mesh_device, dtype, state_dict, layer_num, weight_cache_path):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.num_devices = 1

        self.args = args
        self.hidden_size = args.dim
        self.n_heads = args.n_heads
        self.head_dim = self.hidden_size // self.n_heads
        self.max_seq_len = args.max_seq_len
        self.dim = args.dim
        self.max_batch_size = args.max_batch_size
        self.n_kv_heads = args.n_kv_heads
        self.current = 0
        self.sliding_window = args.sliding_window
        self.model_config = args.get_model_config()

        self.layer_num = layer_num
        self.n_local_heads = self.n_heads // self.num_devices
        self.n_local_kv_heads = self.n_kv_heads // self.num_devices

        self.attention = TtLlamaAttention(
            mesh_device=mesh_device,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            configuration=args,
        )
        self.feed_forward = TtLlamaMLP(
            mesh_device=mesh_device,
            args=args,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            model_config=self.model_config,
        )
        self.attention_norm = RMSNorm(
            device=mesh_device,
            dim=args.dim,
            state_dict=state_dict,
            layer_num=layer_num,
            weight_cache_path=None if args.dummy_weights else weight_cache_path,
            weight_dtype=dtype,
            weight_key="attention_norm",
        )
        self.ffn_norm = RMSNorm(
            device=mesh_device,
            dim=args.dim,
            state_dict=state_dict,
            layer_num=layer_num,
            weight_cache_path=None if args.dummy_weights else weight_cache_path,
            weight_dtype=dtype,
            weight_key="ffn_norm",
        )

    def forward(
        self,
        x: ttnn.Tensor,
        current_pos,
        rot_mat=None,
        transformation_mats=None,
        user_id=0,
        mode="decode",
        page_table=None,
    ) -> ttnn.Tensor:
        if mode == "prefill":
            skip_mem_cfg = ttnn.DRAM_MEMORY_CONFIG
        else:
            skip_mem_cfg = self.model_config["DEC_SKIP_OUTPUT_MEMCFG"]
        attn_norm = self.attention_norm(x)
        # Attention module expects a list of inputs (multi-device support)
        r = self.attention.forward(
            attn_norm,
            current_pos,
            rot_mat,
            transformation_mats,
            user_id,
            mode,
            page_table,
        )
        h = ttnn.add(x, r, memory_config=skip_mem_cfg)
        r = self.feed_forward.forward(self.ffn_norm(h), mode)
        out = ttnn.add(h, r, memory_config=skip_mem_cfg)
        return out
