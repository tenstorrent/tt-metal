# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import ttnn
from typing import Optional
from models.demos.wormhole.llama31_8b.tt.llama_attention import TtLlamaAttention
from models.demos.wormhole.llama31_8b.tt.llama_mlp import TtLlamaMLP
from models.common.rmsnorm import RMSNorm


class TtTransformerBlock(torch.nn.Module):
    def __init__(self, args, device, dtype, state_dict, layer_num, weight_cache_path):
        super().__init__()

        self.state_dict = state_dict
        self.device = device
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
            devices=[device],
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            configuration=args,
        )
        self.feed_forward = TtLlamaMLP(
            device=device,
            args=args,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            model_config=self.model_config,
        )
        self.attention_norm = RMSNorm(
            device=device,
            dim=args.dim,
            state_dict=state_dict,
            layer_num=layer_num,
            weight_cache_path=None if args.dummy_weights else weight_cache_path,
            weight_dtype=dtype,
            weight_key="attention_norm",
        )
        self.ffn_norm = RMSNorm(
            device=device,
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
        current_pos_attn,
        rot_mat=None,
        transformation_mats=None,
        user_id=0,
        mode="decode",
    ) -> ttnn.Tensor:
        if mode == "prefill":
            skip_mem_cfg = ttnn.DRAM_MEMORY_CONFIG
        else:
            skip_mem_cfg = self.model_config["DEC_SKIP_OUTPUT_MEMCFG"]
        attn_norm = self.attention_norm(x)
        # Attention module expects a list of inputs (multi-device support)
        r = self.attention.forward(
            [attn_norm],
            current_pos,
            current_pos_attn,
            rot_mat,
            transformation_mats,
            user_id,
            mode,
        )
        # Attention also returns multiple outputs (multi-device support)
        assert len(r) == 1, "Multiple devices not yet supported"

        if mode == "decode":  # Sharded config on attn and ffn
            r_sharded = r[0]
            x_sharded = ttnn.interleaved_to_sharded(x, self.model_config["SHARDED_SKIP_INPUT_MEMCFG"])
            ttnn.deallocate(x)
            h_sharded = ttnn.add(x_sharded, r_sharded, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)
            ttnn.deallocate(x_sharded)
            ttnn.deallocate(r_sharded)
            ff_norm = self.ffn_norm(h_sharded, in_sharded=True, out_sharded=True)
            # Reshard the activations (grid_config = [4, 8] after attention) to match the MLP sharded grid_config [8, 8]
            ff_norm = ttnn.reshard(ff_norm, self.model_config["SHARDED_MLP_DECODE_INPUT_MEMCFG"])

            r_interleaved = self.feed_forward.forward(ff_norm, mode)
            h_interleaved = ttnn.sharded_to_interleaved(h_sharded, ttnn.L1_MEMORY_CONFIG)  # Final output is interleaved
            ttnn.deallocate(h_sharded)
            out = ttnn.add(h_interleaved, r_interleaved, memory_config=skip_mem_cfg)
            ttnn.deallocate(h_interleaved)
            ttnn.deallocate(r_interleaved)
            return out
        else:  # prefill  (Interleaved configs)
            r = r[0]
            h = ttnn.add(x, r, memory_config=skip_mem_cfg)
            r = self.feed_forward.forward(self.ffn_norm(h), mode)
            out = ttnn.add(h, r, memory_config=skip_mem_cfg)
            return out
