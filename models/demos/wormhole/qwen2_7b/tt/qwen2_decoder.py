# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import ttnn
from typing import Optional
from models.demos.wormhole.qwen2_7b.tt.qwen2_attention import TtQwen2Attention
from models.demos.wormhole.qwen2_7b.tt.qwen2_mlp import TtQwen2MLP
from models.common.rmsnorm import RMSNorm


class TtTransformerBlock(torch.nn.Module):
    def __init__(self, args, device, dtype, state_dict, layer_num, weight_cache_path, rot_mat, start_pos):
        super().__init__()

        self.state_dict = state_dict
        self.device = device
        self.num_devices = 1
        self.start_pos = start_pos

        self.args = args
        self.hidden_size = args.dim
        self.n_heads = args.n_heads
        self.head_dim = self.hidden_size // self.n_heads
        self.max_seq_len = args.max_seq_len
        self.dim = args.dim
        self.max_batch_size = args.max_batch_size
        self.n_kv_heads = args.n_kv_heads
        self.sliding_window = args.sliding_window
        self.model_config = args.get_model_config()

        self.layer_num = layer_num
        self.n_local_heads = self.n_heads // self.num_devices
        self.n_local_kv_heads = self.n_kv_heads // self.num_devices

        self.attention = TtQwen2Attention(
            devices=[device],
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            configuration=args,
            rot_mat=rot_mat,
            start_pos=start_pos,
        )
        self.feed_forward = TtQwen2MLP(
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
            weight_cache_path=weight_cache_path,
            weight_dtype=dtype,
            weight_key=None,
            weight_name=f"model.layers.{layer_num}.input_layernorm.weight",
        )
        self.ffn_norm = RMSNorm(
            device=device,
            dim=args.dim,
            state_dict=state_dict,
            layer_num=layer_num,
            weight_cache_path=weight_cache_path,
            weight_dtype=dtype,
            weight_key=None,
            weight_name=f"model.layers.{layer_num}.post_attention_layernorm.weight",
        )

    def forward(
        self,
        x: ttnn.Tensor,
        current_pos: int,
        attn_masks: Optional[ttnn.Tensor] = None,
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
        # Attention module expects a list of inputs, attn masks (multi-device support)
        r = self.attention.forward(
            [attn_norm],
            current_pos,
            [attn_masks],
            rot_mat,
            transformation_mats,
            user_id,
            mode,
        )
        # Attention also returns multiple outputs (multi-device support)
        assert len(r) == 1, "Multiple devices not yet supported"
        r = r[0]
        # r = ttnn.reshape(r, (1, 1, 32, 4096))
        h = ttnn.add(x, r, memory_config=skip_mem_cfg)
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = ttnn.add(h, r, memory_config=skip_mem_cfg)
        return out
