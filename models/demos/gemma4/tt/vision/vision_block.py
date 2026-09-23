# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.demos.gemma4.tt.vision.vision_attention import VisionAttention
from models.demos.gemma4.tt.vision.vision_mlp import Gemma4VisionMLP
from models.tt_transformers.tt.common import Mode


class VisionBlock(LightweightModule):
    def __init__(
        self,
        args,
        mesh_device,
        dtype,
        tt_ccl,
        state_dict,
        layer_num,
        weight_cache_path,
        transformation_mats,
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
        self.tt_ccl = tt_ccl

        self.layer_num = layer_num

        self.attention = VisionAttention(
            mesh_device=mesh_device,
            state_dict=state_dict,
            tt_ccl=tt_ccl,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            transformation_mats=transformation_mats,
            configuration=args,
        )
        self.feed_forward = Gemma4VisionMLP(
            mesh_device=mesh_device,
            args=args,
            tt_ccl=tt_ccl,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
        )
        self.input_norm = RMSNorm(
            device=mesh_device,
            dim=args.dim,
            eps=1e-6,  # Qwen2_5_VLVisionBlock hard-codes this
            state_dict=state_dict,
            state_dict_prefix=args.get_state_dict_prefix("input_layernorm", layer_num),
            weight_cache_path=None if args.dummy_weights else weight_cache_path,
            weight_dtype=ttnn.bfloat16,
            weight_key="",
            tt_ccl=tt_ccl,
        )
        self.post_attention_norm = RMSNorm(
            device=mesh_device,
            dim=args.dim,
            eps=1e-6,  # Qwen2_5_VLVisionBlock hard-codes this
            state_dict=state_dict,
            state_dict_prefix=args.get_state_dict_prefix("post_attention_layernorm", layer_num),
            weight_cache_path=None if args.dummy_weights else weight_cache_path,
            weight_dtype=ttnn.bfloat16,
            weight_key="",
            tt_ccl=tt_ccl,
        )
        self.pre_ff_norm = RMSNorm(
            device=mesh_device,
            dim=args.dim,
            eps=1e-6,  # Qwen2_5_VLVisionBlock hard-codes this
            state_dict=state_dict,
            state_dict_prefix=args.get_state_dict_prefix("pre_feedforward_layernorm", layer_num),
            weight_cache_path=None if args.dummy_weights else weight_cache_path,
            weight_dtype=ttnn.bfloat16,
            tt_ccl=tt_ccl,
            weight_key="",
        )
        # args.dim = 1280
        self.post_ff_norm = RMSNorm(
            device=mesh_device,
            dim=args.dim,
            eps=1e-6,  # Qwen2_5_VLVisionBlock hard-codes this
            state_dict=state_dict,
            state_dict_prefix=args.get_state_dict_prefix("post_feedforward_layernorm", layer_num),
            weight_cache_path=None if args.dummy_weights else weight_cache_path,
            weight_dtype=ttnn.bfloat16,
            tt_ccl=tt_ccl,
            weight_key="",
        )

    def forward(
        self,
        x: ttnn.Tensor,
        rot_mats,
    ) -> ttnn.Tensor:
        # x is fractured across devices and interleaved in DRAM (for prefill) and sharded in L1 (for decode)
        skip_mem_cfg = ttnn.DRAM_MEMORY_CONFIG
        assert (
            x.memory_config() == skip_mem_cfg
        ), f"VisionBlock input memcfg mismatch: {x.memory_config()} != {skip_mem_cfg}"

        attn_in = self.input_norm(x, mode=Mode.PREFILL)
        attn_out = self.attention.forward(
            attn_in,
            rot_mats=rot_mats,
        )
        attn_normed = self.post_attention_norm(attn_out, mode=Mode.PREFILL)
        h = ttnn.add(x, attn_normed, memory_config=skip_mem_cfg, dtype=None)
        ttnn.deallocate(attn_normed)
        ttnn.deallocate(attn_out)
        ttnn.deallocate(x)

        ff_in = self.pre_ff_norm(h, mode=Mode.PREFILL)
        ff_out = self.feed_forward.forward(ff_in, mode=Mode.PREFILL)
        ff_normed = self.post_ff_norm(ff_out, mode=Mode.PREFILL)
        ttnn.deallocate(ff_in)
        out = ttnn.add(
            h,
            ff_normed,
            memory_config=skip_mem_cfg,
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(h)
        ttnn.deallocate(ff_out)
        ttnn.deallocate(ff_normed)

        return out  # fractured across devices
