# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.demos.qwen25_vl.tt.vision_attention import VisionAttention
from models.demos.qwen25_vl.tt.vision_mlp import MLP


class VisionBlock(LightweightModule):
    def __init__(
        self,
        args,
        mesh_device,
        dtype,
        state_dict,
        layer_num,
        weight_cache_path,
        transformation_mats,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.args = args
        self.hidden_size = args.vision_dim
        self.n_heads = args.vision_n_heads
        self.head_dim = self.hidden_size // self.n_heads
        self.max_seq_len = args.max_seq_len
        self.dim = args.vision_dim
        self.max_batch_size = args.max_batch_size
        self.n_kv_heads = args.vision_n_kv_heads
        self.current = 0
        self.model_config = args.get_model_config()

        self.layer_num = layer_num

        self.attention = VisionAttention(
            mesh_device=mesh_device,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            transformation_mats=transformation_mats,
            configuration=args,
        )
        self.feed_forward = MLP(
            mesh_device=mesh_device,
            args=args,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
        )
        # TODO: remove after https://github.com/tenstorrent/tt-metal/issues/35650 is fixed
        extra_rmsnorm_kwargs = {}
        if args.base_model_name in ("Qwen2.5-VL-7B",):
            extra_rmsnorm_kwargs["fp32_dest_acc_en"] = False
        self.attention_norm = RMSNorm(
            device=mesh_device,
            dim=args.vision_dim,
            eps=1e-6,  # Qwen2_5_VLVisionBlock hard-codes this
            state_dict=state_dict,
            state_dict_prefix=args.get_state_dict_prefix("", layer_num),
            weight_cache_path=None if args.dummy_weights else weight_cache_path,
            weight_dtype=ttnn.bfloat16,
            weight_key="norm1",
            **extra_rmsnorm_kwargs,
        )
        # args.dim = 1280
        self.ff_norm = RMSNorm(
            device=mesh_device,
            dim=args.vision_dim,
            eps=1e-6,  # Qwen2_5_VLVisionBlock hard-codes this
            state_dict=state_dict,
            state_dict_prefix=args.get_state_dict_prefix("", layer_num),
            weight_cache_path=None if args.dummy_weights else weight_cache_path,
            weight_dtype=ttnn.bfloat16,
            weight_key="norm2",
            **extra_rmsnorm_kwargs,
        )

    def forward(
        self,
        x: ttnn.Tensor,
        cu_seqlens,
        rot_mats,
    ) -> ttnn.Tensor:
        # x is fractured across devices and interleaved in DRAM (for prefill) and sharded in L1 (for decode)
        skip_mem_cfg = ttnn.DRAM_MEMORY_CONFIG
        assert (
            x.memory_config() == skip_mem_cfg
        ), f"VisionBlock input memcfg mismatch: {x.memory_config()} != {skip_mem_cfg}"
        # Norms take fractured inputs and output replicated across devices

        attn_in = self.attention_norm(x, mode="prefill")
        # Attention takes replicated inputs and produces fractured outputs
        attn_out = self.attention.forward(
            attn_in,
            cu_seqlens=cu_seqlens,
            rot_mats=rot_mats,
        )

        # Here x and attn_out are both fractured across devices
        h = ttnn.add(x, attn_out, memory_config=skip_mem_cfg, dtype=None)
        ttnn.deallocate(attn_out)
        ttnn.deallocate(x)

        # Norms take fractured inputs and output replicated across devices
        ff_in = self.ff_norm(h, mode="prefill")
        # MLP takes replicated inputs and produces fractured outputs
        ff_out = self.feed_forward.forward(ff_in, mode="prefill")
        ttnn.deallocate(ff_in)
        # ff_out and h are both fractured across devices
        out = ttnn.add(
            h,
            ff_out,
            memory_config=skip_mem_cfg,
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(h)
        ttnn.deallocate(ff_out)

        return out  # fractured across devices
