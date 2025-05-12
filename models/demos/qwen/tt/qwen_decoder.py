# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.demos.qwen.tt.distributed_norm import DistributedNorm
from models.demos.qwen.tt.qwen_attention import TtQwenAttention
from models.demos.qwen.tt.qwen_mlp import TtQwenMLP


class TtTransformerBlock(LightweightModule):
    def __init__(self, args, mesh_device, dtype, state_dict, layer_num, weight_cache_path):
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
        self.sliding_window = args.sliding_window
        self.model_config = args.get_model_config()

        self.layer_num = layer_num

        self.attention = TtQwenAttention(
            mesh_device=mesh_device,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            configuration=args,
        )
        self.feed_forward = TtQwenMLP(
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
                state_dict=state_dict,
                state_dict_prefix=args.get_state_dict_prefix("", layer_num),
                weight_cache_path=None if args.dummy_weights else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                weight_key="input_layernorm",
                is_distributed=self.args.is_distributed_norm,
                sharded_program_config=self.model_config["SHARDED_NORM_ATTN_PRGM_CFG"],
                sharded_output_config=self.model_config["SHARDED_ATTN_INPUT_MEMCFG"],
            ),
            args,
        )
        self.ff_norm = DistributedNorm(
            RMSNorm(
                device=mesh_device,
                dim=args.dim,
                state_dict=state_dict,
                state_dict_prefix=args.get_state_dict_prefix("", layer_num),
                weight_cache_path=None if args.dummy_weights else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                weight_key="post_attention_layernorm",
                is_distributed=self.args.is_distributed_norm,
                sharded_program_config=self.model_config["SHARDED_NORM_MLP_PRGM_CFG"],
                sharded_output_config=self.model_config["SHARDED_MLP_INPUT_MEMCFG"],
            ),
            args,
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
        # x is fractured across devices and interleaved in DRAM (for prefill) and L1 (for decode)
        # FIXME: move to sharded residuals once support for this is added
        # FIXME: Currently, for decode mode, we are using DRAM intereleaved as L1 interleaved results in h being corrupted in MLP
        skip_mem_cfg = (
            ttnn.DRAM_MEMORY_CONFIG
            # self.model_config["DEC_SKIP_OUTPUT_MEMCFG"] if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG
        )

        # Norms take fractured inputs and output replicated across devices
        attn_in = self.attention_norm(x, mode)
        # Attention takes replicated inputs and produces fractured outputs
        attn_out = self.attention.forward(
            attn_in,
            current_pos,
            rot_mat,
            transformation_mats,
            user_id,
            mode,
            page_table,
        )

        # Here x and attn_out are both fractured across devices
        h = ttnn.add(x, attn_out, memory_config=skip_mem_cfg)

        # TODO: This deallocate may cause ND output. The reason seems to be related to either the input being on DRAM/L1 and the sharded spec in MLP using 32 cores instead of 16.
        # ttnn.deallocate(attn_out)

        # Norms take fractured inputs and output replicated across devices
        ff_in = self.ff_norm(h, mode)
        # MLP takes replicated inputs and produces fractured outputs
        ff_out = self.feed_forward.forward(ff_in, mode)

        # ff_out and h are both fractured across devices
        out = ttnn.add(h, ff_out, memory_config=skip_mem_cfg)

        return out  # fractured across devices
