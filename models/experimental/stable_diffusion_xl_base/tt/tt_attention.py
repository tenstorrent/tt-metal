# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.stable_diffusion_xl_base.tt.sdxl_utility import prepare_linear_params


class TtAttention(LightweightModule):
    def __init__(
        self,
        device,
        state_dict,
        module_path,
        model_config,
        query_dim: int,
        heads: int = 8,
        out_dim: int = None,
        kv_heads=None,
        dim_head: int = 64,
    ):
        super().__init__()
        self.device = device

        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.inner_kv_dim = self.inner_dim if kv_heads is None else dim_head * kv_heads
        self.query_dim = query_dim
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.head_dim = dim_head

        q_weights = state_dict[f"{module_path}.to_q.weight"].unsqueeze(0).unsqueeze(0)
        k_weights = state_dict[f"{module_path}.to_k.weight"].unsqueeze(0).unsqueeze(0)
        v_weights = state_dict[f"{module_path}.to_v.weight"].unsqueeze(0).unsqueeze(0)

        out_weights = state_dict[f"{module_path}.to_out.0.weight"].unsqueeze(0).unsqueeze(0)
        out_bias = state_dict[f"{module_path}.to_out.0.bias"]

        self.is_self_attention = (
            q_weights.shape[-1] == k_weights.shape[-1] and q_weights.shape[-1] == v_weights.shape[-1]
        )
        self.sdpa_program_config = model_config.get_sdpa_config(
            module_path=module_path, is_self_attention=self.is_self_attention
        )

        self.sdpa_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        attention_weights_dtype = model_config.attention_weights_dtype

        if self.is_self_attention == True:
            fused_qkv_weights = torch.cat(
                [
                    torch.transpose(q_weights, -2, -1),
                    torch.transpose(k_weights, -2, -1),
                    torch.transpose(v_weights, -2, -1),
                ],
                dim=-1,
            )
            self.tt_qkv_weights = ttnn.from_torch(
                fused_qkv_weights, attention_weights_dtype, device=device, layout=ttnn.TILE_LAYOUT
            )
        else:
            self.tt_q_weights, _ = prepare_linear_params(device, q_weights, None, attention_weights_dtype)
            self.tt_k_weights, _ = prepare_linear_params(device, k_weights, None, attention_weights_dtype)
            self.tt_v_weights, _ = prepare_linear_params(device, v_weights, None, attention_weights_dtype)

            self.k_program_config = model_config.get_matmul_config(f"{module_path}.to_k")
            self.v_program_config = model_config.get_matmul_config(f"{module_path}.to_v")

        self.tt_out_weights, self.tt_out_bias = prepare_linear_params(
            device, out_weights, out_bias, attention_weights_dtype
        )

        self.q_program_config = model_config.get_matmul_config(f"{module_path}.to_q")
        self.q_compute_kernel_config = model_config.get_mm_compute_config(f"{module_path}.to_q")

        self.dense_out_program_config = model_config.get_matmul_config(f"{module_path}.to_out")
        self.default_compute_kernel_config = model_config.get_mm_compute_config(f"{module_path}.to_out")

    def forward(self, hidden_states, attention_mask, encoder_hidden_states=None):
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        B, C, H, W = list(hidden_states.shape)

        if self.is_self_attention:
            qkv_fused = ttnn.matmul(
                hidden_states,
                self.tt_qkv_weights,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                compute_kernel_config=self.q_compute_kernel_config,
                program_config=self.q_program_config,
            )

            (
                q_heads,
                k_heads,
                v_heads,
            ) = ttnn.experimental.nlp_create_qkv_heads(
                qkv_fused, num_heads=self.heads, transpose_k_heads=False, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            ttnn.deallocate(qkv_fused)
        else:
            q_heads = ttnn.matmul(
                hidden_states,
                self.tt_q_weights,
                program_config=self.dense_out_program_config,
                compute_kernel_config=self.q_compute_kernel_config,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            k_heads = ttnn.matmul(
                encoder_hidden_states,
                self.tt_k_weights,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=self.default_compute_kernel_config,
                program_config=self.k_program_config,
            )
            v_heads = ttnn.matmul(
                encoder_hidden_states,
                self.tt_v_weights,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=self.default_compute_kernel_config,
                program_config=self.v_program_config,
            )

            q_heads, _, _ = ttnn.experimental.nlp_create_qkv_heads(
                q_heads,
                num_heads=self.heads,
                num_kv_heads=0,
                transpose_k_heads=False,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

            v_heads, _, _ = ttnn.experimental.nlp_create_qkv_heads(
                v_heads,
                num_heads=self.heads,
                num_kv_heads=0,
                transpose_k_heads=False,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

            k_heads, _, _ = ttnn.experimental.nlp_create_qkv_heads(
                k_heads,
                num_heads=self.heads,
                num_kv_heads=0,
                transpose_k_heads=False,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

        hidden_states = ttnn.transformer.scaled_dot_product_attention(
            q_heads,
            k_heads,
            v_heads,
            is_causal=False,
            attn_mask=attention_mask,
            program_config=self.sdpa_program_config,
            compute_kernel_config=self.sdpa_compute_kernel_config,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        hidden_states = ttnn.experimental.nlp_concat_heads(hidden_states, memory_config=ttnn.L1_MEMORY_CONFIG)

        hidden_states = ttnn.linear(
            hidden_states,
            self.tt_out_weights,
            bias=self.tt_out_bias,
            program_config=self.dense_out_program_config,
            compute_kernel_config=self.default_compute_kernel_config,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG if W == 1280 else ttnn.L1_MEMORY_CONFIG,
        )

        return hidden_states
