# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import time

from models.common.lightweightmodule import LightweightModule
from models.experimental.stable_diffusion_xl_base.tt.sdxl_utility import prepare_linear_params
from models.experimental.stable_diffusion_xl_base.tt.lora_weights_logger import lora_logger


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

        # Log module initialization start
        lora_logger.log_module_start(module_path, "TtAttention")

        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.inner_kv_dim = self.inner_dim if kv_heads is None else dim_head * kv_heads
        self.query_dim = query_dim
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.head_dim = dim_head

        # LORA WEIGHT: Query projection weights - primary target for LoRA fine-tuning
        q_weights = state_dict[f"{module_path}.to_q.weight"].unsqueeze(0).unsqueeze(0)
        # LORA WEIGHT: Key projection weights - primary target for LoRA fine-tuning
        k_weights = state_dict[f"{module_path}.to_k.weight"].unsqueeze(0).unsqueeze(0)
        # LORA WEIGHT: Value projection weights - primary target for LoRA fine-tuning
        v_weights = state_dict[f"{module_path}.to_v.weight"].unsqueeze(0).unsqueeze(0)

        # LORA WEIGHT: Output projection weights - commonly targeted for LoRA adaptation
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
            # self.tt_qkv_weights = ttnn.from_torch(
            #     fused_qkv_weights, attention_weights_dtype, device=device, layout=ttnn.TILE_LAYOUT
            # )
            fused_qkv_weights = fused_qkv_weights.to(torch.bfloat16)
            ttnn.synchronize_device(device)
            start_time = time.perf_counter()
            tt_weights_host = ttnn.from_torch(fused_qkv_weights, attention_weights_dtype)
            # ttnn.synchronize_device(device)
            host_creation_time_ms = (time.perf_counter() - start_time) * 1000
            tt_weights_host = ttnn.to_layout(tt_weights_host, ttnn.TILE_LAYOUT)
            self.tt_qkv_weights = ttnn.allocate_tensor_on_device(tt_weights_host.spec, device)

            ttnn.synchronize_device(device)
            start_time = time.perf_counter()
            ttnn.copy_host_to_device_tensor(tt_weights_host, self.tt_qkv_weights)
            ttnn.synchronize_device(device)
            host_to_device_time_ms = (time.perf_counter() - start_time) * 1000
            # LORA WEIGHT LOG: Fused QKV weights for self-attention
            lora_logger.log_weight_creation(
                module_path,
                "tt_qkv_weights",
                self.tt_qkv_weights.shape,
                attention_weights_dtype,
                device,
                "Fused Q+K+V weights for self-attention",
                tensor_obj=self.tt_qkv_weights,
                host_creation_time_ms=host_creation_time_ms,
                host_to_device_time_ms=host_to_device_time_ms,
            )
        else:
            self.tt_q_weights, _, host_creation_ms, host_to_device_ms = prepare_linear_params(
                device, q_weights, None, attention_weights_dtype, is_lora_impacted=True
            )
            # LORA WEIGHT LOG: Query weights for cross-attention
            lora_logger.log_weight_creation(
                module_path,
                "tt_q_weights",
                self.tt_q_weights.shape,
                attention_weights_dtype,
                device,
                "Query projection weights for cross-attention",
                tensor_obj=self.tt_q_weights,
                host_creation_time_ms=host_creation_ms,
                host_to_device_time_ms=host_to_device_ms,
            )

            self.tt_k_weights, _, host_creation_ms, host_to_device_ms = prepare_linear_params(
                device, k_weights, None, attention_weights_dtype, is_lora_impacted=True
            )
            # LORA WEIGHT LOG: Key weights for cross-attention
            lora_logger.log_weight_creation(
                module_path,
                "tt_k_weights",
                self.tt_k_weights.shape,
                attention_weights_dtype,
                device,
                "Key projection weights for cross-attention",
                tensor_obj=self.tt_k_weights,
                host_creation_time_ms=host_creation_ms,
                host_to_device_time_ms=host_to_device_ms,
            )

            self.tt_v_weights, _, host_creation_ms, host_to_device_ms = prepare_linear_params(
                device, v_weights, None, attention_weights_dtype, is_lora_impacted=True
            )
            # LORA WEIGHT LOG: Value weights for cross-attention
            lora_logger.log_weight_creation(
                module_path,
                "tt_v_weights",
                self.tt_v_weights.shape,
                attention_weights_dtype,
                device,
                "Value projection weights for cross-attention",
                tensor_obj=self.tt_v_weights,
                host_creation_time_ms=host_creation_ms,
                host_to_device_time_ms=host_to_device_ms,
            )

            self.k_program_config = model_config.get_matmul_config(f"{module_path}.to_k")
            self.v_program_config = model_config.get_matmul_config(f"{module_path}.to_v")

            self.k_memory_config = model_config.get_mm_output_memory_config(f"{module_path}.to_k")
            self.v_memory_config = model_config.get_mm_output_memory_config(f"{module_path}.to_v")

        self.tt_out_weights, self.tt_out_bias, host_creation_ms, host_to_device_ms = prepare_linear_params(
            device, out_weights, out_bias, attention_weights_dtype, is_lora_impacted=True
        )
        # LORA WEIGHT LOG: Output projection weights
        lora_logger.log_weight_creation(
            module_path,
            "tt_out_weights",
            self.tt_out_weights.shape,
            attention_weights_dtype,
            device,
            "Output projection weights",
            tensor_obj=self.tt_out_weights,
            host_creation_time_ms=host_creation_ms,
            host_to_device_time_ms=host_to_device_ms,
        )

        self.q_program_config = model_config.get_matmul_config(f"{module_path}.to_q")
        self.q_compute_kernel_config = model_config.get_mm_compute_config(f"{module_path}.to_q")
        self.q_memory_config = model_config.get_mm_output_memory_config(f"{module_path}.to_q")

        self.dense_out_program_config = model_config.get_matmul_config(f"{module_path}.to_out")
        self.default_compute_kernel_config = model_config.get_mm_compute_config(f"{module_path}.to_out")
        self.out_memory_config = model_config.get_mm_output_memory_config(f"{module_path}.to_out")

        # Log module initialization end
        lora_logger.log_module_end(module_path, "TtAttention")

    def forward(self, hidden_states, attention_mask, encoder_hidden_states=None):
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        B, C, H, W = list(hidden_states.shape)

        if self.is_self_attention:
            qkv_fused = ttnn.matmul(
                hidden_states,
                self.tt_qkv_weights,
                memory_config=self.q_memory_config,
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
                program_config=self.q_program_config,
                compute_kernel_config=self.q_compute_kernel_config,
                memory_config=self.q_memory_config,
            )
            k_heads = ttnn.matmul(
                encoder_hidden_states,
                self.tt_k_weights,
                memory_config=self.k_memory_config,
                compute_kernel_config=self.default_compute_kernel_config,
                program_config=self.k_program_config,
            )
            v_heads = ttnn.matmul(
                encoder_hidden_states,
                self.tt_v_weights,
                memory_config=self.v_memory_config,
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
            memory_config=self.out_memory_config,
        )

        return hidden_states
