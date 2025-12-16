"""Attention mechanism implementations for TTNN."""

import torch
from transformers.models.vit.modeling_vit import ViTSelfAttention
from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight

import ttnn
from models.tt_symbiote.core.module import TTNNModule
from models.tt_symbiote.modules.linear import TTNNLinear


class TTNNViTSelfAttention(TTNNModule):
    """TTNN-accelerated ViT Self-Attention layer."""

    def __init__(self, hidden_size, num_attention_heads, should_reallocate_in_attention=False, core_grid=None) -> None:
        super().__init__()

        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.scaling = self.attention_head_size**-0.5
        self.should_reallocate_in_attention = should_reallocate_in_attention
        self.query = None
        self.key = None
        self.value = None
        self.query_key_value_weight = None
        self.query_key_value_bias = None
        self.core_grid = core_grid if core_grid is not None else ttnn.CoreGrid(y=8, x=8)

    @classmethod
    def from_torch(cls, self_attention: ViTSelfAttention):
        """Create TTNNViTSelfAttention from PyTorch ViTSelfAttention."""
        new_self_attention = TTNNViTSelfAttention(
            hidden_size=self_attention.config.hidden_size,
            num_attention_heads=self_attention.num_attention_heads,
        )
        new_self_attention._fallback_torch_layer = self_attention
        new_self_attention.query = TTNNLinear.from_torch(self_attention.query)
        new_self_attention.key = TTNNLinear.from_torch(self_attention.key)
        new_self_attention.value = TTNNLinear.from_torch(self_attention.value)
        return new_self_attention

    def preprocess_weights_impl(self):
        """Preprocess attention weights for TTNN."""
        assert (
            self.torch_layer is not None
        ), "torch_layer must be set for TTNNViTSelfAttention. As bias config for query, key, value layers is needed."
        num_heads = self.num_attention_heads
        hidden_size = self.hidden_size
        head_size = hidden_size // num_heads
        hidden_size = num_heads * head_size * 3
        qkv_weight = torch.cat(
            [
                self.query.torch_layer.weight.reshape([num_heads, head_size, -1]),
                self.key.torch_layer.weight.reshape([num_heads, head_size, -1]),
                self.value.torch_layer.weight.reshape([num_heads, head_size, -1]),
            ],
            dim=1,
        ).reshape([hidden_size, -1])
        qkv_bias = torch.cat(
            [
                self.query.torch_layer.bias.reshape([num_heads, head_size]),
                self.key.torch_layer.bias.reshape([num_heads, head_size]),
                self.value.torch_layer.bias.reshape([num_heads, head_size]),
            ],
            dim=1,
        ).reshape([hidden_size])
        self.query_key_value_weight = preprocess_linear_weight(qkv_weight, dtype=ttnn.bfloat8_b)
        self.query_key_value_bias = preprocess_linear_bias(qkv_bias, dtype=ttnn.bfloat8_b)

    def move_weights_to_device_impl(self):
        """Move attention weights to TTNN device."""
        assert self.device is not None, "Device must be set before moving weights to device."
        self.query_key_value_weight = ttnn.to_device(self.query_key_value_weight, self.device)
        self.query_key_value_bias = ttnn.to_device(self.query_key_value_bias, self.device)

    def forward(self, hidden_states, head_mask, output_attentions: bool = False):
        """Forward pass through ViT self-attention."""
        assert head_mask is None, "head_mask is not supported in TTNNViTSelfAttention"
        assert not output_attentions, "output_attentions is not supported in TTNNViTSelfAttention"
        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        hidden_states = ttnn.unsqueeze(hidden_states, 0)

        query_key_value = ttnn.linear(
            hidden_states,
            self.query_key_value_weight,
            bias=self.query_key_value_bias,
            dtype=ttnn.bfloat8_b,
        )
        query_key_value = ttnn.to_memory_config(query_key_value, ttnn.L1_MEMORY_CONFIG)

        query, key, value = ttnn.experimental.nlp_create_qkv_heads(
            query_key_value,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            num_heads=self.num_attention_heads,
            num_kv_heads=self.num_attention_heads,
            transpose_k_heads=False,
        )

        ttnn.deallocate(query_key_value)
        ttnn.deallocate(hidden_states)
        if self.should_reallocate_in_attention:
            value = ttnn.reallocate(value)

        # SDPA code
        query = ttnn.to_memory_config(query, ttnn.DRAM_MEMORY_CONFIG)
        key = ttnn.to_memory_config(key, ttnn.DRAM_MEMORY_CONFIG)
        value = ttnn.to_memory_config(value, ttnn.DRAM_MEMORY_CONFIG)

        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(self.core_grid.x, self.core_grid.y),
            q_chunk_size=256,
            k_chunk_size=256,
            exp_approx_mode=False,  # NOTE: False is more correct
        )

        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        context_layer = ttnn.transformer.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=False,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        context_layer = ttnn.transformer.concatenate_heads(
            context_layer,
        )
        return (context_layer,)
