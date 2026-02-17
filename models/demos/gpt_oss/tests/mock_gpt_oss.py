"""
Mock GPT-OSS model implementation for testing purposes.

Since the transformers.models.gpt_oss module doesn't exist in the standard
transformers library and no custom modeling file is available, this mock
provides the minimal interface needed for running tests.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GptOssRotaryEmbedding(nn.Module):
    """Mock RoPE implementation for GPT-OSS."""

    def __init__(self, config):
        super().__init__()
        self.dim = config.head_dim
        self.max_position_embeddings = getattr(config, "max_position_embeddings", 131072)
        self.rope_theta = getattr(config, "rope_theta", 150000.0)

        # Precompute frequencies
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids):
        """Apply rotary embeddings.

        Args:
            x: Input tensor (not actually used, just for shape)
            position_ids: Position IDs [batch_size, seq_len] or [batch_size]

        Returns:
            cos, sin: Rotation matrices
        """
        # Handle different position_ids shapes
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)

        batch_size = position_ids.shape[0]
        seq_len = position_ids.shape[-1]

        # Create position embeddings
        position = position_ids.float()
        freqs = torch.einsum("bi,j->bij", position, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        cos = emb.cos()
        sin = emb.sin()

        # Expand to match expected shape [1, batch, seq_len, head_dim]
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

        return cos, sin


class GptOssRMSNorm(nn.Module):
    """Mock RMS Normalization for GPT-OSS."""

    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class GptOssAttention(nn.Module):
    """Mock attention layer for GPT-OSS."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim if hasattr(config, "head_dim") else self.hidden_size // self.num_heads
        self.num_key_value_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=True)

        # GPT-OSS specific: attention sinks
        self.sinks = nn.Parameter(torch.randn(self.num_heads))

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, **kwargs):
        batch_size, seq_len, _ = hidden_states.size()

        # QKV projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for attention
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Repeat KV heads if necessary
        if self.num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.num_heads * self.head_dim)

        # Output projection
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


class GptOssMLP(nn.Module):
    """Mock MLP layer for GPT-OSS."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = getattr(config, "intermediate_size", 2880)

        self.gate_up_proj = nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=True)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        # Fused gate and up projection
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)

        # Apply SwiGLU activation
        intermediate = self.act_fn(gate) * up

        # Down projection
        output = self.down_proj(intermediate)
        return output


class GptOssTopKRouter(nn.Module):
    """Mock TopK router for GPT-OSS MoE."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = getattr(config, "num_local_experts", 128)
        self.num_experts_per_tok = getattr(config, "num_experts_per_tok", 4)
        self.hidden_size = config.hidden_size

        # Create a Linear layer for computation with bias
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=True)
        # Register the weight and bias parameters to match TopKRouter expectations
        # Note: This creates references, not copies
        self.register_parameter("weight", self.gate.weight)
        self.register_parameter("bias", self.gate.bias)

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # Compute router logits
        router_logits = self.gate(hidden_states_flat)

        # Get top-k experts
        routing_weights, selected_experts = torch.topk(router_logits, self.num_experts_per_tok, dim=-1)

        # Normalize routing weights
        routing_weights = F.softmax(routing_weights, dim=-1)

        # Reshape back to 3D
        routing_weights = routing_weights.view(batch_size, seq_len, self.num_experts_per_tok)
        selected_experts = selected_experts.view(batch_size, seq_len, self.num_experts_per_tok)

        return routing_weights, selected_experts, router_logits


class GptOssSparseMoeBlock(nn.Module):
    """Mock Sparse MoE block for GPT-OSS."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts = getattr(config, "num_local_experts", 128)
        self.num_experts_per_tok = getattr(config, "num_experts_per_tok", 4)
        self.hidden_size = config.hidden_size
        self.intermediate_size = getattr(config, "intermediate_size", 2880)

        # Router
        self.router = GptOssTopKRouter(config)

        # Create experts with combined weights to match expected format
        # Instead of ModuleList, we'll use parameters that match the expected format
        # This matches what load_throughput_expert_weights expects
        self.experts = nn.Module()

        # Create combined expert weights (all experts in one tensor)
        # Shape: [num_experts, in_features, out_features]
        self.experts.gate_up_proj = nn.Parameter(
            torch.randn(self.num_experts, self.hidden_size, 2 * self.intermediate_size)
        )
        self.experts.gate_up_proj_bias = nn.Parameter(torch.randn(self.num_experts, 2 * self.intermediate_size))
        self.experts.down_proj = nn.Parameter(torch.randn(self.num_experts, self.intermediate_size, self.hidden_size))
        self.experts.down_proj_bias = nn.Parameter(torch.randn(self.num_experts, self.hidden_size))

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        orig_shape = hidden_states.shape

        # Route tokens to experts (router expects 3D input)
        routing_weights, selected_experts, router_logits = self.router(hidden_states)

        # Now reshape for expert processing
        hidden_states = hidden_states.view(-1, hidden_dim)
        routing_weights = routing_weights.view(-1, self.num_experts_per_tok)
        selected_experts = selected_experts.view(-1, self.num_experts_per_tok)

        # Simplified expert computation (not distributed)
        final_hidden_states = torch.zeros_like(hidden_states)

        # Process each token
        for token_idx in range(hidden_states.shape[0]):
            token_hidden = hidden_states[token_idx : token_idx + 1]

            for expert_idx in range(self.num_experts_per_tok):
                expert_id = selected_experts[token_idx, expert_idx].item()
                expert_weight = routing_weights[token_idx, expert_idx]

                # Apply expert using the combined weights
                # Extract the specific expert's weights
                gate_up_weight = self.experts.gate_up_proj[expert_id]
                gate_up_bias = self.experts.gate_up_proj_bias[expert_id]
                down_weight = self.experts.down_proj[expert_id]
                down_bias = self.experts.down_proj_bias[expert_id]

                # Apply gate_up projection
                gate_up = F.linear(token_hidden, gate_up_weight.t(), gate_up_bias)
                gate, up = gate_up.chunk(2, dim=-1)

                # SwiGLU activation
                intermediate = F.silu(gate) * up

                # Down projection
                expert_output = F.linear(intermediate, down_weight.t(), down_bias)

                final_hidden_states[token_idx : token_idx + 1] += expert_weight * expert_output

        return final_hidden_states.view(orig_shape), router_logits


class GptOssDecoderLayer(nn.Module):
    """Mock decoder layer for GPT-OSS."""

    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Determine attention type from layer_types if available
        layer_types = getattr(config, "layer_types", None)
        if layer_types and layer_idx < len(layer_types):
            self.attention_type = layer_types[layer_idx]
        else:
            self.attention_type = "full_attention"

        # Layer components
        self.self_attn = GptOssAttention(config, layer_idx)
        self.input_layernorm = GptOssRMSNorm(config.hidden_size)
        self.post_attention_layernorm = GptOssRMSNorm(config.hidden_size)

        # MLP or MoE
        if getattr(config, "num_local_experts", 0) > 0:
            self.mlp = GptOssSparseMoeBlock(config)
        else:
            self.mlp = GptOssMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ):
        """
        Forward pass for the decoder layer.
        """
        residual = hidden_states

        # Layer norm before attention
        hidden_states = self.input_layernorm(hidden_states)

        # Self attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        # Add residual
        hidden_states = residual + hidden_states

        # Layer norm before MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # MLP or MoE
        if isinstance(self.mlp, GptOssSparseMoeBlock):
            hidden_states, router_logits = self.mlp(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
            router_logits = None

        # Add residual
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits and router_logits is not None:
            outputs += (router_logits,)

        return outputs


class GptOssModel(nn.Module):
    """Mock GPT-OSS model."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GptOssDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GptOssRMSNorm(config.hidden_size)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        # Decoder layers
        for idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values[idx] if past_key_values else None,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)

        return hidden_states


class GptOssForCausalLM(nn.Module):
    """Mock GPT-OSS model for causal language modeling."""

    def __init__(self, config):
        super().__init__()
        self.model = GptOssModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # Model forward
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
        )

        hidden_states = outputs
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return (loss, logits) if loss is not None else logits
