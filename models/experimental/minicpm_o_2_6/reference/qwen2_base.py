#!/usr/bin/env python3
"""
Qwen2.5 Base Language Model Reference Implementation for MiniCPM-o-2_6

Base Qwen2.5 model (without cross-attention) matching MiniCPM-o specifications:
- Hidden size: 3584
- Num layers: 28
- Num attention heads: 28
- Num key-value heads: 4
- Vocab size: 151700

This provides the foundation language model before adding multimodal capabilities.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List, Dict
import math


class Qwen2Config:
    """Configuration for Qwen2.5 model matching MiniCPM-o"""

    def __init__(
        self,
        vocab_size: int = 151700,
        hidden_size: int = 3584,
        intermediate_size: int = 18944,  # 3584 * 5.333 ≈ 18944
        num_hidden_layers: int = 28,
        num_attention_heads: int = 28,
        num_key_value_heads: int = 4,
        max_position_embeddings: int = 32768,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1000000.0,
        attention_dropout: float = 0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout


class Qwen2RMSNorm(nn.Module):
    """RMS normalization used in Qwen2.5"""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary positional embedding to tensor"""
    return (tensor * cos) + (rotate_half(tensor) * sin)


class Qwen2RotaryEmbedding(nn.Module):
    """Rotary positional embeddings for Qwen2.5"""

    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Create inv_freq - inverse frequencies for rotary embeddings
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class Qwen2Attention(nn.Module):
    """Multi-head attention for Qwen2.5"""

    def __init__(self, config: Qwen2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = Qwen2RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, cos, sin), apply_rotary_pos_emb(
            key_states, cos, sin
        )

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if num_key_value_heads < num_heads
        key_states = torch.repeat_interleave(key_states, self.num_key_value_groups, dim=1)
        value_states = torch.repeat_interleave(value_states, self.num_key_value_groups, dim=1)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Qwen2MLP(nn.Module):
    """MLP for Qwen2.5"""

    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen2DecoderLayer(nn.Module):
    """Single decoder layer for Qwen2.5"""

    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen2Attention(config, layer_idx=layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = None,
        use_cache: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class Qwen2Model(nn.Module):
    """Base Qwen2.5 model without language modeling head"""

    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.padding_idx = config.vocab_size - 1
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, dict]:
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        use_cache = use_cache if use_cache is not None else False
        return_dict = return_dict if return_dict is not None else True

        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_key_values_length = 0

        if use_cache:
            if past_key_values is None:
                past_key_values = [None] * len(self.layers)
            else:
                past_key_values_length = past_key_values[0][0].shape[2]

        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length,
                past_key_values_length + inputs_embeds.shape[1],
                dtype=torch.long,
                device=inputs_embeds.device,
            )
            position_ids = position_ids.unsqueeze(0)

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (inputs_embeds.shape[0], inputs_embeds.shape[1] + past_key_values_length),
                dtype=torch.bool,
                device=inputs_embeds.device,
            )
        else:
            attention_mask = attention_mask.bool()

        # 4d mask is passed through the layers
        attention_mask = attention_mask[:, None, None, :]
        attention_mask = attention_mask.to(dtype=self.embed_tokens.weight.dtype)  # fp16 compatibility
        attention_mask = (1.0 - attention_mask) * torch.finfo(self.embed_tokens.weight.dtype).min

        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    decoder_layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return {
            "last_hidden_state": hidden_states,
            "past_key_values": next_cache,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attns,
        }

    def load_weights(self, weights_dict: Dict[str, torch.Tensor]) -> None:
        """Load weights from dictionary into PyTorch model parameters.

        Maps from weight generator keys to PyTorch model parameter keys.
        Keys follow model_key_mapping.txt structure with 'llm.' prefix.
        """
        weight_mapping = {
            # Embeddings
            "llm.model.embed_tokens.weight": "embed_tokens.weight",
            # Layers
            **{
                f"llm.model.layers.{i}.input_layernorm.weight": f"layers.{i}.input_layernorm.weight"
                for i in range(self.config.num_hidden_layers)
            },
            **{
                f"llm.model.layers.{i}.post_attention_layernorm.weight": f"layers.{i}.post_attention_layernorm.weight"
                for i in range(self.config.num_hidden_layers)
            },
            # Attention weights (Qwen2.5 uses q_proj, k_proj, v_proj, o_proj)
            **{
                f"llm.model.layers.{i}.self_attn.q_proj.weight": f"layers.{i}.self_attn.q_proj.weight"
                for i in range(self.config.num_hidden_layers)
            },
            **{
                f"llm.model.layers.{i}.self_attn.q_proj.bias": f"layers.{i}.self_attn.q_proj.bias"
                for i in range(self.config.num_hidden_layers)
            },
            **{
                f"llm.model.layers.{i}.self_attn.k_proj.weight": f"layers.{i}.self_attn.k_proj.weight"
                for i in range(self.config.num_hidden_layers)
            },
            **{
                f"llm.model.layers.{i}.self_attn.k_proj.bias": f"layers.{i}.self_attn.k_proj.bias"
                for i in range(self.config.num_hidden_layers)
            },
            **{
                f"llm.model.layers.{i}.self_attn.v_proj.weight": f"layers.{i}.self_attn.v_proj.weight"
                for i in range(self.config.num_hidden_layers)
            },
            **{
                f"llm.model.layers.{i}.self_attn.v_proj.bias": f"layers.{i}.self_attn.v_proj.bias"
                for i in range(self.config.num_hidden_layers)
            },
            **{
                f"llm.model.layers.{i}.self_attn.o_proj.weight": f"layers.{i}.self_attn.o_proj.weight"
                for i in range(self.config.num_hidden_layers)
            },
            # FFN weights (Qwen2.5 uses gate_proj, up_proj, down_proj)
            **{
                f"llm.model.layers.{i}.mlp.gate_proj.weight": f"layers.{i}.mlp.gate_proj.weight"
                for i in range(self.config.num_hidden_layers)
            },
            **{
                f"llm.model.layers.{i}.mlp.up_proj.weight": f"layers.{i}.mlp.up_proj.weight"
                for i in range(self.config.num_hidden_layers)
            },
            **{
                f"llm.model.layers.{i}.mlp.down_proj.weight": f"layers.{i}.mlp.down_proj.weight"
                for i in range(self.config.num_hidden_layers)
            },
            # Final layer norm
            "llm.model.norm.weight": "norm.weight",
        }

        loaded_count = 0
        for gen_key, model_key in weight_mapping.items():
            if gen_key in weights_dict:
                try:
                    param = dict(self.named_parameters())[model_key]
                    param.data.copy_(weights_dict[gen_key])
                    loaded_count += 1
                except KeyError as e:
                    print(f"Warning: Parameter {model_key} not found in model (bias parameter may not exist)")
                except Exception as e:
                    print(f"Error loading {model_key}: {e}")

        print(f"Loaded {loaded_count} weights into PyTorch Qwen2Model")


class Qwen2ForCausalLM(nn.Module):
    """Complete Qwen2.5 model with language modeling head"""

    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, dict]:
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        return_dict = return_dict if return_dict is not None else True

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0] if not return_dict else outputs["last_hidden_state"]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": outputs.get("past_key_values"),
            "hidden_states": outputs.get("hidden_states"),
            "attentions": outputs.get("attentions"),
        }


def create_minicpm_qwen_base(config: Qwen2Config = None) -> Qwen2ForCausalLM:
    """Create base Qwen2.5 model matching MiniCPM-o specifications"""
    if config is None:
        config = Qwen2Config()  # Default MiniCPM-o config

    return Qwen2ForCausalLM(config)


# Test function
def test_qwen_base():
    """Test base Qwen2.5 model"""
    print("🤖 Testing Base Qwen2.5 Model...")

    config = Qwen2Config()
    model = Qwen2ForCausalLM(config)

    # Test input
    batch_size, seq_len = 1, 10
    input_ids = torch.randint(0, min(config.vocab_size, 1000), (batch_size, seq_len))

    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    print(f"✅ Input shape: {input_ids.shape}")
    print(f"✅ Logits shape: {outputs['logits'].shape}")
    print(f"   Expected: ({batch_size}, {seq_len}, {config.vocab_size})")

    expected_shape = (batch_size, seq_len, config.vocab_size)
    assert outputs["logits"].shape == expected_shape, f"Shape mismatch: {outputs['logits'].shape} vs {expected_shape}"

    print("✅ Base Qwen2.5 model test passed!")
    return True


if __name__ == "__main__":
    test_qwen_base()
