#!/usr/bin/env python3
"""
PyTorch Reference Implementation for MiniCPM-o-2_6

This module implements the multimodal Qwen2.5 model with cross-attention layers
for fusing vision and audio modalities into the language model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Prefer importing Qwen2 classes from transformers when available, otherwise fall back to
# the local reference implementation `qwen2_base.py` included in this directory.
try:
    from transformers import Qwen2ForCausalLM, Qwen2Config  # type: ignore
except Exception:
    # Local fallback for environments where transformers doesn't expose Qwen2
    from qwen2_base import Qwen2ForCausalLM, Qwen2Config  # type: ignore

# Qwen2 helpers (attention / RMSNorm) are imported from transformers' qwen2 submodule
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm  # type: ignore
import math
from typing import Dict


class Qwen2CrossAttention(nn.Module):
    """Cross-attention layer for multimodal fusion in Qwen2.5"""

    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings

        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

        # Query projection
        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=True)

        # Key and Value projections for cross-attention (multimodal)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)

        # Output projection
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=False)

        # Normalization
        self.q_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, hidden_states, encoder_hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size] - text embeddings
            encoder_hidden_states: [batch_size, seq_len_enc, hidden_size] - vision/audio embeddings
            attention_mask: [batch_size, seq_len, seq_len_enc] - attention mask
        """
        batch_size, seq_len, _ = hidden_states.shape
        _, seq_len_enc, _ = encoder_hidden_states.shape

        # Query projection
        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)

        # Key and Value projections from encoder (multimodal)
        key_states = self.k_proj(encoder_hidden_states)
        value_states = self.v_proj(encoder_hidden_states)

        key_states = key_states.view(batch_size, seq_len_enc, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len_enc, self.num_key_value_heads, self.head_dim).transpose(
            1, 2
        )

        # Repeat K,V for grouped attention
        if self.num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        # Apply normalization
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # Scaled dot-product attention
        scale_factor = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scale_factor

        if attention_mask is not None:
            # attention_mask shape: [batch_size, seq_len, seq_len_enc]
            # Expand to: [batch_size, num_heads, seq_len, seq_len_enc]
            attention_mask = attention_mask.unsqueeze(1).expand(-1, self.num_attention_heads, -1, -1)
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output


class MultimodalQwen2DecoderLayer(nn.Module):
    """Qwen2.5 decoder layer extended with cross-attention for multimodal inputs"""

    def __init__(self, config: Qwen2Config, layer_idx: int, has_cross_attention: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size

        # Self-attention (text-only) - use transformers implementation
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

        self.self_attn = Qwen2Attention(config, layer_idx)

        # MLP (reuse from Qwen2.5)
        self.mlp = Qwen2DecoderLayer(config, layer_idx).mlp

        # Layer norms (reuse from Qwen2.5)
        self.input_layernorm = Qwen2DecoderLayer(config, layer_idx).input_layernorm
        self.post_attention_layernorm = Qwen2DecoderLayer(config, layer_idx).post_attention_layernorm

        # Cross-attention for multimodal (new)
        self.has_cross_attention = has_cross_attention
        if has_cross_attention:
            self.cross_attn = Qwen2CrossAttention(config)
            self.cross_attn_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        **kwargs,
    ):
        # Self-attention (text)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

        # Handle present_key_value for caching
        present_key_value = None
        hidden_states = residual + hidden_states

        # Cross-attention (multimodal fusion)
        if self.has_cross_attention and encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.cross_attn_layernorm(hidden_states)
            hidden_states = self.cross_attn(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            hidden_states = residual + hidden_states

        # MLP
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


class MultimodalQwen2Config(Qwen2Config):
    """Configuration for Multimodal Qwen2.5 with cross-attention layers"""

    def __init__(
        self,
        cross_attention_layers=None,  # List of layer indices that have cross-attention
        vision_hidden_size=1152,  # SigLip hidden size
        audio_hidden_size=1024,  # Whisper hidden size
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cross_attention_layers = cross_attention_layers or []
        self.vision_hidden_size = vision_hidden_size
        self.audio_hidden_size = audio_hidden_size


class MultimodalQwen2Model(nn.Module):
    """Multimodal Qwen2.5 model with cross-attention for vision/audio fusion"""

    def __init__(self, config: MultimodalQwen2Config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Embeddings (reuse from Qwen2.5)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        # Rotary position embeddings
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

        self.rotary_emb = Qwen2RotaryEmbedding(config=config)

        # Layers with cross-attention where specified
        self.layers = nn.ModuleList()
        for layer_idx in range(config.num_hidden_layers):
            has_cross_attn = layer_idx in config.cross_attention_layers
            self.layers.append(MultimodalQwen2DecoderLayer(config, layer_idx, has_cross_attn))

        # Final layer norm
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights following Qwen2.5 pattern"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        vision_embeds=None,
        audio_embeds=None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        # Embed inputs
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Prepare multimodal embeddings for cross-attention
        encoder_hidden_states = None
        encoder_attention_mask = None

        if vision_embeds is not None or audio_embeds is not None:
            multimodal_embeds = []
            if vision_embeds is not None:
                multimodal_embeds.append(vision_embeds)
            if audio_embeds is not None:
                multimodal_embeds.append(audio_embeds)

            # Concatenate vision and audio embeddings
            encoder_hidden_states = torch.cat(multimodal_embeds, dim=1)

            # Create attention mask (all visible)
            if encoder_hidden_states is not None:
                batch_size, seq_len_enc, _ = encoder_hidden_states.shape
                encoder_attention_mask = torch.zeros(
                    (batch_size, seq_length, seq_len_enc), dtype=torch.float32, device=encoder_hidden_states.device
                )

        # Forward through layers
        hidden_states = inputs_embeds

        # Compute rotary position embeddings
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        all_self_attns = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        next_decoder_cache = () if use_cache else None

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values[layer_idx] if past_key_values is not None else None,
                output_attentions=output_attentions,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                # present_key_value is at index 2 if output_attentions else 1
                kv_index = 2 if output_attentions else 1
                if kv_index < len(layer_outputs):
                    next_decoder_cache += (layer_outputs[kv_index],)

            if output_attentions:
                # self_attn_weights is always at index 1 when output_attentions=True
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # Add hidden states from the last decoder layer
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
            # Self-attention weights
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
            # FFN weights
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
            # Cross-attention weights (if layer has cross-attention)
            **{
                f"llm.model.layers.{i}.cross_attn_layernorm.weight": f"layers.{i}.cross_attn_layernorm.weight"
                for i in self.config.cross_attention_layers
            },
            **{
                f"llm.model.layers.{i}.cross_attn.q_proj.weight": f"layers.{i}.cross_attn.q_proj.weight"
                for i in self.config.cross_attention_layers
            },
            **{
                f"llm.model.layers.{i}.cross_attn.q_proj.bias": f"layers.{i}.cross_attn.q_proj.bias"
                for i in self.config.cross_attention_layers
            },
            **{
                f"llm.model.layers.{i}.cross_attn.k_proj.weight": f"layers.{i}.cross_attn.k_proj.weight"
                for i in self.config.cross_attention_layers
            },
            **{
                f"llm.model.layers.{i}.cross_attn.k_proj.bias": f"layers.{i}.cross_attn.k_proj.bias"
                for i in self.config.cross_attention_layers
            },
            **{
                f"llm.model.layers.{i}.cross_attn.v_proj.weight": f"layers.{i}.cross_attn.v_proj.weight"
                for i in self.config.cross_attention_layers
            },
            **{
                f"llm.model.layers.{i}.cross_attn.v_proj.bias": f"layers.{i}.cross_attn.v_proj.bias"
                for i in self.config.cross_attention_layers
            },
            **{
                f"llm.model.layers.{i}.cross_attn.o_proj.weight": f"layers.{i}.cross_attn.o_proj.weight"
                for i in self.config.cross_attention_layers
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
                except Exception as e:
                    print(f"Failed to load {gen_key} -> {model_key}: {e}")
            else:
                print(f"Warning: {gen_key} not found in weights_dict")

        print(f"Loaded {loaded_count}/{len(weight_mapping)} weights into MultimodalQwen2Model")


class MultimodalQwen2ForCausalLM(nn.Module):
    """Multimodal Qwen2.5 model for causal language modeling"""

    def __init__(self, config: MultimodalQwen2Config):
        super().__init__()
        self.config = config
        self.model = MultimodalQwen2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights following Qwen2.5 pattern"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        """

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        vision_embeds=None,
        audio_embeds=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            vision_embeds=vision_embeds,
            audio_embeds=audio_embeds,
        )

        hidden_states = outputs["last_hidden_state"] if return_dict else outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device).bool()]
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device).bool()]
            else:
                shift_logits = logits[..., :-1, :]
                shift_labels = labels[..., 1:]

            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return {
            "loss": loss,
            "logits": logits,
            "last_hidden_state": hidden_states,  # Add this for TTS decoder
            "past_key_values": outputs.get("past_key_values"),
            "hidden_states": outputs.get("hidden_states"),
            "attentions": outputs.get("attentions"),
        }

    @classmethod
    def from_qwen2_checkpoint(cls, qwen2_model_path, cross_attention_layers=None):
        # Load base Qwen2.5 config
        from transformers import Qwen2Config
        base_config = Qwen2Config.from_pretrained(qwen2_model_path)

        # Create multimodal config
        multimodal_config = MultimodalQwen2Config(
            vocab_size=base_config.vocab_size,
            hidden_size=base_config.hidden_size,
            intermediate_size=base_config.intermediate_size,
            num_hidden_layers=base_config.num_hidden_layers,
            num_attention_heads=base_config.num_attention_heads,
            num_key_value_heads=base_config.num_key_value_heads,
            max_position_embeddings=base_config.max_position_embeddings,
            rms_norm_eps=base_config.rms_norm_eps,
            initializer_range=base_config.initializer_range,
            use_cache=base_config.use_cache,
            tie_word_embeddings=base_config.tie_word_embeddings,
            rope_theta=base_config.rope_theta,
            cross_attention_layers=cross_attention_layers or [8, 16, 24],  # Default layers with cross-attention
        )

        # Create model
        model = cls(multimodal_config)

        # Load weights from base Qwen2.5 model
        base_model = Qwen2ForCausalLM.from_pretrained(qwen2_model_path)

        # Copy compatible weights
        model.model.embed_tokens.load_state_dict(base_model.model.embed_tokens.state_dict())
        model.lm_head.load_state_dict(base_model.lm_head.state_dict())

        # Copy layer weights (excluding cross-attention parts)
        for i, layer in enumerate(model.model.layers):
            base_layer = base_model.model.layers[i]
            layer.self_attn.load_state_dict(base_layer.self_attn.state_dict())
            layer.mlp.load_state_dict(base_layer.mlp.state_dict())
            layer.input_layernorm.load_state_dict(base_layer.input_layernorm.state_dict())
            layer.post_attention_layernorm.load_state_dict(base_layer.post_attention_layernorm.state_dict())

        model.model.norm.load_state_dict(base_model.model.norm.state_dict())

        return model


PyTorch Reference Implementation for MiniCPM-o-2_6

This module implements the multimodal Qwen2.5 model with cross-attention layers
for fusing vision and audio modalities into the language model.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

# Prefer importing Qwen2 classes from transformers when available, otherwise fall back to
# the local reference implementation `qwen2_base.py` included in this directory.
try:
    from transformers import Qwen2ForCausalLM, Qwen2Config  # type: ignore
except Exception:
    # Local fallback for environments where transformers doesn't expose Qwen2
    from qwen2_base import Qwen2ForCausalLM, Qwen2Config  # type: ignore

# Qwen2 helpers (attention / RMSNorm) are imported from transformers' qwen2 submodule
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm  # type: ignore
import math
from typing import Dict


class Qwen2CrossAttention(nn.Module):
    """Cross-attention layer for multimodal fusion in Qwen2.5"""

    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings

        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

        # Query projection
        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=True)

        # Key and Value projections for cross-attention (multimodal)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)

        # Output projection
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=False)

        # Normalization
        self.q_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, hidden_states, encoder_hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size] - text embeddings
            encoder_hidden_states: [batch_size, seq_len_enc, hidden_size] - vision/audio embeddings
            attention_mask: [batch_size, seq_len, seq_len_enc] - attention mask
        """
        batch_size, seq_len, _ = hidden_states.shape
        _, seq_len_enc, _ = encoder_hidden_states.shape

        # Query projection
        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)

        # Key and Value projections from encoder (multimodal)
        key_states = self.k_proj(encoder_hidden_states)
        value_states = self.v_proj(encoder_hidden_states)

        key_states = key_states.view(batch_size, seq_len_enc, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len_enc, self.num_key_value_heads, self.head_dim).transpose(
            1, 2
        )

        # Repeat K,V for grouped attention
        if self.num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        # Apply normalization
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # Scaled dot-product attention
        scale_factor = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scale_factor

        if attention_mask is not None:
            # attention_mask shape: [batch_size, seq_len, seq_len_enc]
            # Expand to: [batch_size, num_heads, seq_len, seq_len_enc]
            attention_mask = attention_mask.unsqueeze(1).expand(-1, self.num_attention_heads, -1, -1)
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output


class MultimodalQwen2DecoderLayer(nn.Module):
    """Qwen2.5 decoder layer extended with cross-attention for multimodal inputs"""

    def __init__(self, config: Qwen2Config, layer_idx: int, has_cross_attention: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size

        # Self-attention (text-only) - use transformers implementation
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

        self.self_attn = Qwen2Attention(config, layer_idx)

        # MLP (reuse from Qwen2.5)
        self.mlp = Qwen2DecoderLayer(config, layer_idx).mlp

        # Layer norms (reuse from Qwen2.5)
        self.input_layernorm = Qwen2DecoderLayer(config, layer_idx).input_layernorm
        self.post_attention_layernorm = Qwen2DecoderLayer(config, layer_idx).post_attention_layernorm

        # Cross-attention for multimodal (new)
        self.has_cross_attention = has_cross_attention
        if has_cross_attention:
            self.cross_attn = Qwen2CrossAttention(config)
            self.cross_attn_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        **kwargs,
    ):
        # Self-attention (text)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

        # Handle present_key_value for caching
        present_key_value = None
        hidden_states = residual + hidden_states

        # Cross-attention (multimodal fusion)
        if self.has_cross_attention and encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.cross_attn_layernorm(hidden_states)
            hidden_states = self.cross_attn(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            hidden_states = residual + hidden_states

        # MLP
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


class MultimodalQwen2Config(Qwen2Config):
    """Configuration for Multimodal Qwen2.5 with cross-attention layers"""

    def __init__(
        self,
        cross_attention_layers=None,  # List of layer indices that have cross-attention
        vision_hidden_size=1152,  # SigLip hidden size
        audio_hidden_size=1024,  # Whisper hidden size
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cross_attention_layers = cross_attention_layers or []
        self.vision_hidden_size = vision_hidden_size
        self.audio_hidden_size = audio_hidden_size


class MultimodalQwen2Model(nn.Module):
    """Multimodal Qwen2.5 model with cross-attention for vision/audio fusion"""

    def __init__(self, config: MultimodalQwen2Config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Embeddings (reuse from Qwen2.5)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        # Rotary position embeddings
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

        self.rotary_emb = Qwen2RotaryEmbedding(config=config)

        # Layers with cross-attention where specified
        self.layers = nn.ModuleList()
        for layer_idx in range(config.num_hidden_layers):
            has_cross_attn = layer_idx in config.cross_attention_layers
            self.layers.append(MultimodalQwen2DecoderLayer(config, layer_idx, has_cross_attn))

        # Final layer norm
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights following Qwen2.5 pattern"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        vision_embeds=None,
        audio_embeds=None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        # Embed inputs
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Prepare multimodal embeddings for cross-attention
        encoder_hidden_states = None
        encoder_attention_mask = None

        if vision_embeds is not None or audio_embeds is not None:
            multimodal_embeds = []
            if vision_embeds is not None:
                multimodal_embeds.append(vision_embeds)
            if audio_embeds is not None:
                multimodal_embeds.append(audio_embeds)

            # Concatenate vision and audio embeddings
            encoder_hidden_states = torch.cat(multimodal_embeds, dim=1)

            # Create attention mask (all visible)
            if encoder_hidden_states is not None:
                batch_size, seq_len_enc, _ = encoder_hidden_states.shape
                encoder_attention_mask = torch.zeros(
                    (batch_size, seq_length, seq_len_enc), dtype=torch.float32, device=encoder_hidden_states.device
                )

        # Forward through layers
        hidden_states = inputs_embeds

        # Compute rotary position embeddings
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        all_self_attns = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        next_decoder_cache = () if use_cache else None

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values[layer_idx] if past_key_values is not None else None,
                output_attentions=output_attentions,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                # present_key_value is at index 2 if output_attentions else 1
                kv_index = 2 if output_attentions else 1
                if kv_index < len(layer_outputs):
                    next_decoder_cache += (layer_outputs[kv_index],)

            if output_attentions:
                # self_attn_weights is always at index 1 when output_attentions=True
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # Add hidden states from the last decoder layer
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
            # Self-attention weights
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
            # FFN weights
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
            # Cross-attention weights (if layer has cross-attention)
            **{
                f"llm.model.layers.{i}.cross_attn_layernorm.weight": f"layers.{i}.cross_attn_layernorm.weight"
                for i in self.config.cross_attention_layers
            },
            **{
                f"llm.model.layers.{i}.cross_attn.q_proj.weight": f"layers.{i}.cross_attn.q_proj.weight"
                for i in self.config.cross_attention_layers
            },
            **{
                f"llm.model.layers.{i}.cross_attn.q_proj.bias": f"layers.{i}.cross_attn.q_proj.bias"
                for i in self.config.cross_attention_layers
            },
            **{
                f"llm.model.layers.{i}.cross_attn.k_proj.weight": f"layers.{i}.cross_attn.k_proj.weight"
                for i in self.config.cross_attention_layers
            },
            **{
                f"llm.model.layers.{i}.cross_attn.k_proj.bias": f"layers.{i}.cross_attn.k_proj.bias"
                for i in self.config.cross_attention_layers
            },
            **{
                f"llm.model.layers.{i}.cross_attn.v_proj.weight": f"layers.{i}.cross_attn.v_proj.weight"
                for i in self.config.cross_attention_layers
            },
            **{
                f"llm.model.layers.{i}.cross_attn.v_proj.bias": f"layers.{i}.cross_attn.v_proj.bias"
                for i in self.config.cross_attention_layers
            },
            **{
                f"llm.model.layers.{i}.cross_attn.o_proj.weight": f"layers.{i}.cross_attn.o_proj.weight"
                for i in self.config.cross_attention_layers
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
                except Exception as e:
                    print(f"Failed to load {gen_key} -> {model_key}: {e}")
            else:
                print(f"Warning: {gen_key} not found in weights_dict")

        print(f"Loaded {loaded_count}/{len(weight_mapping)} weights into MultimodalQwen2Model")


class MultimodalQwen2ForCausalLM(nn.Module):
    """Multimodal Qwen2.5 model for causal language modeling"""

    def __init__(self, config: MultimodalQwen2Config):
        super().__init__()
        self.config = config
        self.model = MultimodalQwen2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights following Qwen2.5 pattern"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        vision_embeds=None,
        audio_embeds=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            vision_embeds=vision_embeds,
            audio_embeds=audio_embeds,
        )

        hidden_states = outputs["last_hidden_state"] if return_dict else outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device).bool()]
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device).bool()]
            else:
                shift_logits = logits[..., :-1, :]
                shift_labels = labels[..., 1:]

            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return {
            "loss": loss,
            "logits": logits,
            "last_hidden_state": hidden_states,  # Add this for TTS decoder
            "past_key_values": outputs.get("past_key_values"),
            "hidden_states": outputs.get("hidden_states"),
            "attentions": outputs.get("attentions"),
        }

    @classmethod
    def from_qwen2_checkpoint(cls, qwen2_model_path, cross_attention_layers=None):
        # Load base Qwen2.5 config
        from transformers import Qwen2Config

        base_config = Qwen2Config.from_pretrained(qwen2_model_path)

        # Create multimodal config
        multimodal_config = MultimodalQwen2Config(
            vocab_size=base_config.vocab_size,
            hidden_size=base_config.hidden_size,
            intermediate_size=base_config.intermediate_size,
            num_hidden_layers=base_config.num_hidden_layers,
            num_attention_heads=base_config.num_attention_heads,
            num_key_value_heads=base_config.num_key_value_heads,
            max_position_embeddings=base_config.max_position_embeddings,
            rms_norm_eps=base_config.rms_norm_eps,
            initializer_range=base_config.initializer_range,
            use_cache=base_config.use_cache,
            tie_word_embeddings=base_config.tie_word_embeddings,
            rope_theta=base_config.rope_theta,
            cross_attention_layers=cross_attention_layers or [8, 16, 24],  # Default layers with cross-attention
        )

        # Create model
        model = cls(multimodal_config)

        # Load weights from base Qwen2.5 model
        base_model = Qwen2ForCausalLM.from_pretrained(qwen2_model_path)

        # Copy compatible weights
        model.model.embed_tokens.load_state_dict(base_model.model.embed_tokens.state_dict())
        model.lm_head.load_state_dict(base_model.lm_head.state_dict())

        # Copy layer weights (excluding cross-attention parts)
        for i, layer in enumerate(model.model.layers):
            base_layer = base_model.model.layers[i]
            layer.self_attn.load_state_dict(base_layer.self_attn.state_dict())
            layer.mlp.load_state_dict(base_layer.mlp.state_dict())
            layer.input_layernorm.load_state_dict(base_layer.input_layernorm.state_dict())
            layer.post_attention_layernorm.load_state_dict(base_layer.post_attention_layernorm.state_dict())

        model.model.norm.load_state_dict(base_model.model.norm.state_dict())

        return model
