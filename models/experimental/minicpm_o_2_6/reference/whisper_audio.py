#!/usr/bin/env python3
"""
Whisper Audio Encoder Reference Implementation for MiniCPM-o-2_6

Based on Whisper medium model used in MiniCPM-o:
- d_model: 1024
- encoder_layers: 24
- encoder_attention_heads: 16
- encoder_ffn_dim: 4096

This provides a PyTorch reference for the audio encoder component.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, Dict


class WhisperAudioConfig:
    """Configuration for Whisper audio encoder matching MiniCPM-o"""

    def __init__(
        self,
        d_model: int = 1024,
        encoder_layers: int = 24,
        encoder_attention_heads: int = 16,
        encoder_ffn_dim: int = 4096,
        max_source_positions: int = 1500,
        vocab_size: int = 51865,
        dropout: float = 0.0,
        activation_function: str = "gelu",
        layer_norm_eps: float = 1e-5,
        num_mel_bins: int = 80,  # Number of mel frequency bins
        **kwargs,
    ):
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.max_source_positions = max_source_positions
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.activation_function = activation_function
        self.layer_norm_eps = layer_norm_eps
        self.num_mel_bins = num_mel_bins


class WhisperAudioEmbeddings(nn.Module):
    """Whisper audio embeddings with convolutional positional encoding"""

    def __init__(self, config: WhisperAudioConfig):
        super().__init__()
        self.config = config

        # Convolutional positional encoding (following Whisper architecture)
        self.conv1 = nn.Conv1d(config.d_model, config.d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(config.d_model, config.d_model, kernel_size=3, stride=2, padding=1)

        # Embeddings layer
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_positions = nn.Embedding(config.max_source_positions, config.d_model)

        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            # Use pre-computed embeddings (for encoded audio features)
            hidden_states = inputs_embeds
        elif input_ids is not None:
            # Standard embedding lookup
            input_shape = input_ids.size()
            hidden_states = self.embed_tokens(input_ids)
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # Add positional embeddings if provided
        if position_ids is not None:
            position_embeddings = self.embed_positions(position_ids)
            hidden_states = hidden_states + position_embeddings

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class WhisperAttention(nn.Module):
    """Multi-head attention for Whisper audio encoder"""

    def __init__(self, config: WhisperAudioConfig, is_decoder: bool = False):
        super().__init__()
        self.config = config
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        self.dropout = config.dropout

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * (self.head_dim**-0.5)

        # get key, value proj
        if past_key_value is not None:
            saved_key, saved_value = past_key_value
            # saved states are: (bsz, num_heads, seq_len, head_dim)
        else:
            saved_key = saved_value = None

        if key_value_states is None:
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        else:
            key_states = self.k_proj(key_value_states)
            value_states = self.v_proj(key_value_states)

        # reshape
        query_states = self._shape(query_states, tgt_len, bsz)
        key_states = self._shape(key_states, -1, bsz)
        value_states = self._shape(value_states, -1, bsz)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([saved_key, key_states], dim=2)
            value_states = torch.cat([saved_value, value_states], dim=2)

        past_key_value = (key_states, value_states) if past_key_value is not None else None

        # compute attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights, past_key_value


class WhisperEncoderLayer(nn.Module):
    """Single encoder layer for Whisper audio model"""

    def __init__(self, config: WhisperAudioConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.d_model

        self.self_attn = WhisperAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.config.dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class WhisperEncoder(nn.Module):
    """Whisper encoder consisting of multiple encoder layers"""

    def __init__(self, config: WhisperAudioConfig):
        super().__init__()
        self.config = config
        self.dropout = config.dropout
        self.layerdrop = 0.0

        # Conv1D layers for mel spectrogram projection (like original Whisper)
        self.conv1 = nn.Conv1d(config.num_mel_bins, config.d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(config.d_model, config.d_model, kernel_size=3, stride=2, padding=1)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_positions = nn.Embedding(config.max_source_positions, config.d_model)

        self.layers = nn.ModuleList([WhisperEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        mel_spectrograms: Optional[torch.Tensor] = None,  # Add mel spectrograms input
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, dict]:
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        return_dict = return_dict if return_dict is not None else True

        # Handle different input types
        if input_ids is not None and inputs_embeds is not None and mel_spectrograms is not None:
            raise ValueError("You cannot specify input_ids, inputs_embeds, and mel_spectrograms at the same time")
        elif mel_spectrograms is not None:
            # Process mel spectrograms through convolutional layers
            # mel_spectrograms: [batch, time, mel_bins] -> [batch, mel_bins, time]
            mel_spectrograms = mel_spectrograms.transpose(1, 2)
            inputs_embeds = self.conv1(mel_spectrograms)
            inputs_embeds = nn.functional.gelu(inputs_embeds)
            inputs_embeds = self.conv2(inputs_embeds)
            inputs_embeds = nn.functional.gelu(inputs_embeds)
            # inputs_embeds: [batch, d_model, time] -> [batch, time, d_model]
            inputs_embeds = inputs_embeds.transpose(1, 2)
            input_shape = inputs_embeds.size()[:-1]
        elif input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids, inputs_embeds, or mel_spectrograms")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Create position IDs
        position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=inputs_embeds.device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)

        # Add positional embeddings
        position_embeddings = self.embed_positions(position_ids)
        hidden_states = inputs_embeds + position_embeddings
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        # check if we are using gradient checkpointing
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    encoder_layer,
                    hidden_states,
                    attention_mask,
                    None,  # layer_head_mask
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    layer_head_mask=None,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions += (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

        return {
            "last_hidden_state": hidden_states,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attentions,
        }


class WhisperAudioEncoder(nn.Module):
    """Complete Whisper audio encoder matching MiniCPM-o specifications"""

    def __init__(self, config: WhisperAudioConfig):
        super().__init__()
        self.config = config
        self.encoder = WhisperEncoder(config)

    def load_weights(self, weights_dict: Dict[str, torch.Tensor]) -> None:
        """
        Load weights from dictionary into PyTorch model parameters.

        Args:
            weights_dict: Dictionary containing model weights
        """
        # Mapping from generated weight keys to PyTorch parameter names
        weight_mapping = {
            # Conv layers
            "apm.conv1.weight": "encoder.conv1.weight",
            "apm.conv1.bias": "encoder.conv1.bias",
            "apm.conv2.weight": "encoder.conv2.weight",
            "apm.conv2.bias": "encoder.conv2.bias",
            # Embeddings
            "apm.embed_positions.weight": "encoder.embed_positions.weight",
            # Encoder layers
            **{
                f"apm.layers.{i}.self_attn.q_proj.weight": f"encoder.layers.{i}.self_attn.q_proj.weight"
                for i in range(self.config.encoder_layers)
            },
            **{
                f"apm.layers.{i}.self_attn.q_proj.bias": f"encoder.layers.{i}.self_attn.q_proj.bias"
                for i in range(self.config.encoder_layers)
            },
            **{
                f"apm.layers.{i}.self_attn.k_proj.weight": f"encoder.layers.{i}.self_attn.k_proj.weight"
                for i in range(self.config.encoder_layers)
            },
            **{
                f"apm.layers.{i}.self_attn.k_proj.bias": f"encoder.layers.{i}.self_attn.k_proj.bias"
                for i in range(self.config.encoder_layers)
            },
            **{
                f"apm.layers.{i}.self_attn.v_proj.weight": f"encoder.layers.{i}.self_attn.v_proj.weight"
                for i in range(self.config.encoder_layers)
            },
            **{
                f"apm.layers.{i}.self_attn.v_proj.bias": f"encoder.layers.{i}.self_attn.v_proj.bias"
                for i in range(self.config.encoder_layers)
            },
            **{
                f"apm.layers.{i}.self_attn.out_proj.weight": f"encoder.layers.{i}.self_attn.out_proj.weight"
                for i in range(self.config.encoder_layers)
            },
            **{
                f"apm.layers.{i}.self_attn.out_proj.bias": f"encoder.layers.{i}.self_attn.out_proj.bias"
                for i in range(self.config.encoder_layers)
            },
            # Layer norms
            **{
                f"apm.layers.{i}.self_attn_layer_norm.weight": f"encoder.layers.{i}.self_attn_layer_norm.weight"
                for i in range(self.config.encoder_layers)
            },
            **{
                f"apm.layers.{i}.self_attn_layer_norm.bias": f"encoder.layers.{i}.self_attn_layer_norm.bias"
                for i in range(self.config.encoder_layers)
            },
            **{
                f"apm.layers.{i}.final_layer_norm.weight": f"encoder.layers.{i}.final_layer_norm.weight"
                for i in range(self.config.encoder_layers)
            },
            **{
                f"apm.layers.{i}.final_layer_norm.bias": f"encoder.layers.{i}.final_layer_norm.bias"
                for i in range(self.config.encoder_layers)
            },
            # Feed-forward
            **{
                f"apm.layers.{i}.fc1.weight": f"encoder.layers.{i}.fc1.weight"
                for i in range(self.config.encoder_layers)
            },
            **{f"apm.layers.{i}.fc1.bias": f"encoder.layers.{i}.fc1.bias" for i in range(self.config.encoder_layers)},
            **{
                f"apm.layers.{i}.fc2.weight": f"encoder.layers.{i}.fc2.weight"
                for i in range(self.config.encoder_layers)
            },
            **{f"apm.layers.{i}.fc2.bias": f"encoder.layers.{i}.fc2.bias" for i in range(self.config.encoder_layers)},
            # Final layer norm
            "apm.layer_norm.weight": "encoder.layer_norm.weight",
            "apm.layer_norm.bias": "encoder.layer_norm.bias",
        }

        # Load weights
        for gen_key, model_key in weight_mapping.items():
            if gen_key in weights_dict:
                try:
                    param = dict(self.named_parameters())[model_key]
                    param.data.copy_(weights_dict[gen_key])
                    print(f"Loaded {gen_key} -> {model_key}")
                except KeyError:
                    print(f"Warning: Parameter {model_key} not found in model (bias parameter may not exist)")
            else:
                print(f"Warning: {gen_key} not found in weights_dict")

    def forward(
        self,
        input_features: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, dict]:
        """
        Forward pass for Whisper audio encoder.

        Args:
            input_features: Raw audio features (mel spectrograms), shape (batch, time, mel_bins)
            inputs_embeds: Pre-computed embeddings (for MiniCPM-o integration)
            attention_mask: Attention mask
            output_attentions: Whether to return attentions
            output_hidden_states: Whether to return hidden states
            return_dict: Whether to return dict or tuple
        """
        if input_features is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_features and inputs_embeds")

        if inputs_embeds is not None:
            # Use pre-computed embeddings (MiniCPM-o workflow)
            encoder_outputs = self.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif input_features is not None:
            # Process mel spectrograms through convolutional projection
            encoder_outputs = self.encoder(
                mel_spectrograms=input_features,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            raise ValueError("You must specify either input_features or inputs_embeds")

        return encoder_outputs


def create_minicpm_whisper_encoder(config: WhisperAudioConfig = None) -> WhisperAudioEncoder:
    """Create Whisper audio encoder matching MiniCPM-o specifications"""
    if config is None:
        config = WhisperAudioConfig()  # Default MiniCPM-o config

    return WhisperAudioEncoder(config)


# Test function
def test_whisper_audio():
    """Test Whisper audio encoder"""
    print("🎵 Testing Whisper Audio Encoder...")

    config = WhisperAudioConfig()
    model = WhisperAudioEncoder(config)

    # Test with pre-computed embeddings (MiniCPM-o style)
    batch_size, seq_len, hidden_size = 1, 128, config.d_model
    inputs_embeds = torch.randn(batch_size, seq_len, hidden_size)

    with torch.no_grad():
        outputs = model(inputs_embeds=inputs_embeds)

    print(f"✅ Input embeddings shape: {inputs_embeds.shape}")
    print(f"✅ Last hidden state shape: {outputs['last_hidden_state'].shape}")
    print(f"   Expected shape: ({batch_size}, {seq_len}, {config.d_model})")

    expected_shape = (batch_size, seq_len, config.d_model)
    assert (
        outputs["last_hidden_state"].shape == expected_shape
    ), f"Shape mismatch: {outputs['last_hidden_state'].shape} vs {expected_shape}"

    print("✅ Whisper audio encoder test passed!")
    return True


if __name__ == "__main__":
    test_whisper_audio()
