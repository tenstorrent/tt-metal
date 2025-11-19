#!/usr/bin/env python3
"""
ChatTTS Decoder Reference Implementation for MiniCPM-o-2_6

This module provides a reference implementation of the ChatTTS decoder
based on the configuration found in MiniCPM-o-2_6 and following SpeechT5 patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ChatTTSDecoderConfig:
    """Configuration for ChatTTS decoder based on MiniCPM-o config"""

    def __init__(
        self,
        model_type: str = "conditional_chattts",
        llm_dim: int = 3584,  # From MiniCPM-o config
        hidden_size: int = 512,  # Typical TTS decoder size
        num_layers: int = 6,
        num_heads: int = 8,
        intermediate_size: int = 2048,
        vocab_size: int = 4096,  # ChatTTS semantic token vocabulary
        max_position_embeddings: int = 2048,
        dropout: float = 0.0,  # Disable dropout for deterministic PCC validation
        activation_function: str = "gelu",
    ):
        self.model_type = model_type
        self.llm_dim = llm_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.dropout = dropout
        self.activation_function = activation_function


class ChatTTSDecoderLayer(nn.Module):
    """Single layer of ChatTTS decoder following transformer decoder pattern"""

    def __init__(self, config: ChatTTSDecoderConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size, num_heads=self.num_heads, dropout=config.dropout, batch_first=True
        )

        # Feed-forward network
        self.linear1 = nn.Linear(self.hidden_size, config.intermediate_size)
        self.linear2 = nn.Linear(config.intermediate_size, self.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        # Layer norms
        self.norm1 = nn.LayerNorm(self.hidden_size)
        self.norm2 = nn.LayerNorm(self.hidden_size)

        # Activation
        self.activation = F.gelu if config.activation_function == "gelu" else F.relu

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention with residual connection
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)

        # Add position embeddings if provided
        if position_embeddings is not None:
            hidden_states = hidden_states + position_embeddings

        # Self-attention
        attn_output, _ = self.self_attn(
            query=hidden_states, key=hidden_states, value=hidden_states, attn_mask=attention_mask, need_weights=False
        )

        hidden_states = residual + self.dropout(attn_output)

        # Feed-forward with residual connection
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)

        hidden_states = self.linear2(self.dropout(self.activation(self.linear1(hidden_states))))
        hidden_states = residual + self.dropout(hidden_states)

        return hidden_states


class ChatTTSDecoder(nn.Module):
    """
    ChatTTS decoder that converts LLM outputs to audio tokens.

    This is a reference implementation based on the MiniCPM-o configuration
    and following the general pattern of text-to-speech decoders.
    """

    def __init__(self, config: ChatTTSDecoderConfig):
        super().__init__()
        self.config = config

        # Input projection from LLM dimension to decoder dimension
        self.input_projection = nn.Linear(config.llm_dim, config.hidden_size)

        # Positional embeddings
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # Decoder layers
        self.layers = nn.ModuleList([ChatTTSDecoderLayer(config) for _ in range(config.num_layers)])

        # Output projection to audio vocabulary
        self.output_projection = nn.Linear(config.hidden_size, config.vocab_size)

        # Final layer norm
        self.layer_norm = nn.LayerNorm(config.hidden_size)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights following transformer patterns"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        llm_outputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        target_audio_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Generate audio tokens from LLM outputs.

        Args:
            llm_outputs: [batch_size, seq_len, llm_dim] LLM hidden states
            attention_mask: [batch_size, seq_len] attention mask
            target_audio_tokens: [batch_size, audio_seq_len] for teacher forcing (optional)

        Returns:
            logits: [batch_size, audio_seq_len, vocab_size] audio token logits
            loss: cross-entropy loss if targets provided
        """
        batch_size, seq_len, _ = llm_outputs.shape

        # Project LLM outputs to decoder dimension
        hidden_states = self.input_projection(llm_outputs)
        hidden_states = self.dropout(hidden_states)

        # Add positional embeddings
        position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        position_embeddings = self.position_embeddings(position_ids)
        hidden_states = hidden_states + position_embeddings

        # Create causal attention mask for decoder
        causal_mask = self._create_causal_mask(seq_len, hidden_states.device)

        # Forward through decoder layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states, attention_mask=causal_mask, position_embeddings=None  # Already added
            )

        # Final layer norm
        hidden_states = self.layer_norm(hidden_states)

        # Output projection to audio vocabulary
        logits = self.output_projection(hidden_states)

        # Calculate loss if targets provided
        loss = None
        if target_audio_tokens is not None:
            # For teacher forcing, we predict the next token for each position
            # logits shape: [batch_size, seq_len, vocab_size]
            # target_audio_tokens shape: [batch_size, target_seq_len]

            # Flatten logits and targets, ignoring the last logit if seq_len > target_seq_len
            flat_logits = logits.view(-1, self.config.vocab_size)
            flat_targets = target_audio_tokens.view(-1)

            # Ensure they have the same number of elements
            min_len = min(flat_logits.size(0), flat_targets.size(0))
            flat_logits = flat_logits[:min_len]
            flat_targets = flat_targets[:min_len]

            loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=-100)

        return logits, loss

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask for decoder"""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask.to(device)

    def generate(
        self,
        llm_outputs: torch.Tensor,
        max_length: int = 512,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Autoregressive generation of audio tokens.

        Args:
            llm_outputs: [batch_size, seq_len, llm_dim] LLM hidden states
            max_length: Maximum number of audio tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling

        Returns:
            audio_tokens: [batch_size, max_length] generated audio tokens
        """
        batch_size = llm_outputs.shape[0]

        # Start with BOS token (assuming 0 is BOS)
        generated = torch.zeros(batch_size, 1, dtype=torch.long, device=llm_outputs.device)

        for _ in range(max_length - 1):
            # Get current sequence length
            curr_len = generated.shape[1]

            # Forward pass
            logits, _ = self.forward(llm_outputs[:, :curr_len, :])

            # Get next token logits
            next_logits = logits[:, -1, :] / temperature

            # Apply top-k sampling
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_logits, top_k, dim=-1)
                next_logits = torch.where(
                    next_logits < top_k_logits[:, -1:].unsqueeze(-1),
                    torch.tensor(float("-inf"), device=next_logits.device),
                    next_logits,
                )

            # Apply top-p sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above top_p
                sorted_logits = torch.where(
                    cumulative_probs > top_p, torch.tensor(float("-inf"), device=next_logits.device), sorted_logits
                )

                # Reorder back to original positions
                next_logits = torch.gather(sorted_logits, dim=-1, index=torch.argsort(sorted_indices, dim=-1))

            # Sample next token
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Stop if EOS token generated (assuming 1 is EOS)
            if (next_token == 1).any():
                break

        return generated


def create_chatts_decoder_from_config(minicpm_config: dict) -> ChatTTSDecoder:
    """
    Create ChatTTS decoder from MiniCPM-o configuration.

    Args:
        minicpm_config: Configuration dict from MiniCPM-o model

    Returns:
        Configured ChatTTS decoder
    """
    tts_config = minicpm_config.get("tts_config", {})
    llm_hidden_size = minicpm_config.get("hidden_size", 3584)

    # Create ChatTTS config based on MiniCPM-o settings
    chatts_config = ChatTTSDecoderConfig(
        model_type=tts_config.get("model_type", "conditional_chattts"),
        llm_dim=tts_config.get("llm_dim", llm_hidden_size),
        hidden_size=512,  # Standard decoder size
        num_layers=6,  # Reasonable number of layers
        num_heads=8,  # Standard number of heads
        vocab_size=4096,  # ChatTTS semantic token vocabulary
    )

    return ChatTTSDecoder(chatts_config)


# Test functions
def test_chatts_decoder():
    """Test ChatTTS decoder with dummy data"""

    # Create decoder
    config = ChatTTSDecoderConfig(llm_dim=3584)
    decoder = ChatTTSDecoder(config)

    batch_size, seq_len = 2, 10

    # Dummy LLM outputs
    llm_outputs = torch.randn(batch_size, seq_len, 3584)

    # Test forward pass
    logits, loss = decoder(llm_outputs)
    print(f"Decoder forward: {llm_outputs.shape} -> {logits.shape}")
    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert loss is None  # No targets provided

    # Test with targets
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len + 1))
    logits, loss = decoder(llm_outputs, target_audio_tokens=targets)
    print(f"Decoder with targets: loss = {loss.item():.4f}")
    assert loss is not None

    # Test generation
    generated = decoder.generate(llm_outputs, max_length=5)
    print(f"Decoder generation: {generated.shape}")
    assert generated.shape[1] <= 5  # Should respect max_length

    print("✓ ChatTTS decoder test passed!")


if __name__ == "__main__":
    test_chatts_decoder()
