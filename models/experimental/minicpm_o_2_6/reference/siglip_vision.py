#!/usr/bin/env python3
"""
SigLip Vision Encoder Reference Implementation for MiniCPM-o-2_6

Based on SigLip vision model used in MiniCPM-o:
- Hidden size: 1152
- Num layers: 27
- Num attention heads: 16
- Patch size: 14
- Image size: 980

This provides a PyTorch reference for the vision encoder component.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
import math


class SigLipVisionConfig:
    """Configuration for SigLip vision model matching MiniCPM-o"""

    def __init__(
        self,
        hidden_size: int = 1152,
        image_size: int = 980,
        patch_size: int = 14,
        num_attention_heads: int = 16,
        num_hidden_layers: int = 27,
        intermediate_size: int = 4304,
        layer_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        num_channels: int = 3,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_channels = num_channels

        # Computed values
        self.num_patches = (image_size // patch_size) ** 2


class SigLipVisionEmbeddings(nn.Module):
    """SigLip vision embeddings with patch embedding"""

    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )

        # Position embeddings
        num_patches = (config.image_size // config.patch_size) ** 2
        self.position_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))

        # Class token
        self.class_embedding = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]

        # Patch embedding: [batch, channels, height, width] -> [batch, hidden_size, num_patches_h, num_patches_w]
        patch_embeds = self.patch_embedding(pixel_values)  # [batch, hidden_size, h//patch, w//patch]

        # Flatten patches: [batch, hidden_size, num_patches_h * num_patches_w] -> [batch, num_patches, hidden_size]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        # Add class token
        class_embeds = self.class_embedding.expand(batch_size, -1, -1)  # [batch, 1, hidden_size]
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)  # [batch, num_patches + 1, hidden_size]

        # Add position embeddings
        embeddings = embeddings + self.position_embedding

        # Layer norm
        embeddings = self.layer_norm(embeddings)

        return embeddings


class SigLipAttention(nn.Module):
    """Multi-head attention for SigLip vision model"""

    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_dropout)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Take the dot product between "query" and "key"
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class SigLipMLP(nn.Module):
    """MLP for SigLip vision model"""

    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config

        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SigLipEncoderLayer(nn.Module):
    """Single encoder layer for SigLip vision model"""

    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config

        self.self_attn = SigLipAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SigLipMLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states

        # Self-attention
        hidden_states = self.layer_norm1(hidden_states)
        self_attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self_attn_outputs[0]
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,) + self_attn_outputs[1:]  # add attentions if we output them

        return outputs


class SigLipEncoder(nn.Module):
    """Encoder consisting of multiple SigLip encoder layers"""

    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([SigLipEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        hidden_states = inputs_embeds

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer(
                hidden_states,
                attention_mask,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions += (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return hidden_states, all_hidden_states, all_self_attentions


class SigLipVisionTransformer(nn.Module):
    """Complete SigLip vision transformer matching MiniCPM-o specifications"""

    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config

        self.embeddings = SigLipVisionEmbeddings(config)
        self.encoder = SigLipEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, dict]:
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        return_dict = return_dict if return_dict is not None else True

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        # Pool from class token (index 0)
        pooled_output = last_hidden_state[:, 0, :]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return {
            "last_hidden_state": last_hidden_state,
            "pooler_output": pooled_output,
            "hidden_states": encoder_outputs[1],
            "attentions": encoder_outputs[2],
        }


def create_minicpm_siglip_vision(config: SigLipVisionConfig = None) -> SigLipVisionTransformer:
    """Create SigLip vision model matching MiniCPM-o specifications"""
    if config is None:
        config = SigLipVisionConfig()  # Default MiniCPM-o config

    return SigLipVisionTransformer(config)


# Test function
def test_siglip_vision():
    """Test SigLip vision model"""
    print("🖼️  Testing SigLip Vision Model...")

    config = SigLipVisionConfig()
    model = SigLipVisionTransformer(config)

    # Test input: batch of 1 image
    batch_size = 1
    pixel_values = torch.randn(batch_size, 3, config.image_size, config.image_size)

    with torch.no_grad():
        outputs = model(pixel_values)

    print(f"✅ Input shape: {pixel_values.shape}")
    print(f"✅ Last hidden state: {outputs['last_hidden_state'].shape}")
    print(f"✅ Pooler output: {outputs['pooler_output'].shape}")
    print(f"   Expected pooler shape: ({batch_size}, {config.hidden_size})")

    # Check shapes
    expected_pooled = (batch_size, config.hidden_size)
    expected_last_hidden = (batch_size, config.num_patches + 1, config.hidden_size)

    assert (
        outputs["pooler_output"].shape == expected_pooled
    ), f"Pooler shape mismatch: {outputs['pooler_output'].shape} vs {expected_pooled}"
    assert (
        outputs["last_hidden_state"].shape == expected_last_hidden
    ), f"Last hidden shape mismatch: {outputs['last_hidden_state'].shape} vs {expected_last_hidden}"

    print("✅ SigLip vision model test passed!")
    return True


if __name__ == "__main__":
    test_siglip_vision()
