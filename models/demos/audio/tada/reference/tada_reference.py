# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Standalone reference implementations of TADA modules for testing.

These are self-contained copies of the TADA model components that do NOT
depend on transformers.PreTrainedModel or transformers.PretrainedConfig,
allowing them to work with the transformers version in the tt-metal venv.

The implementations are functionally identical to the original TADA code.
"""

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from dac.nn.layers import Snake1d

# ============================================================================
# Utility functions
# ============================================================================


def modulate(x, shift, scale):
    """Apply modulation to input tensor."""
    return x * (1 + scale) + shift


# ============================================================================
# VibeVoice components
# ============================================================================


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(t.dtype)

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class FeedForwardNetwork(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, embed_dim, ffn_dim):
        super().__init__()
        self.gate_proj = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, embed_dim, bias=False)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class HeadLayer(nn.Module):
    def __init__(self, embed_dim, ffn_dim, cond_dim, norm_eps=1e-5):
        super().__init__()
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)
        self.norm = RMSNorm(embed_dim, eps=norm_eps)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 3 * embed_dim, bias=False),
        )

    def forward(self, x, c):
        shift_ffn, scale_ffn, gate_ffn = self.adaLN_modulation(c).chunk(3, dim=-1)
        x = x + gate_ffn * self.ffn(modulate(self.norm(x), shift_ffn, scale_ffn))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, output_size, cond_size, norm_eps=1e-5):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size, eps=norm_eps, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, output_size, bias=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_size, 2 * hidden_size, bias=False),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class VibeVoiceDiffusionHead(nn.Module):
    """Standalone VibeVoice diffusion head (no PreTrainedModel dependency)."""

    def __init__(self, hidden_size=2048, head_layers=6, head_ffn_ratio=4.0, rms_norm_eps=1e-5, latent_size=528):
        super().__init__()
        self.cond_dim = hidden_size

        self.noisy_images_proj = nn.Linear(latent_size, hidden_size, bias=False)
        self.cond_proj = nn.Linear(hidden_size, self.cond_dim, bias=False)
        self.t_embedder = TimestepEmbedder(self.cond_dim)

        ffn_dim = int(hidden_size * head_ffn_ratio)
        self.layers = nn.ModuleList(
            [HeadLayer(hidden_size, ffn_dim, self.cond_dim, rms_norm_eps) for _ in range(head_layers)]
        )
        self.final_layer = FinalLayer(hidden_size, latent_size, self.cond_dim, rms_norm_eps)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for layer in self.layers:
            nn.init.constant_(layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)

    def forward(self, noisy_images, timesteps, condition):
        x = self.noisy_images_proj(noisy_images)
        t = self.t_embedder(timesteps)
        condition = self.cond_proj(condition)
        c = condition + t
        for layer in self.layers:
            x = layer(x, c)
        return self.final_layer(x, c)


# ============================================================================
# Encoder components
# ============================================================================


def WNConv1d(*args, **kwargs):
    return torch.nn.utils.parametrizations.weight_norm(torch.nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return torch.nn.utils.parametrizations.weight_norm(torch.nn.ConvTranspose1d(*args, **kwargs))


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(dim // 2, dim, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2)),
        )

    def forward(self, x):
        return self.block(x)


class WavEncoder(nn.Module):
    def __init__(self, d_model: int = 64, strides: list = [2, 4, 8, 8], d_latent: int = 64):
        super().__init__()
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride)]
        self.block += [Snake1d(d_model), WNConv1d(d_model, d_latent, kernel_size=3, padding=1)]
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)


class LocalSelfAttention(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int = 8, dropout: float = 0.1, max_seq_len: int = 8192, causal: bool = False
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.register_buffer("rope_freqs", self._compute_rope_freqs(self.head_dim, max_seq_len))

    def _compute_rope_freqs(self, head_dim, max_seq_len):
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)
        return torch.stack([freqs.cos(), freqs.sin()], dim=-1)

    def _apply_rope(self, x, seq_len):
        batch, num_heads, seq_len, head_dim = x.shape
        freqs = self.rope_freqs[:seq_len]
        freqs_cos = freqs[..., 0]
        freqs_sin = freqs[..., 1]
        x_reshaped = x.reshape(batch, num_heads, seq_len, head_dim // 2, 2)
        x0 = x_reshaped[..., 0]
        x1 = x_reshaped[..., 1]
        x_rotated_0 = x0 * freqs_cos.unsqueeze(0).unsqueeze(0) - x1 * freqs_sin.unsqueeze(0).unsqueeze(0)
        x_rotated_1 = x0 * freqs_sin.unsqueeze(0).unsqueeze(0) + x1 * freqs_cos.unsqueeze(0).unsqueeze(0)
        x_rotated = torch.stack([x_rotated_0, x_rotated_1], dim=-1)
        return x_rotated.reshape(batch, num_heads, seq_len, head_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self._apply_rope(q, seq_len)
        k = self._apply_rope(k, seq_len)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is None:
            pass  # No masking
        elif mask.dim() == 2:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        elif mask.dim() == 3:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(1), float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v).transpose(1, 2).reshape(batch_size, seq_len, d_model)
        output = self.out_proj(attn_output)
        output = self.dropout(output)
        return self.layer_norm(x + output)


class LocalAttentionEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads=8, d_ff=None, dropout=0.1, activation="gelu", max_seq_len=8192):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.self_attn = LocalSelfAttention(d_model, num_heads, dropout, max_seq_len)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.self_attn(x, mask=mask)
        x = self.norm(x + self.ffn(x))
        return x


class LocalAttentionEncoder(nn.Module):
    def __init__(
        self,
        d_model,
        d_input=None,
        num_layers=4,
        num_heads=8,
        d_ff=None,
        dropout=0.1,
        activation="gelu",
        max_seq_len=8192,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                LocalAttentionEncoderLayer(d_model, num_heads, d_ff, dropout, activation, max_seq_len)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        if d_input is not None and d_input != d_model:
            self.input_proj = nn.Linear(d_input, d_model)
        else:
            self.input_proj = nn.Identity()

    def forward(self, x, mask=None):
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.final_norm(x)


# ============================================================================
# Segment attention masks
# ============================================================================


def create_segment_attention_mask(text_token_mask: torch.Tensor, version: Literal["v1", "v2"] = "v2") -> torch.Tensor:
    """
    Create segment attention mask.

    Args:
        text_token_mask: (batch_size, seq_len) binary mask where 1 = block boundary
        version: "v1" or "v2"
    Returns:
        (batch_size, seq_len, seq_len) boolean mask where True = masked (cannot attend)
    """
    if version == "v2":
        block_ids = torch.cumsum(text_token_mask, dim=1)
        block_ids_i = block_ids.unsqueeze(2)
        block_ids_j = block_ids.unsqueeze(1)
        is_marked_i = text_token_mask.unsqueeze(2).bool()
        is_marked_j = text_token_mask.unsqueeze(1).bool()
        same_block = block_ids_i == block_ids_j
        same_block_valid = same_block & (~is_marked_j | (is_marked_i & same_block))
        prev_block = block_ids_j == (block_ids_i - 1)
        prev_block_valid = prev_block & ~is_marked_j
        can_attend = same_block_valid | (is_marked_i & prev_block_valid)
        return ~can_attend
    elif version == "v1":
        block_ids = torch.cumsum(text_token_mask, dim=1) - text_token_mask
        block_ids_i = block_ids.unsqueeze(2)
        block_ids_j = block_ids.unsqueeze(1)
        same_block = block_ids_j == block_ids_i
        block_ids_j_excluding_last = torch.where(text_token_mask.bool(), -10, block_ids_j[:, 0, :]).unsqueeze(1)
        next_block = block_ids_j_excluding_last == (block_ids_i + 1)
        can_attend = same_block | next_block
        return ~can_attend
    else:
        raise ValueError(f"Unknown version: {version}")


def create_decoder_segment_attention_mask(
    text_token_mask: torch.Tensor, version: Literal["v1", "v2"] = "v2"
) -> torch.Tensor:
    """Decoder segment attention mask."""
    if version == "v2":
        block_ids = torch.cumsum(text_token_mask, dim=1) - text_token_mask
        block_ids_i = block_ids.unsqueeze(2)
        block_ids_j = block_ids.unsqueeze(1)
        same_block = block_ids_j == block_ids_i
        prev_block = block_ids_j == (block_ids_i - 1)
        can_attend = same_block | prev_block
        return ~can_attend
    else:
        raise ValueError(f"Unknown version: {version}")


# ============================================================================
# Decoder components
# ============================================================================


class DecoderBlock(nn.Module):
    def __init__(self, input_dim=16, output_dim=8, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim, output_dim, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2)
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x):
        return self.block(x)


class DACDecoder(nn.Module):
    def __init__(self, input_channel, channels, rates, d_out=1):
        super().__init__()
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride)]
        layers += [Snake1d(output_dim), WNConv1d(output_dim, d_out, kernel_size=7, padding=3), nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    """Standalone TADA Decoder."""

    def __init__(
        self,
        embed_dim=512,
        hidden_dim=1024,
        num_attn_layers=6,
        num_attn_heads=8,
        attn_dim_feedforward=4096,
        attn_dropout=0.1,
        wav_decoder_channels=1536,
        strides=None,
        block_attention="v2",
    ):
        super().__init__()
        if strides is None:
            strides = [4, 4, 5, 6]
        self.block_attention = block_attention
        self.decoder_proj = nn.Linear(embed_dim, hidden_dim)
        self.local_attention_decoder = LocalAttentionEncoder(
            d_model=hidden_dim,
            num_layers=num_attn_layers,
            num_heads=num_attn_heads,
            d_ff=attn_dim_feedforward,
            dropout=attn_dropout,
            activation="gelu",
            max_seq_len=8192,
        )
        self.wav_decoder = DACDecoder(
            input_channel=hidden_dim,
            channels=wav_decoder_channels,
            rates=strides,
        )

    def forward(self, encoded_expanded, token_masks):
        decoder_input = self.decoder_proj(encoded_expanded)
        attn_mask = create_decoder_segment_attention_mask(token_masks, version=self.block_attention)
        decoded_expanded = self.local_attention_decoder(decoder_input, mask=attn_mask)
        return self.wav_decoder(decoded_expanded.transpose(1, 2))


# ============================================================================
# Full Encoder
# ============================================================================


class Encoder(nn.Module):
    """Standalone TADA Encoder (without Aligner)."""

    def __init__(
        self,
        hidden_dim=1024,
        embed_dim=512,
        strides=None,
        num_attn_layers=6,
        num_attn_heads=8,
        attn_dim_feedforward=4096,
        attn_dropout=0.1,
        block_attention="v2",
    ):
        super().__init__()
        if strides is None:
            strides = [6, 5, 4, 4]
        self.block_attention = block_attention
        self.wav_encoder = WavEncoder(d_model=64, strides=strides, d_latent=hidden_dim)
        self.local_attention_encoder = LocalAttentionEncoder(
            d_model=hidden_dim,
            num_layers=num_attn_layers,
            num_heads=num_attn_heads,
            d_ff=attn_dim_feedforward,
            dropout=attn_dropout,
            activation="gelu",
            max_seq_len=8192,
        )
        if hidden_dim != embed_dim:
            self.hidden_linear = nn.Linear(hidden_dim, embed_dim)
        else:
            self.hidden_linear = nn.Identity()
        self.pos_emb = nn.Embedding(2, hidden_dim)

    def get_encoder_outputs(self, audio, token_masks):
        enc_out = self.wav_encoder(F.pad(audio.unsqueeze(1), (0, 960), value=0)).transpose(1, 2)
        seq_len = enc_out.shape[1]
        padded_token_masks = F.pad(token_masks, (0, seq_len - token_masks.shape[1]), value=0)
        enc_out = enc_out + self.pos_emb(padded_token_masks)
        attn_mask = create_segment_attention_mask(padded_token_masks, version=self.block_attention)
        enc_out = self.local_attention_encoder(enc_out, mask=attn_mask)
        enc_out = self.hidden_linear(enc_out)
        return enc_out, padded_token_masks


# ============================================================================
# Speaker Verification
# ============================================================================


class AcousticSpkrVerf(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=768, embed_dim=192, num_layers=3, dropout=0.1):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            out_d = embed_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_d, out_d))
            if i < num_layers - 1:
                layers.append(nn.LayerNorm(out_d))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)
