# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PyTorch reference implementation of the Qwen3-TTS Vocoder (Code2Wav decoder).

Architecture (from Qwen3TTSTokenizerV2Decoder):
    16 codebook tokens [B, T, 16] → SplitResidualVectorQuantizer → [B, codebook_dim, T]
    → CausalConv1d(codebook_dim, latent_dim, k=3)
    → Transformer (8L, hidden=512, heads=16, sliding_window=72) with input/output proj
    → 2x ConvTranspose1d + ConvNeXtBlock (upsample 2×2=4x)
    → Conv1d(latent_dim, decoder_dim=1536, k=7)
    → 4x DecoderBlock (TransConv + 3 ResidualUnits with SnakeBeta)
    → SnakeBeta(96) + Conv1d(96, 1, k=7) → waveform
    Total upsample: 2*2*8*5*4*3 = 1920x → 12.5 Hz → 24kHz

This standalone reference can load directly from speech_tokenizer/model.safetensors
without requiring the full HF/transformers framework.
"""

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, dilation=dilation, groups=groups)
        self.stride = stride
        self.kernel_size = (kernel_size - 1) * dilation + 1
        self.padding = self.kernel_size - self.stride

    def _get_extra_padding(self, x):
        length = x.shape[-1]
        n_frames = (length - self.kernel_size + self.padding) / self.stride + 1
        ideal_length = (math.ceil(n_frames) - 1) * self.stride + (self.kernel_size - self.padding)
        return ideal_length - length

    def forward(self, x):
        extra_padding = self._get_extra_padding(x)
        x = F.pad(x, (self.padding, int(extra_padding)), mode="constant", value=0)
        return self.conv(x).contiguous()


class CausalTransConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_ch, out_ch, kernel_size, stride=stride)
        self.right_pad = kernel_size - stride

    def forward(self, x):
        x = self.conv(x)
        if self.right_pad > 0:
            x = x[..., : x.shape[-1] - self.right_pad]
        return x.contiguous()


class SnakeBeta(nn.Module):
    def __init__(self, in_features, alpha=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
        self.beta = nn.Parameter(torch.zeros(in_features) * alpha)

    def forward(self, x):
        alpha = torch.exp(self.alpha.unsqueeze(0).unsqueeze(-1))
        beta = torch.exp(self.beta.unsqueeze(0).unsqueeze(-1))
        return x + (1.0 / (beta + 1e-9)) * torch.pow(torch.sin(x * alpha), 2)


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = CausalConv1d(dim, dim, kernel_size=7, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, x):
        residual = x
        x = self.dwconv(x).permute(0, 2, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        return residual + x.permute(0, 2, 1)


class ResidualUnit(nn.Module):
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.act1 = SnakeBeta(dim)
        self.conv1 = CausalConv1d(dim, dim, kernel_size=7, dilation=dilation)
        self.act2 = SnakeBeta(dim)
        self.conv2 = CausalConv1d(dim, dim, kernel_size=1)

    def forward(self, x):
        return x + self.conv2(self.act2(self.conv1(self.act1(x))))


class DecoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, upsample_rate):
        super().__init__()
        self.block = nn.ModuleList([
            SnakeBeta(in_dim),
            CausalTransConv1d(in_dim, out_dim, 2 * upsample_rate, upsample_rate),
        ])
        for dilation in (1, 3, 9):
            self.block.append(ResidualUnit(out_dim, dilation))

    def forward(self, x):
        for layer in self.block:
            x = layer(x)
        return x


class EuclideanCodebook(nn.Module):
    def __init__(self, dim, codebook_size, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.cluster_usage = nn.Parameter(torch.ones(codebook_size))
        self.embedding_sum = nn.Parameter(torch.zeros(codebook_size, dim))

    def decode(self, codes):
        embedding = self.embedding_sum / self.cluster_usage.clamp(min=self.epsilon)[:, None]
        return F.embedding(codes, embedding)


class VectorQuantization(nn.Module):
    def __init__(self, dim, codebook_size, codebook_dim=None, epsilon=1e-5):
        super().__init__()
        codebook_dim = codebook_dim or dim
        requires_projection = codebook_dim != dim
        self.project_out = nn.Linear(codebook_dim, dim) if requires_projection else nn.Identity()
        self._codebook = EuclideanCodebook(codebook_dim, codebook_size, epsilon)

    def decode(self, codes):
        quantized = self._codebook.decode(codes)
        quantized = self.project_out(quantized)
        return quantized.transpose(1, 2)


class ResidualVectorQuantization(nn.Module):
    def __init__(self, num_quantizers, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([VectorQuantization(**kwargs) for _ in range(num_quantizers)])

    def decode(self, codes):
        quantized = torch.zeros([1], device=codes.device)[0]
        for idx, layer_codes in enumerate(codes):
            quantized = quantized + self.layers[idx].decode(layer_codes)
        return quantized


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, dimension=128, input_dimension=None, output_dimension=None,
                 n_q=8, bins=1024, force_projection=False, **kwargs):
        super().__init__()
        self.dimension = dimension
        input_dimension = input_dimension or dimension
        output_dimension = output_dimension or dimension

        if input_dimension == dimension and not force_projection:
            self.input_proj = nn.Identity()
        else:
            self.input_proj = nn.Conv1d(input_dimension, dimension, 1, bias=False)
        if output_dimension == dimension and not force_projection:
            self.output_proj = nn.Identity()
        else:
            self.output_proj = nn.Conv1d(dimension, output_dimension, 1, bias=False)

        self.vq = ResidualVectorQuantization(
            num_quantizers=n_q, dim=dimension, codebook_size=bins,
        )

    def decode(self, codes):
        codes = codes.transpose(0, 1)
        quantized = self.vq.decode(codes)
        return self.output_proj(quantized)


class SplitResidualVectorQuantizer(nn.Module):
    def __init__(self, n_q=8, n_q_semantic=1, **kwargs):
        super().__init__()
        self.n_q_semantic = n_q_semantic
        kwargs.pop("q_dropout", None)
        self.rvq_first = ResidualVectorQuantizer(n_q=n_q_semantic, force_projection=True, **kwargs)
        self.rvq_rest = ResidualVectorQuantizer(n_q=n_q - n_q_semantic, force_projection=True, **kwargs)

    def decode(self, codes):
        quantized = self.rvq_first.decode(codes[:, :self.n_q_semantic])
        if codes.shape[1] > self.n_q_semantic:
            quantized = quantized + self.rvq_rest.decode(codes[:, self.n_q_semantic:])
        return quantized


# --- Transformer components ---

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(dtype)


class LayerScale(nn.Module):
    def __init__(self, hidden_size, initial_scale=0.01):
        super().__init__()
        self.scale = nn.Parameter(torch.full((hidden_size,), initial_scale))

    def forward(self, x):
        return self.scale * x


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_position_embeddings=8000, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position_embeddings = max_position_embeddings

    def forward(self, x, position_ids):
        inv_freq = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids = position_ids[:, None, :].float()
        freqs = (inv_freq @ position_ids).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    return (q * cos + rotate_half(q) * sin), (k * cos + rotate_half(k) * sin)


class TransformerAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, sliding_window=72, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self.scaling = head_dim ** -0.5
        self.sliding_window = sliding_window

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=bias)

    def forward(self, x, cos, sin, attention_mask=None):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.num_kv_groups > 1:
            k = k[:, :, None, :, :].expand(B, self.num_kv_heads, self.num_kv_groups, S, self.head_dim)
            k = k.reshape(B, self.num_heads, S, self.head_dim)
            v = v[:, :, None, :, :].expand(B, self.num_kv_heads, self.num_kv_groups, S, self.head_dim)
            v = v.reshape(B, self.num_heads, S, self.head_dim)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        if attention_mask is not None:
            attn = attn + attention_mask
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, S, -1)
        return self.o_proj(out)


class TransformerMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, ffn_dim,
                 sliding_window=72, layer_scale=0.01, eps=1e-5):
        super().__init__()
        self.self_attn = TransformerAttention(hidden_size, num_heads, num_kv_heads, head_dim, sliding_window)
        self.mlp = TransformerMLP(hidden_size, ffn_dim)
        self.input_layernorm = RMSNorm(hidden_size, eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps)
        self.self_attn_layer_scale = LayerScale(hidden_size, layer_scale)
        self.mlp_layer_scale = LayerScale(hidden_size, layer_scale)

    def forward(self, x, cos, sin, attention_mask=None):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, cos, sin, attention_mask)
        x = residual + self.self_attn_layer_scale(x)

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + self.mlp_layer_scale(x)
        return x


class VocoderTransformer(nn.Module):
    def __init__(self, hidden_size=512, latent_dim=1024, num_layers=8, num_heads=16,
                 num_kv_heads=16, head_dim=64, ffn_dim=1024, sliding_window=72,
                 layer_scale=0.01, eps=1e-5, max_position_embeddings=8000, rope_theta=10000.0):
        super().__init__()
        self.input_proj = nn.Linear(latent_dim, hidden_size)
        self.output_proj = nn.Linear(hidden_size, latent_dim)
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_size, num_heads, num_kv_heads, head_dim, ffn_dim,
                             sliding_window, layer_scale, eps)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size, eps)
        self.rotary_emb = RotaryEmbedding(head_dim, max_position_embeddings, rope_theta)
        self.sliding_window = sliding_window

    def _make_sliding_window_mask(self, S, device, dtype):
        mask = torch.full((S, S), float("-inf"), device=device, dtype=dtype)
        for i in range(S):
            start = max(0, i - self.sliding_window + 1)
            mask[i, start : i + 1] = 0
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        B, S, _ = x.shape
        x = self.input_proj(x)
        position_ids = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)
        cos, sin = self.rotary_emb(x, position_ids)
        attn_mask = self._make_sliding_window_mask(S, x.device, x.dtype)

        for layer in self.layers:
            x = layer(x, cos, sin, attn_mask)

        x = self.norm(x)
        return self.output_proj(x)


class VocoderReference(nn.Module):
    """
    Full Qwen3-TTS Vocoder (Code2Wav decoder) reference implementation.

    Config defaults (from speech_tokenizer/config.json):
        codebook_size=2048, codebook_dim=512, hidden_size=512, latent_dim=1024,
        num_hidden_layers=8, num_attention_heads=16, num_key_value_heads=16,
        head_dim=64, intermediate_size=1024, sliding_window=72,
        num_quantizers=16, decoder_dim=1536,
        upsample_rates=[8,5,4,3], upsampling_ratios=[2,2]
    """

    def __init__(
        self,
        codebook_size=2048,
        codebook_dim=512,
        hidden_size=512,
        latent_dim=1024,
        num_layers=8,
        num_heads=16,
        num_kv_heads=16,
        head_dim=64,
        ffn_dim=1024,
        sliding_window=72,
        num_quantizers=16,
        decoder_dim=1536,
        upsample_rates=None,
        upsampling_ratios=None,
        layer_scale=0.01,
        eps=1e-5,
        max_position_embeddings=8000,
        rope_theta=10000.0,
    ):
        super().__init__()
        if upsample_rates is None:
            upsample_rates = [8, 5, 4, 3]
        if upsampling_ratios is None:
            upsampling_ratios = [2, 2]

        self.total_upsample = int(np.prod(upsample_rates + upsampling_ratios))

        # VQ dequantization
        self.quantizer = SplitResidualVectorQuantizer(
            dimension=codebook_dim // 2,
            n_q=num_quantizers,
            n_q_semantic=1,
            bins=codebook_size,
            input_dimension=codebook_dim,
            output_dimension=codebook_dim,
        )

        # Pre-conv
        self.pre_conv = CausalConv1d(codebook_dim, latent_dim, kernel_size=3)

        # Transformer
        self.pre_transformer = VocoderTransformer(
            hidden_size=hidden_size, latent_dim=latent_dim, num_layers=num_layers,
            num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
            ffn_dim=ffn_dim, sliding_window=sliding_window, layer_scale=layer_scale,
            eps=eps, max_position_embeddings=max_position_embeddings, rope_theta=rope_theta,
        )

        # Upsample stages
        self.upsample = nn.ModuleList()
        for factor in upsampling_ratios:
            self.upsample.append(nn.ModuleList([
                CausalTransConv1d(latent_dim, latent_dim, factor, factor),
                ConvNeXtBlock(latent_dim),
            ]))

        # Decoder stages
        decoder = [CausalConv1d(latent_dim, decoder_dim, 7)]
        for i, rate in enumerate(upsample_rates):
            in_dim = decoder_dim // 2 ** i
            out_dim = decoder_dim // 2 ** (i + 1)
            decoder.append(DecoderBlock(in_dim, out_dim, rate))
        output_dim = decoder_dim // 2 ** len(upsample_rates)
        decoder.append(SnakeBeta(output_dim))
        decoder.append(CausalConv1d(output_dim, 1, 7))
        self.decoder = nn.ModuleList(decoder)

    def forward(self, codes):
        """
        Args:
            codes: [B, num_quantizers, T] codebook indices (int64)

        Returns:
            waveform: [B, 1, num_samples] audio
        """
        hidden = self.quantizer.decode(codes)
        hidden = self.pre_conv(hidden).transpose(1, 2)
        hidden = self.pre_transformer(hidden)
        hidden = hidden.permute(0, 2, 1)

        for blocks in self.upsample:
            for block in blocks:
                hidden = block(hidden)

        wav = hidden
        for block in self.decoder:
            wav = block(wav)
        return wav.clamp(min=-1, max=1)

    def chunked_decode(self, codes, chunk_size=300, left_context_size=25):
        """Decode in chunks to avoid OOM on long sequences."""
        wavs = []
        start_index = 0
        while start_index < codes.shape[-1]:
            end_index = min(start_index + chunk_size, codes.shape[-1])
            context_size = left_context_size if start_index - left_context_size > 0 else start_index
            codes_chunk = codes[..., start_index - context_size: end_index]
            wav_chunk = self(codes_chunk)
            wavs.append(wav_chunk[..., context_size * self.total_upsample:])
            start_index = end_index
        return torch.cat(wavs, dim=-1)

    @classmethod
    def from_pretrained(cls, model_name_or_path="Qwen/Qwen3-TTS-12Hz-1.7B-Base"):
        """Load from HuggingFace pretrained weights.

        The vocoder weights are in speech_tokenizer/model.safetensors.
        We only load the decoder part (encoder is not needed for TTS inference).
        """
        import json

        from safetensors import safe_open

        try:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(model_name_or_path, filename="speech_tokenizer/config.json")
            weights_path = hf_hub_download(model_name_or_path, filename="speech_tokenizer/model.safetensors")
        except Exception:
            from pathlib import Path

            base = Path(model_name_or_path)
            config_path = base / "speech_tokenizer" / "config.json"
            weights_path = base / "speech_tokenizer" / "model.safetensors"

        with open(config_path) as f:
            config = json.load(f)

        decoder_cfg = config.get("decoder_config", {})
        model = cls(
            codebook_size=decoder_cfg.get("codebook_size", 2048),
            codebook_dim=decoder_cfg.get("codebook_dim", 512),
            hidden_size=decoder_cfg.get("hidden_size", 512),
            latent_dim=decoder_cfg.get("latent_dim", 1024),
            num_layers=decoder_cfg.get("num_hidden_layers", 8),
            num_heads=decoder_cfg.get("num_attention_heads", 16),
            num_kv_heads=decoder_cfg.get("num_key_value_heads", 16),
            head_dim=decoder_cfg.get("head_dim", 64),
            ffn_dim=decoder_cfg.get("intermediate_size", 1024),
            sliding_window=decoder_cfg.get("sliding_window", 72),
            num_quantizers=decoder_cfg.get("num_quantizers", 16),
            decoder_dim=decoder_cfg.get("decoder_dim", 1536),
            upsample_rates=list(decoder_cfg.get("upsample_rates", [8, 5, 4, 3])),
            upsampling_ratios=list(decoder_cfg.get("upsampling_ratios", [2, 2])),
            layer_scale=decoder_cfg.get("layer_scale_initial_scale", 0.01),
            eps=decoder_cfg.get("rms_norm_eps", 1e-5),
            max_position_embeddings=decoder_cfg.get("max_position_embeddings", 8000),
            rope_theta=decoder_cfg.get("rope_theta", 10000.0),
        )

        state_dict = {}
        with safe_open(str(weights_path), framework="pt") as f:
            for key in f.keys():
                if key.startswith("decoder."):
                    state_dict[key[len("decoder."):]] = f.get_tensor(key)

        model.load_state_dict(state_dict, strict=True)
        return model
