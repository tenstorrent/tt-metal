# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI
# SPDX-License-Identifier: Apache-2.0

"""
MeloTTS Synthesizer for Text-to-Speech.

Full TTS pipeline: Text → BERT → TextEncoder → Duration → Flow → HiFi-GAN → Audio

This integrates with OpenVoice voice conversion for the complete pipeline:
1. MeloTTS generates speech in a base voice
2. OpenVoice converts to the target speaker's voice
"""

import math
from typing import Any, List, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    import ttnn

    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False

from models.demos.openvoice.tt.generator import TTNNGenerator
from models.demos.openvoice.tt.modules.conv1d import ttnn_conv1d


def _ensure_conv1d_weight(w):
    """Ensure weight tensor has correct shape for F.conv1d [out, in, kernel]."""
    if w is None:
        return None
    if w.dim() == 2:
        return w.unsqueeze(2)
    return w


from models.demos.openvoice.tt.duration_predictor import TTNNDurationPredictor, TTNNStochasticDurationPredictor
from models.demos.openvoice.tt.posterior_encoder import TTNNPosteriorEncoder
from models.demos.openvoice.tt.reference_encoder import TTNNReferenceEncoder
from models.demos.openvoice.tt.residual_coupling import TTNNResidualCouplingBlock
from models.demos.openvoice.tt.transformer_flow import TTNNTransformerCouplingBlock


class LayerNorm1d:
    """Layer normalization for 1D sequences."""

    def __init__(self, channels: int, weight: Any = None, bias: Any = None, eps: float = 1e-5):
        self.channels = channels
        self.weight = weight
        self.bias = bias
        self.eps = eps

    def __call__(self, x: Any) -> Any:
        is_torch = isinstance(x, torch.Tensor)
        if not TTNN_AVAILABLE or is_torch:
            x = x.transpose(1, -1)
            x = F.layer_norm(x, (self.channels,), self.weight, self.bias, self.eps)
            return x.transpose(1, -1)

        x = ttnn.permute(x, (0, 2, 1))
        x = ttnn.layer_norm(x, weight=self.weight, bias=self.bias, epsilon=self.eps)
        x = ttnn.permute(x, (0, 2, 1))
        return x


class MultiHeadAttention:
    """Multi-head attention for encoder."""

    def __init__(
        self,
        channels: int,
        out_channels: int,
        n_heads: int,
        conv_q_weight: Any = None,
        conv_q_bias: Any = None,
        conv_k_weight: Any = None,
        conv_k_bias: Any = None,
        conv_v_weight: Any = None,
        conv_v_bias: Any = None,
        conv_o_weight: Any = None,
        conv_o_bias: Any = None,
        device: Optional[Any] = None,
    ):
        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.k_channels = channels // n_heads
        self.device = device

        self.conv_q_weight = conv_q_weight
        self.conv_q_bias = conv_q_bias
        self.conv_k_weight = conv_k_weight
        self.conv_k_bias = conv_k_bias
        self.conv_v_weight = conv_v_weight
        self.conv_v_bias = conv_v_bias
        self.conv_o_weight = conv_o_weight
        self.conv_o_bias = conv_o_bias

    def __call__(self, x: Any, c: Any, attn_mask: Optional[Any] = None) -> Any:
        is_torch = isinstance(x, torch.Tensor)
        if not TTNN_AVAILABLE or is_torch:
            return self._forward_pytorch(x, c, attn_mask)
        return self._forward_ttnn(x, c, attn_mask)

    def _forward_pytorch(self, x, c, attn_mask):
        q = F.conv1d(x, _ensure_conv1d_weight(self.conv_q_weight), self.conv_q_bias)
        k = F.conv1d(c, _ensure_conv1d_weight(self.conv_k_weight), self.conv_k_bias)
        v = F.conv1d(c, _ensure_conv1d_weight(self.conv_v_weight), self.conv_v_bias)

        b, d, t_s = k.size()
        t_t = q.size(2)

        q = q.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        k = k.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        v = v.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.k_channels)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e4)

        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)

        output = output.transpose(2, 3).contiguous().view(b, d, t_t)
        output = F.conv1d(output, _ensure_conv1d_weight(self.conv_o_weight), self.conv_o_bias)

        return output

    def _forward_ttnn(self, x, c, attn_mask):
        """Attention computation (TTNN) - uses fused attention when available."""
        q = ttnn_conv1d(x, self.conv_q_weight, self.conv_q_bias, device=self.device)
        k = ttnn_conv1d(c, self.conv_k_weight, self.conv_k_bias, device=self.device)
        v = ttnn_conv1d(c, self.conv_v_weight, self.conv_v_bias, device=self.device)

        b, d, t_s = k.shape[0], k.shape[1], k.shape[2]
        t_t = q.shape[2]

        q = ttnn.reshape(q, (b, self.n_heads, self.k_channels, t_t))
        q = ttnn.permute(q, (0, 1, 3, 2))  # [B, n_heads, T, k_channels]
        k = ttnn.reshape(k, (b, self.n_heads, self.k_channels, t_s))
        k = ttnn.permute(k, (0, 1, 3, 2))  # [B, n_heads, T, k_channels]
        v = ttnn.reshape(v, (b, self.n_heads, self.k_channels, t_s))
        v = ttnn.permute(v, (0, 1, 3, 2))  # [B, n_heads, T, k_channels]

        # Try to use fused scaled_dot_product_attention for better performance
        use_fused = hasattr(ttnn, "transformer") and hasattr(ttnn.transformer, "scaled_dot_product_attention")

        if use_fused:
            try:
                # Use FlashAttention-2 via TTNN fused SDPA
                scale = 1.0 / math.sqrt(self.k_channels)
                output = ttnn.transformer.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    is_causal=False,
                    scale=scale,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                # Reshape back
                output = ttnn.permute(output, (0, 1, 3, 2))  # [B, n_heads, k_channels, T]
                output = ttnn.reshape(output, (b, d, t_t))
                output = ttnn_conv1d(output, self.conv_o_weight, self.conv_o_bias, device=self.device)
                return output
            except Exception:
                # Fall back to manual implementation if fused attention fails
                pass

        # Manual attention computation (fallback)
        scale = 1.0 / math.sqrt(self.k_channels)
        k_t = ttnn.permute(k, (0, 1, 3, 2))

        # Attention with L1 memory config for efficient computation
        scores = ttnn.matmul(q, k_t, memory_config=ttnn.L1_MEMORY_CONFIG)
        scores = ttnn.multiply(scores, scale)

        if attn_mask is not None:
            scores = ttnn.where(attn_mask == 0, -1e4, scores)

        attn = ttnn.softmax(scores, dim=-1)
        output = ttnn.matmul(attn, v, memory_config=ttnn.L1_MEMORY_CONFIG)

        output = ttnn.permute(output, (0, 1, 3, 2))
        output = ttnn.reshape(output, (b, d, t_t))
        output = ttnn_conv1d(output, self.conv_o_weight, self.conv_o_bias, device=self.device)

        return output


class FFN:
    """Feed-forward network for encoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_channels: int,
        kernel_size: int,
        conv_1_weight: Any = None,
        conv_1_bias: Any = None,
        conv_2_weight: Any = None,
        conv_2_bias: Any = None,
        device: Optional[Any] = None,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.device = device

        self.conv_1_weight = conv_1_weight
        self.conv_1_bias = conv_1_bias
        self.conv_2_weight = conv_2_weight
        self.conv_2_bias = conv_2_bias

    def __call__(self, x: Any, x_mask: Any) -> Any:
        is_torch = isinstance(x, torch.Tensor)
        if not TTNN_AVAILABLE or is_torch:
            return self._forward_pytorch(x, x_mask)
        return self._forward_ttnn(x, x_mask)

    def _forward_pytorch(self, x, x_mask):
        pad = self.kernel_size // 2
        x = F.conv1d(F.pad(x * x_mask, (pad, pad)), _ensure_conv1d_weight(self.conv_1_weight), self.conv_1_bias)
        x = F.relu(x)
        x = F.conv1d(F.pad(x * x_mask, (pad, pad)), _ensure_conv1d_weight(self.conv_2_weight), self.conv_2_bias)
        return x * x_mask

    def _forward_ttnn(self, x, x_mask):
        pad = self.kernel_size // 2
        x = ttnn.multiply(x, x_mask)
        x = ttnn.pad(x, ((0, 0), (0, 0), (pad, pad)))
        # Fused conv + relu for better performance
        x = ttnn_conv1d(x, self.conv_1_weight, self.conv_1_bias, device=self.device, activation="relu")
        x = ttnn.multiply(x, x_mask)
        x = ttnn.pad(x, ((0, 0), (0, 0), (pad, pad)))
        x = ttnn_conv1d(x, self.conv_2_weight, self.conv_2_bias, device=self.device)
        return ttnn.multiply(x, x_mask)


class MeloTextEncoder:
    """
    MeloTTS Text Encoder.

    Encodes text with:
    - Token embeddings
    - Tone embeddings (for tonal languages)
    - Language embeddings (multi-lingual)
    - BERT features (prosody)
    - Transformer encoder
    """

    def __init__(
        self,
        n_vocab: int,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        num_languages: int = 10,
        num_tones: int = 10,
        gin_channels: int = 0,
        emb_weight: Any = None,
        tone_emb_weight: Any = None,
        language_emb_weight: Any = None,
        bert_proj_weight: Any = None,
        bert_proj_bias: Any = None,
        ja_bert_proj_weight: Any = None,
        ja_bert_proj_bias: Any = None,
        proj_weight: Any = None,
        proj_bias: Any = None,
        attn_layers: Optional[List] = None,
        norm1_layers: Optional[List] = None,
        ffn_layers: Optional[List] = None,
        norm2_layers: Optional[List] = None,
        cond_weight: Any = None,
        cond_bias: Any = None,
        device: Optional[Any] = None,
    ):
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.num_languages = num_languages
        self.num_tones = num_tones
        self.gin_channels = gin_channels
        self.device = device

        self.emb_weight = emb_weight
        self.tone_emb_weight = tone_emb_weight
        self.language_emb_weight = language_emb_weight
        self.bert_proj_weight = bert_proj_weight
        self.bert_proj_bias = bert_proj_bias
        self.ja_bert_proj_weight = ja_bert_proj_weight
        self.ja_bert_proj_bias = ja_bert_proj_bias
        self.proj_weight = proj_weight
        self.proj_bias = proj_bias
        self.attn_layers = attn_layers or []
        self.norm1_layers = norm1_layers or []
        self.ffn_layers = ffn_layers or []
        self.norm2_layers = norm2_layers or []
        self.cond_weight = cond_weight
        self.cond_bias = cond_bias

    def __call__(
        self,
        x: Any,
        x_lengths: Any,
        tone: Any,
        language: Any,
        bert: Any,
        ja_bert: Any,
        g: Optional[Any] = None,
    ) -> Tuple[Any, Any, Any, Any]:
        """
        Encode text.

        Args:
            x: Token indices [B, T]
            x_lengths: Sequence lengths [B]
            tone: Tone indices [B, T]
            language: Language indices [B, T]
            bert: BERT features [B, 1024, T]
            ja_bert: Japanese BERT features [B, 768, T]
            g: Speaker conditioning [B, gin_channels, 1]

        Returns:
            (hidden, mean, log_variance, mask)
        """
        # Check if inputs are PyTorch tensors
        is_torch = isinstance(x, torch.Tensor)

        if not TTNN_AVAILABLE or is_torch:
            return self._forward_pytorch(x, x_lengths, tone, language, bert, ja_bert, g)
        return self._forward_ttnn(x, x_lengths, tone, language, bert, ja_bert, g)

    def _forward_pytorch(self, x, x_lengths, tone, language, bert, ja_bert, g):
        # Embeddings
        x_emb = F.embedding(x, self.emb_weight)
        tone_emb = F.embedding(tone, self.tone_emb_weight)
        lang_emb = F.embedding(language, self.language_emb_weight)

        # BERT projections
        bert_emb = F.conv1d(bert, _ensure_conv1d_weight(self.bert_proj_weight), self.bert_proj_bias).transpose(1, 2)
        ja_bert_emb = F.conv1d(
            ja_bert, _ensure_conv1d_weight(self.ja_bert_proj_weight), self.ja_bert_proj_bias
        ).transpose(1, 2)

        # Combine embeddings
        x = (x_emb + tone_emb + lang_emb + bert_emb + ja_bert_emb) * math.sqrt(self.hidden_channels)
        x = x.transpose(1, 2)  # [B, hidden, T]

        # Create mask
        max_len = x.size(2)
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) < x_lengths.unsqueeze(1)
        x_mask = mask.unsqueeze(1).to(x.dtype)

        # Apply speaker conditioning if present
        if g is not None and self.cond_weight is not None:
            g_cond = F.conv1d(g, _ensure_conv1d_weight(self.cond_weight), self.cond_bias)
            x = x + g_cond

        # Transformer encoder
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask

        for i in range(self.n_layers):
            y = self.attn_layers[i](x, x, attn_mask)
            x = self.norm1_layers[i](x + y)
            y = self.ffn_layers[i](x, x_mask)
            x = self.norm2_layers[i](x + y)

        x = x * x_mask

        # Project to mean/variance
        stats = F.conv1d(x, _ensure_conv1d_weight(self.proj_weight), self.proj_bias) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)

        return x, m, logs, x_mask

    def _forward_ttnn(self, x, x_lengths, tone, language, bert, ja_bert, g):
        # Embeddings
        x_emb = ttnn.embedding(x, self.emb_weight)
        tone_emb = ttnn.embedding(tone, self.tone_emb_weight)
        lang_emb = ttnn.embedding(language, self.language_emb_weight)

        # BERT projections
        bert_emb = ttnn_conv1d(bert, self.bert_proj_weight, self.bert_proj_bias, device=self.device)
        bert_emb = ttnn.permute(bert_emb, (0, 2, 1))
        ja_bert_emb = ttnn_conv1d(ja_bert, self.ja_bert_proj_weight, self.ja_bert_proj_bias, device=self.device)
        ja_bert_emb = ttnn.permute(ja_bert_emb, (0, 2, 1))

        # Combine
        x = ttnn.add(ttnn.add(ttnn.add(ttnn.add(x_emb, tone_emb), lang_emb), bert_emb), ja_bert_emb)
        x = ttnn.multiply(x, math.sqrt(self.hidden_channels))
        x = ttnn.permute(x, (0, 2, 1))

        # Create mask
        max_len = x.shape[2]
        mask = ttnn.arange(0, max_len)
        mask = ttnn.unsqueeze(mask, 0)
        x_lengths_exp = ttnn.unsqueeze(x_lengths, 1)
        mask = ttnn.lt(mask, x_lengths_exp)
        x_mask = ttnn.unsqueeze(mask, 1)
        x_mask = ttnn.to_dtype(x_mask, x.dtype)

        # Speaker conditioning
        if g is not None and self.cond_weight is not None:
            g_cond = ttnn_conv1d(g, self.cond_weight, self.cond_bias, device=self.device)
            x = ttnn.add(x, g_cond)

        # Encoder
        attn_mask = ttnn.multiply(ttnn.unsqueeze(x_mask, 2), ttnn.unsqueeze(x_mask, -1))
        x = ttnn.multiply(x, x_mask)

        for i in range(self.n_layers):
            y = self.attn_layers[i](x, x, attn_mask)
            x = self.norm1_layers[i](ttnn.add(x, y))
            y = self.ffn_layers[i](x, x_mask)
            x = self.norm2_layers[i](ttnn.add(x, y))

        x = ttnn.multiply(x, x_mask)

        # Project
        stats = ttnn_conv1d(x, self.proj_weight, self.proj_bias, device=self.device)
        stats = ttnn.multiply(stats, x_mask)
        m = stats[:, : self.out_channels, :]
        logs = stats[:, self.out_channels :, :]

        return x, m, logs, x_mask


class TTNNMeloTTS:
    """
    MeloTTS Synthesizer for Text-to-Speech.

    Full pipeline:
    1. Encode text (with BERT, tone, language embeddings)
    2. Predict durations
    3. Expand encoded features to audio length
    4. Apply normalizing flow
    5. Decode with HiFi-GAN

    Can work with OpenVoice for voice cloning:
    1. Generate base audio with MeloTTS
    2. Convert voice with OpenVoice ToneColorConverter
    """

    def __init__(
        self,
        n_vocab: int,
        spec_channels: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        n_speakers: int = 0,
        gin_channels: int = 256,
        num_languages: int = 10,
        num_tones: int = 10,
        use_transformer_flow: bool = True,
        n_flow_layers: int = 4,
        n_layers_trans_flow: int = 6,
        enc_p: Optional[MeloTextEncoder] = None,
        dec: Optional[TTNNGenerator] = None,
        enc_q: Optional[TTNNPosteriorEncoder] = None,
        flow: Optional[Any] = None,
        dp: Optional[TTNNDurationPredictor] = None,
        sdp: Optional[TTNNStochasticDurationPredictor] = None,
        emb_g: Any = None,
        ref_enc: Optional[TTNNReferenceEncoder] = None,
        device: Optional[Any] = None,
    ):
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.num_languages = num_languages
        self.num_tones = num_tones
        self.use_transformer_flow = use_transformer_flow
        self.device = device

        self.enc_p = enc_p
        self.dec = dec
        self.enc_q = enc_q
        self.flow = flow
        self.dp = dp
        self.sdp = sdp
        self.emb_g = emb_g
        self.ref_enc = ref_enc

    def infer(
        self,
        x: Any,
        x_lengths: Any,
        sid: Any,
        tone: Any,
        language: Any,
        bert: Any,
        ja_bert: Any,
        noise_scale: float = 0.667,
        length_scale: float = 1.0,
        noise_scale_w: float = 0.8,
        sdp_ratio: float = 0.0,
        max_len: Optional[int] = None,
    ) -> Tuple[Any, Any, Any]:
        """
        Generate speech from text.

        Args:
            x: Token indices [B, T]
            x_lengths: Sequence lengths [B]
            sid: Speaker ID [B] or speaker embedding
            tone: Tone indices [B, T]
            language: Language indices [B, T]
            bert: BERT features [B, 1024, T]
            ja_bert: Japanese BERT features [B, 768, T]
            noise_scale: Noise scale for sampling
            length_scale: Duration scaling factor (>1 = slower)
            noise_scale_w: Noise scale for duration
            sdp_ratio: Stochastic duration predictor ratio
            max_len: Maximum output length

        Returns:
            (audio, attention, y_mask)
        """
        # Check if inputs are PyTorch tensors - use PyTorch path
        is_torch = isinstance(x, torch.Tensor)

        if not TTNN_AVAILABLE or is_torch:
            return self._infer_pytorch(
                x,
                x_lengths,
                sid,
                tone,
                language,
                bert,
                ja_bert,
                noise_scale,
                length_scale,
                noise_scale_w,
                sdp_ratio,
                max_len,
            )
        return self._infer_ttnn(
            x,
            x_lengths,
            sid,
            tone,
            language,
            bert,
            ja_bert,
            noise_scale,
            length_scale,
            noise_scale_w,
            sdp_ratio,
            max_len,
        )

    def _infer_pytorch(
        self,
        x,
        x_lengths,
        sid,
        tone,
        language,
        bert,
        ja_bert,
        noise_scale,
        length_scale,
        noise_scale_w,
        sdp_ratio,
        max_len,
    ):
        # Get speaker embedding
        if self.n_speakers > 0:
            g = F.embedding(sid, self.emb_g).unsqueeze(-1)
        else:
            raise ValueError("Reference audio needed for n_speakers=0")

        # Encode text
        x_enc, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, tone, language, bert, ja_bert, g=g)

        # Predict durations
        if sdp_ratio > 0 and self.sdp is not None:
            logw_sdp = self.sdp(x_enc, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
        else:
            logw_sdp = 0

        logw_dp = self.dp(x_enc, x_mask, g=g)
        logw = logw_sdp * sdp_ratio + logw_dp * (1 - sdp_ratio)

        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()

        # Create output mask
        y_mask = self._sequence_mask(y_lengths, None).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(2) * y_mask.unsqueeze(-1)

        # Generate attention path
        attn = self._generate_path(w_ceil, attn_mask)

        # Expand to output length
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        # Sample from prior
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

        # Apply flow (reverse)
        z = self.flow(z_p, y_mask, g=g, reverse=True)

        # Decode
        o = self.dec((z * y_mask)[:, :, :max_len], g=g)

        return o, attn, y_mask

    def _infer_ttnn(
        self,
        x,
        x_lengths,
        sid,
        tone,
        language,
        bert,
        ja_bert,
        noise_scale,
        length_scale,
        noise_scale_w,
        sdp_ratio,
        max_len,
    ):
        # Fall back to PyTorch for now (complex control flow)
        # Transfer to CPU, run, transfer back
        def to_pytorch(t):
            if not TTNN_AVAILABLE:
                return t
            t = ttnn.to_torch(ttnn.from_device(t))
            # Convert bfloat16 to float32 for PyTorch ops
            if t.dtype == torch.bfloat16:
                t = t.float()
            return t

        x_cpu = to_pytorch(x)
        x_lengths_cpu = to_pytorch(x_lengths)
        sid_cpu = to_pytorch(sid)
        tone_cpu = to_pytorch(tone)
        language_cpu = to_pytorch(language)
        bert_cpu = to_pytorch(bert)
        ja_bert_cpu = to_pytorch(ja_bert)

        o, attn, y_mask = self._infer_pytorch(
            x_cpu,
            x_lengths_cpu,
            sid_cpu,
            tone_cpu,
            language_cpu,
            bert_cpu,
            ja_bert_cpu,
            noise_scale,
            length_scale,
            noise_scale_w,
            sdp_ratio,
            max_len,
        )

        if TTNN_AVAILABLE and self.device:
            o = ttnn.from_torch(o.float(), dtype=ttnn.bfloat16, device=self.device)

        return o, attn, y_mask

    def _sequence_mask(self, lengths, max_len=None):
        """Create sequence mask."""
        if max_len is None:
            max_len = lengths.max()
        mask = torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
        return mask

    def _generate_path(self, duration, mask):
        """Generate monotonic attention path from durations."""
        b, _, t_x = duration.shape
        t_y = mask.shape[2]
        dur_cumsum = torch.cumsum(duration.squeeze(1), dim=1)

        # Vectorized: broadcast compare frame indices against cumulative durations
        # dur_cumsum: [b, t_x], dur_cumsum_shifted: [b, t_x]
        dur_cumsum_shifted = F.pad(dur_cumsum[:, :-1], (1, 0), value=0)  # [b, t_x]
        # frame indices: [1, t_y, 1]
        j_indices = torch.arange(t_y, device=duration.device, dtype=duration.dtype).unsqueeze(0).unsqueeze(2)
        # Compare: j >= start AND j < end -> [b, t_y, t_x]
        path = ((j_indices >= dur_cumsum_shifted.unsqueeze(1)) & (j_indices < dur_cumsum.unsqueeze(1))).float()

        return path.unsqueeze(1) * mask

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict,
        config: dict,
        device: Optional[Any] = None,
    ) -> "TTNNMeloTTS":
        """Create MeloTTS from state dict and config."""
        from models.demos.openvoice.utils.weight_loader import remove_weight_norm_from_state_dict

        # Remove weight normalization (fuse weight_g and weight_v into weight)
        state_dict = remove_weight_norm_from_state_dict(state_dict)

        # Extract config values
        model_cfg = config.get("model", config)
        data_cfg = config.get("data", config)

        # Get n_vocab from symbols list if available
        symbols = config.get("symbols", [])
        n_vocab = len(symbols) if symbols else model_cfg.get("n_vocab", 256)

        spec_channels = data_cfg.get("filter_length", 1024) // 2 + 1
        inter_channels = model_cfg.get("inter_channels", 192)
        hidden_channels = model_cfg.get("hidden_channels", 192)
        filter_channels = model_cfg.get("filter_channels", 768)
        n_heads = model_cfg.get("n_heads", 2)
        n_layers = model_cfg.get("n_layers", 6)
        kernel_size = model_cfg.get("kernel_size", 1)
        # n_speakers is in data section, not model
        n_speakers = data_cfg.get("n_speakers", model_cfg.get("n_speakers", 0))
        gin_channels = model_cfg.get("gin_channels", 256)
        # num_languages and num_tones can be at top level
        num_languages = config.get("num_languages", model_cfg.get("num_languages", 10))
        num_tones = config.get("num_tones", model_cfg.get("num_tones", 10))
        use_transformer_flow = model_cfg.get("use_transformer_flow", True)
        n_flow_layers = model_cfg.get("n_flow_layer", 4)
        n_layers_trans_flow = model_cfg.get("n_layers_trans_flow", 6)
        use_spk_conditioned_encoder = model_cfg.get("use_spk_conditioned_encoder", True)

        # Determine encoder gin_channels
        enc_gin_channels = gin_channels if use_spk_conditioned_encoder and gin_channels > 0 else 0

        # Build text encoder
        enc_p = cls._build_text_encoder(
            state_dict,
            "enc_p",
            n_vocab,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            num_languages,
            num_tones,
            enc_gin_channels,
            device,
        )

        # Build decoder
        dec = TTNNGenerator.from_state_dict(
            state_dict,
            "dec",
            inter_channels,
            model_cfg.get("resblock", "1"),
            model_cfg.get("resblock_kernel_sizes", [3, 7, 11]),
            model_cfg.get("resblock_dilation_sizes", [[1, 3, 5], [1, 3, 5], [1, 3, 5]]),
            model_cfg.get("upsample_rates", [8, 8, 2, 2]),
            model_cfg.get("upsample_initial_channel", 512),
            model_cfg.get("upsample_kernel_sizes", [16, 16, 4, 4]),
            gin_channels,
            device,
        )

        # Build posterior encoder (for training/voice conversion)
        enc_q = TTNNPosteriorEncoder.from_state_dict(
            state_dict,
            "enc_q",
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels,
            device,
        )

        # Build flow
        if use_transformer_flow:
            flow = TTNNTransformerCouplingBlock.from_state_dict(
                state_dict,
                "flow",
                inter_channels,
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers_trans_flow,
                5,
                0.0,
                n_flow_layers,
                gin_channels,
                device,
            )
        else:
            flow = TTNNResidualCouplingBlock.from_state_dict(
                state_dict,
                "flow",
                inter_channels,
                hidden_channels,
                5,
                1,
                n_flow_layers,
                gin_channels,
                device,
            )

        # Build duration predictors
        dp = TTNNDurationPredictor.from_state_dict(
            state_dict,
            "dp",
            hidden_channels,
            256,
            3,
            0.5,
            gin_channels,
            device,
        )

        sdp = cls._build_stochastic_duration_predictor(
            state_dict,
            "sdp",
            hidden_channels,
            192,
            3,
            0.5,
            4,
            gin_channels,
            device,
        )

        # Speaker embedding or reference encoder
        emb_g = state_dict.get("emb_g.weight") if n_speakers > 0 else None
        ref_enc = None
        if n_speakers == 0:
            ref_enc = TTNNReferenceEncoder.from_state_dict(
                state_dict,
                "ref_enc",
                spec_channels,
                gin_channels,
                device,
            )

        return cls(
            n_vocab=n_vocab,
            spec_channels=spec_channels,
            inter_channels=inter_channels,
            hidden_channels=hidden_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            n_layers=n_layers,
            n_speakers=n_speakers,
            gin_channels=gin_channels,
            num_languages=num_languages,
            num_tones=num_tones,
            use_transformer_flow=use_transformer_flow,
            n_flow_layers=n_flow_layers,
            enc_p=enc_p,
            dec=dec,
            enc_q=enc_q,
            flow=flow,
            dp=dp,
            sdp=sdp,
            emb_g=emb_g,
            ref_enc=ref_enc,
            device=device,
        )

    @classmethod
    def _build_text_encoder(
        cls,
        state_dict: dict,
        prefix: str,
        n_vocab: int,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        num_languages: int,
        num_tones: int,
        gin_channels: int,
        device: Optional[Any],
    ) -> MeloTextEncoder:
        """Build MeloTextEncoder from state dict."""

        # Embeddings
        emb_weight = state_dict.get(f"{prefix}.emb.weight")
        tone_emb_weight = state_dict.get(f"{prefix}.tone_emb.weight")
        language_emb_weight = state_dict.get(f"{prefix}.language_emb.weight")

        # BERT projections
        bert_proj_weight = state_dict.get(f"{prefix}.bert_proj.weight")
        bert_proj_bias = state_dict.get(f"{prefix}.bert_proj.bias")
        ja_bert_proj_weight = state_dict.get(f"{prefix}.ja_bert_proj.weight")
        ja_bert_proj_bias = state_dict.get(f"{prefix}.ja_bert_proj.bias")

        # Final projection
        proj_weight = state_dict.get(f"{prefix}.proj.weight")
        proj_bias = state_dict.get(f"{prefix}.proj.bias")

        # Speaker conditioning in encoder
        cond_weight = None
        cond_bias = None
        if gin_channels > 0:
            cond_weight = state_dict.get(f"{prefix}.encoder.spk_emb_linear.weight")
            cond_bias = state_dict.get(f"{prefix}.encoder.spk_emb_linear.bias")

        # Build attention and FFN layers
        attn_layers = []
        norm1_layers = []
        ffn_layers = []
        norm2_layers = []

        for i in range(n_layers):
            # MultiHeadAttention weights
            attn_layers.append(
                MultiHeadAttention(
                    channels=hidden_channels,
                    out_channels=hidden_channels,
                    n_heads=n_heads,
                    conv_q_weight=state_dict.get(f"{prefix}.encoder.attn_layers.{i}.conv_q.weight"),
                    conv_q_bias=state_dict.get(f"{prefix}.encoder.attn_layers.{i}.conv_q.bias"),
                    conv_k_weight=state_dict.get(f"{prefix}.encoder.attn_layers.{i}.conv_k.weight"),
                    conv_k_bias=state_dict.get(f"{prefix}.encoder.attn_layers.{i}.conv_k.bias"),
                    conv_v_weight=state_dict.get(f"{prefix}.encoder.attn_layers.{i}.conv_v.weight"),
                    conv_v_bias=state_dict.get(f"{prefix}.encoder.attn_layers.{i}.conv_v.bias"),
                    conv_o_weight=state_dict.get(f"{prefix}.encoder.attn_layers.{i}.conv_o.weight"),
                    conv_o_bias=state_dict.get(f"{prefix}.encoder.attn_layers.{i}.conv_o.bias"),
                    device=device,
                )
            )

            # LayerNorm 1
            norm1_layers.append(
                LayerNorm1d(
                    channels=hidden_channels,
                    weight=state_dict.get(f"{prefix}.encoder.norm_layers_1.{i}.gamma"),
                    bias=state_dict.get(f"{prefix}.encoder.norm_layers_1.{i}.beta"),
                )
            )

            # FFN
            ffn_layers.append(
                FFN(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    filter_channels=filter_channels,
                    kernel_size=kernel_size,
                    conv_1_weight=state_dict.get(f"{prefix}.encoder.ffn_layers.{i}.conv_1.weight"),
                    conv_1_bias=state_dict.get(f"{prefix}.encoder.ffn_layers.{i}.conv_1.bias"),
                    conv_2_weight=state_dict.get(f"{prefix}.encoder.ffn_layers.{i}.conv_2.weight"),
                    conv_2_bias=state_dict.get(f"{prefix}.encoder.ffn_layers.{i}.conv_2.bias"),
                    device=device,
                )
            )

            # LayerNorm 2
            norm2_layers.append(
                LayerNorm1d(
                    channels=hidden_channels,
                    weight=state_dict.get(f"{prefix}.encoder.norm_layers_2.{i}.gamma"),
                    bias=state_dict.get(f"{prefix}.encoder.norm_layers_2.{i}.beta"),
                )
            )

        return MeloTextEncoder(
            n_vocab=n_vocab,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            num_languages=num_languages,
            num_tones=num_tones,
            gin_channels=gin_channels,
            emb_weight=emb_weight,
            tone_emb_weight=tone_emb_weight,
            language_emb_weight=language_emb_weight,
            bert_proj_weight=bert_proj_weight,
            bert_proj_bias=bert_proj_bias,
            ja_bert_proj_weight=ja_bert_proj_weight,
            ja_bert_proj_bias=ja_bert_proj_bias,
            proj_weight=proj_weight,
            proj_bias=proj_bias,
            attn_layers=attn_layers,
            norm1_layers=norm1_layers,
            ffn_layers=ffn_layers,
            norm2_layers=norm2_layers,
            cond_weight=cond_weight,
            cond_bias=cond_bias,
            device=device,
        )

    @classmethod
    def _build_stochastic_duration_predictor(
        cls,
        state_dict: dict,
        prefix: str,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float,
        n_flows: int,
        gin_channels: int,
        device: Optional[Any],
    ) -> TTNNStochasticDurationPredictor:
        """Build StochasticDurationPredictor from state dict."""
        from models.demos.openvoice.tt.duration_predictor import (
            ElementwiseAffine,
            Flip,
            Log,
            TTNNStochasticDurationPredictor,
        )

        # Note: filter_channels in SDP is overwritten to in_channels in original code
        filter_channels = in_channels

        # Main convs
        convs = cls._build_dds_conv(state_dict, f"{prefix}.convs", filter_channels, kernel_size, 3, p_dropout, device)

        # Post convs
        post_convs = cls._build_dds_conv(
            state_dict, f"{prefix}.post_convs", filter_channels, kernel_size, 3, p_dropout, device
        )

        # Build flows
        flows = []
        flows.append(
            ElementwiseAffine(
                channels=2,
                m=state_dict.get(f"{prefix}.flows.0.m"),
                logs=state_dict.get(f"{prefix}.flows.0.logs"),
            )
        )

        for i in range(n_flows):
            # ConvFlow
            flow_idx = 1 + i * 2
            conv_flow = cls._build_conv_flow(
                state_dict, f"{prefix}.flows.{flow_idx}", 2, filter_channels, kernel_size, 3, device
            )
            flows.append(conv_flow)
            flows.append(Flip())

        # Build post flows
        post_flows = []
        post_flows.append(
            ElementwiseAffine(
                channels=2,
                m=state_dict.get(f"{prefix}.post_flows.0.m"),
                logs=state_dict.get(f"{prefix}.post_flows.0.logs"),
            )
        )

        for i in range(4):  # 4 post flows
            flow_idx = 1 + i * 2
            conv_flow = cls._build_conv_flow(
                state_dict, f"{prefix}.post_flows.{flow_idx}", 2, filter_channels, kernel_size, 3, device
            )
            post_flows.append(conv_flow)
            post_flows.append(Flip())

        # Condition projection
        cond_weight = state_dict.get(f"{prefix}.cond.weight") if gin_channels > 0 else None
        cond_bias = state_dict.get(f"{prefix}.cond.bias") if gin_channels > 0 else None

        return TTNNStochasticDurationPredictor(
            in_channels=in_channels,
            filter_channels=filter_channels,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
            n_flows=n_flows,
            gin_channels=gin_channels,
            log_flow=Log(),
            flows=flows,
            post_pre_weight=state_dict.get(f"{prefix}.post_pre.weight"),
            post_pre_bias=state_dict.get(f"{prefix}.post_pre.bias"),
            post_proj_weight=state_dict.get(f"{prefix}.post_proj.weight"),
            post_proj_bias=state_dict.get(f"{prefix}.post_proj.bias"),
            post_convs=post_convs,
            post_flows=post_flows,
            pre_weight=state_dict.get(f"{prefix}.pre.weight"),
            pre_bias=state_dict.get(f"{prefix}.pre.bias"),
            proj_weight=state_dict.get(f"{prefix}.proj.weight"),
            proj_bias=state_dict.get(f"{prefix}.proj.bias"),
            convs=convs,
            cond_weight=cond_weight,
            cond_bias=cond_bias,
            device=device,
        )

    @classmethod
    def _build_dds_conv(
        cls,
        state_dict: dict,
        prefix: str,
        channels: int,
        kernel_size: int,
        n_layers: int,
        p_dropout: float,
        device: Optional[Any],
    ) -> "DDSConv":
        """Build DDSConv from state dict."""
        from models.demos.openvoice.tt.duration_predictor import DDSConv

        convs_sep_weights = []
        convs_sep_biases = []
        convs_1x1_weights = []
        convs_1x1_biases = []
        norms_1_weights = []
        norms_1_biases = []
        norms_2_weights = []
        norms_2_biases = []

        for i in range(n_layers):
            convs_sep_weights.append(state_dict.get(f"{prefix}.convs_sep.{i}.weight"))
            convs_sep_biases.append(state_dict.get(f"{prefix}.convs_sep.{i}.bias"))
            convs_1x1_weights.append(state_dict.get(f"{prefix}.convs_1x1.{i}.weight"))
            convs_1x1_biases.append(state_dict.get(f"{prefix}.convs_1x1.{i}.bias"))
            norms_1_weights.append(state_dict.get(f"{prefix}.norms_1.{i}.gamma"))
            norms_1_biases.append(state_dict.get(f"{prefix}.norms_1.{i}.beta"))
            norms_2_weights.append(state_dict.get(f"{prefix}.norms_2.{i}.gamma"))
            norms_2_biases.append(state_dict.get(f"{prefix}.norms_2.{i}.beta"))

        return DDSConv(
            channels=channels,
            kernel_size=kernel_size,
            n_layers=n_layers,
            p_dropout=p_dropout,
            convs_sep_weights=convs_sep_weights,
            convs_sep_biases=convs_sep_biases,
            convs_1x1_weights=convs_1x1_weights,
            convs_1x1_biases=convs_1x1_biases,
            norms_1_weights=norms_1_weights,
            norms_1_biases=norms_1_biases,
            norms_2_weights=norms_2_weights,
            norms_2_biases=norms_2_biases,
            device=device,
        )

    @classmethod
    def _build_conv_flow(
        cls,
        state_dict: dict,
        prefix: str,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        n_layers: int,
        device: Optional[Any],
    ) -> "ConvFlow":
        """Build ConvFlow from state dict."""
        from models.demos.openvoice.tt.duration_predictor import ConvFlow

        convs = cls._build_dds_conv(state_dict, f"{prefix}.convs", filter_channels, kernel_size, n_layers, 0.0, device)

        return ConvFlow(
            in_channels=in_channels,
            filter_channels=filter_channels,
            kernel_size=kernel_size,
            n_layers=n_layers,
            pre_weight=state_dict.get(f"{prefix}.pre.weight"),
            pre_bias=state_dict.get(f"{prefix}.pre.bias"),
            proj_weight=state_dict.get(f"{prefix}.proj.weight"),
            proj_bias=state_dict.get(f"{prefix}.proj.bias"),
            convs=convs,
            device=device,
        )
