# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pure-torch MiMo-V2.6 audio input path: log-mel [T, 128] -> LLM input embeddings [n_tokens, 4096].

Mirrors the HF remote code ``modeling_mimo_v2.py`` (``tokenize_audio_batch``, ``AudioTokenizerEncoder``,
``ResidualVectorQuantizer``, ``MiMoAudioEncoder.get_audio_feature``) op for op, without transformers: in fp32 the
codes and the embeddings are bit-identical to the HF modules (tests/test_audio_reference_host.py). One audio item:

  waveform (mono, 24 kHz) -> log-magnitude mel [T, 128], 100 frames/s (vLLM processor; ``log_mel_spectrogram``)
  -> 6000-frame (60 s) segments, encoded as one zero-padded batch
  -> conv1 (k 3) + GELU, conv2 (k 3, stride 2) + GELU                   50 Hz
  -> 24 pre-LN layers (d 1024, 16 heads, RoPE theta 1e4 restarting per segment, causal; layers 0, 2, .. also
     window 128) + skip connection of layer 3's output, LayerNorm
  -> avg-pool conv (k 2, stride 2) + GELU + LayerNorm                   25 Hz
  -> 20-stage residual VQ in fp32 (codebooks 1024, 1024, 256, 128 x 17) -> codes [n, 20]
  -> pad to a multiple of 4 frames by repeating the last frame -> groups [G, 4, 20]
  -> sum of the 20 code embeddings (``speech_embeddings.i``) [G, 4, 1024]
  -> 6-layer Qwen2 transformer, bidirectional within each group (RoPE theta 640000, positions 0..3), RMSNorm
  -> [G, 4096] -> Linear 4096->16384, GELU, Linear 16384->4096 -> [G, 4096]: 6.25 tokens per second of audio

``dtype`` selects the compute dtype of the convolutional / transformer parts (weights ship in bf16, so fp32 is exact
upcasting). The RVQ always runs in fp32, as in HF. ``bf16_buffers=True`` emulates HF's ``load_audio_tokenizer`` and
vLLM's ``_load_audio_tokenizer``, which cast the whole tokenizer module to bf16: that also rounds its fp32 buffers,
the RVQ codebooks (``encode`` casts the quantizer back to fp32 afterwards, the rounding stays) and the RoPE
``inv_freq``.

vLLM 0.31.0 (``mimo_audio.py`` ``MimoAudioEncoder.get_audio_feature``, processor ``MiMoVLProcessor``) vs HF:

* Precision: vLLM runs the tokenizer encoder in bf16 with bf16-rounded buffers (as above) and the patch encoder in
  the model dtype (bf16). We follow fp32 (the HF math; fp64 gives the same codes). bf16 changes at least one of the
  20 codes in ~70 % of the 25 Hz frames (codebook 0: ~2-3 %) and the embeddings' PCC vs fp32 to ~0.98.
* Encoder attention: vLLM ``flash_attn_varlen_func(causal=True, window_size=(128, 0))`` on even layers, causal
  on odd; HF an additive mask (causal, |i - j| <= 128) per segment. Same keys (i - 128 .. i).
* Patch encoder attention: both call transformers' ``Qwen2Model(..., is_causal=False)``; with transformers 5.12.1
  (sdpa and eager alike) that is bidirectional within the 4-frame group, which we follow (``input_full_attention``).
* Weight names: HF ``speech_embeddings.i``, vLLM ``audio_encoder.speech_embeddings.i`` (its weight mapper).
* Token count: the processor's ``audio_token_lens`` is computed on the whole mel, the model's rows per 6000-frame
  segment; they agree for every length because 6000 is a multiple of 4 (``audio_token_len``, ``num_codes``).
* Last group: HF and vLLM both pad the codes to a multiple of 4 frames by repeating the last frame.
* Batching: vLLM tokenizes all audio items of an encoder batch in one ``tokenize_audio_batch`` call. In a padded
  batch a segment shorter than the longest one sees padding rows that depend on the batch (conv1 outputs past its
  end, and the copied last row before the avg-pool), so an item's last code frame can change with what it is
  batched with. ``MiMoAudioReference`` (and the host backend) encode what they are given; one item per call is
  vLLM with one audio item per encoder batch.
"""

import json
import math
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from safetensors import safe_open

AUDIO_TOKENIZER_DIR = "audio_tokenizer"


# ---------------------------------------------------------------------------
# Config and token counts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MiMoAudioConfig:
    # Mel front end (config.json processor_config; vLLM's processor defaults are the same values)
    sampling_rate: int = 24000
    n_fft: int = 960
    hop_length: int = 240
    win_length: int = 960
    n_mels: int = 128
    # Audio tokenizer encoder (audio_tokenizer/config.json)
    d_model: int = 1024
    encoder_layers: int = 24
    encoder_heads: int = 16
    encoder_ffn: int = 4096
    kernel_size: int = 3
    stride_size: int = 2
    avg_pooler: int = 2
    skip_layer_id: int = 3
    window: int = 128
    swa_per_block: int = 2
    hybrid_attention: bool = True
    encoder_causal: bool = True
    encoder_rope_theta: float = 10000.0
    codebook_sizes: tuple = (1024, 1024, 256) + (128,) * 17
    segment_size: int = 6000
    group_by_length: int = 256000
    # Patch encoder (config.json audio_config)
    audio_channels: int = 20
    group_size: int = 4
    local_dim: int = 1024
    local_layers: int = 6
    local_heads: int = 16
    local_ffn: int = 4096
    local_rope_theta: float = 640000.0
    local_rms_eps: float = 1e-6
    local_causal: bool = False  # input_full_attention = True
    add_post_norm: bool = True
    out_hidden_size: int = 4096

    @classmethod
    def from_model_path(cls, model_path):
        with open(os.path.join(model_path, "config.json")) as f:
            cfg = json.load(f)
        with open(os.path.join(model_path, AUDIO_TOKENIZER_DIR, "config.json")) as f:
            tok = json.load(f)
        ac, pc = cfg["audio_config"], cfg.get("processor_config", {})
        assert (
            tok["position_embedding_type"] == "rope" and tok["rope_type"] == "default" and tok["ln_type"] == "LayerNorm"
        )
        assert not tok["scale_embedding"] and tok["activation_function"] == "gelu"
        assert tok["encoder_attn_window_size"][1] == 0 and ac.get("projection_layers", 2) == 2
        assert float(ac.get("partial_rotary_factor", 1.0)) == 1.0
        assert ac["input_local_head_dim"] * ac["input_local_attn_heads"] == ac["input_local_dim"]
        assert ac["input_local_hidden_dropout"] == 0.0
        codebooks = list(tok["codebook_size"])
        codebooks += [codebooks[-1]] * (tok["num_quantizers"] - len(codebooks))
        return cls(
            sampling_rate=pc.get("audio_sampling_rate", tok["sampling_rate"]),
            n_fft=pc.get("audio_nfft", tok["nfft"]),
            hop_length=pc.get("audio_hop_length", tok["hop_length"]),
            win_length=pc.get("audio_window_size", tok["window_size"]),
            n_mels=pc.get("audio_n_mels", tok["n_mels"]),
            d_model=tok["d_model"],
            encoder_layers=tok["encoder_layers"],
            encoder_heads=tok["encoder_attention_heads"],
            encoder_ffn=tok["encoder_ffn_dim"],
            kernel_size=tok["kernel_size"],
            stride_size=tok["stride_size"],
            avg_pooler=tok["avg_pooler"],
            skip_layer_id=tok["encoder_skip_layer_id"],
            window=tok["encoder_attn_window_size"][0],
            swa_per_block=tok["swa_per_block"],
            hybrid_attention=tok["hybrid_attention"],
            encoder_causal=tok["encoder_causal"],
            encoder_rope_theta=float(tok["rope_theta"]),
            codebook_sizes=tuple(codebooks[: tok["num_quantizers"]]),
            segment_size=ac.get("audio_segment_size", 6000),
            audio_channels=ac["audio_channels"],
            group_size=ac["group_size"],
            local_dim=ac["input_local_dim"],
            local_layers=ac["input_local_layers"],
            local_heads=ac["input_local_attn_heads"],
            local_ffn=ac["input_local_intermediate_size"],
            local_rope_theta=float(ac.get("rope_theta", 640000.0)),
            local_causal=not ac.get("input_full_attention", True),
            add_post_norm=ac.get("add_post_norm", True),
            out_hidden_size=ac["out_hidden_size"],
        )

    def window_for_layer(self, i):
        """Left attention window of encoder layer ``i`` (-1 = unlimited); the right window is 0 (causal)."""
        if not self.hybrid_attention or i % self.swa_per_block < self.swa_per_block - 1:
            return self.window
        return -1


def num_mel_frames(num_samples, cfg=MiMoAudioConfig()):
    """Frames of the processor's centered STFT for ``num_samples`` 24 kHz samples."""
    return num_samples // cfg.hop_length + 1


def encoder_output_length(mel_len, cfg=MiMoAudioConfig()):
    """Frames after the two convolutions (``AudioTokenizerEncoder.get_output_length``), int or tensor."""
    tgt_len = mel_len + 3 - cfg.kernel_size
    return (tgt_len + 2 - cfg.kernel_size) // cfg.stride_size + 1


def pooled_length(n, cfg=MiMoAudioConfig()):
    return n // cfg.avg_pooler + (n % cfg.avg_pooler != 0)


def segment_lengths(mel_len, segment_size=6000):
    segs = [segment_size] * (mel_len // segment_size)
    if mel_len % segment_size:
        segs.append(mel_len % segment_size)
    return segs


def num_codes(mel_len, cfg=MiMoAudioConfig()):
    """Code frames (25 Hz) of one item: the sum over its 6000-frame segments, as ``tokenize_audio_batch`` splits."""
    return sum(
        int(pooled_length(encoder_output_length(s, cfg), cfg)) for s in segment_lengths(mel_len, cfg.segment_size)
    )


def audio_token_len(mel_len, cfg=MiMoAudioConfig()):
    """LLM placeholder tokens of one item, as vLLM's ``MiMoVLProcessor.preprocess_audio`` counts them (on the whole
    mel, without segmenting; ``num_codes`` agrees because 6000 is a multiple of 4)."""
    n = int(encoder_output_length(mel_len, cfg))
    n = n // cfg.avg_pooler + int(n % cfg.avg_pooler != 0)
    return math.ceil(n / cfg.group_size)


# ---------------------------------------------------------------------------
# Mel front end (torchaudio.transforms.MelSpectrogram defaults, as the vLLM processor builds it)
# ---------------------------------------------------------------------------


def _hz_to_mel_htk(freq):
    return 2595.0 * math.log10(1.0 + (freq / 700.0))


def mel_filterbank(cfg=MiMoAudioConfig()):
    """torchaudio.functional.melscale_fbanks(n_fft // 2 + 1, 0, sr // 2, n_mels, sr, norm=None, mel_scale="htk")."""
    n_freqs = cfg.n_fft // 2 + 1
    all_freqs = torch.linspace(0, cfg.sampling_rate // 2, n_freqs)
    m_pts = torch.linspace(_hz_to_mel_htk(0.0), _hz_to_mel_htk(float(cfg.sampling_rate // 2)), cfg.n_mels + 2)
    f_pts = 700.0 * (10.0 ** (m_pts / 2595.0) - 1.0)
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)
    down = (-1.0 * slopes[:, :-2]) / f_diff[:-1]
    up = slopes[:, 2:] / f_diff[1:]
    return torch.max(torch.zeros(1), torch.min(down, up))


def log_mel_spectrogram(waveform, cfg=MiMoAudioConfig()):
    """[samples] mono fp32 at ``cfg.sampling_rate`` -> [frames, n_mels] fp32: the processor's
    ``log(clip(MelSpectrogram(..., power=1.0, center=True)(w), 1e-7))``: hann window, centered reflect-padded
    STFT, magnitude (not power), HTK mel scale without area normalization, natural log."""
    assert waveform.dim() == 1, "mono waveform expected (vLLM averages channels when it decodes the file)"
    spec = torch.stft(
        waveform.float()[None],
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window=torch.hann_window(cfg.win_length),
        center=True,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    magnitude = spec.abs()  # [1, freq, frames]; torchaudio's power=1.0 path
    mel = torch.matmul(magnitude.transpose(-1, -2), mel_filterbank(cfg)).transpose(-1, -2)
    return torch.log(torch.clip(mel, min=1e-7))[0].transpose(0, 1)


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


def load_audio_weights(model_path, tokenizer=True, patch=True):
    """Audio-input tensors as stored: ``encoder.*`` of the tokenizer (bf16; RVQ codebooks fp32; ``tokenizer``) and
    the main checkpoint's ``speech_embeddings.*`` / ``audio_encoder.*`` (bf16; ``patch``). The vocoder
    (``decoder.*``) is skipped."""
    out = {}
    if tokenizer:
        with safe_open(os.path.join(model_path, AUDIO_TOKENIZER_DIR, "model.safetensors"), framework="pt") as f:
            for k in f.keys():
                if k.startswith("encoder."):
                    out[k] = f.get_tensor(k)
    if patch:
        with open(os.path.join(model_path, "model.safetensors.index.json")) as f:
            weight_map = json.load(f)["weight_map"]
        by_file = {}
        for k, fname in weight_map.items():
            if k.startswith(("speech_embeddings.", "audio_encoder.")):
                by_file.setdefault(fname, []).append(k)
        for fname, keys in by_file.items():
            with safe_open(os.path.join(model_path, fname), framework="pt") as f:
                for k in keys:
                    out[k] = f.get_tensor(k)
    return out


# ---------------------------------------------------------------------------
# Audio tokenizer encoder: mel -> RVQ codes
# ---------------------------------------------------------------------------


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _rope_cos_sin(inv_freq, position_ids, dtype):
    """AudioTokenizerRotaryEmbedding.forward: fp32 angles, cos / sin cast to the compute dtype."""
    freqs = (inv_freq[:, None].float() @ position_ids[None, :].float()).transpose(0, 1)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _sequence_mask(lengths, tgt_len):
    bsz = lengths.shape[0]
    mask = torch.lt(torch.arange(0, tgt_len), lengths.reshape(bsz, 1)).view(bsz, tgt_len, 1)
    unpacking_index = torch.cumsum(mask.to(torch.int64).view(-1), dim=0) - 1
    return mask, unpacking_index


def _attn_mask(seq_len, causal, window, dtype):
    """AudioTokenizerAttention._build_attn_mask: additive mask, -inf above the diagonal and beyond |i - j| > window."""
    if not causal and window <= 0:
        return None
    mask = torch.zeros(seq_len, seq_len, dtype=dtype)
    if causal:
        mask = mask + torch.triu(torch.full((seq_len, seq_len), float("-inf"), dtype=dtype), diagonal=1)
    if window > 0:
        idx = torch.arange(seq_len)
        mask = mask.masked_fill((idx[:, None] - idx[None, :]).abs() > window, float("-inf"))
    return mask


class AudioTokenizerEncoder:
    """``audio_tokenizer`` encoder + RVQ (codes only). ``weights``: ``load_audio_weights`` output."""

    def __init__(self, weights, cfg=MiMoAudioConfig(), dtype=torch.float32, bf16_buffers=False):
        self.cfg, self.dtype = cfg, dtype
        w = {k[len("encoder.") :]: v for k, v in weights.items() if k.startswith("encoder.")}
        cast = lambda t: t.to(dtype).contiguous()
        self.conv1_w, self.conv1_b = cast(w["conv1.weight"]), cast(w["conv1.bias"])
        self.conv2_w, self.conv2_b = cast(w["conv2.weight"]), cast(w["conv2.bias"])
        self.layers = []
        for i in range(cfg.encoder_layers):
            p = f"layers.{i}."
            self.layers.append(
                {
                    "ln1_w": cast(w[p + "self_attn_layer_norm.weight"]),
                    "ln1_b": cast(w[p + "self_attn_layer_norm.bias"]),
                    "q_w": cast(w[p + "self_attn.q_proj.weight"]),
                    "q_b": cast(w[p + "self_attn.q_proj.bias"]),
                    "k_w": cast(w[p + "self_attn.k_proj.weight"]),
                    "v_w": cast(w[p + "self_attn.v_proj.weight"]),
                    "v_b": cast(w[p + "self_attn.v_proj.bias"]),
                    "o_w": cast(w[p + "self_attn.out_proj.weight"]),
                    "o_b": cast(w[p + "self_attn.out_proj.bias"]),
                    "ln2_w": cast(w[p + "final_layer_norm.weight"]),
                    "ln2_b": cast(w[p + "final_layer_norm.bias"]),
                    "fc1_w": cast(w[p + "fc1.weight"]),
                    "fc1_b": cast(w[p + "fc1.bias"]),
                    "fc2_w": cast(w[p + "fc2.weight"]),
                    "fc2_b": cast(w[p + "fc2.bias"]),
                    "window": cfg.window_for_layer(i),
                }
            )
        self.ln_w, self.ln_b = cast(w["layer_norm.weight"]), cast(w["layer_norm.bias"])
        self.down_w = cast(w["down_sample_layer.0.weight"])
        self.down_ln_w, self.down_ln_b = cast(w["down_sample_norm.weight"]), cast(w["down_sample_norm.bias"])
        # The RVQ runs in fp32 (fp64 for an fp64 encoder, a precision probe).
        self.rvq_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        self.codebooks = []
        for i in range(len(cfg.codebook_sizes)):
            cb = w[f"quantizer.vq.layers.{i}._codebook.embed"].float()
            if bf16_buffers:
                cb = cb.to(torch.bfloat16).float()
            cb = cb.to(self.rvq_dtype)
            assert cb.shape == (cfg.codebook_sizes[i], cfg.d_model)
            self.codebooks.append(cb.contiguous())
        head_dim = cfg.d_model // cfg.encoder_heads
        self.inv_freq = 1.0 / (cfg.encoder_rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        if bf16_buffers:
            self.inv_freq = self.inv_freq.to(torch.bfloat16)

    def _attention(self, q, k, v, window, masks):
        """One segment, [1, heads, n, hd]: SDPA with HF's additive mask (cached in ``masks`` across layers)."""
        key = (q.shape[2], window)
        if key not in masks:
            masks[key] = _attn_mask(q.shape[2], self.cfg.encoder_causal, window, q.dtype)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=masks[key], scale=q.shape[-1] ** -0.5)

    def _layer(self, h, L, cu_seqlens, cos, sin, masks):
        cfg = self.cfg
        total = h.shape[0]
        heads, hd = cfg.encoder_heads, cfg.d_model // cfg.encoder_heads
        residual = h
        h = F.layer_norm(h, (cfg.d_model,), L["ln1_w"], L["ln1_b"])
        q = F.linear(h, L["q_w"], L["q_b"]).view(total, heads, hd)
        k = F.linear(h, L["k_w"]).view(total, heads, hd)
        v = F.linear(h, L["v_w"], L["v_b"]).view(total, heads, hd)
        c, s = cos.unsqueeze(1), sin.unsqueeze(1)
        q, k = (q * c) + (_rotate_half(q) * s), (k * c) + (_rotate_half(k) * s)
        outs = []
        for i in range(len(cu_seqlens) - 1):
            a, b = int(cu_seqlens[i]), int(cu_seqlens[i + 1])
            qs, ks, vs = (t[a:b].transpose(0, 1).unsqueeze(0) for t in (q, k, v))
            outs.append(self._attention(qs, ks, vs, L["window"], masks).squeeze(0).transpose(0, 1))
        h = F.linear(torch.cat(outs, dim=0).reshape(total, cfg.d_model), L["o_w"], L["o_b"])
        h = residual + h
        residual = h
        h = F.layer_norm(h, (cfg.d_model,), L["ln2_w"], L["ln2_b"])
        h = F.linear(F.gelu(F.linear(h, L["fc1_w"], L["fc1_b"])), L["fc2_w"], L["fc2_b"])
        return residual + h

    def features(self, features, input_lens):
        """Packed mel frames [sum(input_lens), n_mels] of a batch of segments -> packed 25 Hz hidden states
        [sum(out_lens), d] (compute dtype) and ``out_lens``. ``AudioTokenizerEncoder.encode`` up to the quantizer."""
        cfg = self.cfg
        out_len = encoder_output_length(input_lens, cfg)
        # _at_unpack_hidden_states: [bsz, max_len, n_mels], zeros past each segment's end.
        bsz, max_len = input_lens.shape[0], int(input_lens.max())
        mask, unpack = _sequence_mask(input_lens, max_len)
        x = torch.where(mask, torch.index_select(features, 0, unpack).view(bsz, max_len, -1), 0)
        x = x.transpose(1, 2).to(self.dtype)
        x = F.gelu(F.conv1d(x, self.conv1_w, self.conv1_b, padding=1))
        x = F.gelu(F.conv1d(x, self.conv2_w, self.conv2_b, stride=cfg.stride_size, padding=1))
        x = x.permute(0, 2, 1)
        tgt_len = x.shape[1]
        offsets = torch.repeat_interleave(
            torch.cat([torch.zeros(1, dtype=out_len.dtype), out_len[:-1].cumsum(0)]), out_len
        )
        position_ids = torch.arange(0, int(out_len.sum())) - offsets
        cos, sin = _rope_cos_sin(self.inv_freq, position_ids, self.dtype)
        mask, unpack = _sequence_mask(out_len, tgt_len)
        h = torch.masked_select(x, mask).view(int(out_len.sum()), cfg.d_model)
        cu_seqlens = F.pad(torch.cumsum(out_len, dim=0), (1, 0), "constant", 0)
        skip, masks = 0.0, {}
        for i, L in enumerate(self.layers):
            h = self._layer(h, L, cu_seqlens, cos, sin, masks)
            if i == cfg.skip_layer_id - 1:
                skip = h.clone()
        h = h + skip
        h = F.layer_norm(h, (cfg.d_model,), self.ln_w, self.ln_b)
        # Avg-pool: back to [bsz, tgt_len, d]; rows past a segment's end repeat its last row (unpacking index),
        # an odd tgt_len gets one zero row.
        h = torch.index_select(h, 0, unpack).view(bsz, tgt_len, cfg.d_model)
        if tgt_len % cfg.avg_pooler:
            h = F.pad(h, (0, 0, 0, cfg.avg_pooler - tgt_len % cfg.avg_pooler), mode="constant", value=0.0)
        h = F.gelu(F.conv1d(h.transpose(1, 2), self.down_w, stride=cfg.avg_pooler))
        out_len = out_len // cfg.avg_pooler + (out_len % cfg.avg_pooler != 0).int()
        h = h.transpose(1, 2)
        mask, _ = _sequence_mask(out_len, h.shape[1])
        h = torch.masked_select(h, mask).view(int(out_len.sum()), cfg.d_model)
        h = F.layer_norm(h, (cfg.d_model,), self.down_ln_w, self.down_ln_b)
        return h, out_len

    def quantize(self, h):
        """ResidualVectorQuantization.encode in fp32: [n, d] -> codes [n_q, n] (int64)."""
        residual = h.to(self.rvq_dtype)
        codes = []
        for embed in self.codebooks:
            e = embed.t()
            dist = -(residual.pow(2).sum(1, keepdim=True) - 2 * residual @ e + e.pow(2).sum(0, keepdim=True))
            idx = dist.max(dim=-1).indices
            codes.append(idx)
            residual = residual - F.embedding(idx, embed)
        return torch.stack(codes)

    def encode(self, features, input_lens):
        h, out_len = self.features(features, input_lens)
        return self.quantize(h), out_len


@torch.no_grad()
def tokenize_audio_batch(mels, encoder, segment_size=None):
    """HF / vLLM ``tokenize_audio_batch``: list of mels [T_i, n_mels] -> list of codes [n_i, n_q] (int64).

    Every item is cut into ``segment_size``-frame segments; all segments of all items are encoded together in
    groups of at most ``group_by_length`` frames, as one padded batch per group. Batching is not numerically
    neutral (module docstring): it can change an item's last code frame."""
    cfg = encoder.cfg
    segment_size = segment_size or cfg.segment_size
    if not mels:
        return []
    segs_per_mel = [segment_lengths(m.shape[0], segment_size) for m in mels]
    lens = torch.tensor([s for segs in segs_per_mel for s in segs], dtype=torch.long)
    features = torch.cat(list(mels), dim=0)
    # _at_group_by_length
    split_points, current = [], 0
    for i, n in enumerate(lens.tolist()):
        if current + n > cfg.group_by_length and current > 0:
            split_points.append(i)
            current = n
        else:
            current += n
    sizes, prev = [], 0
    for p in split_points:
        sizes.append(p - prev)
        prev = p
    if prev < len(lens):
        sizes.append(len(lens) - prev)
    len_groups = torch.split(lens, sizes)
    feat_groups = torch.split(features, [int(g.sum()) for g in len_groups])
    parts = [encoder.encode(f, l)[0] for f, l in zip(feat_groups, len_groups)]
    codes = torch.cat(parts, dim=-1).transpose(0, 1)
    code_lengths = [sum(int(pooled_length(encoder_output_length(s, cfg), cfg)) for s in segs) for segs in segs_per_mel]
    return list(torch.split(codes, code_lengths))


# ---------------------------------------------------------------------------
# Patch encoder: codes -> LLM embeddings
# ---------------------------------------------------------------------------


def pad_and_group_codes(codes, cfg=MiMoAudioConfig()):
    """_pad_and_group_audio_codes: [T, >= C] -> [ceil(T / 4), 4, C], padded by repeating the last frame."""
    codes = codes[:, : cfg.audio_channels]
    T = codes.shape[0]
    padded = ((T + cfg.group_size - 1) // cfg.group_size) * cfg.group_size
    if padded > T:
        codes = torch.cat([codes, codes[-1:].expand(padded - T, -1)], dim=0)
    return codes.reshape(padded // cfg.group_size, cfg.group_size, cfg.audio_channels)


def _rms_norm(x, w, eps):
    dtype = x.dtype
    x = x.to(torch.float32)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return w * x.to(dtype)


class AudioPatchEncoder:
    """``speech_embeddings`` + ``audio_encoder`` (``MiMoAudioEncoder``): grouped codes [G, 4, 20] -> [G, 4096]."""

    def __init__(self, weights, cfg=MiMoAudioConfig(), dtype=torch.float32):
        self.cfg, self.dtype = cfg, dtype
        cast = lambda t: t.to(dtype).contiguous()
        self.embeddings = [cast(weights[f"speech_embeddings.{i}.weight"]) for i in range(cfg.audio_channels)]
        p = "audio_encoder.input_local_transformer."
        self.layers = []
        for i in range(cfg.local_layers):
            q = f"{p}layers.{i}."
            self.layers.append(
                {
                    "in_norm": cast(weights[q + "input_layernorm.weight"]),
                    "q_w": cast(weights[q + "self_attn.q_proj.weight"]),
                    "q_b": cast(weights[q + "self_attn.q_proj.bias"]),
                    "k_w": cast(weights[q + "self_attn.k_proj.weight"]),
                    "k_b": cast(weights[q + "self_attn.k_proj.bias"]),
                    "v_w": cast(weights[q + "self_attn.v_proj.weight"]),
                    "v_b": cast(weights[q + "self_attn.v_proj.bias"]),
                    "o_w": cast(weights[q + "self_attn.o_proj.weight"]),
                    "post_norm": cast(weights[q + "post_attention_layernorm.weight"]),
                    "gate_w": cast(weights[q + "mlp.gate_proj.weight"]),
                    "up_w": cast(weights[q + "mlp.up_proj.weight"]),
                    "down_w": cast(weights[q + "mlp.down_proj.weight"]),
                }
            )
        self.norm = cast(weights[p + "norm.weight"]) if cfg.add_post_norm else None
        self.proj0 = cast(weights["audio_encoder.projection.mlp.0.weight"])
        self.proj2 = cast(weights["audio_encoder.projection.mlp.2.weight"])
        hd = cfg.local_dim // cfg.local_heads
        self.inv_freq = 1.0 / (
            cfg.local_rope_theta ** (torch.arange(0, hd, 2, dtype=torch.int64).to(dtype=torch.float) / hd)
        )

    def embed_codes(self, grouped):
        """MiMoAudioEncoder._apply_speech_embeddings: sum of the 20 code embeddings, in channel order."""
        out = torch.zeros((grouped.shape[0], self.cfg.group_size, self.cfg.local_dim), dtype=self.dtype)
        for i in range(self.cfg.audio_channels):
            out.add_(F.embedding(grouped[:, :, i].long(), self.embeddings[i]))
        return out

    def local_transformer(self, x):
        """Qwen2Model(inputs_embeds=x, is_causal=False) (transformers 5.x sdpa path): [G, 4, d] -> [G, 4, d]."""
        cfg = self.cfg
        G, S, _ = x.shape
        heads, hd = cfg.local_heads, cfg.local_dim // cfg.local_heads
        pos = torch.arange(S)[None]
        freqs = (self.inv_freq[None, :, None].float() @ pos[:, None, :].float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos().to(x.dtype)[:, None], emb.sin().to(x.dtype)[:, None]  # [1, 1, S, hd]
        h = x
        for L in self.layers:
            residual = h
            t = _rms_norm(h, L["in_norm"], cfg.local_rms_eps)
            q = F.linear(t, L["q_w"], L["q_b"]).view(G, S, heads, hd).transpose(1, 2)
            k = F.linear(t, L["k_w"], L["k_b"]).view(G, S, heads, hd).transpose(1, 2)
            v = F.linear(t, L["v_w"], L["v_b"]).view(G, S, heads, hd).transpose(1, 2)
            q, k = (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)
            o = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, scale=hd**-0.5, is_causal=cfg.local_causal, enable_gqa=True
            )
            o = o.transpose(1, 2).contiguous().reshape(G, S, -1).contiguous()
            h = residual + F.linear(o, L["o_w"])
            residual = h
            t = _rms_norm(h, L["post_norm"], cfg.local_rms_eps)
            h = residual + F.linear(F.silu(F.linear(t, L["gate_w"])) * F.linear(t, L["up_w"]), L["down_w"])
        if self.norm is not None:
            h = _rms_norm(h, self.norm, cfg.local_rms_eps)
        return h

    def project(self, h):
        """AudioProjection: [G, 4, d] -> [G, 4 * d] -> Linear, GELU, Linear -> [G, out_hidden_size]."""
        h = h.reshape(h.shape[0], -1)
        return F.linear(F.gelu(F.linear(h, self.proj0)), self.proj2)

    @torch.no_grad()
    def __call__(self, grouped):
        return self.project(self.local_transformer(self.embed_codes(grouped)))


# ---------------------------------------------------------------------------
# Whole path
# ---------------------------------------------------------------------------


class MiMoAudioReference:
    """mels -> codes -> embeddings, as ``MiMoAudioEncoder.get_audio_feature`` (HF) / ``MimoAudioEncoder``
    (vLLM) for one batch of items. fp32 by default: the reference."""

    def __init__(
        self,
        weights,
        cfg=MiMoAudioConfig(),
        encoder_dtype=torch.float32,
        patch_dtype=torch.float32,
        bf16_buffers=False,
    ):
        self.cfg = cfg
        self.encoder = AudioTokenizerEncoder(weights, cfg, encoder_dtype, bf16_buffers)
        self.patch = AudioPatchEncoder(weights, cfg, patch_dtype)

    @classmethod
    def from_model_path(cls, model_path, **kwargs):
        return cls(load_audio_weights(model_path), MiMoAudioConfig.from_model_path(model_path), **kwargs)

    @torch.no_grad()
    def codes(self, mels):
        """list of mels [T_i, 128] -> list of codes [n_i, 20] (int64, 25 Hz)."""
        return tokenize_audio_batch(mels, self.encoder)

    @torch.no_grad()
    def embed_codes(self, codes_list):
        """list of codes [n_i, 20] -> (embeddings [sum G_i, 4096] in the patch dtype, token counts G_i)."""
        grouped = [pad_and_group_codes(c, self.cfg) for c in codes_list]
        return self.patch(torch.cat(grouped, dim=0)), [g.shape[0] for g in grouped]

    @torch.no_grad()
    def embed(self, mels):
        """list of mels [T_i, 128] -> (embeddings [sum G_i, 4096], token counts G_i == audio_token_len(T_i))."""
        return self.embed_codes(self.codes(mels))
