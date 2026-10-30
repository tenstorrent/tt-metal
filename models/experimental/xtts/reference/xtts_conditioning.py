# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch import einsum, nn
from torch.nn import functional as F

from models.experimental.xtts.config import (  # noqa: F401
    COND_CHUNK_FRAMES,
    COND_CHUNK_SAMPLES,
    COND_CHUNK_SEC,
    COND_MAX_FRAMES,
    COND_MAX_SAMPLES,
    COND_MIN_CHUNK_FRAMES,
    COND_MIN_CHUNK_SAMPLES,
    COND_N_MELS as N_MELS,
    COQUI_TESTS_WAV_URL,
    GPT_COND_CHUNK_SEC,
    GPT_COND_LEN_SEC,
    HF_REPO_ID,
    HF_REVISION,
    HIDDEN_SIZE,
    MEL_FMAX,
    MEL_FMIN,
    MEL_HOP,
    MEL_N_FFT,
    MEL_SR,
    MEL_WIN,
    NUM_ATTN_HEADS,
    NUM_LATENTS,
    PERCEIVER_DEPTH,
    PERCEIVER_FF_MULT,
    PERCEIVER_HEAD_DIM,
    PERCEIVER_HEADS,
)


def chunk_cond_mel(mel, chunk_frames=COND_CHUNK_FRAMES, min_frames=COND_MIN_CHUNK_FRAMES, max_frames=COND_MAX_FRAMES):
    """Split conditioning mel into fixed-size chunks, dropping short tails."""
    mel = mel[..., :max_frames]
    s = mel.shape[-1]
    if s <= chunk_frames:
        return [mel]
    chunks = [mel[..., i : i + chunk_frames] for i in range(0, s, chunk_frames)]
    kept = [c for c in chunks if c.shape[-1] >= min_frames]
    return kept or [mel]


def chunk_wav(wav, chunk_samples=COND_CHUNK_SAMPLES, min_samples=COND_MIN_CHUNK_SAMPLES, max_samples=COND_MAX_SAMPLES):
    """Split waveform into fixed-size chunks, dropping short tails."""
    wav = wav[..., :max_samples]
    length = wav.shape[-1]
    if length <= chunk_samples:
        return [wav]
    chunks = [wav[..., i : i + chunk_samples] for i in range(0, length, chunk_samples)]
    kept = [c for c in chunks if c.shape[-1] >= min_samples]
    return kept or [wav]


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        """Apply GroupNorm in float32 then cast back to input dtype."""
        return super().forward(x.float()).type(x.dtype)


def normalization(channels):
    """Create GroupNorm32 with a channel-compatible group count."""
    groups = 32
    if channels <= 16:
        groups = 8
    elif channels <= 64:
        groups = 16
    while channels % groups != 0:
        groups = int(groups / 2)
    assert groups > 2
    return GroupNorm32(groups, channels)


class QKVAttention(nn.Module):
    def __init__(self, n_heads):
        """Store head count for packed QKV attention."""
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """Compute multi-head attention from packed QKV channels."""
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1):
        """Build a residual QKV attention block over 1D features."""
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = normalization(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)
        self.x_proj = nn.Identity()
        self.proj_out = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        """Apply normed self-attention with residual projection."""
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        x = self.norm(x)
        qkv = self.qkv(x)
        h = self.attention(qkv)
        h = self.proj_out(h)
        xp = self.x_proj(x)
        return (xp + h).reshape(b, xp.shape[1], *spatial)


class ConditioningEncoder(nn.Module):
    def __init__(self, spec_dim, embedding_dim, attn_blocks=6, num_attn_heads=NUM_ATTN_HEADS):
        """Build mel-to-embedding conv plus stacked attention blocks."""
        super().__init__()
        self.init = nn.Conv1d(spec_dim, embedding_dim, kernel_size=1)
        self.attn = nn.Sequential(*[AttentionBlock(embedding_dim, num_attn_heads) for _ in range(attn_blocks)])
        self.dim = embedding_dim

    def forward(self, x):
        """Encode mel spectrogram into conditioning embeddings."""
        return self.attn(self.init(x))


def _exists(x):
    """Return True if value is not None."""
    return x is not None


class RMSNorm(nn.Module):
    def __init__(self, dim, scale=True):
        """Initialize RMSNorm with optional learned gamma."""
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim)) if scale else None

    def forward(self, x):
        """Normalize last dim by RMS and apply scale/gamma."""
        gamma = self.gamma if _exists(self.gamma) else 1
        return F.normalize(x, dim=-1) * self.scale * gamma


class GEGLU(nn.Module):
    def forward(self, x):
        """Apply gated GELU activation on split channels."""
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


def FeedForward(dim, mult=4):
    """Build a GEGLU feed-forward Sequential for the given dim."""
    dim_inner = int(dim * mult * 2 / 3)
    return nn.Sequential(nn.Linear(dim, dim_inner * 2), GEGLU(), nn.Linear(dim_inner, dim))


class PerceiverAttention(nn.Module):
    def __init__(self, dim, dim_head=PERCEIVER_HEAD_DIM, heads=PERCEIVER_HEADS, cross_attn_include_queries=True):
        """Build cross-attention over latents and optional query context."""
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.cross_attn_include_queries = cross_attn_include_queries
        dim_inner = dim_head * heads
        self.to_q = nn.Linear(dim, dim_inner, bias=False)
        self.to_kv = nn.Linear(dim, dim_inner * 2, bias=False)
        self.to_out = nn.Linear(dim_inner, dim, bias=False)

    def _split(self, t):
        """Reshape projected tensor into multi-head layout."""
        b, n, _ = t.shape
        return t.reshape(b, n, self.heads, self.dim_head).permute(0, 2, 1, 3)

    def forward(self, x, context):
        """Cross-attend latents to context and project the output."""
        if self.cross_attn_include_queries:
            context = torch.cat((x, context), dim=-2)
        q, k, v = self.to_q(x), *self.to_kv(context).chunk(2, dim=-1)
        q, k, v = self._split(q), self._split(k), self._split(v)
        scale = self.dim_head**-0.5
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * scale
        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        b, h, n, d = out.shape
        out = out.permute(0, 2, 1, 3).reshape(b, n, h * d)
        return self.to_out(out)


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        dim,
        depth=PERCEIVER_DEPTH,
        num_latents=NUM_LATENTS,
        dim_head=PERCEIVER_HEAD_DIM,
        heads=PERCEIVER_HEADS,
        ff_mult=PERCEIVER_FF_MULT,
    ):
        """Build learned latents with stacked Perceiver attention/FF layers."""
        super().__init__()
        self.proj_context = nn.Identity()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.layers = nn.ModuleList(
            [nn.ModuleList([PerceiverAttention(dim, dim_head, heads), FeedForward(dim, ff_mult)]) for _ in range(depth)]
        )
        self.norm = RMSNorm(dim)

    def forward(self, x):
        """Resample context features into a fixed set of latents."""
        batch = x.shape[0]
        x = self.proj_context(x)
        latents = self.latents.unsqueeze(0).expand(batch, -1, -1)
        for attn, ff in self.layers:
            latents = attn(latents, x) + latents
            latents = ff(latents) + latents
        return self.norm(latents)


def wav_to_mel(wav, mel_norms):
    """Convert waveform to log-mel spectrogram normalized by mel_norms."""
    import librosa

    window = torch.hann_window(MEL_WIN, dtype=torch.float32)
    stft = torch.stft(
        wav.float(),
        n_fft=MEL_N_FFT,
        hop_length=MEL_HOP,
        win_length=MEL_WIN,
        window=window,
        center=True,
        pad_mode="reflect",
        return_complex=True,
    )
    power = stft.abs() ** 2

    fb = librosa.filters.mel(
        sr=MEL_SR, n_fft=MEL_N_FFT, n_mels=N_MELS, fmin=MEL_FMIN, fmax=MEL_FMAX, htk=True, norm="slaney"
    )
    fb = torch.from_numpy(fb).to(power.dtype)
    mel = torch.matmul(fb, power)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    return mel / mel_norms.unsqueeze(0).unsqueeze(-1)


def load_reference_audio(sample="en_sample.wav", max_seconds=COND_CHUNK_SEC):
    """Download and resample an HF sample wav for conditioning."""
    import math

    import soundfile as sf
    from huggingface_hub import hf_hub_download
    from scipy.signal import resample_poly

    path = hf_hub_download(repo_id=HF_REPO_ID, filename=f"samples/{sample}", revision=HF_REVISION)
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != MEL_SR:
        g = math.gcd(MEL_SR, sr)
        audio = resample_poly(audio, MEL_SR // g, sr // g)
    audio = audio[: MEL_SR * max_seconds]
    return torch.from_numpy(audio.astype("float32")).unsqueeze(0)


def load_coqui_test_audio(samples=("LJ001-0001.wav",), max_seconds=GPT_COND_LEN_SEC):
    """Download, concat, and resample Coqui test wavs for conditioning."""
    import math
    import os

    import numpy as np
    import soundfile as sf
    import torch.hub
    from scipy.signal import resample_poly

    cache = os.path.join(torch.hub.get_dir(), "xtts_coqui_ref")
    os.makedirs(cache, exist_ok=True)

    parts = []
    for sample in samples:
        path = os.path.join(cache, sample)
        if not os.path.exists(path):
            torch.hub.download_url_to_file(f"{COQUI_TESTS_WAV_URL}/{sample}", path)
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != MEL_SR:
            g = math.gcd(MEL_SR, sr)
            audio = resample_poly(audio, MEL_SR // g, sr // g)
        parts.append(audio.astype("float32"))

    audio = np.concatenate(parts)[: MEL_SR * max_seconds]
    return torch.from_numpy(audio).unsqueeze(0)


class XttsReferenceConditioning(nn.Module):
    def __init__(self):
        """Build conditioning encoder and Perceiver resampler."""
        super().__init__()
        self.conditioning_encoder = ConditioningEncoder(N_MELS, HIDDEN_SIZE)
        self.conditioning_perceiver = PerceiverResampler(HIDDEN_SIZE)

    def forward(self, mel):
        """Encode mel into fixed conditioning latents via Perceiver."""
        conds = self.conditioning_encoder(mel)
        conds = self.conditioning_perceiver(conds.permute(0, 2, 1)).transpose(1, 2)
        return conds


def reference_conditioning(state_dict):
    """Load conditioning encoder and perceiver weights from checkpoint."""
    module = XttsReferenceConditioning()

    prefix = "gpt.conditioning_encoder."
    enc_state = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
    module.conditioning_encoder.load_state_dict(enc_state)

    prefix = "gpt.conditioning_perceiver."
    perc_state = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
    module.conditioning_perceiver.load_state_dict(perc_state)

    module.eval()
    return module
