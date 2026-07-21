# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Reference (pure-PyTorch) XTTS-v2 audio *conditioning* path — ref audio -> GPT prompt.

This is the speaker-conditioning branch that produces the latents XTTS prepends
to the GPT input stream (the "prompt" in ``GPT.get_logits``). The flow, mirroring
coqui ``Xtts.get_gpt_cond_latents`` + ``GPT.get_style_emb``:

    wav  --preprocess-->  mel [b, 80, s]        (torch.stft + mel filterbank)
    mel  --conditioning_encoder-->  [b, 1024, s]   (init conv1d + 6 attn blocks)
    x    --conditioning_perceiver-->  [b, 1024, 32] (PerceiverResampler, 32 latents)

The ``ConditioningEncoder``/``AttentionBlock`` and ``PerceiverResampler`` classes
are faithful ports of coqui's ``TTS/tts/layers/xtts/latent_encoder.py`` and
``TTS/tts/layers/xtts/perceiver_encoder.py`` (flash-attention path removed — CPU
math path only). Weights come from the upstream checkpoint at
https://huggingface.co/coqui/XTTS-v2 (``gpt.conditioning_encoder.*`` and
``gpt.conditioning_perceiver.*``); ``mel_stats`` (the log-mel normalizer) is the
checkpoint buffer of the same name.

NOTE: coqui computes the mel with ``torchaudio``; torchaudio is unavailable in
this env, so the mel here is an equivalent ``torch.stft`` + a slaney-normalized
htk mel filterbank (``librosa.filters.mel``). The mel is computed once on the
host and fed identically to the reference and the TTNN port, so PCC between them
is independent of the exact mel implementation.
"""

import math

import torch
from torch import einsum, nn
from torch.nn import functional as F

from models.experimental.xtts.reference.xtts_gpt_block import HF_REPO_ID, HIDDEN_SIZE

N_MELS = 80
NUM_ATTN_HEADS = 16  # GPT.__init__ passes heads (=16) to ConditioningEncoder
NUM_LATENTS = 32

# Mel-spectrogram config for the perceiver-resampler path (coqui get_gpt_cond_latents).
MEL_N_FFT = 2048
MEL_HOP = 256
MEL_WIN = 1024
MEL_SR = 22050
MEL_FMIN = 0
MEL_FMAX = 8000
COND_CHUNK_SEC = 6  # legacy single-window length (kept for load_reference_audio default)

# coqui get_gpt_cond_latents: condition on up to gpt_cond_len=30 s of reference audio,
# split into gpt_cond_chunk_len=4 s windows, run get_style_emb per chunk and AVERAGE the
# 32-latent style embeddings. We chunk the precomputed log-mel along time (equivalent to
# chunking the audio, up to a few STFT-overlap frames at boundaries) and average.
GPT_COND_LEN_SEC = 30  # gpt_cond_len / max_ref_len
GPT_COND_CHUNK_SEC = 4  # gpt_cond_chunk_len
COND_CHUNK_FRAMES = int(round(GPT_COND_CHUNK_SEC * MEL_SR / MEL_HOP))  # ~344 mel frames / chunk
COND_MIN_CHUNK_FRAMES = 32  # drop a tiny trailing chunk (also keeps lengths tile-sane)


def chunk_cond_mel(mel, chunk_frames=COND_CHUNK_FRAMES, min_frames=COND_MIN_CHUNK_FRAMES):
    """Split a log-mel ``[b, 80, s]`` along time into ``gpt_cond_chunk_len`` windows for
    coqui-style style-embedding averaging. Returns a list of ``[b, 80, <=chunk_frames]``
    mels. A trailing window shorter than ``min_frames`` is dropped; a mel already shorter
    than one chunk is returned as-is (single window == the previous behaviour, no change)."""
    s = mel.shape[-1]
    if s <= chunk_frames:
        return [mel]
    chunks = [mel[..., i : i + chunk_frames] for i in range(0, s, chunk_frames)]
    kept = [c for c in chunks if c.shape[-1] >= min_frames]
    return kept or [mel]


COND_CHUNK_SAMPLES = int(round(GPT_COND_CHUNK_SEC * MEL_SR))  # 88200 samples / 4 s window
COND_MIN_CHUNK_SAMPLES = COND_MIN_CHUNK_FRAMES * MEL_HOP  # drop a tiny trailing chunk


def chunk_wav(wav, chunk_samples=COND_CHUNK_SAMPLES, min_samples=COND_MIN_CHUNK_SAMPLES):
    """Split a waveform ``[1, L]`` (22.05 kHz) into ``gpt_cond_chunk_len`` windows — the
    on-device analogue of :func:`chunk_cond_mel` (coqui chunks the audio, then mels each
    chunk). Returns a list of ``[1, <=chunk_samples]`` wavs; a trailing window shorter than
    ``min_samples`` is dropped; a wav already under one chunk is returned as-is."""
    length = wav.shape[-1]
    if length <= chunk_samples:
        return [wav]
    chunks = [wav[..., i : i + chunk_samples] for i in range(0, length, chunk_samples)]
    kept = [c for c in chunks if c.shape[-1] >= min_samples]
    return kept or [wav]


# ---------------------------------------------------------------------------
# ConditioningEncoder — port of TTS/tts/layers/xtts/latent_encoder.py
# ---------------------------------------------------------------------------
class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def normalization(channels):
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
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """qkv: [N, (H*3*C), T] -> [N, (H*C), T]."""
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
    """Spatial self-attention block (non-causal), channels-first [b, c, t]."""

    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = normalization(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)
        self.x_proj = nn.Identity()
        self.proj_out = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
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
        super().__init__()
        self.init = nn.Conv1d(spec_dim, embedding_dim, kernel_size=1)
        self.attn = nn.Sequential(*[AttentionBlock(embedding_dim, num_attn_heads) for _ in range(attn_blocks)])
        self.dim = embedding_dim

    def forward(self, x):  # x: (b, 80, s) -> (b, dim, s)
        return self.attn(self.init(x))


# ---------------------------------------------------------------------------
# PerceiverResampler — port of TTS/tts/layers/xtts/perceiver_encoder.py
# ---------------------------------------------------------------------------
def _exists(x):
    return x is not None


class RMSNorm(nn.Module):
    def __init__(self, dim, scale=True):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim)) if scale else None

    def forward(self, x):
        gamma = self.gamma if _exists(self.gamma) else 1
        return F.normalize(x, dim=-1) * self.scale * gamma


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


def FeedForward(dim, mult=4):
    dim_inner = int(dim * mult * 2 / 3)
    return nn.Sequential(nn.Linear(dim, dim_inner * 2), GEGLU(), nn.Linear(dim_inner, dim))


class PerceiverAttention(nn.Module):
    """Cross-attention; latents attend to [latents ; context] (CPU math path)."""

    def __init__(self, dim, dim_head=64, heads=8, cross_attn_include_queries=True):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.cross_attn_include_queries = cross_attn_include_queries
        dim_inner = dim_head * heads
        self.to_q = nn.Linear(dim, dim_inner, bias=False)
        self.to_kv = nn.Linear(dim, dim_inner * 2, bias=False)
        self.to_out = nn.Linear(dim_inner, dim, bias=False)

    def _split(self, t):  # [b, n, h*d] -> [b, h, n, d]
        b, n, _ = t.shape
        return t.reshape(b, n, self.heads, self.dim_head).permute(0, 2, 1, 3)

    def forward(self, x, context):
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
    def __init__(self, dim, depth=2, num_latents=NUM_LATENTS, dim_head=64, heads=8, ff_mult=4):
        super().__init__()
        self.proj_context = nn.Identity()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.layers = nn.ModuleList(
            [nn.ModuleList([PerceiverAttention(dim, dim_head, heads), FeedForward(dim, ff_mult)]) for _ in range(depth)]
        )
        self.norm = RMSNorm(dim)

    def forward(self, x):  # x: (b, s, dim) -> (b, num_latents, dim)
        batch = x.shape[0]
        x = self.proj_context(x)
        latents = self.latents.unsqueeze(0).expand(batch, -1, -1)
        for attn, ff in self.layers:
            latents = attn(latents, x) + latents
            latents = ff(latents) + latents
        return self.norm(latents)


# ---------------------------------------------------------------------------
# audio preprocessing (torch.stft + mel filterbank; torchaudio-free)
# ---------------------------------------------------------------------------
def wav_to_mel(wav, mel_norms):
    """waveform [1, samples] (22050 Hz) -> normalized log-mel [1, 80, frames].

    Equivalent to coqui ``wav_to_mel_cloning`` (perceiver path params), computed
    with ``torch.stft`` + a slaney-normalized htk mel filterbank.
    """
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
    power = stft.abs() ** 2  # [1, n_freqs, frames]

    fb = librosa.filters.mel(
        sr=MEL_SR, n_fft=MEL_N_FFT, n_mels=N_MELS, fmin=MEL_FMIN, fmax=MEL_FMAX, htk=True, norm="slaney"
    )
    fb = torch.from_numpy(fb).to(power.dtype)  # [n_mels, n_freqs]
    mel = torch.matmul(fb, power)  # [1, n_mels, frames]
    mel = torch.log(torch.clamp(mel, min=1e-5))
    return mel / mel_norms.unsqueeze(0).unsqueeze(-1)


def load_reference_audio(sample="en_sample.wav", max_seconds=COND_CHUNK_SEC):
    """Download an XTTS-v2 sample wav, load mono @ 22050 Hz, clip to one chunk.

    Returns waveform ``[1, samples]``.
    """
    import math

    import soundfile as sf
    from huggingface_hub import hf_hub_download
    from scipy.signal import resample_poly

    path = hf_hub_download(repo_id=HF_REPO_ID, filename=f"samples/{sample}")
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != MEL_SR:
        g = math.gcd(MEL_SR, sr)
        audio = resample_poly(audio, MEL_SR // g, sr // g)  # e.g. 24000 -> 22050
    audio = audio[: MEL_SR * max_seconds]
    return torch.from_numpy(audio.astype("float32")).unsqueeze(0)


# ---------------------------------------------------------------------------
# reference conditioning module
# ---------------------------------------------------------------------------
class XttsReferenceConditioning(nn.Module):
    """conditioning_encoder + conditioning_perceiver -> GPT conditioning latents.

    ``forward(mel)`` takes normalized log-mel ``[b, 80, s]`` and returns the
    conditioning latents ``[b, 1024, 32]`` (coqui ``get_style_emb``).
    """

    def __init__(self):
        super().__init__()
        self.conditioning_encoder = ConditioningEncoder(N_MELS, HIDDEN_SIZE)
        self.conditioning_perceiver = PerceiverResampler(HIDDEN_SIZE)

    def forward(self, mel):
        conds = self.conditioning_encoder(mel)  # (b, 1024, s)
        conds = self.conditioning_perceiver(conds.permute(0, 2, 1)).transpose(1, 2)  # (b, 1024, 32)
        return conds


def reference_conditioning(state_dict):
    """Build the audio conditioning path with real weights, in eval mode."""
    module = XttsReferenceConditioning()

    prefix = "gpt.conditioning_encoder."
    enc_state = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
    module.conditioning_encoder.load_state_dict(enc_state)

    prefix = "gpt.conditioning_perceiver."
    perc_state = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
    module.conditioning_perceiver.load_state_dict(perc_state)

    module.eval()
    return module
