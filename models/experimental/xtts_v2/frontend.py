# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Coqui-free CPU front-end for XTTS-v2: tokenizer, mel/STFT front-ends, prompt assembly.

Replaces phase A (`pipeline/phase_coqui_pre.py`, which needs the coqui `TTS` package in a
separate venv) with an in-process implementation that runs in tt-metal's python_env. Everything
here is a *transcription* of coqui-tts 0.27.5 behavior with parameters pinned from source —
not a redesign. Coqui source references are given per function; validate any change against
captured phase-A tensors (see `pipeline/validate_frontend.py`).

The front-end jobs, and what consumes them:

  * `load_reference_audio(path, sr)`         -> ([1,N] mono wav, sr) (reference-clip IO)
  * `XttsTokenizer.encode(text, lang)`       -> token ids            (prompt assembly)
  * `conditioning_mels(audio, sr, stats)`    -> [1,80,T] mel chunks  (Block 1 input)
  * `speaker_logmel(audio, sr)`              -> [1,64,T] logmel      (Block 2 input)
  * `assemble_prompt(ids, cond_latents, tb)` -> [1,P,1024] prefix    (Block 3 prefill)

DSP building blocks (`sinc_resample`, `melscale_fbanks`, `mel_spectrogram`) reimplement the
exact torchaudio functions coqui calls (torchaudio is not in tt-metal's env). They are pure
torch and deterministic.

Language support: **en only** for text normalization. The BPE step itself (vocab.json) is
multilingual, but coqui's cleaner tables (abbreviations/symbols/ordinals per language, plus
zh/ja/ko transliteration) are transcribed only for "en" — extend from
TTS/tts/layers/xtts/tokenizer.py when another language is needed. Number expansion needs the
`num2words` package (tiny, pure python); without it, texts containing digits raise.
"""

import math
import os
import re

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------------------
# DSP building blocks (torchaudio-equivalent, pure torch)
# ---------------------------------------------------------------------------------------


def sinc_resample(waveform, orig_freq: int, new_freq: int):
    """torchaudio.functional.resample with default args (sinc_interp_hann, width 6,
    rolloff 0.99), which is what coqui's front-end calls. Pure-torch transcription of
    torchaudio's kernel construction (functional.py `_get_sinc_resample_kernel`) so the
    resampled samples match bit-for-bit without the torchaudio dependency."""
    if orig_freq == new_freq:
        return waveform
    lowpass_filter_width = 6
    rolloff = 0.99
    gcd = math.gcd(int(orig_freq), int(new_freq))
    orig = int(orig_freq) // gcd
    new = int(new_freq) // gcd

    base_freq = min(orig, new) * rolloff
    width = math.ceil(lowpass_filter_width * orig / base_freq)
    # torchaudio builds the kernel in the waveform's dtype (float32 here) — match it exactly
    dt = waveform.dtype
    idx = torch.arange(-width, width + orig, dtype=dt)[None, None] / orig
    t = torch.arange(0, -new, -1, dtype=dt)[:, None, None] / new + idx
    t *= base_freq
    t = t.clamp_(-lowpass_filter_width, lowpass_filter_width)
    window = torch.cos(t * math.pi / lowpass_filter_width / 2) ** 2
    t *= math.pi
    kernel = torch.where(t == 0, torch.tensor(1.0, dtype=t.dtype), t.sin() / t)
    kernel *= window * (base_freq / orig)

    shape = waveform.size()
    wav = waveform.reshape(-1, shape[-1])
    num, length = wav.shape
    wav = F.pad(wav, (width, width + orig))
    out = F.conv1d(wav[:, None], kernel, stride=orig)
    out = out.transpose(1, 2).reshape(num, -1)
    target_len = int(math.ceil(new * length / orig))
    return out[..., :target_len].view(shape[:-1] + (-1,))


def melscale_fbanks(n_freqs: int, f_min: float, f_max: float, n_mels: int, sample_rate: int, norm=None):
    """torchaudio.functional.melscale_fbanks with mel_scale="htk" (torchaudio's default,
    which both coqui mel front-ends use — note the cloning mel passes norm="slaney" but
    KEEPS the htk mel scale). Returns (n_freqs, n_mels)."""

    def hz_to_mel(f):
        return 2595.0 * math.log10(1.0 + f / 700.0)

    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)
    m_pts = torch.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    f_pts = 700.0 * (10.0 ** (m_pts / 2595.0) - 1.0)
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)  # (n_freqs, n_mels + 2)
    down = (-1.0 * slopes[:, :-2]) / f_diff[:-1]
    up = slopes[:, 2:] / f_diff[1:]
    fb = torch.max(torch.zeros(1), torch.min(down, up))
    if norm == "slaney":
        fb *= (2.0 / (f_pts[2 : n_mels + 2] - f_pts[:n_mels])).unsqueeze(0)
    return fb


def mel_spectrogram(wav, n_fft, hop_length, win_length, window, fb):
    """torchaudio.transforms.MelSpectrogram forward (power=2, center=True, reflect pad,
    not normalized): |STFT|^2 -> mel. wav [B,N] -> [B,n_mels,T]."""
    spec = torch.stft(
        wav,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = spec.abs().pow(2.0)  # [B, n_freqs, T]
    return torch.matmul(spec.transpose(-1, -2), fb).transpose(-1, -2)


# ---------------------------------------------------------------------------------------
# Reference-clip loading (shared by demo.py / demo_server.py / pipeline/validate_frontend.py)
# ---------------------------------------------------------------------------------------


def load_reference_audio(path, fallback_sr=22050):
    """Load a reference-voice clip -> (waveform tensor [1, N] float, sample rate).

    .wav/.flac/.ogg via soundfile (multi-channel is downmixed to mono by averaging the
    channels); .pt via torch.load handling a raw tensor/ndarray, a (tensor, sr) tuple, or
    a HuggingFace audio-sample dict {"audio": {"array": ..., "sampling_rate": ...}}.
    `fallback_sr` is used when the .pt carries no sample rate. soundfile/numpy are imported
    lazily so importing this module needs neither."""
    import numpy as np

    ext = os.path.splitext(path)[1].lower()
    if ext in (".wav", ".flac", ".ogg"):
        import soundfile as sf

        data, sr = sf.read(path, dtype="float32")
        wav = torch.from_numpy(data)
        if wav.dim() > 1:  # [N, C] -> mono
            wav = wav.mean(dim=1)
        return wav.reshape(1, -1), sr

    raw = torch.load(path, map_location="cpu", weights_only=False)
    sr = fallback_sr
    if isinstance(raw, dict) and isinstance(raw.get("audio"), dict) and "array" in raw["audio"]:
        sr = int(raw["audio"].get("sampling_rate", sr))
        raw = raw["audio"]["array"]
    elif isinstance(raw, dict):
        for k in ("array", "audio", "waveform", "wav"):
            if k in raw:
                raw = raw[k]
                break
    elif isinstance(raw, (tuple, list)):
        if len(raw) == 2 and isinstance(raw[1], (int, float)):
            raw, sr = raw[0], int(raw[1])
        else:
            raw = raw[0]
    wav = raw if torch.is_tensor(raw) else torch.as_tensor(np.asarray(raw))
    wav = wav.squeeze()
    if wav.dim() > 1:
        wav = wav[0]
    return wav.float().reshape(1, -1), sr


# ---------------------------------------------------------------------------------------
# Block 1 front-end: conditioning mel chunks (coqui Xtts.get_gpt_cond_latents)
# ---------------------------------------------------------------------------------------

COND_CHUNK_SECONDS = 6  # coqui chunk_length: style embeddings are computed per chunk and averaged
COND_MIN_SECONDS = 0.33  # chunks shorter than this are skipped
COND_MAX_SECONDS = 30  # coqui `length`: reference audio is truncated to this


def conditioning_mels(audio, sr: int, mel_norms):
    """Reference waveform -> list of [1,80,T] mel chunks for the conditioning encoder.

    Transcribes Xtts.get_gpt_cond_latents + wav_to_mel_cloning (perceiver-resampler branch:
    n_fft=2048, hop=256, win=1024, hann, power=2, mel htk scale + slaney norm, f 0..8000,
    log(clamp 1e-5), then divided per-band by the checkpoint's `mel_stats`).

    Returns a LIST because coqui computes one style embedding per <=6s chunk and averages the
    embeddings — so the caller must run Block 1 once per chunk and mean the latents. A <=6s
    reference clip yields exactly one chunk (the shape every PCC test used)."""
    audio = audio.reshape(1, -1).float()
    if sr != 22050:
        audio = sinc_resample(audio, sr, 22050)
    audio = audio[:, : 22050 * COND_MAX_SECONDS]

    fb = melscale_fbanks(2048 // 2 + 1, 0.0, 8000.0, 80, 22050, norm="slaney")
    window = torch.hann_window(1024, periodic=True)
    mel_norms = mel_norms.reshape(1, 80, 1)

    chunks = []
    step = 22050 * COND_CHUNK_SECONDS
    for i in range(0, audio.shape[1], step):
        chunk = audio[:, i : i + step]
        if chunk.shape[-1] < 22050 * COND_MIN_SECONDS:
            continue
        mel = mel_spectrogram(chunk, 2048, 256, 1024, window, fb)
        mel = torch.log(torch.clamp(mel, min=1e-5)) / mel_norms
        chunks.append(mel)
    if not chunks:
        raise ValueError(f"reference audio too short (min {COND_MIN_SECONDS}s)")
    return chunks


# ---------------------------------------------------------------------------------------
# Block 2 front-end: speaker-encoder logmel (coqui ResNetSpeakerEncoder.torch_spec)
# ---------------------------------------------------------------------------------------


def speaker_logmel(audio, sr: int):
    """Reference waveform -> [1,64,T] logmel, the ResNet speaker encoder's instancenorm
    input (= what the TT Block 2 consumes; see tests/test_speaker_pcc.py).

    Transcribes Xtts.get_speaker_embedding (resample to 16k) + the encoder's torch_spec:
    PreEmphasis(0.97) -> MelSpectrogram(16000, n_fft=512, win=400, hop=160, hamming window,
    64 mels, htk scale, no norm, f 0..8000) -> log(x + 1e-6)."""
    audio = audio.reshape(1, -1).float()
    if sr != 16000:
        audio = sinc_resample(audio, sr, 16000)

    # PreEmphasis: y[t] = x[t] - 0.97 * x[t-1], reflect-padded (coqui base_encoder.PreEmphasis)
    pre = F.pad(audio.unsqueeze(1), (1, 0), "reflect")
    kernel = torch.tensor([[[-0.97, 1.0]]], dtype=audio.dtype)
    audio = F.conv1d(pre, kernel).squeeze(1)

    fb = melscale_fbanks(512 // 2 + 1, 0.0, 8000.0, 64, 16000, norm=None)
    window = torch.hamming_window(400, periodic=True)
    mel = mel_spectrogram(audio, 512, 160, 400, window, fb)
    return (mel + 1e-6).log()


# ---------------------------------------------------------------------------------------
# Block 0: tokenizer (coqui VoiceBpeTokenizer, en cleaners only)
# ---------------------------------------------------------------------------------------

# TTS/tts/layers/xtts/tokenizer.py `_abbreviations["en"]`
_EN_ABBREVIATIONS = [
    (re.compile(rf"\b{abbr}\.", re.IGNORECASE), full)
    for abbr, full in [
        ("mrs", "misess"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]

# TTS/tts/layers/xtts/tokenizer.py `_symbols_multilingual["en"]`
_EN_SYMBOLS = [
    (re.compile(re.escape(sym), re.IGNORECASE), rep)
    for sym, rep in [
        ("&", " and "),
        ("@", " at "),
        ("%", " percent "),
        ("#", " hash "),
        ("$", " dollar "),
        ("£", " pound "),
        ("°", " degree "),
    ]
]

_EN_ORDINAL_RE = re.compile(r"([0-9]+)(st|nd|rd|th)")
_NUMBER_RE = re.compile(r"[0-9]+")
_COMMA_NUMBER_RE = re.compile(r"\b\d{1,3}(,\d{3})*(\.\d+)?\b")
_DECIMAL_RE = re.compile(r"([0-9]+[.,][0-9]+)")
_CURRENCY_RE = {
    "USD": re.compile(r"((\$[0-9\.\,]*[0-9]+)|([0-9\.\,]*[0-9]+\$))"),
    "GBP": re.compile(r"((£[0-9\.\,]*[0-9]+)|([0-9\.\,]*[0-9]+£))"),
    "EUR": re.compile(r"(([0-9\.\,]*[0-9]+€)|((€[0-9\.\,]*[0-9]+)))"),
}
_WHITESPACE_RE = re.compile(r"\s+")


def _num2words(*args, **kwargs):
    try:
        from num2words import num2words
    except ImportError as e:
        raise ImportError(
            "text contains digits; number expansion needs the `num2words` package "
            "(pip install num2words — pure python, no further deps)"
        ) from e
    return num2words(*args, **kwargs)


def _expand_currency_en(m, currency):
    amount = float(re.sub(r"[^\d.]", "", m.group(0).replace(",", ".")))
    full = _num2words(amount, to="currency", currency=currency, lang="en")
    if amount.is_integer():
        last_and = full.rfind(", ")
        if last_and != -1:
            full = full[:last_and]
    return full


def _expand_numbers_en(text):
    """TTS/tts/layers/xtts/tokenizer.py expand_numbers_multilingual, lang="en"."""
    if not _NUMBER_RE.search(text):
        return text  # avoid the num2words import for digit-free text
    text = _COMMA_NUMBER_RE.sub(lambda m: m.group(0).replace(",", ""), text)
    try:
        text = _CURRENCY_RE["GBP"].sub(lambda m: _expand_currency_en(m, "GBP"), text)
        text = _CURRENCY_RE["USD"].sub(lambda m: _expand_currency_en(m, "USD"), text)
        text = _CURRENCY_RE["EUR"].sub(lambda m: _expand_currency_en(m, "EUR"), text)
    except ImportError:
        raise
    except Exception:
        pass  # coqui swallows currency-expansion failures; the number res below still apply
    text = _DECIMAL_RE.sub(lambda m: _num2words(float(m.group(1).replace(",", ".")), lang="en"), text)
    text = _EN_ORDINAL_RE.sub(lambda m: _num2words(int(m.group(1)), ordinal=True, lang="en"), text)
    text = _NUMBER_RE.sub(lambda m: _num2words(int(m.group(0)), lang="en"), text)
    return text


def clean_text_en(text):
    """coqui multilingual_cleaners(text, "en"): strip double quotes, lowercase, expand
    numbers/abbreviations/symbols, collapse whitespace. Idempotent on its own output."""
    text = text.replace('"', "")
    text = text.lower()
    text = _expand_numbers_en(text)
    for regex, replacement in _EN_ABBREVIATIONS:
        text = regex.sub(replacement, text)
    for regex, replacement in _EN_SYMBOLS:
        text = regex.sub(replacement, text)
    text = text.strip()  # coqui expand_symbols_multilingual ends with .strip()
    text = _WHITESPACE_RE.sub(" ", text)
    return text


class XttsTokenizer:
    """coqui VoiceBpeTokenizer on the checkpoint's vocab.json (a HF `tokenizers` file).

    encode(): clean -> "[en]" language tag -> " " -> "[SPACE]" -> BPE ids. Matches coqui's
    Xtts.inference tokenization (which also does sent.strip().lower() first — .lower() is
    already part of the cleaner). en only; extend clean_text_* tables for other languages."""

    CHAR_LIMIT_EN = 250  # coqui warns (and audio may truncate) past this

    def __init__(self, vocab_file):
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(vocab_file)

    def encode(self, text, lang="en"):
        lang = lang.split("-")[0]
        if lang != "en":
            raise NotImplementedError(
                f"only 'en' normalization is transcribed (got {lang!r}); add the language's "
                "cleaner tables from coqui TTS/tts/layers/xtts/tokenizer.py"
            )
        text = clean_text_en(text.strip())
        text = f"[{lang}]{text}"
        text = text.replace(" ", "[SPACE]")
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        if torch.is_tensor(ids):
            ids = ids.tolist()
        text = self.tokenizer.decode(ids, skip_special_tokens=False).replace(" ", "")
        return text.replace("[SPACE]", " ").replace("[STOP]", "").replace("[UNK]", "")


# ---------------------------------------------------------------------------------------
# Block 3 front-end: GPT prompt-prefix assembly (coqui GPT.compute_embeddings)
# ---------------------------------------------------------------------------------------

START_TEXT_TOKEN = 261  # coqui GPT defaults (gpt.py); confirmed against config.json
STOP_TEXT_TOKEN = 0
MAX_TEXT_TOKENS = 402  # gpt_max_text_tokens: text_pos_embedding is [404,1024], minus start/stop


class PromptTables:
    """The three checkpoint tensors prompt assembly needs (text emb/pos + mel_stats).
    Loaded via reference/xtts_gpt_ref.load_full_state, which is lru_cached — cheap to share
    with the GPT weight preprocessing."""

    def __init__(self, ckpt_path=None):
        from models.experimental.xtts_v2.reference.xtts_gpt_ref import load_full_state

        sd = load_full_state(ckpt_path)
        self.text_emb = sd["gpt.text_embedding.weight"].float()  # [6681, 1024]
        self.text_pos = sd["gpt.text_pos_embedding.emb.weight"].float()  # [404, 1024]
        self.mel_stats = sd["mel_stats"].float()  # [80] per-band mel normalizers


def assemble_prompt(token_ids, cond_latents, tables: PromptTables):
    """token ids + [1,32,1024] conditioning latents -> [1,P,1024] GPT prefill prefix.

    Transcribes GPT.compute_embeddings: pad text with start/stop tokens, embed, add learned
    positions (row i of the pos table for sequence position i), prepend the conditioning
    latents. The START_AUDIO embedding is NOT part of the prefix — the decode driver feeds it
    as the first step (see phase_tt.py / TTNNGPTTracedDecoder)."""
    ids = torch.as_tensor(token_ids, dtype=torch.long).reshape(1, -1)
    assert ids.shape[1] < MAX_TEXT_TOKENS, f"text too long: {ids.shape[1]} tokens (max {MAX_TEXT_TOKENS})"
    ids = F.pad(ids, (0, 1), value=STOP_TEXT_TOKEN)
    ids = F.pad(ids, (1, 0), value=START_TEXT_TOKEN)
    emb = tables.text_emb[ids[0]] + tables.text_pos[: ids.shape[1]]  # [S,1024]
    return torch.cat([cond_latents.reshape(1, -1, 1024), emb.unsqueeze(0)], dim=1)
