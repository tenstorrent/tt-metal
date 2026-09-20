"""Real CosyVoice2 preprocessing, reimplemented from real upstream source
(fetched fresh from FunAudioLLM/CosyVoice's cosyvoice/cli/frontend.py and
Matcha-TTS's matcha/utils/audio.py), not guessed -- since neither `matcha-tts`
nor `cosyvoice` packages are installed in this environment (and installing
matcha-tts via pip destabilized the venv -- torchaudio/torch got silently
swapped mid-install -- so it was abandoned; this reimplements only the small
piece actually needed).

Three real preprocessing paths, each confirmed against the real function body:
- prompt speech tokens: `whisper.log_mel_spectrogram(wav16k, n_mels=128)` ->
  `speech_tokenizer_v2.onnx` (two inputs: feat, feat_length)
- speaker embedding (xvec): `torchaudio.compliance.kaldi.fbank(wav16k,
  num_mel_bins=80, dither=0, sample_frequency=16000)`, mean-subtracted ->
  `campplus.onnx`
- prompt mel (flow decoder's own `prompt_feat`): `mel_spectrogram(wav24k,
  n_fft=1920, num_mels=80, sampling_rate=24000, hop_size=480, win_size=1920,
  fmin=0, fmax=8000, center=False)` -- exact params from `cosyvoice2.yaml`'s
  `feat_extractor` block, exact math from Matcha-TTS's `mel_spectrogram`
  (STFT -> librosa mel filterbank -> log compression).
"""
from __future__ import annotations

import numpy as np
import torch
import torchaudio.compliance.kaldi as kaldi
from librosa.filters import mel as librosa_mel_fn


def mel_spectrogram(
    y: torch.Tensor,
    n_fft: int = 1920,
    num_mels: int = 80,
    sampling_rate: int = 24000,
    hop_size: int = 480,
    win_size: int = 1920,
    fmin: int = 0,
    fmax: int = 8000,
    center: bool = False,
) -> torch.Tensor:
    """y: [1, T] float32 in [-1, 1] @ sampling_rate. Returns [1, num_mels, T']."""
    mel_basis = torch.from_numpy(librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)).float()
    hann = torch.hann_window(win_size)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect")
    y = y.squeeze(1)
    spec = torch.stft(
        y, n_fft, hop_length=hop_size, win_length=win_size, window=hann,
        center=center, pad_mode="reflect", normalized=False, onesided=True, return_complex=True,
    )
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
    spec = torch.matmul(mel_basis, spec)
    spec = torch.log(torch.clamp(spec, min=1e-5))
    return spec


def extract_prompt_feat(wav24k: torch.Tensor) -> torch.Tensor:
    """wav24k: [1, T] @ 24kHz -> [1, T', 80] channels-last, this port's convention."""
    feat = mel_spectrogram(wav24k)  # [1, 80, T']
    return feat.transpose(1, 2)  # [1, T', 80]


def extract_spk_embedding(campplus_session, wav16k: torch.Tensor) -> torch.Tensor:
    """wav16k: [1, T] @ 16kHz -> [1, 192] raw xvec (spk_embed_affine_layer normalizes internally)."""
    feat = kaldi.fbank(wav16k, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    embedding = campplus_session.run(
        None, {campplus_session.get_inputs()[0].name: feat.unsqueeze(0).numpy()}
    )[0]
    return torch.from_numpy(embedding).float()


def extract_speech_tokens(speech_tokenizer_session, wav16k: torch.Tensor) -> torch.Tensor:
    """wav16k: [1, T] @ 16kHz -> [1, N] int speech token ids, via the real speech tokenizer."""
    import whisper

    feat = whisper.log_mel_spectrogram(wav16k, n_mels=128)
    speech_token = speech_tokenizer_session.run(
        None,
        {
            speech_tokenizer_session.get_inputs()[0].name: feat.detach().numpy(),
            speech_tokenizer_session.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32),
        },
    )[0].flatten().tolist()
    return torch.tensor([speech_token], dtype=torch.long)
