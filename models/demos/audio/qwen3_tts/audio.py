# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Host audio path for Qwen3-TTS: a file or a waveform in, the tensors the encoders want out.

Two consumers, one sample rate. The codec encoder takes the waveform itself at 24 kHz and
the speaker encoder takes a 128-bin log-mel of it, and `config.json` happens to put both at
24 kHz, so a reference clip is resampled once and read twice.

Nothing here runs on device and nothing here is going to: an STFT is not a TTNN op, and the
whole thing is 4 ms of work on a 3 s clip.

`mel_spectrogram` is a faithful port of the function inside
`reference/qwen/speaker_encoder.py`, which is vendored upstream and stays an oracle;
`tests/pcc/test_speaker_pcc.py` checks the two against each other for exact equality. The
port exists so the pipeline does not import the oracle, and so the mel front-end is
something this directory owns rather than borrows.

Three details in it are easy to get wrong and all three are load-bearing:

  * the STFT is **not** centred. Upstream pads the signal by `(n_fft - hop) / 2` on each
    side in reflect mode and then passes `center=False`, which is not the same as letting
    torch centre it: the frame count differs by one and every frame shifts.
  * the magnitude is `sqrt(real^2 + imag^2 + 1e-9)`, not `abs()`. The epsilon inside the
    square root is what keeps its gradient finite, and it changes the quiet bins.
  * the filterbank is librosa's slaney-normalised mel, and the compression is a plain
    natural log clamped at 1e-5. No power, no decibels, no reference level.
"""

import functools

import torch

from models.demos.audio.qwen3_tts import weights


@functools.lru_cache(maxsize=4)
def _mel_basis(sampling_rate, n_fft, num_mels, fmin, fmax):
    """librosa's slaney-normalised filterbank, [num_mels, n_fft // 2 + 1]."""
    from librosa.filters import mel as librosa_mel

    basis = librosa_mel(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
    return torch.from_numpy(basis).float()


@functools.lru_cache(maxsize=4)
def _hann(win_size):
    return torch.hann_window(win_size)


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax=None, center=False):
    """Waveform [B, N] -> log-mel [B, num_mels, T]. Upstream's function, ported."""
    basis = _mel_basis(sampling_rate, n_fft, num_mels, fmin, fmax)
    window = _hann(win_size)

    padding = (n_fft - hop_size) // 2
    padded = torch.nn.functional.pad(y.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)

    spectrum = torch.stft(
        padded,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    magnitude = torch.sqrt(torch.view_as_real(spectrum).pow(2).sum(-1) + 1e-9)
    return torch.log(torch.clamp(basis @ magnitude, min=1e-5))


def speaker_mel(waveform, sample_rate=None):
    """Waveform [N] or [1, N] at 24 kHz -> log-mel [1, T, 128], the speaker encoder's input.

    The trailing transpose is upstream's, from `extract_speaker_embedding`.
    """
    expected = weights.SPEAKER_MEL["sampling_rate"]
    if sample_rate is not None and int(sample_rate) != expected:
        raise ValueError(f"the speaker encoder wants {expected} Hz audio, got {sample_rate}")

    audio = torch.as_tensor(waveform, dtype=torch.float32)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    if audio.dim() != 2 or audio.shape[0] != 1:
        raise ValueError(f"expected mono audio shaped [N] or [1, N], got {tuple(audio.shape)}")

    return mel_spectrogram(audio, **weights.SPEAKER_MEL).transpose(1, 2)


def read_clip(path, sample_rate=None):
    """An audio file -> mono waveform [N] at the codec's rate, resampled if it has to be.

    Both encoders want 24 kHz, so a clip is converted once here rather than twice later.
    """
    import librosa
    import soundfile

    target = int(sample_rate or weights.codec_config()["input_sample_rate"])
    audio, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)  # a stereo reference clip is mixed down, not split
    if rate != target:
        audio = librosa.resample(y=audio, orig_sr=int(rate), target_sr=target)
    return torch.from_numpy(audio.copy()).float()
