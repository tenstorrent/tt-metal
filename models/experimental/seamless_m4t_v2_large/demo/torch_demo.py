"""
Seamless M4T v2 smoke test — CPU-friendly (no torchaudio CUDA extension).
source: https://huggingface.co/facebook/seamless-m4t-v2-large
"""

import io
import os
import urllib.request

import numpy as np
import scipy.io.wavfile
import soundfile as sf
import torch
from transformers import AutoProcessor, SeamlessM4Tv2Model


def load_audio(path_or_url: str) -> tuple[torch.Tensor, int]:
    """Load mono/stereo float waveform [channels, samples] and sample rate."""
    if path_or_url.startswith(("http://", "https://")):
        with urllib.request.urlopen(path_or_url, timeout=30) as resp:
            data = resp.read()
        audio, sr = sf.read(io.BytesIO(data))
    else:
        local = os.path.expanduser(path_or_url)
        if not os.path.isfile(local):
            raise FileNotFoundError(f"No such file or not a regular file: {local}")
        audio, sr = sf.read(local)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    else:
        audio = audio.T
    return torch.from_numpy(audio.astype(np.float32)), int(sr)


def resample_waveform(wave: torch.Tensor, orig_freq: int, new_freq: int) -> torch.Tensor:
    """Linear resample [C, T] using torch only (no torchaudio)."""
    if orig_freq == new_freq:
        return wave
    duration = wave.shape[-1] / orig_freq
    new_len = max(1, int(round(duration * new_freq)))
    x = wave.unsqueeze(0)
    y = torch.nn.functional.interpolate(x, size=new_len, mode="linear", align_corners=False)
    return y.squeeze(0)


processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")
model.eval().to("cpu")

# from text
text_inputs = processor(text="Hello, my dog is cute", src_lang="eng", return_tensors="pt")
with torch.no_grad():
    audio_array_from_text = model.generate(**text_inputs, tgt_lang="hin")[0].cpu().numpy().squeeze()

# from audio
audio, orig_freq = load_audio("https://www.cs.kzoo.edu/cs107/MediaSources/preamble10.wav")
audio = resample_waveform(audio, orig_freq=orig_freq, new_freq=16_000)
audio_inputs = processor(audios=audio, return_tensors="pt")
with torch.no_grad():
    audio_array_from_audio = model.generate(**audio_inputs, tgt_lang="hin")[0].cpu().numpy().squeeze()


def _as_wav_samples(data: np.ndarray) -> np.ndarray:
    """scipy.io.wavfile expects mono (N,) or multichannel (N, C)."""
    x = np.asarray(data)
    x = np.squeeze(x)
    if x.ndim == 2 and x.shape[0] in (1, 2) and x.shape[0] < x.shape[1]:
        x = x.T
    return x


sample_rate = model.config.sampling_rate
scipy.io.wavfile.write("out_from_text.wav", rate=sample_rate, data=_as_wav_samples(audio_array_from_text))
scipy.io.wavfile.write("out_from_audio.wav", rate=sample_rate, data=_as_wav_samples(audio_array_from_audio))

print("ok", audio_array_from_text.shape, audio_array_from_audio.shape)
