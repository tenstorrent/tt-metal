# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from scipy.io import wavfile
from transformers import WhisperFeatureExtractor

from models.demos.audio.whisper.demo.demo import read_wav_as_float


def _tone(rate, seconds=1.0):
    t = np.arange(int(rate * seconds)) / rate
    return (0.5 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 1200 * t)).astype(np.float32)


def _features(fe, data, rate):
    return fe(data, sampling_rate=rate, return_tensors="np").input_features


@pytest.fixture(scope="module")
def fe():
    return WhisperFeatureExtractor()


@pytest.mark.parametrize("kind", ["int16_stereo", "float_stereo", "int16_mono"])
def test_wav_layout_matches_mono_float(tmp_path, fe, kind):
    ref = _tone(16000)
    if kind == "int16_mono":
        arr = np.round(ref * 32767).astype(np.int16)
    elif kind == "int16_stereo":
        arr = np.round(np.stack([ref, ref], axis=1) * 32767).astype(np.int16)
    else:
        arr = np.stack([ref, ref], axis=1)
    path = tmp_path / "a.wav"
    wavfile.write(path, 16000, arr)
    rate, data = read_wav_as_float(str(path))
    assert rate == 16000 and data.ndim == 1 and data.dtype == np.float32
    diff = np.abs(_features(fe, data, rate) - _features(fe, ref, 16000)).max()
    assert diff < 1e-2


def test_wav_44k_resampled_to_16k(tmp_path, fe):
    ref = _tone(16000)
    src = _tone(44100)
    arr = np.round(np.stack([src, src], axis=1) * 32767).astype(np.int16)
    path = tmp_path / "b.wav"
    wavfile.write(path, 44100, arr)
    rate, data = read_wav_as_float(str(path))
    assert rate == 16000 and data.shape[0] == ref.shape[0]
    diff = np.abs(_features(fe, data, rate) - _features(fe, ref, 16000)).max()
    assert diff < 0.1
