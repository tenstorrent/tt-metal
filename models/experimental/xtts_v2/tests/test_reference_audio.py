# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Reference-clip loading (frontend.load_reference_audio), the one file the demos read.

Host-only: no device, no checkpoint. Clips are written with soundfile and read back, so this
also pins soundfile as a real dependency of the front-end.

Run:
    pytest -svv models/experimental/xtts_v2/tests/test_reference_audio.py
"""
import numpy as np
import pytest
import soundfile as sf
import torch

from models.experimental.xtts_v2.frontend import load_reference_audio

SAMPLES = 24000
RATES = (16000, 22050, 24000, 44100)


def _signal(seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(SAMPLES, generator=g) * 0.1).numpy().astype("float32")


@pytest.mark.parametrize("sr", RATES)
def test_float_wav_round_trips(tmp_path, sr):
    """Float wav carries the samples and the rate through untouched."""
    x = _signal()
    p = tmp_path / f"m{sr}.wav"
    sf.write(p, x, sr, subtype="FLOAT")
    wav, got = load_reference_audio(str(p))
    assert got == sr
    assert wav.shape == (1, SAMPLES)
    assert np.array_equal(wav[0].numpy(), x)


def test_multichannel_is_averaged(tmp_path):
    """The mono downmix is a mean, so an anti-phase pair cancels."""
    x = _signal()
    p = tmp_path / "stereo.wav"
    sf.write(p, np.stack([x, -x], axis=1), 24000, subtype="FLOAT")
    wav, _ = load_reference_audio(str(p))
    assert wav.shape == (1, SAMPLES)
    assert np.allclose(wav[0].numpy(), 0.0)


def test_pcm16_is_quantised_not_mangled(tmp_path):
    """The common real-world encoding: samples land within one 16-bit step."""
    x = _signal()
    p = tmp_path / "i16.wav"
    sf.write(p, x, 24000, subtype="PCM_16")
    wav, got = load_reference_audio(str(p))
    assert got == 24000
    assert np.abs(wav[0].numpy() - x).max() <= 1.0 / 32767


@pytest.mark.parametrize("ext", ("flac", "ogg"))
def test_other_advertised_containers(tmp_path, ext):
    p = tmp_path / f"a.{ext}"
    sf.write(p, _signal(), 24000)
    wav, got = load_reference_audio(str(p))
    assert got == 24000
    assert wav.shape[0] == 1 and wav.shape[1] > 0


def test_unsupported_extension_is_rejected(tmp_path, expect_error):
    """A pickle cannot reach torch.load through this path."""
    p = tmp_path / "ref.pt"
    torch.save(torch.randn(SAMPLES), p)
    with expect_error(ValueError, "must be .wav/.flac/.ogg"):
        load_reference_audio(str(p))
