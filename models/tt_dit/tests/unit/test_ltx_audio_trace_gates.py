# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Audio-decode trace defaults. No device needed."""

import pytest

from models.tt_dit.models.audio_vae.audio_decoder_ltx import _trace_gates


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("LTX_VOC_TRACE", "LTX_VAE_TRACE"):
        monkeypatch.delenv(k, raising=False)


def test_traced_pipeline_traces_vocoder_bwe_and_mel_vae_by_default():
    assert _trace_gates(True) == (True, True, True)


def test_untraced_pipeline_traces_nothing(monkeypatch):
    monkeypatch.setenv("LTX_VOC_TRACE", "1")
    monkeypatch.setenv("LTX_VAE_TRACE", "1")
    assert _trace_gates(False) == (False, False, False)


def test_kill_switches_force_eager(monkeypatch):
    monkeypatch.setenv("LTX_VOC_TRACE", "0")
    monkeypatch.setenv("LTX_VAE_TRACE", "0")
    assert _trace_gates(True) == (False, True, False)
