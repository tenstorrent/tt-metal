# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Phase 0 validation: the HF reference harness loads, the speech encoder runs and
exposes all intermediate captures, a single decoder step produces vocab-sized
logits, and end-to-end en->ja generation yields non-empty text.

These tests need the ~9GB `facebook/seamless-m4t-v2-large` checkpoint. If it is
not available they skip rather than fail, so the suite stays green pre-download.
"""

import numpy as np
import pytest
import torch

from models.demos.seamless_m4t_v2.reference.reference_s2tt import S2TTReference
from models.demos.seamless_m4t_v2.tt.model_config import SeamlessS2TTConfig


def _synthetic_audio(seconds: float = 3.0, sr: int = 16000) -> np.ndarray:
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    # mixture of tones so the fbank features are non-degenerate
    wave = 0.5 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 440 * t)
    return wave.astype(np.float32)


@pytest.fixture(scope="module")
def reference():
    try:
        return S2TTReference.load()
    except Exception as e:  # noqa: BLE001 - missing weights / offline
        pytest.skip(f"seamless-m4t-v2-large checkpoint unavailable: {e}")


def test_config_from_hf():
    c = SeamlessS2TTConfig.from_hf()
    assert c.hidden_size == 1024
    assert c.head_dim == 64
    assert c.num_distance_positions == 73  # left(64) + right(8) + 1
    assert c.vocab_size == 256102
    assert c.speech_encoder_layers == 24 and c.decoder_layers == 24


def test_speech_encoder_captures(reference):
    audio = _synthetic_audio()
    feats, mask = reference.extract_features(audio)
    assert feats.shape[-1] == reference.config.feature_projection_input_dim  # 160

    enc = reference.run_speech_encoder(feats, mask, capture=True)
    assert enc.shape[0] == 1 and enc.shape[-1] == 1024
    assert not torch.isnan(enc).any()

    caps = reference.captures
    assert "feature_projection" in caps
    assert "conformer_layer_0" in caps and "conformer_layer_23" in caps
    assert "encoder_layer_norm" in caps
    assert "intermediate_ffn" in caps
    assert "adapter_layer_0" in caps
    assert "inner_layer_norm" in caps
    # adapter (stride 8) downsamples sequence length vs the pre-adapter states
    assert caps["adapter_layer_0"].shape[1] < caps["conformer_layer_23"].shape[1]


def test_decoder_single_step(reference):
    audio = _synthetic_audio()
    feats, mask = reference.extract_features(audio)
    enc = reference.run_speech_encoder(feats, mask)

    start = reference.config.decoder_start_token_id
    dec_ids = torch.tensor([[start]], dtype=torch.long)
    logits = reference.decoder_logits(dec_ids, enc)
    assert logits.shape == (1, 1, reference.config.vocab_size)
    assert not torch.isnan(logits).any()


@pytest.mark.slow
def test_generate_en_to_ja(reference):
    audio = _synthetic_audio()
    feats, mask = reference.extract_features(audio)
    text = reference.generate_text(feats, mask, tgt_lang="jpn", num_beams=1)
    assert isinstance(text, str)
