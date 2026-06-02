# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Phase 5 E2E: TtSpeechEncoder + TtTextDecoder greedy generation vs HF.

Builds the TT pipeline and HF model from one checkpoint load, runs greedy S2TT
(tgt_lang=jpn) on the same audio, and checks that the TT token sequence matches
HF's. Synthetic audio is fine here — this validates functional TT==HF agreement,
not translation quality (that is Phase 8). Skips if the checkpoint is unavailable.
"""

import numpy as np
import pytest
import torch

import ttnn
from models.demos.seamless_m4t_v2.tt.conformer_encoder import TtSpeechEncoder
from models.demos.seamless_m4t_v2.tt.generator import SeamlessS2TTGenerator
from models.demos.seamless_m4t_v2.tt.model_config import SeamlessS2TTConfig, DEFAULT_MODEL_ID
from models.demos.seamless_m4t_v2.tt.text_decoder import TtTextDecoder


def _synthetic_audio(seconds=5.0, sr=16000):
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    wave = 0.5 * np.sin(2 * np.pi * 180 * t) + 0.3 * np.sin(2 * np.pi * 350 * t) + 0.1 * np.random.randn(t.size)
    return wave.astype(np.float32)


@pytest.fixture(scope="module")
def hf_and_processor():
    try:
        from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText

        proc = AutoProcessor.from_pretrained(DEFAULT_MODEL_ID)
        hf = SeamlessM4Tv2ForSpeechToText.from_pretrained(DEFAULT_MODEL_ID).eval().float()
        return hf, proc
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"checkpoint unavailable: {e}")


def test_e2e_greedy_matches_hf(device, hf_and_processor):
    torch.manual_seed(0)
    np.random.seed(0)
    hf, proc = hf_and_processor
    cfg = SeamlessS2TTConfig.from_hf()
    max_new = 16  # each TT decode step recompiles kernels for a new seq length (no KV cache yet)

    audio = _synthetic_audio()
    try:
        feats = proc(audios=audio, sampling_rate=16000, return_tensors="pt")["input_features"].float()
    except (TypeError, ValueError):
        feats = proc(audio=audio, sampling_rate=16000, return_tensors="pt")["input_features"].float()

    # HF golden greedy
    with torch.no_grad():
        hf_out = hf.generate(input_features=feats, tgt_lang="jpn", num_beams=1, max_new_tokens=max_new)
    hf_ids = hf_out[0].tolist() if hasattr(hf_out[0], "tolist") else list(hf_out[0])

    # TT pipeline (reuse the already-loaded weights)
    tt_enc = TtSpeechEncoder(hf.speech_encoder.state_dict(), cfg, device, dtype=ttnn.bfloat16)
    tt_dec = TtTextDecoder(hf.text_decoder.state_dict(), cfg, device, dtype=ttnn.bfloat16)
    gen = SeamlessS2TTGenerator(tt_enc, tt_dec, proc, cfg, hf.generation_config, dtype=ttnn.bfloat16)
    tt_text, tt_ids = gen.generate(audio, device, tgt_lang="jpn", max_new_tokens=max_new)

    hf_text = proc.decode(hf_ids, skip_special_tokens=True)
    n = min(len(hf_ids), len(tt_ids))
    agree = sum(int(a == b) for a, b in zip(hf_ids[:n], tt_ids[:n])) / max(n, 1)
    print(f"\nHF  ids: {hf_ids}\nTT  ids: {tt_ids}\nagreement: {agree:.3f}")
    print(f"HF text: {hf_text!r}\nTT text: {tt_text!r}")

    # tokens 0,1 are the fixed seed [decoder_start, lang]; index 2 is the first
    # actually-generated token and must match HF.
    assert tt_ids[:2] == hf_ids[:2], f"seed mismatch: {tt_ids[:2]} vs {hf_ids[:2]}"
    assert tt_ids[2] == hf_ids[2], f"first generated token mismatch: TT {tt_ids[2]} vs HF {hf_ids[2]}"
    assert agree >= 0.9, f"token agreement {agree:.3f} < 0.9"
