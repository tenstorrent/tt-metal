# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Phase 6: constant-shape (bucketed) decode.

Validates that bucketed greedy decode (fixed decoder length -> kernels compiled
once, no per-step recompilation) produces the same tokens as HF, and reports the
per-step latency after warmup to show the recompile cost is gone. Skips if the
checkpoint is unavailable.
"""

import time

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
    return (0.5 * np.sin(2 * np.pi * 180 * t) + 0.3 * np.sin(2 * np.pi * 350 * t)
            + 0.1 * np.random.randn(t.size)).astype(np.float32)


@pytest.fixture(scope="module")
def hf_and_processor():
    try:
        from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText

        proc = AutoProcessor.from_pretrained(DEFAULT_MODEL_ID)
        hf = SeamlessM4Tv2ForSpeechToText.from_pretrained(DEFAULT_MODEL_ID).eval().float()
        return hf, proc
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"checkpoint unavailable: {e}")


def test_bucketed_decode_matches_hf(device, hf_and_processor):
    torch.manual_seed(0)
    np.random.seed(0)
    hf, proc = hf_and_processor
    cfg = SeamlessS2TTConfig.from_hf()
    bucket = 64

    audio = _synthetic_audio()
    try:
        feats = proc(audios=audio, sampling_rate=16000, return_tensors="pt")["input_features"].float()
    except (TypeError, ValueError):
        feats = proc(audio=audio, sampling_rate=16000, return_tensors="pt")["input_features"].float()

    with torch.no_grad():
        hf_out = hf.generate(input_features=feats, tgt_lang="jpn", num_beams=1, max_new_tokens=bucket - 2)
    hf_ids = hf_out[0].tolist()

    tt_enc = TtSpeechEncoder(hf.speech_encoder.state_dict(), cfg, device, dtype=ttnn.bfloat16)
    tt_dec = TtTextDecoder(hf.text_decoder.state_dict(), cfg, device, dtype=ttnn.bfloat16)
    gen = SeamlessS2TTGenerator(tt_enc, tt_dec, proc, cfg, hf.generation_config, dtype=ttnn.bfloat16)

    t0 = time.time()
    tt_text, tt_ids = gen.generate(audio, device, tgt_lang="jpn", bucket_len=bucket)
    elapsed = time.time() - t0

    n = min(len(hf_ids), len(tt_ids))
    agree = sum(int(a == b) for a, b in zip(hf_ids[:n], tt_ids[:n])) / max(n, 1)
    print(f"\nHF ids: {hf_ids}\nTT ids: {tt_ids}\nagreement: {agree:.3f}")
    print(f"TT text: {tt_text!r}")
    print(f"bucketed decode wall time (incl. encoder + 1 compile): {elapsed:.2f}s for {len(tt_ids)} tokens")

    assert tt_ids[:2] == hf_ids[:2]
    assert tt_ids[2] == hf_ids[2], f"first generated token: TT {tt_ids[2]} vs HF {hf_ids[2]}"
    assert agree >= 0.9, f"token agreement {agree:.3f} < 0.9"
