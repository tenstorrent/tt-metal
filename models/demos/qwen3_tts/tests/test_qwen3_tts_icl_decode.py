# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
ICL post-processing gate for qwen3_tts (host only, no device).

Generated codes must come back as exactly ``num_frames * 1920`` samples of
target speech: nothing trimmed from the head, and decoded with the reference
codes as decoder context (HF ``generate_voice_clone``).

Run:
    pytest -s -v models/demos/qwen3_tts/tests/test_qwen3_tts_icl_decode.py
"""

from pathlib import Path

import pytest
import torch

from models.demos.qwen3_tts.tt.server import TTSConfig, decode_audio, decode_icl_audio

SAMPLES_PER_FRAME = 1920  # 24 kHz / 12.5 fps
REF_CACHE = Path(__file__).resolve().parents[1] / "demo" / "jim_reference.refcache.pt"


@pytest.fixture(scope="module")
def decoder_weights():
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    sd = load_file(hf_hub_download("Qwen/Qwen3-TTS-12Hz-1.7B-Base", "speech_tokenizer/model.safetensors"))
    return {k[8:]: v.float() for k, v in sd.items() if k.startswith("decoder.")}


def test_no_default_codec_frame_trim():
    assert TTSConfig().trim_codec_frames == 0


@pytest.mark.parametrize("split", [15, 35])
def test_decode_icl_audio_keeps_every_generated_frame(decoder_weights, split):
    # Real speech codes stand in for (reference, generated continuation).
    codes_all = torch.load(REF_CACHE, weights_only=True)["ref_codes"].long()
    ref_codes, gen_codes = codes_all[:split], codes_all[split:]

    audio = decode_icl_audio(ref_codes, gen_codes, decoder_weights).squeeze()
    assert audio.shape[-1] == gen_codes.shape[0] * SAMPLES_PER_FRAME

    # Must equal the generated span of one full-context decode.
    expected = decode_audio(codes_all, decoder_weights).squeeze()[split * SAMPLES_PER_FRAME :]
    torch.testing.assert_close(audio, expected, atol=1e-4, rtol=0)
