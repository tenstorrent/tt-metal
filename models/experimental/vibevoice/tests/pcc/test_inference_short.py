# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Short reference inference smoke test (PyTorch gold before TTNN PCC).

Runs 1p_short.txt on CPU with SDPA; checks non-empty speech output.
"""

from pathlib import Path

import pytest
import torch

from models.experimental.vibevoice.common.config import (
    DEFAULT_TXT_PATH,
    MODEL_PATH,
    VOICES_DIR,
)
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor


pytestmark = pytest.mark.skipif(
    not Path(MODEL_PATH).is_dir(),
    reason=f"VibeVoice weights not found at {MODEL_PATH} (set VIBEVOICE_MODEL_PATH)",
)


@pytest.mark.timeout(600)
def test_inference_short_cpu(tmp_path):
    assert DEFAULT_TXT_PATH.is_file()
    assert VOICES_DIR.is_dir()

    with open(DEFAULT_TXT_PATH, encoding="utf-8") as f:
        script = f.read().strip().replace("\u2019", "'")

    voice_path = VOICES_DIR / "en-Alice_woman.wav"
    if not voice_path.is_file():
        wavs = list(VOICES_DIR.glob("*.wav"))
        assert wavs, f"No voice WAV in {VOICES_DIR}"
        voice_path = wavs[0]

    processor = VibeVoiceProcessor.from_pretrained(MODEL_PATH)
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        device_map="cpu",
        attn_implementation="sdpa",
    )
    model.eval()
    model.set_ddpm_inference_steps(num_steps=10)

    inputs = processor(
        text=[script],
        voice_samples=[[str(voice_path)]],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=None,
        cfg_scale=1.3,
        tokenizer=processor.tokenizer,
        generation_config={"do_sample": False},
        verbose=False,
        is_prefill=True,
    )

    assert outputs.speech_outputs is not None
    assert len(outputs.speech_outputs) >= 1
    speech = outputs.speech_outputs[0]
    assert speech is not None
    num_samples = speech.shape[-1] if hasattr(speech, "shape") else len(speech)
    assert num_samples > 1000

    out_wav = tmp_path / "1p_short_generated.wav"
    processor.save_audio(speech, output_path=str(out_wav))
    assert out_wav.stat().st_size > 1000
