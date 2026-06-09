# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end generate() PCC test — reference vs TTNN.

Uses the same 1p_short.txt + voice prompt as test_inference_short.py.
Compares concatenated speech waveform PCC (target >= 0.95; tighten toward 0.99 as
generator/CFG parity improves).
"""

import sys
from pathlib import Path

import pytest
import torch

from models.common.utility_functions import comp_pcc
from models.experimental.vibevoice.common.config import (
    MODEL_PATH,
    DEFAULT_TXT_PATH,
    VOICES_DIR,
)
from models.experimental.vibevoice.tt.ttnn_vibevoice_model import TTVibeVoiceModel

_VIBEVOICE_ROOT = Path(__file__).resolve().parent.parent.parent
_REFERENCE_DIR = _VIBEVOICE_ROOT / "reference"
for _p in (_REFERENCE_DIR, _VIBEVOICE_ROOT.parent.parent.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_VOICE_PATH = VOICES_DIR / "en-Alice_woman.wav"
_TEXT_PATH = DEFAULT_TXT_PATH

CFG_SCALE = 1.3
NUM_DIFFUSION_STEPS = 10
SPEECH_PCC = 0.99
TOKEN_MATCH = 0.99
# Cap AR steps for CI; remove or raise for full 1p_short parity runs.
MAX_NEW_TOKENS = 128


def _load_script() -> str:
    assert _TEXT_PATH.is_file()
    with open(_TEXT_PATH, encoding="utf-8") as f:
        return f.read().strip().replace("\u2019", "'")


def _voice_path() -> str:
    if _VOICE_PATH.is_file():
        return str(_VOICE_PATH)
    wavs = list(VOICES_DIR.glob("*.wav"))
    assert wavs, f"No voice WAV in {VOICES_DIR}"
    return str(wavs[0])


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_e2e_generate_speech_pcc(mesh_device):
    """Full generate() speech output PCC vs reference (greedy, 10 DPM steps)."""
    from vibevoice.modular.modeling_vibevoice_inference import (
        VibeVoiceForConditionalGenerationInference,
    )
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

    script = _load_script()
    voice_path = _voice_path()

    processor = VibeVoiceProcessor.from_pretrained(MODEL_PATH)
    ref_model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        device_map="cpu",
        attn_implementation="sdpa",
    )
    ref_model.eval()
    ref_model.set_ddpm_inference_steps(num_steps=NUM_DIFFUSION_STEPS)

    inputs = processor(
        text=[script],
        voice_samples=[[voice_path]],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    prefill_len = inputs["input_ids"].shape[1]

    # Pre-compute reference speech embeddings under seed 0 (matches ref generate prefill).
    torch.manual_seed(0)
    with torch.no_grad():
        _, prefill_speech_embeds = ref_model._process_speech_inputs(
            inputs["speech_tensors"].to(ref_model.dtype),
            inputs["speech_masks"],
        )

    torch.manual_seed(0)
    ref_out = ref_model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        cfg_scale=CFG_SCALE,
        tokenizer=processor.tokenizer,
        generation_config={"do_sample": False},
        verbose=False,
        is_prefill=True,
    )

    assert ref_out.speech_outputs and ref_out.speech_outputs[0] is not None
    ref_speech = ref_out.speech_outputs[0].to(torch.float32).reshape(-1)
    assert ref_speech.numel() > 1000

    tt_model = TTVibeVoiceModel.from_checkpoint(
        mesh_device,
        MODEL_PATH,
        cfg_scale=CFG_SCALE,
        num_diffusion_steps=NUM_DIFFUSION_STEPS,
    )
    tt_model.set_speech_scale_bias(
        ref_model.model.speech_scaling_factor.item(),
        ref_model.model.speech_bias_factor.item(),
    )

    # Seed global RNG then replay the EXACT same acoustic-encoding RNG draw that
    # reference does on its first generate step.  Using the same _process_speech_inputs
    # call (rather than torch.randn(*shape)) handles any std_dist_type automatically
    # ("gaussian" draws batch_size extra values beyond randn_like(mean)).
    torch.manual_seed(0)
    with torch.no_grad():
        ref_model._process_speech_inputs(
            inputs["speech_tensors"].to(ref_model.dtype),
            inputs["speech_masks"],
        )
    tt_out = tt_model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        speech_input_mask=inputs["speech_input_mask"],
        prefill_speech_embeds=prefill_speech_embeds,
        tokenizer=processor.tokenizer,
        cfg_scale=CFG_SCALE,
        num_diffusion_steps=NUM_DIFFUSION_STEPS,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    assert tt_out.speech_outputs and tt_out.speech_outputs[0].numel() > 0
    tt_speech = tt_out.speech_outputs[0].to(torch.float32).reshape(-1)

    ref_gen = ref_out.sequences[0, prefill_len:]
    tt_gen = tt_out.sequences[0, prefill_len:]
    cmp_len = min(ref_gen.numel(), tt_gen.numel())
    token_match = 1.0
    if cmp_len > 0:
        token_match = (ref_gen[:cmp_len] == tt_gen[:cmp_len]).float().mean().item()
        assert (
            token_match >= TOKEN_MATCH
        ), f"Generated token match rate {token_match:.4f} < {TOKEN_MATCH} over {cmp_len} tokens"

    n = min(ref_speech.numel(), tt_speech.numel())
    passed, pcc_val = comp_pcc(ref_speech[:n], tt_speech[:n], pcc=SPEECH_PCC)
    assert passed, (
        f"Speech PCC {pcc_val:.6f} < {SPEECH_PCC} (ref len={ref_speech.numel()}, "
        f"tt len={tt_speech.numel()}, token_match={token_match:.4f})"
    )
