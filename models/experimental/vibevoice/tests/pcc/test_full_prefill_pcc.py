# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Full VibeVoice prefill-chain PCC vs HuggingFace reference.

Exercises the integrated prefill path that ``test_lm_prefill_pcc.py`` does not cover:

  voice audio → acoustic tokenizer encode → scale/bias → acoustic connector
  → scatter into text embeddings → LM prefill → last_hidden_state

Speech-embed and LM-hidden PCC are both gated at >= 0.99 (float32 comparison).
"""

import sys
from pathlib import Path

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.vibevoice.common.config import MODEL_PATH, TEXT_EXAMPLES_DIR, VOICES_DIR
from models.experimental.vibevoice.common.resource_utils import load_script
from models.experimental.vibevoice.tests.pcc.lm_pcc_common import (
    PCC_THRESHOLD,
    PREFILL_CHUNK_SIZE,
    compare_prefill_hidden_pcc,
)
from models.experimental.vibevoice.tt.ttnn_vibevoice_model import TTVibeVoiceModel

_VIBEVOICE_ROOT = Path(__file__).resolve().parent.parent.parent
_REFERENCE_DIR = _VIBEVOICE_ROOT / "reference"
for _p in (_REFERENCE_DIR, _VIBEVOICE_ROOT.parent.parent.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_TEXT_PATH = TEXT_EXAMPLES_DIR / "2p_short.txt"
_VOICE_PATH = VOICES_DIR / "en-Alice_woman.wav"
CFG_SCALE = 1.3
NUM_DIFFUSION_STEPS = 10


def _voice_path() -> str:
    if _VOICE_PATH.is_file():
        return str(_VOICE_PATH)
    wavs = list(VOICES_DIR.glob("*.wav"))
    assert wavs, f"No voice WAV in {VOICES_DIR}"
    return str(wavs[0])


def _build_processor_batch():
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

    assert _TEXT_PATH.is_file(), f"Missing demo text: {_TEXT_PATH}"
    script = load_script(_TEXT_PATH)
    processor = VibeVoiceProcessor.from_pretrained(MODEL_PATH)
    inputs = processor(
        text=[script],
        voice_samples=[[_voice_path()]],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    return processor, inputs


def reference_speech_embeds(ref_model, inputs: dict) -> torch.Tensor:
    with torch.no_grad():
        _, speech_embeds = ref_model._process_speech_inputs(
            inputs["speech_tensors"].to(ref_model.dtype),
            inputs["speech_masks"],
        )
    return speech_embeds.to(torch.float32)


def reference_full_prefill_hidden(ref_model, inputs: dict) -> torch.Tensor:
    with torch.no_grad():
        out = ref_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            speech_tensors=inputs["speech_tensors"].to(ref_model.dtype),
            speech_masks=inputs["speech_masks"],
            speech_input_mask=inputs["speech_input_mask"],
            use_cache=True,
            return_dict=True,
        )
    return out.last_hidden_state.to(torch.float32)


def _tt_prefill_hidden_from_embeds(lm_tt, inputs_embeds: ttnn.Tensor, kv_cache) -> torch.Tensor:
    """Return full-sequence LM hidden states for chunked embed prefill."""
    seq_len = inputs_embeds.shape[2]
    if seq_len <= PREFILL_CHUNK_SIZE:
        _, tt_hidden = lm_tt.prefill_embeds(inputs_embeds, kv_cache=kv_cache, return_last_hidden=True)
        return ttnn.to_torch(tt_hidden).to(torch.float32).squeeze(1)

    hidden_dim = inputs_embeds.shape[-1]
    hidden_parts = []
    for start in range(0, seq_len, PREFILL_CHUNK_SIZE):
        end = min(start + PREFILL_CHUNK_SIZE, seq_len)
        chunk = ttnn.slice(
            inputs_embeds,
            [0, 0, start, 0],
            [1, 1, end, hidden_dim],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        _, tt_hidden = lm_tt.forward(
            chunk,
            start_pos=start,
            kv_cache=kv_cache,
            return_last_hidden=True,
        )
        hidden_parts.append(ttnn.to_torch(tt_hidden).to(torch.float32).squeeze(1))
    return torch.cat(hidden_parts, dim=1)


def _make_prefill_generator(tt_model: TTVibeVoiceModel, processor):
    generator = tt_model._make_generator(
        processor.tokenizer,
        cfg_scale=CFG_SCALE,
        num_diffusion_steps=NUM_DIFFUSION_STEPS,
        max_new_tokens=1,
    )
    # Deterministic VAE mode — match reference ``std_dist_type="none"``.
    generator.acoustic_fix_std = 0.0
    return generator


def _tt_prefill_inputs_embeds(generator, inputs: dict):
    speech_tensors = inputs["speech_tensors"]
    speech_masks = inputs["speech_masks"]
    speech_input_mask = inputs["speech_input_mask"]

    speech_embeds = generator._process_speech_prefill(speech_tensors, speech_masks)
    inputs_embeds = generator._build_prefill_embeds(
        inputs["input_ids"],
        speech_tensors,
        speech_masks,
        speech_input_mask,
    )
    return speech_embeds, inputs_embeds


def tt_full_prefill_chain(tt_model: TTVibeVoiceModel, processor, inputs: dict):
    """Run TT acoustic encode → connector → embed scatter → LM prefill."""
    generator = _make_prefill_generator(tt_model, processor)
    speech_embeds, inputs_embeds = _tt_prefill_inputs_embeds(generator, inputs)

    prefill_len = inputs_embeds.shape[2]
    kv_cache = generator.lm.alloc_kv_cache(prefill_len + 8)
    lm_hidden = _tt_prefill_hidden_from_embeds(generator.lm, inputs_embeds, kv_cache)
    return speech_embeds, lm_hidden


def _load_ref_model():
    from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference

    ref_model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        device_map="cpu",
        attn_implementation="sdpa",
    )
    ref_model.eval()
    ref_model.set_ddpm_inference_steps(num_steps=NUM_DIFFUSION_STEPS)
    ref_model.model.acoustic_tokenizer.std_dist_type = "none"
    return ref_model


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_full_prefill_chain_pcc(mesh_device):
    """PCC for speech embeds and LM hidden states across the full voice-clone prefill."""
    processor, inputs = _build_processor_batch()
    prefill_len = inputs["input_ids"].shape[1]

    ref_model = _load_ref_model()

    ref_speech_embeds = reference_speech_embeds(ref_model, inputs)
    ref_hidden = reference_full_prefill_hidden(ref_model, inputs)

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

    tt_speech_embeds, tt_hidden = tt_full_prefill_chain(tt_model, processor, inputs)

    assert (
        ref_speech_embeds.shape == tt_speech_embeds.shape
    ), f"speech embed shape mismatch: ref={tuple(ref_speech_embeds.shape)} tt={tuple(tt_speech_embeds.shape)}"
    passed_embeds, pcc_embeds = comp_pcc(ref_speech_embeds, tt_speech_embeds, pcc=PCC_THRESHOLD)
    print(
        f"[test_full_prefill_pcc] speech_embeds shape={tuple(ref_speech_embeds.shape)} "
        f"PCC={pcc_embeds:.6f} threshold={PCC_THRESHOLD}"
    )
    assert passed_embeds, f"Full-prefill speech_embeds PCC {pcc_embeds:.6f} < {PCC_THRESHOLD}"

    passed_hidden, pcc_hidden, per_pos = compare_prefill_hidden_pcc(ref_hidden, tt_hidden, prefill_len)
    print(
        f"[test_full_prefill_pcc] prefill_len={prefill_len} overall_PCC={pcc_hidden:.6f} "
        f"last_token_PCC={per_pos[-1]:.5f} min_PCC={min(per_pos):.5f} threshold={PCC_THRESHOLD}"
    )
    assert passed_hidden, f"Full-prefill LM hidden PCC {pcc_hidden:.6f} < {PCC_THRESHOLD}"
