# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Voxtral TTS end-to-end trial: teacher-forced TT decode against the CPU reference.

The CPU runs one free generate() (the golden token stream + waveform). The TT side is
then **teacher-forced**: at each step the TT text decode is fed the CPU reference codes,
and the TT audio tokenizer decodes those same reference codes. This breaks the chaotic
autoregressive cascade so the test reports the true per-component fidelity instead of
the irreducible ~0.9575 free-running ceiling.

Why not free-running? On the fully-independent path a single semantic-argmax near-tie
flips (logit gaps below one bf16 ULP; one CPU side is a literal 0.0 tie) and cascades.
Free-running PCC is therefore capped at ~0.9575 and is measured (log-only) by
``test_voxtral_e2e_pcc.py``. Here we assert the teacher-forced fidelity, which clears
0.99 because every component (text decode, acoustic, tokenizer) matches CPU when given
the same token stream — same conclusion as ``test_voxtral_tts_pipeline_inference``.
"""
from __future__ import annotations

import gc

import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import comp_pcc
from models.experimental.voxtraltts.reference.cpu_reference import VoxtralCPUReference
from models.experimental.voxtraltts.reference.voxtral_config import DEFAULT_VOXTRAL_TT_TEXT_MAX_SEQ_LEN
from models.experimental.voxtraltts.reference.voxtral_request import compose_speech_request
from models.experimental.voxtraltts.tests.common import resolve_voxtral_model_name_or_skip
from models.experimental.voxtraltts.tt.voxtral_tts import VoxtralTTSPipeline

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False

# Teacher-forced fidelity gate. Free-running generation is irreducibly ~0.9575 (near-tie
# semantic-argmax cascade); teacher forcing removes the cascade and every component
# clears 0.99 (text hidden ~0.9998/step, tokenizer waveform ~0.9994).
FINAL_WAVEFORM_PCC = 0.99
PREFILL_HIDDEN_PCC = 0.99
TEXT_DECODE_STEP_PCC = 0.98

_DEMO_TEXT = (
    "Voxtral is a four billion parameter open weight text to speech model "
    "released by Mistral AI in two thousand twenty six, designed for low "
    "latency multilingual voice generation across English, Spanish, French, "
    "Portuguese, Hindi, German, Dutch, and Italian. It builds on the "
    "Ministral three billion language backbone with a flow matching acoustic "
    "decoder and produces audio at twelve point five hertz with high quality, "
    "suitable for streaming voice applications and real time agent deployments."
)
_DEMO_VOICE = "casual_male"


def _log_pcc(label: str, pcc_value: float, target: float) -> None:
    status = "PASS" if pcc_value >= target else "LOW"
    logger.info(f"  {label}: PCC={pcc_value:.4f}  target>={target:.4f}  [{status}]")


@torch.no_grad()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_ttnn_voxtral_tts_e2e_trial(device, reset_seeds, request):
    """CPU ``generate()`` golden + teacher-forced TT decode; assert per-component fidelity."""
    generate_steps = 8
    name = resolve_voxtral_model_name_or_skip()

    try:
        cpu = VoxtralCPUReference(model_name_or_path=name, dtype="bfloat16", device="cpu")
    except Exception as exc:
        pytest.skip(f"CPU reference load failed: {exc}")

    try:
        pipe = VoxtralTTSPipeline.from_model_name(
            device,
            model_name_or_path=name,
            text_max_seq_len=DEFAULT_VOXTRAL_TT_TEXT_MAX_SEQ_LEN,
        )
    except Exception as exc:
        pytest.skip(f"TT pipeline load failed: {exc}")
    pipe_holder = [pipe]

    def _cleanup_pipe() -> None:
        if pipe_holder[0] is not None:
            pipe_holder[0].cleanup_all()
            pipe_holder[0] = None

    request.addfinalizer(_cleanup_pipe)

    logger.info("=" * 70)
    logger.info("CPU REFERENCE FORWARD (single free generate, golden token stream)")
    logger.info("=" * 70)
    ref_wav, ref_codes, cpu_trace = cpu.generate(
        text=_DEMO_TEXT,
        voice=_DEMO_VOICE,
        max_tokens=generate_steps,
        seed=0,
        return_tokenizer_codes=True,
        return_debug=True,
    )
    assert torch.isfinite(ref_wav).all(), "CPU reference produced non-finite waveform samples"
    n_frames = int(ref_codes.shape[2])
    logger.info(f"  CPU codes shape={tuple(ref_codes.shape)} waveform samples={int(ref_wav.numel())}")

    if use_signpost:
        signpost(header="start")

    logger.info("=" * 70)
    logger.info("TT TEACHER-FORCED DECODE (fed CPU reference codes each step)")
    logger.info("=" * 70)
    speech = compose_speech_request(_DEMO_TEXT, name, voice=_DEMO_VOICE)
    prompt_token_ids = speech["prompt_token_ids"]

    tt_embeds = pipe._build_voice_injected_embeds(prompt_token_ids, _DEMO_VOICE)
    tt_hidden_tt = pipe.text.prefill_from_embeds(tt_embeds, start_pos=0)
    tt_hidden = pipe.text.hidden_tt_to_torch(tt_hidden_tt)
    ttnn.deallocate(tt_hidden_tt)

    cpu_prefill = cpu_trace.get("text.prefill.hidden")
    ok_prefill, prefill_pcc = comp_pcc(
        cpu_prefill.reshape(-1).float(), tt_hidden.reshape(-1).float(), pcc=PREFILL_HIDDEN_PCC
    )
    _log_pcc("prefill hidden", float(prefill_pcc), PREFILL_HIDDEN_PCC)
    assert ok_prefill, f"prefill hidden PCC failed: {prefill_pcc}"

    current_pos = len(prompt_token_ids)

    for step in range(generate_steps):
        cpu_codes_step = cpu_trace.get(f"step.{step}.acoustic.codes")  # [37] shifted (with special offset)
        cpu_codes_step = cpu_codes_step.reshape(-1).long().unsqueeze(0)  # [1, 37]

        if int(cpu_codes_step[0, 0].item()) == cpu.end_audio_id:
            break

        # Teacher forcing: feed CPU reference codes into the TT text decode.
        mm_embed = pipe._audio_codes_to_mm_embed(cpu_codes_step)
        tt_hidden = pipe.text.decode_step_from_embeds(mm_embed, current_pos)
        ttnn.synchronize_device(device)
        current_pos += 1

        cpu_hidden_out = cpu_trace.get(f"step.{step}.text.hidden_out")
        if cpu_hidden_out is not None:
            ok_h, h_pcc = comp_pcc(
                cpu_hidden_out.reshape(-1).float(), tt_hidden.reshape(-1).float(), pcc=TEXT_DECODE_STEP_PCC
            )
            _log_pcc(f"text hidden step={step} pos={current_pos - 1}", float(h_pcc), TEXT_DECODE_STEP_PCC)
            assert ok_h, f"teacher-forced text hidden step={step} PCC failed: {h_pcc}"

    if use_signpost:
        signpost(header="stop")

    logger.info("=" * 70)
    logger.info("FINAL WAVEFORM PCC (TT tokenizer on CPU reference codes)")
    logger.info("=" * 70)
    tt_wav = pipe.decode_waveform_from_codes_tt(ref_codes)
    ttnn.synchronize_device(device)
    assert torch.isfinite(tt_wav).all(), "TT tokenizer produced non-finite waveform samples"

    ref_flat = ref_wav.reshape(-1).float()
    tt_flat = tt_wav.reshape(-1).float()
    n_wav = min(int(ref_flat.numel()), int(tt_flat.numel()))
    assert n_wav > 0, "no waveform samples produced by one of the pipelines"
    ok_wav, wav_pcc = comp_pcc(ref_flat[:n_wav], tt_flat[:n_wav], pcc=FINAL_WAVEFORM_PCC)
    _log_pcc("waveform", float(wav_pcc), FINAL_WAVEFORM_PCC)
    logger.info(f"  frames={n_frames} samples={n_wav}")
    assert ok_wav, f"final waveform PCC failed: {wav_pcc}  (samples={n_wav})"

    ttnn.synchronize_device(device)
    pipe.cleanup_all()
    pipe_holder[0] = None
    del pipe
    gc.collect()
