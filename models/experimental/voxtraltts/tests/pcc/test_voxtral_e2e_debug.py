# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Voxtral TTS end-to-end trial: exactly one CPU forward and one TT forward.

Runs ``cpu.generate()`` once and ``forward_device_resident()`` once on the same
``text`` / ``voice`` / ``max_tokens`` / ``seed``. Pass/fail is final waveform PCC.

For per-step code match and aggregate stats, see logged ``PER-STEP CODE MATCH`` section.
For continuous per-stage PCC (``return_debug=True``), use ``log_voxtral_staged_pcc_report``.
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
from models.experimental.voxtraltts.tests.common import log_per_step_code_match, resolve_voxtral_model_name_or_skip
from models.experimental.voxtraltts.tt.voxtral_tts import VoxtralTTSPipeline
from models.experimental.voxtraltts.utils.debug_trace import log_voxtral_staged_pcc_report

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False

FINAL_WAVEFORM_PCC = 0.99

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
    """One CPU ``generate()`` + one TT ``forward_device_resident()``; assert waveform PCC."""
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
    logger.info("CPU REFERENCE FORWARD (single generate, return_debug trace)")
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
    logger.info(f"  CPU codes shape={tuple(ref_codes.shape)} waveform samples={int(ref_wav.numel())}")

    logger.info("=" * 70)
    logger.info("TT FORWARD (single forward_device_resident, return_debug trace)")
    logger.info("=" * 70)
    if use_signpost:
        signpost(header="start")
    tt_out = pipe.forward_device_resident(
        text=_DEMO_TEXT,
        voice=_DEMO_VOICE,
        max_tokens=generate_steps,
        seed=0,
        return_debug=True,
    )
    ttnn.synchronize_device(device)
    if use_signpost:
        signpost(header="stop")

    tt_wav = tt_out.waveform
    tt_codes = tt_out.codes_b37t
    assert torch.isfinite(tt_wav).all(), "TT forward produced non-finite waveform samples"
    assert tt_codes.dim() == 3 and tuple(tt_codes.shape[:2]) == (1, 37)
    logger.info(
        f"  TT codes shape={tuple(tt_codes.shape)} waveform shape={tuple(tt_wav.shape)} "
        f"hit_end_audio={tt_out.hit_end_audio}"
    )

    assert tt_out.debug is not None, "TT forward missing debug trace"
    logger.info("=" * 70)
    logger.info("PER-MODULE / PER-STAGE PCC (same forward as staged; trace only)")
    logger.info("=" * 70)
    staged = log_voxtral_staged_pcc_report(
        cpu_trace,
        tt_out.debug,
        target=FINAL_WAVEFORM_PCC,
        ref_waveform=ref_wav,
        tt_waveform=tt_wav,
    )
    if staged.first_low_stage is not None:
        logger.info(f"  → first stage below target {FINAL_WAVEFORM_PCC}: {staged.first_low_stage}")
    if staged.first_semantic_argmax_mismatch_step is not None:
        logger.info(f"  → first semantic-argmax mismatch at AR step {staged.first_semantic_argmax_mismatch_step}")

    logger.info("=" * 70)
    logger.info("FINAL WAVEFORM PCC")
    logger.info("=" * 70)
    n_frames = min(int(tt_codes.shape[2]), int(ref_codes.shape[2]))
    assert n_frames > 0, "no frames produced by one of the pipelines"
    tt_codes_aligned = tt_codes[:, :, :n_frames]
    ref_codes_aligned = ref_codes[:, :, :n_frames]

    log_per_step_code_match(ref_codes_aligned, tt_codes_aligned)

    sem_matches = int((tt_codes_aligned[:, 0] == ref_codes_aligned[:, 0]).sum().item())
    sem_total = int(tt_codes_aligned[:, 0].numel())
    ac_matches = int((tt_codes_aligned[:, 1:] == ref_codes_aligned[:, 1:]).sum().item())
    ac_total = int(tt_codes_aligned[:, 1:].numel())
    logger.info(f"  semantic-code match: {sem_matches / sem_total:.4f}  ({sem_matches}/{sem_total})")
    logger.info(f"  acoustic-code match: {ac_matches / ac_total:.4f}  ({ac_matches}/{ac_total})  (informational)")

    _, codes_pcc = comp_pcc(ref_codes_aligned.float(), tt_codes_aligned.float(), pcc=FINAL_WAVEFORM_PCC)
    logger.info(f"  codes PCC={float(codes_pcc):.4f}  (informational)")

    ref_flat = ref_wav.reshape(-1).float()
    tt_flat = tt_wav.reshape(-1).float()
    n_wav = min(int(ref_flat.numel()), int(tt_flat.numel()))
    assert n_wav > 0, "no waveform samples produced by one of the pipelines"
    ok_wav, wav_pcc = comp_pcc(ref_flat[:n_wav], tt_flat[:n_wav], pcc=FINAL_WAVEFORM_PCC)
    _log_pcc("waveform", float(wav_pcc), FINAL_WAVEFORM_PCC)
    assert ok_wav, f"final waveform PCC failed: {wav_pcc}  (samples={n_wav})"

    ttnn.synchronize_device(device)
    pipe.cleanup_all()
    pipe_holder[0] = None
    del pipe
    del tt_out
    gc.collect()
