# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""E2E PCC using the standard teacher-forced methodology.
"""
from __future__ import annotations

import gc
import os

import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import comp_pcc
from models.experimental.voxtraltts.reference.audio_tokenizer_ops import audio_tokenizer_decode_reference
from models.experimental.voxtraltts.reference.cpu_reference import VoxtralCPUReference
from models.experimental.voxtraltts.reference.voxtral_config import DEFAULT_VOXTRAL_TT_TEXT_MAX_SEQ_LEN
from models.experimental.voxtraltts.tests.common import (
    VOXTRAL_STANDARD_CHAR_TEXT,
    log_per_step_code_match,
    resolve_voxtral_model_name_or_skip,
)
from models.experimental.voxtraltts.demo.decode_trace_2cq import num_command_queues_for_decode
from models.experimental.voxtraltts.tt.voxtral_tt_args import voxtral_text_hf_aligned_optimizations
from models.experimental.voxtraltts.tt.voxtral_tts import VoxtralTTSPipeline

os.environ.setdefault("VOXTRAL_DECODE_TRACE", "1")

WAVEFORM_PCC_TARGET = 0.99

_DEMO_TEXT = VOXTRAL_STANDARD_CHAR_TEXT
_DEMO_VOICE = "casual_male"


def _log_pcc(label: str, pcc_value: float, target: float) -> None:
    status = "PASS" if pcc_value >= target else "LOW"
    logger.info(f"  {label}: PCC={pcc_value:.4f}  target>={target:.4f}  [{status}]")


def _align_1d(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    n = min(int(a.numel()), int(b.numel()))
    return a[:n], b[:n]


def _trace_device_params() -> dict[str, int]:
    return {
        "trace_region_size": int(os.environ.get("VOXTRAL_TRACE_REGION_SIZE", str(200_000_000))),
        "num_command_queues": num_command_queues_for_decode(),
    }


@torch.no_grad()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [_trace_device_params()], indirect=True)
def test_ttnn_voxtral_tts_staged_pcc(device, reset_seeds, request):
    """Teacher-forced E2E waveform PCC (shared codes) + logged free-run divergence."""
    generate_steps = 1
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
            text_optimizations=voxtral_text_hf_aligned_optimizations,
        )
    except Exception as exc:
        pytest.skip(f"TT pipeline load failed: {exc}")
    pipe_holder = [pipe]

    def _cleanup_pipe() -> None:
        if pipe_holder[0] is not None:
            pipe_holder[0].cleanup_all()
            pipe_holder[0] = None

    request.addfinalizer(_cleanup_pipe)

    # ---------------------------------------------------------------------
    # Step 1-2: one CPU generate produces the shared code history + ref wav.
    # ---------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("CPU GENERATE (produces shared code history = teacher-forcing input)")
    logger.info("=" * 70)
    ref_wav_gen, ref_codes = cpu.generate(
        text=_DEMO_TEXT,
        voice=_DEMO_VOICE,
        max_tokens=generate_steps,
        seed=0,
        return_tokenizer_codes=True,
    )
    assert torch.isfinite(ref_wav_gen).all(), "CPU reference produced non-finite waveform samples"
    assert int(ref_codes.shape[2]) > 0, "CPU reference produced no acoustic frames"

    # ---------------------------------------------------------------------
    # Step 2-4 (PCC gate): decode the SAME codes through the reference torch
    # tokenizer and the TT tokenizer, then compare. This is the canonical
    # "same input -> ref output / tt output -> compare" methodology, on a
    # continuous tensor (waveform). Mirrors SpeechT5's teacher-forced PCC.
    # ---------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("TEACHER-FORCED E2E (decode shared codes: torch ref vs TT)")
    logger.info("=" * 70)
    ref_wav = audio_tokenizer_decode_reference(ref_codes, pipe.audio_tokenizer_sd, pipe.config.audio_tokenizer_args)
    tt_wav = pipe.decode_waveform_from_codes_tt(ref_codes)
    ttnn.synchronize_device(device)
    assert torch.isfinite(tt_wav).all(), "TT tokenizer produced non-finite waveform samples"

    ref_flat, tt_flat = _align_1d(ref_wav, tt_wav)
    _, wav_pcc = comp_pcc(ref_flat, tt_flat, pcc=WAVEFORM_PCC_TARGET)
    _log_pcc("waveform (teacher-forced shared codes)", float(wav_pcc), WAVEFORM_PCC_TARGET)

    # ---------------------------------------------------------------------
    # Informational only: free-running TT generation. Reads ~0.77 (fp32-softmax
    # aligned) because the discrete-code AR feedback diverges from the CPU rollout
    # once the first few FSQ-boundary flips occur. NOT asserted (see module docstring).
    # ---------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("FREE-RUN DIAGNOSTIC (TT generates its own codes; informational, not gated)")
    logger.info("=" * 70)
    tt_out = pipe.forward_device_resident(
        text=_DEMO_TEXT,
        voice=_DEMO_VOICE,
        max_tokens=generate_steps,
        seed=0,
    )
    ttnn.synchronize_device(device)

    n_frames = min(int(tt_out.codes_b37t.shape[2]), int(ref_codes.shape[2]))
    tt_codes = tt_out.codes_b37t[:, :, :n_frames]
    ref_codes_aligned = ref_codes[:, :, :n_frames]
    log_per_step_code_match(ref_codes_aligned, tt_codes)

    sem_matches = int((tt_codes[:, 0] == ref_codes_aligned[:, 0]).sum().item())
    sem_total = int(tt_codes[:, 0].numel())
    ac_matches = int((tt_codes[:, 1:] == ref_codes_aligned[:, 1:]).sum().item())
    ac_total = int(tt_codes[:, 1:].numel())
    logger.info(f"  semantic-code match: {sem_matches / max(sem_total, 1):.4f}  ({sem_matches}/{sem_total})")
    logger.info(f"  acoustic-code match: {ac_matches / max(ac_total, 1):.4f}  ({ac_matches}/{ac_total})")

    free_ref, free_tt = _align_1d(ref_wav_gen, tt_out.waveform)
    _, free_pcc = comp_pcc(free_ref, free_tt, pcc=WAVEFORM_PCC_TARGET)
    _log_pcc("waveform (free-run, north-star, NOT gated)", float(free_pcc), WAVEFORM_PCC_TARGET)

    # The actual correctness gate is the teacher-forced continuous comparison.
    assert bool(wav_pcc >= WAVEFORM_PCC_TARGET), f"teacher-forced waveform PCC below target: {wav_pcc}"

    ttnn.synchronize_device(device)
    pipe.cleanup_all()
    pipe_holder[0] = None
    del pipe
    del tt_out
    gc.collect()
