# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Production E2E PCC log: one CPU generate + one TT forward (prod path, no debug trace).

Same ``forward_device_resident()`` compute as ``test_ttnn_trial.py`` (trial passes
``return_debug=True`` only to collect staged-PCC tensors; numerics are identical).
Pipeline changes that move waveform PCC in trial will show the same number here.
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
from models.experimental.voxtraltts.tt.voxtral_tt_args import voxtral_text_high_accuracy_optimizations
from models.experimental.voxtraltts.tt.voxtral_tts import VoxtralTTSPipeline

WAVEFORM_PCC_TARGET = 0.99

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
def test_ttnn_voxtral_tts_staged_pcc(device, reset_seeds, request):
    """One CPU ``generate()`` + one TT ``forward_device_resident()`` (production path)."""
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
            text_optimizations=voxtral_text_high_accuracy_optimizations,
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
    logger.info("CPU FORWARD (single generate, production path)")
    logger.info("=" * 70)
    ref_wav, ref_codes = cpu.generate(
        text=_DEMO_TEXT,
        voice=_DEMO_VOICE,
        max_tokens=generate_steps,
        seed=0,
        return_tokenizer_codes=True,
    )
    assert torch.isfinite(ref_wav).all(), "CPU reference produced non-finite waveform samples"

    logger.info("=" * 70)
    logger.info("TT FORWARD (single forward_device_resident, production path)")
    logger.info("=" * 70)
    tt_out = pipe.forward_device_resident(
        text=_DEMO_TEXT,
        voice=_DEMO_VOICE,
        max_tokens=generate_steps,
        seed=0,
    )
    ttnn.synchronize_device(device)
    assert torch.isfinite(tt_out.waveform).all(), "TT forward produced non-finite waveform samples"

    n_frames = min(int(tt_out.codes_b37t.shape[2]), int(ref_codes.shape[2]))
    tt_codes = tt_out.codes_b37t[:, :, :n_frames]
    ref_codes_aligned = ref_codes[:, :, :n_frames]

    log_per_step_code_match(ref_codes_aligned, tt_codes)

    sem_matches = int((tt_codes[:, 0] == ref_codes_aligned[:, 0]).sum().item())
    sem_total = int(tt_codes[:, 0].numel())
    ac_matches = int((tt_codes[:, 1:] == ref_codes_aligned[:, 1:]).sum().item())
    ac_total = int(tt_codes[:, 1:].numel())
    logger.info(f"  semantic-code match: {sem_matches / sem_total:.4f}  ({sem_matches}/{sem_total})")
    logger.info(f"  acoustic-code match: {ac_matches / ac_total:.4f}  ({ac_matches}/{ac_total})")

    ref_flat = ref_wav.reshape(-1).float()
    tt_flat = tt_out.waveform.reshape(-1).float()
    n_wav = min(int(ref_flat.numel()), int(tt_flat.numel()))
    _, wav_pcc = comp_pcc(ref_flat[:n_wav], tt_flat[:n_wav], pcc=WAVEFORM_PCC_TARGET)
    _log_pcc("waveform", float(wav_pcc), WAVEFORM_PCC_TARGET)

    ttnn.synchronize_device(device)
    pipe.cleanup_all()
    pipe_holder[0] = None
    del pipe
    del tt_out
    gc.collect()
