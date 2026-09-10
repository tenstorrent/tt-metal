# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Single eager speech-frame workload for Tracy / device-perf profiling.

Profiles **one AR speech-diffusion frame** on the eager path: neg-LM → diffusion
(CFG × steps) → post-diffusion chain → pos-LM → argmax.  Signposts are emitted
inside ``TTVibeVoiceGenerator.generate()`` when ``VV_PROFILE_SPEECH_FRAME=1``
(frame 1).

Standalone Tracy capture::

    VV_TRACE_SEGMENT=0 VV_PROFILE_SPEECH_FRAME=1 VV_PROFILE_SPEECH_FRAME_EXIT=1 \\
    python -m tracy -p -v -r --dump-device-data-mid-run --op-support-count 100000 -m pytest \\
        models/experimental/vibevoice/tests/perf/test_profile_single_step_decode.py \\
        ::test_profile_single_step_decode -v

Device perf CSV/JSON dump (outer driver spawns this under Tracy)::

    python models/experimental/vibevoice/tests/perf/test_device_perf_single_step_decode.py

Env:
  ``VV_DECODE_PERF_WARMUP_TOKENS`` — warmup AR steps before the measured frame (default 32).
"""

from __future__ import annotations

import os
import time

import pytest
import torch
import ttnn
from loguru import logger

from models.experimental.vibevoice.common.config import MODEL_PATH, TEXT_EXAMPLES_DIR, VOICES_DIR
from models.experimental.vibevoice.common.resource_utils import load_script
from models.experimental.vibevoice.tt.ttnn_vibevoice_model import TTVibeVoiceModel

_CFG_SCALE = 1.3
_NUM_DIFFUSION_STEPS = 10
_TEXT_PATH = TEXT_EXAMPLES_DIR / "1p_vibevoice.txt"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return int(raw)


def _voice_path() -> str:
    voice = VOICES_DIR / "en-Alice_woman.wav"
    if voice.is_file():
        return str(voice)
    wavs = list(VOICES_DIR.glob("*.wav"))
    assert wavs, f"No voice WAV in {VOICES_DIR}"
    return str(wavs[0])


def _build_processor_batch():
    from models.experimental.vibevoice.reference.processor.vibevoice_processor import VibeVoiceProcessor

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


def _generate_kwargs(processor, inputs: dict, *, max_new_tokens: int) -> dict:
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "speech_input_mask": inputs["speech_input_mask"],
        "speech_tensors": inputs["speech_tensors"],
        "speech_masks": inputs["speech_masks"],
        "tokenizer": processor.tokenizer,
        "cfg_scale": _CFG_SCALE,
        "num_diffusion_steps": _NUM_DIFFUSION_STEPS,
        "max_new_tokens": max_new_tokens,
    }


def _tracy_signpost_available() -> bool:
    try:
        from tracy import signpost  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.timeout(3600)
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_profile_single_step_decode(mesh_device, device_params, model_path):
    """Profile one warm eager speech-diffusion frame (generator signposts)."""
    del device_params
    use_signpost = _tracy_signpost_available()
    if not use_signpost:
        logger.info("tracy.signpost unavailable; running profile workload without signpost markers.")

    os.environ["VV_TRACE_SEGMENT"] = "0"
    if use_signpost:
        os.environ["VV_PROFILE_SPEECH_FRAME"] = "1"
        os.environ["VV_PROFILE_SPEECH_FRAME_EXIT"] = "1"
    else:
        os.environ.pop("VV_PROFILE_SPEECH_FRAME", None)
        os.environ.pop("VV_PROFILE_SPEECH_FRAME_EXIT", None)

    processor, inputs = _build_processor_batch()
    warmup_tokens = _env_int("VV_DECODE_PERF_WARMUP_TOKENS", 32)

    tt_model = TTVibeVoiceModel.from_checkpoint(
        mesh_device,
        model_path,
    )

    warm_kw = _generate_kwargs(processor, inputs, max_new_tokens=warmup_tokens)
    torch.manual_seed(0)
    _ = tt_model.generate(**warm_kw)
    ttnn.synchronize_device(mesh_device)
    ttnn.ReadDeviceProfiler(mesh_device)

    profile_kw = _generate_kwargs(processor, inputs, max_new_tokens=warmup_tokens)
    torch.manual_seed(0)
    t0 = time.perf_counter()
    out = tt_model.generate(**profile_kw)
    ttnn.synchronize_device(mesh_device)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    ttnn.ReadDeviceProfiler(mesh_device)

    assert out is not None
    logger.info(
        f"Profile workload complete: eager_speech_frame1, "
        f"signposts={'on' if use_signpost else 'off'}, host_wall_ms={wall_ms:.2f}"
    )
    if use_signpost:
        print(f"\nDECODE_STEP_HOST_WALL_MS={wall_ms:.2f}", flush=True)
