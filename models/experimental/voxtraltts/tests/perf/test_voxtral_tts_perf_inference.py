# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device-perf inference test: measures one forward_device_resident() call.

Model loading dispatches ~68k device ops which overflow the profiler DRAM
buffer (op_support_count=75000).  ttnn.ReadDeviceProfiler() is called after
loading to clear the buffer.  Signpost("start"/"stop") markers bracket the
measured inference so test_voxtral_tts_device_perf.py can filter to inference
ops only (has_signposts=True).
"""

from __future__ import annotations

import gc

import pytest
import torch
import ttnn

try:
    from tracy import signpost as _signpost
except ImportError:

    def _signpost(name: str) -> None:  # no-op when not running under Tracy
        pass


from models.experimental.voxtraltts.reference.voxtral_config import DEFAULT_VOXTRAL_TT_TEXT_MAX_SEQ_LEN
from models.experimental.voxtraltts.tests.common import resolve_voxtral_model_name_or_skip
from models.experimental.voxtraltts.tt.voxtral_tt_args import voxtral_text_default_optimizations
from models.experimental.voxtraltts.tt.voxtral_tts import VoxtralTTSPipeline

_DEMO_TEXT = "Voxtral is a text to speech model by Mistral AI released in two thousand twenty six."
_DEMO_VOICE = "casual_male"
_MAX_TOKENS = 2


@torch.no_grad()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_voxtral_tts_device_perf_inference(device, reset_seeds):
    """Single forward_device_resident() at max_tokens=2 for device-kernel perf measurement.

    ttnn.ReadDeviceProfiler() flushes the ~68k model-loading ops from the device
    DRAM profiler buffer before inference so only inference ops are captured.
    signpost("start"/"stop") brackets the measured region so test_voxtral_tts_device_perf.py
    can aggregate only inference kernel durations (has_signposts=True).
    """
    name = resolve_voxtral_model_name_or_skip()

    try:
        pipe = VoxtralTTSPipeline.from_model_name(
            device,
            model_name_or_path=name,
            text_max_seq_len=DEFAULT_VOXTRAL_TT_TEXT_MAX_SEQ_LEN,
            text_optimizations=voxtral_text_default_optimizations,
        )
    except Exception as exc:
        pytest.skip(f"TT pipeline load failed: {exc}")

    # Flush device profiler buffer so model-loading ops don't pollute the measurement.
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)

    try:
        _signpost("start")
        tt_out = pipe.forward_device_resident(
            text=_DEMO_TEXT,
            voice=_DEMO_VOICE,
            max_tokens=_MAX_TOKENS,
            seed=0,
        )
        ttnn.synchronize_device(device)
        _signpost("stop")
        # Drain inference ops before teardown — without this the C++ profiler
        # segfaults during device.close() when cleaning up overflow-marked buffers.
        ttnn.ReadDeviceProfiler(device)
        assert torch.isfinite(tt_out.waveform).all(), "TT forward produced non-finite waveform"
    finally:
        pipe.cleanup_all()
        del pipe
        gc.collect()
