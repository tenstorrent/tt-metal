# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Bark Small demo test — standard CI entry point.

Validates that the full Bark pipeline produces valid audio on device.
Uses the repo-level ``device`` fixture provided by conftest.py.

Usage:
    pytest models/demos/wormhole/bark/tests/test_bark_demo.py -v
"""

import numpy as np
import pytest

from models.demos.wormhole.bark.demo.demo import run_demo


@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "text, min_audio_duration",
    [
        ("Hello from Tenstorrent!", 0.5),
        ("Testing the Bark model on Wormhole hardware.", 0.5),
    ],
)
def test_demo(device, text, min_audio_duration):
    """Standard CI entry point for Bark Small demo.

    Runs the full text-to-audio pipeline and validates:
    - Output is a non-empty numpy array
    - Audio duration exceeds minimum threshold
    - Audio is finite (no NaN/Inf)
    - Audio is not silent
    """
    audio = run_demo(text=text, verbose=True, top_k=0, device=device)

    assert audio is not None, "Demo returned None"
    assert isinstance(audio, np.ndarray), f"Expected numpy array, got {type(audio)}"

    duration = len(audio) / 24000
    assert duration >= min_audio_duration, f"Audio too short: {duration:.2f}s (expected >= {min_audio_duration}s)"
    assert np.isfinite(audio).all(), "Audio contains NaN/Inf"
    assert np.max(np.abs(audio)) > 0.001, "Audio is silent"
