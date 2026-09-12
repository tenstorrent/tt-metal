# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Prefill-mode PCC gate for Qwen3-TTS: Talker + CodePredictor.

Talker prefill is the full 28-layer pass over the whole sequence (1024 in the
``max1024`` case), followed by ``codec_head`` at EVERY position. The
CodePredictor's prefill is its 2-position step -- the Talker hidden state at
position 0 and ``codec_embedding[code0]`` at position 1 -- producing code 1
through ``lm_head[0]``.

That asymmetry is architectural, not a shortcut: the CodePredictor's sequence
axis is the CODEBOOK INDEX (0..15), not time, so there is no "CP prefill over
1024 tokens" to run. Only the Talker has a time axis to prefill.

Method, thresholds, measurements and the reference chains all live in
``qwen3_tts_full_model_pcc_common.py`` — read its module docstring first. The
companion file is ``test_qwen3_tts_decode_pcc.py``.

Run (N300; ``MESH_DEVICE=N150`` for one chip)::

    MESH_DEVICE=N300 pytest -s models/demos/qwen3_tts/tests/test_qwen3_tts_prefill_pcc.py
"""

import pytest
import torch

from models.demos.qwen3_tts.tests.qwen3_tts_full_model_pcc_common import PREFILL_CASES, run_full_model_pcc

# pcc_device / pcc_model come from conftest.py — see the note there on why a
# fixture cannot be imported into two test modules.


@torch.no_grad()
@pytest.mark.timeout(5400)
@pytest.mark.parametrize("prefill_case", PREFILL_CASES)
def test_prefill_pcc(pcc_device, pcc_model, prefill_case):
    """Talker + CodePredictor in PREFILL mode.

    Talker prefill is the full 28-layer pass over the whole sequence (1024 in the
    ``max1024`` case). The CodePredictor's prefill is its 2-position step — the
    Talker hidden state at position 0 and ``codec_embedding[code0]`` at position 1,
    producing code 1 through ``lm_head[0]``.

    That asymmetry is architectural, not a shortcut. The CodePredictor's sequence
    axis is the CODEBOOK INDEX (0..15), not time: it predicts one frame's 16 codes
    from that frame's single Talker hidden row. There is no "CP prefill over 1024
    tokens" to run — see the module docstring.
    """
    run_full_model_pcc(pcc_device, pcc_model, prefill_case, phase="prefill")
