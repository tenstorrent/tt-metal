# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Decode-mode PCC gate for Qwen3-TTS: Talker + CodePredictor.

Walks ``QWEN3_TTS_PCC_FRAMES`` autoregressive frames (default 24). Each frame is
the Talker's single-token step against the full KV cache, its ``codec_head``, and
the CodePredictor's decode steps for codes 2..15.

The prefill runs first because decode needs a populated KV cache, but only
decode-mode stages are asserted -- the prefill stages are owned by
``test_qwen3_tts_prefill_pcc.py``.

Method, thresholds, measurements and the reference chains all live in
``qwen3_tts_full_model_pcc_common.py`` — read its module docstring first. The
companion file is ``test_qwen3_tts_prefill_pcc.py``.

Run (N300; ``MESH_DEVICE=N150`` for one chip)::

    MESH_DEVICE=N300 pytest -s models/demos/qwen3_tts/tests/test_qwen3_tts_decode_pcc.py
"""

import pytest
import torch

from models.demos.qwen3_tts.tests.qwen3_tts_full_model_pcc_common import PREFILL_CASES, run_full_model_pcc

# pcc_device / pcc_model come from conftest.py — see the note there on why a
# fixture cannot be imported into two test modules.


@torch.no_grad()
@pytest.mark.timeout(5400)
@pytest.mark.parametrize("prefill_case", PREFILL_CASES)
def test_decode_pcc(pcc_device, pcc_model, prefill_case):
    """Talker + CodePredictor in DECODE mode, walked over QWEN3_TTS_PCC_FRAMES frames.

    Runs the prefill first because decode needs a populated KV cache, but asserts
    only the decode-mode stages: the Talker's single-token step against the full
    cache, its codec_head, and the CodePredictor's decode steps for codes 2..15.
    """
    run_full_model_pcc(pcc_device, pcc_model, prefill_case, phase="decode")
