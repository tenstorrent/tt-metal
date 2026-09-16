# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Chained end-to-end PCC gate for Qwen3-TTS: the whole device pipeline in one test.

Every stage, prefill and decode, asserted in a single run:

    speaker encoder (ECAPA, device mel)
      -> Talker prefill, 28 layers          -> codec_head at every position
      -> CodePredictor prefill              -> code 1
      -> CodePredictor decode x14           -> codes 2..15
      -> next-frame input embedding         -> Talker decode -> ...

Because the torch reference runs as an independent model, each stage's PCC
carries the error of everything upstream of it — which is the point. Per-block
numbers can all pass while the chain drifts.

The two companion files assert the same walk with one half of the stages each,
for when this one fails and you want to know which half:
``test_qwen3_tts_prefill_pcc.py`` and ``test_qwen3_tts_decode_pcc.py``.

Method, thresholds, measurements and the reference chains live in
``qwen3_tts_full_model_pcc_common.py`` — read its module docstring first.

Run (N300; ``MESH_DEVICE=N150`` for one chip)::

    MESH_DEVICE=N300 pytest -s models/demos/qwen3_tts/tests/test_qwen3_tts_full_model_pcc.py
"""

import pytest
import torch

from models.demos.qwen3_tts.tests.qwen3_tts_full_model_pcc_common import PREFILL_CASES, run_full_model_pcc

# pcc_device / pcc_model come from conftest.py — see the note there on why a
# fixture cannot be imported into two test modules.


@torch.no_grad()
@pytest.mark.timeout(5400)
@pytest.mark.parametrize("prefill_case", PREFILL_CASES)
def test_full_model_pcc(pcc_device, pcc_model, prefill_case):
    """Talker + CodePredictor, prefill AND decode, every stage asserted together."""
    run_full_model_pcc(pcc_device, pcc_model, prefill_case, phase="all")
