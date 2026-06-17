# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""E2E Full-Model Logit PCC Test (``models/tt_transformers/tests/test_model.py`` pattern).

Validates ``encoder → text_decoder (all layers) → final norm`` during teacher-forced greedy
decode. Per tt_transformers doc naming: "Logit PCC" compares activations **before** ``lm_head``
(not logits or tokens).

Scope:
  - All five tasks (T2TT, T2ST, S2TT, S2ST, ASR) via **text-decoder intermediates**
    (encoder hidden → decoder hidden before ``lm_head``). T2ST/S2ST stop before T2U/vocoder.
  - Fixed short prompt / mel input (encoder timeline ≤ 256)
  - Eager KV decode (no trace / 2CQ)
  - Quick mode: 3 decode steps; full mode: 9 decode steps

Out of scope: ``lm_head``, T2U, vocoder waveform/units, prefill-length sweeps.

Encoder PCC thresholds: text **0.99**; speech **0.97** (~0.978 observed at enc_seq≈256 on BH 1×4).
Decoder prefill/decode thresholds: text **0.99**; speech prefill **0.97** (S2TT/ASR) / **0.93** (S2ST);
speech decode **0.96** (S2TT/ASR) / **0.93** (S2ST).
"""

from __future__ import annotations

import pytest
import torch

from models.experimental.seamless_m4t_v2_large.tests.pcc.decoder_pcc_fixtures import load_hf_model_and_processor
from models.experimental.seamless_m4t_v2_large.tests.pcc.e2e_logit_pcc_helpers import (
    PCC_DECODE_FULL,
    PCC_DECODE_QUICK,
    PCC_DECODE_SPEECH_FULL,
    PCC_DECODE_SPEECH_QUICK,
    PCC_DECODE_S2ST,
    PCC_PREFILL_S2ST,
    PCC_PREFILL_SPEECH,
    run_speech_e2e_logit_pcc,
    run_t2tt_e2e_logit_pcc,
    weights_dir_or_skip,
)
from models.experimental.seamless_m4t_v2_large.tests.pcc.e2e_token_matching_helpers import (
    ALL_E2E_TASKS,
    SPEECH_INPUT_TASKS,
    TASK_TGT_LANG,
    TEXT_INPUT_TASKS,
)
from models.experimental.seamless_m4t_v2_large.tt.mesh_helpers import (
    MESH_DEVICE_PARAMETRIZE_TEXT,
    mesh_default_device,
)

_DECODE_STEPS = {
    "quick": 3,
    "full": 9,
}

_TASK_PREFILL_PCC = {
    "s2tt": PCC_PREFILL_SPEECH,
    "s2st": PCC_PREFILL_S2ST,
    "asr": PCC_PREFILL_SPEECH,
}


@pytest.mark.timeout(5400)
@pytest.mark.parametrize("task", list(ALL_E2E_TASKS))
@pytest.mark.parametrize("mode", ["quick", "full"], ids=["quick", "full"])
@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_TEXT, indirect=["mesh_device", "device_params"])
def test_seamless_e2e_logit_pcc(mesh_device, device_params, reset_seeds, task, mode):
    """E2E Full-Model Logit PCC: encoder + teacher-forced decoder vs HF (before ``lm_head``)."""
    _ = reset_seeds
    _ = device_params
    weights_dir = weights_dir_or_skip()
    decode_steps = _DECODE_STEPS[mode]
    if task in TEXT_INPUT_TASKS:
        pcc_decode = PCC_DECODE_QUICK if mode == "quick" else PCC_DECODE_FULL
    elif task == "s2st":
        pcc_decode = PCC_DECODE_S2ST
    else:
        assert task in SPEECH_INPUT_TASKS
        pcc_decode = PCC_DECODE_SPEECH_QUICK if mode == "quick" else PCC_DECODE_SPEECH_FULL
    tgt_lang = TASK_TGT_LANG[task]
    log_label = f"{task.upper()}-{mode}"

    torch.manual_seed(0)
    with mesh_default_device(mesh_device):
        hf_model, processor, _ = load_hf_model_and_processor(weights_dir, dtype=torch.bfloat16)
        if task in TEXT_INPUT_TASKS:
            run_t2tt_e2e_logit_pcc(
                mesh_device,
                hf_model,
                processor,
                tgt_lang=tgt_lang,
                decode_steps=decode_steps,
                pcc_decode=pcc_decode,
                log_label=log_label,
            )
        else:
            run_speech_e2e_logit_pcc(
                mesh_device,
                hf_model,
                processor,
                tgt_lang=tgt_lang,
                decode_steps=decode_steps,
                pcc_decode=pcc_decode,
                pcc_prefill=_TASK_PREFILL_PCC[task],
                log_label=log_label,
            )
