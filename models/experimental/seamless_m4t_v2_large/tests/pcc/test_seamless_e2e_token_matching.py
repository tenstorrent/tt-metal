# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""E2E Token Matching Test (``models/tt_transformers/demo/simple_text_demo.py`` pattern).

Compares TT ``lm_head`` greedy predictions to offline HuggingFace top-1 / top-5 during
teacher-forced decode. All five tasks via **text-decoder intermediates** (T2ST/S2ST stop
before T2U/vocoder).

Reference files under ``tests/reference_outputs/`` — generate via::

    python models/experimental/seamless_m4t_v2_large/scripts/generate_t2tt_token_accuracy_reference.py --task all

Requires eager KV decode (no trace / 2CQ).
"""

from __future__ import annotations

import pytest
import torch

from models.experimental.seamless_m4t_v2_large.tests.pcc.decoder_pcc_fixtures import load_hf_model_and_processor
from models.experimental.seamless_m4t_v2_large.tests.pcc.e2e_token_matching_helpers import (
    ALL_E2E_TASKS,
    SPEECH_INPUT_TASKS,
    SPEECH_TOP1_THRESHOLD,
    SPEECH_TOP5_THRESHOLD,
    S2ST_MIN_TOKEN_REF_STEPS,
    TEXT_INPUT_TASKS,
    T2TT_TOP1_THRESHOLD,
    T2TT_TOP5_THRESHOLD,
    load_speech_token_accuracy_reference,
    load_t2tt_token_accuracy_reference,
    refpt_or_skip,
    run_speech_e2e_token_accuracy,
    run_t2tt_e2e_token_accuracy,
    weights_dir_or_skip,
)
from models.experimental.seamless_m4t_v2_large.tt.mesh_helpers import MESH_DEVICE_PARAMETRIZE_TEXT, mesh_default_device

_DECODE_STEPS = {
    "quick": 32,
    "full": 128,
}


@pytest.mark.timeout(5400)
@pytest.mark.parametrize("task", list(ALL_E2E_TASKS))
@pytest.mark.parametrize("mode", ["quick", "full"], ids=["quick", "full"])
@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_TEXT, indirect=["mesh_device", "device_params"])
def test_seamless_e2e_token_matching(mesh_device, device_params, reset_seeds, task, mode):
    """E2E Token Matching Test: TT greedy top-1/top-5 vs offline HF reference (teacher forced)."""
    _ = reset_seeds
    _ = device_params
    weights_dir = weights_dir_or_skip()
    ref_path = refpt_or_skip(task)
    decode_steps = _DECODE_STEPS[mode]
    log_label = f"{task.upper()}-{mode}"

    if task in TEXT_INPUT_TASKS:
        top1_threshold = T2TT_TOP1_THRESHOLD
        top5_threshold = T2TT_TOP5_THRESHOLD
    else:
        assert task in SPEECH_INPUT_TASKS
        top1_threshold = SPEECH_TOP1_THRESHOLD
        top5_threshold = SPEECH_TOP5_THRESHOLD

    torch.manual_seed(0)
    with mesh_default_device(mesh_device):
        hf_model, _, _ = load_hf_model_and_processor(weights_dir, dtype=torch.bfloat16)
        if task in TEXT_INPUT_TASKS:
            ref = load_t2tt_token_accuracy_reference(ref_path)
            run_t2tt_e2e_token_accuracy(
                mesh_device,
                hf_model,
                ref,
                decode_steps=decode_steps,
                top1_threshold=top1_threshold,
                top5_threshold=top5_threshold,
                log_label=log_label,
            )
        else:
            ref = load_speech_token_accuracy_reference(ref_path)
            if task == "s2st" and int(ref.teacher_tokens.numel()) < S2ST_MIN_TOKEN_REF_STEPS:
                gen = "models/experimental/seamless_m4t_v2_large/scripts/generate_t2tt_token_accuracy_reference.py"
                pytest.skip(
                    f"S2ST reference has {ref.teacher_tokens.numel()} decode steps (need "
                    f">={S2ST_MIN_TOKEN_REF_STEPS}). Regenerate with preamble audio: "
                    f"python {gen} --task s2st"
                )
            run_speech_e2e_token_accuracy(
                mesh_device,
                hf_model,
                ref,
                decode_steps=decode_steps,
                top1_threshold=top1_threshold,
                top5_threshold=top5_threshold,
                log_label=log_label,
            )
