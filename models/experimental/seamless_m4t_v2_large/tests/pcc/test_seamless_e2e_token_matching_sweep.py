# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""E2E token matching across README sequence lengths (32→4096, all five tasks).

Devstral-style **input-length** sweep: for each N, build inputs of length N (source tokens
or mel frames, same as ``demo_perf_sweep.py``), teacher-force decode for 128 steps, and
compare TT greedy top-1/top-5 to offline HF references.

Run::

    pytest models/experimental/seamless_m4t_v2_large/tests/pcc/test_seamless_e2e_token_matching_sweep.py -k sanity
    pytest models/experimental/seamless_m4t_v2_large/tests/pcc/test_seamless_e2e_token_matching_sweep.py -k sweep

References auto-generate on first run (``SEAMLESS_SWEEP_AUTO_REF=1``, default). Pre-generate::

    python models/experimental/seamless_m4t_v2_large/scripts/generate_t2tt_token_accuracy_reference.py --sweep --task all
"""

from __future__ import annotations

import pytest
import torch

from models.experimental.seamless_m4t_v2_large.tests.pcc.decoder_pcc_fixtures import load_hf_model_and_processor
from models.experimental.seamless_m4t_v2_large.tests.pcc.e2e_token_matching_helpers import (
    ALL_E2E_TASKS,
    SPEECH_INPUT_TASKS,
    TEXT_INPUT_TASKS,
    load_speech_token_accuracy_reference,
    load_t2tt_token_accuracy_reference,
    run_speech_e2e_token_accuracy,
    run_t2tt_e2e_token_accuracy,
    weights_dir_or_skip,
)
from models.experimental.seamless_m4t_v2_large.tests.pcc.e2e_token_matching_sweep_helpers import (
    SANITY_SWEEP_LENGTHS,
    SWEEP_EVAL_STEPS,
    ensure_sweep_reference,
    maybe_skip_short_speech_sweep,
    sweep_sequence_lengths,
    sweep_thresholds_for_task,
)
from models.experimental.seamless_m4t_v2_large.tt.mesh_helpers import MESH_DEVICE_PARAMETRIZE_TEXT, mesh_default_device


def _run_sweep_point(mesh_device, task: str, seq_len: int) -> None:
    weights_dir = weights_dir_or_skip()
    ref_path = ensure_sweep_reference(task, seq_len, weights_dir, max_decode_steps=SWEEP_EVAL_STEPS)
    top1_threshold, top5_threshold = sweep_thresholds_for_task(task, seq_len)
    log_label = f"{task.upper()}-len{seq_len}"

    torch.manual_seed(0)
    with mesh_default_device(mesh_device):
        hf_model, _, _ = load_hf_model_and_processor(weights_dir, dtype=torch.bfloat16)
        if task in TEXT_INPUT_TASKS:
            ref = load_t2tt_token_accuracy_reference(ref_path)
            run_t2tt_e2e_token_accuracy(
                mesh_device,
                hf_model,
                ref,
                decode_steps=SWEEP_EVAL_STEPS,
                top1_threshold=top1_threshold,
                top5_threshold=top5_threshold,
                log_label=log_label,
            )
        else:
            assert task in SPEECH_INPUT_TASKS
            ref = load_speech_token_accuracy_reference(ref_path)
            maybe_skip_short_speech_sweep(task, int(ref.teacher_tokens.numel()), seq_len)
            run_speech_e2e_token_accuracy(
                mesh_device,
                hf_model,
                ref,
                decode_steps=SWEEP_EVAL_STEPS,
                top1_threshold=top1_threshold,
                top5_threshold=top5_threshold,
                log_label=log_label,
            )


@pytest.mark.sanity
@pytest.mark.timeout(5400)
@pytest.mark.parametrize("seq_len", list(SANITY_SWEEP_LENGTHS))
@pytest.mark.parametrize("task", list(ALL_E2E_TASKS))
@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_TEXT, indirect=["mesh_device", "device_params"])
def test_seamless_e2e_token_matching_sweep_sanity(mesh_device, device_params, reset_seeds, task, seq_len):
    """CI sanity: token matching at input lengths 32, 64, 128 for all five tasks."""
    _ = reset_seeds
    _ = device_params
    _run_sweep_point(mesh_device, task, seq_len)


@pytest.mark.sweep
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("seq_len", sweep_sequence_lengths())
@pytest.mark.parametrize("task", list(ALL_E2E_TASKS))
@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_TEXT, indirect=["mesh_device", "device_params"])
def test_seamless_e2e_token_matching_sweep(mesh_device, device_params, reset_seeds, task, seq_len):
    """Nightly: token matching at all README lengths (32→4096) for all five tasks."""
    _ = reset_seeds
    _ = device_params
    _run_sweep_point(mesh_device, task, seq_len)
