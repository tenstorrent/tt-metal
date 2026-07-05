# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""E2E logits PCC ISL sweep for text-output tasks (T2TT, S2TT, ASR).

Devstral-style **input-length** sweep: for each N, build inputs of length N (source tokens
or mel frames, same as ``demo_perf_sweep.py`` / token-matching sweep), compare last-prefill
logits and up to ``LOGIT_PCC_DECODE_STEPS`` (10) decode logits (full vocab) between live HF
and TT. Speech tasks cap at ``min(10, len(teacher_tokens))`` when HF greedy hits EOS early.
Decode follows HF greedy (temperature=0 argmax) — the same token is fed to both models after
each logits comparison.

Speech-output tasks (T2ST, S2ST) use WER tests instead.

Run::

    pytest models/experimental/seamless_m4t_v2_large/tests/pcc/test_seamless_e2e_logit_pcc_sweep.py -k sanity -v
    pytest models/experimental/seamless_m4t_v2_large/tests/pcc/test_seamless_e2e_logit_pcc_sweep.py -k sweep -v

Reuses token-matching sweep ``.refpt`` inputs under
``tests/teacher_forced_sweep_outputs/references/`` (auto-generated on first run when
``SEAMLESS_SWEEP_AUTO_REF=1``, default).
"""

from __future__ import annotations

import pytest
import torch

from models.experimental.seamless_m4t_v2_large.tests.pcc.decoder_pcc_fixtures import load_hf_model_and_processor
from models.experimental.seamless_m4t_v2_large.tests.pcc.e2e_logit_pcc_helpers import (
    LOGIT_PCC_DECODE_STEPS,
    effective_logit_pcc_decode_steps,
    run_speech_e2e_logits_pcc_from_ref,
    run_t2tt_e2e_logits_pcc_from_ref,
)
from models.experimental.seamless_m4t_v2_large.tests.pcc.e2e_token_matching_helpers import (
    SPEECH_INPUT_TASKS,
    TEXT_INPUT_TASKS,
    TEXT_OUTPUT_TASKS,
    load_speech_token_accuracy_reference,
    load_t2tt_token_accuracy_reference,
    weights_dir_or_skip,
)
from models.experimental.seamless_m4t_v2_large.tests.pcc.e2e_token_matching_sweep_helpers import (
    SANITY_SWEEP_LENGTHS,
    SWEEP_EVAL_STEPS,
    ensure_sweep_reference,
    maybe_save_speech_sweep_mel_env,
    sweep_sequence_lengths,
)
from models.experimental.seamless_m4t_v2_large.tt.mesh_helpers import MESH_DEVICE_PARAMETRIZE_TEXT, mesh_default_device


def _run_sweep_point(mesh_device, task: str, seq_len: int) -> None:
    weights_dir = weights_dir_or_skip()
    ref_path = ensure_sweep_reference(task, seq_len, weights_dir, max_decode_steps=SWEEP_EVAL_STEPS)

    torch.manual_seed(0)
    with mesh_default_device(mesh_device):
        hf_model, _, _ = load_hf_model_and_processor(weights_dir, dtype=torch.bfloat16)
        if task in TEXT_INPUT_TASKS:
            ref = load_t2tt_token_accuracy_reference(ref_path)
            run_t2tt_e2e_logits_pcc_from_ref(
                mesh_device,
                hf_model,
                ref,
                seq_len=seq_len,
                decode_steps=LOGIT_PCC_DECODE_STEPS,
            )
        else:
            assert task in SPEECH_INPUT_TASKS
            ref = load_speech_token_accuracy_reference(ref_path)
            decode_steps = effective_logit_pcc_decode_steps(int(ref.teacher_tokens.numel()), task=task, seq_len=seq_len)
            maybe_save_speech_sweep_mel_env(
                task=task,
                seq_len=seq_len,
                input_features=ref.input_features,
                mel_attention_mask=ref.mel_attention_mask,
                seed_ids=ref.seed_ids,
                teacher_tokens=ref.teacher_tokens,
            )
            run_speech_e2e_logits_pcc_from_ref(
                mesh_device,
                hf_model,
                ref,
                task=task,
                seq_len=seq_len,
                decode_steps=decode_steps,
            )


@pytest.mark.sanity
@pytest.mark.timeout(5400)
@pytest.mark.parametrize("seq_len", list(SANITY_SWEEP_LENGTHS))
@pytest.mark.parametrize("task", list(TEXT_OUTPUT_TASKS))
@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_TEXT, indirect=["mesh_device", "device_params"])
def test_seamless_e2e_logit_pcc_sweep_sanity(mesh_device, device_params, reset_seeds, task, seq_len):
    """CI sanity: logits PCC at input lengths 32, 64, 128 for text-output tasks."""
    _ = reset_seeds
    _ = device_params
    _run_sweep_point(mesh_device, task, seq_len)


@pytest.mark.sweep
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("seq_len", sweep_sequence_lengths())
@pytest.mark.parametrize("task", list(TEXT_OUTPUT_TASKS))
@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_TEXT, indirect=["mesh_device", "device_params"])
def test_seamless_e2e_logit_pcc_sweep(mesh_device, device_params, reset_seeds, task, seq_len):
    """Nightly: logits PCC at all README lengths (32→4096) for text-output tasks."""
    _ = reset_seeds
    _ = device_params
    _run_sweep_point(mesh_device, task, seq_len)
