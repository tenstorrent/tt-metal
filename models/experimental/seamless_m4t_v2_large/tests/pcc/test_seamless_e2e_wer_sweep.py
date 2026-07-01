# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""E2E WER ISL sweep for speech-output tasks (T2ST, S2ST).

Devstral-style **input-length** sweep: for each N, build inputs of length N (source tokens for
T2ST; mel frames for S2ST — same as ``demo_perf_sweep.py``), run full TT
``generate(generate_speech=True)``, and compare intermediate translation text to offline HF
references via ``jiwer.wer`` (Whisper-demo pattern).

T2ST text inputs use *A Tale of Two Cities* from
``models/tt_transformers/tests/tale-of-two-cities.txt.bz2`` (via ``demo_perf_sweep.ensure_long_story``,
same corpus as tt-transformers / Devstral prefill tests). S2ST uses concatenated preamble audio.

Text-output tasks (T2TT, S2TT, ASR) use ``test_seamless_e2e_token_matching_sweep.py`` instead.

Run::

    pytest models/experimental/seamless_m4t_v2_large/tests/pcc/test_seamless_e2e_wer_sweep.py -k sanity
    pytest models/experimental/seamless_m4t_v2_large/tests/pcc/test_seamless_e2e_wer_sweep.py -k sweep

Run on P150 (1×1)::

    MESH_DEVICE=P150 pytest models/experimental/seamless_m4t_v2_large/tests/pcc/test_seamless_e2e_wer_sweep.py -k sanity

Run on Blackhole QB (1×4)::

    MESH_DEVICE=BH-QB pytest models/experimental/seamless_m4t_v2_large/tests/pcc/test_seamless_e2e_wer_sweep.py -k sanity

References auto-generate on first run (``SEAMLESS_SWEEP_AUTO_REF=1``, default). Pre-generate::

    python models/experimental/seamless_m4t_v2_large/scripts/generate_wer_sweep_reference.py --sweep --task all
"""

from __future__ import annotations

import pytest
import torch

from models.experimental.seamless_m4t_v2_large.tests.pcc.decoder_pcc_fixtures import load_hf_model_and_processor
from models.experimental.seamless_m4t_v2_large.tests.pcc.e2e_token_matching_helpers import SPEECH_OUTPUT_TASKS
from models.experimental.seamless_m4t_v2_large.tests.pcc.e2e_wer_helpers import (
    SANITY_SWEEP_LENGTHS,
    ensure_wer_sweep_reference,
    load_wer_sweep_reference,
    maybe_skip_empty_wer_reference,
    maybe_skip_short_s2st_wer,
    run_speech_output_wer,
    sweep_sequence_lengths,
    sweep_wer_threshold_for_task,
    weights_dir_or_skip,
    wer_sweep_mesh_parametrize,
)
from models.experimental.seamless_m4t_v2_large.tt.mesh_helpers import mesh_default_device, mesh_num_devices


def _mesh_key(mesh_device) -> str:
    return "1x1" if mesh_num_devices(mesh_device) == 1 else "1x4"


def _run_wer_sweep_point(mesh_device, device_params, task: str, seq_len: int) -> None:
    _ = device_params
    weights_dir = weights_dir_or_skip()
    ref_path = ensure_wer_sweep_reference(task, seq_len, weights_dir)
    wer_threshold = sweep_wer_threshold_for_task(task, seq_len, mesh_id=_mesh_key(mesh_device))
    log_label = f"{task.upper()}-len{seq_len}"
    ref = load_wer_sweep_reference(ref_path)
    maybe_skip_empty_wer_reference(ref.reference_text, task=task, seq_len=seq_len)
    maybe_skip_short_s2st_wer(task, ref.reference_text, seq_len)

    torch.manual_seed(0)
    with mesh_default_device(mesh_device):
        hf_model, _, tokenizer = load_hf_model_and_processor(weights_dir, dtype=torch.bfloat16)
        run_speech_output_wer(
            mesh_device,
            hf_model,
            tokenizer,
            ref,
            wer_threshold=wer_threshold,
            log_label=log_label,
        )


@pytest.mark.sanity
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("seq_len", list(SANITY_SWEEP_LENGTHS))
@pytest.mark.parametrize("task", list(SPEECH_OUTPUT_TASKS))
@pytest.mark.parametrize(*wer_sweep_mesh_parametrize(), indirect=["mesh_device", "device_params"])
def test_seamless_e2e_wer_sweep_sanity(mesh_device, device_params, reset_seeds, task, seq_len):
    """CI sanity: WER at input lengths 32, 64, 128 for T2ST and S2ST."""
    _ = reset_seeds
    _run_wer_sweep_point(mesh_device, device_params, task, seq_len)


@pytest.mark.sweep
@pytest.mark.timeout(10800)
@pytest.mark.parametrize("seq_len", sweep_sequence_lengths())
@pytest.mark.parametrize("task", list(SPEECH_OUTPUT_TASKS))
@pytest.mark.parametrize(*wer_sweep_mesh_parametrize(), indirect=["mesh_device", "device_params"])
def test_seamless_e2e_wer_sweep(mesh_device, device_params, reset_seeds, task, seq_len):
    """Nightly: WER at all README lengths (32→4096) for T2ST and S2ST."""
    _ = reset_seeds
    _run_wer_sweep_point(mesh_device, device_params, task, seq_len)
