# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""E2E WER ISL sweep for T2ST, S2ST, and ASR.

Devstral-style **input-length** sweep: for each N, build inputs of length N (source tokens for
T2ST; mel frames for S2ST/ASR — same as ``demo_perf_sweep.py``), run the TT pipeline and compare
the output text to offline HF references via ``jiwer.wer`` (Whisper-demo pattern).

T2ST text inputs use *A Tale of Two Cities* from
``models/tt_transformers/tests/tale-of-two-cities.txt.bz2`` (via ``demo_perf_sweep.ensure_long_story``,
same corpus as tt-transformers). S2ST/ASR audio is length-dependent: mel
``<= S2ST_PREAMBLE_MAX_MEL`` (128) uses the demo preamble (its short opening translates cleanly);
longer mel uses distinct *LibriSpeech* utterances (coherent, non-repeating) so references are long
enough for a stable WER rather than a degenerate repeated loop.

Text-output translation tasks T2TT / S2TT use ``test_seamless_e2e_token_matching_sweep.py`` instead.

This file has two flavors:

* **Teacher-forced** (``test_seamless_e2e_teacher_forced_wer_sweep*``): HF reference tokens are fed
  to the TT decoder step-by-step, so no cascade — a stable *fidelity* gate. Select with
  ``-k teacher_forced``.
* **Whisper round-trip** (``test_seamless_e2e_whisper_wer_sweep*``): for the speech-output tasks
  (T2ST, S2ST) this is **pure vocoder fidelity** — HF's *exact* vocoder input (the offset-applied
  unit ids HF fed to its own vocoder) is fed straight into the TT vocoder
  (``tt_model.vocode_units(...)``), the TT waveform is transcribed with Whisper, and WER'd against the
  HF waveform's transcription. Both the text decode AND the T2U (text→units) are teacher-forced from
  HF, so the only variable is the TT vocoder (+ Whisper ASR). **ASR** emits text (no waveform to
  transcribe), so its whisper-sweep point falls back to the teacher-forced text-decoder WER
  (``run_speech_output_teacher_forced_wer``) — same gate as the teacher-forced flavor. Select with
  ``-k whisper``. (voxtral ``test_voxtral_e2e_quality_metrics`` uses the same transcribe-and-WER idea.)

  Both speech-output tasks target **Spanish** here (``WER_TASK_TGT_LANG``: T2ST=spa, S2ST=spa) so the
  Whisper round-trip is a faithful metric. whisper-large-v3 transcribes Spanish reliably; on Hindi it
  amplifies perceptually-inaudible vocoder deltas into huge word divergence (measured: TT-vs-HF vocoder
  log-mel PCC 0.994 on Hindi yet whisper WER 1.77, vs Spanish 0.03) — so T2ST uses Spanish, unlike the
  token-matching suite's Hindi. (The teacher-forced text flavor shares the same refpts and therefore
  also runs T2ST as eng→spa here.)

Teacher-forced WER (fidelity gate; HF tokens fed to the TT decoder)::

    # full sweep (32→4096), all tasks (T2ST + S2ST + ASR = 24 points):
    MESH_DEVICE=BH-QB pytest models/experimental/seamless_m4t_v2_large/tests/pcc/test_seamless_e2e_wer_sweep.py -k "teacher_forced and not sanity" -s
    # CI sanity only (lengths 32/64/128):
    MESH_DEVICE=BH-QB pytest models/experimental/seamless_m4t_v2_large/tests/pcc/test_seamless_e2e_wer_sweep.py -k "teacher_forced and sanity" -s
    # one task / one length:
    MESH_DEVICE=BH-QB pytest models/experimental/seamless_m4t_v2_large/tests/pcc/test_seamless_e2e_wer_sweep.py -k "teacher_forced and not sanity and t2st-512" -s
    MESH_DEVICE=BH-QB pytest models/experimental/seamless_m4t_v2_large/tests/pcc/test_seamless_e2e_wer_sweep.py -k "teacher_forced and not sanity and s2st" -s

Whisper round-trip WER (transcribe generated speech for T2ST/S2ST; teacher-forced text for ASR)::

    MESH_DEVICE=BH-QB pytest models/experimental/seamless_m4t_v2_large/tests/pcc/test_seamless_e2e_wer_sweep.py -k "whisper and not sanity" -s

Mesh selection: ``MESH_DEVICE=P150`` → 1×1, ``MESH_DEVICE=BH-QB`` → 1×4 (unset → both when hardware allows).

References auto-generate on first run (``SEAMLESS_SWEEP_AUTO_REF=1``, default). Pre-generate::

    python models/experimental/seamless_m4t_v2_large/scripts/generate_wer_sweep_reference.py --sweep --task all
"""

from __future__ import annotations

import pytest
import torch

from models.experimental.seamless_m4t_v2_large.tests.pcc.decoder_pcc_fixtures import load_hf_model_and_processor
from models.experimental.seamless_m4t_v2_large.tests.pcc.e2e_token_matching_helpers import (
    SANITY_SWEEP_LENGTHS,
    sweep_sequence_lengths,
)
from models.experimental.seamless_m4t_v2_large.tests.pcc.e2e_wer_helpers import (
    WER_SWEEP_TASKS,
    ensure_wer_sweep_reference,
    load_wer_sweep_reference,
    maybe_skip_empty_wer_reference,
    maybe_skip_short_speech_wer,
    run_speech_output_teacher_forced_wer,
    run_speech_output_whisper_wer,
    sweep_teacher_forced_wer_threshold_for_task,
    sweep_whisper_wer_threshold_for_task,
    wer_sweep_mesh_parametrize,
)
from models.experimental.seamless_m4t_v2_large.tests.pcc.pcc_test_common import weights_dir_or_skip
from models.experimental.seamless_m4t_v2_large.tt.mesh_helpers import mesh_default_device, mesh_num_devices


def _mesh_key(mesh_device) -> str:
    return "1x1" if mesh_num_devices(mesh_device) == 1 else "1x4"


def _run_teacher_forced_wer_sweep_point(mesh_device, device_params, task: str, seq_len: int) -> None:
    _ = device_params
    weights_dir = weights_dir_or_skip()
    ref_path = ensure_wer_sweep_reference(task, seq_len, weights_dir)
    wer_threshold = sweep_teacher_forced_wer_threshold_for_task(task, seq_len, mesh_id=_mesh_key(mesh_device))
    log_label = f"{task.upper()}-len{seq_len}-TF"
    ref = load_wer_sweep_reference(ref_path)
    maybe_skip_empty_wer_reference(ref.reference_text, task=task, seq_len=seq_len)
    maybe_skip_short_speech_wer(task, ref.reference_text, seq_len)

    torch.manual_seed(0)
    with mesh_default_device(mesh_device):
        hf_model, _, tokenizer = load_hf_model_and_processor(weights_dir, dtype=torch.bfloat16)
        run_speech_output_teacher_forced_wer(
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
@pytest.mark.parametrize("task", list(WER_SWEEP_TASKS))
@pytest.mark.parametrize(*wer_sweep_mesh_parametrize(), indirect=["mesh_device", "device_params"])
def test_seamless_e2e_teacher_forced_wer_sweep_sanity(mesh_device, device_params, reset_seeds, task, seq_len):
    """CI sanity: teacher-forced (fidelity) WER at lengths 32, 64, 128 for T2ST, S2ST, ASR.

    Select with ``-k teacher_forced``. Feeds HF reference tokens to the TT decoder so a single
    low-margin token flip cannot cascade — a stable fidelity gate, unlike free-running WER.
    """
    _ = reset_seeds
    _run_teacher_forced_wer_sweep_point(mesh_device, device_params, task, seq_len)


@pytest.mark.sweep
@pytest.mark.timeout(10800)
@pytest.mark.parametrize("seq_len", sweep_sequence_lengths())
@pytest.mark.parametrize("task", list(WER_SWEEP_TASKS))
@pytest.mark.parametrize(*wer_sweep_mesh_parametrize(), indirect=["mesh_device", "device_params"])
def test_seamless_e2e_teacher_forced_wer_sweep(mesh_device, device_params, reset_seeds, task, seq_len):
    """Nightly: teacher-forced (fidelity) WER at all README lengths (32→4096) for T2ST, S2ST, ASR."""
    _ = reset_seeds
    _run_teacher_forced_wer_sweep_point(mesh_device, device_params, task, seq_len)


def _run_whisper_wer_sweep_point(mesh_device, device_params, task: str, seq_len: int) -> None:
    _ = device_params
    weights_dir = weights_dir_or_skip()
    ref_path = ensure_wer_sweep_reference(task, seq_len, weights_dir)
    ref = load_wer_sweep_reference(ref_path)
    maybe_skip_empty_wer_reference(ref.reference_text, task=task, seq_len=seq_len)
    maybe_skip_short_speech_wer(task, ref.reference_text, seq_len)

    torch.manual_seed(0)
    with mesh_default_device(mesh_device):
        hf_model, _, tokenizer = load_hf_model_and_processor(weights_dir, dtype=torch.bfloat16)
        if task == "asr":
            # ASR emits text, not speech — there is no TT waveform for Whisper to transcribe. Fall
            # back to the teacher-forced text-decoder WER so the whisper sweep still gates ASR.
            wer_threshold = sweep_teacher_forced_wer_threshold_for_task(task, seq_len, mesh_id=_mesh_key(mesh_device))
            run_speech_output_teacher_forced_wer(
                mesh_device,
                hf_model,
                tokenizer,
                ref,
                wer_threshold=wer_threshold,
                log_label=f"{task.upper()}-len{seq_len}-WHISPER-TF",
            )
        else:
            wer_threshold = sweep_whisper_wer_threshold_for_task(task, seq_len, mesh_id=_mesh_key(mesh_device))
            run_speech_output_whisper_wer(
                mesh_device,
                hf_model,
                tokenizer,
                ref,
                wer_threshold=wer_threshold,
                log_label=f"{task.upper()}-len{seq_len}-WHISPER",
            )


@pytest.mark.sanity
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("seq_len", list(SANITY_SWEEP_LENGTHS))
@pytest.mark.parametrize("task", list(WER_SWEEP_TASKS))
@pytest.mark.parametrize(*wer_sweep_mesh_parametrize(), indirect=["mesh_device", "device_params"])
def test_seamless_e2e_whisper_wer_sweep_sanity(mesh_device, device_params, reset_seeds, task, seq_len):
    """CI sanity: whisper round-trip WER at lengths 32, 64, 128 for T2ST, S2ST, ASR.

    Select with ``-k whisper``. For T2ST/S2ST, HF's exact vocoder input is fed to the TT vocoder,
    the TT speech is transcribed with Whisper, and WER'd vs the HF speech transcription — isolating
    the vocoder (+ Whisper ASR). ASR has no waveform, so it uses the teacher-forced text-decoder WER.
    """
    _ = reset_seeds
    _run_whisper_wer_sweep_point(mesh_device, device_params, task, seq_len)


@pytest.mark.sweep
@pytest.mark.timeout(10800)
@pytest.mark.parametrize("seq_len", sweep_sequence_lengths())
@pytest.mark.parametrize("task", list(WER_SWEEP_TASKS))
@pytest.mark.parametrize(*wer_sweep_mesh_parametrize(), indirect=["mesh_device", "device_params"])
def test_seamless_e2e_whisper_wer_sweep(mesh_device, device_params, reset_seeds, task, seq_len):
    """Nightly: whisper round-trip WER at all README lengths (32→4096) for T2ST, S2ST, ASR."""
    _ = reset_seeds
    _run_whisper_wer_sweep_point(mesh_device, device_params, task, seq_len)
