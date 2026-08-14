# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Pytest wrapper around the Kokoro-82M TTNN demo (``demo.py``).

Follows the demo-test convention used by the other tt-metal demos (see
``models/demos/deepseek_v3/demo/test_demo.py``): every case drives the shared ``run_demo``
programmatic entrypoint and asserts the demo produced real audio, hit its parity floor and reported
the performance statistics it advertises. ``run_demo`` opens and closes the device itself (it sizes the
trace region from its own flags), so these tests take no device fixture.

The matrix covers the vocoder fallback / STFT-formulation configs (as in
``tests/test_tt_kmodel_mel_pcc.py``) plus the demo's own moving parts: metal tracing, the L1-activation
opt-in and the eager path. Note the split of responsibility — these cases gate that the *demo* still
produces sane audio and reports its statistics; the strict spectral/perceptual parity gates live in
``tests/test_tt_kmodel_{mel_pcc,asr_wer,speaker_cosine,cfw2vd}.py``, which score duration-matched audio
(see the mel-PCC note on the floors below).

Run::

    # single case (what CI runs)
    pytest "models/experimental/kokoro/demo/test_demo.py::test_demo[default]" -s --timeout=1200

    # host-only guard tests, no device needed
    pytest models/experimental/kokoro/demo/test_demo.py -k rejects_host_work

    # whole matrix — the two eager cases have no trace replay and dominate the runtime
    pytest models/experimental/kokoro/demo/test_demo.py -s --timeout=3600
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

# Upstream kokoro (G2P + chunking) and librosa (mel PCC) are demo-only deps:
# models/experimental/kokoro/requirements.txt. Skip rather than fail when absent.
pytest.importorskip("kokoro", reason='requires upstream kokoro: pip install "kokoro>=0.9.2"')
pytest.importorskip("librosa", reason="requires librosa for the mel-PCC parity check")

from models.experimental.kokoro.demo.demo import DEFAULT_VOICE, run_demo
from models.experimental.kokoro.tt.tt_kmodel import KokoroConfig

# Checkpoint resolution is delegated to the demo (``checkpoint=None`` -> the KOKORO_CHECKPOINT env var,
# which may be a file OR a directory to scan -> the HF hub cache -> download from HuggingFace). Passing
# the env var through as ``checkpoint=`` would reject the directory form, which run_demo requires to be
# a file, so leave it to find_local_checkpoint().
_SHORT_TEXT = "Hello from Tenstorrent Kokoro full TTNN."

# No multi-chunk case, deliberately. Upstream KPipeline packs sentences GREEDILY up to the 510-phoneme
# context cap instead of splitting per sentence, so forcing a second chunk needs > 510 phonemes of
# input — 2x476-phoneme chunks in the shortest text that splits. Measured on p150, that lands outside
# what the vocoder can run: the generator's harmonic-source ``ttnn.concat`` needs a 1,695,872 B CB page
# against 1,461,376 B of per-core L1 ("op cannot fit on this device"), and it is ~13 min per attempt.
# It is also far outside the README's documented optimized range (20-150 phonemes). Single-chunk cases
# still exercise the per-chunk loop; raise the model's length ceiling before adding a multi-chunk case.

# mel-PCC floors gate gross demo breakage (silence, garbage, wrong voice) — NOT spectral parity. This
# metric is FRAME-ALIGNED and the demo runs free (no teacher-forced durations), so any drift between
# the on-device ``pred_dur`` sum and the reference's shifts every later frame: for _SHORT_TEXT the
# device predicts 262 frames against the reference's 266, and that 4-frame shift alone pins mel PCC at
# ~0.83. Measured identically before and after this refactor, and identical traced (0.8297) vs eager
# with both CPU fallbacks on (0.8297) — i.e. the drift is in the prosody path and the vocoder config
# barely moves it. The strict > 0.95 spectral gate lives in tests/test_tt_kmodel_mel_pcc.py, which
# scores duration-matched audio.
#
# Floors sit ~0.08 below measured (per-case measurements in the matrix table below).
MEL_PCC_MIN_SHORT = 0.75  # measured 0.8297-0.8352 across every config below


def _demo_case(
    *,
    text: str,
    trace: bool,
    torch_stft_fallback: bool,
    torch_phase_fallback: bool,
    disable_complex: bool,
    l1_activations: bool,
    pcc_check: bool,
    mel_pcc_min: float | None,
    case_id: str,
    marks=None,
):
    return pytest.param(
        {
            "text": text,
            "trace": trace,
            "torch_stft_fallback": torch_stft_fallback,
            "torch_phase_fallback": torch_phase_fallback,
            "disable_complex": disable_complex,
            "l1_activations": l1_activations,
            "pcc_check": pcc_check,
            "mel_pcc_min": mel_pcc_min,
        },
        id=case_id,
        marks=marks or [],
    )


# Test matrix. ``trace`` is off for every config that does host work inside the decoder (the CPU
# fallbacks and the chunked CustomSTFT path) — run_demo rejects those combinations up front, because
# a host round-trip aborts the metal-trace capture (see test_demo_rejects_host_work_under_trace).
# +-----------------+-------------+-------+---------+----------+-----------------+---------+-----------+-------------+----------+
# | id              | text        | trace | stft fb | phase fb | disable_complex | l1 acts | pcc_check | mel_pcc_min | measured |
# +-----------------+-------------+-------+---------+----------+-----------------+---------+-----------+-------------+----------+
# | default         | short       | True  | off     | off      | False           | False   | True      | 0.75        | 0.8297   |
# | l1_activations  | short       | True  | off     | off      | False           | True    | True      | 0.75        | 0.8297   |
# | config_e        | short       | False | on      | on       | False           | False   | True      | 0.75        | 0.8297   |
# | disable_complex | short       | False | off     | off      | True            | False   | True      | 0.75        | 0.8352   |
# | no_trace        | short       | False | off     | off      | False           | False   | False     | None        | n/a      |
# +-----------------+-------------+-------+---------+----------+-----------------+---------+-----------+-------------+----------+


@pytest.mark.parametrize(
    # update the test matrix table above if new test cases are added
    "case",
    [
        # The CI case: the demo exactly as a user gets it (traced, fully on-device, PCC on).
        _demo_case(
            text=_SHORT_TEXT,
            trace=True,
            torch_stft_fallback=False,
            torch_phase_fallback=False,
            disable_complex=False,
            l1_activations=False,
            pcc_check=True,
            mel_pcc_min=MEL_PCC_MIN_SHORT,
            case_id="default",
            marks=pytest.mark.timeout(1800),
        ),
        # L1-resident generator loop: a perf opt-in that must stay PCC-neutral, so it carries the
        # same floor as `default` (both measured 0.8297).
        _demo_case(
            text=_SHORT_TEXT,
            trace=True,
            torch_stft_fallback=False,
            torch_phase_fallback=False,
            disable_complex=False,
            l1_activations=True,
            pcc_check=True,
            mel_pcc_min=MEL_PCC_MIN_SHORT,
            case_id="l1_activations",
            marks=pytest.mark.timeout(1800),
        ),
        # Config E — both CPU fallbacks, the highest-parity vocoder path. Eager only: the fallbacks
        # read/compute on the host inside the decoder, which a trace capture forbids.
        _demo_case(
            text=_SHORT_TEXT,
            trace=False,
            torch_stft_fallback=True,
            torch_phase_fallback=True,
            disable_complex=False,
            l1_activations=False,
            pcc_check=True,
            mel_pcc_min=MEL_PCC_MIN_SHORT,
            case_id="config_e",
            marks=pytest.mark.timeout(2400),
        ),
        # On-device CustomSTFT formulation. Eager only: its chunked iSTFT branch round-trips each
        # conv_transpose2d chunk through torch.
        _demo_case(
            text=_SHORT_TEXT,
            trace=False,
            torch_stft_fallback=False,
            torch_phase_fallback=False,
            disable_complex=True,
            l1_activations=False,
            pcc_check=True,
            mel_pcc_min=MEL_PCC_MIN_SHORT,
            case_id="disable_complex",
            marks=pytest.mark.timeout(2400),
        ),
        # Eager decoder: no trace capture/replay, so this is the slow path (trace replay is 74-301x
        # faster). PCC is skipped here — parity is covered by the cases above; this case exists to
        # keep the --no-trace code path from rotting.
        _demo_case(
            text=_SHORT_TEXT,
            trace=False,
            torch_stft_fallback=False,
            torch_phase_fallback=False,
            disable_complex=False,
            l1_activations=False,
            pcc_check=False,
            mel_pcc_min=None,
            case_id="no_trace",
            marks=pytest.mark.timeout(2400),
        ),
    ],
)
def test_demo(case: dict, tmp_path):
    out_wav = tmp_path / "kokoro_demo.wav"

    results = run_demo(
        case["text"],
        voice=DEFAULT_VOICE,
        output_path=out_wav,
        trace=case["trace"],
        torch_stft_fallback=case["torch_stft_fallback"],
        torch_phase_fallback=case["torch_phase_fallback"],
        disable_complex=case["disable_complex"],
        l1_activations=case["l1_activations"],
        pcc_check=case["pcc_check"],
    )

    generations = results["generations"]
    statistics = results["statistics"]

    # --- Chunk records ---
    assert generations, "demo produced no chunk records"
    assert len(generations) == statistics["chunks"]
    for record in generations:
        assert record["phonemes"] > 0
        assert record["samples"] > 0
        assert record["infer_s"] > 0
        assert record["rtf"] > 0

    # --- Audio artifact: the demo's actual product ---
    assert results["audio_path"] is not None and out_wav.exists(), "demo did not write the output WAV"
    audio, sample_rate = sf.read(str(out_wav))
    assert sample_rate == KokoroConfig.sample_rate_hz
    assert audio.shape[0] == statistics["audio_samples"]
    assert np.isfinite(audio).all(), "generated audio has NaN/Inf"
    assert float(np.abs(audio).max()) > 1e-3, "generated audio is silent"

    # --- Performance statistics the demo advertises ---
    assert statistics["input_characters"] > 0
    assert statistics["generated_audio_s"] > 0
    assert statistics["total_latency_s"] > 0
    assert statistics["time_to_first_audio_s"] > 0
    assert np.isfinite(statistics["real_time_factor"]) and statistics["real_time_factor"] > 0
    assert np.isfinite(statistics["throughput_char_s"]) and statistics["throughput_char_s"] > 0
    assert statistics["program_cache_entries"] > 0, "program cache was not populated"

    # --- Trace bookkeeping: warmup must capture one trace per distinct aligned length ---
    if case["trace"]:
        assert statistics["warmup_s"] is not None
        assert statistics["trace_captures_decoder"] >= 1, "trace warmup captured no decoder trace"
    else:
        assert statistics["warmup_s"] is None
        assert statistics["trace_captures_decoder"] == 0

    # --- Parity gate vs the reference CPU KModel ---
    if case["mel_pcc_min"] is not None:
        assert results["reference_audio_path"] is not None, "pcc_check did not write the reference WAV"
        assert statistics["mel_pcc_min"] is not None
        for record in generations:
            assert record["mel_pcc"] >= case["mel_pcc_min"], (
                f"chunk {record['index']} mel PCC {record['mel_pcc']:.4f} below "
                f"gate {case['mel_pcc_min']} (log-mel, phase/shift-tolerant)"
            )
    else:
        assert statistics["mel_pcc_mean"] is None
        assert results["reference_audio_path"] is None


@pytest.mark.parametrize(
    "flag",
    ["torch_stft_fallback", "torch_phase_fallback", "disable_complex"],
)
def test_demo_rejects_host_work_under_trace(flag: str):
    """Host work inside the traced decoder must be rejected up front, not abort mid-capture.

    Each of these paths does a device<->host round-trip inside the decoder (``ttnn.to_torch`` in the
    SineGen phase chain, host ``torch.stft``, the chunked CustomSTFT iSTFT branch), which trace
    capture forbids -- on hardware it aborts with an opaque
    ``TT_FATAL ... Reads/Writes are not supported during trace capture`` deep in the vocoder. Runs
    host-side only: run_demo validates before it opens a device.
    """
    with pytest.raises(SystemExit, match="cannot be combined with trace=True"):
        run_demo(_SHORT_TEXT, trace=True, output_path=None, **{flag: True})
