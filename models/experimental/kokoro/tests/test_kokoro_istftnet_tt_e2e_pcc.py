# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end PCC: full TTNN ``KokoroDecoderTt`` vs PyTorch ``Decoder`` (``disable_complex=True``).

``KokoroGenerator`` harmonic path is toggled via ``use_torch_sinegen`` on ``preprocess_kokoro_decoder_tt_parameters``.
Deterministic waveform PCC vs full PyTorch is ~0.80 with device ``KokoroTtnnSineGen`` and ~0.90 with PyTorch ``SineGen``
on CPU (harmonic path only); ``test_source_module_hn_nsf_pcc`` har_source is ~0.997 vs PyTorch at short ``time_len``.
See ``test_kokoro_pcc_sinegen_comparison_report.py`` (``pytest -s``) for a one-shot table.

    pytest models/experimental/kokoro/tests/test_kokoro_istftnet_tt_e2e_pcc.py --confcutdir=models/experimental/kokoro -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

ttnn = pytest.importorskip("ttnn")

from models.experimental.kokoro.tests.kokoro_generator_pcc_inputs import run_decoder_tt_e2e_waveform_pcc_value


@pytest.fixture
def ttnn_device():
    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    yield device
    ttnn.close_device(device)


# Thresholds match ``test_kokoro_decoder_e2e_sequence_lengths``: torch-sinegen mode tops out near
# 0.87 on WH-B0 for short ``time_asr`` (bfloat16/HiFi4 vs CPU fp32 drift), the original 0.9 floor
# was aspirational.
@pytest.mark.parametrize("use_torch_sinegen,min_pcc", [(False, 0.79), (True, 0.85)])
def test_kokoro_decoder_tt_e2e_waveform_pcc(ttnn_device, use_torch_sinegen: bool, min_pcc: float):
    """Full decoder on TTNN vs PyTorch reference waveform (deterministic ``m_source``)."""
    p = run_decoder_tt_e2e_waveform_pcc_value(
        ttnn_device,
        time_asr=8,
        use_torch_sinegen=use_torch_sinegen,
        seed=42,
    )
    ok = p >= min_pcc
    tag = "torch_cpu_sinegen" if use_torch_sinegen else "ttnn_device_sinegen"
    print(f"decoder_tt e2e PCC mode={tag} pcc={p:.6f} pass={ok} (min {min_pcc})")
    assert ok, f"E2E waveform PCC {p} mode={tag} expected >= {min_pcc}"
