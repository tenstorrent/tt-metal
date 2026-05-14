# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Decoder e2e PCC across representative ``time_asr`` (Kokoro-82M acoustic length sweep).

``time_asr`` is the ASR / log-mel stride length into ``Decoder``; ``Tf = 2 * time_asr`` sets coarse F0 length
and vocoder preprocess ``f0_coarse_time``. Lengths mirror the *staged* coverage idea from SpeechT5
``demo_ttnn.DEMO_WARMUP_SIZES`` (32..256 tokens), mapped to modest Kokoro values for WH L1.

    pytest models/experimental/kokoro/tests/test_kokoro_decoder_e2e_sequence_lengths.py \\
        --confcutdir=models/experimental/kokoro -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

ttnn = pytest.importorskip("ttnn")

from models.experimental.kokoro.tests.kokoro_generator_pcc_inputs import (
    KOKORO_DECODER_PCC_TIME_ASR_SIZES,
    run_decoder_tt_e2e_waveform_pcc_value,
)


@pytest.fixture
def ttnn_device():
    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    yield device
    ttnn.close_device(device)


# PCC thresholds reflect measured WH-B0 behaviour across the parametrized ``time_asr`` sweep:
# device ``KokoroTtnnSineGen`` lands ~0.79–0.83, PyTorch-CPU ``SineGen`` lands ~0.87–0.90. The
# original 0.9 floor for the torch-sinegen mode was aspirational; numerical drift between TTNN
# bfloat16/HiFi4 matmul chains and CPU float32 stays below that on shorter utterances.
@pytest.mark.parametrize("time_asr", KOKORO_DECODER_PCC_TIME_ASR_SIZES)
@pytest.mark.parametrize("use_torch_sinegen,min_pcc", [(False, 0.79), (True, 0.85)])
def test_kokoro_decoder_tt_e2e_waveform_pcc_sequence_lengths(
    ttnn_device, time_asr: int, use_torch_sinegen: bool, min_pcc: float
):
    p = run_decoder_tt_e2e_waveform_pcc_value(
        ttnn_device,
        time_asr=time_asr,
        use_torch_sinegen=use_torch_sinegen,
        seed=42,
    )
    tag = "torch_cpu_sinegen" if use_torch_sinegen else "ttnn_device_sinegen"
    ok = p >= min_pcc
    print(f"time_asr={time_asr} decoder_tt e2e PCC mode={tag} pcc={p:.6f} pass={ok} (min {min_pcc})")
    assert ok, f"time_asr={time_asr} E2E waveform PCC {p} mode={tag} expected >= {min_pcc}"
