# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Profiling harness: ONE speaker-encoder forward (log-mel -> 512-d ``g``) for
device-time measurement. Not a correctness test (see ``test_speaker_encoder.py``).

``mel_len`` defaults to 801 frames — what the demo actually feeds it (8 s of 16 kHz
reference audio through the mel frontend: 1 + 128000/160). Run under tracy:

    python -m tracy -v -r -p -o spk_enc \\
        -m "pytest models/experimental/xtts/tests/test_speaker_encoder_profile.py"

then summarise the emitted ops_perf_results CSV (group by OP CODE, sum
DEVICE FW DURATION [ns]). NOTE: the CSV holds **two** passes — the warmup below and the
measured one — so halve the totals (and the per-op-code counts) for a per-pass figure.
The warmup is what keeps conv2d's one-time weight preprocessing out of the numbers; at
~330 device ops per pass, both fit under the on-device profiler buffer.
"""
import pytest
import torch

import ttnn
from models.experimental.xtts.reference.xtts_gpt_block import load_xtts_state_dict
from models.experimental.xtts.reference.xtts_speaker_encoder import build_reference_speaker_encoder
from models.experimental.xtts.tt.xtts_speaker_encoder import TtResNetSpeakerEncoder


@pytest.fixture(scope="module")
def xtts_state_dict():
    return load_xtts_state_dict()


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("mel_len", [801])
def test_speaker_encoder_profile(device, xtts_state_dict, mel_len):
    reference = build_reference_speaker_encoder(xtts_state_dict)
    tt_enc = TtResNetSpeakerEncoder(device, reference)

    mel = torch.randn(1, 64, mel_len).abs() + 0.1
    mel_dev = ttnn.from_torch(mel.float(), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)

    # Warmup: compiles kernels and runs conv2d's one-time weight preprocessing, so neither
    # lands in the measured pass. Both passes still appear in the CSV (see the module docstring).
    tt_enc(mel_dev)
    ttnn.synchronize_device(device)

    g = tt_enc(mel_dev)
    ttnn.synchronize_device(device)
    print(f"SPKENCINFO mel_len={mel_len} out={tuple(g.shape)}")
