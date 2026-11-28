# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.pyannote.reference.wespeaker import SAMPLE_RATE, compute_fbank, load_wespeaker
from models.experimental.pyannote.tt.wespeaker import TtWeSpeakerResNet34

CHUNK_SAMPLES = 10 * SAMPLE_RATE  # pyannote/speaker-diarization-3.1 embeds 10 s chunks
MASK_FRAMES = 589  # segmentation-3.0 frames per 10 s chunk
SPEAKERS = 3


def _inputs(chunks, seed=0):
    """Speech-like waveforms and per-speaker activity masks, one inactive speaker per chunk."""
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(CHUNK_SAMPLES) / SAMPLE_RATE
    f0 = 100 + 150 * torch.rand(chunks, 1, generator=g)
    envelope = (torch.sin(2 * torch.pi * 3 * t + 6 * torch.rand(chunks, 1, generator=g)) > 0).float()
    voiced = sum(torch.sin(2 * torch.pi * k * f0 * t) / k for k in range(1, 8)) * envelope
    wav = 0.1 * voiced + 0.01 * torch.randn(chunks, CHUNK_SAMPLES, generator=g)
    masks = (torch.rand(chunks, SPEAKERS, MASK_FRAMES, generator=g) > 0.5).float()
    masks[:, -1] = 0.0
    return wav, masks


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("chunks", [1, 4, 21])  # 21: the 10 s windows of 30 s audio at a 1 s step
def test_wespeaker_embedding(device, chunks, reset_seeds):
    ref = load_wespeaker()
    wav, masks = _inputs(chunks)
    fbank = compute_fbank(wav)
    with torch.no_grad():
        expected = ref(fbank, masks)
    model = TtWeSpeakerResNet34(device, ref)

    tt_out = ttnn.to_torch(model(fbank, masks)).float()[:chunks, :SPEAKERS]
    active = expected[:, :-1]
    passing, msg = comp_pcc(active, tt_out[:, :-1], 0.99)
    cos = _cosine(active, tt_out[:, :-1])
    logger.info(f"chunks={chunks}: {msg}, cosine min {cos.min():.5f}")

    ttnn.synchronize_device(device)
    start = time.perf_counter()
    ttnn.to_torch(model(compute_fbank(wav), masks))
    logger.info(f"chunks={chunks}: {1e3 * (time.perf_counter() - start):.1f} ms warm, fbank included")
    assert passing, msg
    assert cos.min() > 0.99, f"cosine {cos.min():.5f}"


def _cosine(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten(0, -2), b.flatten(0, -2), dim=-1)
