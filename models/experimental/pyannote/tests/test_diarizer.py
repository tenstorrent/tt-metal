# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from loguru import logger

from models.experimental.pyannote.reference.diarization import diarize, embedding_masks
from models.experimental.pyannote.reference.segmentation import SAMPLE_RATE, load_segmentation, to_multilabel, windows
from models.experimental.pyannote.reference.wespeaker import compute_fbank, load_wespeaker
from models.experimental.pyannote.tests.test_wespeaker import _inputs
from models.experimental.pyannote.tt.diarizer import TtDiarizer


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("starts_before", [None, 20 * SAMPLE_RATE])
def test_diarizer_matches_reference(device, starts_before, reset_seeds):
    """A 30 s waveform: device per-window data against the host reference, then speaker turns."""
    wav = _inputs(3, seed=1)[0].reshape(-1)
    out = TtDiarizer(device)(wav, starts_before)

    chunks, starts = windows(wav, starts_before)
    with torch.no_grad():
        seg = to_multilabel(load_segmentation()(chunks[:, None])).numpy()
        emb = load_wespeaker()(compute_fbank(chunks), torch.from_numpy(embedding_masks(seg))).numpy()

    assert np.array_equal(out["starts"], np.asarray(starts, np.float32) / SAMPLE_RATE)
    assert np.array_equal(out["segmentation"], seg.astype(np.uint8))
    active = seg.sum(axis=1) > 0
    a, b = emb[active], out["embeddings"][active]
    cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    logger.info(f"windows={len(starts)} active pairs={active.sum()} cosine min {cos.min():.5f}")
    assert cos.min() > 0.99, f"cosine {cos.min():.5f}"
    assert diarize(out["segmentation"], out["starts"], out["embeddings"]) == diarize(seg, out["starts"], emb)
