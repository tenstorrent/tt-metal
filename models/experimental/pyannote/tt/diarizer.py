# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Per-window speaker data for pyannote/speaker-diarization-3.1: segmentation-3.0 on host,
WeSpeaker embeddings on device. Clustering across windows is
models.experimental.pyannote.reference.diarization.diarize, on host."""

import numpy as np
import torch

from models.experimental.pyannote.reference.diarization import embedding_masks
from models.experimental.pyannote.reference.segmentation import (
    SAMPLE_RATE,
    load_segmentation,
    to_multilabel,
    windows,
)
from models.experimental.pyannote.reference.wespeaker import compute_fbank, load_wespeaker
from models.experimental.pyannote.tt.wespeaker import TtWeSpeakerResNet34

import ttnn


class TtDiarizer:
    def __init__(self, device):
        self.segmentation = load_segmentation()
        self.embedding = TtWeSpeakerResNet34(device, load_wespeaker())

    def __call__(self, waveform: torch.Tensor, starts_before: int | None = None) -> dict[str, np.ndarray]:
        """waveform [samples] at 16 kHz -> per 10 s window: `segmentation` [windows, frames, 3] binary,
        `embeddings` [windows, 3, 256] and `starts` [windows] in seconds from the waveform start.
        `starts_before`: see reference.segmentation.windows."""
        chunks, starts = windows(waveform.float(), starts_before)
        with torch.no_grad():
            segmentation = to_multilabel(self.segmentation(chunks[:, None])).numpy()
        masks = torch.from_numpy(embedding_masks(segmentation))
        embeddings = ttnn.to_torch(self.embedding(compute_fbank(chunks), masks)).float()
        return {
            "segmentation": segmentation.astype(np.uint8),
            "embeddings": embeddings[: len(starts), : segmentation.shape[-1]].numpy(),
            "starts": np.asarray(starts, dtype=np.float32) / SAMPLE_RATE,
        }
