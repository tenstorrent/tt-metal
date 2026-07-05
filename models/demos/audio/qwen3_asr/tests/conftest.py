# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the Qwen3-ASR PCC / regression tests.

The tests compare the ttnn port against CPU-reference "golden" tensors captured by
``reference/dump_reference.py`` (large, kept outside the repo) and against the
extracted Qwen3-1.7B text-decoder checkpoint. Both are provided via env vars and the
fixtures below ``pytest.skip`` cleanly when they are absent, so the suite is safe to
collect on a machine without the data (e.g. a generic CI runner) and runs for real on a
P150 box that has them staged.

Env vars (all optional; defaults match the server container layout):
  QWEN3ASR_GOLDEN_DIR   dir with input_features.npy / conv_out.npy / audio_tower.npy /
                        inputs_embeds.npy / lm_head.npy   (default: $GOLDEN_DIR or /golden)
  QWEN3ASR_SNAP         HF snapshot root for Qwen/Qwen3-ASR-1.7B (audio-tower weights)
  QWEN3ASR_TEXT_DECODER extracted text-decoder checkpoint dir (ModelArgs / HF_MODEL)
"""
import os

import numpy as np
import pytest
import torch

GOLDEN_DIR = os.environ.get("QWEN3ASR_GOLDEN_DIR", os.environ.get("GOLDEN_DIR", "/golden"))
SNAP_ROOT = os.environ.get(
    "QWEN3ASR_SNAP", "/root/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots"
)
TEXT_DECODER = os.environ.get(
    "QWEN3ASR_TEXT_DECODER", os.environ.get("HF_MODEL", "/models/qwen3_asr_text_decoder")
)


def _load_golden(name):
    path = os.path.join(GOLDEN_DIR, name)
    if not os.path.isfile(path):
        pytest.skip(f"golden tensor not found: {path} (set QWEN3ASR_GOLDEN_DIR)")
    return torch.from_numpy(np.load(path)).float()


@pytest.fixture(scope="session")
def golden():
    """Callable ``golden(name)`` -> torch.float32 tensor from the golden dir (skips if absent)."""
    if not os.path.isdir(GOLDEN_DIR):
        pytest.skip(f"golden dir not found: {GOLDEN_DIR} (set QWEN3ASR_GOLDEN_DIR)")
    return _load_golden


@pytest.fixture(scope="session")
def snap_dir():
    """Resolved HF snapshot dir for the audio-tower weights (skips if absent)."""
    if not os.path.isdir(SNAP_ROOT):
        pytest.skip(f"HF snapshot not found: {SNAP_ROOT} (set QWEN3ASR_SNAP)")
    snaps = [d for d in os.listdir(SNAP_ROOT) if os.path.isdir(os.path.join(SNAP_ROOT, d))]
    if not snaps:
        pytest.skip(f"no snapshot under {SNAP_ROOT}")
    return os.path.join(SNAP_ROOT, snaps[0])


@pytest.fixture(scope="session")
def audio_tower_weights(snap_dir):
    """Session-cached CPU audio-tower reference weights."""
    from models.demos.audio.qwen3_asr.reference import audio_encoder_ref as ref

    return ref.load_audio_tower_weights(snap_dir=snap_dir, dtype=torch.float32)


@pytest.fixture(scope="session")
def text_decoder_ckpt():
    """Path to the extracted Qwen3-1.7B text-decoder checkpoint (skips if absent).

    ``ModelArgs`` reads the checkpoint from the ``HF_MODEL`` env var, so we also export it
    here for tests that build the decoder.
    """
    if not os.path.isdir(TEXT_DECODER):
        pytest.skip(f"text-decoder checkpoint not found: {TEXT_DECODER} (set QWEN3ASR_TEXT_DECODER)")
    os.environ.setdefault("HF_MODEL", TEXT_DECODER)
    return TEXT_DECODER
