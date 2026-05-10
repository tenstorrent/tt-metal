# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Experimental bring-up for mistralai/Mistral-Small-4-119B-2603 (TTNN)."""

from models.experimental.mistral_small_4_119b.constants import (
    HF_MODEL_ID,
    EXPECTED_HIDDEN_SIZE,
    EXPECTED_NUM_LAYERS,
    EXPECTED_VOCAB_SIZE,
    EXPECTED_RMS_NORM_EPS,
    EXPECTED_NUM_EXPERTS,
    EXPECTED_NUM_EXPERTS_PER_TOK,
    EXPECTED_VISION_HIDDEN_SIZE,
)

__all__ = [
    "HF_MODEL_ID",
    "EXPECTED_HIDDEN_SIZE",
    "EXPECTED_NUM_LAYERS",
    "EXPECTED_VOCAB_SIZE",
    "EXPECTED_RMS_NORM_EPS",
    "EXPECTED_NUM_EXPERTS",
    "EXPECTED_NUM_EXPERTS_PER_TOK",
    "EXPECTED_VISION_HIDDEN_SIZE",
]
