# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""The utterance the e2e reference is generated from, and where each size keeps it."""

import os

from models.demos.audio.qwen3_tts import weights

TEXT = "The kettle is on, and the rain has not let up since yesterday morning."
SPEAKER = "ryan"
LANGUAGE = "English"
SEED = 0
MAX_FRAMES = 200


def path():
    """`reference_outputs/<size>.refpt`, for the ambient checkpoint's size."""
    return os.path.join(os.path.dirname(__file__), "reference_outputs", f"{weights.model_size()}.refpt")
