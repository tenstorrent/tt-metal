# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from models.experimental.mistral_small_4_119b.tt.vision import TtMistralSmall4VisionStack
from models.tt_transformers.tt.multimodal.mistral_24b.vision_model import (
    TtMistralVisionTransformer as CanonicalVision,
)


def test_vision_stack_is_canonical_tt_transformers_class():
    assert TtMistralSmall4VisionStack is CanonicalVision
