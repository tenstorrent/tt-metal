# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Vision + multimodal projector for Mistral Small 4.

The HF module tree matches Mistral-Small-3.1-24B-Instruct-2503: ``PixtralVisionModel``,
``Mistral3MultiModalProjector``, same patch geometry and 1024-d vision width. Only the
projector MLP maps into ``hidden_size=4096`` (24B used 5120). Existing TTNN Pixtral
implementations load shapes from ``state_dict`` / ``ModelArgs``, so we reuse the
canonical ``tt_transformers`` stack unchanged.
"""

from models.tt_transformers.tt.multimodal.mistral_24b.vision_model import (
    TtMistralVisionTransformer as TtMistralSmall4VisionStack,
)

__all__ = ["TtMistralSmall4VisionStack"]
