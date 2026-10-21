# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0


def build_token_handler(handler_type: str = "standard"):
    """
    Factory function to create token handlers for multimodal input

    Args:
        handler_type: Type of token handler
            - "standard": Standard token weaving (LLaVA 1.5)
            - Future: "anyres": S2 anyres handler (LLaVA 1.6)
            - Future: "spatial": Spatial reasoning (OneVision)
            - Future: "codec": Codec-aligned tokens (OneVision-2)

    Returns:
        Token handler instance

    Reference:
        - LLaVA/llava/model/llava_arch.py:prepare_inputs_labels_for_multimodal
        - qwen25_vl/tt/common.py:merge_vision_tokens
    """
    handler_type = handler_type.lower()

    if handler_type == "standard":
        from .standard_interleaver import StandardInterleaver

        return StandardInterleaver()

    # Future handlers
    # elif handler_type == "anyres":
    #     from .anyres_handler import AnyresHandler
    #     return AnyresHandler()
    #
    # elif handler_type == "spatial":
    #     from .spatial_handler import SpatialHandler
    #     return SpatialHandler()
    #
    # elif handler_type == "codec":
    #     from .codec_handler import CodecHandler
    #     return CodecHandler()

    else:
        raise ValueError(
            f"Unknown token handler type: {handler_type}. " f"Supported: standard. " f"Future: anyres, spatial, codec"
        )
