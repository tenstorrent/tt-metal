# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any, Dict


def build_projector(args, dtype, state_dict: Dict[str, Any], weight_cache_path: Path):
    """
    Factory function to create vision-to-LLM projectors

    Args:
        args: VisionModelArgs with projector configuration
        dtype: ttnn data type (ttnn.bfloat16)
        state_dict: Model weights dictionary
        weight_cache_path: Path to cached weights

    Returns:
        Projector module (LightweightModule)

    Supported projectors:
        - "mlp": 2-layer MLP with GELU (LLaVA 1.5, LLaVA-MORE)
        - Future: "moe": Mixture of Experts projector (MoE-LLaVA)

    Reference:
        - LLaVA/llava/model/multimodal_projector/builder.py
        - MoE-LLaVA/moellava/model/multimodal_projector/
    """
    projector_type = args.projector_type.lower()

    if projector_type == "mlp":
        from .mlp_projector import MLPProjector

        return MLPProjector(args, dtype, state_dict, weight_cache_path)

    # Future: MoE projector for MoE-LLaVA
    # elif projector_type == "moe":
    #     from .moe_projector import MoEProjector
    #     return MoEProjector(args, dtype, state_dict, weight_cache_path)

    else:
        raise ValueError(f"Unknown projector type: {projector_type}. " f"Supported: mlp. " f"Future: moe")
