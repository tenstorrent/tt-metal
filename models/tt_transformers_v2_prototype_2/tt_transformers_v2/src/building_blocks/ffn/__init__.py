"""
Feed-Forward Network (FFN) building blocks for transformers.

This module provides various FFN implementations including:
- Standard MLP
- Gated MLPs (SwiGLU, GeGLU)
- Mixture of Experts (MoE)
"""

__all__ = [
    # Standard MLP
    "MLP",
    "MLPSpec",
    "MLPImplConfig",
    "mlp_prefill_forward",
    "mlp_decode_forward",
    "get_default_mlp_impl_config",
    # Gated MLPs
    "GatedMLP",
    "GatedMLPSpec",
    "GatedMLPImplConfig",
    "SwiGLU",
    "GeGLU",
    "gated_mlp_prefill_forward",
    "gated_mlp_decode_forward",
    "get_default_gated_mlp_impl_config",
    # Mixture of Experts
    "MoE",
    "MoESpec",
    "MoEImplConfig",
    "MoEOutput",
    "moe_prefill_forward",
    "moe_decode_forward",
    "get_default_moe_impl_config",
    "compute_load_balancing_loss",
]

# Standard MLP imports
from .mlp import (
    MLPSpec,
    MLPImplConfig,
    prefill_forward as mlp_prefill_forward,
    decode_forward as mlp_decode_forward,
    get_default_impl_config as get_default_mlp_impl_config,
)

# Gated MLP imports
from .gated_mlp import (
    GatedMLPSpec,
    GatedMLPImplConfig,
    SwiGLU,
    GeGLU,
    prefill_forward as gated_mlp_prefill_forward,
    decode_forward as gated_mlp_decode_forward,
    get_default_impl_config as get_default_gated_mlp_impl_config,
)

# MoE imports
from .moe import (
    MoESpec,
    MoEImplConfig,
    MoEOutput,
    prefill_forward as moe_prefill_forward,
    decode_forward as moe_decode_forward,
    get_default_impl_config as get_default_moe_impl_config,
    compute_load_balancing_loss,
)

# Convenience class names
MLP = MLPSpec
GatedMLP = GatedMLPSpec
MoE = MoESpec
