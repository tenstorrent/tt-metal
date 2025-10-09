"""
Normalization building blocks for transformers.

Provides various normalization techniques including RMSNorm and LayerNorm.
"""

__all__ = [
    # RMSNorm exports
    "RMSNormSpec",
    "RMSNormImplConfig",
    "rmsnorm_forward",
    "get_default_rmsnorm_impl_config",
    # LayerNorm exports
    "LayerNormSpec",
    "LayerNormImplConfig",
    "layernorm_forward",
    "get_default_layernorm_impl_config",
]

# RMSNorm imports
from .rmsnorm import (
    RMSNormSpec,
    RMSNormImplConfig,
    forward as rmsnorm_forward,
    get_default_impl_config as get_default_rmsnorm_impl_config,
)

# LayerNorm imports
from .layernorm import (
    LayerNormSpec,
    LayerNormImplConfig,
    forward as layernorm_forward,
    get_default_impl_config as get_default_layernorm_impl_config,
)
