"""
Building blocks for transformer models.

This module provides the core functional components for building transformers:
- Attention mechanisms (MHA, GQA, Flash, Sliding Window)
- Feed-forward networks (MLP, SwiGLU, MoE)
- Normalization layers (RMSNorm, LayerNorm)
- Embeddings (Token, Position)
- Collective Communication Layer (CCL) for distributed operations
- Language Model Head (LM Head) for output projection
- Distributed Normalization wrapper for multi-device normalization
"""

# Import submodules
from . import attention
from . import ffn
from . import embeddings
from . import normalization
from . import ccl
from . import lm_head

__all__ = ["attention", "ffn", "embeddings", "normalization", "ccl", "lm_head"]

# For backward compatibility with the old flat structure,
# re-export commonly used items at the building_blocks level
from .attention import (
    MultiHeadAttentionSpec as AttentionSpec,
    MultiHeadAttentionImplConfig as AttentionImplConfig,
    mha_prefill_forward as attention_prefill_forward,
    mha_decode_forward as attention_decode_forward,
    get_default_mha_impl_config as get_default_attention_impl_config,
)

from .ffn import (
    MLPSpec as FFNSpec,
    MLPImplConfig as FFNImplConfig,
    MoESpec,
    mlp_prefill_forward as ffn_prefill_forward,
    mlp_decode_forward as ffn_decode_forward,
    moe_prefill_forward,
    moe_decode_forward,
    get_default_mlp_impl_config as get_default_ffn_impl_config,
)

from .normalization import (
    RMSNormSpec,
    RMSNormImplConfig,
    LayerNormSpec,
    LayerNormImplConfig,
    rmsnorm_forward,
    layernorm_forward,
    get_default_rmsnorm_impl_config as get_default_norm_impl_config,
)

from .embeddings import (
    TokenEmbeddingSpec as EmbeddingSpec,
    TokenEmbeddingImplConfig as EmbeddingImplConfig,
    PositionEmbeddingSpec as PositionalEmbeddingSpec,
    token_embedding_forward as embedding_forward,
    position_embedding_forward as positional_embedding_forward,
    get_default_token_impl_config as get_default_embedding_impl_config,
)

from .ccl import (
    # Manager
    CCLManager,
    # All-reduce
    AllReduceSpec,
    AllReduceImplConfig,
    all_reduce_forward,
    get_all_reduce_default_impl_config,
    # All-gather
    AllGatherSpec,
    AllGatherImplConfig,
    all_gather_forward,
    get_all_gather_default_impl_config,
    # Distributed RMS norm
    DistributedRMSNormSpec,
    DistributedRMSNormImplConfig,
    distributed_rmsnorm_forward,
    get_distributed_rmsnorm_default_impl_config,
)

from .lm_head import (
    LMHeadSpec,
    LMHeadImplConfig,
    prepare_weights as prepare_lm_head_weights,
    lm_head_forward,
    prefill_forward as lm_head_prefill_forward,
    decode_forward as lm_head_decode_forward,
    get_default_impl_config as get_default_lm_head_impl_config,
)

# Note: Distributed normalization is available via ccl.distributed_norm
