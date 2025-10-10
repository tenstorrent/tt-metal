# TTTv2 Source - Public API
# This file controls what is exposed to users of the library
# Note: Advanced users may need direct access to TTNN interfaces

# Building blocks - commented out to allow testing imports without building_blocks errors
# from . import building_blocks
# from .building_blocks import (
#     # Attention
#     AttentionSpec,
#     AttentionImplConfig,
#     attention_prefill_forward,
#     attention_decode_forward,
#     get_default_attention_impl_config,
#     # FFN
#     FFNSpec,
#     FFNImplConfig,
#     MoESpec,
#     ffn_prefill_forward,
#     ffn_decode_forward,
#     moe_prefill_forward,
#     moe_decode_forward,
#     get_default_ffn_impl_config,
#     # Normalization
#     RMSNormSpec,
#     RMSNormImplConfig,
#     LayerNormSpec,
#     LayerNormImplConfig,
#     rmsnorm_forward,
#     layernorm_forward,
#     get_default_norm_impl_config,
#     # Embeddings
#     EmbeddingSpec,
#     EmbeddingImplConfig,
#     PositionalEmbeddingSpec,
#     embedding_forward,
#     positional_embedding_forward,
#     get_default_embedding_impl_config,
#     # CCL
#     CCLManager,
#     AllReduceSpec,
#     AllReduceImplConfig,
#     all_reduce_forward,
#     get_all_reduce_default_impl_config,
#     AllGatherSpec,
#     AllGatherImplConfig,
#     all_gather_forward,
#     get_all_gather_default_impl_config,
#     DistributedRMSNormSpec,
#     DistributedRMSNormImplConfig,
#     distributed_rmsnorm_forward,
#     get_distributed_rmsnorm_default_impl_config,
#     # LM Head
#     LMHeadSpec,
#     LMHeadImplConfig,
#     prepare_lm_head_weights,
#     lm_head_forward,
#     lm_head_prefill_forward,
#     lm_head_decode_forward,
#     get_default_lm_head_impl_config,
#     # Note: Distributed normalization is available via building_blocks.ccl.distributed_norm
# )

# Patterns are optional - uncomment when implemented
# from .patterns import (
#     DecoderLayerSpec,
#     DecoderLayerImplConfig,
#     CausalLM,
# )

# Testing utilities
from . import testing

__all__ = [
    # Module namespaces (lazy-loaded via __getattr__)
    "building_blocks",
    "testing",
]
