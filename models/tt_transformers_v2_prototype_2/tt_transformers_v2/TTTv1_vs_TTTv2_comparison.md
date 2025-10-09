# TTTv1 vs TTTv2 Building Blocks Comparison

## Summary

This document compares the building blocks available in TTTv1 (`/models/tt_transformers`) with those implemented in TTTv2 to identify any missing components.

## TTTv1 Components

### Core Building Blocks
1. **attention.py** - TtAttention (multi-head attention with various features)
2. **mlp.py** - TtMLP (standard feed-forward network)
3. **mixtral_mlp.py** / **mixtral_moe.py** - MoE-specific implementations
4. **embedding.py** - TtEmbedding, TtLearnedPositionalEmbedding
5. **rope.py** - Various RoPE implementations (TtRoPE, TtCompactRoPE, TtMultiscaleRoPE, etc.)
6. **distributed_norm.py** - DistributedNorm (wrapper for distributed normalization)
7. **lm_head.py** - LMHead (language model output projection)

### Higher-Level Components (Not Building Blocks)
- **decoder.py** - Full decoder block implementations
- **model.py** - Complete model orchestration
- **generator.py** / **generator_vllm.py** - Generation logic
- **ccl.py** - Communication collectives
- **load_checkpoints.py** - Model loading utilities
- **model_config.py** - Configuration management

### Multimodal Components (Found in other models)
- **patch embeddings** - Found in various models (gemma3, qwen25_vl, swin, etc.)
- **vision components** - Found in model-specific implementations

## TTTv2 Components

### Current Building Blocks (12 files in 4 subdirectories)

#### attention/ (4 files)
- **mha.py** - MultiHeadAttentionSpec (includes GQA support)
- **gqa.py** - GroupedQueryAttentionSpec
- **flash.py** - FlashAttentionImplConfig
- **sliding.py** - SlidingWindowAttentionSpec

#### ffn/ (3 files)
- **mlp.py** - MLPSpec (standard MLP)
- **gated_mlp.py** - GatedMLPSpec, SwiGLU, GeGLU
- **moe.py** - MoESpec (Mixture of Experts)

#### embeddings/ (3 files)
- **token.py** - TokenEmbeddingSpec
- **position.py** - PositionEmbeddingSpec (learned, sinusoidal, ALiBi)
- **rotary.py** - RoPESpec (moved from attention/)

#### normalization/ (2 files)
- **rmsnorm.py** - RMSNormSpec
- **layernorm.py** - LayerNormSpec

## Missing from TTTv2

### Core Components That Should Be Added
1. **LM Head** - Language model output projection layer
   - **Recommendation**: Add as `building_blocks/embeddings/lm_head.py` or create a new `building_blocks/heads/` subdirectory
   - This is essential for language models

2. **Distributed Normalization** - Wrapper for distributed/sharded normalization
   - **Recommendation**: Add as `building_blocks/normalization/distributed.py`
   - Important for multi-device deployments

### Components That Belong in Models (Not Core)
1. **Patch Embeddings** - Vision-specific component
   - **Recommendation**: Keep in individual model implementations
   - Too vision-specific for core library

2. **Vision Components** - Vision transformers, vision MLPs, etc.
   - **Recommendation**: Keep in model implementations
   - Models can compose TTTv2 building blocks for vision tasks

3. **Multimodal Components** - Cross-modal attention, fusion layers
   - **Recommendation**: Keep in model implementations
   - Too model-specific for core library

### Components Already Covered
- ✅ Attention mechanisms (MHA, GQA, Flash, Sliding Window)
- ✅ FFN variants (MLP, Gated MLP, MoE)
- ✅ Embeddings (Token, Position)
- ✅ RoPE (now in embeddings/)
- ✅ Normalization (RMSNorm, LayerNorm)

## Recommendations

### Immediate Additions (High Priority)
1. **Add LM Head building block**
   ```python
   # building_blocks/embeddings/lm_head.py
   # or building_blocks/heads/lm.py
   @dataclass
   class LMHeadSpec:
       hidden_dim: int
       vocab_size: int
       tie_word_embeddings: bool = False
   ```

2. **Add Distributed Normalization**
   ```python
   # building_blocks/normalization/distributed.py
   @dataclass
   class DistributedNormSpec:
       base_norm: Union[RMSNormSpec, LayerNormSpec]
       distributed_strategy: str = "all_reduce"
       num_devices: int = 1
   ```

### Keep Outside Core (Low Priority)
- Patch embeddings (vision-specific)
- Vision transformer components
- Multimodal fusion layers
- Model-specific optimizations

### Design Principle Reminder
Per TTTv2 design: "Library, not Framework" - TTTv2 should provide general-purpose building blocks that can be composed by any transformer model. Model-specific components belong in the model implementations, not the core library.
