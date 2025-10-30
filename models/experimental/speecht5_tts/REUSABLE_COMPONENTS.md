# Reusable Components from Existing T5 Implementation

## Source Files
- `models/experimental/stable_diffusion_35_large/reference/t5_encoder.py`
- `models/experimental/stable_diffusion_35_large/tt/t5_encoder.py`

## Components That Can Be Reused/Adapted

### 1. Attention Mechanism
**From**: `T5Attention` class
**Reusable for**: SpeechT5 encoder self-attention, decoder self-attention, and cross-attention

**Similarities**:
- Q/K/V projections using `torch.nn.Linear(hidden_size, num_heads * head_dim)`
- Output projection
- Multi-head attention computation
- Softmax over attention scores

**Adaptations needed**:
- Position bias handling (T5 uses relative, SpeechT5 uses scaled positional)
- Cross-attention variant (K/V from encoder, Q from decoder)
- Causal masking for decoder self-attention

### 2. Feed-Forward Network
**From**: `T5DenseGatedActDense` / `T5LayerFF`
**Reusable for**: SpeechT5 encoder and decoder FFN blocks

**Similarities**:
- Two linear layers with intermediate expansion
- GELU activation (T5 uses NewGELU, SpeechT5 uses standard GELU)
- Residual connection + layer norm

**Can reuse directly**: Feed-forward structure is nearly identical

### 3. Layer Normalization
**From**: `T5LayerNorm`
**Reusable for**: All layer norms in SpeechT5

**Similarities**:
- RMS-style normalization
- Learnable weight parameter
- Applied before sublayers (pre-norm)

**Note**: Both use `eps=1e-5` by default

### 4. Encoder Block Structure
**From**: `T5Block` / `T5EncoderLayer`
**Reusable for**: SpeechT5 encoder layers

**Similarities**:
- Self-attention → Add & Norm → FFN → Add & Norm
- Same residual connection pattern
- Dropout after each sublayer

### 5. Parameter Loading Pattern (ttnn)
**From**: `TtT5EncoderParameters.from_torch()`
**Reusable for**: SpeechT5 parameter conversion

**Pattern to follow**:
- Use dataclasses for parameter grouping
- Implement `from_torch()` classmethod
- Convert weights using `ttnn.from_torch()` with appropriate dtypes
- Handle weight transposition for linear layers
- Support mesh device sharding

### 6. Linear Layer Wrapper (ttnn)
**From**: `TtLinear` / `TtLinearParameters`
**Reusable for**: All linear projections in SpeechT5

**Pattern**:
```python
@dataclass
class TtLinearParameters:
    weight: ttnn.Tensor
    bias: ttnn.Tensor | None
```

### 7. Testing Pattern
**From**: `test_t5_encoder.py`
**Reusable for**: SpeechT5 component tests

**Pattern**:
- Load HuggingFace model
- Create PyTorch reference
- Convert to ttnn
- Compare outputs with `assert_quality(output, tt_output, pcc=0.945)`
- Use mesh device fixtures

## Components Requiring New Implementation

### 1. Speech Decoder Pre-Net
- Linear layers for mel-spectrogram preprocessing
- Speaker embedding integration
- Different from T5's text embedding

### 2. Cross-Attention Layer ✓ **EXISTS IN CODEBASE!**
**Source**: `models/experimental/t5/tt/t5_layer_cross_attention.py`

The T5 decoder already has cross-attention implementation in ttnn! Key features:
- `TtT5LayerCrossAttention` class
- Handles `key_value_states` from encoder
- Query from decoder hidden states
- Supports KV caching
- Layer norm + residual connection

**Also available**:
- `models/tt_transformers/tt/multimodal/llama_cross_attention.py` - Multimodal cross-attention
- `models/experimental/trocr/tt/trocr_decoder_layer.py` - TrOCR encoder-decoder attention

**Adaptation needed**: Minimal! The T5 cross-attention can be directly adapted for SpeechT5 decoder

### 3. Convolutional Post-Net
- Conv1D layers
- BatchNorm layers
- Tanh activation
- No equivalent in T5

### 4. Scaled Positional Encoding
- Different from T5's relative position bias
- Uses learnable scaling parameter `alpha`
- Applied as additive encoding

### 5. Autoregressive Decoder Logic
- Causal masking
- KV cache management (for inference)
- Stop token prediction
- Not needed in T5 encoder

### 6. Reduction Factor Handling
- Output multiple mel frames per decoder step
- Reshape and unfold operations
- T5 doesn't have this concept

## Adaptation Strategy

1. **Phase 1**: Reuse T5 attention and FFN for encoder
2. **Phase 2**: Extend attention mechanism for cross-attention
3. **Phase 3**: Implement speech-specific pre/post-nets
4. **Phase 4**: Adapt ttnn conversion patterns from T5

## Code Reuse Estimate (UPDATED!)
- **Encoder attention & FFN**: ~70% reusable from T5 encoder
- **Decoder self-attention**: ~70% reusable from T5 decoder
- **Decoder cross-attention**: ~90% reusable - **ALREADY IMPLEMENTED!** ✓
- **Layer norm & residuals**: ~90% reusable
- **Parameter conversion patterns**: ~80% reusable
- **Overall codebase**: ~60-70% can be adapted from existing T5 implementation

**Major Discovery**: T5 decoder cross-attention already exists in ttnn, significantly reducing implementation effort!
