# T5 Cross-Attention Reference

## Key Finding

**T5 decoder cross-attention is already implemented in ttnn!**

This significantly simplifies the SpeechT5 decoder implementation since we can directly adapt the existing T5 cross-attention code.

## File Locations

### Primary Reference (T5 Decoder with Cross-Attention)
- **Cross-Attention Layer**: `models/experimental/t5/tt/t5_layer_cross_attention.py`
- **T5 Block (Decoder)**: `models/experimental/t5/tt/t5_block.py`
- **T5 Attention Base**: `models/experimental/t5/tt/t5_attention.py`
- **T5 Model**: `models/experimental/t5/tt/t5_model.py`

### Additional References
- **Llama Cross-Attention**: `models/tt_transformers/tt/multimodal/llama_cross_attention.py`
- **TrOCR Decoder**: `models/experimental/trocr/tt/trocr_decoder_layer.py`

## T5 Cross-Attention Implementation Summary

From `models/experimental/t5/tt/t5_layer_cross_attention.py`:

### Class: `TtT5LayerCrossAttention`

**Key Features**:
1. **Encoder-Decoder Attention**:
   - Query (Q) from decoder hidden states
   - Key (K) and Value (V) from encoder output

2. **Components**:
   - `EncDecAttention`: TtT5Attention module with `has_relative_attention_bias=False`
   - `layer_norm`: Pre-normalization before attention
   - Residual connection: `hidden_states + attention_output`

3. **Forward Signature**:
```python
def forward(
    self,
    hidden_states,           # Decoder hidden states (Query)
    key_value_states,        # Encoder output (Key & Value)
    attention_mask=None,
    position_bias=None,
    layer_head_mask=None,
    past_key_value=None,     # KV cache for cross-attention
    use_cache=False,
    query_length=None,
    output_attentions=False,
)
```

4. **Processing Flow**:
   - Layer norm on decoder hidden states
   - Cross-attention with encoder outputs
   - Residual add
   - Return outputs with optional attention weights

## Adaptation for SpeechT5

### What Can Be Reused Directly
1. ✅ Cross-attention mechanism structure
2. ✅ Q/K/V projection pattern
3. ✅ Attention mask handling
4. ✅ KV cache management
5. ✅ Layer norm + residual connection pattern

### What Needs Adaptation
1. **Configuration**: Adapt T5Config → SpeechT5Config
2. **Hidden Dimensions**: Match SpeechT5 dimensions (768, not T5's dimensions)
3. **Integration**: Combine with speech decoder pre-net
4. **Weight Loading**: Load from SpeechT5 checkpoint instead of T5

## Implementation Strategy

### Step 1: Understand T5 Decoder Structure
Read the following files in order:
1. `models/experimental/t5/tt/t5_attention.py` - Base attention mechanism
2. `models/experimental/t5/tt/t5_layer_cross_attention.py` - Cross-attention layer
3. `models/experimental/t5/tt/t5_block.py` - Complete decoder block

### Step 2: Copy and Adapt for SpeechT5
Create new files in `models/experimental/speecht5_tts/tt/`:
1. `ttnn_speecht5_attention.py` - Adapt from T5 attention
2. `ttnn_speecht5_decoder_layer.py` - Adapt from T5 cross-attention layer
3. `ttnn_speecht5_decoder.py` - Complete decoder with pre-net

### Step 3: Modify for SpeechT5 Specifics
Key changes needed:
- Update dimensions (768 hidden size)
- Add speech decoder pre-net integration
- Handle speaker embeddings
- Support reduction factor (2 mel frames per step)

## Code Comparison

### T5 Decoder Block Structure
```
Input: decoder_hidden_states
  ↓
Self-Attention
  ↓
Add & Norm
  ↓
Cross-Attention (key_value_states from encoder)
  ↓
Add & Norm
  ↓
Feed-Forward
  ↓
Add & Norm
  ↓
Output: decoder_output
```

### SpeechT5 Decoder Block (Same Structure!)
```
Input: decoder_hidden_states (from speech pre-net)
  ↓
Self-Attention (causal)
  ↓
Add & Norm
  ↓
Cross-Attention (key_value_states from text encoder)
  ↓
Add & Norm
  ↓
Feed-Forward
  ↓
Add & Norm
  ↓
Output: decoder_output → post-net → mel-spectrogram
```

**They're nearly identical!** The main difference is the input source (speech pre-net vs text embedding).

## Parameter Mapping

### T5 State Dict → SpeechT5 State Dict

T5 Decoder Cross-Attention Keys:
```
decoder.block.{i}.layer.1.EncDecAttention.q.weight
decoder.block.{i}.layer.1.EncDecAttention.k.weight
decoder.block.{i}.layer.1.EncDecAttention.v.weight
decoder.block.{i}.layer.1.EncDecAttention.o.weight
decoder.block.{i}.layer.1.layer_norm.weight
```

SpeechT5 Decoder Cross-Attention Keys:
```
speecht5.decoder.wrapped_decoder.layers.{i}.encoder_attn.q_proj.weight
speecht5.decoder.wrapped_decoder.layers.{i}.encoder_attn.k_proj.weight
speecht5.decoder.wrapped_decoder.layers.{i}.encoder_attn.v_proj.weight
speecht5.decoder.wrapped_decoder.layers.{i}.encoder_attn.out_proj.weight
speecht5.decoder.wrapped_decoder.layers.{i}.encoder_attn_layer_norm.weight
```

**Mapping**: Just need to adjust the key names!

## Next Steps

1. ✅ **Copy T5 cross-attention implementation**
2. ✅ **Rename classes** for SpeechT5
3. ✅ **Update configuration** to use SpeechT5Config
4. ✅ **Adjust parameter loading** for SpeechT5 state dict keys
5. ✅ **Test with SpeechT5 weights**
6. ✅ **Integrate with speech pre-net and post-net**

## Estimated Implementation Time

With existing T5 cross-attention:
- **Original estimate**: 6-8 hours for decoder
- **New estimate**: 3-4 hours for decoder ⚡
- **Time saved**: ~50% reduction!

The cross-attention is the most complex part of the decoder, and it's already done!

## Files to Create

1. `tt/ttnn_speecht5_decoder_layer.py` - Adapt from `t5_layer_cross_attention.py`
2. `tt/ttnn_speecht5_decoder.py` - Complete decoder with all 6 layers
3. `tests/test_ttnn_decoder.py` - Test cross-attention with PCC validation

## Success Criteria

- Load SpeechT5 cross-attention weights into adapted T5 structure
- Forward pass matches PyTorch reference
- PCC > 0.94 for cross-attention layer
- PCC > 0.94 for complete decoder
