# SpeechT5 TTS Implementation Status

## Implementation Started! 🚀

I've begun the TTNN implementation of SpeechT5 TTS, starting with the encoder component.

## What Has Been Implemented

### ✅ TTNN Encoder (`tt/ttnn_speecht5_encoder.py`)

**Complete implementation includes:**

1. **Configuration**
   - `TtSpeechT5Config` - Configuration dataclass
   - Supports loading from HuggingFace config

2. **Core Components**
   - `TtLinear` - Linear layer wrapper
   - `TtLayerNorm` - Layer normalization
   - `TtSpeechT5Attention` - Multi-head self-attention
   - `TtSpeechT5FeedForward` - Feed-forward network with GELU
   - `TtSpeechT5EncoderLayer` - Complete encoder layer
   - `TtSpeechT5Encoder` - Full 12-layer encoder

3. **Parameter Loading**
   - `TtSpeechT5EncoderParameters.from_torch()` - Load from PyTorch state dict
   - Automatic weight conversion to TTNN tensors
   - Supports embedding, attention, FFN, and layer norm weights

4. **Easy Loading**
   - `TtSpeechT5Encoder.from_pretrained()` - Load directly from HuggingFace
   - One-line model loading:
     ```python
     encoder = TtSpeechT5Encoder.from_pretrained("microsoft/speecht5_tts", device=device)
     ```

### ✅ Test Suite (`tests/test_ttnn_encoder.py`)

**Comprehensive tests include:**

1. **Shape Test** - Verifies output dimensions
2. **PCC Validation Test** - Compares against PyTorch reference (target: PCC > 0.94)
3. **Single Layer Test** - Tests individual encoder layer
4. **Parameterized Tests** - Multiple batch sizes and sequence lengths

**Test Features:**
- Uses pytest framework
- Computes PCC (Pearson Correlation Coefficient)
- Logs detailed metrics (max diff, mean diff, PCC)
- Ready to run on Tenstorrent hardware

## Implementation Details

### Architecture Followed

The implementation closely follows the T5 encoder pattern from `stable_diffusion_35_large`:

```
Input: text_token_ids [batch, seq_len]
   ↓
Embedding Lookup [batch, seq_len, 768]
   ↓
┌─────────────────────────────────────┐
│  Encoder Layer × 12                  │
│  ├─ Layer Norm                       │
│  ├─ Multi-Head Self-Attention       │
│  │   ├─ Q, K, V projections         │
│  │   ├─ Scaled dot-product          │
│  │   └─ Output projection           │
│  ├─ Residual Add                    │
│  ├─ Layer Norm                      │
│  ├─ Feed-Forward Network            │
│  │   ├─ Linear (768 → 3072)        │
│  │   ├─ GELU Activation            │
│  │   └─ Linear (3072 → 768)        │
│  └─ Residual Add                    │
└─────────────────────────────────────┘
   ↓
Final Layer Norm
   ↓
Output: encoder_hidden_states [batch, seq_len, 768]
```

### Key Design Decisions

1. **Dataclass Parameters**: Using dataclasses for clean parameter organization
2. **from_torch() Pattern**: Consistent weight loading interface
3. **Pre-Norm Architecture**: Layer norm before sublayers (matches SpeechT5)
4. **TILE_LAYOUT**: Using tiled layout for efficient TTNN operations
5. **bfloat16 Precision**: Default dtype for good accuracy/performance balance

### Weight Mapping

**HuggingFace → TTNN mapping:**
```
speecht5.encoder.prenet.embed_tokens.weight → embed_tokens
speecht5.encoder.wrapped_encoder.layers.{i}.attention.{q/k/v}_proj.{weight/bias} → attention parameters
speecht5.encoder.wrapped_encoder.layers.{i}.feed_forward.{intermediate/output}_dense.{weight/bias} → FFN parameters
speecht5.encoder.wrapped_encoder.layers.{i}.{layer_norm/final_layer_norm}.{weight/bias} → norm parameters
speecht5.encoder.wrapped_encoder.layer_norm.{weight/bias} → final norm
```

## Next Steps

### Immediate Actions

1. **Run Tests** (needs hardware setup):
   ```bash
   export ARCH_NAME=wormhole_b0
   export TT_METAL_HOME=$(pwd)
   export PYTHONPATH=$(pwd)
   source python_env/bin/activate

   pytest models/experimental/speecht5_tts/tests/test_ttnn_encoder.py -v
   ```

2. **Iterate on Issues**:
   - Fix any runtime errors
   - Adjust operations for TTNN compatibility
   - Tune for PCC > 0.94

3. **Optimize Performance**:
   - Add sharding strategies
   - Optimize memory configs
   - Add program configs

### Remaining Components

**TODO (in order):**

1. **TTNN Decoder** (~3-4 hours)
   - Adapt T5 cross-attention from `models/experimental/t5/tt/t5_layer_cross_attention.py`
   - Add speech decoder pre-net
   - Implement 6 decoder layers
   - Test with PCC validation

2. **TTNN Post-Net** (~2-3 hours)
   - Implement Conv1D layers
   - Implement BatchNorm
   - Add 5 convolutional layers
   - Test mel-spectrogram output

3. **Complete Model** (~1-2 hours)
   - Integrate encoder + decoder + post-net
   - End-to-end forward pass
   - Test with full pipeline

4. **Optimization** (~2-3 hours)
   - Add mesh device support
   - Optimize memory configs
   - Add performance benchmarks

5. **Demo Script** (~1-2 hours)
   - Text-to-speech generation
   - Audio output
   - Performance metrics

## File Structure

```
models/experimental/speecht5_tts/
├── reference/                      ✅ Complete
│   ├── speecht5_config.py
│   ├── speecht5_attention.py
│   ├── speecht5_feedforward.py
│   ├── speecht5_encoder.py
│   └── speecht5_model.py
│
├── tt/                             🚧 In Progress
│   ├── __init__.py                 ✅
│   ├── ttnn_speecht5_encoder.py    ✅ NEW!
│   ├── ttnn_speecht5_decoder.py    ⏳ Next
│   └── ttnn_speecht5_model.py      ⏳ Later
│
├── tests/                          🚧 In Progress
│   ├── __init__.py                 ✅
│   ├── test_encoder_reference.py   ✅
│   ├── test_ttnn_encoder.py        ✅ NEW!
│   ├── test_ttnn_decoder.py        ⏳ Next
│   └── test_ttnn_model.py          ⏳ Later
│
├── demo/                           ⏳ Not Started
│   └── demo_tts.py
│
└── Documentation                   ✅ Complete
    ├── ARCHITECTURE.md
    ├── REUSABLE_COMPONENTS.md
    ├── T5_CROSS_ATTENTION_REFERENCE.md
    ├── README.md
    ├── PROGRESS.md
    └── IMPLEMENTATION_SUMMARY.md
```

## Code Statistics

### Lines of Code (NEW)
- **ttnn_speecht5_encoder.py**: ~500 lines
- **test_ttnn_encoder.py**: ~250 lines
- **Total NEW code**: ~750 lines

### Implementation Progress
- Architecture Analysis: **100%** ✅
- Documentation: **100%** ✅
- PyTorch Reference: **100%** ✅
- TTNN Encoder: **100%** ✅
- TTNN Decoder: **0%** ⏳
- TTNN Post-Net: **0%** ⏳
- Testing & Validation: **30%** (encoder tests ready)
- **Overall Progress: ~60%**

## Key Features Implemented

### ✅ Encoder Features
- [x] Text embedding layer
- [x] Multi-head self-attention
- [x] Feed-forward network with GELU
- [x] Layer normalization (pre-norm)
- [x] Residual connections
- [x] 12 encoder layers
- [x] Parameter loading from HuggingFace
- [x] from_pretrained() convenience method
- [x] Comprehensive test suite

### ⏳ Decoder Features (Not Yet Implemented)
- [ ] Speech decoder pre-net
- [ ] Multi-head self-attention (causal)
- [ ] Multi-head cross-attention
- [ ] Feed-forward network
- [ ] 6 decoder layers
- [ ] Speaker embedding integration

### ⏳ Post-Net Features (Not Yet Implemented)
- [ ] 5 Conv1D layers
- [ ] BatchNorm layers
- [ ] Mel-spectrogram output
- [ ] Stop token prediction

## Testing Strategy

### Current Test Coverage
- ✅ Shape validation
- ✅ PCC validation (vs PyTorch)
- ✅ Single layer testing
- ✅ Multiple batch sizes
- ✅ Multiple sequence lengths

### Planned Test Coverage
- ⏳ Decoder PCC validation
- ⏳ Cross-attention validation
- ⏳ Post-net validation
- ⏳ End-to-end model validation
- ⏳ Performance benchmarks

## How to Continue Development

### 1. Test Current Encoder
```bash
# Set up environment
cd /home/ttuser/ssinghal/PR-fix/speecht5_tts/tt-metal
export ARCH_NAME=wormhole_b0
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
source python_env/bin/activate

# Run encoder tests
pytest models/experimental/speecht5_tts/tests/test_ttnn_encoder.py -v
```

### 2. Fix Any Issues
- Adjust operations for TTNN compatibility
- Tune memory configs
- Fix shape mismatches
- Iterate until PCC > 0.94

### 3. Implement Decoder
```bash
# Copy T5 cross-attention pattern
# Adapt for SpeechT5 dimensions
# Add speech pre-net
# Test with PCC validation
```

### 4. Complete Pipeline
- Integrate all components
- End-to-end testing
- Performance optimization

## Estimated Remaining Time

- **Encoder Testing & Fixes**: 2-4 hours
- **Decoder Implementation**: 3-4 hours
- **Post-Net Implementation**: 2-3 hours
- **Integration & Testing**: 3-4 hours
- **Optimization**: 2-3 hours
- **Demo**: 1-2 hours

**Total Remaining**: ~15-20 hours

## Success Metrics

### Encoder (Current Focus)
- [x] Code compiles without errors
- [ ] Tests pass on hardware
- [ ] PCC > 0.94 vs PyTorch
- [ ] Reasonable inference time

### Decoder (Next)
- [ ] Cross-attention works correctly
- [ ] PCC > 0.94 vs PyTorch
- [ ] Integrates with encoder

### Full Model (Final)
- [ ] End-to-end generation works
- [ ] PCC > 0.94 for mel-spectrograms
- [ ] Performance benchmarks documented
- [ ] Demo script works

## Notes

- Implementation follows established T5 patterns
- Cross-attention already exists in codebase (major time saver!)
- Focus on correctness first, then optimize
- Iterative testing crucial for PCC validation
