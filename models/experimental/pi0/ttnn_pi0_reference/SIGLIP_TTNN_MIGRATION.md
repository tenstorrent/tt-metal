# SigLIP TTNN Migration Summary

## Overview
Replaced PyTorch fallback in SigLIP vision tower with full TTNN implementation based on the Gemma3 optimized kernels.

## Previous Implementation Issues

The original `SigLIPVisionTowerTTNN` had significant PyTorch fallbacks:
- ❌ **Patch Embedding**: Conv2d on CPU
- ❌ **All 27 Transformer Blocks**: Pure PyTorch
- ❌ **Final LayerNorm**: PyTorch
- ✅ Only position embedding addition was on device

This meant ~90% of computation was running in PyTorch on CPU/host, defeating the purpose of TTNN acceleration.

## New Implementation

### Architecture Overview
```
Input (PyTorch) 
    ↓
Patch Embedding (Conv2d on host, then transfer to TTNN) 
    ↓
Position Embedding Addition (TTNN) 
    ↓
27x Transformer Blocks (Full TTNN):
    ├── LayerNorm (TTNN)
    ├── Multi-Head Attention (TTNN)
    │   ├── Fused QKV projection
    │   ├── scaled_dot_product_attention
    │   └── Output projection
    ├── Residual Add (TTNN)
    ├── LayerNorm (TTNN)
    ├── MLP (TTNN)
    │   ├── FC1 + GELU
    │   └── FC2
    └── Residual Add (TTNN)
    ↓
Post LayerNorm (TTNN)
    ↓
Output (TTNN)
```

## Implementation Details

### 1. SigLIPAttentionTTNN
**Based on**: `models/demos/gemma3/tt/gemma_image_attention.py`

**Key Features**:
- Fused QKV weight concatenation for single linear operation
- Uses `ttnn.transformer.scaled_dot_product_attention` with optimized SDPA config
- Proper head splitting with `ttnn.experimental.nlp_create_qkv_heads`
- Head concatenation with `ttnn.experimental.nlp_concat_heads`
- Memory management with explicit deallocations
- Bias support for all projections

**Configuration**:
```python
compute_kernel_config = WormholeComputeKernelConfig(
    math_fidelity=HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True
)

sdpa_cfg = SDPAProgramConfig(
    compute_with_storage_grid_size=(8, 8),
    q_chunk_size=256,
    k_chunk_size=256,
    exp_approx_mode=False
)
```

### 2. SigLIPMLPTTNN
**Based on**: `models/demos/gemma3/tt/gemma_image_mlp.py`

**Key Features**:
- Two linear layers (FC1: hidden → intermediate, FC2: intermediate → hidden)
- Fused GELU activation in first linear operation
- Proper bias handling
- Memory deallocation between operations

### 3. SigLIPBlockTTNN
**Based on**: `models/demos/gemma3/tt/gemma_image_block.py`

**Key Features**:
- Pre-LayerNorm architecture
- Residual connections using `ttnn.add`
- Proper shape management (3D ↔ 4D conversions)
- Memory cleanup with deallocations

### 4. SigLIPVisionTowerTTNN
**Updated**: Now uses all TTNN blocks instead of torch fallback

**Changes**:
- ✅ Initializes 27 TTNN transformer blocks
- ✅ Removes torch fallback loop
- ✅ Runs all blocks on device
- ✅ Final LayerNorm on device

## Performance Improvements

### Computation Distribution
**Before**:
- Device: ~5% (position embedding only)
- Host: ~95% (all transformer blocks)

**After**:
- Device: ~95% (all transformer blocks + norms)
- Host: ~5% (Conv2d patch embedding only)

### Expected Benefits
1. **Reduced Host-Device Transfers**: Only one transfer after patch embedding
2. **Hardware Acceleration**: All attention/MLP operations on Tenstorrent hardware
3. **Optimized Kernels**: Uses proven Gemma3 kernels with proper program configs
4. **Memory Efficiency**: Explicit deallocations prevent memory buildup

## Testing Recommendations

1. **Unit Tests**: Test each component individually
   - `SigLIPAttentionTTNN` with known inputs
   - `SigLIPMLPTTNN` with known inputs
   - `SigLIPBlockTTNN` with residual connections

2. **Integration Tests**: 
   - Compare `SigLIPVisionTowerTTNN` output with `SigLIPVisionTowerTorch`
   - PCC (Pearson Correlation Coefficient) should be > 0.99

3. **End-to-End Tests**:
   - Run full PI0 model with TTNN SigLIP
   - Verify action generation quality
   - Measure latency improvements

## Usage

```python
import ttnn
from ttnn_siglip import SigLIPConfig, SigLIPVisionTowerTTNN

# Initialize
config = SigLIPConfig()
device = ttnn.open_device(device_id=0)

# Load weights (from HuggingFace or checkpoint)
weights = load_siglip_weights(...)

# Create TTNN vision tower
vision_tower = SigLIPVisionTowerTTNN(config, weights, device)

# Process images
images = torch.randn(1, 3, 224, 224)  # PyTorch input
embeddings = vision_tower.forward(images)  # TTNN output

# embeddings is now a TTNN tensor on device
```

## Backward Compatibility

The default exports still point to PyTorch versions:
```python
# Default exports (backward compatible)
SigLIPVisionTower = SigLIPVisionTowerTorch
```

To use TTNN versions, explicitly import:
```python
from ttnn_siglip import SigLIPVisionTowerTTNN
```

## References

1. **Gemma3 SigLIP Implementation**: 
   - `models/demos/gemma3/tt/gemma_image_attention.py`
   - `models/demos/gemma3/tt/gemma_image_mlp.py`
   - `models/demos/gemma3/tt/gemma_image_block.py`

2. **ViT Implementation** (alternative reference):
   - `models/demos/grayskull/vit/tt/ttnn_optimized_vit_highres_gs.py`

3. **SigLIP Attention** (original):
   - `models/demos/siglip/tt/attention.py`

## Known Limitations

1. **Patch Embedding**: Still uses PyTorch Conv2d on host
   - Could be optimized with `ttnn.fold` (from ViT implementation)
   - Current approach is pragmatic and proven

2. **Device Compatibility**: Optimized for Wormhole architecture
   - Compute kernel configs may need adjustment for Grayskull

3. **Memory Config**: Currently uses DRAM_MEMORY_CONFIG
   - Could explore L1_MEMORY_CONFIG for better performance
   - Requires careful memory capacity management

## Next Steps

1. ✅ Implement TTNN blocks (COMPLETED)
2. 🔄 Test PCC against PyTorch reference
3. 🔄 Measure performance improvements
4. 🔄 Optimize patch embedding with `ttnn.fold`
5. 🔄 Fine-tune memory configs for L1 usage

