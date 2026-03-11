# Linear Attention Implementation Status

## Summary

We've implemented all four core methods in `tt/linear_attention.py`:

### ✅ 1. Causal Conv1D (`_causal_conv1d`)
**Status**: Implemented for both prefill and decode modes

**Implementation**:
- Uses `ttnn.conv1d` with depthwise (grouped) convolution
- **Prefill mode**: Full sequence convolution with left-padding (padding=3)
- **Decode mode**: Maintains sliding window state, concatenates with new input
- State management: Stores last (kernel_size - 1) positions for decode

**TTNN Operations Used**:
- `ttnn.conv1d` - 1D convolution
- `ttnn.permute` - dimension reordering
- `ttnn.concat` - concatenating conv state with new input
- `ttnn.slice` - extracting conv state from sequence

**Based on**: Mamba conv1d implementation (`models/demos/wormhole/mamba/tt/mamba_conv.py`)

---

### ✅ 2. Query Gating (`_query_gating`)
**Status**: Fully implemented

**Implementation**:
```
gate = sigmoid(linear(flatten(Q)) + bias)
output = gate * attention_output
```

**Flow**:
1. Flatten Q: `[B, L, H, D] -> [B, L, H*D]`
2. Linear projection: `[B, L, H*D] -> [B, L, H]`
3. Add bias and apply sigmoid
4. Expand and multiply with attention output

**TTNN Operations Used**:
- `ttnn.reshape` - tensor reshaping
- `linear` (from ttnn_functional_common) - linear projection
- `ttnn.add` - bias addition
- `ttnn.sigmoid` - activation
- `ttnn.mul` - element-wise gating

---

### ✅ 3. Output Gating (`_output_gating`)
**Status**: Fully implemented

**Implementation**:
```
gate = sigmoid(linear(x))
output = gate * attention_output
```

**Flow**:
1. Project input: `[B, L, dim] -> [B, L, H*D]`
2. Apply sigmoid
3. Reshape and multiply with attention output

**TTNN Operations Used**:
- `linear` - projection
- `ttnn.sigmoid` - activation
- `ttnn.reshape` - dimension matching
- `ttnn.mul` - gating

---

### ⚠️ 4. Delta Rule (`_delta_rule`)
**Status**: Initial implementation (needs refinement)

**Implementation**:

#### Decode Mode (Single Token)
- ✅ State management with dict structure (`kv_state`, `k_sum`, `beta`)
- ✅ Cumulative K sum update
- ✅ Beta gating computation: `beta = sigmoid(cumsum(K))`
- ✅ Delta beta: `delta_beta = beta[t] - beta[t-1]`
- ⚠️ KV outer product with head grouping (needs validation)
- ⚠️ Output computation (placeholder - needs correct implementation)

#### Prefill Mode (Full Sequence)
- ✅ Cumulative sum along sequence: `ttnn.cumsum(k, dim=1)`
- ✅ Beta computation: `sigmoid(K_cumsum)`
- ✅ Delta beta computation
- ⚠️ Actual attention computation (placeholder - needs proper implementation)

**Known Issues**:
1. **Dimension mismatches**: Need to handle head count differences
   - K: 16 heads × 128 dim = 2048
   - V: 32 heads × 128 dim = 4096
   - Q: 16 heads × 256 dim = 4096

2. **Output computation**: The matmul between state and Q needs proper dimension alignment

3. **Prefill mode**: Currently returns zeros - needs chunked iteration or custom kernel

4. **State structure**: Returns dict instead of tensor (test expects tensor)

**TTNN Operations Used**:
- `ttnn.cumsum` - cumulative sum
- `ttnn.sigmoid` - beta gating
- `ttnn.subtract` - delta computation
- `ttnn.squeeze` / `ttnn.reshape` - dimension manipulation
- `ttnn.matmul` - outer products and attention
- `ttnn.add` / `ttnn.mul` - state updates
- `ttnn.zeros` - initialization

---

## Testing

### Created Test Files

1. **`tests/test_linear_attention_basic.py`** (NEW)
   - Smoke tests without full model weights
   - Tests instantiation, weight shapes, individual operations
   - Can run with `dummy_weights=True`

2. **`tests/test_linear_attention.py`** (EXISTING)
   - Full integration tests with real weights
   - Tests prefill and decode modes
   - Compares against PyTorch reference
   - Requires real model weights

### Running Tests

```bash
# Basic smoke tests (no weights needed)
pytest models/demos/qwen35/tests/test_linear_attention_basic.py -v

# Full integration tests (requires weights)
pytest models/demos/qwen35/tests/test_linear_attention.py -v

# Specific test
pytest models/demos/qwen35/tests/test_linear_attention_basic.py::test_causal_conv1d_operation -v
```

---

## Next Steps

### High Priority
1. **Fix Delta Rule Output Computation**
   - Correctly implement `state @ Q` matmul with proper dimension handling
   - Handle head count mismatches (16 K heads, 32 V heads, 16 Q heads)
   - Validate against PyTorch reference implementation

2. **Test Basic Operations**
   - Run `test_linear_attention_basic.py` to verify individual ops
   - Fix any API or dimension errors
   - Ensure operations execute without crashes

3. **Delta Rule Prefill Mode**
   - Implement proper chunked processing
   - Accumulate weighted KV contributions correctly
   - Consider iterative approach if parallel scan not feasible

### Medium Priority
4. **State Management**
   - Decide on dict vs tensor for recurrent_state
   - Update tests if using dict structure
   - Ensure decode-to-decode continuity works

5. **Conv1D Decode Mode**
   - Test state updates between decode steps
   - Verify causal property maintained

### Low Priority
6. **Optimization**
   - Profile memory usage
   - Optimize tensor layouts
   - Consider fused kernels for delta rule

7. **Reference Comparison**
   - Create PyTorch reference implementation
   - Compare outputs for correctness (PCC > 0.99)
   - Test with real model weights

---

## Implementation Metrics

| Component | Lines of Code | Complexity | Estimated Correctness |
|-----------|--------------|------------|----------------------|
| Causal Conv1D | ~100 | Medium | 70% (needs testing) |
| Query Gating | ~25 | Low | 90% (straightforward) |
| Output Gating | ~20 | Low | 90% (straightforward) |
| Delta Rule | ~150 | High | 40% (needs major work) |

**Total**: ~295 lines of implementation code added

---

## References

- TTNN Operations: All required ops available (`conv1d`, `cumsum`, `sigmoid`, `matmul`, etc.)
- Mamba Conv1D: `models/demos/wormhole/mamba/tt/mamba_conv.py`
- Flash Linear Attention: https://github.com/fla-org/flash-linear-attention
- Gated DeltaNet Paper: https://arxiv.org/abs/2412.06464

---

## Files Modified

1. **`models/demos/qwen35/tt/linear_attention.py`**
   - Implemented all four methods
   - Added implementation status comments
   - ~295 lines added

2. **`models/demos/qwen35/tests/test_linear_attention_basic.py`** (NEW)
   - Basic smoke tests
   - ~150 lines

3. **`models/demos/qwen35/LINEAR_ATTENTION_OPS.md`** (NEW)
   - Detailed operations analysis
   - Implementation guide

4. **`models/demos/qwen35/IMPLEMENTATION_STATUS.md`** (THIS FILE)
   - Current status tracking
