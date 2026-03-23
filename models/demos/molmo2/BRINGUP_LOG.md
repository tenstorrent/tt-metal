# Molmo2-8B Bringup Log

## Current Status: FIXED - MLP Bug + Video Reshape Bug Resolved

### Summary
- Model loads and initializes on T3K (8 devices)
- Vision backbone runs (~86ms traced, ~2s untraced)
- Prefill runs (~145-190ms)
- **Fixed MLP gate/up order bug - model now outputs correct tokens!**
- **Fixed video reshape bug - video understanding now works!**
- Text model now matches HuggingFace reference exactly (PCC > 0.999)

### Performance (Current - Non-optimized)
- Vision processing: ~2000ms (needs optimization, should be ~86ms traced)
- Prefill TTFT: ~145-190ms
- Decode throughput: ~5.3 tok/s (expected ~35 tok/s)

### Bug Fixed: MLP Gate/Up Order
**Root cause**: The SwiGLU MLP had the gate and up projections swapped.

HuggingFace does:
```python
x, gate = ff_proj_out.chunk(2, dim=-1)  # First half = x, Second half = gate
output = silu(gate) * x
```

Our code was doing:
```python
gate = ff_proj_out[:intermediate_dim]   # WRONG: treated first half as gate
up = ff_proj_out[intermediate_dim:]     # WRONG: treated second half as up
output = silu(gate) * up                # WRONG: silu(first) * second
```

**Fix**: Swap the order in text_mlp.py and test_pcc_all_layers.py:
```python
up_proj = ff_proj[:intermediate_dim, :]    # First half = up (value)
gate_proj = ff_proj[intermediate_dim:, :]  # Second half = gate (activation)
output = silu(gate) * up                   # Correct: silu(second) * first
```

### Files Changed This Session
- `models/demos/molmo2/tt/text_mlp.py` - **FIXED** gate/up order in weight loading
- `models/demos/molmo2/tests/test_pcc_all_layers.py` - **FIXED** reference MLP implementation
- `models/demos/molmo2/tt/text_rotary_setup.py` - Fixed RoPE format (HF style, return full cache)
- `models/demos/molmo2/demo/demo.py` - Added chat template wrapping
- `models/demos/molmo2/tt/vision_attention.py` - **FIXED** video reshape (add divisibility check)
- `models/demos/molmo2/tt/vision_mlp.py` - **FIXED** video reshape (add divisibility check)
- `models/demos/molmo2/tt/image_projector.py` - **FIXED** video reshape (add divisibility check)

### Verification - All Tests Pass ✓
| Prompt | Output | Status |
|--------|--------|--------|
| "The capital of France is" | "Paris" | ✅ |
| "The largest planet in our solar system is" | "Jupiter" | ✅ |
| "Water boils at" | "100°C (212°F)" | ✅ |
| "What is 1 + 1?" | "1 + 1 = 2" | ✅ |
| Image of dog (multimodal) | "a dog. Specifically, it appears to be a small puppy" | ✅ |
| Video (letter writing) | "B. a" (correct answer) | ✅ |

### PCC Verification Results (Post-Fix)
```
MLP Block 0: PCC=0.999992
```

### Performance (Current)
- Vision processing: ~4000ms (needs optimization with tracing)
- Prefill TTFT: ~196ms
- Decode throughput: ~5.0 tok/s

### Next Steps
1. ✅ Re-run full model generation test - PASSED
2. ✅ Test with vision inputs - PASSED
3. Optimize vision processing (currently ~4s, target ~86ms with tracing)
4. Optimize decode throughput (currently ~5.0 tok/s, target ~35 tok/s)

### Bug Fixed: Video Reshape
**Root cause**: Vision modules assumed sequence length divisible by 1024/2048.
Video input has 8 frames × 729 patches = 5832 tokens (not divisible).

**Fix**: Add divisibility check before reshape:
```python
# Before:
if seq_len > 1024:
    x = ttnn.reshape(x, [1, seq_len // 1024, 1024, -1])
# After:
if seq_len > 1024 and seq_len % 1024 == 0:
    x = ttnn.reshape(x, [1, seq_len // 1024, 1024, -1])
```

### Technical Notes
- Chat template format: `<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n`
- Image patch token ID: 151938 (`<im_patch>`)
- layer_norm_eps: 1e-06 (HF config)
- SwiGLU: `silu(gate) * up` where gate is second half of ff_proj output
- Video: 8 frames × 729 patches = 5832 visual tokens

---
Last updated: 2026-03-23
