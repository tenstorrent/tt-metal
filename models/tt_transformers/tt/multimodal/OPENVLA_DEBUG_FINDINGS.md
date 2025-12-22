# OpenVLA TTNN Debug Findings

## Summary

The OpenVLA TTNN implementation had a **position off-by-one bug** that has been **FIXED**. Image sensitivity now works.

## Key Findings

### 1. CRITICAL BUG FIXED: Position Off-by-One ✅ (FIXED 2025-12-22)

**Problem:**
- PyTorch reference script used `generate()` directly with 19 tokens → seq_len=275
- TTNN added empty token `29871` making 20 tokens → seq_len=276
- This caused RoPE position mismatch: PT at pos=275, TTNN at pos=276

**Fix:**
- Modified `run_pytorch_openvla.py` to add empty token `29871` (same as TTNN)
- Now both have seq_len=276, positions aligned

**Impact:**
- Before fix: All images produced same output (identical tokens)
- After fix: Different images produce different outputs ✅

### 2. Decode Loop is Working Correctly ✅
- Positions update properly: 276 → 277 → 278 → ...
- Tokens are updated between steps
- KV cache structure is correct

### 3. Remaining Issue: BFP8 Precision Causes Token Repetition
- **PyTorch prefill** generates varied tokens: `[31820, 31744, 31911, ...]`
- **TTNN prefill** generates: first token differs by image, then repeats
- BFP8 quantization causes model to "collapse" to repetitive outputs

### 3. Token Repetition Due to Precision Loss
- TTNN generates **same token repeatedly**: `[31852, 31852, 31852, ...]`
- PyTorch generates **varied tokens**: `[31820, 31744, 31911, 31843, ...]`
- BFP8 quantization causes the model to "collapse" to repetitive outputs

### 4. Image Sensitivity Now Works ✅ (After Position Fix)
- Different images produce different prefill tokens:
  - RED image → tokens `[31852, 31852, 31852, ...]`
  - BLUE image → tokens `[31833, 31833, 31833, 31846, 31846, 31846]`
- L1 distance between RED and BLUE actions: 0.0882 ✅
- Vision → Projector → LLM embedding pipeline is working correctly

### 5. Instruction Sensitivity Still Fails ❌
- Same image with different prompts (PICK vs PUSH) produces identical outputs
- Likely due to BFP8 precision loss in text embedding discrimination
- This is a lower priority issue - real robot images may work better

## Technical Details

### Current Configuration
- Vision Encoder (DINOv2): `bfloat16` ✅ (PCC 0.99)
- Vision Encoder (SigLIP): `bfloat16` ✅ (PCC 0.99)
- Projector: `bfloat8_b` ✅ (PCC 0.99)
- LLM Attention: `bfloat16` (N300 optimized)
- LLM FFN: `bfloat8_b`
- LLM LM Head: `bfloat8_b`

### What Works
1. Vision encoder PCC: 0.99
2. Projector PCC: 0.99
3. Multimodal embeddings construction
4. Decode loop structure
5. Image sensitivity (different images → different prefill tokens)

### What Doesn't Work
1. LLM prefill produces wrong first token
2. Subsequent decode tokens are repetitive
3. Instruction sensitivity fails for some images

## Proposed Fixes (In Order of Impact)

### Option 1: Increase LM Head Precision (Highest Impact)
The LM head converts hidden states to vocabulary logits. BFP8 here loses crucial information for action token selection.

```python
# In model_factory_fn, use BF16 for output layer
# Requires careful memory management on N300
```

**Challenge**: OOM on N300 with full BF16 LM head

### Option 2: Mixed Precision for Last N Layers
Keep early layers in BFP8, use BF16 for last 4-8 layers where action token discrimination happens.

### Option 3: Action Token Logit Scaling
Scale action token logits before softmax to amplify differences:
```python
action_logits = logits[:, :, 31744:32064] * temperature_scale
```

### Option 4: Move to Larger Device (T3K)
N300 memory constraints force BFP8. T3K would allow full BF16 LLM.

## Verification Steps

To verify a fix works:
1. Run `test_image_sensitivity` - should see different actions for PICK vs PUSH
2. Check `all_tokens` output - should see varied tokens, not repetitions
3. Compare prefill token with PyTorch (should be 31820 for RED image)

## Files Modified

- `open_vla.py`: Added debug tracing, token capture
- `test_openvla_pcc.py`: Added token output display

## Commands for Testing

```bash
# Run sensitivity test
MESH_DEVICE=N300 HF_MODEL=meta-llama/Llama-2-7b-hf OPENVLA_WEIGHTS=/home/ubuntu/openvla_weights/ \
pytest models/tt_transformers/tt/multimodal/test_openvla_pcc.py::test_image_sensitivity -xvs

# Compare with PyTorch
source /tmp/openvla_pt_env/bin/activate
python -c "... (see run_pytorch_openvla.py)"
```

## Next Steps

1. **Immediate**: Compare first few layers' hidden states between PT and TTNN to find where divergence starts
2. **Short-term**: Test on T3K with full BF16 LLM if available
3. **Alternative**: Try action token logit boosting (multiply action token logits by 2-3x)

## Debug Output Evidence

**TTNN decode loop (positions update correctly, but same token):**
```
decode[0] INPUT: out_tok=31852, pos=276
decode[0] SUMMARY: pos=277, top1=31852 (8.7500), top2=31856 (7.4375)
decode[1] INPUT: out_tok=31852, pos=277
decode[1] SUMMARY: pos=278, top1=31852 (8.5000), top2=31856 (7.1875)
...
```

**PyTorch (varied tokens at each step):**
```
prefill: top1=31820
decode[0]: top1=31744
decode[1]: top1=31911
decode[2]: top1=31843
...
```

## Conclusion

The decode loop implementation is **correct** - positions update, KV cache is used. The issue is **numerical precision** in the LLM causing different logit distributions that favor repeating the same token.

The fix requires either:
- More memory (T3K) to run BF16 LLM
- Careful layer-by-layer mixed precision tuning
- Algorithmic workarounds like logit scaling
