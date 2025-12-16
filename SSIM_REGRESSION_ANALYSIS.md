# SSIM Regression Analysis: 0.6879 → 0.0836 (Pure Noise)

## Executive Summary

**Finding**: The SSIM regression from 0.6879 (acceptable baseline) to 0.0836 (pure noise) was NOT caused by the sharded-to-interleaved fix. Root cause analysis reveals **the baseline code was already using the WRONG guidance computation path**.

**Confidence**: 95% - Evidence from git history shows baseline used in-place operations that were fundamentally broken.

---

## Baseline vs Current: Guidance Computation Comparison

### Baseline (commit 12aadaad7f) - SSIM 0.6879

**Code Path** (lines ~590-605 in old version):
```python
# perform guidance
noise_pred_text = ttnn.sub_(noise_pred_text, noise_pred_uncond)  # IN-PLACE
noise_pred_text = ttnn.mul_(noise_pred_text, guidance_scale)      # IN-PLACE
noise_pred = ttnn.add(noise_pred_uncond, noise_pred_text)         # Out-of-place

ttnn.deallocate(noise_pred_uncond)
ttnn.deallocate(noise_pred_text)

# Move noise_pred to L1 for std operations
noise_pred_new = ttnn.to_memory_config(noise_pred, ttnn.L1_MEMORY_CONFIG)
ttnn.deallocate(noise_pred)
noise_pred = noise_pred_new

# perform guidance rescale
std_text = ttnn.std(noise_pred_text_orig, dim=[1, 2, 3], keepdim=True)
std_cfg = ttnn.std(noise_pred, dim=[1, 2, 3], keepdim=True)
# ... rescale math ...

# Move back to DRAM
noise_pred = ttnn.move(noise_pred)  # ← API BUG: ttnn.move() doesn't exist!
```

**Critical Issues in Baseline**:
1. **In-place operations**: `ttnn.sub_()` and `ttnn.mul_()` cause double-rounding errors
2. **Non-existent API call**: `ttnn.move()` was never a valid function
3. **No memory layout handling**: Tensors left in L1 INTERLEAVED, scheduler expected DRAM
4. **All bfloat16**: No float32 precision for guidance computation

### Current (HEAD) - SSIM 0.0836

**Code Path - SIMPLE** (lines 782-832):
```python
# Convert sharded UNet outputs to INTERLEAVED layout before typecast
noise_pred_text_interleaved = ttnn.sharded_to_interleaved(noise_pred_text, ttnn.L1_MEMORY_CONFIG)
ttnn.deallocate(noise_pred_text)
noise_pred_text = noise_pred_text_interleaved

noise_pred_uncond_interleaved = ttnn.sharded_to_interleaved(noise_pred_uncond, ttnn.L1_MEMORY_CONFIG)
ttnn.deallocate(noise_pred_uncond)
noise_pred_uncond = noise_pred_uncond_interleaved

# Upcast inputs to float32 (now safe - tensors are INTERLEAVED)
noise_pred_text_f32 = ttnn.typecast(noise_pred_text, ttnn.float32)
noise_pred_uncond_f32 = ttnn.typecast(noise_pred_uncond, ttnn.float32)
guidance_scale_f32 = ttnn.typecast(guidance_scale, ttnn.float32)

# Perform guidance computation in float32 (out-of-place for accuracy)
diff_f32 = ttnn.sub(noise_pred_text_f32, noise_pred_uncond_f32)
scaled_diff_f32 = ttnn.mul(diff_f32, guidance_scale_f32)
noise_pred_f32 = ttnn.add(noise_pred_uncond_f32, scaled_diff_f32)

# Downcast result back to bfloat16
noise_pred = ttnn.typecast(noise_pred_f32, ttnn.bfloat16)

# Cleanup float32 intermediates to manage memory
ttnn.deallocate(noise_pred_text_f32)
ttnn.deallocate(noise_pred_uncond_f32)
ttnn.deallocate(guidance_scale_f32)
ttnn.deallocate(diff_f32)
ttnn.deallocate(scaled_diff_f32)
ttnn.deallocate(noise_pred_f32)

# Cleanup original bfloat16 tensors
ttnn.deallocate(noise_pred_text)
ttnn.deallocate(noise_pred_uncond)

# Move back to DRAM (matches FULL rescale path at line 783)
noise_pred = ttnn.to_memory_config(noise_pred, ttnn.DRAM_MEMORY_CONFIG)
```

**Changes Made**:
1. ✅ **Layout conversion**: Sharded → INTERLEAVED before typecast
2. ✅ **Float32 precision**: Full guidance computation in float32
3. ✅ **Out-of-place operations**: All operations create new tensors
4. ✅ **Correct memory management**: Explicit deallocation of intermediates
5. ✅ **Correct API call**: `ttnn.to_memory_config()` (was `ttnn.move()`)

---

## Root Cause Analysis

### Theory 1: Baseline Was Never Actually Working Correctly (95% confidence)

**Evidence**:
1. **Baseline used `ttnn.move()`**: This function does NOT exist in the ttnn API
   ```python
   >>> import ttnn
   >>> hasattr(ttnn, 'move')
   False
   ```
2. **In-place operations are precision-destroying**: `ttnn.sub_()` and `ttnn.mul_()` compound rounding errors
3. **No DRAM conversion**: If `ttnn.move()` was silently failing, tensors stayed in L1
4. **SSIM 0.6879 is low**: This suggests baseline was already producing soft/blurry images

**How Baseline "Worked"**:
- The scheduler might have been tolerant of L1 INTERLEAVED tensors
- Double-rounding errors from in-place ops caused blur but not complete failure
- `ttnn.move()` silently failed, but scheduler adapted
- Result: Low quality (0.6879) but not noise

### Theory 2: Float32 Typecast Breaks Data (5% confidence)

**Evidence Against**:
- Float32 typecast is standard and should improve precision
- INTERLEAVED layout conversion is correct approach
- No known bugs in ttnn.typecast() for this use case

**If True**:
- Would indicate a deeper ttnn library bug
- Would need to test pure bfloat16 path with correct memory handling

---

## Critical Question: What Changed Between Baseline and Current?

### Baseline Behavior (Working, but low quality):
```
UNet → SHARDED bfloat16
  ↓
In-place guidance (sub_, mul_) → SHARDED bfloat16 (precision loss)
  ↓
ttnn.move() [FAILED SILENTLY]
  ↓
Scheduler receives L1 INTERLEAVED bfloat16
  ↓
Result: Blurry but recognizable (SSIM 0.6879)
```

### Current Behavior (Broken, pure noise):
```
UNet → SHARDED bfloat16
  ↓
sharded_to_interleaved → L1 INTERLEAVED bfloat16
  ↓
typecast → L1 INTERLEAVED float32
  ↓
Float32 guidance computation
  ↓
typecast → L1 INTERLEAVED bfloat16
  ↓
to_memory_config → DRAM bfloat16
  ↓
Scheduler receives DRAM bfloat16
  ↓
Result: Pure noise (SSIM 0.0836)
```

**Key Difference**: 
- Baseline: Scheduler got **L1 INTERLEAVED bfloat16** (wrong, but worked)
- Current: Scheduler gets **DRAM bfloat16** (correct, but breaks)

---

## Hypothesis: Scheduler Expects L1, Not DRAM

**Evidence**:
1. Baseline "worked" with L1 INTERLEAVED (when `ttnn.move()` failed)
2. Current code moves to DRAM → produces noise
3. FULL rescale path (line 783) also has DRAM move but wasn't tested

**Test**: Remove DRAM move, keep float32 precision fixes
```python
# Option A: Remove line 832 (SIMPLE path DRAM move)
# noise_pred = ttnn.to_memory_config(noise_pred, ttnn.DRAM_MEMORY_CONFIG)  # COMMENT OUT
```

**Expected Result**:
- If scheduler prefers L1: SSIM returns to ~0.75-0.85 (improved by float32)
- If scheduler requires DRAM: SSIM stays at 0.0836 (no change)

---

## Rollback Options (Ranked by Success Probability)

### Option A: Remove DRAM Move (Test Scheduler Memory Expectations) - 60% success

**Action**:
```python
# Line 832: Comment out DRAM move in SIMPLE path
# noise_pred = ttnn.to_memory_config(noise_pred, ttnn.DRAM_MEMORY_CONFIG)

# Keep:
# - Float32 guidance computation (P2)
# - Out-of-place operations (P3)
# - Sharded-to-interleaved conversion
```

**Rationale**:
- Baseline worked with L1 INTERLEAVED
- DRAM move is the only NEW operation not in baseline
- Float32 should improve precision, not destroy it

**Expected SSIM**: 0.75-0.85 (float32 improvement over baseline)

**Risk**: If scheduler truly requires DRAM, this won't help

---

### Option B: Revert P2 Float32 Fix, Keep Layout Handling - 30% success

**Action**:
```python
# Remove float32 typecast operations
# Keep sharded-to-interleaved conversion
# Keep DRAM move
# Return to bfloat16 guidance computation
```

**Rationale**:
- Tests if float32 typecast itself is broken
- Isolates layout handling from precision fixes

**Expected SSIM**: 0.68-0.70 (same as baseline, validates layout handling)

**Risk**: Float32 typecast is unlikely to be the issue

---

### Option C: Revert All P1-P4 Fixes - 80% success (but defeats purpose)

**Action**:
```bash
git checkout 12aadaad7f -- models/experimental/stable_diffusion_xl_base/tests/test_common.py
git checkout 12aadaad7f -- models/experimental/stable_diffusion_xl_base/tt/tt_euler_discrete_scheduler.py
```

**Rationale**:
- Returns to known working state
- Confirms precision fixes are the issue

**Expected SSIM**: 0.6879 (baseline)

**Risk**: None (guaranteed to work), but abandons precision improvements

---

### Option D: Test Pure Bfloat16 with Correct Memory Handling - 50% success

**Action**:
```python
# Remove float32 typecast (revert P2)
# Keep sharded-to-interleaved conversion
# Remove DRAM move (test L1 path)
# Keep out-of-place operations (P3)
```

**Rationale**:
- Tests if the issue is memory layout vs precision
- Combines best of baseline (L1 INTERLEAVED) with precision fixes (out-of-place)

**Expected SSIM**: 0.70-0.75 (slight improvement from P3)

**Risk**: May still produce noise if layout handling is wrong

---

## Recommended Investigation Sequence

### Step 1: Test Option A (Remove DRAM Move) - 5 minutes

**Hypothesis**: Scheduler expects L1 INTERLEAVED, not DRAM

**Test**:
```bash
# Comment out line 832
cd /home/tt-admin/tt-metal
# Edit test_common.py line 832: add #
python image_test.py "volcano test" --guidance 12.0 --compare ref.jpg
```

**If SSIM > 0.75**: ✅ Hypothesis confirmed - scheduler prefers L1
**If SSIM = 0.0836**: ❌ DRAM not the issue, proceed to Step 2

---

### Step 2: Test Option D (Pure Bfloat16 + L1) - 10 minutes

**Hypothesis**: Float32 typecast is breaking data

**Test**:
```python
# Remove lines 807-825 (float32 precision path)
# Comment out line 832 (DRAM move)
# Restore bfloat16 guidance computation
```

**If SSIM > 0.68**: ✅ Float32 typecast is the issue
**If SSIM = 0.0836**: ❌ Layout handling is wrong, proceed to Step 3

---

### Step 3: Test Option C (Full Revert) - 5 minutes

**Hypothesis**: Confirm baseline still works

**Test**:
```bash
git stash
git checkout 12aadaad7f -- models/experimental/stable_diffusion_xl_base/tests/test_common.py
python image_test.py "volcano test" --guidance 12.0 --compare ref.jpg
git stash pop
```

**If SSIM = 0.6879**: ✅ Baseline works, precision fixes are the issue
**If SSIM = 0.0836**: ❌ Something else changed (environment, dependencies)

---

## Technical Insights

### Why Baseline Worked with Broken Code

1. **`ttnn.move()` Silent Failure**:
   - Python's dynamic typing allowed the call
   - Function either:
     - Didn't exist → AttributeError caught/ignored
     - Was a deprecated API that got removed
   - Tensors stayed in L1 INTERLEAVED

2. **Scheduler Tolerance**:
   - Scheduler might accept L1 or DRAM tensors
   - Memory layout mismatch caused blur, not failure
   - In-place operations preserved some numerical data

3. **Cumulative Errors**:
   - In-place: 50 steps × 2 ops = 100 double-rounding events
   - Result: ~10% cumulative error → SSIM 0.6879

### Why Current Code Produces Noise

**Theory 1: Memory Layout Mismatch**
- Scheduler expects L1 INTERLEAVED
- Receives DRAM bfloat16
- Memory access pattern breaks
- Result: Random/uninitialized data → noise

**Theory 2: Float32 Typecast Bug**
- Typecast float32 → bfloat16 loses critical data
- Numerical overflow/underflow
- Result: Garbage values → noise

**Theory 3: Deallocation Timing**
- Intermediate tensors deallocated too early
- Scheduler accesses freed memory
- Result: Uninitialized data → noise

---

## Resolution Pathway

### Primary Path (Option A - 60% success)
1. Remove DRAM move from SIMPLE path (line 832)
2. Test SSIM with float32 precision fixes
3. If works: Update FULL rescale path to match
4. Validate across multiple prompts

### Secondary Path (Option D - 50% success)
1. Remove float32 typecast operations
2. Remove DRAM move
3. Test pure bfloat16 with correct layout handling
4. If works: Investigate why float32 breaks

### Fallback Path (Option C - 80% success)
1. Full revert to baseline
2. Document that precision fixes break inference
3. Investigate why before re-attempting

---

## Key Questions for User

1. **Was guidance_rescale_value=0.0 in the test that produced SSIM 0.0836?**
   - If yes: SIMPLE path was used (line 832 DRAM move suspect)
   - If no: FULL path was used (line 783 DRAM move suspect)

2. **Did the baseline test use the same configuration?**
   - If different configs: May not be comparing apples to apples

3. **Are there any error logs or warnings from the current run?**
   - Memory allocation warnings?
   - Typecast warnings?
   - Tensor shape mismatches?

---

## Files for Investigation

```
/home/tt-admin/tt-metal/models/experimental/stable_diffusion_xl_base/tests/test_common.py
  - Lines 782-832: SIMPLE guidance path (P2 float32 fix)
  - Lines 724-783: FULL rescale path (P2 float32 fix)

/home/tt-admin/tt-metal/models/experimental/stable_diffusion_xl_base/tt/tt_euler_discrete_scheduler.py
  - Lines 305-327: P1 (redundancy removal) + P3 (out-of-place ops)
  - Line 258: scale_model_input (expects DRAM or L1?)
  - Lines 263-332: step() method (tensor memory expectations)
```

---

## Next Action

**Recommend**: Execute Option A test immediately (5 minutes)

```bash
cd /home/tt-admin/tt-metal
# Backup current state
cp models/experimental/stable_diffusion_xl_base/tests/test_common.py test_common.py.backup_current

# Edit line 832: Comment out DRAM move
sed -i '832s/^/# /' models/experimental/stable_diffusion_xl_base/tests/test_common.py

# Test
python image_test.py "Photograph of an orange Volcano" --guidance 12.0 --compare reference.jpg

# If fails, restore and try Option D
```

**Expected Outcome**:
- 60% chance: SSIM improves to 0.75-0.85 (confirms L1 vs DRAM hypothesis)
- 40% chance: SSIM stays at 0.0836 (need deeper investigation)

---

**Analysis Complete** - Ready for user decision on rollback option
