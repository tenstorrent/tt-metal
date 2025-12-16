# Implementation Plan: Option A - Deep Dive UNet Analysis for SSIM Degradation Root Cause

## Plan Overview

**Goal:** Identify the numerical differences in UNet outputs that cause SSIM degradation from 0.95 (target) to 0.6879 (current), specifically determining where "soft/less crisp" image artifacts originate in the denoising pipeline.

**Key Constraints:**
- No direct access to tt-media-server source code (the reference implementation)
- Previous logger/instrumentation attempts have had reliability issues
- Need to compare against PyTorch reference (diffusers library) as ground truth
- Must work within existing test infrastructure

**High-Level Success Criteria:**
- Identify the specific timestep(s) where TT UNet output diverges significantly from PyTorch reference
- Determine whether divergence is in raw UNet output, post-guidance computation, or scheduler step
- Narrow down to specific UNet component(s) if UNet output is the source
- Provide actionable data for numerical precision fixes

---

## Phases & Milestones

### Milestone 1: Instrumentation Infrastructure
**Deliverables:**
- Standalone comparison script that runs both PyTorch reference and TT UNet side-by-side
- Reliable file-based statistics capture (avoiding previous logger issues)
- JSON/CSV output format for easy analysis

**Success Criteria:**
- Script runs without errors on single device
- Captures complete statistics for all 50 denoising steps
- Data format allows programmatic comparison

### Milestone 2: Timestep-Level Divergence Analysis
**Deliverables:**
- Per-timestep PCC/correlation data between TT and PyTorch UNet outputs
- Identification of divergence onset point (first timestep with significant deviation)
- Visualization of divergence progression over timesteps

**Success Criteria:**
- Clear identification of when divergence starts (step 0? step 25? gradual?)
- Quantified divergence magnitude at each step

### Milestone 3: Output Component Analysis
**Deliverables:**
- Separate analysis of: raw UNet output, post-guidance output, scheduler step output
- Identification of which component introduces the largest error
- Error accumulation vs single-point error assessment

**Success Criteria:**
- Root cause narrowed to one of: UNet itself, guidance computation, scheduler step
- Quantified contribution of each component to final error

### Milestone 4: UNet Layer-Level Analysis (Conditional)
**Deliverables:**
- If UNet is identified as root cause: per-block analysis (down/mid/up blocks)
- Specific layer identification (attention, resnet, conv, norm)
- Precision/numerical stability assessment

**Success Criteria:**
- Identification of specific problematic layer(s) or operation(s)
- Actionable fix recommendations

---

## Detailed Action Sequence

### Phase 1: Instrumentation Strategy (Steps 1-4)

#### Step 1: Create Standalone Comparison Script
**File:** `/home/tt-admin/tt-metal/models/experimental/stable_diffusion_xl_base/tests/pcc/test_unet_deep_dive.py`

**Owner/Resources:** Developer with access to TT device
**Dependencies:** None (starts from existing test infrastructure)
**Estimated Effort:** 2-3 hours
**Expected Output:** New test file that initializes both PyTorch and TT UNet

**Approach:**
- Adapt from existing `test_unet_loop.py` and `test_module_tt_unet.py`
- Create parallel execution framework
- Use file-based output (JSON) instead of logger for reliability
- Key function: `run_comparison_analysis()`

```python
# Pseudo-structure
def run_comparison_analysis(ttnn_device, num_steps=50, output_dir="/tmp/unet_deep_dive"):
    # Initialize both UNets
    # Run side-by-side with identical inputs
    # Capture statistics at each step
    # Write to JSON files
```

#### Step 2: Design Statistics Capture Format
**File:** `/home/tt-admin/tt-metal/models/experimental/stable_diffusion_xl_base/tests/pcc/unet_analysis_utils.py`

**Owner/Resources:** Developer
**Dependencies:** Step 1 complete
**Estimated Effort:** 1 hour
**Expected Output:** Utility module for tensor statistics

**Metrics to Capture (per tensor):**
- Shape
- min, max, mean, std, median
- p1, p5, p10, p25, p50, p75, p90, p95, p99 (percentiles)
- L1 norm, L2 norm
- Number of NaN/Inf values
- Histogram bins (10 bins)

**Output Format:**
```json
{
  "step": 0,
  "timestep": 999,
  "pytorch": {
    "raw_unet_output": {...stats...},
    "post_guidance_output": {...stats...},
    "scheduler_step_output": {...stats...}
  },
  "tt": {
    "raw_unet_output": {...stats...},
    "post_guidance_output": {...stats...},
    "scheduler_step_output": {...stats...}
  },
  "comparison": {
    "raw_unet_pcc": 0.997,
    "raw_unet_mse": 1.2e-5,
    "post_guidance_pcc": 0.995,
    ...
  }
}
```

#### Step 3: Implement Reliable Output Capture
**Owner/Resources:** Developer
**Dependencies:** Steps 1-2
**Estimated Effort:** 1 hour
**Expected Output:** File-based capture mechanism

**Strategy:**
- Write JSON files directly (not via logger)
- Use `flush()` after each write
- Create separate files per timestep for resilience
- Aggregate at end of run

**File Structure:**
```
/tmp/unet_deep_dive/
  run_<timestamp>/
    step_000.json
    step_001.json
    ...
    step_049.json
    summary.json
```

#### Step 4: Implement Reference Pipeline Extraction
**Owner/Resources:** Developer
**Dependencies:** Existing diffusers reference code
**Estimated Effort:** 2 hours
**Expected Output:** PyTorch reference that matches TT pipeline exactly

**Key Points:**
- Use existing reference implementation as starting point
- Ensure identical:
  - Seed
  - Scheduler configuration
  - Guidance scale
  - Input latents
  - Prompt embeddings

---

### Phase 2: Data Collection (Steps 5-8)

#### Step 5: Define Timestep Sampling Strategy
**Owner/Resources:** Developer
**Dependencies:** Phase 1 complete
**Estimated Effort:** 30 minutes
**Expected Output:** Defined sampling points

**Initial Strategy:**
- Full capture: All 50 steps (steps 0-49)
- Key checkpoints: Step 0 (first), Step 24 (middle), Step 49 (last)
- Early steps: 0, 1, 2, 3, 4 (high noise, initial structure)
- Mid steps: 20, 21, 22, 23, 24, 25 (detail formation)
- Late steps: 45, 46, 47, 48, 49 (final refinement)

**Rationale:**
- Diffusion literature suggests different errors manifest at different stages
- Early steps: overall structure
- Late steps: fine details (likely source of "soft" artifacts)

#### Step 6: Implement Data Collection Run
**Owner/Resources:** Developer with device access
**Dependencies:** Steps 1-5
**Estimated Effort:** 1-2 hours (including runtime)
**Expected Output:** Complete dataset for one prompt

**Test Configuration:**
- Prompt: "A cat sitting on a windowsill" (deterministic test prompt)
- Seed: 42 (fixed for reproducibility)
- Steps: 50
- Guidance scale: 5.0
- Resolution: 1024x1024

**Runtime Estimate:** ~10-15 minutes per full run

#### Step 7: Implement Memory-Efficient Capture
**Owner/Resources:** Developer
**Dependencies:** Step 6
**Estimated Effort:** 1 hour
**Expected Output:** Memory-optimized capture

**Strategy:**
- Convert tensors to CPU immediately after capture
- Compute statistics on CPU
- Do not store full raw tensors (too large: 4x128x128 = 65536 floats per sample)
- Option: Store subset of raw values for debugging (first 1000 values)

#### Step 8: Run Initial Data Collection
**Owner/Resources:** Developer with device access
**Dependencies:** Steps 1-7
**Estimated Effort:** 30 minutes runtime
**Expected Output:** First complete dataset

---

### Phase 3: Comparison Method (Steps 9-12)

#### Step 9: Implement PCC/MSE Comparison
**File:** `unet_analysis_utils.py` (extension)

**Owner/Resources:** Developer
**Dependencies:** Phase 2 data available
**Estimated Effort:** 1 hour
**Expected Output:** Comparison metrics computed

**Metrics:**
- Pearson Correlation Coefficient (PCC) - already used in tests
- Mean Squared Error (MSE)
- Max Absolute Error
- Relative Error Distribution

#### Step 10: Implement Divergence Detection Algorithm
**Owner/Resources:** Developer
**Dependencies:** Step 9
**Estimated Effort:** 1-2 hours
**Expected Output:** Automated divergence point identification

**Algorithm:**
```python
def find_divergence_onset(step_data):
    pcc_threshold = 0.99  # TT UNet single-pass PCC is ~0.997
    for step in sorted(step_data.keys()):
        if step_data[step]['raw_unet_pcc'] < pcc_threshold:
            return step
    return None  # No significant divergence
```

**Analysis Questions:**
1. Does PCC degrade monotonically or suddenly?
2. Is degradation in specific spatial regions?
3. Does degradation correlate with timestep value?

#### Step 11: Implement Spatial Analysis
**Owner/Resources:** Developer
**Dependencies:** Step 9
**Estimated Effort:** 1-2 hours
**Expected Output:** Spatial error maps

**Approach:**
- Reshape tensors to spatial format (B, C, H, W)
- Compute per-pixel absolute difference
- Identify high-error regions (corners? edges? center?)
- Generate heatmaps

#### Step 12: Create Analysis Report Generator
**Owner/Resources:** Developer
**Dependencies:** Steps 9-11
**Estimated Effort:** 1 hour
**Expected Output:** Automated report generation

**Report Contents:**
- PCC progression graph
- Error accumulation graph
- Spatial error heatmaps at key steps
- Summary statistics table

---

### Phase 4: Root Cause Analysis (Steps 13-17)

#### Step 13: Analyze Pre vs Post Guidance
**Owner/Resources:** Developer
**Dependencies:** Phase 3 analysis complete
**Estimated Effort:** 1 hour
**Expected Output:** Determination of whether guidance computation introduces error

**Key Question:** Is `noise_pred` (raw UNet) correct, but `noise_pred_guided` (after guidance formula) incorrect?

**Guidance Formula Analysis:**
```python
# noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
# Check:
# 1. noise_pred_text - noise_pred_uncond precision
# 2. multiplication by guidance_scale precision
# 3. final addition precision
```

#### Step 14: Analyze Scheduler Step
**Owner/Resources:** Developer
**Dependencies:** Step 13
**Estimated Effort:** 1 hour
**Expected Output:** Determination of whether scheduler step introduces error

**Key Files to Analyze:**
- `/home/tt-admin/tt-metal/models/experimental/stable_diffusion_xl_base/tt/tt_euler_discrete_scheduler.py`

**Focus Areas:**
- `scale_model_input()` - division by norm factor
- `step()` - the actual Euler step computation
- Sigma values precision

#### Step 15: UNet Block-Level Instrumentation (Conditional)
**Owner/Resources:** Developer
**Dependencies:** Steps 13-14 indicate UNet is root cause
**Estimated Effort:** 3-4 hours
**Expected Output:** Per-block analysis data

**UNet Architecture (from code analysis):**
```
conv_in ->
  down_blocks[0] (DownBlock2D) ->
  down_blocks[1] (CrossAttnDownBlock2D: 640 dim, 10 heads) ->
  down_blocks[2] (CrossAttnDownBlock2D: 1280 dim, 20 heads) ->
mid_block (CrossAttnMidBlock2D: 1280 dim, 20 heads) ->
  up_blocks[0] (CrossAttnUpBlock2D: 1280 dim, 20 heads) ->
  up_blocks[1] (CrossAttnUpBlock2D: 640 dim, 10 heads) ->
  up_blocks[2] (UpBlock2D) ->
norm_out -> conv_out
```

**Instrumentation Points:**
1. After conv_in
2. After each down_block
3. After mid_block
4. After each up_block
5. After norm_out (before conv_out)
6. Final output

#### Step 16: Attention vs ResNet vs Conv Analysis (Conditional)
**Owner/Resources:** Developer
**Dependencies:** Step 15 identifies specific block
**Estimated Effort:** 2-3 hours
**Expected Output:** Layer-type identification

**Key Files:**
- `/home/tt-admin/tt-metal/models/experimental/stable_diffusion_xl_base/tt/tt_attention.py`
- `/home/tt-admin/tt-metal/models/experimental/stable_diffusion_xl_base/tt/tt_resnetblock2d.py`
- `/home/tt-admin/tt-metal/models/experimental/stable_diffusion_xl_base/tt/tt_transformermodel.py`

**Focus Areas:**
- Attention softmax precision
- Group normalization precision
- Conv2D precision
- SDPA (scaled dot product attention) precision

#### Step 17: Document Findings and Recommendations
**Owner/Resources:** Developer
**Dependencies:** All previous steps
**Estimated Effort:** 1-2 hours
**Expected Output:** Final analysis document

---

## Risk & Mitigation

### Risk 1: Instrumentation Fails Again
**Probability:** Medium
**Impact:** High - blocks entire analysis

**Mitigation:**
- Use file I/O instead of logger (more reliable)
- Add try/catch with fallback to stderr
- Create minimal instrumentation first, add complexity gradually
- Test instrumentation on single step before full run

### Risk 2: UNet is Correct, Issue is Elsewhere
**Probability:** Medium
**Impact:** Medium - redirects investigation

**Mitigation:**
- Phase 3 explicitly checks post-UNet components
- Have backup investigation paths ready:
  - VAE decoder analysis
  - Prompt embedding analysis
  - Final image processing analysis

### Risk 3: Error is Accumulated, Not Single-Point
**Probability:** High
**Impact:** Low - changes fix approach but still actionable

**Mitigation:**
- Track cumulative error vs per-step error
- Analyze error propagation pattern
- May indicate need for higher precision in specific operations

### Risk 4: Data Volume Too Large
**Probability:** Low
**Impact:** Low - solvable

**Mitigation:**
- Compute statistics on-device/on-CPU, don't store raw tensors
- Use streaming writes
- Downsample if needed for storage

### Risk 5: Reference Mismatch
**Probability:** Medium
**Impact:** High - invalidates comparison

**Mitigation:**
- Verify identical configurations explicitly
- Check scheduler state at each step
- Compare final latents before VAE to isolate denoising from decoding

---

## Backup Investigation Paths

### If UNet Output Matches Reference:
1. **Guidance Computation Path:**
   - Check in-place operations in `run_tt_image_gen()`
   - Verify `guidance_rescale` path (currently at 0.0)
   - Check tensor memory configurations

2. **Scheduler Step Path:**
   - Compare sigma values (TT vs PyTorch)
   - Check division/multiplication precision in Euler step
   - Verify timestep handling

3. **VAE Decoder Path:**
   - Run separate VAE analysis
   - Compare TT VAE vs PyTorch VAE outputs
   - Check VAE scaling factor handling

### If Error Source is Attention:
1. Focus on SDPA implementation
2. Check softmax precision
3. Analyze head count effects (10 vs 20 heads)
4. Check memory config effects on precision

### If Error Source is Normalization:
1. Check GroupNorm implementation
2. Analyze epsilon handling
3. Check affine parameter precision

---

## Next Immediate Actions

### Action 1: Create Test Scaffold (Today)
Create the basic test file with parallel PyTorch/TT execution:
```
/home/tt-admin/tt-metal/models/experimental/stable_diffusion_xl_base/tests/pcc/test_unet_deep_dive.py
```

### Action 2: Implement Statistics Utility (Today)
Create the analysis utilities module:
```
/home/tt-admin/tt-metal/models/experimental/stable_diffusion_xl_base/tests/pcc/unet_analysis_utils.py
```

### Action 3: Run First Comparison (Day 2)
Execute the comparison with a single prompt and capture full statistics.

---

## Implementation Approach Summary

### Should We Modify test_common.py?
**Recommendation:** No, create separate analysis files.

**Rationale:**
- `test_common.py` is used by production code
- Modifications could affect existing tests
- Separate files allow aggressive instrumentation without risk
- Can import from `test_common.py` as needed

### Separate Comparison Script?
**Recommendation:** Yes, create dedicated analysis script.

**Location:** `/home/tt-admin/tt-metal/models/experimental/stable_diffusion_xl_base/tests/pcc/test_unet_deep_dive.py`

### Reference Without tt-media-server Source?
**Recommendation:** Use diffusers library directly.

**Approach:**
- Use `DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")`
- Extract UNet and scheduler
- Run identical inputs through both
- Compare outputs tensor-by-tensor

### Data Volume Strategy?
**Recommendation:** Statistics only, no raw tensor storage.

**Implementation:**
- Compute all statistics immediately after tensor generation
- Write JSON summaries
- Store only anomalous raw values (NaN, Inf, extreme outliers)

---

## Estimated Total Effort

| Phase | Hours |
|-------|-------|
| Phase 1: Instrumentation | 6-8 |
| Phase 2: Data Collection | 3-4 |
| Phase 3: Comparison | 4-6 |
| Phase 4: Root Cause | 8-12 (conditional) |
| **Total** | **21-30 hours** |

---

## Key File References

| File | Purpose |
|------|---------|
| `/home/tt-admin/tt-metal/models/experimental/stable_diffusion_xl_base/tests/test_common.py` | Core pipeline functions (existing) |
| `/home/tt-admin/tt-metal/models/experimental/stable_diffusion_xl_base/tt/tt_unet.py` | TT UNet implementation (302 lines) |
| `/home/tt-admin/tt-metal/models/experimental/stable_diffusion_xl_base/tt/tt_sdxl_pipeline.py` | Pipeline wrapper (759 lines) |
| `/home/tt-admin/tt-metal/models/experimental/stable_diffusion_xl_base/tt/tt_euler_discrete_scheduler.py` | Scheduler (320 lines) |
| `/home/tt-admin/tt-metal/models/experimental/stable_diffusion_xl_base/tests/pcc/test_unet_loop.py` | Existing UNet loop test (reference) |
| `/home/tt-admin/tt-metal/models/experimental/stable_diffusion_xl_base/tests/pcc/test_module_tt_unet.py` | Existing UNet module test (reference) |
| `/home/tt-admin/tt-metal/models/experimental/stable_diffusion_xl_base/reference/test_torch_sdxl_base.py` | PyTorch reference test |
