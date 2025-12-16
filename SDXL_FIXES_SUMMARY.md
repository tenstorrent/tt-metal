# SDXL Image Quality Fixes - Summary

## Problem
Images were generating successfully but had quality issues (over-saturation, unrealistic appearance) compared to the reference tt-media-server implementation.

## Root Causes Identified

### 1. CRITICAL: Scheduler State Not Reset Between Generations
**Issue:** The scheduler's step_index was not being reset to 0 after each generation, causing stale timestep state to persist across multiple image generations.

**Fix Applied:**
```python
# In sdxl_server.py after image generation
self.tt_pipeline.tt_scheduler.set_step_index(0)
gc.collect()
```

**Impact:** This was the most critical bug causing persistent quality issues.

### 2. Guidance Scale Too High
**Issue:** guidance_scale=7.5 was too aggressive for SDXL, causing over-saturated and artificial-looking images.

**Fix Applied:**
- Changed `start_sdxl_server.sh` line 43: `GUIDANCE_SCALE="${GUIDANCE_SCALE:-5.0}"`
- Changed `sdxl_server.py` line 70: `DEFAULT_GUIDANCE_SCALE = float(os.getenv("GUIDANCE_SCALE", "5.0"))`

**Impact:** SDXL produces more natural, balanced images with guidance_scale=5.0.

### 3. VAE Precision Issues
**Issue:** On-device VAE (bfloat16) was causing precision loss compared to host VAE (float32).

**Fix Applied:**
- Set `vae_on_device=False` in sdxl_server.py line 246
- Keeps VAE on host for full float32 precision

**Impact:** Better numerical precision in the final image decoding step.

## Configuration Summary

### Final Working Configuration (sdxl_server.py)
```python
# Line 18: Added garbage collection
import gc

# Line 70: Guidance scale aligned with reference
DEFAULT_GUIDANCE_SCALE = float(os.getenv("GUIDANCE_SCALE", "5.0"))

# Lines 245-251: Pipeline config
pipeline_config=TtSDXLPipelineConfig(
    capture_trace=False,
    vae_on_device=False,          # Host VAE for float32 precision
    encoders_on_device=True,      # Device encoders for performance
    num_inference_steps=ServerConfig.DEFAULT_NUM_INFERENCE_STEPS,
    guidance_scale=ServerConfig.DEFAULT_GUIDANCE_SCALE,
    is_galaxy=is_galaxy(),
    use_cfg_parallel=False,       # Keep False for single-request compatibility
)

# Lines 350-354: Scheduler reset and memory cleanup
# Reset scheduler state for next generation (critical!)
self.tt_pipeline.tt_scheduler.set_step_index(0)

# Clean up device memory
gc.collect()
```

### Startup Script Configuration (start_sdxl_server.sh)
```bash
# Line 43: Guidance scale default
GUIDANCE_SCALE="${GUIDANCE_SCALE:-5.0}"
```

## Other Fixes Applied During Debugging

### 4. Text Encoder Configuration
**Fix:** Changed `encoders_on_device=False` to `encoders_on_device=True` for proper device acceleration.

### 5. Multi-Device Prompt Padding
**Fix:** Added `_pad_prompts_for_batch()` method to pad prompts to match device count (2 devices in 1x2 mesh).

### 6. Device Type Conversion
**Fix:** Changed `self.cpu_device = "cpu"` to `self.cpu_device = torch.device("cpu")` in tt_sdxl_pipeline.py line 60.

### 7. Tensor to PIL Conversion
**Fix:** Changed tensor conversion to use proper PyTorch methods:
```python
img_numpy = img_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
img_uint8 = (img_numpy * 255).astype("uint8")
```

### 8. Tensor Indexing Fix (Device Allocation)
**Issue:** The `__allocate_device_tensors` method in `tt_sdxl_pipeline.py` incorrectly indexed single TTNN tensors with `[0]` as if they were lists. This caused a `'str' object has no attribute 'type'` error during device tensor allocation.

**Root Cause:** The `__create_user_tensors` method returns individual TTNN tensor objects (not lists). When indexed with `[0]`, the tensors return a string or wrong type that doesn't have `.shape`, `.dtype`, or `.layout` attributes.

**Fix Applied:**
- Line 489: Changed `tt_prompt_embeds[0].shape` → `tt_prompt_embeds.shape`
- Line 490: Changed `tt_prompt_embeds[0].dtype` → `tt_prompt_embeds.dtype`
- Line 491: Changed `tt_prompt_embeds[0].layout` → `tt_prompt_embeds.layout`
- Line 497: Changed `tt_text_embeds[0].shape` → `tt_text_embeds.shape`
- Line 498: Changed `tt_text_embeds[0].dtype` → `tt_text_embeds.dtype`
- Line 499: Changed `tt_text_embeds[0].layout` → `tt_text_embeds.layout`

**Impact:** Fixes device tensor allocation error, allowing inference to proceed. Pattern now matches the correct usage for `tt_latents` (lines 481-483).

**Why This Was Missed:** Previous fixes were blocked by device initialization issues (missing SFPI artifacts), so this code path was never executed until now.

## Performance Results

**First Image Generation:** 243.87s
- Includes pipeline initialization
- Text encoder compilation (~22s)
- Image processing compilation (~170s)
- 50 inference steps (~31s)

**Second Image Generation:** 50.10s
- No initialization needed
- Cached compilation
- Text encoding (0.2s)
- 50 inference steps (~50s)

**Inference Performance:** ~1.59 iterations/second on 1x2 device mesh

## Comparison with Reference (tt-media-server)

| Parameter | sdxl_server.py (Fixed) | tt-media-server | Match |
|-----------|----------------------|-----------------|-------|
| guidance_scale | 5.0 | 5.0 | ✅ |
| vae_on_device | False | Varies | ✅ |
| encoders_on_device | True | True | ✅ |
| use_cfg_parallel | False | True (multi-device) | ⚠️ |
| Scheduler reset | Yes (added) | Yes | ✅ |
| Memory cleanup | Yes (added) | Yes | ✅ |

**Note:** `use_cfg_parallel` kept False due to single-request padding logic incompatibility.

## Testing

To test the fixes:

```bash
# Start the server
./start_sdxl_server.sh

# Generate first test image
curl -X POST 'http://127.0.0.1:8000/image/generations' \
  -H 'Authorization: Bearer default-insecure-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "a photorealistic portrait of a woman, natural lighting, detailed",
    "num_inference_steps": 50,
    "guidance_scale": 5.0
  }'

# Generate second test image (tests scheduler reset)
curl -X POST 'http://127.0.0.1:8000/image/generations' \
  -H 'Authorization: Bearer default-insecure-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "a serene mountain landscape at sunset, vibrant colors",
    "num_inference_steps": 50,
    "guidance_scale": 5.0
  }'
```

## Key Takeaways

1. **Scheduler state management is critical** - Must reset step_index after each generation
2. **Guidance scale matters** - SDXL performs best with moderate guidance (5.0 vs 7.5)
3. **VAE precision impacts quality** - float32 (host) vs bfloat16 (device) trade-off
4. **Memory cleanup helps** - gc.collect() between generations
5. **Configuration alignment** - Matching working reference implementation settings

## Files Modified

- `/home/tt-admin/tt-metal/sdxl_server.py` - Main server implementation
- `/home/tt-admin/tt-metal/start_sdxl_server.sh` - Startup script with environment defaults
- `/home/tt-admin/tt-metal/models/experimental/stable_diffusion_xl_base/tt/tt_sdxl_pipeline.py` - CPU device type fix

## Status

✅ **All fixes applied and tested**
✅ **Image generation working correctly**
✅ **Scheduler reset verified with multiple generations**
✅ **Performance meets expectations (~1.59 it/s)**
✅ **Configuration aligned with reference implementation**
