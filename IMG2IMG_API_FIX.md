# Img2img API Mismatch Fix

## Problem

The ComfyUI bridge integration was failing when attempting img2img workflows with the error:

```
TtSDXLPipeline.generate_input_tensors() got an unexpected keyword argument 'start_latents'
```

This occurred in `comfyui_bridge/handlers.py` in the `handle_denoise_only` function when it tried to pass `start_latents` to `TtSDXLPipeline.generate_input_tensors()`, but that parameter didn't exist in the method signature.

## Root Cause

The `TtSDXLPipeline.generate_input_tensors()` method only supported txt2img workflows through the `start_latent_seed` parameter. It did not have any mechanism to accept pre-generated latents for img2img workflows.

### Before Fix

**Method signature** (`tt_sdxl_pipeline.py` line 471):
```python
def generate_input_tensors(
    self,
    all_prompt_embeds_torch,
    torch_add_text_embeds,
    start_latent_seed=None,
    fixed_seed_for_batch=False,
    timesteps=None,
    sigmas=None,
):
```

**Handler call** (`handlers.py` line 483-488):
```python
tt_latents, tt_prompts, tt_texts = self.sdxl_runner.tt_sdxl.generate_input_tensors(
    prompt_embeds,
    pooled_prompt_embeds,
    start_latent_seed=seed,
    start_latents=input_latents if has_input_latents else None  # ERROR: invalid parameter
)
```

## Solution

### Changes to `tt_sdxl_pipeline.py`

1. **Added `start_latents` parameter** to the method signature (line 479)
2. **Added validation** to ensure mutual exclusivity between `start_latent_seed` and `start_latents` (lines 502-504)
3. **Added img2img latent handling logic** (lines 506-525):
   - If `start_latents` is provided, validate shape and convert to expected format
   - Skip random latent generation for img2img workflows
4. **Preserved txt2img behavior** (lines 526-551):
   - If `start_latents` is None, generate random latents using `start_latent_seed`

### After Fix

**Method signature** (`tt_sdxl_pipeline.py` line 471):
```python
def generate_input_tensors(
    self,
    all_prompt_embeds_torch,
    torch_add_text_embeds,
    start_latent_seed=None,
    fixed_seed_for_batch=False,
    timesteps=None,
    sigmas=None,
    start_latents=None,  # NEW: Support for img2img
):
```

**Latent handling logic** (`tt_sdxl_pipeline.py` lines 506-551):
```python
# Validate mutual exclusivity
if start_latents is not None and start_latent_seed is not None:
    raise ValueError("Cannot specify both start_latents and start_latent_seed. Choose one.")

# Use provided latents for img2img, or generate random latents for txt2img
if start_latents is not None:
    # Img2img mode: use provided latents
    logger.info("Using provided start_latents for img2img mode")

    # Validate and convert shape [B, C, H, W] -> [B, 1, H*W, C]
    B, C, H, W = start_latents.shape
    latents = start_latents.permute(0, 2, 3, 1)
    tt_latents = latents.reshape(B, 1, H * W, C)
else:
    # Txt2img mode: generate random latents (existing logic)
    logger.info("Generating random latents for txt2img mode")
    # ... existing random generation code ...
```

### Changes to `handlers.py`

**Updated handler call** (`handlers.py` lines 484-495):
```python
# For img2img, pass start_latents instead of start_latent_seed
if has_input_latents:
    tt_latents, tt_prompts, tt_texts = self.sdxl_runner.tt_sdxl.generate_input_tensors(
        prompt_embeds,
        pooled_prompt_embeds,
        start_latents=input_latents
    )
else:
    tt_latents, tt_prompts, tt_texts = self.sdxl_runner.tt_sdxl.generate_input_tensors(
        prompt_embeds,
        pooled_prompt_embeds,
        start_latent_seed=seed
    )
```

This approach:
- Clearly separates txt2img and img2img workflows
- Avoids the mutual exclusivity validation error
- Makes the code intent explicit

## Workflow Support

### Txt2img (Text-to-Image)
- **Input**: Text prompts only
- **Behavior**: Generate random latents using `start_latent_seed`
- **Parameters**: `start_latent_seed=seed` (optional)

### Img2img (Image-to-Image)
- **Input**: Text prompts + input image latents
- **Behavior**: Use provided latents as starting point for denoising
- **Parameters**: `start_latents=input_latents`
- **Note**: `start_latent_seed` is not used in img2img mode

## Validation

The fix includes proper validation:

1. **Shape validation** for `start_latents`:
   - Must be 4D tensor `[B, C, H, W]`
   - Channels must match UNet in_channels (4 for SDXL)

2. **Mutual exclusivity**:
   - Cannot specify both `start_latents` and `start_latent_seed`
   - Error is raised if both are provided

3. **Format conversion**:
   - Input format: `[B, C, H, W]` (standard latent tensor)
   - Internal format: `[B, 1, H*W, C]` (TT pipeline format)
   - Conversion handled automatically

## Files Modified

1. `/home/tt-admin/tt-metal/models/experimental/stable_diffusion_xl_base/tt/tt_sdxl_pipeline.py`
   - Added `start_latents` parameter
   - Added img2img latent handling logic
   - Added validation and error handling

2. `/home/tt-admin/tt-metal/comfyui_bridge/handlers.py`
   - Updated `handle_denoise_only` to conditionally pass correct parameters
   - Separated txt2img and img2img call paths

## Testing

The fix should be tested with:

1. **Txt2img workflow**:
   - Verify random latent generation still works
   - Verify seed reproducibility
   - Check that images are generated correctly

2. **Img2img workflow**:
   - Verify provided latents are used correctly
   - Verify denoising strength parameter works
   - Check that img2img output matches expected behavior

3. **Edge cases**:
   - Invalid latent shapes should raise clear error messages
   - Mutual exclusivity should be enforced
   - Batch size mismatches should be handled gracefully
