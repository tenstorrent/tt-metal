# SDXL Tensor Concatenation Fix - Implementation Summary

## Problem

The SDXL server was failing during warmup with a tensor concatenation error:
```
TT_FATAL @ ttnn/cpp/ttnn/operations/data_movement/concat/concat.cpp:270: shapes_match
All dimensions must be the same size except for the dimension along which the concatenation is taking place.
```

## Root Cause Analysis

After thorough investigation, the issue was found in `tt_unet.py:174-175`:

```python
temb_add = ttnn.reshape(temb_add, (text_embeds.shape[0], -1))  # UNSAFE device tensor shape access
temb_add = ttnn.concat([text_embeds, temb_add], -1)            # Concat fails due to shape mismatch
```

**The Real Problem**: The issue wasn't just the shape access, but how `text_embeds` was being prepared in `test_common.py`:

1. `tt_text_embeds` is created as a sharded tensor with shape `[2, 1280]` (2 prompts: conditional/unconditional)
2. When indexed with `tt_text_embeds[unet_slice]`, sharded tensors don't behave like normal arrays
3. The original code used `ttnn.unsqueeze(tt_text_embeds[unet_slice], dim=0)` which resulted in shape `[1, 2, 1280]` instead of `[1, 1280]`
4. This 3D tensor couldn't be concatenated with the 2D `temb_add` tensor

## Solution Implemented

### Fix 1: Add `batch_size` parameter to UNet forward method
**File**: `models/experimental/stable_diffusion_xl_base/tt/tt_unet.py`

Changed line 166:
```python
# Before:
def forward(self, sample, input_shape, timestep, encoder_hidden_states, time_ids, text_embeds):

# After:
def forward(self, sample, input_shape, timestep, encoder_hidden_states, time_ids, text_embeds, batch_size=1):
```

Changed line 174:
```python
# Before:
temb_add = ttnn.reshape(temb_add, (text_embeds.shape[0], -1))

# After:
temb_add = ttnn.reshape(temb_add, (batch_size, -1))
```

**Rationale**: Avoid accessing `.shape[0]` on device tensors, which can be unreliable. Use explicit parameter instead.

### Fix 2: Pass `batch_size` in all forward calls
**Files Modified**:
- `models/experimental/stable_diffusion_xl_base/tests/test_common.py` (line 486)
- `models/experimental/stable_diffusion_xl_base/tests/pcc/test_module_tt_unet.py` (lines 139, 175)

Added `batch_size=B` parameter to all `tt_unet.forward()` calls.

### Fix 3: Correct text_embeds shape preparation
**File**: `models/experimental/stable_diffusion_xl_base/tests/test_common.py`

Changed lines 528-546:
```python
# Before:
for unet_slice in range(tt_prompt_embeds.shape[0]):
    latent_model_input = tt_latents
    noise_pred, _ = run_tt_iteration(
        tt_unet,
        tt_scheduler,
        latent_model_input,
        input_shape,
        tt_prompt_embeds[unet_slice] if not use_cfg_parallel else tt_prompt_embeds,
        tt_time_ids if use_cfg_parallel else tt_time_ids[unet_slice],
        ttnn.unsqueeze(tt_text_embeds[unet_slice], dim=0) if not use_cfg_parallel else tt_text_embeds,
    )

# After:
for unet_slice in range(tt_prompt_embeds.shape[0]):
    latent_model_input = tt_latents
    # Extract and reshape text_embeds for this slice
    if not use_cfg_parallel:
        text_embeds_slice = tt_text_embeds[unet_slice]
        # Ensure correct shape [1, 1280] by reshaping
        text_embeds_slice = ttnn.reshape(text_embeds_slice, (1, 1280))
    else:
        text_embeds_slice = tt_text_embeds

    noise_pred, _ = run_tt_iteration(
        tt_unet,
        tt_scheduler,
        latent_model_input,
        input_shape,
        tt_prompt_embeds[unet_slice] if not use_cfg_parallel else tt_prompt_embeds,
        tt_time_ids if use_cfg_parallel else tt_time_ids[unet_slice],
        text_embeds_slice,
    )
```

**Rationale**: Explicitly reshape `text_embeds` to `[1, 1280]` instead of using `unsqueeze`, which was creating an incorrect 3D tensor due to sharding behavior.

## Expected Tensor Shapes

After the fix, the tensors should have these shapes:

| Tensor | Shape | Notes |
|--------|-------|-------|
| `text_embeds` (input to UNet) | `[1, 1280]` | Text encoder projection embeddings |
| `temb_add` (after time projection) | `[1, 256]` | Time embedding projection |
| `temb_add` (after concat) | `[1, 1536]` | Combined: 1280 + 256 |

The concatenation on dimension -1 combines:
- Text embeddings: 1280 dimensions (from CLIP text encoder 2)
- Time embeddings: 256 dimensions (from time embedding projection)
- Result: 1536 dimensions total

## Files Modified

1. `/home/tt-admin/tt-metal/models/experimental/stable_diffusion_xl_base/tt/tt_unet.py`
   - Lines 166, 174

2. `/home/tt-admin/tt-metal/models/experimental/stable_diffusion_xl_base/tests/test_common.py`
   - Lines 486, 528-546

3. `/home/tt-admin/tt-metal/models/experimental/stable_diffusion_xl_base/tests/pcc/test_module_tt_unet.py`
   - Lines 139, 175

## Testing

The fix was tested with:
```bash
./launch_sdxl_server.sh --clear-cache --dev
```

This runs the SDXL server in development mode with a single worker and reduced warmup steps, which exercises the code path that was failing.

## Key Learnings

1. **Device tensor shape access is unsafe**: Accessing `.shape` attributes on device tensors (especially sharded ones) can return unexpected values.

2. **Sharded tensor indexing behavior**: When indexing a sharded tensor, the result may not behave like a simple array slice. Explicit reshaping is safer.

3. **Explicit is better than implicit**: Passing `batch_size` as a parameter is clearer and more reliable than trying to infer it from tensor shapes at runtime.

4. **Debug logging is essential**: Adding temporary print statements to show tensor shapes was crucial for diagnosing the actual problem.

## Commit Message Template

```
Fix SDXL tensor concatenation error during warmup

- Add batch_size parameter to TtUNet2DConditionModel.forward() with default value 1
- Replace unsafe device tensor shape access (text_embeds.shape[0]) with explicit batch_size parameter
- Fix text_embeds preparation in test_common.py to use explicit reshape to [1, 1280]
- Update all UNet forward() call sites to pass batch_size=B parameter
- Fixes warmup crash due to incompatible tensor shapes in concat operation

The issue was caused by:
1. Accessing .shape[0] on sharded device tensors, which is unreliable
2. Using unsqueeze on indexed sharded tensors, which created incorrect 3D shapes
3. Shape mismatches between text_embeds and temb_add tensors during concatenation

Files modified:
- models/experimental/stable_diffusion_xl_base/tt/tt_unet.py
- models/experimental/stable_diffusion_xl_base/tests/test_common.py
- models/experimental/stable_diffusion_xl_base/tests/pcc/test_module_tt_unet.py
```

## Next Steps

1. Remove debug print statements from tt_unet.py (lines 174-179)
2. Run full test suite to ensure no regressions
3. Test with multi-device configurations
4. Commit changes with descriptive message
