# SDXL Server - Fixes Applied

## Issue Found

When the server was first run, it failed with:
```
ImportError: cannot import name 'get_model_config' from 'models.experimental.stable_diffusion_xl_base.tt.model_configs'
```

## Root Cause

The initial server implementation used an incorrect API that doesn't exist in tt-metal's actual SDXL code. The server was written based on the API patterns from tt-media-server rather than the actual tt-metal SDXL implementation.

**What was wrong:**
- Used non-existent `get_model_config()` function
- Used incorrect parameter signatures for `create_tt_clip_text_encoders()`
- Tried to use functions that don't work with the high-level pipeline API
- Dependency on external `conftest.py`

## Solution Applied

Updated `sdxl_server.py` to use the actual tt-metal SDXL implementation:

### 1. Corrected Imports
```python
# BEFORE (incorrect):
from models.experimental.stable_diffusion_xl_base.tests.test_common import (
    create_tt_clip_text_encoders,
    warmup_tt_text_encoders,
    batch_encode_prompt_on_device,
    run_tt_image_gen,
)

# AFTER (correct):
from models.experimental.stable_diffusion_xl_base.tt.tt_sdxl_pipeline import (
    TtSDXLPipeline,
    TtSDXLPipelineConfig,
)
```

### 2. Pipeline Initialization
Changed to use the proper `TtSDXLPipeline` class with configuration:

```python
self.tt_pipeline = TtSDXLPipeline(
    ttnn_device=self.device,
    torch_pipeline=self.torch_pipeline,
    pipeline_config=TtSDXLPipelineConfig(
        capture_trace=False,
        vae_on_device=False,
        encoders_on_device=False,
        num_inference_steps=ServerConfig.DEFAULT_NUM_INFERENCE_STEPS,
        guidance_scale=ServerConfig.DEFAULT_GUIDANCE_SCALE,
        is_galaxy=is_galaxy(),
        use_cfg_parallel=False,
    ),
)
```

### 3. Image Generation Process
Updated to follow the actual tt-metal pipeline flow:

1. Encode prompts
2. Generate input tensors
3. Prepare inputs
4. Compile image processing
5. Generate images

```python
# Encode prompts
prompt_embeds_torch, add_text_embeds_torch = self.tt_pipeline.encode_prompts(
    [prompt], [negative_prompt]
)

# Generate input tensors
tt_latents, tt_prompt_embeds, tt_add_text_embeds = self.tt_pipeline.generate_input_tensors(
    prompt_embeds_torch,
    add_text_embeds_torch,
)

# Prepare and compile
self.tt_pipeline.prepare_input_tensors([...])
self.tt_pipeline.compile_image_processing()

# Generate
images = self.tt_pipeline.generate_images()
```

### 4. Removed External Dependencies
- Removed dependency on `conftest.py` which had import chain issues
- Implemented simple `is_galaxy()` detection inline

### 5. Fixed Pydantic Warning
Changed `model_loaded` to `pipeline_loaded` in `HealthResponse` to avoid Pydantic protected namespace warning.

## Files Changed

- **`sdxl_server.py`** - Updated imports, pipeline initialization, and generation process

## Testing

The corrected server now:
- ✅ Imports successfully
- ✅ Has proper function signatures
- ✅ Uses the actual tt-metal SDXL API
- ✅ No external dependency issues
- ✅ Ready to test with actual device

## Next Steps

To test the corrected server:

```bash
cd /home/tt-admin/tt-metal

# Run diagnostics
./check_sdxl_readiness.sh

# Start the server
./start_sdxl_server.sh
```

The server will now properly initialize the TtSDXLPipeline and be ready for image generation requests.

## Architecture Notes

The TtSDXLPipeline is designed for:
- Multi-batch inference
- Device-specific optimizations
- Tensor compilation and tracing
- Complex multi-stage denoising

This is more complex than the tt-media-server wrapper approach but provides direct access to tt-metal's optimized SDXL implementation.
