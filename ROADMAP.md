# ComfyUI-tt_standalone Roadmap

**Version:** 1.0.0  
**Date:** 2025-12-12

---

## Current Status

### Version 1.0.0 - Initial Release

**Release Date:** December 2025

**Features:**
- SDXL (Stable Diffusion XL) support
- Full text-to-image inference
- ComfyUI custom nodes integration
- Unix socket bridge architecture
- Zero-copy shared memory transfers
- Thread-safe backend client

---

## Known Limitations

### Model Support

| Model | Status | Notes |
|-------|--------|-------|
| SDXL (stabilityai/stable-diffusion-xl-base-1.0) | Supported | Full inference |
| SD 3.5 | Placeholder | Not implemented |
| SD 1.4/1.5 | Placeholder | Not implemented |
| SDXL Turbo | Not tested | May work with SDXL backend |

### Feature Limitations

1. **Full Inference Only**
   - Currently runs complete pipeline (CLIP + UNet + VAE)
   - Per-step sampling not yet supported
   - Cannot intercept intermediate latents

2. **Single Device Support**
   - T3000 multi-device support requires additional configuration
   - Device selection limited to single device per bridge instance

3. **No Advanced Features**
   - LoRA: Not supported
   - ControlNet: Not supported
   - IP-Adapter: Not supported
   - Inpainting: Not supported
   - Image-to-image: Not supported

4. **Fixed Resolution**
   - Optimal at 1024x1024
   - Other resolutions work but may have quality variations

### Performance Limitations

1. **First Inference Latency**
   - Cold start includes trace compilation
   - First image takes 30-60s
   - Subsequent images much faster

2. **Memory Requirements**
   - Requires ~16GB device memory
   - ~8GB system RAM for bridge

---

## Planned Enhancements

### Phase 1: Core Improvements (Q1 2026)

#### 1.1 Per-Step Sampling Support
- Enable step-by-step denoising
- Return intermediate latents
- Support custom scheduling

**Implementation:**
```python
# New operation
def handle_denoise_step(self, data):
    """Run single denoising step."""
    latent = self.tensor_bridge.tensor_from_shm(data["latent"])
    t = data["timestep"]
    
    denoised = self.sdxl_runner.denoise_step(latent, t)
    
    return {"latent_shm": self.tensor_bridge.tensor_to_shm(denoised)}
```

#### 1.2 Image-to-Image Support
- Accept input images
- Encode to latent space
- Run partial denoising

#### 1.3 Improved Error Handling
- Better error messages
- Recovery from transient failures
- Automatic reconnection

### Phase 2: Multi-Model Support (Q2 2026)

#### 2.1 SD 3.5 Support
- DiT architecture
- Flow matching scheduler
- T5 text encoder

#### 2.2 SD 1.4/1.5 Support
- Legacy UNet architecture
- Single text encoder
- Smaller memory footprint

#### 2.3 SDXL Turbo
- Fewer steps (1-4)
- Modified scheduler
- Lower latency

### Phase 3: Advanced Features (Q3 2026)

#### 3.1 LoRA Support
- Load LoRA weights
- Apply during inference
- Multiple LoRA stacking

**API Design:**
```python
def load_lora(self, model_id: str, lora_path: str, weight: float):
    """Load and apply LoRA weights."""
    pass

def unload_lora(self, model_id: str, lora_name: str):
    """Remove LoRA weights."""
    pass
```

#### 3.2 ControlNet Integration
- Canny edge
- Depth estimation
- OpenPose

#### 3.3 IP-Adapter Support
- Image prompt conditioning
- Style transfer
- Face preservation

### Phase 4: Performance Optimization (Q4 2026)

#### 4.1 Multi-Device Support
- T3000 parallel inference
- Device mesh configuration
- Load balancing

#### 4.2 Batch Processing
- Multiple images per request
- Pipeline parallelism
- Improved throughput

#### 4.3 Caching Improvements
- Model caching
- Latent caching
- Text embedding caching

---

## Community Requests

### High Priority (Frequently Requested)

1. **LoRA support** - Most requested feature
2. **ControlNet** - For guided generation
3. **Lower resolution support** - For faster testing
4. **Batch generation** - Multiple images at once

### Medium Priority

1. **Upscaling** - ESRGAN, Real-ESRGAN
2. **Inpainting** - Selective regeneration
3. **Outpainting** - Extend images
4. **Video generation** - Future models

### Low Priority

1. **Custom model loading** - From .safetensors
2. **Model merging** - Combine weights
3. **Training/fine-tuning** - LoRA training

---

## API Roadmap

### Current API (v1.0)

```python
# Operations
- ping() -> status
- init_model(model_type) -> model_id
- full_denoise(model_id, prompt, ...) -> images
- unload_model(model_id)
```

### Planned API (v2.0)

```python
# New Operations
- list_models() -> [model_info]
- get_model_info(model_id) -> model_details
- encode_prompt(model_id, text) -> embeddings
- denoise_step(model_id, latent, t) -> latent
- decode_latent(model_id, latent) -> image
- encode_image(model_id, image) -> latent

# LoRA Operations
- load_lora(model_id, path, weight)
- unload_lora(model_id, name)
- list_loras(model_id) -> [lora_info]

# ControlNet Operations
- init_controlnet(model_id, type)
- apply_controlnet(model_id, image, strength)
```

---

## Breaking Changes

### Planned for v2.0

1. **Response format changes**
   - Add `api_version` field
   - Standardize error codes
   - Include timing information

2. **Protocol changes**
   - Add message compression option
   - Support streaming responses
   - Add progress callbacks

3. **Configuration changes**
   - New config file format
   - Environment variable changes
   - Default value updates

---

## Contributing

### How to Contribute

1. **Bug Reports**
   - File issues with reproduction steps
   - Include system information
   - Attach relevant logs

2. **Feature Requests**
   - Describe use case
   - Propose API design
   - Consider backward compatibility

3. **Pull Requests**
   - Follow coding standards
   - Include tests
   - Update documentation

### Priority Areas

- [ ] Test coverage improvement
- [ ] Documentation expansion
- [ ] Performance benchmarks
- [ ] Error message improvements

---

## Feedback

For feature requests, bug reports, or feedback:
- File issues on the repository
- Include use case description
- Provide system information

---

**Document Version:** 1.0.0  
**Last Updated:** 2025-12-12  
**Maintainer:** Tenstorrent AI ULC
