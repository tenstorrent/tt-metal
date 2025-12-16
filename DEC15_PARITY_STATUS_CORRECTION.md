# Correction: ControlNet/LoRA/IP-Adapter Support Clarification

**Date:** December 15, 2025
**Corrects:** DEC15_PARITY_STATUS.md (Section 4: Known Limitations)
**Issue:** Misleading terminology about extension feature requirements

---

## Executive Summary

**The Error:** DEC15_PARITY_STATUS.md incorrectly stated that ControlNet, IP-Adapter, and LoRA "require native integration." This is misleading.

**The Truth:**
- ControlNet/IP-Adapter: **CAN work with bridge** via API extension (per-timestep calling)
- LoRA: **Better suited for native integration** (requires weight patching)

**What v2.0 Actually Achieved:**
- ✅ Architectural foundation for extensions (solved loop ownership conflict)
- ❌ Not implemented: Per-timestep API and extension integration code

---

## The Confusion Explained

### Original Statement (MISLEADING)

From DEC15_PARITY_STATUS.md, Section 4 "Known Limitations":

> #### 3. Missing Features
>
> **Not yet implemented:**
> - ❌ LoRA (Low-Rank Adaptation)
> - ❌ ControlNet
> - ❌ IP-Adapter
> - ❌ Custom samplers beyond Euler
>
> **Reason:** Phase 2 focused on core pipeline parity first
>
> **Future work:** These require native integration (Option A gradual evolution)

### Corrected Statement

**Not yet implemented:**
- ❌ ControlNet: Architecturally possible with bridge, needs per-timestep API (2-4 weeks)
- ❌ IP-Adapter: Architecturally possible with bridge, needs per-timestep API (2-4 weeks)
- ❌ LoRA: Better suited for native integration due to weight patching requirements (12-17 weeks via Option A)
- ❌ Custom samplers: Possible with per-timestep API extension

**Reason:** Phase 2 focused on architectural foundation (latent output), not feature implementation

**Future work:** Two paths available:
1. **Bridge Extension** (2-4 weeks): Add per-timestep API → enables ControlNet/IP-Adapter
2. **Native Integration** (12-17 weeks): Full ecosystem support → enables everything including LoRA

---

## What v2.0 Bridge Actually Achieved

### Architectural Foundation (COMPLETE ✅)

**Problem Solved:** v1.0 loop ownership conflict

```
v1.0 Architecture (BLOCKED extensions):
┌──────────────────────────────┐
│ Bridge owns complete loop    │
│ for step in range(20):       │
│     latents = denoise(...)   │
│ return final_image           │ ← ComfyUI gets ONLY final image
└──────────────────────────────┘
Result: No intermediate access = ControlNet IMPOSSIBLE

v2.0 Architecture (ENABLES extensions):
┌──────────────────────────────┐
│ Bridge returns latents       │
│ for step in range(20):       │
│     latents = denoise(...)   │
│ return latents               │ ← ComfyUI gets latents
└──────────────────────────────┘
Result: Intermediate access = ControlNet POSSIBLE
```

**Key Achievement:** The v2.0 `denoise_only` operation returns latents (not images), which is the architectural prerequisite for external extensions.

---

### What's NOT Implemented (INCOMPLETE ❌)

**Per-timestep calling API:**

```python
# Current v2.0 implementation
def handle_denoise_only(params):
    """
    Runs FULL denoising loop internally (e.g., 20 steps).
    Returns final latents after all steps complete.
    """
    for step in range(num_steps):
        latents = self.denoise_step(latents, ...)
    return latents  # Final latents after 20 steps

# What's NEEDED for ControlNet/IP-Adapter
def handle_denoise_step_single(params):
    """
    Runs SINGLE denoising step.
    ComfyUI calls this 20 times, injecting ControlNet guidance each time.

    NOT YET IMPLEMENTED.
    """
    latent = params["latent"]
    timestep = params["timestep"]
    conditioning = params["conditioning"]
    control_hint = params.get("control_hint")  # From ControlNet

    # Single UNet forward pass
    output = self.unet.forward(latent, timestep, conditioning, control_hint)
    return output  # After 1 step
```

**Critical Distinction:**
- v2.0 enables **latent manipulation** between full passes (img2img works)
- v2.0 does NOT enable **per-timestep injection** (ControlNet doesn't work)

---

## Three Implementation Levels

### Level 1: v1.0 Bridge (BLOCKS Extensions)

```
ComfyUI: "Generate image"
    ↓
Bridge: Runs 20 steps → returns IMAGE
    ↓
ComfyUI: Gets final image only
```

**Status:** Deprecated
**ControlNet:** IMPOSSIBLE (no access to intermediate states)

---

### Level 2: v2.0 Bridge (ENABLES Extensions, Not Implemented)

```
ComfyUI: "Run denoising"
    ↓
Bridge: Runs 20 steps → returns LATENTS
    ↓
ComfyUI: Can manipulate latents, send to VAE
```

**Status:** ✅ CURRENT (Dec 15, 2025)
**ControlNet:** ARCHITECTURAL FOUNDATION EXISTS
- Can get latents out ✅
- Cannot inject guidance per-step ❌ (needs API extension)

---

### Level 3: Per-Timestep API (FULLY ENABLES Extensions)

```
ComfyUI: For each of 20 steps:
    1. "Run step N with ControlNet guidance"
        ↓
    Bridge: Runs 1 step with control_hint → returns latents
        ↓
    ComfyUI: 2. Apply ControlNet conditioning
    ComfyUI: 3. Send latents + conditioning for next step
```

**Status:** NOT IMPLEMENTED
**ControlNet:** FULLY FUNCTIONAL
- Per-step calling ✅
- ControlNet injection per-step ✅
- IP-Adapter injection per-step ✅

---

## Feature Support Matrix (CORRECTED)

| Feature | v1.0 Bridge | v2.0 Bridge (Current) | Bridge + Per-Step API | Native Integration |
|---------|-------------|----------------------|----------------------|-------------------|
| txt2img | ✅ | ✅ | ✅ | ✅ |
| img2img | ❌ | ✅ | ✅ | ✅ |
| Latent manipulation | ❌ | ✅ | ✅ | ✅ |
| **ControlNet** | ❌ | ❌ | ✅ **POSSIBLE** | ✅ |
| **IP-Adapter** | ❌ | ❌ | ✅ **POSSIBLE** | ✅ |
| **LoRA** | ❌ | ❌ | ⚠️ Difficult | ✅ **BEST PATH** |
| Custom samplers | ❌ | ❌ | ✅ **POSSIBLE** | ✅ |
| Full ecosystem | ❌ | ❌ | ⚠️ Partial | ✅ |

---

## Two Paths Forward (CORRECTED)

### Path A: Bridge Extension (2-4 weeks)

**What to implement:**
1. Add `handle_denoise_step_single` operation to bridge server
2. Implement per-timestep calling from ComfyUI side
3. Add ControlNet conditioning hooks
4. Test with ComfyUI's ControlNet nodes

**Enables:**
- ✅ ControlNet
- ✅ IP-Adapter
- ✅ Custom samplers
- ⚠️ LoRA (difficult, needs weight patching workaround)

**Timeline:** 2-4 weeks

**Code location to modify:**
- `/home/tt-admin/tt-metal/comfyui_bridge/handlers.py` (add new operation)
- `/home/tt-admin/ComfyUI-tt_standalone/custom_nodes/tenstorrent_nodes/nodes.py` (add per-step nodes)

**Advantages:**
- Fast implementation
- Reuses existing bridge infrastructure
- ControlNet/IP-Adapter fully functional

**Disadvantages:**
- LoRA still difficult (weight patching at IPC boundary)
- Still two-process architecture
- Not full ComfyUI ecosystem integration

---

### Path B: Native Integration (12-17 weeks)

**What to implement:**
Follow Option A (gradual evolution) from BRIDGE_TO_INTEGRATION_ANALYSIS.md:
- Phase 1: Extract reusable core (2-3 weeks)
- Phase 2: Native model loading (3-4 weeks)
- Phase 3: Native sampling (4-6 weeks)
- Phase 4: Native VAE (2-3 weeks)
- Phase 5: Deprecate bridge (1 week)

**Enables:**
- ✅ ControlNet
- ✅ IP-Adapter
- ✅ LoRA (full weight patching support)
- ✅ Custom samplers
- ✅ Full ComfyUI ecosystem

**Timeline:** 12-17 weeks

**Advantages:**
- Full ecosystem support
- LoRA works properly
- Single process (no IPC overhead)
- Better long-term maintainability

**Disadvantages:**
- Longer timeline
- More complex implementation
- Higher risk

---

## Why LoRA is Different

### ControlNet/IP-Adapter: Conditioning Injection

```python
# ControlNet injects additional conditioning at each timestep
def apply_controlnet(latent, timestep, base_conditioning):
    control_features = controlnet_model(latent, timestep)
    enhanced_conditioning = base_conditioning + control_features
    return unet_forward(latent, timestep, enhanced_conditioning)
```

**Bridge compatibility:** ✅ CAN WORK
- ControlNet runs on ComfyUI side (CPU/GPU)
- Conditioning passed to bridge per-step
- Bridge runs UNet with enhanced conditioning

---

### LoRA: Weight Patching

```python
# LoRA modifies model weights at load time
def load_model_with_lora(base_model, lora_weights):
    for layer in base_model.layers:
        # Apply LoRA deltas to layer weights
        layer.weight = layer.weight + lora_weights[layer.name]
    return modified_model
```

**Bridge compatibility:** ⚠️ DIFFICULT
- LoRA needs to modify model weights before they're loaded to TT device
- Bridge loads model internally, ComfyUI doesn't control this
- Would require:
  1. Serialize LoRA weights via IPC (slow)
  2. Apply patches on bridge side (complex)
  3. Reload model with patches (expensive)

**Native integration:** ✅ NATURAL FIT
- ComfyUI controls model loading
- Apply LoRA patches before TT conversion
- Standard ComfyUI LoRA workflow

---

## Terminology Clarification

### "Bridge Extension" vs "Native Integration"

**Bridge Extension:**
- Add features to existing bridge server
- Still two processes (ComfyUI + Bridge)
- Unix socket + shared memory communication
- **Use for:** Features that work at inference time (ControlNet, IP-Adapter)

**Native Integration:**
- Move TT-Metal into ComfyUI process
- Single process, direct API calls
- No IPC overhead
- **Use for:** Features that need model-level control (LoRA, full ecosystem)

### Original Error in DEC15_PARITY_STATUS.md

The document said:

> **Future work:** These require native integration (Option A gradual evolution)

**Should have said:**

> **Future work:** Two options available:
> 1. Bridge extension (2-4 weeks) for ControlNet/IP-Adapter
> 2. Native integration (12-17 weeks) for full ecosystem including LoRA

---

## Implementation Recommendations

### If You Want ControlNet/IP-Adapter Soon

**Choose:** Bridge Extension (Path A)

**Steps:**
1. Read OPTION_2_5A_IMPLEMENTATION_PROMPT.md for detailed plan
2. Implement `handle_denoise_step_single` in handlers.py
3. Create `TT_KSampler` node in ComfyUI that calls per-step
4. Test with ControlNet nodes (may need wrapper)

**Timeline:** 2-4 weeks
**Outcome:** ControlNet ✅, IP-Adapter ✅, LoRA ❌

---

### If You Want Full Ecosystem (Including LoRA)

**Choose:** Native Integration (Path B)

**Steps:**
1. Read BRIDGE_TO_INTEGRATION_ANALYSIS.md for detailed plan
2. Follow Option A (gradual evolution) 5-phase approach
3. Keep bridge as reference implementation during transition
4. Test each phase against bridge for validation

**Timeline:** 12-17 weeks
**Outcome:** Everything ✅ (ControlNet, IP-Adapter, LoRA, full ecosystem)

---

### If You're Unsure

**Recommendation:** Start with Bridge Extension

**Reasoning:**
1. Quick validation (2-4 weeks vs 12-17 weeks)
2. Proves ControlNet/IP-Adapter demand before major investment
3. Bridge extension work is NOT wasted:
   - Validates per-timestep patterns
   - Tests ControlNet integration approach
   - Can inform native integration design
4. Can always do native integration later

**Sequential approach:**
- Week 1-4: Bridge extension (ControlNet/IP-Adapter working)
- Week 5-6: Evaluate usage and demand
- Week 7+: Start native integration if needed (for LoRA + full ecosystem)

---

## Code Examples

### Current v2.0 Implementation

**Location:** `/home/tt-admin/tt-metal/comfyui_bridge/handlers.py:574-602`

```python
def handle_denoise_only(self, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Current implementation: Runs FULL denoising loop.
    Returns latents after all steps complete.
    """
    # CLIP encoding
    embeddings = self.encode_clip(params["prompt"], ...)

    # Initialize latents
    if "latent_image_shm" in params:
        tt_latents = self._get_latents_from_shm_and_convert(...)
    else:
        tt_latents = self.generate_random_latents(...)

    # FULL DENOISING LOOP (all 20 steps)
    for step_index in range(num_inference_steps):
        tt_latents = self.sdxl_runner.tt_sdxl.denoise_step(
            tt_latents, embeddings, step_index, ...
        )

    # Convert and return final latents
    latents_torch = ttnn.to_torch(tt_latents, ...)
    latents_torch = _detect_and_convert_tt_to_standard_format(latents_torch)

    return {"latents_shm": self.tensor_bridge.tensor_to_shm(latents_torch)}
```

**What this enables:**
- ✅ ComfyUI can call multiple times (img2img pipeline)
- ✅ ComfyUI can manipulate latents between calls
- ❌ ComfyUI cannot inject guidance per-timestep (ControlNet blocked)

---

### What's Needed for ControlNet (NOT YET IMPLEMENTED)

**New operation to add:**

```python
def handle_denoise_step_single(self, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    NEW OPERATION: Runs SINGLE denoising step.
    ComfyUI calls this 20 times for 20 steps.

    NOT YET IMPLEMENTED - This is what needs to be added.
    """
    # Get inputs for this single step
    latent = self._get_latents_from_shm(params["latent_shm"])
    timestep = params["timestep"]
    timestep_index = params["timestep_index"]
    conditioning = self._get_conditioning_from_shm(params["conditioning_shm"])

    # Optional: ControlNet conditioning hint
    control_hint = None
    if "control_hint_shm" in params:
        control_hint = self._get_tensor_from_shm(params["control_hint_shm"])

    # Convert to TT format
    tt_latent = convert_to_tt_format(latent)

    # SINGLE UNet forward pass
    tt_output = self.sdxl_runner.tt_sdxl.unet.forward(
        tt_latent,
        timestep,
        conditioning,
        control_hint=control_hint  # Inject ControlNet guidance here
    )

    # Convert back to standard format
    output_torch = ttnn.to_torch(tt_output, ...)
    output_torch = _detect_and_convert_tt_to_standard_format(output_torch)

    return {"latent_shm": self.tensor_bridge.tensor_to_shm(output_torch)}
```

**What this would enable:**
- ✅ ComfyUI controls loop (calls 20 times)
- ✅ ControlNet injects guidance each step
- ✅ IP-Adapter modifies conditioning each step
- ✅ Custom samplers control schedule

---

### ComfyUI Side Integration Example

**New node to add:** `/home/tt-admin/ComfyUI-tt_standalone/custom_nodes/tenstorrent_nodes/nodes.py`

```python
class TT_KSampler:
    """
    NEW NODE: Per-step sampler with ControlNet support.

    NOT YET IMPLEMENTED - This is what needs to be added.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "latent_image": ("LATENT",),
                "conditioning": ("CONDITIONING",),
                "steps": ("INT", {"default": 20}),
            },
            "optional": {
                "control_hint": ("IMAGE",),  # From ControlNet preprocessor
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "Tenstorrent/sampling"

    def sample(self, model, latent_image, conditioning, steps, control_hint=None):
        """
        Per-step sampling with ControlNet support.
        """
        backend = model.backend
        latents = latent_image["samples"]

        # Per-timestep loop (ComfyUI controls this)
        for step in range(steps):
            # Calculate timestep for this step
            timestep = self.get_timestep_for_step(step, steps)

            # Call bridge for SINGLE step
            response = backend._send_receive("denoise_step_single", {
                "model_id": model.model_id,
                "latent_shm": backend.tensor_bridge.tensor_to_shm(latents),
                "timestep": timestep,
                "timestep_index": step,
                "conditioning_shm": backend.tensor_bridge.tensor_to_shm(conditioning),
                "control_hint_shm": backend.tensor_bridge.tensor_to_shm(control_hint) if control_hint else None,
            })

            # Get result for next step
            latents = backend.tensor_bridge.tensor_from_shm(response["latent_shm"])

            # ControlNet node can modify latents here if chained

        return ({"samples": latents},)
```

**This would enable standard ComfyUI workflows:**
```
LoadImage → ControlNetPreprocessor → TT_KSampler (with control_hint) → TT_VAEDecode
```

---

## Summary of Corrections

### Incorrect Statements in DEC15_PARITY_STATUS.md

**Section 4, Line 345:**
> **Reason:** Phase 2 focused on core pipeline parity first
> **Future work:** These require native integration (Option A gradual evolution)

**Should read:**
> **Reason:** Phase 2 focused on architectural foundation (latent output), not feature implementation
> **Future work:** Two paths available:
> 1. Bridge extension (2-4 weeks): Per-timestep API → ControlNet/IP-Adapter
> 2. Native integration (12-17 weeks): Full ecosystem → everything including LoRA

---

**Section 4, Lines 338-343:**
> **Missing Features:**
> - ❌ LoRA, ❌ ControlNet, ❌ IP-Adapter

**Should read:**
> **Architecturally Enabled but Not Implemented:**
> - ⏳ ControlNet (needs per-timestep API, 2-4 weeks)
> - ⏳ IP-Adapter (needs per-timestep API, 2-4 weeks)
>
> **Better Suited for Native Integration:**
> - ⚠️ LoRA (needs weight patching, 12-17 weeks via native integration)

---

## File References

**Primary documentation:**
- `/home/tt-admin/tt-metal/DEC15_PARITY_STATUS.md` - Main checkpoint (contains errors corrected here)
- `/home/tt-admin/tt-metal/DEC15_PARITY_STATUS_CORRECTION.md` - **This document**
- `/home/tt-admin/tt-metal/BRIDGE_TO_INTEGRATION_ANALYSIS.md` - Native integration path
- `/home/tt-admin/tt-metal/PICKUP_COMFYUI.md` - v1.0 vs v2.0 history

**Implementation references:**
- `/home/tt-admin/tt-metal/OPTION_2_5A_IMPLEMENTATION_PROMPT.md` - Bridge extension detailed plan
- `/home/tt-admin/tt-metal/ROADMAP.md` - Feature timeline

**Code locations:**
- `/home/tt-admin/tt-metal/comfyui_bridge/handlers.py` - Current v2.0 implementation
- `/home/tt-admin/ComfyUI-tt_standalone/custom_nodes/tenstorrent_nodes/nodes.py` - ComfyUI nodes

---

## For Future LLMs

**If reading DEC15_PARITY_STATUS.md, also read this correction document.**

**Key takeaway:** The v2.0 bridge achieved the architectural foundation for ControlNet/IP-Adapter (solved loop ownership conflict, returns latents). What's missing is the per-timestep calling API implementation. This is a 2-4 week bridge extension task, NOT a 12-17 week native integration requirement.

**Decision point:**
- Want ControlNet/IP-Adapter soon? → Bridge extension (2-4 weeks)
- Want full ecosystem with LoRA? → Native integration (12-17 weeks)
- Unsure? → Start with bridge extension, evaluate, then decide on native integration

---

**End of correction document. Future LLMs: Read both DEC15_PARITY_STATUS.md and this correction together.**
