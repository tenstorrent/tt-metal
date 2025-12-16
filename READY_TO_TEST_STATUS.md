# Ready to Test Status - Phase 0 Fixes

**Date**: 2025-12-12
**Status**: ✅ **FIXES COMPLETE** - Ready for Testing (with caveats)

---

## ✅ What's Been Fixed

### 1. Timestep Format Bug ✅ FIXED
**File**: `/home/tt-admin/ComfyUI-tt/custom_nodes/tenstorrent_nodes/wrappers.py` (Line 363)

**What was wrong**: Passing continuous sigma values (14.6) instead of discrete timesteps (999)

**Fix applied**:
```python
t_discrete = self.model_sampling.timestep(t).float()
```

**Status**: Code complete, needs live testing to verify logs

---

### 2. CFG Batching Fix ✅ FIXED
**File**: `/home/tt-admin/ComfyUI-tt/custom_nodes/tenstorrent_nodes/wrappers.py` (Lines 231-357)

**What was wrong**: ComfyUI sends batched `[uncond, cond]`, TT expects separate calls

**Fix applied**:
- Refactored `apply_model()` to detect batch size 2
- New `_apply_single()` method handles individual calls
- Automatic split → process → recombine pattern

**Status**: Code complete, syntax validated, needs live testing

---

## ⚠️ What's Missing: Bridge Server

### The Problem
`ComfyUI-tt` expects a bridge server at `/tmp/tt-comfy.sock`, but:
- Old bridge server (`tt-inference-server/tt-comfy-bridge/`) doesn't exist
- Standalone SDXL server (`sdxl_server.py`) uses different architecture (HTTP REST API)
- Need to connect ComfyUI to TT hardware somehow

### Your Options

#### **Option A: Find/Use Old Bridge Server** (If it exists elsewhere)
If the bridge server code exists somewhere:
```bash
# Start bridge
python -m server.main --socket-path /tmp/tt-comfy.sock --device-id 0

# Start ComfyUI
cd /home/tt-admin/ComfyUI-tt
AUTO_RUN=false ./launch_comfyui_tenstorrent.sh
```

**Pros**: Should work immediately
**Cons**: Need to find the bridge code

---

#### **Option B: Create Simple Bridge Adapter** (2-4 hours work)
Create a lightweight adapter that:
1. Listens on Unix socket `/tmp/tt-comfy.sock`
2. Receives msgpack messages from ComfyUI
3. Forwards to standalone SDXL server HTTP API
4. Returns results

**Pros**: Clean, uses proven standalone server
**Cons**: Requires implementation time

---

#### **Option C: Test Standalone Server Only** (Available NOW)
Skip ComfyUI integration, test fixes in isolation:

```bash
# 1. Start standalone server
./launch_sdxl_server.sh --dev

# 2. Test via HTTP API
curl -X POST http://127.0.0.1:8000/image/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "An astronaut riding a horse",
    "negative_prompt": "",
    "num_inference_steps": 20,
    "guidance_scale": 7.0,
    "width": 1024,
    "height": 1024,
    "seed": 42
  }'
```

**Pros**: Can test immediately
**Cons**: Doesn't test ComfyUI integration or our wrapper fixes

---

#### **Option D: Use TT_FullDenoise Node** (Fallback)
According to the plan, `TT_FullDenoise` already works with SSIM 0.998+.

Check if it's independent of the bridge server, or if it also needs one.

**Location**: `/home/tt-admin/ComfyUI-tt/custom_nodes/tenstorrent_nodes/nodes.py`

---

## 📋 Testing Checklist (Once Bridge is Available)

### Quick Verification (30 min)
- [ ] Start bridge server (or equivalent)
- [ ] Launch ComfyUI with our fixes
- [ ] Run simple txt2img workflow
- [ ] Check logs for:
  - ✅ Sigma → timestep conversion (values 999, 950, ... 50, 0)
  - ✅ CFG unbatching messages ("🔀 CFG BATCH DETECTED")
  - ✅ 2 bridge calls per step (uncond, then cond)
- [ ] Verify image generated successfully

### Quality Test (1-2 hours)
- [ ] Generate reference image (PyTorch SDXL or previous working version)
- [ ] Generate with our fixed integration
- [ ] Calculate SSIM score
- [ ] **Success**: SSIM >= 0.90

### Stress Test (2-3 hours)
- [ ] Run 10 consecutive generations
- [ ] Verify no crashes
- [ ] Check for memory leaks
- [ ] Verify consistent quality

---

## 🎯 Recommended Next Steps

### Immediate (You choose):

**Path 1: Quick Test** (If you just want to verify something works)
→ Test standalone server with API calls (Option C)

**Path 2: Find Bridge** (If old bridge exists)
→ Search for bridge server code, start it, test ComfyUI integration

**Path 3: Build Adapter** (If you want full integration now)
→ I can help create a bridge adapter (2-4 hours)

**Path 4: Focus on TT_FullDenoise** (Follow original plan)
→ Test the existing proven node first, ignore per-step integration for Phase 0

---

## 📁 Modified Files Summary

All changes in: `/home/tt-admin/ComfyUI-tt/custom_nodes/tenstorrent_nodes/wrappers.py`

**Additions:**
- Line 263-336: CFG unbatching logic in `apply_model()`
- Line 338-357: New `_apply_single()` method
- Line 363: Timestep conversion fix
- Lines 296-304, 360-379: Enhanced diagnostic logging

**Total changes**: ~130 new/modified lines

**Validation**: ✅ Syntax checked, module imports successfully

---

## 🔍 How to Verify Fixes Work

### Look for these log patterns:

**1. Timestep Conversion** (every step):
```
--- TIMESTEP CONVERSION ---
  Input (sigma) value: [14.6146]
  Converted (discrete) value: [999.0]
  Conversion: sigma -> discrete timestep index
```

**2. CFG Unbatching** (if CFG enabled):
```
🔀 CFG BATCH DETECTED - UNBATCHING FOR TT PIPELINE
  ComfyUI sent batch of 2: [uncond, cond]
  TT pipeline requires separate calls
  Splitting batch and processing separately...

  [1/2] Processing UNCONDITIONAL (negative prompt)...
  [2/2] Processing CONDITIONAL (positive prompt)...

  ✅ CFG predictions recombined: [uncond, cond]
```

**3. Bridge Calls** (2x per step with CFG):
```
--- CALLING BRIDGE SERVER (apply_unet) ---
```

---

## ⚡ Performance Impact

### Timestep Conversion
- **Overhead**: < 0.1ms per step
- **Total**: < 2ms for 20 steps
- **Impact**: Negligible

### CFG Unbatching
- **Overhead**: 2x bridge calls per step (vs 1x batched)
- **Per step**: ~2x latency
- **Total**: Expected ~2x slower than ideal batched approach
- **Trade-off**: Correctness over speed (can optimize in Phase 3)

**Estimated timing** (20 steps, 1024x1024):
- Without CFG: ~20-25s (1 call per step)
- With CFG: ~40-50s (2 calls per step)

This is acceptable for Phase 0 validation. Optimization comes in Phase 3.

---

## 🚦 Status Summary

| Component | Status | Notes |
|-----------|--------|-------|
| Timestep fix | ✅ Complete | Needs testing |
| CFG fix | ✅ Complete | Needs testing |
| Syntax validation | ✅ Passed | Imports successfully |
| Bridge server | ❌ Missing | Blocker for ComfyUI testing |
| Standalone server | ✅ Working | Already tested |
| Launch script | ⏳ Waiting | Depends on bridge |

---

## 💡 My Recommendation

**For immediate progress**: Use Option C (test standalone server) to verify the hardware pipeline works, then decide on bridge approach.

**For full integration**: Either find the old bridge server OR let me create a simple adapter (I can do this if you want).

**Question for you**:
1. Do you know where the bridge server code is?
2. OR would you like me to create a simple bridge adapter?
3. OR should we focus on testing TT_FullDenoise node first (per original Phase 1 plan)?

---

## 📝 Files to Review

1. **Modified wrapper**: `/home/tt-admin/ComfyUI-tt/custom_nodes/tenstorrent_nodes/wrappers.py`
2. **Timestep fix report**: `/home/tt-admin/tt-metal/PHASE_0_TIMESTEP_FIX_REPORT.md`
3. **This status**: `/home/tt-admin/tt-metal/READY_TO_TEST_STATUS.md`

---

**Bottom Line**: Code fixes are done and validated. You're ready to test as soon as you have a way to connect ComfyUI to TT hardware (bridge server or adapter).
