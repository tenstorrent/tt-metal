# Molmo2 vLLM Server Status

## Goal
Get the vLLM server working with **text, images, and video** using the same server code path.

## Current Status: ALL MODALITIES WORKING (2026-03-27)

**Note:** Simple prompts work well, complex descriptive prompts may produce lower quality output.

| Input Type | Demo (paged attn) | vLLM Server | Notes |
|------------|-------------------|-------------|-------|
| Text-only | WORKING | WORKING | Both use traces |
| Images | WORKING | WORKING | Output is low quality but no crash |
| Video | WORKING | WORKING | ~22s response time, content generated |

### Fixes Applied (2026-03-26)

**1. TT Model Runner Video Support** (`tt_model_runner.py`):
- Extended `_validate_mm_feature` to accept both "image" and "video" modalities
- Added video field arrays to `_gather_multi_modal_inputs`: `pixel_values_videos`, `video_grid_thw`, `video_token_pooling`

**2. Video Path Tensor Conversion** (`generator_vllm.py`):
- Fixed tensor type mismatch in video path: `_run_prefill()` returns ttnn tensor, but `torch.cat()` expects torch tensor
- Added proper conversion: `ttnn.to_torch()` → extract last token → `output_logits.append()`

**3. Robust Video/Image Detection** (`generator_vllm.py`):
- Fixed issue where image requests were incorrectly entering video path
- Problem: `pixel_values_videos = [[None]]` for images, which passes `is not None` check but fails after unwrapping
- Solution: Unwrap nested lists first, then check if actual data exists before entering video path

**4. Chat Template Placeholder Fix** (`generator_vllm.py` - 2026-03-27):
- Fixed `<|image|>` placeholder insertion to go inside user message, not before chat template
- Before: `<|image|><|im_start|>user\n...` - image tokens outside chat structure
- After: `<|im_start|>user\n<|image|>...` - image tokens inside user message
- This matches how demo.py constructs prompts via `apply_chat_template()`
- Result: Simple prompts like "What animal is this?" now return coherent answers ("This animal is a dog")

### vLLM Test Results (2026-03-26)

**Text test:**
```json
{"content": "22"}  // Correct response
```

**Image test:**
```json
{"content": "TheImageShowsAnAnimalAnAnimalIn..."}  // No crash, ~28s
```

**Video test:**
```json
{"content": "The main subject matter involves a scene with a man sitting at a table..."}  // ~22s
```

### Demo Test Results (2026-03-26)

#### Optimal Config: `--paged-attention --use-decode-trace`

| Modality | TTFT | Decode | Output |
|----------|------|--------|--------|
| Text | 66ms | 33.01 tok/s | "The capital of France is Paris." |
| Image | 143.80ms | 32.74 tok/s | "The animal in this image is a dog." |
| Video | 582.39ms | 31.95 tok/s | "In this video, a man is sitting at a table..." |

```bash
# Text - PASSED
python demo.py --prompt "What is capital of France?" --paged-attention --use-decode-trace
# Image - PASSED
python demo.py --image dog.jpg --prompt "<|image|>What animal?" --paged-attention --use-decode-trace
# Video - PASSED
python demo.py --video video.mp4 --prompt "<|video|>What is happening?" --paged-attention --use-decode-trace
```

#### Without Decode Trace: `--paged-attention` (no trace flags)

| Modality | TTFT | Decode | Notes |
|----------|------|--------|-------|
| Video | 576.82ms | 6.18 tok/s | 5x slower decode than with --use-decode-trace |

#### With Prefill Trace: `--paged-attention --use-trace --use-decode-trace`

| Modality | Status | Notes |
|----------|--------|-------|
| Text | PASSED | Works fine |
| Image | PASSED | seq_len=256, fast compile |
| Video | **HANGS** | seq_len=2048, trace capture hangs during prefill warmup |

**Key findings:**
1. Paged attention works for all modalities when prefill traces are disabled
2. Decode trace works with paged attention (30+ tok/s vs 6 tok/s without)
3. Prefill trace capture HANGS for large seq_len (2048+) with paged attention
4. The optimal config is: `--paged-attention --use-decode-trace` (no `--use-trace`)

## Architecture Overview

### Two Code Paths

**1. Standalone Demo (`demo/demo.py`)**
- Works with images + traces
- `use_paged_attention=False` by default
- Captures traces on-the-fly with vision-fused hidden states
- Vision processing fully in TTNN

**2. vLLM Server (`tt/generator_vllm.py`)**
- Text works with traces
- Images/video have `use_trace=False` (line 1660)
- Always uses paged attention (vLLM requirement)
- Vision processing fully in TTNN

### Key Difference: Paged Attention

| Feature | Demo | vLLM |
|---------|------|------|
| Paged Attention | OFF by default | Always ON |
| Prefill Traces | Work with images | Disabled for images |
| KV Cache | Standard fill_cache | paged_fill_cache |

## Root Cause Analysis

### The Problem
Line 1660 in `generator_vllm.py`:
```python
use_trace=False,  # DISABLED: text-only traces incompatible with vision input
```

This forces non-traced execution for images, which crashes or times out.

### Key Finding (2026-03-26)

**Demo with paged attention WORKS for all modalities!** The issue is NOT:
- Paged attention + vision (demo works)
- Non-traced prefill conceptually (demo uses it)

**The issue IS something specific to the vLLM code path that differs from demo.**

### What Demo Does Differently

When `--paged-attention` is enabled:
1. `--use-unified-trace` auto-disabled (incompatible)
2. Falls back to `run_prefill()` line 1660: "Preparing inputs (vision processing)..."
3. Calls `_prepare_text_inputs()` for vision+text fusion
4. Calls `warmup_prefill()` then direct `text_model.forward()`
5. Works correctly!

### What vLLM Does

1. During warmup: captures prefill traces with text-only dummy tensors
2. During inference with images: `use_trace=False` forces non-traced path
3. Calls `_prepare_text_inputs()` - same as demo
4. Calls `text_model.forward()` - same as demo
5. **CRASHES** with memory manager error in `text_attention.py` line 319

### The Real Question

Why does the **same** `_prepare_text_inputs()` + `text_model.forward()` sequence work in demo but crash in vLLM?

Possible differences:
1. KV cache initialization (vLLM paged cache vs demo paged cache)
2. Page table tensor format/shape
3. Warmup sequence differences
4. State management (vLLM has multiple requests, demo has one)

### Trace Capture Flow

**vLLM Warmup (text-only):**
```
warmup_model_prefill()
  → _allocate_prefill_trace_tensors() [dummy tensors]
  → _capture_prefill_trace() [captures with paged_fill_cache ops]
  → stores trace in self.prefill_traces[seq_len]
```

**Runtime with Images:**
```
prefill_forward()
  → _prepare_text_inputs() [vision + text fusion in TTNN]
  → _run_prefill(use_trace=False) [non-traced forward]
  → TIMEOUT or CRASH
```

### Why Demo Works

Demo captures traces **on-the-fly** with actual vision-fused input:
```python
# demo.py lines 1666-1677
if use_trace:
    if seq_len not in self.prefill_traces:
        # Capture trace with ACTUAL vision-fused hidden_states
        self.warmup_prefill(hidden_states_ttnn, trace_tensors, use_trace=True)
        trace_id, trace_output = self._capture_prefill_trace(trace_tensors)
```

## Comparison with Qwen VL

Qwen VL works with images in vLLM because:
- Vision processing on **CPU** (HuggingFace model)
- Only text model runs on TTNN
- Sidesteps the TTNN vision + paged attention + trace issue

Molmo2 does everything in TTNN, which is the desired approach.

## Files Involved

| File | Purpose |
|------|---------|
| `models/demos/molmo2/tt/generator_vllm.py` | vLLM integration, prefill/decode |
| `models/demos/molmo2/demo/demo.py` | Standalone demo (works) |
| `models/demos/molmo2/tt/text_model.py` | Text transformer layers |
| `models/demos/molmo2/tt/text_attention.py` | Attention with paged_fill_cache |
| `models/demos/molmo2/tt/molmo2_model.py` | Full model (vision + text) |

## Potential Solutions

### Option A: Enable Traces for Images (Quick Test)
Change line 1660 to `use_trace=enable_trace` and test if traces work.

**Risk:** May cause device hang (was the original issue).

### Option B: Fix Non-Traced Path
Debug why `use_trace=False` causes crashes in text_attention.py.

**Error:** RuntimeError in ttnn.reshape at line 319.

### Option C: Capture Vision-Aware Traces
Modify warmup to capture traces with vision-fused input, not text-only.

**Complexity:** Medium - need to run vision during warmup.

### Option D: Hybrid Approach
Use traces for text model only, run vision without traces.

**Status:** This is essentially what's attempted now, but fails.

## Test Commands

**Start Server:**
```bash
cd /home/ttuser/ssinghal/PR-fix/tt-metal/tt-inference-server
python run.py --model Molmo2-8B --workflow server --tt-device t3k --local-server
```

**Test Text:**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "allenai/Molmo2-8B", "messages": [{"role": "user", "content": "What is 2+2?"}]}'
```

**Test Image:**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "allenai/Molmo2-8B",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg"}},
        {"type": "text", "text": "Describe this image briefly."}
      ]
    }]
  }'
```

## Next Steps

1. [x] Test demo with `--paged-attention` + images - **CONFIRMED WORKING**
2. [x] Test demo with `--paged-attention` + video - **CONFIRMED WORKING**
3. [x] Test demo with `--paged-attention --use-decode-trace` - **OPTIMAL CONFIG FOUND**
4. [x] Test demo with `--paged-attention --use-trace --use-decode-trace` - **HANGS for video (large seq_len)**
5. [ ] Compare demo vs vLLM code paths to find the difference
6. [ ] Debug why vLLM non-traced path crashes in text_attention.py line 319
7. [ ] Fix the vLLM path to match demo behavior

## Recommended vLLM Fix

Based on demo testing, vLLM should:
1. **Disable prefill traces for vision** (already done - line 1660)
2. **Keep decode traces enabled** (verify this is happening)
3. **Add compile pass for non-traced prefill** (IMPLEMENTED - line 1199-1226)

### Fix Applied (2026-03-26) - v2

**Root Cause:** vLLM's `warmup_model_prefill()` skipped vision warmup when traces disabled.

**Fix v2 (Improved):** Added vision compile warmup in `warmup_model_prefill()`:

1. **Vision Warmup During Initialization** (`_warmup_vision_compile()`):
   - Creates dummy image (378x378 RGB)
   - Creates dummy pooling indices for vision tokens
   - Runs vision + prefill path to compile all TTNN ops
   - Runs ALWAYS, regardless of trace setting
   - Marks bucket as compiled in `_prefill_compiled_buckets`

2. **Fallback Compile in `_run_prefill()`**:
   - For bucket sizes not covered by warmup
   - Runs compile pass, then **re-prepares ALL inputs** (hidden_states + rot_mats)
   - Ensures correct values for actual forward

```python
# In warmup_model_prefill():
self._warmup_vision_compile(kv_cache, num_blocks)  # Always runs

# In _run_prefill() fallback:
if padded_seq_len not in self._prefill_compiled_buckets:
    # Compile pass
    self.model.text_model.forward(...)
    # CRITICAL: Re-prepare ALL inputs after compile
    hidden_states_ttnn = self._prepare_text_inputs(...)
    rot_mats = ...
```

This mirrors tt_transformers' pattern of vision warmup during initialization.

## Immediate Investigation

Compare these code paths side-by-side:

**Demo (works):** `demo.py:run_prefill()` → `_prepare_text_inputs()` → `text_model.forward()`

**vLLM (crashes):** `generator_vllm.py:_run_prefill()` → `_prepare_text_inputs()` → `text_model.forward()`

The functions look similar but something is different in:
- KV cache setup
- Page table handling
- Tensor shapes/memory configs

## Error Log Reference

**With use_trace=True for images:**
```
TT_THROW: Device 0: Timeout (10000 ms) waiting for physical cores to finish
```

**With use_trace=False for images:**
```
RuntimeError: TT_THROW @ system_memory_manager.cpp:561
(in ttnn.reshape at text_attention.py:319)
```

---
Last Updated: 2026-03-26
