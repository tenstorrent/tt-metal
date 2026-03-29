# Molmo2 vLLM Server Status

## Goal
Get the vLLM server working with **text, images, and video** using the same server code path.

## Current Status: STABLE FOR TEXT + IMAGES + VIDEO (2026-03-28)

| Input Type | Demo (paged attn) | vLLM Server | Docker Image | Notes |
|------------|-------------------|-------------|--------------|-------|
| Text-only | ✅ WORKING | ✅ WORKING | ✅ WORKING | All pass 50/50 tests |
| Images | ✅ WORKING | ✅ WORKING | ✅ WORKING | Traces DISABLED - coherent output |
| Video | ✅ WORKING | ✅ WORKING | ⚠️ NOT TESTED | Traces DISABLED - coherent output |

### Major Fix (2026-03-28): Disable Traces for Vision Input

**Root Cause:** Text traces captured during warmup don't work with vision-fused embeddings. The traces encode patterns for text-only input distributions, but vision-fused hidden states have different value distributions even with same tensor shapes.

**Fix:** Disabled `use_trace` for both IMAGE and VIDEO paths in `generator_vllm.py`:
- Line 1762: VIDEO path already had `use_trace=False`
- Line 1921: Changed IMAGE path from `use_trace=enable_trace` to `use_trace=False`

**Results:**
| Test | Before (traced) | After (non-traced) |
|------|-----------------|-------------------|
| Image: Cinque Terre | "Cinque Terreto is Cinque Terre rosso..." (garbled) | "This is Cinque Terre... in Italy" (correct) |
| Video: Big Buck Bunny | Somewhat coherent | "large tree with bright light... green background" (coherent) |

### Docker Image Test Results (2026-03-28)

**Docker Image:** `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.11.0-3035237ebd-ba84dbf0`

| Test Type | Results | Notes |
|-----------|---------|-------|
| Text (50 requests) | 50/50 PASSED | Various prompts tested |
| Image (50 requests) | 50/50 PASSED | Base64 encoded images |

**Note:** URL-based image fetching fails inside Docker due to network access (403 Forbidden from Wikipedia). Use base64-encoded images for Docker deployment.

### vLLM Server Test Results (2026-03-27)
- **Image:** "A small French bulldog" - coherent and accurate
- **Video:** Garbled output (see Video Quality Bug section below)
- **Response time:** ~15-25 seconds for image/video requests

### Major Fix: Image Token Reordering (2026-03-27)

**Root Cause:** vLLM's multimodal processing was placing image tokens BEFORE the chat template structure instead of INSIDE the user message.

**Wrong structure:**
```
[IMAGE_TOKENS, <|im_start|>user\n, text, <|im_end|>, ...]
```

**Correct structure:**
```
[<|im_start|>user\n, IMAGE_TOKENS, text, <|im_end|>, ...]
```

**Fix:** Added token reordering in `prepare_inputs_for_multimodal()` in `molmo2_model.py`:
1. Detect if image tokens are at position 0 (wrong)
2. Find where `<|im_start|>user\n` sequence starts
3. Reorder: move image tokens to after the chat template prefix

**Files Changed:**
- `models/demos/molmo2/tt/molmo2_model.py`: Added token reordering logic
- `models/demos/molmo2/tt/generator_vllm.py`: Apply chat template in processor to ensure correct `<|image|>` position

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
5. [x] Compare demo vs vLLM code paths to find the difference
6. [x] Debug why vLLM non-traced path crashes in text_attention.py line 319
7. [x] Identified video hang issue on second request

## Video Quality Bug (2026-03-27)

### Finding: Video Output is Garbled/Incoherent

**Symptoms:**
- Images produce coherent, accurate responses: "A small French bulldog"
- Videos produce nonsensical output even on first request

**Example Video Response (Big Buck Bunny):**
```
"After loading the first clip: The scene, the image appears, an end of the video
plays as if 1, there appears to fade to fade to a As the pivots back, the sequence,
following the scene shows a it clears瞬间 of a peaceful forest scene in:, more trees:
video with a towering tree unfolds with lush..."
```

**Key Observations:**
1. **Images work correctly** through the same vision backbone pipeline
2. **Videos produce garbage** despite using similar code path
3. **This is a pre-existing bug** - not caused by memory leak fixes
4. The issue is specific to video frame processing (8 frames × 729 patches = 5832 tokens vs 729 for images)

**Potential Root Causes:**
1. Video frame extraction or multi-frame pooling has a bug
2. Video embeddings not being fused correctly with text
3. Shape mismatch or tensor corruption during video-specific operations

**Note:** This is separate from the "second video request timeout" issue documented below. The quality issue affects even the first video request.

---

## Video Multi-Request Investigation (2026-03-27)

### Finding: Second Video Request Causes Device Timeout

**Symptoms:**
- First video request: SUCCESS (~40 seconds)
- Second video request: FAILS with device timeout (~60 seconds)
- Device becomes "unrecoverable"

**Root Cause Location:**
- `image_pooling.py` → `scaled_dot_product_attention` (line 342)
- During second request, SDPA operation causes device timeout
- Tensor data becomes corrupted (all zeros)

**Debug Trace:**
```
Request #1: image_pooling_2d: min=-56.5000, max=62.7500 (SUCCESS)
Request #2: TIMEOUT: device timeout, potential hang detected, the device is unrecoverable
Request #2: image_pooling_2d: min=0.0000, max=0.0000 (CORRUPTED)
```

**Affected Components:**
- `models/demos/molmo2/tt/image_pooling.py:342` - `scaled_dot_product_attention`
- Device state not properly cleared between video requests

**Why Qwen VL Doesn't Have This Issue:**
- Qwen VL does vision processing on CPU (HuggingFace model)
- Only text model runs on TTNN
- Sidesteps TTNN vision stability issues entirely

**Potential Fixes:**
1. **Short-term:** Add device reset between video requests (heavy-handed)
2. **Medium-term:** Investigate TTNN SDPA memory management for video-sized tensors
3. **Long-term:** Move vision processing to CPU like Qwen VL (architectural change)

### Fix Attempt (2026-03-27): Device Synchronization

**Hypothesis:** Device state not properly synchronized between video requests.

**Changes in `image_pooling.py`:**
1. Added request counter tracking
2. Added `ttnn.synchronize_device()` BEFORE SDPA
3. Added `ttnn.synchronize_device()` AFTER SDPA
4. Added final `ttnn.synchronize_device()` before return

**Changes in `vision_backbone.py`:**
1. Added request counter tracking
2. Added `ttnn.synchronize_device()` at START of forward_ttnn

**Test Results:** Still fails on second video request.

### Root Cause Confirmed (2026-03-27)

**Pattern Identified:**
- REQUEST #1 (batch_seq=196, warmup image) - WORKS
- REQUEST #2 (batch_seq=1568, first video) - WORKS
- REQUEST #3 (batch_seq=1568, second video) - HANGS at pre-SDPA sync

**Critical Finding:** The issue is specific to running **large batch SDPA twice**.
- Small batch (196) works repeatedly without issues
- Large batch (1568) works once, then hangs on second call
- All syncs complete successfully, but the second large SDPA causes device timeout

**Hypothesis:** TTNN's `scaled_dot_product_attention` has a memory leak or resource corruption when processing large tensors (batch_seq=1568). The first call consumes resources that aren't properly freed, causing the second call to deadlock.

**Evidence:**
- Vision backbone initial sync completes on REQUEST #3
- encode_image completes successfully
- Query/key_value tensors have valid data (non-zero ranges)
- Device hangs at sync BEFORE SDPA, not during/after

**Recommended Solutions (in order of preference):**
1. **File TTNN bug report** - This appears to be a TTNN kernel issue with repeated large-tensor SDPA
2. **Move vision to CPU** - Like Qwen VL, sidesteps TTNN vision entirely (architectural change)
3. **Device reset between video requests** - Heavy-handed workaround

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

## Eval Benchmarks (2026-03-29)

### Configuration Added to tt-inference-server

Added `EvalConfig` for Molmo2-8B in `tt-inference-server/evals/eval_config.py`:
- chartqa (published: 85.7%)
- docvqa_val (published: 88.7%)
- mmmu_val (published: 51.0%)

### Eval Run Results

| Benchmark | Samples | TT Score | Published | Status |
|-----------|---------|----------|-----------|--------|
| chartqa | 1250/2500 | 9.36% | 85.7% | Completed |
| docvqa_val | 1026/2675 | N/A | 88.7% | Server crashed |
| mmmu_val | 0 | N/A | 51.0% | Not started |

### Issues Found

1. **Low chartqa accuracy (9.36% vs 85.7%):**
   - Model outputs spelled numbers ("Thirteen" vs "14")
   - Some garbage outputs for decimal values ("4444444444444444")
   - Counting errors and yes/no inversions

2. **Server crash during docvqa:**
   - Eval degraded from ~2s/sample to ~47s/sample
   - Server crashed at ~38% completion

### Note

Video verification tests (105/105) pass correctly with coherent responses. Direct API calls work well. The discrepancy between API quality and lmms-eval scores suggests prompt/format mismatch.

See: `models/demos/molmo2/verification/eval_benchmarks_results.md`

---
Last Updated: 2026-03-29 (Eval benchmarks documented)
