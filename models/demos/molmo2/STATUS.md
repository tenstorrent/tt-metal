# Molmo2-8B TTNN Status

## Comprehensive Test Results (2026-03-31)

### Summary Matrix

| Test Type | Batch 1 | Batch 32 | Notes |
|-----------|---------|----------|-------|
| **Demo Text** | ✅ WORKING | ⚠️ PARTIAL (~40%) | Batch 32 has RoPE/attention issues |
| **Demo Image** | ✅ WORKING | ⚠️ PARTIAL (1/4) | Batched images show repetition |
| **Demo Video (8 frames)** | ✅ WORKING | - | Must limit frames |
| **Demo Video (all frames)** | ❌ GARBAGE | - | Chunked processing broken |
| **Local Server Text** | ✅ WORKING | ✅ WORKING (97%) | Concurrent text excellent |
| **Local Server Image** | ✅ WORKING | ❌ CRITICAL | Concurrent images corrupt server |
| **Local Server Video** | ❌ seq_len error | - | max_model_len=4096 too small |
| **Docker Server Text** | ✅ WORKING | ✅ WORKING (97%) | Same as local server |
| **Docker Server Image** | ✅ WORKING | ❌ CRITICAL | Same corruption issue |
| **Docker Server Video** | - | - | Not tested (seq_len issue) |

### Docker Image Used (2026-03-31)
```
ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.11.0-f47e93fe7d-ba84dbf0
```
- Built from tt-metal commit: `f47e93fe7d`
- vLLM commit: `ba84dbf0`

### What Is Working

| Feature | Status | Performance | Notes |
|---------|--------|-------------|-------|
| Demo Batch 1 Text | ✅ | 32.68 tok/s | Coherent output |
| Demo Batch 1 Image | ✅ | 35.47 tok/s | Minor repetition |
| Demo Batch 1 Video (8 frames) | ✅ | 35.08 tok/s | Use `--max-video-frames 8` |
| Server Batch 1 Text | ✅ | ~1s latency | Coherent output |
| Server Batch 1 Image | ✅ | ~3s latency | Coherent output |
| Server Batch 32 Text (concurrent) | ✅ | 97% accuracy | 31/32 correct |

### What Is NOT Working

| Feature | Status | Issue |
|---------|--------|-------|
| Demo Batch 32 Text | ⚠️ | ~60% outputs wrong/garbage - RoPE/attention batching bug |
| Demo Batch 4 Image | ⚠️ | 3/4 outputs repetitive/garbled |
| Demo Video (>8 frames) | ❌ | Chunked vision processing produces garbage ("coffee, coffee...") |
| Server Video | ❌ | Sequence length 6974 > max_model_len 4096 |
| Server Concurrent Images | ❌ | **CRITICAL:** Corrupts server state - all subsequent requests fail |

### Critical Issues

1. **Server Concurrent Images (CRITICAL)**
   - 4 concurrent image requests cause complete server state corruption
   - After corruption, even text requests return garbage
   - Requires server restart + device reset to recover

2. **Video Chunked Processing**
   - Videos with >8 frames processed in chunks of 8
   - Cross-frame attention lost between chunks
   - Output degrades to garbage with more frames

3. **Demo Batch 32 Coherence**
   - Many batch slots produce wrong answers or garbage
   - Example: User 4 expects 11, gets "1111111111"
   - Likely RoPE position encoding or attention mask issue

### Recommendations

| Use Case | Recommended Config |
|----------|-------------------|
| Text (production) | Server concurrent (97% accurate) |
| Image (production) | Batch 1 only (demo or server sequential) |
| Video | Demo with `--max-video-frames 8` only |
| **Avoid** | Server concurrent images, demo batch 32, video >8 frames |

---

## What Is Working

### Core Model (Image)
- **Full E2E image inference** on T3K (8-device mesh) — demo runs end-to-end
- **Vision backbone** (ViT + pooling + projector) — all blocks PCC > 0.99
- **Text model** (36-layer Qwen2 transformer) — prefill + decode, PCC > 0.95 E2E
- **KV cache** — allocated per-layer, filled during prefill, read during decode
- **CPU-free decode loop** — `ttnn.plus_one` for position, `ttnn.argmax` on-device
- **Tracing** — prefill trace, vision trace, unified vision+prefill trace, decode trace
- **RoPE** — on-device embedding lookup via `get_rot_mats_decode_traced`

### Measured Performance (T3K)
| Metric | Measured |
|--------|----------|
| Vision (traced) | ~86 ms |
| Prefill TTFT | ~85 ms |
| Decode (traced) | ~28 ms/token (35.6 tok/s) |
| Decode (no trace) | ~181 ms/token (5.5 tok/s) |

### Video Pipeline (new — this session)
- `preprocess_video_molmo2()` — uses `molmo-utils` + `decord` to extract frames, preprocesses each frame to `[n_frames, 3, 378, 378]`
- `get_video_tokens()` — generates correct `<frame_start>...<im_patch>*196...<frame_end>` token string with timestamps
- `run_video_inference()` — full prefill + decode using multi-frame visual input
- `run_video_demo()` — CLI entry point with video-specific perf reporting
- `eval_video.py` — batch eval script for test.jsonl (downloads, runs, reports)
- `--video` / `--max-seq-len` / `--max-video-frames` / `--max-video-fps` CLI args added

---

## What Needs To Be Done for `test.jsonl`

### test.jsonl Structure
- **Location:** `models/demos/molmo2/verification/test.jsonl`
- **105 entries**, 103 unique video URLs (.mp4 on GCS)
- **1 video repeated 3×** with the same question
- Each entry: video URL + multiple-choice question, `max_tokens: 16`
- No ground-truth answer in the file — predictions only (accuracy requires human or reference labels)

### Blockers

#### 1. Dependency conflict — `transformers` / `accelerate` / `numpy`
The `pip install molmo-utils --target python_env/...` during setup polluted `python_env` with:
- `numpy-2.2.6.dist-info` (dist metadata only, mismatching the actual numpy 1.26.4 runtime)
- A newer `accelerate` `other.py` that uses `np._core.multiarray` (not present in numpy 1.26.4)

**Symptom:** `AttributeError: module 'numpy._core' has no attribute 'multiarray'` when loading `transformers.generation.utils` → `accelerate.hooks`

**Fix applied:** Removed `numpy-2.2.6.dist-info` from `python_env`. Verify with:
```bash
python -c "import importlib.metadata; print(importlib.metadata.version('numpy'))"
# Must print 1.26.4
python -c "from transformers import AutoTokenizer; print('OK')"
```

**If the above fails:** Reinstall accelerate cleanly:
```bash
pip uninstall -y accelerate && pip install "accelerate>=0.30.0,<1.0.0"
```

#### 2. Segfault on test startup (intermittent)
`test_generation_pcc.py` crashes with segfault during pytest collection after a device was left in a bad state by a prior run.

**Fix:** Reset device before running tests:
```bash
tt-smi -r
```

#### 3. `max_seq_len` for video
Each video frame uses **196 visual tokens** (`pooled_h=14 × pooled_w=14`). At 8 frames:
- Visual tokens: 8 × 196 = 1,568
- Frame token string overhead (timestamps + special tokens): ~10 tokens/frame = 80
- Question tokens: ~30–60
- **Total input: ~1,700–1,700 tokens** for 8 frames at default settings

Current default `max_seq_len=2048` is sufficient for 8 frames. For more frames:

| Frames | Visual tokens | Total approx | `max_seq_len` needed |
|--------|--------------|--------------|----------------------|
| 8      | 1,568        | ~1,700       | 2048 ✓              |
| 16     | 3,136        | ~3,300       | 4096                |
| 21     | 4,116        | ~4,300       | 8192                |

**Default in `eval_video.py` is `max_seq_len=16384`** — this over-allocates KV cache memory. For 8-frame eval you can pass `--max-seq-len 2048` to save memory.

#### 4. Video download required before eval
The 103 unique .mp4 files must be downloaded from GCS. `eval_video.py` does this automatically but requires network access:
```bash
python models/demos/molmo2/demo/eval_video.py \
    --test-jsonl test.jsonl \
    --cache-dir /tmp/molmo2_video_cache \
    --num-samples 5   # start with 5 to verify
```

#### 5. No reference answers in `test.jsonl`
The file only contains questions, not ground-truth answer letters. `eval_video.py` reports the model's predicted answer letter but cannot compute accuracy without a separate answer key.

To run against the full test set and track accuracy you need either:
- A companion `test_answers.jsonl`, or
- Human review of the output

---

## Quick Start: Run First 5 Videos

```bash
cd /home/ttuser/ssinghal/tt-metal
export ARCH_NAME=wormhole_b0
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
source python_env/bin/activate

# Reset device first
tt-smi -r

# Run 5 videos from test.jsonl
python models/demos/molmo2/demo/eval_video.py \
    --test-jsonl models/demos/molmo2/verification/test.jsonl \
    --cache-dir /tmp/molmo2_video_cache \
    --num-samples 5 \
    --max-seq-len 2048 \
    --max-video-frames 8 \
    --max-video-fps 2.0
```

## Quick Start: Single Video Demo

```bash
python models/demos/molmo2/demo/demo.py \
    --video /path/to/video.mp4 \
    --prompt "<|video|> Describe what happens."
```

---

## Known Limitations

| Limitation | Detail |
|------------|--------|
| TextBlock PCC | Single-layer PCC ~0.98 (target 0.99 — open issue) |
| Decode weights | bfloat16 (bfloat8_b causes numerical overflow) |
| Simplified pooling mean | `forward_ttnn` uses `1/k_pool` divisor instead of counting valid positions |
| Skipped attention mask | `image_pooling` called with `attn_mask=None` for trace compatibility |
| Video multi-crop not supported | Each video frame uses simple 378×378 resize only (no high-res crops) |
| No video tracing | Vision trace is per-batch-size; video batch size varies by frame count |
| **Video >8 frames broken** | Chunked vision processing loses cross-frame attention → garbage output |
| **Server concurrent images** | Causes state corruption → server must be restarted |
| **Demo batch 32 text** | ~60% of outputs are wrong/garbage due to RoPE/attention bug |
| **Server video** | max_model_len=4096 too small for video tokens (need 8k-16k) |

---
Last Updated: 2026-03-31
