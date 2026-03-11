# Molmo2-8B TTNN Status

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
