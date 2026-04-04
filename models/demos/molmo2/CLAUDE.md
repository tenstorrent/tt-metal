# Molmo2 Development Guide

## Quick Reference

- **Server Status:** See [SERVER_STATUS.md](./SERVER_STATUS.md) for current vLLM server status
- **Eval Results:** See [verification/eval_benchmarks_results.md](./verification/eval_benchmarks_results.md)
- **Video Tests:** See [verification/video_test_results_full.md](./verification/video_test_results_full.md)
- **Demo:** `python models/demos/molmo2/demo/demo.py --use-trace`
- **Batched Demo (parallel):** `python models/demos/molmo2/demo/demo.py --input-file models/demos/molmo2/demo/sample_prompts/multi_prompts.json --batch-size 4 --use-decode-trace`
- **Server:** `cd tt-inference-server && python run.py --model Molmo2-8B --workflow server --tt-device t3k --local-server`

## Current Status (2026-04-04)

**HF parity workstreams (see `.cursor/plans/molmo2_video_hf_parity_*.plan.md` if present):**

1. **Bidirectional attention for image tokens** — `token_type_ids` from HF processor + additive prefill mask (`tt/prefill_attention_mask.py`); wired through `molmo2_model.forward` and `demo.run_prefill`. Prefill traces are disabled when this mask is used.
2. **Pooling** — Masked mean query, pooling attention mask, unified `MAX_FRAMES_FOR_SINGLE_POOL`; chunked path no longer applies global norm+clamp before the projector.
3. **Text RMSNorm** — `rms_norm_eps=1e-6` aligned with HF; Q/K RMSNorm uses block epsilon from `TextAttention`.
4. **ViT MLP** — `activation="gelu"` in TTNN documented as matching HF `gelu_pytorch_tanh` for this model; re-validate with PCC if logits drift.

**Residual / verification:** Run `pytest models/demos/molmo2/tests/test_prefill_attention_mask.py` and `pytest models/demos/molmo2/tests/test_hf_verification_harness.py` (includes `test_jsonl_rows_align_with_test_results`: each `test_results.jsonl` row matches the same index in `test.jsonl`). HF greedy vs recorded text for test 0: put the eval `.mp4` (hash from URL) under the repo root or `verification/videos/`, then `MOLMO2_VERIFY_HF_JSONL=1 pytest ...`. Full TTNN vs HF parity on device: `MOLMO2_RUN_HW_PARITY=1`. If a recorded result disagrees with a fresh HF run, update artifacts per HF.

## Memory and long-sequence knobs

- **ViT:** `total_patches ≈ frames × 729` per video; data-parallel paths split frames across devices, but pooling may still need the full pooled sequence for global indices.
- **Pooling:** Large `frames × n_out` increases peak memory; chunked pooling trades extra passes for lower peak — tune `max_frames_per_pool_chunk` in chunked pool paths if OOM.
- **Text prefill:** Sequence length grows with image/video tokens; the demo uses chunked prefill (`demo.py` `_run_chunked_prefill`) with paged attention when over `max_prefill_chunk_size`. **Chunked prefill does not apply the multimodal `token_type_ids` mask** — long sequences with that mask use full prefill (higher memory).
- **OOM:** Reduce `VIDEO_MAX_FRAMES`, resolution, or chunk sizes; truncating inputs has no silent correctness guarantee.

## In-repo TTNN references (before inventing new patterns)

| Area | Path |
|------|------|
| Multimodal / VLM patterns | `models/demos/qwen3_vl/` |
| ViT / bidirectional SDPA | `models/demos/vision/classification/vit/` |
| Cross-attention / pooling | `models/tt_transformers/tt/multimodal/llama_cross_attention.py` |

## TTNN API audit (SDPA and friends)

Canonical parameter docs: `ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa_nanobind.cpp`. Prefill SDPA tests: `tests/ttnn/unit_tests/operations/sdpa/test_sdpa_prefill.py`. When changing `scaled_dot_product_attention` or chunked SDPA, re-check docstrings and those tests against `text_attention.py` / `vision_attention.py` call sites.

## Previous Status (2026-03-30)

**vLLM Server:** WORKING for concurrent text and image requests (max_concurrency=32)
- True batched decode with 32 concurrent requests verified working
- Each request gets correct contextual response (e.g., "User 2: 2+2=4")
- Uses vLLM's start_pos for per-request positions via `prepare_decode_inputs()`
- Batched RoPE via `get_rot_mats_decode_batched()` with ttnn.embedding lookup

**Concurrent Request Support Matrix:**

| Request Type | Sequential | Concurrent |
|--------------|-----------|------------|
| Text-only | ✓ Working | ✓ Working (32 tested) |
| Image | ✓ Working | ✓ Working (via vLLM API) |
| Video | ✓ Working | ✓ Working (via video_url) |

**Video Demo with Decode Trace:** 35.69 tok/s (8 frames, --use-decode-trace)

**vLLM Video API Format:**
```json
{"type": "video_url", "video_url": {"url": "file:///path/to/video.mp4"}}
```

**Docker:** Requires newer image - current image has ETH firmware mismatch on T3K.

**Key Fixes for vLLM Batched Decode:**
1. Token reshape: `[batch, 1]` → flatten → pad to 32 → `[1, 32]` for embed_tokens
2. current_pos: 1D `[batch]` for paged_update_cache (was 2D `[1, batch]`)
3. Traced decode disabled for batch>1 (trace captured with scalar positions)
4. Logits reshape: to_torch returns `[1, batch, vocab]` (3D) → extract `[0, :batch, :]` → unsqueeze to `[batch, 1, vocab]`
5. rot_mat_idxs passed to forward_decode (not rot_mats) so it uses batched RoPE for batch>1

**Demo Batched Inference:** WORKING - true parallel batch processing
- 4 prompts: 141.03 tok/s total (35.26 tok/s/user)
- 32 prompts: 767.52 tok/s total (23.99 tok/s/user)
- Each user gets coherent output for their specific prompt

**vLLM Batched Decode Implementation:**
- `generator_vllm.py`: `prepare_decode_inputs()` creates batch-sized current_pos and rot_mat_idxs
- `text_rotary_setup.py`: `get_rot_mats_decode_batched()` using ttnn.embedding lookup
- `text_model.py`: Auto-detects batch>1 and uses batched rot_mats
- `text_attention.py`: Manual RoPE application for batched case (element-wise ops)

**Verified:**
- Text: 50/50 tests passed
- Images: 50/50 tests passed
- Video: 105/105 tests passed (local + Docker)

**Eval Benchmarks:** Added to tt-inference-server
- chartqa: 9.36% (published 85.7%) - format mismatch issues
- docvqa_val: Server crashed at 38%
- mmmu_val: Not tested yet

## Key Files

| File | Purpose |
|------|---------|
| `tt/generator_vllm.py` | vLLM integration - **main file to modify** |
| `demo/demo.py` | Standalone demo (works with images) |
| `tt/text_attention.py` | Attention layer with paged cache |

## Critical Code Locations

- **Line 1660 in generator_vllm.py:** `use_trace=False` for images - this is the problem
- **Line 319 in text_attention.py:** Reshape crash in non-traced path

## Testing

Always reset device before testing after a hang:
```bash
pkill -9 -f "EngineCore|APIServer"
tt-smi -r
```

## Constraints

- **Inference:** ViT, pooling, projector, and text transformer run on TTNN. HuggingFace is used for **preprocessing only** (processor outputs: `input_ids`, pixels, pooling indices, masks, `token_type_ids`).
- Must work with vLLM's paged attention
- PCC > 0.99 required for all TTNN blocks
