# Molmo2 Development Guide

## Quick Reference

- **Server Status:** See [SERVER_STATUS.md](./SERVER_STATUS.md) for current vLLM server status
- **Eval Results:** See [verification/eval_benchmarks_results.md](./verification/eval_benchmarks_results.md)
- **Video Tests:** See [verification/video_test_results_full.md](./verification/video_test_results_full.md)
- **Demo:** `python models/demos/molmo2/demo/demo.py --use-trace`
- **Server:** `cd tt-inference-server && python run.py --model Molmo2-8B --workflow server --tt-device t3k --local-server`

## Current Status (2026-03-29)

**vLLM Server:** STABLE for text, images, and video (traces disabled for vision)

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

- Do NOT use HuggingFace for vision processing (must stay fully in TTNN)
- Must work with vLLM's paged attention
- PCC > 0.99 required for all TTNN blocks
