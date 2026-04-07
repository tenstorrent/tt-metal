# GPT-OSS-120B Token Accuracy Report

## Result

| Metric | Value |
| ------ | ----- |
| **Top1 Accuracy** | **6.25%** (4/64) |
| **Top5 Accuracy** | **12.50%** (8/64) |

## Test Configuration

- **Model**: `openai/gpt-oss-120b`
- **Hardware**: Wormhole Galaxy, 4×8 mesh (32 devices)
- **Mesh config**: `mesh_4x8`, `fabric_1d_ring`
- **Benchmark**: `.refpt` teacher-forced token accuracy — first half of reference token sequence ("Tale of Two Cities") fed as prefill, model predicts the next 64 tokens with ground-truth feedback at each step.
- **Prefill length**: 64 tokens
- **Decode iterations**: 64
- **Sampling**: greedy (temperature=0, top_p=0.08)
- **Decode trace**: disabled (required for teacher forcing)

## Command

```bash
HF_MODEL=/path/to/gpt-oss-120b \
TT_CACHE_PATH=/path/to/tt_cache/gpt-oss-120b \
pytest models/demos/gpt_oss/demo/text_demo.py -k 'token_accuracy and mesh_4x8' -v
```

## Environment

- Container: `ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64:latest`
- Build: `./build_metal.sh` (Release, clang-20 toolchain, Python 3.10, transformers 4.57.0)
- Branch: `dgolubovic/acc-test-gpt-oss-20b`
- Date: 2026-04-17

## Observed Decode Performance

- Iteration 0 (first token): ~49 s (includes setup)
- Iterations 1–63: ~242 ms each → **4.1 tok/s/user**

## Test Log

`./.build_logs/token_accuracy.log` — full pytest output.
