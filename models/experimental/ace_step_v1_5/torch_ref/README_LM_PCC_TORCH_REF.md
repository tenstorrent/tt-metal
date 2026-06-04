# LM torch reference vs TTNN (`torch_ref/five_hz_lm`)

## Layout

| Path | Role |
|------|------|
| `five_hz_lm/` | Production vendored LM + TTNN assist when `set_ttnn_logits_device` / experimental causal LM is used. |
| `torch_ref/five_hz_lm/` | **PyTorch-only mirror** (same files, patched handler) for PCC tests. |
| `ttnn_impl/five_hz_lm/` | **Re-export** of production `five_hz_lm` inference + processor modules for symmetric imports in PCC tests (implementation stays in `five_hz_lm/`). |
| `ttnn_impl/lm_constrained_logits_ttnn.py`, `lm_logits_ttnn.py`, … | TTNN logits kernels used from production `five_hz_lm`. |

## Entry script

- **`torch_ref/run_ace_step_ttmetal_demo_torch_ref_lm.py`** — copy of `run_prompt_to_wav.py` that imports
  `LocalFiveHzLMHandler` from **`models.experimental.ace_step_v1_5.torch_ref.five_hz_lm`** so the full
  DiT + Qwen caption + LM pipeline runs with the **torch reference** LM package.

- **`torch_ref/run_prompt_to_wav.py`** — existing **torch-only DiT/VAE** demo (not the same as the
  tt-metal `run_prompt_to_wav.py`); keep using it for that pipeline.

## Suggested PCC test pattern

1. Run one decode step (or full generation) with production `five_hz_lm` + TTNN device attached; capture logits / ids.
2. Run the same inputs through `torch_ref.five_hz_lm` (no TTNN assist); compare with PCC / atol on host tensors.

## Automated tests (logits kernels + handler wiring)

- `models/experimental/ace_step_v1_5/tests/test_llm_handler_logits_pcc.py` — compares **PyTorch goldens** to
  `ttnn_impl/lm_constrained_logits_ttnn.py` / `lm_logits_ttnn.py` on device, checks **production vs `torch_ref`**
  handler parity on the CPU path, and checks handler **TTNN delegations** against the same kernels.
  Top-p PCC threshold is relaxed (~0.72) vs top-k (~0.87) because nucleus uses softmax + scatter on BF16 TILE.
