The `tt-train/sources/examples/grpo` folder provides an end-to-end **GRPO training/evaluation example** on GSM8K for `unsloth/Llama-3.2-1B-Instruct` using TT-Train/TTML. It includes:
- a GRPO training entrypoint,
- an accuracy/eval entrypoint,
- shared utilities for setup, inference, reward/loss computation, and run bookkeeping,
- config variants for different device meshes,
- helper scripts and tests for decoding consistency.

### Most important implementation methods

`_completion_batched_impl`, `create_casual_mask`, `compute_nlog_probs`, `compute_grpo_loss`

### File-by-file overview

### Main scripts

- `grpo_training.py`
  Main GRPO training loop: loads model/config, samples completions in batches, computes rewards/advantages, computes GRPO loss, runs backward/optimizer steps, logs metrics, and writes checkpoints.

- `grpo_model_accuracy.py`
  Offline accuracy runner on GSM8K test prompts: generates completions, extracts final numeric answers, compares against gold answers, and logs running accuracy/results.

### Utils

- `utils/setup.py`
  Shared runtime/model initialization: YAML parsing, tokenizer/model load (HF or checkpoint), DDP mapper/composer wiring, GRPO config extraction, optimizer creation, and checkpoint loading helpers.

- `utils/inference.py`
  Batched generation internals: prompt padding, KV cache handling, causal mask creation, decode loop, stop-token handling, and tensor deallocation helpers.

- `utils/loss.py`
  GRPO math utilities: reward/advantage computation from extracted answers, negative log-prob computation over prompt+completion tokens, and clipped surrogate GRPO loss.

- `utils/gsm8k.py`
  GSM8K dataset plumbing: loads dataset, builds few-shot chat prompts, and extracts `#### <number>` answers.

- `utils/bookkeeping.py`
  Run lifecycle/logging utilities: output directory creation, source/config archival, metadata capture, tee logging, checkpoint save helper, and CSV metric trackers (training + accuracy).

- `utils/ttml_operators.py`
  Custom autograd ops (`Exp`, `Clip`, `Min`) used in GRPO loss construction.

- `utils/llama_overrides.py`
  Llama attention overrides using composite SDPA and KV-cache-aware decode path tailored for this GRPO workflow.

- `utils/plot_rewards.py`
  Post-run visualization helper that parses `reward_mean/reward_std` from logs and writes a rewards plot PNG.

- `test_batched_vs_single_completion.py`
  Regression-style test ensuring batched decoding matches one-by-one decoding for deterministic settings (`temperature=0`, `group_size=1`).

#### Configs

Training configs are located in the tt-train/configs. Accuracy & test configs are in the tt-train/sources/examples/grpo folder.

#### Helper scripts

- `run_grpo_training_nohup.sh`
  Convenience wrapper to launch `grpo_training.py` under `nohup` from repo root.

- `run_model_accuracy_nohup.sh`
  Convenience wrapper to launch `grpo_model_accuracy.py` under `nohup` from repo root.

- `conftest.py`
  Pytest path setup so sibling `utils` modules import cleanly.


#### `grpo_training.py` arguments
- `--yaml-path`
  Default: `tt-train/configs/training_configs/training_grpo_gsm8k_unsloth_llama_3_2_1b_instruct.yaml`
  **Default config is without DDP**
- `--checkpoint-interval`, default: 50
- `--start-checkpoint-path`
  Optional checkpoint to initialize from instead of HF pretrained weights.
- `--output-dir`
  Optional output directory for logs/checkpoints/metadata.
  By default outputs the logs into ``tt-metal/generated/tt-train/grpo_training_runs/TIMESTAMP`` folder.
---
#### `grpo_model_accuracy.py` arguments
- `--num-prompts`  Number of prompts to evaluate (if omitted, evaluates all).
- `--yaml-path`  Default: `tt-train/sources/examples/grpo/grpo_model_accuracy.yaml`
- `--checkpoint-path` Optional checkpoint to evaluate. By default evaluates the hugging face model.
- `--output-dir` Optional output directory for logs/results/metadata.  By default outputs the logs into ``tt-metal/generated/tt-train/grpo_accuracy_runs/TIMESTAMP`` folder.
---

### Environment variables to set before running training

`TT_METAL_RUNTIME_ROOT`, `HF_TOKEN`, `TT_MESH_GRAPH_DESC_PATH`
