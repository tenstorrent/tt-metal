# EAGLE3 DeepSeek R1 Demo Runbook

This folder contains a PyTorch CPU-first research implementation of EAGLE3-style speculative decoding.

## Reliability Note

- Results produced with very small CPU smoke-test base models (for example `sshleifer/tiny-gpt2`) and/or mismatched draft-base hidden spaces are **not reliable performance conclusions**.
- Use those runs for integration validation, trend inspection, and debugging only.
- Final claims should be made only with aligned target checkpoints and appropriate hardware.

## Draft branching: `top_k` vs `temperature_top_p`

By default the **draft** expands each beam using the **`top_k` highest logits** (deterministic).

Optional mode **`draft_branching=temperature_top_p`** (CLI: `--draft-branching temperature_top_p`) keeps the same beam structure and **`top_k` as the number of branches to draw**, but picks those branches by **sampling** from a shaped distribution:

- **`draft_temperature` (T):** logits are divided by T before softmax. **Larger T** flattens probabilities so lower-ranked tokens are chosen more often (more diverse speculative paths). **Smaller T** concentrates mass on the head (closer to argmax). **`T ≤ 0`** uses argmax (one branch unless `top_k` still loops—actually returns single id).

- **`draft_top_p` (p):** **nucleus (top-p) filtering** on that tempered distribution: sort tokens by probability, keep the smallest set whose cumulative mass is at least **p**, drop the tail, renormalize, then sample **without replacement** up to **`min(top_k, nucleus_size)`** tokens. **Lower p** shrinks the candidate set (more aggressive truncation); **p = 1** keeps the full tempered distribution before sampling.

**Verification** is unchanged: the base still accepts/rejects using its own logits (and optional probabilistic rule). For **`q` in `min(1, q/p)`**, draft token probabilities **`q`** are taken from the **untempered** draft softmax (temperature 1), not from the sampling distribution—so `q` remains the draft model’s plain likelihood of the proposed token.

**Reproducibility:** set **`random_seed`** when using `temperature_top_p` so `torch.Generator` and verification RNG are fixed.

**Why a numeric `top_k` at all?** In `temperature_top_p` mode it is only a **cap on how many distinct tokens you sample** from the nucleus each step. To match “top-p only” with **no extra k limit**, set **`top_k <= 0`** (e.g. `--top-k 0`): every token in the nucleus becomes a branch (sorted by probability). That can be **very** wide.

**`max_paths`:** by default beams are pruned to the best `max_paths`. To disable that pruning, set **`max_paths <= 0`** (e.g. `--max-paths 0`). Without both caps, cost grows with **nucleus_size^depth** — use only for shallow depth or small nuclei.

Defaults when using nucleus mode: **`draft_temperature=0.6`**, **`draft_top_p=0.95`** (see `EagleConfig` and script `--draft-temperature` / `--draft-top-p`).

## Environment

Use the project Python environment:

```bash
/proj_sw/user_dev/$USER/tt-metal/python_env/bin/python
```

In commands below, replace `<PY>` with that path for convenience.

Recommended cache environment:

```bash
export HF_HOME=/proj_sw/user_dev/$USER/hf_cache
unset TRANSFORMERS_CACHE
```

`TRANSFORMERS_CACHE` is deprecated in newer `transformers` versions; prefer `HF_HOME`.

## Multi-GPU TP (8xH200)

For large R1 checkpoints, use tensor-parallel launch with `torchrun` and `--tp-size 8`.

Pre-flight checklist (minimize deployment errors):

- install TP backend: `pip install deepspeed ninja`
- verify 8 visible GPUs: `nvidia-smi -L`
- use `torchrun --nproc_per_node=8 ...`
- ensure `HF_HOME` points to a large disk
- run baseline smoke first with short generation (8-16 tokens)
- only rank 0 writes outputs in TP mode (handled by scripts)

Automated preflight helper:

```bash
bash models/demos/speculative_deepseek_r1_broad/scripts/run_gpu_preflight.sh \
  --expected-gpus 8 \
  --tp-size 8
```

Checks-only mode (skip smoke launch):

```bash
bash models/demos/speculative_deepseek_r1_broad/scripts/run_gpu_preflight.sh \
  --expected-gpus 8 \
  --tp-size 8 \
  --skip-smoke
```

## Run On GPU Without Pushing

You do not need to push to GitHub for preliminary GPU tests.

Option A: shared filesystem (fastest)

- If GPU node sees the same workspace path, just activate env and run commands directly.

Option B: copy local working tree to GPU node with `rsync`

```bash
rsync -av --delete \
  /proj_sw/user_dev/$USER/tt-metal/ \
  <gpu-user>@<gpu-host>:/proj_sw/user_dev/<gpu-user>/tt-metal/
```

Option C: tarball transfer (no git remote usage)

```bash
tar -czf /tmp/tt-metal-prelim.tar.gz -C /proj_sw/user_dev/$USER tt-metal
scp /tmp/tt-metal-prelim.tar.gz <gpu-user>@<gpu-host>:/tmp/
ssh <gpu-user>@<gpu-host> "mkdir -p /proj_sw/user_dev/<gpu-user> && tar -xzf /tmp/tt-metal-prelim.tar.gz -C /proj_sw/user_dev/<gpu-user>"
```

Recommended practice for prelim runs:

- keep changes on a local branch (no push)
- sync/copy code to GPU host
- run preflight script first
- run short TP smoke, then full experiments

## Base Model Presets

Most scripts now support `--base-model-preset`:

- `r1_0528` -> `deepseek-ai/DeepSeek-R1-0528`
- `r1` -> `deepseek-ai/DeepSeek-R1`
- `distill_llama_8b` -> `deepseek-ai/DeepSeek-R1-Distill-LLaMA-8B` (matches EAGLE3 draft head)
- `distill_qwen_1_5b` -> `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (lightweight CPU)
- `tiny_gpt2` -> `sshleifer/tiny-gpt2` (fastest debug smoke tests)

Most scripts also support `--base-impl`:

- `reference` (default): reference-style base loader under `reference/` (more detailed structure).
- `adapter`: current lightweight adapter path.

## Draft Modes

Scripts support `--draft-mode`:

- **MTP-style draft without loading a full DeepSeek base**: use `scripts/run_nextn_mtp_from_record_cpu.py` — record replay base + `lmsys/DeepSeek-R1-NextN` fusion weights from the Hub (optional `--embed-head-aux-safetensors` if embed/head are not inside the fusion file). No full checkpoint load in this repo.
- `draft_r1`: Traditional speculative decoding with [`jukofyork/DeepSeek-R1-DRAFT-0.6B-v3.0`](https://huggingface.co/jukofyork/DeepSeek-R1-DRAFT-0.6B-v3.0) — a 0.6B model with vocabulary transplanted to match R1-0528's tokenizer. Can run on CPU. No hidden-state coupling needed.
- `eagle3_8b`: EAGLE3 hidden-state draft head (`yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B`). Must be paired with `distill_llama_8b` as base.
- `self`: Reuses the base model as its own draft (for debugging/integration tests only).

Draft model presets (`--draft-model-preset`):

- `r1_draft_0_6b` -> `jukofyork/DeepSeek-R1-DRAFT-0.6B-v3.0` (for `draft_r1` mode)
- `eagle3_8b` -> `yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B` (for `eagle3_8b` mode)

**For R1-class MTP on CPU from a recorded trace**: `run_nextn_mtp_from_record_cpu.py`. **For live EAGLE with a smaller base**: `--draft-mode draft_r1` or `eagle3_8b` with `run_eagle3_deepseek_cpu.py`.

## Useful Commands

### 1) Baseline decode (no EAGLE)

```bash
<PY> models/demos/speculative_deepseek_r1_broad/scripts/run_baseline_deepseek_cpu.py \
  --prompt "Explain speculative decoding in one sentence." \
  --base-model-preset distill_qwen_1_5b \
  --base-impl reference \
  --max-new-tokens 32 \
  --device cpu
```

TP example (8 GPUs):

```bash
torchrun --nproc_per_node=8 \
models/demos/speculative_deepseek_r1_broad/scripts/run_baseline_deepseek_cpu.py \
  --prompt "Explain speculative decoding in one sentence." \
  --base-model-preset r1_0528 \
  --base-impl reference \
  --tp-size 8 \
  --device cuda \
  --dtype bfloat16 \
  --max-new-tokens 16
```

### 2a) Record + NextN MTP (CPU, no full DeepSeek load)

Default record path is `DEFAULT_MTP_RECORD_PATH` in `models/demos/speculative_deepseek_r1_broad/default_paths.py` (this workspace: `.../deepseek-v3-cache/test_io_cache/mtp_full_model_seq128.pt`).

```bash
cd /proj_sw/user_dev/dchrysostomou/tt-metal
export PYTHONPATH=/proj_sw/user_dev/dchrysostomou/tt-metal:$PYTHONPATH
python models/demos/speculative_deepseek_r1_broad/scripts/run_nextn_mtp_from_record_cpu.py
```

**`aux.safetensors`:** only if `nextn_layer_parameters.safetensors` from the Hub does **not** already include both **token embeddings** and **`model.layers.0.shared_head.head.weight`**. Otherwise NextN + record is enough. Inspect keys with the one-liner in `agent_plan.md` (**Primary goal** section).

**Optional case study (separate script):** full Hub `AutoModelForCausalLM` NextN as draft (MoE/MLA stack on tokens, not MTP-head-on-record-hidden) — `scripts/case_study_nextn_full_layer_draft_from_record_cpu.py` and `case_studies/nextn_full_layer_draft_from_record.md`.

### 2b) EAGLE3 run (hidden-state draft head)

Requires paired base: `distill_llama_8b` + EAGLE3 draft head.

```bash
<PY> models/demos/speculative_deepseek_r1_broad/scripts/run_eagle3_deepseek_cpu.py \
  --prompt "Explain speculative decoding in one sentence." \
  --base-model-preset distill_llama_8b \
  --base-impl reference \
  --draft-mode eagle3_8b \
  --max-new-tokens 32 \
  --top-k 2 \
  --depth 2 \
  --num-steps 1 \
  --device cpu
```

### 2c) Self-draft smoke test (debugging)

Uses the base model as its own draft. Only useful for verifying pipeline mechanics.

```bash
<PY> models/demos/speculative_deepseek_r1_broad/scripts/run_eagle3_deepseek_cpu.py \
  --prompt "Speculative decoding is" \
  --base-model-preset tiny_gpt2 \
  --base-impl reference \
  --draft-mode self \
  --max-new-tokens 16 \
  --top-k 2 \
  --depth 2 \
  --num-steps 1 \
  --device cpu
```

TP example (8 GPUs):

```bash
torchrun --nproc_per_node=8 \
models/demos/speculative_deepseek_r1_broad/scripts/run_eagle3_deepseek_cpu.py \
  --prompt "Explain speculative decoding in one sentence." \
  --base-model-preset r1_0528 \
  --base-impl reference \
  --draft-mode eagle3_8b \
  --tp-size 8 \
  --device cuda \
  --dtype bfloat16 \
  --max-new-tokens 16 \
  --top-k 2 \
  --depth 2 \
  --num-steps 1
```

EAGLE shape/cache diagnostics (per round):

```bash
<PY> models/demos/speculative_deepseek_r1_broad/scripts/run_eagle3_deepseek_cpu.py \
  --prompt "Explain speculative decoding in one sentence." \
  --base-model-preset distill_qwen_1_5b \
  --draft-mode eagle3_8b \
  --max-new-tokens 16 \
  --top-k 2 \
  --depth 2 \
  --num-steps 1 \
  --verbose \
  --verbose-shapes \
  --device cpu
```

Probabilistic verification acceptance (optional; default is argmax):

```bash
<PY> models/demos/speculative_deepseek_r1_broad/scripts/run_eagle3_deepseek_cpu.py \
  --prompt "Explain speculative decoding in one sentence." \
  --base-model-preset distill_qwen_1_5b \
  --draft-mode eagle3_8b \
  --max-new-tokens 16 \
  --top-k 2 \
  --depth 2 \
  --num-steps 1 \
  --verification-acceptance probabilistic \
  --random-seed 0 \
  --device cpu
```

Notes:

- `active_paths_in` / `active_paths_out` refer to the number of currently active draft paths at each depth step.
- In eagle3_8b draft mode, draft scoring now runs as a batched pass once per depth (`batched_draft_forward` in logs).
- Baseline and EAGLE scripts strip `<think>...</think>` style segments in displayed text by default.
  - Use `--no-strip-think-tags` to print raw output.
- For token-level debugging, use `--print-raw-tokens` (and optionally `--raw-token-max`).
- Metric naming:
  - `Accepted-per-depth rate`: accepted tokens on selected path normalized by depth.
  - `Rounds-with-any-accept rate`: fraction of proposal rounds where at least one token was accepted.

**EAGLE3 FC mode (paper vs released):** By default the draft uses the FC at every depth (matches released checkpoints; acceptance ~0.69). For the paper formulation (FC only at depth 0), pass `--no-eagle3-fc-every-depth`. To compare acceptance rates:

```bash
<PY> models/demos/speculative_deepseek_r1_broad/scripts/compare_eagle3_fc_modes.py \
  --prompt "The capital of France is" \
  --base-model-preset distill_qwen_1_5b \
  --draft-model-preset eagle3_8b \
  --max-new-tokens 32 --depth 2 --top-k 4 --random-seed 42 \
  --device cpu
```

### 3) Yuhuili draft sanity check (M1-style)

```bash
<PY> models/demos/speculative_deepseek_r1_broad/scripts/run_yuhuili_draft_sanity_cpu.py \
  --prompt "Speculative decoding is" \
  --base-model-preset distill_qwen_1_5b \
  --base-impl reference \
  --draft-model-id yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B \
  --sample-steps 32 \
  --device cpu
```

Draft/base overlap diagnostics:

```bash
<PY> models/demos/speculative_deepseek_r1_broad/scripts/run_yuhuili_draft_sanity_cpu.py \
  --prompt "Speculative decoding is" \
  --base-model-preset distill_qwen_1_5b \
  --base-impl reference \
  --draft-model-id yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B \
  --sample-steps 32 \
  --draft-top-k 4 \
  --base-top-k 5 \
  --diagnose-overlap \
  --device cpu
```

Per-step token text check-up (optional, off by default):

```bash
<PY> models/demos/speculative_deepseek_r1_broad/scripts/run_yuhuili_draft_sanity_cpu.py \
  --prompt "Speculative decoding is" \
  --base-model-preset distill_qwen_1_5b \
  --base-impl reference \
  --draft-model-id yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B \
  --sample-steps 16 \
  --draft-top-k 4 \
  --base-top-k 5 \
  --diagnose-overlap \
  --print-step-overlap \
  --print-max-steps 6 \
  --device cpu
```

### 4) Baseline vs EAGLE benchmark table sweep

```bash
<PY> models/demos/speculative_deepseek_r1_broad/scripts/run_eagle3_benchmark_table_cpu.py \
  --prompt "Speculative decoding is" \
  --base-model-preset distill_qwen_1_5b \
  --base-impl reference \
  --draft-mode eagle3_8b \
  --draft-model-id yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B \
  --max-new-tokens-list 32,128 \
  --top-k-list 2,4 \
  --depth-list 1,2 \
  --num-steps-list 1,2 \
  --device cpu \
  --json-out /tmp/eagle3_benchmark_table.json
```

### 5) Acceptance/iteration sweep (top-k, depth, num_steps)

```bash
<PY> models/demos/speculative_deepseek_r1_broad/scripts/run_eagle3_acceptance_sweep_cpu.py \
  --prompt "Speculative decoding is" \
  --base-model-preset distill_qwen_1_5b \
  --base-impl reference \
  --draft-mode eagle3_8b \
  --draft-model-id yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B \
  --max-new-tokens 64 \
  --top-k-list 1,2,4 \
  --depth-list 1,2,3 \
  --num-steps-list 1,2 \
  --device cpu \
  --json-out /tmp/eagle3_acceptance_sweep.json
```

### 6) GPU trace collection -> CPU replay

Collect base decode trace on GPU:

```bash
<PY> models/demos/speculative_deepseek_r1_broad/scripts/collect_base_trace_gpu.py \
  --prompt "Speculative decoding is" \
  --base-model-preset distill_qwen_1_5b \
  --base-impl reference \
  --max-new-tokens 64 \
  --topk 8 \
  --device cuda \
  --dtype bfloat16 \
  --out /tmp/deepseek_trace.pt
```

TP example (8 GPUs):

```bash
torchrun --nproc_per_node=8 \
models/demos/speculative_deepseek_r1_broad/scripts/collect_base_trace_gpu.py \
  --prompt "Speculative decoding is" \
  --base-model-preset r1_0528 \
  --base-impl reference \
  --tp-size 8 \
  --device cuda \
  --dtype bfloat16 \
  --max-new-tokens 64 \
  --topk 8 \
  --out /tmp/deepseek_trace.pt
```

Replay EAGLE on CPU from collected trace:

```bash
<PY> models/demos/speculative_deepseek_r1_broad/scripts/run_eagle3_from_trace_cpu.py \
  --trace /tmp/deepseek_trace.pt \
  --draft-model-id yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B \
  --max-new-tokens 64 \
  --top-k 4 \
  --depth 2 \
  --num-steps 1 \
  --device cpu
```

Replay caveat:

- Replay is exact only on the recorded base trajectory.
- Speculative branches that diverge from recorded tokens are conservatively rejected.

## Prompt File Format

All scripts that support `--prompts-file` expect either:

- a top-level list:
  - `[{"prompt": "..."}, {"prompt": "..."}]`
- or an object:
  - `{"prompts": [{"prompt": "..."}, {"prompt": "..."}]}`

## Reference Runner

```bash
<PY> models/demos/speculative_deepseek_r1_broad/reference/test_run_model.py \
  --model-id sshleifer/tiny-gpt2 \
  --prompt "Speculative decoding is" \
  --max-new-tokens 16 \
  --device cpu
```

