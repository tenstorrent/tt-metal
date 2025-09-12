# DeepSeek‑V3 Demo (TT‑NN)

This folder contains a small, self‑contained CLI demo that runs DeepSeek‑V3 decode on Tenstorrent hardware using TT‑NN. It performs greedy generation, emulating prefill by stepping through the prompt and then decoding new tokens.

The script supports two practical modes:
- Full‑model mode using local Hugging Face safetensors (accuracy‑oriented)
- Single‑layer with random weights (fast, no safetensors; useful for quick pipeline/hardware checks)

## Requirements
- TT‑Metalium / TT‑NN installed and working (see INSTALLING.md)
- Tenstorrent device visible (check with `tt-smi`)
- Python environment for this repo (`./create_venv.sh && source python_env/bin/activate && pip install -e .`)
 - A local DeepSeek‑V3‑compatible HF model directory for config
  - Full‑model mode: requires config, tokenizer, and `.safetensors`
  - Random‑weights mode: requires config; tokenizer is optional (dummy tokenization is used if missing)

Common environment variables:
- `DEEPSEEK_V3_HF_MODEL`: Path to a local Hugging Face model directory that contains config (and tokenizer + safetensors for full‑model mode). Defaults to `models/demos/deepseek_v3/reference`.
- `DEEPSEEK_V3_CACHE`: Optional path where converted TTNN weights and caches are written. Defaults to `generated/deepseek_v3`.

Tokenizer (full‑model mode): expect one of `tokenizer.model`, `tokenizer.json`, `spiece.model`, or `tokenizer_config.json`.

## Quick Start

### 1) Full model with local safetensors
Use this when you have a local DeepSeek‑V3 model (config, tokenizer, and `.safetensors`).

```bash
# Point to a local HF model directory (contains tokenizer + .safetensors)
export DEEPSEEK_V3_HF_MODEL=/proj_sw/user_dev/deepseek-ai/DeepSeek-R1-0528
# Optional: where to store converted TTNN weights
export DEEPSEEK_V3_CACHE=/proj_sw/user_dev/deepseek_ttnn_cache_all_61_layers

python models/demos/deepseek_v3/demo/demo.py \
  "Explain quantum tunneling in one paragraph." \
  --max-new-tokens 64
```

You can also pass `--model-path` instead of setting the env var:

```bash
python models/demos/deepseek_v3/demo/demo.py \
  "Write a haiku about compilers." \
  --model-path /proj_sw/user_dev/deepseek-ai/DeepSeek-R1-0528 \
  --max-new-tokens 32
```

### 2) Single‑layer with random weights (fast)
Runs a minimal single‑layer pipeline with randomly initialized weights. This does not require `.safetensors`, and the tokenizer is optional. If no tokenizer is found, the demo synthesizes simple token IDs. The prompt is not used in this mode, so you don’t need to provide one.

```bash
# Model path only needs config (tokenizer optional) and defaults to models/demos/deepseek_v3/reference

python models/demos/deepseek_v3/demo/demo.py \
  --random-weights --single-layer mlp \
  --max-new-tokens 16
```

Notes:
- `--single-layer=mlp` is supported; `--single-layer=moe` is not supported and will error out.
- Output will be nonsense (random weights); this mode is for quick validation of the data path, caching, and device bring‑up.

## CLI Reference

```text
usage: DeepSeek-V3 Demo on TT-NN [-h] [--model-path PATH] [--max-new-tokens N]
                                 [--cache-dir PATH] [--random-weights]
                                 [--single-layer {mlp,moe}] [prompt]
```

- `prompt`: Text prompt to generate from (required in full‑model mode; ignored in `--random-weights`).
- `--model-path PATH`: Local HF model directory. Defaults to `$DEEPSEEK_V3_HF_MODEL` or `models/demos/deepseek_v3/reference`.
  - Full‑model mode requires tokenizer files and at least one `.safetensors` shard.
  - Random‑weights mode only requires `config.json`; tokenizer is optional.
- `--max-new-tokens N`: Number of tokens to generate (default: 32). Greedy decoding only.
- `--cache-dir PATH`: Where to store converted TTNN weights and caches (default: `$DEEPSEEK_V3_CACHE` or `generated/deepseek_v3`).
- `--random-weights`: Use randomly initialized weights derived from the HF config (no safetensors).
- `--single-layer {mlp,moe}`: With `--random-weights`, request a single‑layer run. `mlp` is supported; `moe` is not.

## Examples

- Explicit model and cache paths:
  ```bash
  python models/demos/deepseek_v3/demo/demo.py \
    "Summarize the history of RISC architectures." \
    --model-path /proj_sw/user_dev/deepseek-ai/DeepSeek-R1-0528 \
    --cache-dir /proj_sw/user_dev/deepseek_ttnn_cache_all_61_layers \
    --max-new-tokens 80
  ```

- Fast pipeline check, no safetensors, no prompt:
  ```bash
  python models/demos/deepseek_v3/demo/demo.py \
    --model-path /proj_sw/user_dev/deepseek-ai/DeepSeek-R1-0528 \
    --random-weights --single-layer mlp \
    --max-new-tokens 8
  ```

## Behavior and Output
- The demo opens a mesh device based on available device IDs (e.g., `4x8` mesh for 32 devices, otherwise a single row).
- Prefill is emulated by iterating decode steps over the prompt before generating new tokens.
- Prints the generated text between separators:
  ```
  ===== Generated =====
  ... text ...
  =====================
  ```

## Troubleshooting
- "Tokenizer files not found": Ensure the `--model-path` (or `$DEEPSEEK_V3_HF_MODEL`) directory includes tokenizer assets
  such as `tokenizer.model` or `tokenizer.json`.
- "No .safetensors files found": You are in full‑model mode. Either provide a directory with safetensors or use
  `--random-weights --single-layer mlp` for a quick run without weights.
- "--single-layer=moe not supported": Only `mlp` is supported in the single‑layer random‑weights demo.

## Notes
- Converted weights are cached under `--cache-dir/weights` to speed up subsequent runs.
- This script focuses on decode and greedy sampling for simplicity; stopping at EOS, temperature/top‑k/p are not exposed here.
