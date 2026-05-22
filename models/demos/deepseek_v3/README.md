# DeepSeek-V3

## Platforms
- Galaxy (WH) QUAD (4x): primary supported path for full demo and integration runs
- Single Galaxy (TG): supported for selected module/unit tests and reduced-layer debug runs

## Introduction
This demo targets the [deepseek-ai/DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) model and is compatible with other DeepSeek-V3 checkpoints. The TT-NN pipeline supports full-model execution, teacher-forced accuracy verification, random-weight smoke tests, and multiple prompt ingestion patterns for throughput benchmarking.

- [deepseek-ai/DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)
- [deepseek-ai/DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3)

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

## Preferred Checkpoint Format

The recommended DeepSeek-V3 runtime path is:
- export a stacked dequantized checkpoint with `models/demos/deepseek_v3/scripts/dequantize_hf_checkpoint.py`
- prepare a quad-ring overlay checkpoint with `models/demos/deepseek_v3/scripts/prepare_quad_ring_hf_checkpoint.py` (recommended for QUAD/128-device runs)
- point `DEEPSEEK_V3_HF_MODEL` or `--model-path` at the resulting `*-dequantized-stacked-quad-ring` directory (fallback: `*-dequantized-stacked`)
- run without an on-disk TT weight cache

`--cache-dir` and `DEEPSEEK_V3_CACHE` remain available for reference/test caches, but DeepSeek weights are converted
directly in memory on this path. If you explicitly want to consume a prebuilt legacy TT weight cache
(for example BSPM output), pass `--use-weight-cache --cache-dir <cache-root>`. Legacy caches generated before the
current DeepSeek SavedWeight metadata/versioning must be regenerated first; older-format and unversioned caches are rejected.

### DS-RC1 quad-ring checkpoint preparation

For DeepSeek-R1-0528 / DS-RC1 on QUAD, run the quad-ring preparation script with `--num-devices 128`.

```bash
python models/demos/deepseek_v3/scripts/prepare_quad_ring_hf_checkpoint.py \
  /data/deepseek/DeepSeek-R1-0528-dequantized-stacked \
  --output-model-path /data/deepseek/DeepSeek-R1-0528-dequantized-stacked-quad-ring \
  --num-devices 128
```

Fresh prepare (overwrite/rebuild output checkpoint):

```bash
export STACKED=/data/deepseek/DeepSeek-R1-0528-dequantized-stacked
export QUAD=/data/deepseek/DeepSeek-R1-0528-dequantized-stacked-quad-ring

python_env/bin/python models/demos/deepseek_v3/scripts/prepare_quad_ring_hf_checkpoint.py \
  "$STACKED" \
  --output-model-path "$QUAD" \
  --num-devices 128 \
  --force
```

Resume after interruption (reuses existing layers and skips completed work):

```bash
python_env/bin/python models/demos/deepseek_v3/scripts/prepare_quad_ring_hf_checkpoint.py \
  "$STACKED" \
  --output-model-path "$QUAD" \
  --num-devices 128 2>&1 | tee /tmp/quad_prepare_resume.log
```

After preparation, set `DEEPSEEK_V3_HF_MODEL` (or `--model-path`) to `$QUAD`.

## Running on Multi-Host Galaxy (QUAD primary)

DeepSeek-V3 practical multi-host support is QUAD (4x). DUAL (2x) remains in some launch tooling for legacy compatibility, but is deprecated and no longer the recommended path.

### Quick Start (`launch_multihost_galaxy.py`)

```bash
# Run tests on QUAD Galaxy (4 hosts)
./models/demos/deepseek_v3/scripts/launch_multihost_galaxy.py 4x -- pytest models/demos/deepseek_v3/tests/test_model.py

# Run the demo on QUAD Galaxy
./models/demos/deepseek_v3/scripts/launch_multihost_galaxy.py 4x -- python models/demos/deepseek_v3/demo/demo.py \
  --model-path \$DEEPSEEK_V3_HF_MODEL \
  "Your prompt here!"

# Dry run (print command without executing)
./models/demos/deepseek_v3/scripts/launch_multihost_galaxy.py -d 4x -- pytest models/demos/deepseek_v3/tests/test_model.py
```

### Recommended QUAD runner (`run_quad_galaxy_tests.sh`)

```bash
tests/scripts/multihost/run_quad_galaxy_tests.sh quad_demo 8 \
  --model-path /data/deepseek/DeepSeek-R1-0528-dequantized-stacked \
  --cache-path /data/deepseek/DeepSeek-R1-0528-Cache/
```

- `quad_demo` runs the QUAD demo test entrypoint.
- The second argument is UPR mode (`8`, `32`, or `all`).
- `--model-path` and `--cache-path` override the default model/cache paths.
- With torus mode enabled (default), if `<model-path>-quad-ring` exists, this script automatically switches to that prepared checkpoint.

### `launch_multihost_galaxy.py` configuration

The script automatically:
- Detects the current hostname and selects the appropriate cluster configuration
- Sources the Python virtual environment (`python_env/bin/activate`)
- Sets `MESH_DEVICE=QUAD` when running with `4x`
- Exports `DEEPSEEK_V3_HF_MODEL` and `DEEPSEEK_V3_CACHE`
- Defaults `DEEPSEEK_V3_HF_MODEL` to the stacked dequantized checkpoint path
- Leaves `DEEPSEEK_V3_CACHE` available for reference/test caches; DeepSeek weights do not use it as an on-disk TT weight cache
- Wraps your command with **`tt-run`** (MPI) for multi-host execution; see [tt-run README](../../../ttnn/ttnn/distributed/README_ttrun.md) for **auto allocation** (`--mesh-graph-descriptor`, `--hosts`) vs **legacy** (`--rank-binding`, rankfile)

### Special Commands

```bash
# Reset the Galaxy cluster (kills python processes, resets devices, clears shared memory)
./models/demos/deepseek_v3/scripts/launch_multihost_galaxy.py 4x -- reset
```

### Supported Hosts

Supported clusters:
- **g05glx01-04**: QUAD (all four hosts)

Legacy 2x host mappings remain in launcher scripts for compatibility, but are deprecated for DeepSeek-V3 workflows.

To add new host configurations, edit `models/demos/deepseek_v3/scripts/launch_multihost_galaxy.py`.

## Running on Single Galaxy

Many unit tests and submodule tests can run on a single Galaxy without requiring multi-host setup:

```bash
# Run tests directly (no launch script needed)
pytest models/demos/deepseek_v3/tests/test_mlp.py
pytest models/demos/deepseek_v3/tests/test_attention.py
```

For development, you can also run a reduced-layer demo on a single Galaxy:

```bash
MESH_DEVICE=TG python models/demos/deepseek_v3/demo/demo.py \
             --prompts-file models/demos/deepseek_v3/demo/test_prompts.json \
             --output-path deepseek_tt_out_batch_4.json \
             --max-new-tokens 128 \
             --max-users-per-row 8 \
             --override-num-layers 5 \
             --disable-trace \
             --model-path $DEEPSEEK_V3_HF_MODEL
```

This is useful for development and testing when multi-host resources are not available.
By default, the demo stops recording output once EOS is produced. Add `--no-stop-at-eos` when you need fixed-length outputs for stress or benchmark-style runs.

## Demo

Running the demo on QUAD Galaxy (4x):

```bash
# On 4x Galaxy
./models/demos/deepseek_v3/scripts/launch_multihost_galaxy.py 4x -- python models/demos/deepseek_v3/demo/demo.py \
  --model-path \$DEEPSEEK_V3_HF_MODEL \
  --early_print_first_user \
  "Write a haiku about autumnal days by the sea"
```

The `launch_multihost_galaxy` script automatically sets `DEEPSEEK_V3_HF_MODEL` and `DEEPSEEK_V3_CACHE` environment variables. You can reference them directly:

```bash
./models/demos/deepseek_v3/scripts/launch_multihost_galaxy.py 4x -- python models/demos/deepseek_v3/demo/demo.py \
  --model-path \$DEEPSEEK_V3_HF_MODEL \
  --early_print_first_user \
  "Write a haiku about autumnal days by the sea"
```

### Supported arguments

- `prompts`: Positional prompt text. Required unless `--random-weights` is set.
- `--prompts-file FILE`: Load prompts from a JSON file (see below). CLI prompts are ignored when this flag is present.
- `--num-prompts N`: Limit the number of prompts loaded from `--prompts-file`.
- `--output-path FILE`: Save generations/statistics to JSON when using `--prompts-file`. Defaults to `<prompts-file-stem>_output.json`.
- `--model-path PATH`: Local HF model directory (required). In practice this should usually be a `*-dequantized-stacked-quad-ring` checkpoint on QUAD.
- `--cache-dir PATH`: Optional directory for reference/test caches. Also used as the legacy TT weight-cache root when `--use-weight-cache` is enabled.
- `--use-weight-cache`: Load a prebuilt current-format legacy TT weight cache from `--cache-dir` instead of converting weights in memory. Use this for workflows such as BSPM-generated caches. Caches generated before the current DeepSeek SavedWeight metadata/versioning must be regenerated first.
- `--max-new-tokens N`: Number of tokens to generate (default: 32).
- `--max-users-per-row N`: Maximum active users per row for decode (default from `USERS_PER_ROW`).
- `--stop-at-eos`: Stop recording output tokens once EOS is generated. This is the default.
- `--no-stop-at-eos`: Always record `max-new-tokens`, even after EOS. Use this for fixed-length stress or perf runs.
- `--early_print_first_user`: Stream tokens for the first prompt as they are produced.
- `--generator {bp}`: Select batch-parallel generator implementation (default: `bp`).
- `--disable-trace`: Disable tracing for decode forward pass (trace is enabled by default).
- `--random-weights`: Use randomly initialized weights (single dense layer only). Does not require tokenizer or safetensors.
- `--single-layer {mlp,moe}`: When combined with `--random-weights`, request a single-layer run (`mlp` only).
- `--override-num-layers N`: Override model depth for reduced debug/development runs (for example, TG reduced-layer runs).
- `--token-accuracy`: Enable teacher-forcing decode and report accuracy (requires full-model mode plus tokenizer and reference file).
- `--reference-file PATH`: Path to `.pt/.refpt` reference file (see below).
- `--tf-prompt-len N`: Override the teacher-forcing prompt length pulled from the reference file.

You should also provide one or more prompts (each in quotes as in the above example) as positional arguments, unless using `--random-weights`. In `--random-weights` mode, prompts are optional.

### Prompt files and batch generation

The CLI accepts JSON files in either of the following layouts:

```json
[
  {"prompt": "First prompt"},
  {"prompt": "Second prompt"}
]
```

```json
{
  "prompts": [
    {"prompt": "First prompt"},
    {"prompt": "Second prompt"}
  ]
}
```

Use `--num-prompts` to truncate large prompt sets. For example, there are 256 total prompts in `models/demos/deepseek_v3/demo/test_prompts.json`, but you can limit it to a subset.

### Programmatic usage

```python
from models.demos.deepseek_v3.demo.demo import run_demo

# Full-model generation (prompt required)
run_demo(["Write a haiku about hardware"], model_path="/abs/path/to/deepseek-v3")

# Random-weights smoke test (prompt optional)
run_demo(None, random_weights=True)

# Fixed-length generation even after EOS
run_demo(["Write a haiku about hardware"], model_path="/abs/path/to/deepseek-v3", stop_at_eos=False)

# Consume a prebuilt BSPM / legacy TT weight cache
# Regenerate the cache first if it predates the current DeepSeek SavedWeight metadata/versioning.
run_demo(
    ["Write a haiku about hardware"],
    model_path="/abs/path/to/deepseek-v3-dequantized-stacked",
    cache_dir="/abs/path/to/bspm_cache",
    use_weight_cache=True,
)
```

### Performance metrics

The demo logs wall-clock statistics (prefill/decode times, tokens per second, and total runtime) when available. When writing JSON output, the statistics block is included so that automated benchmarks can consume it downstream.

### Teacher Forcing Accuracy Verification

You can verify accuracy under teacher forcing using a reference file with tokenized ground-truth. The expected format matches the tt_transformers demos/tests:

- Keys: `reference_tokens` (LongTensor [1, T]) and optional `top5_tokens` (LongTensor [T, 5]).
- The demo splits `reference_tokens` at `T//2 + 1` into input prompt and ground-truth continuation.

Generate a compatible reference file with the tt_transformers helper (use the same tokenizer/model family as your DeepSeek model to ensure token IDs match):

- Hugging Face path:
  - `python models/tt_transformers/tests/generate_reference_outputs.py --total_length 2048 --output_file models/tt_transformers/tests/reference_outputs/DeepSeek-V3.refpt --model deepseek-ai/DeepSeek-V3`
- Or use the HF-only variant:
  - `python models/tt_transformers/tests/generate_reference_hf.py --total_length 2048 --output_file models/tt_transformers/tests/reference_outputs/DeepSeek-V3.refpt --model deepseek-ai/DeepSeek-V3`

Run the DeepSeek-V3 demo with teacher forcing:

- `python models/demos/deepseek_v3/demo/demo.py --model-path /path/to/deepseek-v3 --token-accuracy --reference-file models/tt_transformers/tests/reference_outputs/DeepSeek-V3.refpt --max-new-tokens 256`
  - Optionally control prompt length independently with `--tf-prompt-len`, e.g.:
  - `... --tf-prompt-len 1024 --max-new-tokens 256`

Notes:

- `--token-accuracy` is not compatible with `--random-weights` and requires tokenizer files in `--model-path`.
- The demo decodes a single sequence in teacher-forcing mode. `--max-new-tokens` is capped to the number of available ground-truth tokens in the reference file.
- If `top5_tokens` is present in the reference, the demo reports both top-1 and top-5 accuracies; otherwise, only top-1.

## How to develop

If you are not running on Tenstorrent internal infrastructure, you need to set the following environment variables:

- `DEEPSEEK_V3_HF_MODEL`: Path to a directory containing the DeepSeek-V3 Hugging Face model weights. Defaults to `models/demos/deepseek_v3/reference`. In practice this should normally point at a `*-dequantized-stacked-quad-ring` checkpoint (preferred for QUAD) produced from a stacked checkpoint created by `models/demos/deepseek_v3/scripts/dequantize_hf_checkpoint.py`.
- `DEEPSEEK_V3_CACHE`: Path to a directory where reference outputs, test inputs/outputs, and similar artifacts can be stored. This is no longer a TT weight cache for the DeepSeek-V3 runtime.

These variables are used in scripts for generating test data and running tests.

This codebase separates model execution into three distinct stages, each of which can be run independently:
1. Convert PyTorch weights to TTNN tensors and generate the WeightConfig
2. Generate ModelConfigs for prefill and decode modes
3. Merge the converted weights with model state/config to create a RunPrefillConfig or RunDecodeConfig and execute the model

The modules are not instantiated directly, but rather used as a namespace for the methods that define the model's behavior in prefill and decode. This is to make it easy to separate the stateful and stateless parts of the model, and allow for easy re-use of the methods.

### Weight Configuration
Generated by static method `convert_weights` on each module class. For the DeepSeek-V3 runtime these weights are typically materialized directly as TTNN tensors in memory rather than written to an on-disk TT cache.
```python
{
    "w1": <ttnn.Tensor>,
    "w2": <ttnn.Tensor>,
    "w3": <ttnn.Tensor>,
}
```

### Per-Submodule Model Configs
Generated by static methods `prefill_model_config` and `decode_model_config` on each module class. Contains operator configurations using dataclasses from `config_dataclass.py`:
```python
{
    "w1": LinearConfig(
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        program_config=matmul_program_config,
        compute_kernel_config=ttnn.experimental.tensor.CoreRangeSet(...)
    ),
    "mul_activation": MulConfig(
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        input_tensor_a_activations=[ttnn.UnaryOpType.SILU] # list because ttnn.mul expects a list
    ),
    "all_reduce": AllReduceConfig(
        cluster_axis=0,
        topology=ttnn.Topology.Ring,
        dtype=ttnn.bfloat8_b,
        dim=3,
        num_reduce_scatter_links=1,
        num_all_gather_links=1,
        use_composite=True
    )
}
```

### Example Usage
```python
# Stage 1: Convert weights and get weight_config (DeepSeek-V3 runtime keeps these in memory)
weight_config = MLP.convert_weights(hf_config, torch_state_dict, Path("weights/mlp"), mesh_device)

# Stage 2: Generate operator configs (returns nested dicts with TTNN objects)
model_config = MLP.prefill_model_config(hf_config, mesh_device) # Or decode_model_config(hf_config, mesh_device) for decode

# Stage 3: Generate the runtime state of the model
model_state = MLP.create_state(hf_config, mesh_device)

# Stage 3: Runtime execution
run_config = MLP.run_config(model_config, weight_config, model_state)
output = MLP.forward_prefill(input_tensor, run_config) # or forward_decode(input_tensor, run_config)
```

## Details
###  Folder Contents
- [reference](./reference): Reference model code from HuggingFace, cleaned up and extracted submodule code etc.
- [tests](./tests): pytests for submodules
- [tt](./tt): ttnn submodule code

## VLLM Server (TT backend)

To run the DeepSeek-V3 model via vLLM with the TT device backend, configure the environment and launch the server as shown below. Adjust the paths for your workspace.

```bash
export TT_METAL_HOME=<path to tt metal home>
export VLLM_DIR=<path to vllm dir>
export PYTHON_ENV_DIR=$TT_METAL_HOME/build/python_env_vllm

source $VLLM_DIR/tt_metal/setup-metal.sh
source $PYTHON_ENV_DIR/bin/activate

export VLLM_TARGET_DEVICE="tt"
export ARCH_NAME=wormhole_b0
export HF_HOME=<hugging face home directory>
export HF_MODEL="deepseek-ai/DeepSeek-R1-0528"
export DEEPSEEK_V3_CACHE=<deepseek v3 cache dir>
export DEEPSEEK_V3_HF_MODEL=<deepseek v3 model dir>
export HF_TOKEN=<HF token>
```

Launch the server with long-lived RPC settings and TT mesh sizing:

```bash
VLLM_RPC_TIMEOUT=1000000 \
MESH_DEVICE="(4,8)" \
python examples/server_example_tt.py \
  --model "deepseek-ai/DeepSeek-R1-0528" \
  --max_model_len 1024 \
  --block_size 32 \
  --override_tt_config '{"trace_mode": false}'
```

In another terminal, send a client request:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{ "model": "deepseek-ai/DeepSeek-R1-0528", "prompt": "San Francisco is a", "max_tokens": 32, "temperature": 0, "top_p": 0.9, "top_k": 10 }'
```
