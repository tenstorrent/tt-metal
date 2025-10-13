# DeepSeek-V3

## Platforms:
    Galaxy (WH)

## Introduction
This codebase was designed for the model: [deepseek-ai/DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) but should also work with other DeepseekV3 models such as:

- [deepseek-ai/DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)
- [deepseek-ai/DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3)

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

## Demo CLI

Example (replace the placeholder paths with your environment):

```bash
python models/demos/deepseek_v3/demo/demo.py \
  --model-path /abs/path/to/load/hf/deepseek-v3 \
  --cache-dir /abs/path/to/save/ttnn/cache \
  --early_print_first_user \
  "Write a haiku about autumnal days by the sea"
```

Supported arguments:

- `--model-path PATH`: Local HF model directory. Defaults to `$DEEPSEEK_V3_HF_MODEL` or `models/demos/deepseek_v3/reference`.
- `--cache-dir PATH`: Where to store converted TTNN weights and caches. Defaults to `$DEEPSEEK_V3_CACHE` or `generated/deepseek_v3`.
- `--early_print_first_user`: Stream generated tokens for the first prompt as they are produced instead of printing only at the end.
- `--max-new-tokens N`: Number of tokens to generate (default: 32).
- `--random-weights`: Use randomly initialized weights (single dense layer only). Does not require tokenizer or safetensors.
- `--single-layer {mlp,moe}`: With `--random-weights`, request a single layer (mlp supported).
- `--token-accuracy`: Enable teacher‑forcing decode and report accuracy (requires full‑model mode + tokenizer).
- `--reference-file PATH`: Path to `.pt/.refpt` reference file (see below).
- `--tf-prompt-len N`: Override the teacher‑forcing prompt length in tokens (taken from the reference file). Defaults to half+1 if omitted.

You should also provide one or more prompts (each in quotes as in the above example) as positional arguments, unless using `--random-weights`. In `--random-weights` mode, prompts are optional.

Programmatic usage:

```python
from models.demos.deepseek_v3.demo.demo import run_demo

# Full-model generation (prompt required)
run_demo(["Write a haiku about hardware"], model_path="/abs/path/to/deepseek-v3")

# Random-weights smoke test (prompt optional)
run_demo(None, random_weights=True)
```

### Teacher Forcing Accuracy Verification

You can verify accuracy under teacher forcing using a reference file with tokenized ground‑truth. The expected format matches the tt_transformers demos/tests:

- Keys: `reference_tokens` (LongTensor [1, T]) and optional `top5_tokens` (LongTensor [T, 5]).
- The demo splits `reference_tokens` at `T//2 + 1` into input prompt and ground‑truth continuation.

Generate a compatible reference file with the tt_transformers helper (use the same tokenizer/model family as your DeepSeek model to ensure token IDs match):

- Hugging Face path:
  - `python models/tt_transformers/tests/generate_reference_outputs.py --total_length 2048 --output_file models/tt_transformers/tests/reference_outputs/DeepSeek-V3.refpt --model deepseek-ai/DeepSeek-V3`
- Or use the HF‑only variant:
  - `python models/tt_transformers/tests/generate_reference_hf.py --total_length 2048 --output_file models/tt_transformers/tests/reference_outputs/DeepSeek-V3.refpt --model deepseek-ai/DeepSeek-V3`

Run the DeepSeek‑V3 demo with teacher forcing:

- `python models/demos/deepseek_v3/demo/demo.py --model-path /path/to/deepseek-v3 --token-accuracy --reference-file models/tt_transformers/tests/reference_outputs/DeepSeek-V3.refpt --max-new-tokens 256`
  - Optionally control prompt length independently with `--tf-prompt-len`, e.g.:
  - `... --tf-prompt-len 1024 --max-new-tokens 256`

Notes:

- `--token-accuracy` is not compatible with `--random-weights` and requires tokenizer files in `--model-path`.
- The demo decodes a single sequence in teacher‑forcing mode. `--max-new-tokens` is capped to the number of available GT tokens in the reference file.
- If `top5_tokens` is present in the reference, the demo reports both top‑1 and top‑5 accuracies; otherwise, only top‑1.

## How to develop

If you are not running on Tenstorrent internal infrastructure, you need to set the following environment variables:

- `DEEPSEEK_V3_HF_MODEL`: Path to a directory containing the DeepSeek-V3 Hugging Face model weights. Defaults to `models/demos/deepseek_v3/reference`. Download the model from Hugging Face set this to the model directory.
- `DEEPSEEK_V3_CACHE`: Path to a directory where cached data such as converted weights and test inputs/outputs will be stored.


These variables are used in scripts for generating test data and running tests.

This codebase separates model execution into three distinct stages, each of which can be run independently:
1. Convert PyTorch weights to TTNN tensor files and generate the WeightConfig
2. Generate ModelConfigs for prefill and decode modes
3. Load TTNN tensor files using WeightConfig, create a shared state using create_state, merge them with ModelPrefillConfig and ModelDecodeConfig to create a RunPrefillConfig and RunDecodeConfig, and execute the model with either of the model configs

The modules are not instantiated directly, but rather used as a namespace for the methods that define the model's behavior in prefill and decode. This is to make it easy to separate the stateful and stateless parts of the model, and allow for easy re-use of the methods.

### Weight Configuration
Generated by static method `convert_weights` on each module class. Returns a dict mapping operation names to their TTNN weight file paths:
```python
{
    "w1": "/path/to/weights/w1.input_tensor_b",
    "w2": "/path/to/weights/w2.input_tensor_b",
    "w3": "/path/to/weights/w3.input_tensor_b"
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
# Stage 1: Convert weights and get weight_config (saves to disk in standard format)
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
