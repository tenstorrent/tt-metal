# Llama3 TG

## Introduction

Llama 3 is a state-of-the-art large language model (LLM) developed for high-performance natural language processing tasks, including text generation, summarization, and question answering. This repository provides a comprehensive demo for running Llama 3 on Tenstorrent hardware platforms, leveraging advanced features such as paged attention, high batch throughput, and performance benchmarking.

## Supported Platforms

- TG

## Model Architectures

- Llama 3.1-70B
- Llama 3.3-70B

## Key Features

- **Paged Attention**: Efficient memory management for long-context inference.
- **Batch Inference**: Supports up to 32 users per batch.
- **Flexible Sequence Lengths**: Powers of 2 up to 128k tokens.
- **Performance Benchmarking**: Built-in profiler and throughput analysis.
- **Stress Testing**: Long-context and high-throughput stress test modes.
- **Sampling Controls**: Supports temperature, top-p, and top-k sampling.

## Prerequisites

This guide requires the installation / build of `tt-metal`. Please refer to the [installation instructions](/INSTALLING.md) for the release corresponding to [README](/README.md#llms).

## Weights

### Download Llama weights directly from Meta

You can download Llama models [directly from Meta](https://llama.meta.com/llama-downloads/), this will mean accepting their license terms.

The downloaded directories include weight files (e.g. `consolidated.00.pth`), the tokenizer `tokenizer.model` and configuration file `params.json`.

##### Repack weights
Meta's Llama3.1/3.3-70B requires repacked weights. We provide scripts to facilitate this in `models/tt_transformers/scripts/repack_weights_70b.py`.

The repacked output directory can be same as the checkpoint directory, since the new files will have different names.
If providing a different path, please make sure that you keep the string `3.1-70B` or `3.3-70B` in the new path name, since the Llama3 codebase relies on the weights directory name to identify the correct model.

Note: Use the default value of `10` for `chunk_size`.

```
# This concatenates the sharded checkpoints and makes it easier for us to load.
python models/tt_transformers/scripts/repack_weights_70b.py <path_to_checkpoint_dir> <repacked_output_dir>
```

If providing a different output directory, please copy the `params.json` and the `tokenizer.model` files to the new directory.

## Setting the Environment Variables

```
export LLAMA_DIR=<path_to_llama3.1/3.3-70B-instruct>
export TT_METAL_HOME=<path_to_tt_metal>
export PYTHONPATH=<path_to_tt_metal>
export ARCH_NAME=wormhole_b0
export TT_METAL_ENABLE_ERISC_IRAM=1
export FAKE_DEVICE=TG
```

## Running the Demo

### To run the Llama 3 demo:

```
pytest -n auto models/demos/llama3_subdevices/demo/demo_decode.py -k "full"
```

#### Demo Decode Arguments
- **--weights (str)**: Model weights to use (instruct, random, etc.)
- **--layers (int)**: Number of transformer layers (e.g., 1, 10, 80)
- **--input_prompts (str)**: Path to JSON file with input prompts
- **--instruct (bool)**: Use instruct-tuned weights
- **--repeat_batches (int)**: Number of consecutive batches to run
- **--max_seq_len (int)**: Maximum context length (up to 128k)
- **--batch_size (int)**: Number of users per batch (1, 2, 4, 8, 16, 32)
- **--max_generated_tokens (int)**: Max tokens to generate per user
- **--paged_attention (bool)**: Enable paged attention
- **--page_params (dict)**: Paged attention parameters (page_block_size, page_max_num_blocks)
- **--sampling_params (dict)**: Sampling parameters (temperature, top_p, top_k, seed)
- **--stress_test (bool)**: Enable stress test mode
- **--start_pos (int)**: Start position for decoding
- **--optimizations (str)**: Optimization level (performance, accuracy)

### To run the text demo:

```
pytest -n auto models/demos/llama3_subdevices/demo/text_demo.py -k "repeat"
```

#### Text Demo Arguments

- **--input_prompts (str)**: Input JSON file with prompts to process.
- **--instruct (bool)**: Whether to use instruct-tuned weights or general weights.
- **--repeat_batches (int)**: Number of consecutive batches of users to run (default: 1).
- **--max_seq_len (int)**: Maximum context length supported by the model (up to 128k for Llama3.1/3.2).
- **--batch_size (int)**: Number of users per batch (supports: 1, 2, 4, 8, 16, 32).
- **--max_generated_tokens (int)**: Maximum number of tokens to generate per user (stops earlier if EoS token is reached).
- **--paged_attention (bool)**: Whether to use paged attention (required for long contexts and vLLM compatibility).
- **--page_params (dict)**: Parameters for paged attention `{block_size, max_num_blocks}`. Use smaller values (e.g., 32/1024) for short contexts; larger (64/2048) for long contexts.
- **--sampling_params (dict)**: Sampling parameters for decoding `{temperature, top_p}`. If `temperature = 0`, uses greedy decoding.
- **--stop_at_eos (bool)**: Whether to stop decoding when the model generates an end-of-sequence (EoS) token.

## Input Prompts

Input prompts should be provided as a JSON file, with each entry containing a prompt and optionally a context and max_length.

```
[
  {
    "prompt": "What is the capital of France?",
    "context": "https://example.com/context.txt",
    "max_length": 2048
  },
  {
    "prompt": "Explain the theory of relativity."
  }
]
```

## Performance Benchmarking

The demo includes built-in profiling and throughput analysis:

- **Tokens/sec/user**: Measured at each iteration and summarized at the end.
- **TSU Thresholds**: Configurable per model/layer count; demo will assert if throughput falls below target.
- **Stress Testing**: Run with long context and high token generation to validate stability.
