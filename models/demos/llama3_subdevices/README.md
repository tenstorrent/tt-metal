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

**⚠️ Warning**
>
> Weights downloaded from the `huggingface-cli` via
>```
>huggingface-cli download meta-llama/Meta-Llama-3-70B-Instruct --include "original/*" --local-dir Meta-Llama-3-70B-Instruct
>```
> will be in the same format as a direct download from Meta (i.e. as `consolidated.xx.pth` files). Hence, you will still need to repack your weights and export `LLAMA_DIR` as before. This is contrary to if you downloaded your weights directly from `huggingface`, as those weights will be downloaded as sharded `.safetensors` files.

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
pytest models/demos/llama3_subdevices/demo/demo_decode.py -k "full"
```

#### Demo Decode Arguments
- **weights (str)**: Model weights to use (instruct, random, etc.)
- **layers (int)**: Number of transformer layers (e.g., 1, 10, 80)
- **input_prompts (str)**: Path to JSON file with input prompts
- **instruct (bool)**: Use instruct-tuned weights
- **repeat_batches (int)**: Number of consecutive batches to run
- **max_seq_len (int)**: Maximum context length (up to 128k)
- **batch_size (int)**: Number of users per batch (1, 2, 4, 8, 16, 32)
- **max_generated_tokens (int)**: Max tokens to generate per user
- **paged_attention (bool)**: Enable paged attention
- **page_params (dict)**: Paged attention parameters (page_block_size, page_max_num_blocks)
- **sampling_params (dict)**: Sampling parameters (temperature, top_p, top_k, seed)
- **stress_test (bool)**: Enable stress test mode
- **start_pos (int)**: Start position for decoding
- **optimizations (str)**: Optimization level (performance, accuracy)

### To run the text demo:

```
pytest models/demos/llama3_subdevices/demo/text_demo.py -k "repeat2"
```

#### Text Demo Arguments

- **input_prompts (str)**: Input JSON file with prompts to process.
- **instruct (bool)**: Whether to use instruct-tuned weights or general weights.
- **repeat_batches (int)**: Number of consecutive batches of users to run (default: 1).
- **max_seq_len (int)**: Maximum context length supported by the model (up to 128k for Llama3.1/3.2).
- **batch_size (int)**: Number of users per batch (supports: 1, 2, 4, 8, 16, 32).
- **max_generated_tokens (int)**: Maximum number of tokens to generate per user (stops earlier if EoS token is reached).
- **paged_attention (bool)**: Whether to use paged attention (required for long contexts and vLLM compatibility).
- **page_params (dict)**: Parameters for paged attention `{block_size, max_num_blocks}`. Use smaller values (e.g., 32/1024) for short contexts; larger (64/2048) for long contexts.
- **sampling_params (dict)**: Sampling parameters for decoding `{temperature, top_p}`. If `temperature = 0`, uses greedy decoding.
- **stop_at_eos (bool)**: Whether to stop decoding when the model generates an end-of-sequence (EoS) token.

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

## Serving the model from vLLM

1. **Ensure that tt-metal is installed and set up correctly. Optional check: `python -c "import tt_lib"`.**

2. **Install vLLM**

    ```bash
    # Installing from within `tt-metal`
    git clone https://github.com/tenstorrent/vllm.git
    cd vllm
    git checkout dev
    VLLM_USE_PRECOMPILED=1 pip install -e .
    ```

3. **Ensure weights are downloaded and repacked as described above, and that the environment variables are set.**

    ```bash
    export VLLM_TARGET_DEVICE="tt"
    export MESH_DEVICE=TG
    export TT_LLAMA_TEXT_VER="llama3_subdevices"
    export PYTHONPATH=<path_to_tt_metal>:<path_to_vllm>:$PYTHONPATH
    ```

4. **Running the server**

    ```bash
    python examples/server_example_tt.py
    ```

5. **Interact with server**

    In a separate terminal window, run:

    ```bash
    curl http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "meta-llama/Meta-Llama-3.1-70B",
            "prompt": "Write a poem about RISC-V",
            "max_tokens": 128,
            "temperature": 1,
            "top_p": 0.9,
            "top_k": 10,
            "stream": false
        }'
    ```
This codebase includes Llama3.1-70B on TG.

## Debugging
### Mixing topologies in prefill ccl ops
When running `text_demo.py` on a machine with torus, all ops will by default use ring topology. To use line implementation of ops you can set enviroment variables:
- LINE_RS = 1: to use line for all ReduceScatter ops
- LINE_AG = 1: use line for all AllGather ops

To use line for only some of the AG ops, you can set USE_LINE_AG set in `llama_ccl.py`, for example to use line for all RS and just QKV AG, and ring for the rest of AG set:
- LINE_RS = 1
- LINE_AG = 0
- USE_LINE_AG = {"QKV"}
