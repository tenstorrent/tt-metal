# Llama3 TG

## Introduction

Llama 3 is a state-of-the-art large language model (LLM) developed for high-performance natural language processing tasks, including text generation, summarization, and question answering. This repository provides a comprehensive demo for running Llama 3 on Tenstorrent hardware platforms, leveraging advanced features such as paged attention, high batch throughput, and performance benchmarking.

The demo supports a variety of configurations, including different model sizes, batch sizes, and sequence lengths, and is designed to facilitate both research and production benchmarking.

## Supported Platforms

- TG

## Model Architectures

- Llama 3.1-70B
- Llama 3.3-70B

## Key Features

- **Paged Attention**: Efficient memory management for long-context inference.
- **Batch Inference**: Supports up to 32 users per batch.
- **Flexible Sequence Lengths**: Up to 128k tokens.
- **Performance Benchmarking**: Built-in profiler and throughput analysis.
- **Stress Testing**: Long-context and high-throughput stress test modes.
- **Sampling Controls**: Supports temperature, top-p, and top-k sampling.

## Weights

The Llama 3 demo supports two methods for loading model weights: Meta-style local checkpoints and Hugging Face-hosted models.

### Meta Checkpoints

To use a local Meta-style checkpoint, set the following environment variable:
```
export LLAMA_DIR=/path/to/llama3.1-70B-instruct
```

### Hugging Face Models (HF_MODEL)
To load weights and tokenizer files from Hugging Face, set:
```
export HF_MODEL=meta-llama/Llama-3.1-70B-Instruct
```

**Note**: Only one of LLAMA_DIR or HF_MODEL should be set.

## Running the Demo

To run the Llama 3 demo:

```
pytest models/demos/llama3_subdevices/demo/demo.py::test_llama_demo
```

**Example: Full Demo (Batch 32, 80 Layers)**

```
pytest models/demos/llama3_subdevices/demo/demo_decode.py::test_llama_demo --batch_size=32 --layers=80 --input_prompts=models/demos/llama3_subdevices/demo/input_data_questions_prefill_128.json
```

**Example: Quick 1-Layer Demo**

```
pytest models/demos/llama3_subdevices/demo/demo_decode.py::test_llama_demo --batch_size=32 --layers=1 --input_prompts=models/demos/llama3_subdevices/demo/input_data_questions_prefill_128.json
```

**Example: Stress Test**

```
pytest models/demos/llama3_subdevices/demo/demo_decode.py::test_llama_demo --batch_size=32 --layers=80 --max_generated_tokens=500000 --stress_test=True
```

## Command-Line Parameters
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
- **--FAKE_DEVICE (str)**: Emulate device for testing (TG)

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
