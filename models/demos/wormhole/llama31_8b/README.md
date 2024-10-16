# Llama 3.1 8B Demo

Demo showcasing Llama 3.1 8B running on Wormhole, using ttnn.

## How to Run

### Download the weights

Download the weights [directly from Meta](https://llama.meta.com/llama-downloads/), this will mean accepting their license terms.

The directory includes a single file `consolidated.00.pth` and tokenizer `tokenizer.model`.

### Set up environment

1. Prepare the weight cache directory:

```
# Make a directory for ttnn to cache weights into. This speeds up subsequent runs.
mkdir <weight_cache_dir>
```

2. Set up environment variables:
```
export LLAMA_CKPT_DIR=<meta_dir>
export LLAMA_TOKENIZER_PATH=<meta_dir>
export LLAMA_CACHE_PATH=<meta_dir>
```

A typical environment will have all the above point to the same folder.

Note that the cached weights folder structure will contain, after being generated, the general and instruct cached weights in separate directories, like so:

```
<weights_cache_dir>
  /llama_tensor_cache_bfp8
  /llama_tensor_cache_instruct_bfp8
  ...
```

3. Cache the weights (first-time setup).
If the cached weights have not yet been created the first execution will take care of generating them. You can run the model test for this step:

```
# Build a full 32 layer model to cache the weights. This will take some time (1 time only).
pytest models/demos/wormhole/llama31_8b/tests/test_llama_model.py
```

### Run the demo

Llama 3.1 8B runs fast prefill upto sequence lengths of 512.

For decode-only, the largest context length supported is currently 1024 tokens.

Llama 3.1 8B is running on a single chip. If you are running on a N300 or T3000 please set the following: `export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml`

Note that while running the demo you might see the warning: `Op | WARNING  | TILE layout does not have multicore implementation yet. Falling back to 1 core.` This is expected and can be ignored; the demo will run after the warnings.

**Update**: The latest demo includes ttnn trace support, achieving a performance of 23 tokens/sec/user.

```
# Run the latest Llama3.1-8B trace demo with general weights
pytest models/demos/wormhole/llama31_8b/demo/demo_trace.py -k "general and 1_batch"

# Run the latest Llama3.1-8B trace demo with instruct weights
pytest models/demos/wormhole/llama31_8b/demo/demo_trace.py -k "instruct and 1_batch"
```

### Older demo scripts
```
# Run the demo with a pre-written batch of 8 user prompts:

# Old Prefill & Decode demo
pytest models/demos/wormhole/llama31_8b/demo/demo_with_prefill.py::test_llama_demo[general_weights-1_batch]

# Old Decode-only demo
pytest models/demos/wormhole/llama31_8b/demo/demo.py::test_llama_demo[general_weights]

# Old Prefill & Decode with continuous batching
pytest models/demos/wormhole/llama31_8b/demo/demo_continuous_batching.py::test_LlamaModel_demo[batch_4-greedy-32L-text_completion-llama3]

# Old Prefill & Decode with continuous batching & paged attention
pytest models/demos/wormhole/llama31_8b/demo/demo_continuous_batching_paged_attention.py::test_LlamaModel_demo[batch_4-greedy-32L-text_completion-llama3]
```

We also provide an input file with 32 user question-prompt for instruct weights (don't forget to update your env flags to the correct instruct weights folder):
```
# Old Prefill & Decode demo
pytest models/demos/wormhole/llama31_8b/demo/demo_with_prefill.py::test_llama_demo[instruct_weights-1_batch]

# Old Decode-only demo
pytest models/demos/wormhole/llama31_8b/demo/demo.py::test_llama_demo[instruct_weights]

# Old Prefill & Decode with continuous batching
pytest models/demos/wormhole/llama31_8b/demo/demo_continuous_batching.py::test_LlamaModel_demo[batch_4-greedy-32L-chat_completion-llama3]

# Old Prefill & Decode with continuous batching & paged attention
pytest models/demos/wormhole/llama31_8b/demo/demo_continuous_batching_paged_attention.py::test_LlamaModel_demo[batch_4-greedy-32L-chat_completion-llama3]
```

Both input files are provided inside `models/demos/wormhole/llama31_8b/demo/`.

If you wish you to run the model using a different set of input prompts you can provide a different path to pytest inside the demo code. At this time our code does not add system, user or assistant prompts to the input, so the output should be used for reference only.
