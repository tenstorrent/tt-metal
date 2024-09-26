# Llama 3.1 8B Demo

Demo showcasing Llama 3.1 8B running on Wormhole N300 (tensor-parallel on 2 chips), using ttnn.

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
pytest models/demos/wormhole/llama31_8b_N300/tests/test_llama_model.py
```

### Run the demo

Llama 3.1 8B runs fast prefill upto sequence lengths of 16k.

For decode-only, the largest context length supported is 128k tokens.

Llama 3.1 8B is running on a single chip. If you are running on a T3000 please set the following: `export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml`

```
# Run the demo with a pre-written batch of 8 user prompts:

# Prefill & Decode demo
pytest models/demos/wormhole/llama31_8b_N300/demo/demo_with_prefill.py::test_llama_demo[general_weights-1_batch]

# Decode-only demo
pytest models/demos/wormhole/llama31_8b_N300/demo/demo.py::test_llama_demo[general_weights]
```

We also provide an input file with 32 user question-prompt for instruct weights (don't forget to update your env flags to the correct instruct weights folder):
```
# Prefill & Decode demo
pytest models/demos/wormhole/llama31_8b_N300/demo/demo_with_prefill.py::test_llama_demo[instruct_weights-1_batch]

# Decode-only demo
pytest models/demos/wormhole/llama31_8b_N300/demo/demo.py::test_llama_demo[instruct_weights]
```

Both input files are provided inside `models/demos/wormhole/llama31_8b_N300/demo/`.

If you wish you to run the model using a different set of input prompts you can provide a different path to pytest inside the demo code. At this time our code does not add system, user or assistant prompts to the input, so the output should be used for reference only.
