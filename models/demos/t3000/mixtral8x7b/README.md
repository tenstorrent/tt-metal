# Mixtral8x7b Demo

## How to Run

### Download the weights and repack

1. Download the weights from Huggingface.
- [Instruct weights](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
- [General weights](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)

2. Repack the weights using the provided script in `models/demos/t3000/mixtral8x7b/scripts/repack_weights.py`. It requires the consolidated weights from Huggingface to be inside `<path_to_checkpoint_dir>`.

```
# This separates the 8 experts to facilitate sending them to multiple devices.
python models/demos/t3000/mixtral8x7b/scripts/repack_weights.py <path_to_checkpoint_dir> <repacked_output_dir>
```

### Set up environment
1. Set async and dispatch over ethernet cores env vars:
```
export TT_METAL_ASYNC_DEVICE_QUEUE=1
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
```

2. Prepare the weight cache directory:

```
# Make a directory for ttnn to cache weights into. This speeds up subsequent runs.
mkdir <weight_cache_dir>
```

3. Set up environment variables:
```
export MIXTRAL_CKPT_DIR=<repacked_output_dir>
export MIXTRAL_TOKENIZER_PATH=<path_to_tokenizer_dir>
export MIXTRAL_CACHE_PATH=<weights_cache_dir>
```

Note that the cached weights folder structure will contain the general and instruct cached weights in separate directories, like so:

```
<weights_cache_dir>
  /mixtral_tensor_cache_bfp8
  /mixtral_tensor_cache_instruct_bfp8
  ...
```

4. Cache the weights (first-time setup):
```
# Build a full 32 layer model to cache the weights. This will take some time.
pytest -svv models/demos/t3000/mixtral8x7b/tests/test_mixtral_model.py::test_mixtral_model_inference[wormhole_b0-True-1-32-output]
```

### Run the demo
Mixtral prefill support is now available. We include two different demos: a decode-only mode where the prompts are decoded token-by-token and force-pushed until the user starts generating; and a prefill&decode mode, where the KV-caches are first prefilled for the prompt length (e.g. 128 tokens) and then decode as normal.

The `demo_with_prefill.py` supports context lengths up to 32k tokens. Since this is the upper limit to fit on devices, we cap the demo to prefill up to 16k tokens, and then prefill-as-decode if necessary.

To facilitate this, we provide a very large input with over 100k tokens (the majority of the novel `Tale of Two Cities`, with a question asking which book the excerpt comes from). Inside `tt/mixtral_common.py` we slice this large input to 16k tokens, which will then be prefilled normally. This avoids going over the 32k limit stated above.

- For context lengths higher than 16k tokens, we support a maximum batch size of 4.
- For context lengths higher than 8k tokens, we support a maximum batch size of 8.
- For context lengths higher than 4k tokens, we support a maximum batch size of 16.
- For context lenghts below 4k tokens, we support a maximum batch size of 32.

```
# Run the demo with a pre-written batch of 32 user prompts

# Prefill & Decode demo
pytest -svv models/demos/t3000/mixtral8x7b/demo/demo_with_prefill.py::test_mixtral8x7b_demo[wormhole_b0-True-16k-general]

# Decode-only demo
pytest -svv models/demos/t3000/mixtral8x7b/demo/demo.py::test_mixtral8x7b_demo[wormhole_b0-True-general]
```

We also provide an input file with 32 user question-prompt for instruct weights (don't forget to update your flags to the correct weights!):
```
# Prefill & Decode demo
pytest -svv models/demos/t3000/mixtral8x7b/demo/demo_with_prefill.py::test_mixtral8x7b_demo[wormhole_b0-True-16k-instruct]

# Decode-only demo
pytest -svv models/demos/t3000/mixtral8x7b/demo/demo.py::test_mixtral8x7b_demo[wormhole_b0-True-instruct]
```

All input files are provided inside `models/demos/t3000/mixtral8x7b/demo/`.

If you wish you to run the model using a different set of input prompts you can provide a different path to pytest inside the demo code. Keep in mind that for the instruct-mode, the prompts are automatically prefixed and suffixed by `[INST]` and `[/INST]`, respectively, so there's no need to add them to your file.
