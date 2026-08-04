# Grok-1 Demo

## How to Run

### Download the weights and repack

1. Download the PyTorch version of the weights from Huggingface.
- [General weights](https://huggingface.co/hpcai-tech/grok-1)

2. Repack the weights using the provided script in `models/experimental/grok/scripts/repack_weights.py`. It requires the consolidated weights from Huggingface to be inside `<path_to_checkpoint_dir>`.

```
# This separates the 8 experts to facilitate sending them to multiple devices.
python models/experimental/grok/scripts/repack_weights.py <path_to_checkpoint_dir> <repacked_output_dir>
```

### Set up environment
1. Set async env var:
```
export TT_METAL_ASYNC_DEVICE_QUEUE=1
```

2. Prepare the weight cache directory:

```
# Make a directory for ttnn us to cache weights into. This speeds up subsequent runs.
mkdir <weight_cache_dir>
```

3. Set up environment variables:
```
export GROK_CKPT_DIR=<repacked_output_dir>
export GROK_TOKENIZER_PATH=<path_to_tokenizer_dir>
export GROK_CACHE_PATH=<weights_cache_dir>
```

Note that the cached weights folder structure will contain the general and instruct cached weights in separate directories, like so:

```
<weights_cache_dir>
  /grok_tensor_cache_bfp8
  /grok_tensor_cache_instruct_bfp8
  ...
```

4. Cache the weights (first-time setup):
```
# Build a full 32 layer model to cache the weights. This will take some time.
pytest -svv models/experimental/grok/tests/test_grok_model.py::test_grok_model_inference[wormhole_b0-True-1-32-output]
```

### Run the demo
```
# Run the demo with a pre-written batch of 32 user prompts
pytest -svv models/experimental/grok/demo/demo.py::test_grok_demo[wormhole_b0-True-general_weights]
```

Input files are provided inside `models/experimental/grok/demo/`.

If you wish you to run the model using a different set of input prompts you can provide a different path to pytest inside the demo code. Keep in mind that for the instruct-mode, the prompts are automatically prefixed and suffixed by `[INST]` and `[/INST]`, respectively, so there's no need to add them to your file.
