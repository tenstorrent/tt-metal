# Mistral7B Demo

Demo showcasing Mistral-7B running on Wormhole, using ttnn.

## How to Run

⚠️ **WARNING: DO NOT RUN THIS DEMO ON WORMHOLE FIRMWARE OLDER THAN 80.13.0.0** ⚠️

This demo has been linked to hardware failures that can permanently damage cards. A patch was released for firmware 80.13.0.0. Please do not attempt to run it on anything older.

### Download the weights

Download the weights tarfile directly from Mistral-AI:
- General weights: [Mistral-7B-v0.1](https://models.mistralcdn.com/mistral-7b-v0-1/mistral-7B-v0.1.tar)
- Finetune instruct weights: [Mistral-7B-Instruct-v0.2](https://models.mistralcdn.com/mistral-7b-v0-2/Mistral-7B-v0.2-Instruct.tar)

Both the above tarfiles consolidate the weights into a single file `consolidated.00.pth`. They also contain the tokenizer `tokenizer.model`.

We also include a script to download and untar the weight files inside `models/demos/wormhole/mistral7b/scripts/get_mistral_weights.py`.

```
# Download general weights
python models/demos/wormhole/mistral7b/scripts/get_mistral_weights.py --weights_path=<FOLDER_TO_SAVE_WEIGHTS>

# To download instruct weights add --instruct flag

python models/demos/wormhole/mistral7b/scripts/get_mistral_weights.py --weights_path=<FOLDER_TO_SAVE_WEIGHTS> --instruct
```

### Set up environment

1. Prepare the weight cache directory:

```
# Make a directory for ttnn to cache weights into. This speeds up subsequent runs.
mkdir <weight_cache_dir>
```

2. Set up environment variables:
```
export MISTRAL_CKPT_DIR=<weights_dir>
export MISTRAL_TOKENIZER_PATH=<path_to_tokenizer_dir>
export MISTRAL_CACHE_PATH=<weights_cache_dir>
```

A typical environment will have all the above point to the same folder.

Note that the cached weights folder structure will contain, after being generated, the general and instruct cached weights in separate directories, like so:

```
<weights_cache_dir>
  /mistral_tensor_cache_bfp8
  /mistral_tensor_cache_instruct_bfp8
  ...
```

3. Cache the weights (first-time setup).
If the cached weights have not yet been created the first execution will take care of generating them. You can run the model test for this step:

```
# Build a full 32 layer model to cache the weights. This will take some time (1 time only).
pytest models/demos/wormhole/mistral7b/tests/test_mistral_model.py::test_mistral_model_inference[17-generative]
```

### Run the demo

Mistral-7B runs fast prefill upto sequence lengths of 4096.

For decode-only, the largest context length supported is currently 1024 tokens.

Mistral-7B is running on a single chip. If you are running on a T3000 please set the following: `export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml`

Note that while running the demo you might see the warning: `Op | WARNING  | TILE layout does not have multicore implementation yet. Falling back to 1 core.` This is expected and can be ignored; the demo will run after the warning.

Batches of 1 to 5 are supported. To set the number of batches for the demo, replace `<number_of_batches>` with a value between 1 and 5 in the CLI command.

```
# Run the demo with a pre-written batch of 8 user prompts:

# Prefill & Decode demo
pytest models/demos/wormhole/mistral7b/demo/demo_with_prefill.py::test_mistral7B_demo[general_weights-<number_of_batches>_batch]

# Decode-only demo
pytest models/demos/wormhole/mistral7b/demo/demo.py::test_mistral7B_demo[general_weights-<number_of_batches>_batch]
```

We also provide an input file with 32 user question-prompt for instruct weights (don't forget to update your env flags to the correct instruct weights folder):
```
# Run the demo with a pre-written batch of 8 user question-prompts:

# Prefill & Decode demo
pytest models/demos/wormhole/mistral7b/demo/demo_with_prefill.py::test_mistral7B_demo[instruct_weights-<number_of_batches>_batch]

# Decode-only demo
pytest models/demos/wormhole/mistral7b/demo/demo.py::test_mistral7B_demo[instruct_weights-<number_of_batches>_batch]
```

Both input files are provided inside `models/demos/wormhole/mistral7b/demo/`.

If you wish you to run the model using a different set of input prompts you can provide a different path to pytest inside the demo code. Keep in mind that for the instruct-mode, the prompts are automatically prefixed and suffixed by `[INST]` and `[/INST]`, respectively, so there's no need to add them to your file.
