# Llama-like Models

This code can run Llama3 family of models and other similar models including QwQ, Qwen2.5 and DeepSeek-R1-Distill variants.

The current version is known to support the following Llama3 models:
- Llama3.2-1B
- Llama3.2-3B
- Llama3.1-8B
- Llama3.2-11B
- Llama3.1-70B (T3000 and TG-only)
- Qwen2.5-7B
- Qwen2.5-72B
- QwQ-32B
- DeepSeek R1 Distill Llama 3.3 70B (T3000 and TG-only)

All the above llama models (with the exception of 70B due to its large size) are compatible and tested on the following Tenstorrent hardware:
- N150 (1-chip)
- N300 (2-chips)
- T3000 (8-chips)
- TG (32-chips)

Qwen-7B requires N300
Qwen-72B requires T3K

## How to Run

### Llama models: download the weights

Download the weights [directly from Meta](https://llama.meta.com/llama-downloads/), this will mean accepting their license terms.

The downloaded directories include weight files (e.g. `consolidated.00.pth`), the tokenizer `tokenizer.model` and configuration file `params.json`.

#### Llama3.1-70B only
Llama3.1-70B requires repacked weights. We provide a script to facilitate this in `models/demos/llama3/scripts/repack_weights_70b.py`.

The repacked output directory can be same as the checkpoint directory, since the new files will have different names.
If providing a different path, please make sure that you keep the string `3.1-70B` in the new path name, since the Llama3 codebase relies on the weights directory name to identify the correct model.

Note: Use the default value of `10` for `chunk_size`.

```
# This concatenates the sharded checkpoints and makes it easier for us to load.
python models/demos/llama3/scripts/repack_weights_70b.py <path_to_checkpoint_dir> <repacked_output_dir>
```

If providing a different output directory, please copy the `params.json` and the `tokenizer.model` files to the new directory.

#### Additional package
The library requires extra python dependencies. Install them from:

```
pip install -r models/demos/llama3/requirements.txt
```

### HuggingFace models (e.g. DeepSeek R1 Distill Llama 3.3 70B, Qwen 2.5 7B, ...)

Download the weights from [HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B). Your model directory should have the following structure:

```
DeepSeek-R1-Distill-Llama-70B/
    config.json
    generation_config.json
    model-00001-of-00062.safetensors
    ...
```

#### Running llama-similar models other than DeepSeek R1 Distill and Qwen 2.5

If you are bringing up a new model that is similar to these but is not listed above, you will also need to set additional environment variables:
- `MAX_PREFILL_CHUNK_SIZE` - this determines how many thousands of tokens are prefilled in one go. For optimal performance pick 128. Depending on the model dimensions and hardware you're running on, there may not be enough L1 to prefill 128K tokens at once, in which case you can reduce this in powers of 2 down to 4.
- `PAD_MLP_CORES` - models with a hidden_dim that is not a nice power of 2 may not have a valid layout or may run with lower performance. You can set this to a multiple of 8 between 8 and 64; `16` and `32` commonly work well if this is required.

You should also watch out for:
- RoPE encoding style. `llama3` and of course none are both supported. We have a [branch](https://github.com/tenstorrent/tt-metal/tree/llama-yarn) with `yarn` support in progress.
- Our [accuracy test](tests/test_llama_accuracy.py) will require you to [generate some reference logits](tests/generate_reference_hf.py) and perhaps update the test to use them.
- We parallelise attention over the number of heads. If this number is e.g. 14 then you will not be able to run it on more than 2 chips (because 14/2=7, a prime number). We do not support head-padding or similar mitigations at this time but a PR would be cool.

### Setup TT environment

1. Set up environment variables:
```
export LLAMA_DIR=<model_dir>
```

On N150, N300 and T3K:
```
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
```

- `$LLAMA_DIR` sets the path for the Llama3 model weights and caches.

- `$WH_ARCH_YAML` sets the dispatch over ethernet cores. This is optional for N150 and required for N300 and T3000, enabling a full core grid utilization (8x8), allowing for maximum performance of LLama3 models. Do not set this for TG.

On the first execution of each model, TTNN will create weight cache files for that model, to speed up future runs.
These cache files only need to be created once for each model and each weight (i.e. new finetuned weights will need to be cached) and will be stored accordingly to the machine you are running the models:
```
$LLAMA_DIR/N150  # For N150
$LLAMA_DIR/N300  # For N300
$LLAMA_DIR/T3K   # For T3000
$LLAMA_DIR/TG   # For TG
```

### Run the demo

The Llama3 demo includes 3 main modes of operation and is fully parametrized to support other configurations.

- `batch-1`: Runs a small prompt (128 tokens) for a single user
- `batch-32`: Runs a small prompt (128 tokens) for a a batch of 32 users
- `long-context`: Runs a large prompt (64k tokens) for a single user
- `reasoning-1`: Runs a reasoning prompt for a single user (generates up to 15k tokens)

If you want to provide your own demo configuration, please take a look at the pytest parametrize calls in `models/demos/llama3/demo/simple_text_demo.py`. For convenience we list all the supported params below:

- `input_prompts (string)`: input json file with prompts to process. See `models/demos/llama3/demo/*.json` for a list of input files
- `instruct (bool)`: Whether to use Llama instruct weights or general weights
- `repeat_batches (int)`: Number of consecutive batches of users to run (default: 1)
- `max_seq_len (int)`: Maximum context length supported by the model (refer to the table above)
- `batch_size (int)`: Number of users in a batch (Supports 1/2/4/8/16/32 batches)
- `max_generated_tokens (int)`: Maximum number of tokens to generate for each user (Note that the users will stop generation before this limit if they reach a eos token)
- `paged_attention (bool)`: Whether to use paged attention or default attention (vLLM support (WIP) requires paged attention)
- `page_params (dict)`: Page parameters for paged attention - [`block_size`, `max_num_blocks`]. For smaller context lengths use `block_size=32` and `max_num_blocks=1024`, for larger context use block_size=64 and max_num_blocks=2048
- `sampling_params (dict)`: Sampling parameters for decoding -[`temperature`, `top_p`]. If temperature is set to 0, argmax (greedy decode) is used.
- `stop_at_eos (bool)`: Flag to stop decoding when the model generates an EoS token
- `optimization (LlamaOptimizations)`: Optimization level to use for the model [`performance`, `accuracy`]

Please note that using `argmax` with `batch_size > 1` or using `top-p` sampling with any batch size, these ops will be run on host. This is because those ops are not yet fully supported on device. A decrease in performance is expected when these configurations are enabled.

When running the demo, do not forget to setup the `$LLAMA_DIR` environment variable to the corresponding Llama3 model weights.

Additionally, we also support the use of a fake device. This enables running a smaller chip demo in a larger multichip device.
Supported devices: [`N150`, `N300`, `T3K`, `TG`].

Example: `export FAKE_DEVICE=N150`, will enable running a single-chip demo on a multi-chip system.

```
# Examples of how to run the demo for any supported Llama3 models

# Batch-1
pytest models/demos/llama3/demo/simple_text_demo.py -k "performance and batch-1"

# Batch-32
pytest models/demos/llama3/demo/simple_text_demo.py -k "performance and batch-32"

# Long-context
pytest models/demos/llama3/demo/simple_text_demo.py -k "performance and long"
```

The above examples are run in `LlamaOptimizations.performance` mode.
You can override this by setting the `optimizations` argument in the demo. To use instead the accuracy mode you can call the above tests with `-k "accuracy and ..."` instead of performance.

#### Custom input arguments
To facilitate testing different configurations, `simple_text_demo.py` supports argument overrides. The full list of overrides is included in `models/demos/llama3/demo/conftest.py`.

An example usage where the `batch-1` test is modified to run with 16 users and keep generating tokens until 1024 are generated:

```
pytest models/demos/llama3/demo/simple_text_demo.py -k "performance and batch-1" --batch_size 16 --max_generated_tokens 1024 --stop_at_eos 0
```

### Expected performance and accuracy

See [PERF.md](PERF.md) for expected performance and accuracy across different configurations.

### Implementation notes

**Chunked prefill (text-only)**: All of the compatible model/device combinations support a max prefill context-length of 128k, with the exception of Llama3.1-8B and Llama3.2-11B on N150 which have a max of 64k (due to a lack of memory). To support these large max context-lengths, chunked prefill is performed with different max chunk sizes as shown in the table below.

Max Prefill Chunk Sizes (text-only):
|              |      N150     |      N300     |      T3K       |      TG     |
|--------------|---------------|---------------|----------------|-------------|
| Llama3.2-1B  | 128k tokens   | 128k tokens   | 128k tokens    | 128k tokens |
| Llama3.2-3B  | 8k tokens     | 128k tokens   | 128k tokens    | 128k tokens |
| Llama3.1-8B  | 4k tokens     | 64k tokens    | 128k tokens    | 128k tokens |
| Llama3.2-11B | 4k tokens     | 64k tokens    | 128k tokens    | 128k tokens |
| Llama3.1-70B | Not supported | Not supported | 32k tokens     | 128k tokens |
| DeepSeek-R1-Distill-Llama3.3-70B | Not supported | Not supported | 32k tokens     | 128k tokens |

- These max chunk sizes are specific to max context length 128k and are configured via `MAX_PREFILL_CHUNK_SIZES_DIV1024` in [model_config.py](https://github.com/tenstorrent/tt-metal/blob/main/models/demos/llama3/tt/model_config.py). If the max context length is set to a smaller value using the `max_seq_len` flag (see [Run the demo](#run-the-demo)), these chunk sizes can possibly be increased due to using a smaller KV cache.

**Chunked prefill (Llama3.2-11B multimodal)**: Llama3.2-11B multimodal is currently only supported on N300 and T3000. On N300, a max prefill context length of 8k is supported, while T3000 supports a max context length of 128k.
