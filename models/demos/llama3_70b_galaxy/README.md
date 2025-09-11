# Llama3.3-70

## Platforms:
    Galaxy (WH)

## Introduction
This version of LLama-3.3-70B is tuned for inference performance, achieving competitive prefill and decode times on Wormhole Galaxy Systems.

Read more about this model at the huggingface page for [Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct).

### Key Features
- **Paged Attention**: Efficient memory management for long-context inference
- **Batch Inference**: Supports up to 32 users per batch
- **Flexible Sequence Lengths**: Up to 128k tokens
- **Performance Benchmarking**: Built-in profiler and throughput analysis
- **Sampling Controls**: Supports temperature, top-p, and top-k sampling
- **vLLM-compatible**: Can be used with as an inferencer server engine

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
- Accepting meta license terms

## How to Run
### Download Llama-3.3-70B weights
#### Option 1: From Meta or Huggingface-cli
You can download llama models [directly from Meta](https://www.llama.com/llama-downloads/), or through the `huggingface-cli` via:
```
huggingface-cli download meta-llama/Meta-Llama-3-70B-Instruct --include "original/*" --local-dir Meta-Llama-3-70B-Instruct
```
- **The downloaded weights directories** include weight files (e.g. `consolidated.00.pth`), the tokenizer `tokenizer.model` and configuration file `params.json`.

Then set the following environment variable:
```
export LLAMA_DIR=<path_to_Llama-3.3-70B-instruct>
```

#### Option 2: Straight from Huggingface
If you get the weights [directly from huggingface](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) they will be `.safetensors` files instead. These are supported by setting one of the following environment variables:
```
# For automatic download
export HF_MODEL=meta-llama/Llama-3.3-70B-Instruct
```
```
# If you manually downloaded the weight files
export HF_MODEL=<PATH_TO_HF_WEIGHTS>
```

### Running the Demo
The full Llama-3.3-70B demo can be run with the following command:
```
pytest tt-metal/models/demos/llama3_70b_galaxy/demo/text_demo.py -k "performance-batch-32"
```
- The above run command will execute a short prompt file with 32 users each with up to 128 tokens in length. It will prefill said users and execute 128 decode iterations, i.e. it will generate 128 new tokens.

To run different input prompt files, try these parametrized demo pre-configs:
- `performance-long-4k-b1`, # 4k input prompt context for 1 user
- `performance-long-8k-b1`, # 8k input prompt context for 1 user
- `performance-long-16k-b32`, # 16K input prompt context for 32 users
- `performance-long-32k-b1`, # 32k input prompt context for 1 user
- `performance-long-64k-b1`, # 64k input prompt context for 1 user
- `performance-long-128k-b1`, # 64k input prompt context for 1 user

We also provide other input prompt files with longer sequence lengths. They can be found at `models/demos/llama3_70b_galaxy/demo/sample_prompts/`.

## Testing
### Dev-only and debugging
#### Decode-only Demo
We also provide a decode-only demo. This demo will run prefill as decode and is intended for developers actively working on the decode side of the model. It can be run with the command:
```
pytest models/demos/llama3_70b_galaxy/demo/demo_decode.py -k "full"
```

It supports the following parameters:
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

#### Mixing topologies in prefill ccl ops [Debug-Only]
Please note that using line topology on a galaxy system might affect model accuracy. This functionality is for debug purposes only!

When running `text_demo.py` on a machine with torus, all ops will by default use ring topology. To use line implementation of ops you can set enviroment variables:

- LINE_RS = 1: to use line for all ReduceScatter ops
- LINE_AG = 1: use line for all AllGather ops

To use line for only some of the AG ops, you can set USE_LINE_AG set in `llama_ccl.py`, for example to use line for all RS and just QKV AG, and ring for the rest of AG set:

- LINE_RS = 1
- LINE_AG = 0
- USE_LINE_AG = {"QKV"}

Please note that when using line CCL implementations the maximum sequence length we have validated is 16K tokens. This should just be used for debugging purposes!

## Details
### Demo parameters
- All of the parametrized input prompt files will run prefill (up to their specified sequence lengths) and run 128 decode iterations.
- We also support any arbitrary sequence input sequence length up to 128K tokens. You can test with your own input prompts. Check the next section for more information on the input prompt file format.

- Below is a list of all the parameters that can be configure in the demo. For convenience you can override most of these by adding it's command name to your run.

**Example**: If you want to run text_demo.py -k performance-long-64k-b1 but with a 128K input instead, you would run:
```
text_demo.py -k performance-long-64k-b1 --input_prompts "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_long_128k.json"
```
This would run the same test as 64K but with a 128K input prompt instead.

- **input_prompts (str)**: Input JSON file with prompts to process.
- **instruct (bool)**: Whether to use instruct-tuned weights or general weights.
- **repeat_batches (int)**: Number of consecutive batches of users to run (default: 1).
- **max_seq_len (int)**: Maximum context length supported by the model (up to 128k).
- **batch_size (int)**: Number of users per batch (supports up to 32).
- **max_generated_tokens (int)**: Maximum number of tokens to generate (decode) per user (might stop earlier if EoS token is reached).
- **paged_attention (bool)**: Whether to use paged attention (required for long contexts and vLLM compatibility). On by default.
- **page_params (dict)**: Parameters for paged attention `{block_size, max_num_blocks}`.
- **sampling_params (dict)**: Sampling parameters for decoding `{temperature, top_p}`. If `temperature = 0`, it uses greedy decoding.
- **stop_at_eos (bool)**: Whether to stop decoding when the model generates an end-of-sequence (EoS) token. This is currently hard set to False in `text_demo.py` for testing purposes.
- **apc_test**: [Dev Flag] Runs a specific internal CI test.
- **pcc_check**: [Dev Flag] Enables PCC comparison. To be used by specific internal CI tests.
- **prefill-only profile**: [Dev Flag] Runs prefill only. To be used when measuring prefill OP performance.
- **num_layers**: Specifies how many layers of the model to run. Default = 80.
- **print_outputs**: [Debug Flag] Prints each user's tokens at every decode iteration. Leave False for accurate e2e performance numbers.

### Input Prompts
Input prompts should be provided as a JSON file, with each entry containing a prompt and optionally a context and max_length (equivalent to the number of characters in the prompt, not tokens!). Please refer to the ones already provided in `models/demos/llama3_70b_galaxy/demo/sample_prompts/`.

You can try and change the `max_length` parameter on the provided prompt files to test different input sequence lengths, as long as they don't surpass 128K tokens.
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

### vLLM Model Serving
Ensure first you have a proper TT-Metal installation. (Optional check: `python -c "import tt_lib"`).

vLLM can be install from the TT fork over at https://github.com/tenstorrent/vllm/tree/dev (make sure you're at `dev` branch).

Please follow the [README from vLLM](https://github.com/tenstorrent/vllm/blob/dev/tt_metal/README.md) for the latest instructions on how to build vLLM.

#### Running the vLLM server
To run a vLLM server on a Galaxy system with Llama-3.3-70B you can execute the following command:
```
VLLM_RPC_TIMEOUT=900000 python examples/server_example_tt.py --model "meta-llama/Llama-3.3-70B-Instruct" --override_tt_config '{"dispatch_core_axis": "col", "sample_on_device_mode": "all", "fabric_config": "FABRIC_1D_RING", "worker_l1_size": 1344544, "trace_region_size": 95693824}' --num_scheduler_steps 30
```

After the server is up and running you can interact with it by sending prompt files.

For convenience you can use the official [tt-inference-server](https://github.com/tenstorrent/tt-inference-server/tree/dev) and run the following command:
```
export HF_MODEL_REPO_ID='meta-llama/Llama-3.3-70B-Instruct'

cd tt-inference-server/vllm-tt-metal-llama3/src
python example_requests_client.py --num_concurrent 32 --prompt_json_path "vllm_server_prompts.json"
```

You can find an example server_prompts file in tt-metal at `models/demos/llama3_70b_galaxy/demo/sample_prompts/vllm_server_prompts.json`.
