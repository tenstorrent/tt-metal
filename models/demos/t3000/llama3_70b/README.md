# Llama3/3.1-70B Demo

## Table of Contents

- [One command run](#one-command-run)
- [How to Run](#how-to-run)
  - [Running the demo from TT-Metalium](#running-the-demo-from-tt-metalium)
  - [Serving the model from vLLM](#serving-the-model-from-vllm)

## One command run

```bash
chmod +x ./models/demos/t3000/llama3_70b/setup_llama.sh && ./models/demos/t3000/llama3_70b/setup_llama.sh <MODEL_TYPE> <TT_METAL_COMMIT_SHA_OR_TAG> <TT_VLLM_COMMIT_SHA_OR_TAG>
```

Where, `TT_METAL_COMMIT_SHA_OR_TAG` and `TT_VLLM_COMMIT_SHA_OR_TAG` are found in the root [README](/README.md#llms) under "Release" version, respectively.

Example:

```bash
./models/demos/t3000/llama3_70b/setup_llama.sh llama-3.1-70b-instruct v0.54.0-rc2 953161188c50f10da95a88ab305e23977ebd3750
```

Follow prompts as they come up in CLI to select appropriate weights for Llama 3.1 70B Instruct.

Prerequisites:

- Submit request to access weights from Meta: [Llama Downloads](https://www.llama.com/llama-downloads)
- Submit permissions on HuggingFace and have a HF personal access token: [Llama 3.1 70B Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct)

Steps run:

- Setup environment
- Build `tt-metal`
- Download Llama 3.1 70B Instruct weights
- Install vLLM
- Deploy vLLM server

## How to Run

Note: This guide requires the installation / build of `tt-metal`. Please refer to the [installation instructions](/INSTALLING.md) for the release corresponding to [README](/README.md#llms).

1. **Download the Llama3/3.1-70B weights from Meta (<https://llama.meta.com/>):**

2. **Repack the weights:**

    ```bash
    # This concatenates the sharded checkpoints and makes it easier for us to load.
    python models/demos/t3000/llama2_70b/scripts/repack_weights.py <path_to_checkpoint_dir> <repacked_output_dir> <chunk_size>
    ```

    Note: Use `5` for `chunk_size`.

    Once the weights are repacked, move the `params.json` file from the `checkpoint_dir` to the `repacked_output_dir`.

### Running the demo from TT-Metalium

After setting up the repacked weights and tokenizer, you can run the demo using the commands below:

1. **Prepare the weight cache directory:**

    ```bash
    # Make a directory for us to cache weights into. This speeds up subsequent runs.
    mkdir <weight_cache_dir>
    ```

2. **Set up environment variables:**

    ```bash
    export LLAMA3_CKPT_DIR=<repacked_output_dir>
    export LLAMA3_TOKENIZER_PATH=<path_to_checkpoint_dir>/tokenizer.model  # Path needs to include the tokenizer.model file
    export LLAMA3_CACHE_PATH=<weight_cache_dir>

    export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
    export TIKTOKEN_CACHE_DIR=""

    pip install -r models/demos/t3000/llama2_70b/reference/llama/requirements.txt

    # Example:
    # export LLAMA3_CKPT_DIR="/home/llama-data-repacked/llama-3-70b/"
    # export LLAMA3_TOKENIZER_PATH="/home/llama-data-repacked/tokenizer.model"
    # export LLAMA3_CACHE_PATH="/home/llama-data-cache/weights-cache"
    ```

3. **Run the demo:**

    Note: Run the following command twice.
    1. The first run will cache the weights. This will take some time.
    2. The second run will use the cached weights, thereby running much faster.

    ```bash
    # Run the demo using sampling decode
    pytest -svv models/demos/t3000/llama3_70b/demo/demo.py::test_LlamaModel_demo[wormhole_b0-True-device_params0-short_context-check_disabled-sampling-tt-70b-T3000-80L-decode_only-trace_mode_on-text_completion-llama3]
    ```

4. **Run the performance test:**

    The above demo does not achieve peak performance because we log outputs to the screen. The following perf test will print an accurate end-to-end throughput number.
    For best performance, ensure that tt-metal is built in release mode (default), and ensure the host's CPU frequency governors are set to `performance` -- instructions for setting the frequency governor vary by machine.
    This performance test runs with sequence length 128 and batch size 32.

    ```bash
    pytest -svv models/demos/t3000/llama2_70b/tests/test_llama_perf_decode.py::test_Llama_perf_host[wormhole_b0-True-device_params0-gen128-llama3]
    ```

#### Details

Supported context lengths and batch sizes for the Llama3.1-70B demo are as follows:

| Context Length | Max Batch Size |
|----------------|----------------|
| 2k             | 32             |
| 8k             | 16             |
| 128k           | 1              |

- **Input File:** Uses `./demo/data/multi_prompt.json`.
- **Model Configuration:** Utilizes a pretrained model.
- **Hardware Requirements:** Runs on an 8-chip T3000 machine using tensor parallelism. The host machine must have at least 512 GB of memory.
- **Demo arguments:**
  - `context: [short_context, long_context, 128k_context]`: Select between short context (batch 32, sequence_length 2k) and long context (batch 16, sequence length 8k) and full context (batch 1, sequence length 128k)
  - `ground_truth: [check_disabled, check_enabled]`: Enable or disable ground truth checking, used for testing
  - `sampling: [greedy, sampling]`: Select between greedy decoding and top-k/top-p sampling
  - `implementation: [tt-70b-T3000]`: Run the 70B model on the Tenstorrent backend
  - `num_layers: [1L, 2L, 10L, 80L]`: Select 80L to run the full model
  - `decode_only: [decode_only, prefill_decode]`: Use `prefill_decode`. Alternately, `decode_only` implements prefill via decode.
  - `chat: [text_completion, chat_completion]`: Run in text_completion mode for the pretrained model or chat_completion for the finetuned model
  - `llama_version: [llama3, llama2]`: Select the Llama3 model

Ensure you follow these guidelines to successfully run the Llama3-70B demo.

### Serving the model from vLLM

1. Complete Step 1 and Step 2 of [Running the Demo from TT-Metalium](#running-the-demo-from-tt-metalium)

2. **Install vLLM**

    ```bash
    # Installing from within `tt-metal`
    export VLLM_TARGET_DEVICE="tt"
    git clone https://github.com/tenstorrent/vllm.git
    cd vllm
    git checkout TT_VLLM_COMMIT_SHA_OR_TAG
    pip install -e .
    cd ..
    ```

    > **Note:** TT_VLLM_COMMIT_SHA_OR_TAG is the vLLM Release version from [README](/README.md#llms)

3. **Running the server**

    ```bash
    python vllm/examples/server_example_tt.py
    ```

4. **Interact with server**

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
