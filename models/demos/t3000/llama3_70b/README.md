# Llama3-70B Demo

## How to Run

1. **Download the Llama3-70B weights from Meta (https://llama.meta.com/):**

2. **Repack the weights:**
    ```bash
    # This concatenates the sharded checkpoints and makes it easier for us to load.
    python models/demos/t3000/llama2_70b/scripts/repack_weights.py <path_to_checkpoint_dir> <repacked_output_dir> <chunk_size>
    ```
    Note: Use `5` for `chunk_size`.

    Once the weights are repacked, move the `params.json` file from the `checkpoint_dir` to the `repacked_output_dir`.

### Running the Demo

After setting up the repacked weights and tokenizer, you can run the demo using the commands below:

1. **Prepare the weight cache directory:**
    ```bash
    # Make a directory for us to cache weights into. This speeds up subsequent runs.
    mkdir <weight_cache_dir>
    ```

2. **Set up environment variables:**
    ```bash
    export LLAMA3_CKPT_DIR=<repacked_output_dir>
    export LLAMA3_TOKENIZER_PATH=<path_to_checkpoint_dir> # Path needs to include the tokenizer.model file
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

    NOTE: Run the following comand twice.
    1. The first run will cache the weights. This will take some time.
    2. The second run will use the cached weights, thereby running much faster.

    ```bash
    # Run the demo using sampling decode
    pytest --co models/demos/t3000/llama3_70b/demo/demo.py::test_LlamaModel_demo[wormhole_b0-True-device_params0-short_context-check_disabled-sampling-tt-70b-T3000-80L-decode_only-trace_mode_on-text_completion-llama3]
    ```

4. **Run the performance test:**

    The above demo does not achieve peak performance because we log outputs to the screen. The following perf test will print an accurate end-to-end throughput number.
    For best performance numbers, we recommend building `tt-metal` with `CONFIG=Release` env var, and ensuring the host's CPU frequency governors are set to `performance`.
    ```bash
    pytest -svv models/demos/t3000/llama2_70b/tests/test_llama_perf_decode.py::test_Llama_perf_host[wormhole_b0-True-device_params0-gen128-llama3]
    ```

## Details

- **Batch Size:** Supports batch size 16 and 32.
- **Input File:** Uses `./demo/data/multi_prompt.json`.
- **Model Configuration:** Utilizes a pretrained model.
- **Hardware Requirements:** Runs on an 8-chip T3000 machine using tensor parallelism. The host machine must have at least 512 GB of memory.
- **Model Functionality:**
    - The demo can run in `decode_only` mode in which we use decode mode to consume the context one token at a time, or `prefill_decode` mode in which we prefill the context and then decode.

- **Demo arguments:**
    - `context: [short_context, long_context]`: Select between short context (batch 32, sequence_length 2k) and long context (batch 16, sequence length 8k)
    - `ground_truth: [check_disabled, check_enabled]`: Enable or disable ground truth checking, used for testing
    - `sampling: [greedy, sampling]`: Select between greedy decoding and top-k/top-p sampling
    - `implementation: [tt-70b-T3000]`: Run the 70B model on the Tenstorrent backend
    - `num_layers: [1L, 2L, 10L, 80L]`: Select 80L to run the full model
    - `decode_only: [decode_only, prefill_decode]`: Use `decode_only`. Alternately, choose `prefill_decode` to enable prefill-decode mode
    - `chat: [text_completion, chat_completion]`: Run in text_completion mode for the pretrained model or chat_completion for the finetuned model
    - `llama_version: [llama3]`: Select the Llama3 model

Ensure you follow these guidelines to successfully run the Llama3-70B demo.
