# Llama3-70B Demo

## How to Run

1. **Download the Llama3-70B weights from Meta (https://llama.meta.com/):**

2. **Repack the weights:**
    ```bash
    # This concatenates the sharded checkpoints and makes it easier for us to load.
    python models/demos/t3000/llama2_70b/scripts/repack_weights.py <path_to_checkpoint_dir> <repacked_output_dir>
    ```

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
    pytest -svv models/demos/t3000/llama3_70b/demo/demo.py::test_LlamaModel_demo[wormhole_b0-True-check_disabled-sampling-tt-70b-T3000-80L-decode_only-text_completion-llama3]
    ```

4. **Run the performance test:**

    The above demo does not achieve peak performance because we log outputs to the screen. The following perf test will print an accurate end-to-end throughput number.
    For best performance numbers, we recommend building `tt-metal` with `CONFIG=Release` env var, and ensuring the host's CPU governors are set to `performance`.
    ```bash
    pytest -svv models/demos/t3000/llama2_70b/tests/test_llama_perf_decode.py::test_Llama_perf_host[wormhole_b0-True-gen128-llama3]
    ```

## Details

- **Batch Size:** Supports batch size 32.
- **Input File:** Uses `./demo/data/multi_prompt.json`.
- **Model Configuration:** Utilizes a pretrained model.
- **Hardware Requirements:** Runs on an 8-chip T3000 machine using tensor parallelism. The host machine must have at least 512 GB of memory.
- **Model Functionality:** Implements decode-to-prefill strategy, where prompts are processed token-by-token to produce KV caches, followed by token generation in decode mode.

Ensure you follow these guidelines to successfully run the Llama2-70B demo.
