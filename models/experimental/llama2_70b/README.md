# Llama2-70B Demo

This experimental folder contains the latest performance optimizations and newest features, but is not as stable as the `models/demos/llama2_70b` folder.
The following commands will run the Llama2-70B or Llama3-70B demo depending on which weights are provided.

## How to Run

1. **Download the Llama weights from Meta (https://llama.meta.com/):**
    We recommend Llama2-70B or Llama3-70B weights for this demo.

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

2. **Set up environment:**
    Follow the Wormhole [installation instructions](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md).

    Then export either the Llama2-70B or Llama3-70B checkpoint directory, the tokenizer path, and the weight cache directory.
    ```bash
    # Llama2-70B requires the following environment variables:
    export LLAMA2_CKPT_DIR=<repacked_output_dir>
    export LLAMA2_TOKENIZER_PATH=<path_to_checkpoint_dir>
    export LLAMA2_CACHE_PATH=<weight_cache_dir>

    # Llama3-70B requires the following environment variables:
    export LLAMA3_CKPT_DIR=<repacked_output_dir>
    export LLAMA3_TOKENIZER_PATH=<path_to_checkpoint_dir>
    export LLAMA3_CACHE_PATH=<weight_cache_dir>

    export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
    export TIKTOKEN_CACHE_DIR=""

    pip install -r models/experimental/llama2_70b/reference/llama/requirements.txt
    ```

3. **Run the demo:**
    The first run will take quite a while to cache the weights. Weight caching tilizes and converts Llama weights to our internal format, stored in `LLAMA_CACHE_PATH`.
    Subsequent runs will load cached weights much faster.
    ```bash
    # Llama3-70B
    pytest -svv models/experimental/llama2_70b/demo/demo.py::test_LlamaModel_demo[wormhole_b0-True-check_disabled-greedy-tt-70b-T3000-80L-decode_only-text_completion-llama3]

    # Llama2-70B
    pytest -svv models/experimental/llama2_70b/demo/demo.py::test_LlamaModel_demo[wormhole_b0-True-check_disabled-greedy-tt-70b-T3000-80L-decode_only-text_completion-llama2]
    ```

4. **Run the performance test:**
    The above demo does not achieve peak performance because we log outputs to the screen. The following perf test will print an accurate end-to-end throughput number.
    For best performance numbers, we recommend building `tt-metal` with `CONFIG=Release` env var, and ensuring the host's CPU governors are set to `performance`.
    ```bash
    pytest -svv models/experimental/llama2_70b/tests/test_llama_perf_decode.py::test_Llama_perf_host[wormhole_b0-True-gen128]
    ```
## Details

- **Batch Size:** Supports batch size 32.
- **Input File:** Uses `./demo/data/multi_prompt.json`.
- **Model Configuration:** Utilizes a pretrained model.
- **Hardware Requirements:** Runs on an 8-chip T3000 machine using tensor parallelism. The host machine must have at least 512 GB of memory.

Ensure you follow these guidelines to successfully run the Llama2-70B demo.
