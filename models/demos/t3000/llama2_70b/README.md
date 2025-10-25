# Llama2-70

## Platforms:
> [!NOTE]
> This model is no longer supported on QuietBox/LoudBox setups.

## Introduction
Read more about llama2_70b at [llama.com/llama2](https://www.llama.com/llama2/).

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
- The host machine must have at least 512 GB of memory.

## How to Run
1. Download the Llama2-70B weights from [Meta](https://llama.meta.com/llama2/):

2. Repack the weights:
    ```bash
    python models/demos/t3000/llama2_70b/scripts/repack_weights.py <path_to_checkpoint_dir> <repacked_output_dir> <chunk_size>
    ```
    Note: Use `5` for `chunk_size`.

3. Once the weights are repacked, move the `params.json` file from the `checkpoint_dir` to the `repacked_output_dir`.

### Running the Demo
After setting up the repacked weights and tokenizer, you can run the demo using the commands below:

1. Prepare the weight cache directory:
    ```bash
    mkdir <weight_cache_dir>
    ```

2. Set up environment variables:
    ```bash
    export LLAMA2_CKPT_DIR=<repacked_output_dir>
    export LLAMA2_TOKENIZER_PATH=<path_to_checkpoint_dir> # Path needs to include the tokenizer.model file
    export LLAMA2_CACHE_PATH=<weight_cache_dir>
    export TIKTOKEN_CACHE_DIR=""

    pip install -r models/demos/t3000/llama2_70b/reference/llama/requirements.txt

    # Example:
    # export LLAMA2_CKPT_DIR="/home/llama-data-repacked/llama-2-70b/"
    # export LLAMA2_TOKENIZER_PATH="/home/llama-data-repacked/tokenizer.model"
    # export LLAMA2_CACHE_PATH="/home/llama-data-cache/weights-cache"
    ```

3. Run the demo:

    NOTE: Run the following comand twice.
    1. The first run will cache the weights. This will take some time.
    2. The second run will use the cached weights, thereby running much faster.

    ```bash
    # Run the demo using sampling decode
    pytest -svv models/demos/t3000/llama2_70b/demo/demo.py::test_LlamaModel_demo[wormhole_b0-True-device_params0-short_context-check_disabled-sampling-tt-70b-T3000-80L-decode_only-trace_mode_on-text_completion-llama2]
    ```

## Testing
- Performance test:

    The above demo does not achieve peak performance because we log outputs to the screen. The following perf test will print an accurate end-to-end throughput number.
    For best performance, ensure that tt-metal is built in release mode (default), and ensure the host's CPU frequency governors are set to `performance` -- instructions for setting the frequency governor vary by machine.

    - This performance test runs with sequence length 128 and batch size 32.
    ```bash
    pytest -svv models/demos/t3000/llama2_70b/tests/test_llama_perf_decode.py::test_Llama_perf_host[wormhole_b0-True-device_params0-gen128-llama2]
    ```

## Details
Supported context lengths and batch sizes for the Llama2-70B demo are as follows:

| Context Length | Max Batch Size |
|----------------|------------|
| 2k             | 32         |

- **Input File:** Uses `./demo/data/multi_prompt.json`.
- **Model Configuration:** Utilizes a pretrained model.
- **Demo arguments:**
    - `context: [short_context, long_context]`: Select between short context (batch 32, sequence_length 2k) and long context (batch 16, sequence length 8k)
    - `ground_truth: [check_disabled, check_enabled]`: Enable or disable ground truth checking, used for testing
    - `sampling: [greedy, sampling]`: Select between greedy decoding and top-k/top-p sampling
    - `implementation: [tt-70b-T3000]`: Run the 70B model on the Tenstorrent backend
    - `num_layers: [1L, 2L, 10L, 80L]`: Select 80L to run the full model
    - `decode_only: [decode_only, prefill_decode]`: Use `prefill_decode`. Alternately, `decode_only` implements prefill via decode.
    - `trace_mode: [trace_mode_on, trace_mode_off]`: Use `trace_mode_on`. Alternately, choose `trace_mode_off` to disable trace mode
    - `chat: [text_completion, chat_completion]`: Run in text_completion mode for the pretrained model or chat_completion for the finetuned model
    - `llama_version: [llama3, llama2]`: Select the Llama2 model

Ensure you follow these guidelines to successfully run the Llama2-70B demo.
