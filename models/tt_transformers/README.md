# TT-Transformers

This code can run large language models such as the Llama3 family, Qwen2.5, Mistral, DeepSeek-R1-Distill variants and similar. Tensor-parallelism automatically distributes workloads across all available chips.

The current version is verified to work with the following models:
| Model                                                                                            | Hardware                    | <org/model>                                      |
|--------------------------------------------------------------------------------------------------|-----------------------------|-------------------------------------------------|
| [DeepSeek R1 Distill Llama 70B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B)| LoudBox / QuietBox / Galaxy | ```deepseek-ai/DeepSeek-R1-Distill-Llama-70B``` |
| [Llama 3.1 8B](https://huggingface.co/meta-llama/Llama-3.1-8B)                                   | n150 / p100 / p150          | ```meta-llama/Llama-3.1-8B```                   |
| [Llama 3.1 70B](https://huggingface.co/meta-llama/Llama-3.1-70B)                                 | LoudBox / QuietBox / Galaxy | ```meta-llama/Llama-3.1-70B```                  |
| [Llama 3.2 1B](https://huggingface.co/meta-llama/Llama-3.2-1B)                                   | n150                        | ```meta-llama/Llama-3.2-1B```                   |
| [Llama 3.2 3B](https://huggingface.co/meta-llama/Llama-3.2-3B)                                   | n150                        | ```meta-llama/Llama-3.2-3B```                   |
| [Llama 3.2 11B Vision](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision)                   | n300                        | ```meta-llama/Llama-3.2-11B-Vision```           |
| [Llama 3.2 90B Vision](https://huggingface.co/meta-llama/Llama-3.2-90B-Vision)                   | LoudBox / QuietBox          | ```meta-llama/Llama-3.2-90B-Vision```           |
| [Mistral 7B Instruct v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)            | n150                        | ```mistralai/Mistral-7B-Instruct-v0.3```        |
| [Qwen 2.5 7B](https://huggingface.co/Qwen/Qwen2.5-7B)                                            | n300                        | ```Qwen/Qwen2.5-7B```                           |
| [Qwen 2.5 Coder 32B](https://huggingface.co/Qwen/Qwen2.5-Coder-32B)                              | LoudBox / QuietBox          | ```Qwen/Qwen2.5-Coder-32B```                    |
| [Qwen 2.5 72B](https://huggingface.co/Qwen/Qwen2.5-72B)                                          | LoudBox / QuietBox          | ```Qwen/Qwen2.5-72B```                          |
| [Qwen 3 32B](https://huggingface.co/Qwen/Qwen3-32B)                                              | LoudBox / QuietBox          | ```Qwen/Qwen3-32B```                            |
| [Phi-3-mini-128k-instruct](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct)            | n150 / n300                 | ```microsoft/Phi-3-mini-128k-instruct```        |

<details>
<summary>  Explore an extended list of compatible models (experimental) </summary>

| Model                                                                                                                     | Hardware | <org/model>.                                                 |
|---------------------------------------------------------------------------------------------------------------------------|----------|-------------------------------------------------------------|
| [Bellatrix-Tiny-0.5B](https://huggingface.co/prithivMLmods/Bellatrix-Tiny-0.5B)                                           | N150     | `prithivMLmods/Bellatrix-Tiny-0.5B`                         |
| [Bellatrix-Tiny-1B](https://huggingface.co/prithivMLmods/Bellatrix-Tiny-1B)                                               | N150     | `prithivMLmods/Bellatrix-Tiny-1B`                           |
| [Bellatrix-Tiny-1B-R1](https://huggingface.co/prithivMLmods/Bellatrix-Tiny-1B-R1)                                         | N150     | `prithivMLmods/Bellatrix-Tiny-1B-R1`                        |
| [deepseek-coder-6.7b-instruct](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct)                           | N300     | `deepseek-ai/deepseek-coder-6.7b-instruct`                  |
| [deepseek-llm-7b-base](https://huggingface.co/deepseek-ai/deepseek-llm-7b-base)                                           | N300     | `deepseek-ai/deepseek-llm-7b-base`                          |
| [deepseek-llm-7b-chat](https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat)                                           | N300     | `deepseek-ai/deepseek-llm-7b-chat`                          |
| [DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B)                           | N300     | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`                  |
| [DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B)                               | N300     | `unsloth/DeepSeek-R1-Distill-Llama-8B`                      |
| [DeepSeek-R1-Distill-Llama-8B-abliterated](https://huggingface.co/huihui-ai/DeepSeek-R1-Distill-Llama-8B-abliterated)     | N300     | `huihui-ai/DeepSeek-R1-Distill-Llama-8B-abliterated`        |
| [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)                         | N150     | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`                 |
| [DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B)                           | T3K      | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`                  |
| [DeepSeek-R1-Distill-Qwen-32B-Japanese](https://huggingface.co/cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese)          | T3K      | `cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese`          |
| [DeepSeek-R1-Distill-Qwen-32B-abliterated](https://huggingface.co/huihui-ai/DeepSeek-R1-Distill-Qwen-32B-abliterated)     | T3K      | `huihui-ai/DeepSeek-R1-Distill-Qwen-32B-abliterated`        |
| [DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)                             | N300     | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`                   |
| [DeepSeek-R1-Distill-Qwen-7B-abliterated-v2](https://huggingface.co/huihui-ai/DeepSeek-R1-Distill-Qwen-7B-abliterated-v2) | N300     | `huihui-ai/DeepSeek-R1-Distill-Qwen-7B-abliterated-v2`      |
| [DeepSeek-R1-Distill-Qwen-7B-Japanese](https://huggingface.co/lightblue/DeepSeek-R1-Distill-Qwen-7B-Japanese)             | N300     | `lightblue/DeepSeek-R1-Distill-Qwen-7B-Japanese`            |
| [Dobby-Mini-Leashed-Llama-3.1-8B](https://huggingface.co/SentientAGI/Dobby-Mini-Leashed-Llama-3.1-8B)                     | N300     | `SentientAGI/Dobby-Mini-Leashed-Llama-3.1-8B`               |
| [Dobby-Mini-Unhinged-Llama-3.1-8B](https://huggingface.co/SentientAGI/Dobby-Mini-Unhinged-Llama-3.1-8B)                   | N300     | `SentientAGI/Dobby-Mini-Unhinged-Llama-3.1-8B`              |
| [Dolphin3.0-Mistral-24B](https://huggingface.co/cognitivecomputations/Dolphin3.0-Mistral-24B)                             | T3K      | `cognitivecomputations/Dolphin3.0-Mistral-24B`              |
| [Dolphin3.0-R1-Mistral-24B](https://huggingface.co/cognitivecomputations/Dolphin3.0-R1-Mistral-24B)                       | T3K      | `cognitivecomputations/Dolphin3.0-R1-Mistral-24B`           |
| [EuroLLM-9B-Instruct](https://huggingface.co/utter-project/EuroLLM-9B-Instruct)                                           | N300     | `utter-project/EuroLLM-9B-Instruct`                         |
| [GPT4chan-24B](https://huggingface.co/v2ray/GPT4chan-24B)                                                                 | T3K      | `v2ray/GPT4chan-24B`                                        |
| [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)                                                     | N300     | `meta-llama/Llama-2-7b-hf`                                  |
| [Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)                                                          | N300     | `meta-llama/Llama-2-7b-hf`                                  |
| [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)                                          | N300     | `meta-llama/Llama-3.1-8B-Instruct`                          |
| [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)                                          | N150     | `meta-llama/Llama-3.2-1B-Instruct`                          |
| [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)                                          | N150     | `meta-llama/Llama-3.2-3B-Instruct`                          |
| [Llama-3.2-6B-AlgoCode](https://huggingface.co/prithivMLmods/Llama-3.2-6B-AlgoCode)                                       | N150     | `prithivMLmods/Llama-3.2-6B-AlgoCode`                       |
| [Llama-Krikri-8B-Base](https://huggingface.co/ilsp/Llama-Krikri-8B-Base)                                                  | N150     | `ilsp/Llama-Krikri-8B-Base`                                 |
| [Llama-Krikri-8B-Instruct](https://huggingface.co/ilsp/Llama-Krikri-8B-Instruct)                                          | N300     | `ilsp/Llama-Krikri-8B-Instruct`                             |
| [LwQ-10B-Instruct](https://huggingface.co/prithivMLmods/LwQ-10B-Instruct)                                                 | N300     | `prithivMLmods/LwQ-10B-Instruct`                            |
| [Magnum-v4-Cydonia-vXXX-22B](https://huggingface.co/Kaoeiri/Magnum-v4-Cydonia-vXXX-22B)                                   | T3K      | `Kaoeiri/Magnum-v4-Cydonia-vXXX-22B`                        |
| [Megatron-Opus-7B-Exp](https://huggingface.co/prithivMLmods/Megatron-Opus-7B-Exp)                                         | N300     | `prithivMLmods/Megatron-Opus-7B-Exp`                        |
| [Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)                                                      | N300     | `meta-llama/Meta-Llama-3-8B`                                |
| [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)                                    | N300     | `meta-llama/Meta-Llama-3-8B-Instruct`                       |
| [Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)                                     | N150     | `mistralai/Mistral-7B-Instruct-v0.1`                        |
| [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)                                     | N150     | `mistralai/Mistral-7B-Instruct-v0.2`                        |
| [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)                                                       | N150     | `mistralai/Mistral-7B-v0.1`                                 |
| [Mistral-Small-24B-Base-2501](https://huggingface.co/mistralai/Mistral-Small-24B-Base-2501)                               | T3K      | `mistralai/Mistral-Small-24B-Base-2501`                     |
| [Mistral-Small-24B-Instruct-2501](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501)                       | T3K      | `mistralai/Mistral-Small-24B-Instruct-2501`                 |
| [NuminaMath-7B-TIR](https://huggingface.co/AI-MO/NuminaMath-7B-TIR)                                                       | N300     | `AI-MO/NuminaMath-7B-TIR`                                   |
| [OpenThinker-7B](https://huggingface.co/open-thoughts/OpenThinker-7B)                                                     | N300     | `open-thoughts/OpenThinker-7B`                              |
| [Primal-Mini-3B-Exp](https://huggingface.co/prithivMLmods/Primal-Mini-3B-Exp)                                             | N150     | `prithivMLmods/Primal-Mini-3B-Exp`                          |
| [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)                                                                  | N150     | `Qwen/Qwen2.5-0.5B`                                         |
| [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)                                                | N150     | `Qwen/Qwen2.5-0.5B-Instruct`                                |
| [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B)                                                                  | N150     | `Qwen/Qwen2.5-1.5B`                                         |
| [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)                                                | N150     | `Qwen/Qwen2.5-1.5B-Instruct`                                |
| [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)                                                    | N150     | `Qwen/Qwen2.5-3B-Instruct`                                  |
| [Qwen2.5-32B](https://huggingface.co/Qwen/Qwen2.5-32B)                                                                    | T3K      | `Qwen/Qwen2.5-32B`                                          |
| [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)                                        | N300     | `Qwen/Qwen2.5-Coder-7B-Instruct`                            |
| [Qwen2.5-Math-1.5B](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B)                                                        | N150     | `Qwen/Qwen2.5-Math-1.5B`                                    |
| [Qwen2.5-Math-7B](https://huggingface.co/Qwen/Qwen2.5-Math-7B)                                                            | N300     | `Qwen/Qwen2.5-Math-7B`                                      |
| [Reasoning-Distilled-ta-7B](https://huggingface.co/prithivMLmods/Reasoning-Distilled-ta-7B)                               | N300     | `prithivMLmods/Reasoning-Distilled-ta-7B`                   |
| [s1-32B](https://huggingface.co/simplescaling/s1-32B)                                                                     | T3K      | `simplescaling/s1-32B`                                      |
| [Selene-1-Mini-Llama-3.1-8B](https://huggingface.co/AtlaAI/Selene-1-Mini-Llama-3.1-8B)                                    | N300     | `AtlaAI/Selene-1-Mini-Llama-3.1-8B`                         |
| [Sky-T1-32B-Preview](https://huggingface.co/NovaSky-AI/Sky-T1-32B-Preview)                                                | T3K      | `NovaSky-AI/Sky-T1-32B-Preview`                             |
| [SmallThinker-3B-Preview](https://huggingface.co/PowerInfer/SmallThinker-3B-Preview)                                      | N150     | `PowerInfer/SmallThinker-3B-Preview`                        |
| [SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct)                                       | N150     | `HuggingFaceTB/SmolLM2-1.7B-Instruct`                       |
| [Sqweeks-7B-Instruct](https://huggingface.co/prithivMLmods/Sqweeks-7B-Instruct)                                           | N300     | `prithivMLmods/Sqweeks-7B-Instruct`                         |
| [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)                                     | N150     | `TinyLlama/TinyLlama-1.1B-Chat-v1.0`                        |
| [TinySwallow-1.5B](https://huggingface.co/SakanaAI/TinySwallow-1.5B)                                                      | N150     | `SakanaAI/TinySwallow-1.5B`                                 |
| [TinySwallow-1.5B-Instruct](https://huggingface.co/SakanaAI/TinySwallow-1.5B-Instruct)                                    | N150     | `SakanaAI/TinySwallow-1.5B-Instruct`                        |
| [Triangulum-v2-10B](https://huggingface.co/prithivMLmods/Triangulum-v2-10B)                                               | N300     | `prithivMLmods/Triangulum-v2-10B`                           |
| [WebMind-7B-v0.1](https://huggingface.co/prithivMLmods/WebMind-7B-v0.1)                                                   | N300     | `prithivMLmods/WebMind-7B-v0.1`                             |
| [WizardLM-7B-Uncensored](https://huggingface.co/cognitivecomputations/WizardLM-7B-Uncensored)                             | N300     | `cognitivecomputations/WizardLM-7B-Uncensored`              |
| [Xwen-7B-Chat](https://huggingface.co/xwen-team/Xwen-7B-Chat)                                                             | N300     | `xwen-team/Xwen-7B-Chat`                                    |
| [YuE-s1-7B-anneal-en-cot](https://huggingface.co/m-a-p/YuE-s1-7B-anneal-en-cot)                                           | N300     | `m-a-p/YuE-s1-7B-anneal-en-cot`                             |
| [YuE-s2-1B-general](https://huggingface.co/m-a-p/YuE-s2-1B-general)                                                       | N150     | `m-a-p/YuE-s2-1B-general`                                   |

</details>

## Prerequisites

1. Install [TT-Metalium and TTNN](../../INSTALLING.md).
2. Install additional python dependencies:
    ```
    pip install -r models/tt_transformers/requirements.txt
    ```

## How to Run

To configure the weights and run a demo, choose [Automatic Download](#automatic-download) for a quick setup or [Manual Download](#manual-download) for more control over the files.

---

### Automatic Setup

#### Login to HuggingFace using your token

Use `huggingface-cli login` or set the token with the command `export HF_TOKEN=<token>`
- To obtain a HuggingFace token visit: https://huggingface.co/docs/hub/security-tokens

#### Choose your Model

Set the `HF_MODEL` environment variable to the HuggingFace org/name of the model to be run.

This will automatically download the weights into your HuggingFace cache directory and run the model directly.

- Check the models chart on the top of the page and substitute the `<org/model>` on the following command:
```
export HF_MODEL=<org/name>
```

#### You are all set!

- Skip to section: [Run the Demo](#run-the-demo)

---

### Manual Setup

If you wish, you can manually download the weights either [from Meta](#option-1-download-llama-weights-from-meta) or [from HuggingFace](#option-2-download-weights-from-huggingface) as described by the two following sections:

#### Option 1: Download Llama Weights from Meta

You can download Llama models [directly from Meta](https://llama.meta.com/llama-downloads/), this will mean accepting their license terms.

The downloaded directories include weight files (e.g. `consolidated.00.pth`), the tokenizer `tokenizer.model` and configuration file `params.json`.

- If using Meta-provided weights you should set `LLAMA_DIR` to the path of the downloaded directory instead of setting `HF_MODEL`:
```
export LLAMA_DIR=<path_to_meta_downloaded_model_directory>
```

##### Repack Weights (Llama3.1-70B and Llama3.2-90B from Meta only)
Meta's Llama3.1-70B and Llama3.2-90B requires repacked weights. We provide scripts to facilitate this in `models/tt_transformers/scripts/repack_weights_70b.py` and `models/tt_transformers/scripts/repack_weights_90b.py`.

The repacked output directory can be same as the checkpoint directory, since the new files will have different names.
If providing a different path, please make sure that you keep the string `3.1-70B` or `3.2-90B` in the new path name, since the Llama3 codebase relies on the weights directory name to identify the correct model.

Note: Use the default value of `10` for `chunk_size`.

```
# This concatenates the sharded checkpoints and makes it easier for us to load.
python models/tt_transformers/scripts/repack_weights_70b.py <path_to_checkpoint_dir> <repacked_output_dir>
```

If providing a different output directory, please copy the `params.json` and the `tokenizer.model` files to the new directory.

**⚠️ Warning**
>
> For Llama3 models, weights downloaded from the `huggingface-cli` via
>```
>huggingface-cli download meta-llama/Meta-Llama-3-70B-Instruct --include "original/*" --local-dir Meta-Llama-3-70B-Instruct
>```
> will be in the same format as a direct download from Meta (i.e. as `consolidated.xx.pth` files). Hence, you will still need to repack your weights and export `LLAMA_DIR` as before. This is contrary to if you downloaded your weights directly from `huggingface`, as those weights will be downloaded as sharded `.safetensors` files.

#### You are all set!

- Skip to section: [Run the Demo](#run-the-demo)

---

#### Option 2: Download Weights from HuggingFace

Navigate to [HuggingFace](https://huggingface.co) and download the weights of the model you wish to run (check [models chart](#tt-transformers) for compatibility and direct links to HuggingFace models).

- Ensure your model directory has the following structure:
```
/path_to/org/model/
    config.json
    generation_config.json
    model-00001-of-00062.safetensors
    ...
```

- Set `HF_MODEL` to the directory of the downloaded weights:
```
export HF_MODEL=<path_to_downloaded_directory>
```

#### You are all set!

- Skip to section: [Run the Demo](#run-the-demo)

---

### Run the Demo

- Make sure either `HF_MODEL` or `LLAMA_DIR` has been set:

`export HF_MODEL=<hf_model_name or hf_downloaded_directory>`

`export LLAMA_DIR=<path_to_meta_downloaded_model_directory>`

- Run the Demo

These are some example commands on how to run the demo. For more information about the parameters, check [parameters description.](#parameters-description)

```
# Batch-1
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-1"

# Batch-32
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-32"

# Long context with custom parameters
pytest models/tt_transformers/demo/simple_text_demo.py -k "long-context" --max_seq_len=16384

# Long-context
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and long"
```

The above examples are run in `ModelOptimizations.performance` mode. You can override this by setting the `optimizations` or the `decoder_config_file` argument in the demo. To use instead the accuracy mode you can call the above tests with `-k "accuracy and ..."` instead of performance.

## Details

### Extra compatibility settings for non-Llama models

If you are bringing up a new model that is similar to these but is not listed above, you may also need to set additional environment variables:
- `MAX_PREFILL_CHUNK_SIZE` - this determines how many thousands of tokens are prefilled in one go. For optimal performance pick 128. Depending on the model dimensions and hardware you're running on, there may not be enough L1 to prefill 128K tokens at once, in which case you can reduce this in powers of 2 down to 4.
- `PAD_MLP_CORES` - models with a hidden_dim that is not a nice power of 2 may not have a valid layout or may run with lower performance. You can set this to a multiple of 8 between 8 and 64; `16` and `32` commonly work well if this is required.

You should also watch out for:
- RoPE encoding style. `llama3`, `yarn` and of course `none` are supported. HuggingFace models encode the complex numbers in RoPE as r1, r2, ..., i1, i2, ... whereas Meta models encode the complex numbers as r1, i1, r2, i2, ... - TTT uses a Meta-style implementation of the RoPE op and when loading a HuggingFace it will reshuffle the weights of the pre-rope attention weights to interleave their outputs in this style (see `reverse_permute` in [load_checkpoints.py](tt/load_checkpoints.py)). When _using_ TTT this happens invisibly and should not affect you, but when bringing up a new model or modifying TTT being aware of this will make some otherwise confusing things clearer.
- Our [accuracy test](tests/test_accuracy.py) will require you to [generate some reference logits](tests/generate_reference_hf.py) and perhaps update the test to use them.
- We parallelise attention over the number of heads. If this number is e.g. 14 then you will not be able to run it on more than 2 chips (because 14/2=7, a prime number). We do not support head-padding or similar mitigations at this time but a PR would be cool.

Huggingface models specify their architecture in the `config.json` file. The following architectures are known to work:

- LlamaForCausalLM
- Qwen2ForCausalLM
- Qwen3ForCausalLM
- MistralForCausalLM
- Phi3ForCausalLM

At the time of writing this covers the majority of popular HuggingFace text-generation models. If you find another architecture that works or extend TT-Transformers to support one we would love to accept a PR!

---

### Environment Variables Description

- `HF_MODEL` is the HuggingFace org/name of the model you want to run or the path to the downloaded Huggingface weights.
- `LLAMA_DIR` sets the path for Meta-provided Llama3 model weights if you are not using HuggingFace.
- `TT_CACHE_PATH` is optional. It sets the path for ttnn's weight cache files. See below for more details.
- `MESH_DEVICE` is optional. It allows you to use fewer devices than are available. See below for more details.

On the first execution of each model, TTNN will create weight cache files for that model, to speed up future runs. These cache files only need to be created once for each model and device. These files are stored in one of three places:

1. `TT_CACHE_PATH` if you have set it.
2. `HF_MODEL/device_name` if a path to downloaded weights was specified using `HF_MODEL`.
3. `LLAMA_DIR/device_name` if a path to downloaded Meta-provided weights was specified using `LLAMA_DIR`.
4. `model_cache/HF_MODEL/device_name` if a HuggingFace model name was specified using `HF_MODEL`.

The device name used is:

- `N150` for N150
- `N300` for N300
- `T3K` for LoudBox / QuietBox
- `TG` for Galaxy

By default tensor parallelism is used to run the model over all available chips. You can instead run on a smaller mesh either for testing or for performance reasons (for very small models the communication overhead of tensor parallelism may be larger than the performance gained). To use a smaller mesh, set `MESH_DEVICE` to one of the supported devices: `N150`, `N300`, `T3K` or `TG`.

Example: `export MESH_DEVICE=N150`, will enable running one a single chip of a multi-chip system.

---

### Parameters Description

The `simple_text_demo.py` script includes the following main modes of operation and is parametrized to support other configurations.

- `batch-1`: Runs a small prompt (128 tokens) for a single user
- `batch-32`: Runs a small prompt (128 tokens) for a a batch of 32 users
- `long-context`: Runs a large prompt (64k tokens) for a single user
- `reasoning-1`: Runs a reasoning prompt for a single user (generates up to 15k tokens)

If you want to provide your own demo configuration, please take a look at the pytest parametrize calls in `models/tt_transformers/demo/simple_text_demo.py`. For convenience we list all the supported params below:

- `input_prompts (string)`: input json file with prompts to process. See `models/tt_transformers/demo/*.json` for a list of input files
- `instruct (bool)`: Whether to use Llama instruct weights or general weights
- `repeat_batches (int)`: Number of consecutive batches of users to run (default: 1)
- `max_seq_len (int)`: Maximum context length supported by the model (refer to the table above)
- `batch_size (int)`: Number of users in a batch (Supports 1/2/4/8/16/32 batches)
- `max_generated_tokens (int)`: Maximum number of tokens to generate for each user (Note that the users will stop generation before this limit if they reach a eos token)
- `paged_attention (bool)`: Whether to use paged attention or default attention (vLLM support (WIP) requires paged attention)
- `page_params (dict)`: Page parameters for paged attention - [`block_size`, `max_num_blocks`]. For smaller context lengths use `block_size=32` and `max_num_blocks=1024`, for larger context use block_size=64 and max_num_blocks=2048
- `sampling_params (dict)`: Sampling parameters for decoding -[`temperature`, `top_p`]. If temperature is set to 0, argmax (greedy decode) is used.
- `stop_at_eos (bool)`: Flag to stop decoding when the model generates an EoS token
- `optimizations (ModelOptimizations)`: Optimization level to use for the model [`accuracy`, `performance`]. Applied uniformly across all decoders unless an override config exists in `models/tt_transformers/model_params/<model-name>`
- `decoder_config_file (DecodersPrecision)`: Fine-grained optimization control that allows specifying a configuration file to set different settings for each decoder.

Please note that using `argmax` with `batch_size > 1` or using `top-p` sampling with any batch size, these ops will be run on host. This is because those ops are not yet fully supported on device. A decrease in performance is expected when these configurations are enabled.

---

### Optimization overrides
Some models require a unique set of optimizations defined in `models/tt_transformers/model_params/<model-name>`. To override the default optimizations, you can define files named `models/tt_transformers/tt/model_config/PERFORMANCE_DECODER_CONFIG_FILENAME` and `models/tt_transformers/tt/model_config/ACCURACY_DECODER_CONFIG_FILENAME` in the appropriate `models/tt_transformers/model_params/<model-name>` directory to override the `ModelOptimizations.performance` and `ModelOptimizations.accuracy` optimizations respectively. For example, to override the default "performance" optimizations for Llama3.1-8B-Instruct, a file named `performance_decoder_config.json` has been created in the `models/tt_transformers/model_params/Llama3.1-8B-Instruct` directory. The content to write in override files is described in [the custom optimizations section](#custom-optimizations). Optimizations are applied with the following prioritization:
1. from override config (if it exists)
2. from the `optimizations` argument

---

### Custom input arguments
To facilitate testing different configurations, `simple_text_demo.py` supports argument overrides. The full list of overrides is included in `models/tt_transformers/demo/conftest.py`.

An example usage where the `batch-1` test is modified to run with 16 users and keep generating tokens until 1024 are generated:

```
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-1" --batch_size 16 --max_generated_tokens 1024 --stop_at_eos 0
```

---

### Custom optimizations
To apply the same settings across all decoders, the `optimizations` argument can be used. `optimizations` offers a wide range of configurations for precision and math fidelity. The user can override the configurations of the data types of the weight tensors and activation tensors and the math fidelity of the kernels that works on those tensors, using the `--optimizations` argument on the command line. For example:

```
pytest models/tt_transformers/demo/simple_text_demo.py -k "accuracy and batch-1" --optimizations 'precision_cfg = {ff1_3: bfp4, ff2: bfp4, wqkv: bfp8, wo: bfp8, kv_cache: bfp8, activation: mixed}, fidelity_cfg = {li_ff1_3: hifi2, li_ff2: lofi, li_qkv_decode: hifi2, li_o_decode: hifi2, sdpa_decode: hifi2na, li_qkv_prefill: hifi2, li_o_prefill: hifi2fp16, sdpa_prefill: hifi4}'
```

Please refer to [model_config.py](models/tt_transformers/tt/model_config.py) for the full list of supported key-value pairs in the `--optimizations` argument. Also, please refer to the [PERF.md](PERF.md) file for performance and accuracy across a select range of configurations for an example Pareto front analysis.

To apply non-uniform settings across the decoders, the user can provide a JSON file using the `decoder_config_file` argument to specify the configuration for each decoder. For example

```
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-1" --decoder_config_file 'models/tt_transformers/demo/config_16_decoders.json'
```

When a component is not specified (e.g., FF2 is missing for decoder 2 in `models/tt_transformers/demo/config_16_decoders.json`), the baseline configuration is used for that component.

Using the lt tool (`models/tt_transformers/lt`), the user can also provide multiple JSON configurations in the `models/tt_transformers/tests/configurations` folder and run a Pareto analysis on them using the `pareto_from_json` command.

---

### Expected performance and accuracy

See [PERF.md](PERF.md) for expected performance and accuracy across different configurations.
Accuracy of the network architectures is measured by exact token matching using teacher forcing method. During inference the previous token is replaced by the ground truth token while the network generates the next token. This allows to avoid accumulating errors when comparisons on a finer level (tokens) assessed in comparison to other known metrics that compare quality and context of the answer. Token accuracy can be reported by passing the argument shown below:

```
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-1" --token_accuracy True
```

---

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
| Llama3.2-90B | Not supported | Not supported | 32k tokens     | Not supported |
| DeepSeek-R1-Distill-Llama3.3-70B | Not supported | Not supported | 32k tokens | 128k tokens |

- These max chunk sizes are specific to max context length 128k and are configured via `MAX_PREFILL_CHUNK_SIZES_DIV1024` in [model_config.py](https://github.com/tenstorrent/tt-metal/blob/main/models/demos/llama3/tt/model_config.py). If the max context length is set to a smaller value using the `max_seq_len` flag (see [Run the demo](#run-the-demo)), these chunk sizes can possibly be increased due to using a smaller KV cache.

**Chunked prefill (Llama3.2-11B multimodal)**: Llama3.2-11B multimodal is currently only supported on N300 and T3000. On N300, a max prefill context length of 8k is supported, while T3000 supports a max context length of 128k.

---

### Memory Optimization

#### HuggingFace Model Caching Control

To help manage memory usage, you can control whether the HuggingFace model is cached in memory using the `cache_hf` parameter via command line or code:

```python
# Default: disables caching to conserve memory usage
model_args = ModelArgs(
    mesh_device,
    cache_hf=False,  # Default: Reduces memory usage by not keeping HF model in memory
    max_batch_size=1,
    max_seq_len=2048
)

# Optional: enables caching for faster repeated access
model_args = ModelArgs(
    mesh_device,
    cache_hf=True,  # Cache HF model for better performance running reference tests
    max_batch_size=4,
    max_seq_len=4096
)
```

**When to disable caching (`cache_hf=False`):**
- Running on systems with limited memory (< 256GB)
- Loading large models (70B+ parameters)
- Using the model for single inference runs
- When you don't need reference model comparisons

**When to keep caching enabled (`cache_hf=True`, default):**
- Sufficient memory available
- Comparisons with torch model is needed
- Minimizing test duration is prioritized over memory usage
- Running reference model tests

The `cache_hf` parameter affects:
- `load_state_dict()` method: Controls whether HF model is cached after loading
- `reference_transformer()` method: Controls whether to reuse cached model or load fresh

**Memory Impact:**
- Disabling caching saves approximately the full model size in memory
- For a 70B model, this can save ~140GB+ of memory usage
- Increased test duration as model needs to be reloaded for reference operations
