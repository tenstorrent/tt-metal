# Model Updates

> [!NOTE]
>
> Please refer to the front-page [README](../README.md) for the latest verified release for each model.

## June 9, 2025

### [Qwen 3](tt_transformers)
- Added support for dense Qwen3 models (0.6B, 1.7B, 4B, 8B, 14B, 32B) on Wormhole devices.

### [Llama 3.1-70B - Galaxy](demos/llama3_subdevices)
- Integrated Llama 3.1-70B on Galaxy into the [vLLM fork](https://github.com/tenstorrent/vllm/tree/dev/tt_metal).
- Added initial support for sampling tokens on device with batch size 32.

## May 26, 2025

### [Llama 3.1-8B](tt_transformers)
- Added support for Llama 3.1 8B on Blackhole P100, P150, 2xP150.

### [Mistral 7B](tt_transformers)
- Added support for Mistral 7B in [models/tt_transformers](tt_transformers).
- Integrated Mistral 7B into the [vLLM fork](https://github.com/tenstorrent/vllm/tree/dev/tt_metal).

## May 5, 2025

### [Llama 3.2-90B-Vision](tt_transformers)
- Added support for Llama 3.2 90B Vision on QuietBox in [models/tt_transformers](tt_transformers).

## April 22, 2025

### [TT-Transformers](tt_transformers)
- Added support for non-uniform data format configurations in different decoder layers via json files.

## April 7, 2025

### [Llama 3.1-70B - Galaxy](demos/llama3_subdevices)
- Achieved 45 t/s/u (and still working on further improvements) on Wormhole Galaxy for decode mode, with batch size 32 and 128 input sequence length. The included optimizations were: 1) using DRAM prefetching to remove memory bottlenecks for matmuls, 2) using [Sub-Devices](../tech_reports/SubDevices/SubDevices.md) to run multiple ops in parallel, 3) using CCLs enabled by [TT-Fabric](../tech_reports/TT-Fabric/TT-Fabric-Architecture.md).
- Created a functional prefill + decode demo which can be run via [text_demo.py](demos/llama3_subdevices/demo/text_demo.py).

## March 24, 2025

### [TT-Transformers](tt_transformers)
- Moved and renamed `models/demos/llama3` to [models/tt_transformers](tt_transformers) which is a commonized library for running LLMs similar to the Llama3 family.
- Added support for hybrid data / tensor parallelism to the models that are part of [TT-Transformers](tt_transformers).

### [Whisper](demos/whisper)
- Added support for the Whisper (distil-large-v3) model on N150.

## March 10, 2025

### [QwQ-32B](tt_transformers)
- Added support for QwQ-32B on QuietBox.

## February 24, 2025

### [DeepSeek R1 Distill Llama 3.3 70B](tt_transformers)
- Added support for DeepSeek R1 Distill Llama 3.3 70B on QuietBox.

### [Qwen 2.5](tt_transformers)
- Added support for Qwen2.5-7B on N300 and Qwen2.5-72B on QuietBox.

### [Llama 3.1/3.2](tt_transformers)
> **Note:** This feature is available as of release [v0.56.0-rc37](https://github.com/tenstorrent/tt-metal/tree/v0.56.0-rc37)
- Overhauled the demo script (now called [simple_text_demo.py](tt_transformers/demo/simple_text_demo.py)) to use a simplified causal generation interface.
- Added support for custom input argument overrides to the demo.

## February 10, 2025

### [Llama 3.1/3.2](tt_transformers)
> **Note:** This feature is available as of release [v0.56.0-rc16](https://github.com/tenstorrent/tt-metal/tree/v0.56.0-rc16)
- Added support for loading HuggingFace model formats (previously loaded Meta checkpoint formats), which will also enable easier adoption of future derivative models.

### [Llama 3.2-11B-Vision](tt_transformers)
- Added support for processing text-only prompts to the model and the [vLLM fork](https://github.com/tenstorrent/vllm/tree/dev/tt_metal).

## January 13, 2025

- Integrated Llama3 models (1B/3B/8B/11B/70B) into [vLLM fork](https://github.com/tenstorrent/vllm/tree/dev/tt_metal) for all compatible Tenstorrent devices (N150/N300/QuietBox/Galaxy).
- Enabled prefill with the maximum context length (131072) when running the Llama3 text models on smaller devices (N150/N300) via chunked prefill.

## December 16, 2024

### [Llama 3.1/3.2](tt_transformers)
- Added support for batch size 32 and the maximum context length (131072 tokens).
- Added full hardware compatibilty for the 1B/3B/8B/11B/70B models (all models are now compatible with N150, N300, QuietBox, Galaxy except for 70B which is only supported on QuietBox and Galaxy due to its large size).

## December 2, 2024

### [Llama 3.1/3.2](tt_transformers)
- Improved the decode performance of the 1B/3B/8B/11B text models (for 8B, increased from ~23 t/s/u to ~28 t/s/u) by using BFP4 weights (instead of BFP8) for FF1 and FF3 in the MLP.
- Added the option to specify custom model configurations, with two defaults for performance and accuracy already provided.

## November 18, 2024

### [Llama 3.2 - 1B/3B/11B](tt_transformers)
- Created a new shared codebase for the Llama3 family of models, with newly added support for Llama3.2-1B/3B/11B.

### [Llama 3/3.1 - 70B](demos/t3000/llama3_70b)
- Added support for the `ttnn.experimental.rotary_embedding_llama` op in decode mode, eliminating unnecessary device transfers of rotation matrices.

## October 21, 2024

### [Llama 3/3.1 - 70B](demos/t3000/llama3_70b)
- Enabled prefill workloads to pad to multiples of 1024 instead of powers of 2, improving overall performance for longer sequences

## October 7, 2024

### [Llama 3.1 - 8B](demos/wormhole/llama31_8b)
- Added support for continuous batching
- Added paged caching support for PagedAttention
- Added a demo which runs with TT-NN tracing (23 t/s/u decode on main)

## September 23, 2024

### [Llama 3/3.1 - 70B](demos/t3000/llama3_70b)
- Added support for 128K context length using PagedAttention
- Added a continuous batching demo for running multiple batches of users consecutively
- Added the option to enable TT-NN tracing

## September 9, 2024

### [Mixtral7Bx8](demos/t3000/mixtral8x7b)
> **Note:** This feature is available as of release [v0.52.0-rc1](https://github.com/tenstorrent/tt-metal/tree/v0.52.0-rc1)
- Added support for any user prompt size up to a maximum of 32k tokens

## August 26, 2024

### [Falcon7B](demos/falcon7b_common)
- Added data parallel demo for a single Galaxy (32 chips)
- Refactored all modules and tests to use ttnn multi-device tensors

### [Llama 3.1 - 8B](demos/wormhole/llama31_8b)
> **Note:** This feature is available as of release [v0.51.0-rc33](https://github.com/tenstorrent/tt-metal/tree/v0.51.0-rc33)
- Added multi-batching support to the demo for running multiple batches of users consecutively

### [Mixtral7Bx8](demos/t3000/mixtral8x7b)
- Improved end-to-end performance through optimizations to the attention mask in flash decoding

## August 12, 2024

### [Llama 3.1 - 8B](demos/wormhole/llama31_8b)
- Added support for flash decoding

### [Mistral7B](demos/wormhole/mistral7b)
- Updated the demo to support multiple batches of users

### [Mamba-2.8B](demos/wormhole/mamba)
- Updated the demo to use the full prefill graph instead of processing a single token of the prompt at a time using decode

### [Mixtral7Bx8](demos/t3000/mixtral8x7b)
- Added support for decode with 32K context length using flash decoding
- Fused mixture of experts into a single operation using `ttnn.moe`

## July 29, 2024

### [Llama 3.1 - 8B](demos/wormhole/llama31_8b)
- Added support for LLaMA 3.1 - 8B
- Runs fast prefill for sequence lengths of up to 512 tokens
- Supports a maximum context length of 8K tokens

### [Llama 3/3.1 - 70B](demos/t3000/llama3_70b)
- Added support for LLaMA 3.1 70B (new scaled rotary position embeddings)
- Prefill and decode now support 8K context length with batch size 16

### [Mistral7B](demos/wormhole/mistral7b)
- Added prefill support for 4K context length, using scaled dot product attention
