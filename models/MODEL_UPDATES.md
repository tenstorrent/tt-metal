# Model Updates - July 29, 2024

> [!NOTE]
>
> Please refer to the front-page [README](../README.md) for the latest stable release version for each model.

## [Llama 3.1 - 8B](demos/wormhole/llama31_8b)
- Added support for LLaMA 3.1 - 8B
- Runs fast prefill for sequence lengths of up to 512 tokens
- Supports a maximum context length of 8K tokens

## [Llama 3/3.1 - 70B](demos/t3000/llama3_70b)
- Added support for LLaMA 3.1 70B (new scaled rotary position embeddings)
- Prefill and decode now support 8K context length with batch size 16

## [Mistral7B](demos/wormhole/mistral7b)
- Added prefill support for 4K context length, using scaled dot product attention
