# DeepSeek V3 on Galaxy

Model: [deepseek-ai/DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528)
Platform: 2x 6U Galaxy machines

Other DeepSeekV3 models should also work on this codebase:
- [deepseek-ai/DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)
- [deepseek-ai/DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3)

## Submodule bring-up

This has changed from TTT - we are moving to a more modular approach. See [tt/README.md](./tt/README.md) for details.

## Contents

- [reference](./reference): Reference model code from HuggingFace, cleaned up and extracted submodule code etc.
- [tests](./tests): pytests for submodules
- [tt](./tt): ttnn submodule code
- [demo](./demo): demo code for DeepSeek V3 on Galaxy
