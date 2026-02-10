# [GPT-OSS-120B Galaxy] [Demo] Long Prompts

GPT-OSS-120B needs long prompt/context length validation on the Galaxy Model Demo pipeline.

## Description
This test validates model accuracy and performance with extended context lengths, testing various sequence lengths up to the model's maximum supported length (128K).

## Current Status
Current demo tests focus on standard sequence lengths. Extended context length testing needs to be added.

## Action Items
- [ ] Implement long prompt tests for various sequence lengths (8K, 32K, 64K, 128K)
- [ ] Validate accuracy at different context lengths
- [ ] Measure performance degradation with longer contexts
- [ ] Add to Galaxy demo test pipeline
- [ ] Document memory requirements for different sequence lengths

## Reference
Similar to Llama-3.3-70B issue: https://github.com/tenstorrent/tt-metal/issues/37423
