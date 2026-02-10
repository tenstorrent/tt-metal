# [GPT-OSS-120B Galaxy] [Unit] 1L Decoder PCC

GPT-OSS-120B is missing 1-layer decoder PCC (Pearson Correlation Coefficient) validation tests on the Galaxy pipeline.

## Description
This test should validate a single decoder layer's output against the reference PyTorch implementation to ensure numerical accuracy.

## Current Status
Unit tests exist but need to be refactored and added to the Galaxy pipeline specifically for the 120B model.

## Action Items
- [ ] Create or adapt 1L decoder PCC test for GPT-OSS-120B
- [ ] Ensure PCC threshold meets requirements (typically > 0.99)
- [ ] Add to Galaxy unit test pipeline
- [ ] Validate on 4x8 Galaxy configuration

## Reference
Similar to Llama-3.3-70B issue: https://github.com/tenstorrent/tt-metal/issues/37420
