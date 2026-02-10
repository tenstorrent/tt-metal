# [GPT-OSS-120B Galaxy] [Unit] Individual Module tests

GPT-OSS-120B is missing comprehensive unit tests on the Galaxy pipeline.

## Description
Individual module tests should validate each component of the GPT-OSS-120B model independently:
- Attention modules
- MLP modules
- RMS Norm
- Experts/MoE components
- RoPE (Rotary Position Embedding)
- Top-K routing

## Current Status
Some unit tests exist in `models/demos/gpt_oss/tests/unit/` but they need to be expanded and integrated into the Galaxy pipeline.

## Action Items
- [ ] Audit existing unit tests in `models/demos/gpt_oss/tests/unit/test_modules.py`
- [ ] Ensure all modules have comprehensive PCC validation
- [ ] Add Galaxy pipeline configuration for 120B-specific unit tests
- [ ] Validate tests pass on 4x8 Galaxy configuration

## Reference
Similar to Llama-3.3-70B issue: https://github.com/tenstorrent/tt-metal/issues/37419
