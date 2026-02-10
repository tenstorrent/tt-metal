# [GPT-OSS-120B Galaxy] [Param Sweep] Sequence Length Sweep

GPT-OSS-120B is missing the complete sequence length sweep, from 128 up to 128K.

## Description
This test should validate model correctness and performance across the full range of supported sequence lengths.

## Sequence Lengths to Test
- 128 (minimum)
- 512
- 1K
- 2K
- 4K
- 8K
- 16K
- 32K
- 64K
- 128K (maximum)

## Current Status
Basic sequence length testing exists but comprehensive sweep is needed.

## Action Items
- [ ] Implement sequence length sweep test
- [ ] Validate accuracy across all sequence lengths
- [ ] Measure performance and memory usage for each configuration
- [ ] Document performance characteristics at different sequence lengths
- [ ] Add to Galaxy param sweep pipeline

## Reference
Similar to Llama-3.3-70B issue: https://github.com/tenstorrent/tt-metal/issues/37424
