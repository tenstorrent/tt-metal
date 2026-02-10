# [GPT-OSS-120B Galaxy] [Param Sweep] Batch Size Sweep

GPT-OSS-120B is missing comprehensive batch size parameter sweep testing.

## Description
This test should sweep through various batch sizes to validate model correctness and measure performance characteristics across different batch configurations.

## Batch Sizes to Test
- batch=1 (minimum, latency-focused)
- batch=8
- batch=16
- batch=32
- batch=64
- batch=128 (maximum supported)

## Current Status
Basic batch size testing exists but comprehensive sweep is needed for Galaxy pipeline.

## Action Items
- [ ] Implement batch size sweep test
- [ ] Validate accuracy across all batch sizes
- [ ] Measure throughput and latency for each configuration
- [ ] Document optimal batch sizes for different use cases
- [ ] Add to Galaxy param sweep pipeline

## Reference
Similar to Llama-3.3-70B issue: https://github.com/tenstorrent/tt-metal/issues/37424
