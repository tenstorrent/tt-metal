# [GPT-OSS-120B Galaxy] [OP Perf] Performance Targets

GPT-OSS-120B is missing the required OP Device performance pipeline.

## Description
This test validates that individual operations within the GPT-OSS-120B model meet performance targets on Galaxy hardware.

## Operations to Validate
- Attention (prefill and decode)
- MLP/FFN
- Expert layers (MoE)
- Top-K routing
- RMS Norm
- RoPE
- Matrix multiplications

## Current Status
Performance targets exist in `models/demos/gpt_oss/perf_targets.json` but need validation in dedicated pipeline.

## Action Items
- [ ] Create OP device performance test suite
- [ ] Define performance targets for each operation
- [ ] Measure actual performance on Galaxy 4x8
- [ ] Identify and address performance bottlenecks
- [ ] Add to Galaxy OP device perf pipeline
- [ ] Set up continuous monitoring

## Reference
Similar to Llama-3.3-70B issue: https://github.com/tenstorrent/tt-metal/issues/37426
