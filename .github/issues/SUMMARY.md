# GPT-OSS-120B Missing Tests Issue Templates - Summary

## What Was Created

This PR creates comprehensive documentation templates for tracking missing/failing tests for the GPT-OSS-120B model on WH Galaxy, mirroring the structure of issue #37410 (Llama-3.3-70B Galaxy Missing/Failing Tests).

## Files Structure

```
.github/issues/
├── GPT-OSS-120B-Galaxy-Missing-Tests.md    # Main tracking issue template
├── INSTRUCTIONS.md                         # Step-by-step guide for creating issues
├── README.md                               # Overview and usage guide
└── sub-issues/                            # Individual test tracking templates
    ├── GPT-OSS-120B-Unit-Individual-Modules.md
    ├── GPT-OSS-120B-Unit-1L-Decoder-PCC.md
    ├── GPT-OSS-120B-Demo-Evals-Batch-1.md
    ├── GPT-OSS-120B-Demo-Evals-Batch-32.md
    ├── GPT-OSS-120B-Demo-Long-Prompts.md
    ├── GPT-OSS-120B-Param-Sweep-Batch-Size.md
    ├── GPT-OSS-120B-Param-Sweep-Sequence-Length.md
    └── GPT-OSS-120B-OP-Device-Perf.md
```

## Issue Categories

### Model Unit Test Pipeline (2 issues)
1. Individual Module tests - Validate attention, MLP, RMS norm, experts, RoPE, top-K
2. 1L Decoder PCC - Single layer decoder validation

### Model Demo Pipeline (3 issues)
1. Evals-batch-1 - Evaluation with batch size 1
2. Evals-batch-32 - Evaluation with batch size 32
3. Long Prompts - Extended context length validation (8K-128K)

### Model Param Sweep Pipeline (2 issues)
1. Batch Size Sweep - Test batch sizes 1-128
2. Sequence Length Sweep - Test sequence lengths 128-128K

### Model OP Device Perf Pipeline (1 issue)
1. Performance Targets - Operation-level performance validation

### Other Pipelines (3 items)
- vLLM Nightly
- Stress tests
- PR/Merge Gate

## How to Use

See `INSTRUCTIONS.md` for detailed step-by-step guide to create the GitHub issues.

Quick steps:
1. Create main tracking issue from `GPT-OSS-120B-Galaxy-Missing-Tests.md`
2. Create 8 sub-issues from templates in `sub-issues/`
3. Update main issue with links to sub-issues
4. Apply appropriate labels (`LLMs on Metal`, `models-ci`, `Tier 1 Model`, `P1`)

## Comparison with Issue #37410

| Llama-3.3-70B Issue | GPT-OSS-120B Template |
|---------------------|----------------------|
| #37410 (main) | GPT-OSS-120B-Galaxy-Missing-Tests.md |
| #37419 (Unit: Individual Modules) | GPT-OSS-120B-Unit-Individual-Modules.md |
| #37420 (Unit: 1L Decoder PCC) | GPT-OSS-120B-Unit-1L-Decoder-PCC.md |
| #37421 (Demo: Evals-batch-1) | GPT-OSS-120B-Demo-Evals-Batch-1.md |
| #37422 (Demo: Evals-batch-32) | GPT-OSS-120B-Demo-Evals-Batch-32.md |
| #37423 (Demo: 128sl-accuracy) | GPT-OSS-120B-Demo-Long-Prompts.md |
| #37424 (Param: seqlen sweep) | GPT-OSS-120B-Param-Sweep-Sequence-Length.md |
| #37425 (Param: sampling features) | GPT-OSS-120B-Param-Sweep-Batch-Size.md |
| #37426 (OP Perf) | GPT-OSS-120B-OP-Device-Perf.md |

## Current GPT-OSS Test Status

Existing tests:
- Basic demo tests in `tests/pipeline_reorg/galaxy_demo_tests.yaml` (lines 79-85)
- Unit tests in `models/demos/gpt_oss/tests/unit/`
- Accuracy tests in `models/demos/gpt_oss/tests/accuracy/`

Gaps to address:
- Comprehensive unit test pipeline
- Extended demo evaluations (batch variations, long prompts)
- Parameter sweep testing
- OP-level performance validation
- vLLM integration
- Stress testing
- PR/Merge gate integration

## GPT-OSS Specifics

The GPT-OSS model has unique characteristics:
- Mixture of Experts (MoE) architecture
- Supports both 20B and 120B variants
- Maximum sequence length: 128K
- Expert routing via top-K selection
- Performance targets defined in `models/demos/gpt_oss/perf_targets.json`

These templates account for these specifics in the test descriptions.

## Next Steps

1. Review templates with model owners
2. Create GitHub issues using the templates
3. Assign issues to appropriate team members
4. Track progress on implementing missing tests
5. Update internal test matrix spreadsheet

## References

- Original issue: https://github.com/tenstorrent/tt-metal/issues/37410
- Test matrix: [Internal Google Spreadsheet](https://docs.google.com/spreadsheets/d/1a75kfIoVXrlbri0BtwKJsFJxj6Qa8mbJENPZGgvJjPA/edit?usp=sharing)
- GPT-OSS model: `models/demos/gpt_oss/`
