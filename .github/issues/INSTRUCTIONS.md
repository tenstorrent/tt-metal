# How to Create the GitHub Issues for GPT-OSS-120B

This guide explains how to create GitHub issues similar to #37410 for GPT-OSS-120B using the templates in this repository.

## Overview

This PR creates documentation templates that mirror issue #37410 (Llama-3.3-70B Galaxy Missing/Failing Tests) but targeting the GPT-OSS-120B model. Since the agent cannot directly create GitHub issues, these markdown templates are provided for manual creation.

## Files Created

### Main Tracking Issue Template
- **Location**: `.github/issues/GPT-OSS-120B-Galaxy-Missing-Tests.md`
- **Purpose**: Main meta-issue to track all missing GPT-OSS-120B tests
- **GitHub Issue Title**: `[Models CI] [GPT-OSS-120B Galaxy] Missing/Failing Tests`

### Sub-Issue Templates
Located in `.github/issues/sub-issues/`:

1. **GPT-OSS-120B-Unit-Individual-Modules.md** - Unit tests for individual modules
2. **GPT-OSS-120B-Unit-1L-Decoder-PCC.md** - 1-layer decoder PCC validation
3. **GPT-OSS-120B-Demo-Evals-Batch-1.md** - Demo evaluations with batch=1
4. **GPT-OSS-120B-Demo-Evals-Batch-32.md** - Demo evaluations with batch=32
5. **GPT-OSS-120B-Demo-Long-Prompts.md** - Long context/prompt testing
6. **GPT-OSS-120B-Param-Sweep-Batch-Size.md** - Batch size parameter sweep
7. **GPT-OSS-120B-Param-Sweep-Sequence-Length.md** - Sequence length parameter sweep
8. **GPT-OSS-120B-OP-Device-Perf.md** - Operation-level performance targets

## Step-by-Step Instructions

### 1. Create the Main Tracking Issue

1. Navigate to https://github.com/tenstorrent/tt-metal/issues/new
2. Copy the entire content from `.github/issues/GPT-OSS-120B-Galaxy-Missing-Tests.md`
3. Paste into the issue description
4. Set the issue title: `[Models CI] [GPT-OSS-120B Galaxy] Missing/Failing Tests`
5. Add labels:
   - `LLMs on Metal`
   - `models-ci`
   - `Tier 1 Model`
6. Assign to appropriate team member (e.g., GPT-OSS model owner)
7. Create the issue and note the issue number (e.g., #XXXXX)

### 2. Create Sub-Issues

For each file in `.github/issues/sub-issues/`:

1. Navigate to https://github.com/tenstorrent/tt-metal/issues/new
2. Copy the content from the respective template file
3. Use the filename (without extension) as the issue title
4. Add labels:
   - `P1`
   - `models-ci`
   - `Tier 1 Model`
5. Add issue type: `Task`
6. Assign to appropriate team member
7. Create the issue and note the issue number

### 3. Update Main Tracking Issue

Once all sub-issues are created:

1. Edit the main tracking issue (#XXXXX from step 1)
2. Replace each checklist item with a link to its corresponding sub-issue:

```markdown
### Model Unit Test Pipeline
- https://github.com/tenstorrent/tt-metal/issues/[UNIT-MODULE-ISSUE]
- https://github.com/tenstorrent/tt-metal/issues/[UNIT-PCC-ISSUE]

### Model Demo Pipeline
- https://github.com/tenstorrent/tt-metal/issues/[DEMO-BATCH1-ISSUE]
- https://github.com/tenstorrent/tt-metal/issues/[DEMO-BATCH32-ISSUE]
- https://github.com/tenstorrent/tt-metal/issues/[DEMO-LONG-ISSUE]

### Model Param Sweep Pipeline
- https://github.com/tenstorrent/tt-metal/issues/[PARAM-BATCH-ISSUE]
- https://github.com/tenstorrent/tt-metal/issues/[PARAM-SEQLEN-ISSUE]

### Model OP Device Perf Pipeline
- https://github.com/tenstorrent/tt-metal/issues/[OP-PERF-ISSUE]
```

## Issue Structure Comparison

This structure mirrors issue #37410:
- **Main issue**: #37410 → New GPT-OSS-120B main issue
- **Unit tests**: #37419, #37420 → New GPT-OSS-120B unit issues
- **Demo tests**: #37421, #37422, #37423 → New GPT-OSS-120B demo issues
- **Param sweep**: #37424, #37425 → New GPT-OSS-120B param sweep issues
- **OP perf**: #37426 → New GPT-OSS-120B OP perf issue

## Current GPT-OSS Test Status

GPT-OSS-120B currently has basic tests in:
- **Galaxy Demo Pipeline**: `tests/pipeline_reorg/galaxy_demo_tests.yaml` (lines 79-85)
  - Text demo tests for both 20B and 120B models
  - Accuracy validation tests

These need to be expanded to comprehensive tier 1 model coverage.

## Additional Notes

- The GPT-OSS model uses a Mixture of Experts (MoE) architecture, which differs from Llama
- Performance targets are defined in `models/demos/gpt_oss/perf_targets.json`
- Unit test thresholds are in `models/demos/gpt_oss/unit_test_thresholds.json`
- The model supports sequence lengths up to 128K
- Both 20B and 120B variants should be considered in testing

## Reference

- Original Llama issue: https://github.com/tenstorrent/tt-metal/issues/37410
- Internal test matrix: [Google Spreadsheet](https://docs.google.com/spreadsheets/d/1a75kfIoVXrlbri0BtwKJsFJxj6Qa8mbJENPZGgvJjPA/edit?usp=sharing)
