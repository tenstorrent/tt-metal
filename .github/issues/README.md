# GPT-OSS-120B Galaxy Missing/Failing Tests Documentation

This directory contains issue templates for tracking missing and failing tests for the GPT-OSS-120B model on WH Galaxy, similar to issue #37410 for Llama-3.3-70B.

## Main Tracking Issue

**File**: `GPT-OSS-120B-Galaxy-Missing-Tests.md`

This is the main tracking issue that provides an overview of all missing tests for GPT-OSS-120B on Galaxy. It mirrors the structure of [issue #37410](https://github.com/tenstorrent/tt-metal/issues/37410).

## Sub-Issues

The `sub-issues/` directory contains detailed templates for each specific missing test:

### Model Unit Test Pipeline
1. **GPT-OSS-120B-Unit-Individual-Modules.md** - Individual module tests for attention, MLP, RMS norm, experts, etc.
2. **GPT-OSS-120B-Unit-1L-Decoder-PCC.md** - Single layer decoder PCC validation

### Model Demo Pipeline
3. **GPT-OSS-120B-Demo-Evals-Batch-1.md** - Evaluation tests with batch size 1
4. **GPT-OSS-120B-Demo-Evals-Batch-32.md** - Evaluation tests with batch size 32
5. **GPT-OSS-120B-Demo-Long-Prompts.md** - Long context/prompt validation

### Model Param Sweep Pipeline
6. **GPT-OSS-120B-Param-Sweep-Batch-Size.md** - Batch size parameter sweep (1-128)
7. **GPT-OSS-120B-Param-Sweep-Sequence-Length.md** - Sequence length sweep (128-128K)

### Model OP Device Perf Pipeline
8. **GPT-OSS-120B-OP-Device-Perf.md** - Operation-level performance validation

## Usage

These templates can be used to create GitHub issues for tracking the implementation of missing tests. Each template includes:
- Description of the test requirement
- Current status
- Action items (checklist)
- Reference to corresponding Llama-3.3-70B issue

## Creating GitHub Issues

Since these are markdown templates, you can create issues in the GitHub UI by:
1. Go to https://github.com/tenstorrent/tt-metal/issues/new
2. Copy the content from the relevant template file
3. Paste into the issue body
4. Add appropriate labels:
   - `LLMs on Metal`
   - `models-ci`
   - `Tier 1 Model`
   - `P1`
5. Assign to appropriate team member

## Labels to Use

- **LLMs on Metal**: Indicates this is for LLM models
- **models-ci**: Indicates this is CI/testing related
- **Tier 1 Model**: Marks GPT-OSS-120B as a high-priority model
- **P1**: Priority level

## Current Status

GPT-OSS-120B currently has basic demo tests in the Galaxy pipeline (see `tests/pipeline_reorg/galaxy_demo_tests.yaml` line 79-85), but comprehensive testing across all pipelines is needed to match tier 1 model requirements.

## Reference

This documentation mirrors the structure of issue #37410 and its sub-issues:
- Main: https://github.com/tenstorrent/tt-metal/issues/37410
- Sub-issues: #37419, #37420, #37421, #37422, #37423, #37424, #37425, #37426
