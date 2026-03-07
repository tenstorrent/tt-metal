# Llama-3.1-8B Blackhole Missing Tests - Issue Documents

## Overview

This document contains all the information needed to create a tracking issue for Llama-3.1-8B missing/failing tests on Blackhole platforms (P100/P150/N150), similar to issue [#37410](https://github.com/tenstorrent/tt-metal/issues/37410) for Llama-3.3-70B Galaxy.

**Reference Issue:** [#37410 - Llama-3.3-70B Galaxy Missing/Failing Tests](https://github.com/tenstorrent/tt-metal/issues/37410)

## Model Information

- **Model:** Llama-3.1-8B-Instruct (meta-llama/Llama-3.1-8B-Instruct)
- **Platform:** Blackhole (P100, P150, N150)
- **Classification:** Tier 1 Model

## Current Test Coverage

### ✅ Tests Currently Running

**Single-card Demo Tests (P100/P150):**
- llama3-8b performance test

**Multi-card Demo Tests (BH-LLMBox):**
- Performance tests: batch-1, batch-32
- Data-parallel performance and stress tests
- TP device performance (decode/prefill)

**Nightly Tests (P100/P150):**
- Stress test (performance-ci-stress-1)

**vLLM Tests:**
- Multiple configurations for WH-T3K and BH variants

### ❌ Missing Test Coverage

The following test categories need to be added or fixed:

1. **Unit Tests**
   - Individual Module tests
   - 1L Decoder PCC tests

2. **Demo Tests**
   - Evals-batch-1 (repeat batches with perf output)
   - Evals-batch-32 (repeat batches with perf output)
   - 128sl-accuracy (top-1/top-5 token accuracy)

3. **Parameter Sweep Tests**
   - Sequence length sweep (128 to 128K)
   - Sampling features: Seed, Logprobs, Penalties, n
   - Prefetcher off test

4. **Performance Tests**
   - Comprehensive OP Device Perf pipeline

---

## Main Tracking Issue

### Title
```
[Models CI] [Llama-3.1-8B Blackhole] Missing/Failing Tests
```

### Labels
- `LLMs on Metal`
- `models-ci`
- `Tier 1 Model`

### Content

```markdown
This issue contains all the missing Llama-3.1-8B tests on Blackhole (P100/P150/N150). Since this is a tier 1 model, all of the tests listed in this issue are required and thus need to be fixed/added.

Each missing test will have its correspondent sub-issue assigned.

Consult the [internal Matrix spreadsheet](https://docs.google.com/spreadsheets/d/1a75kfIoVXrlbri0BtwKJsFJxj6Qa8mbJENPZGgvJjPA/edit?usp=sharing) for the latest test status.

# Required tests

If the tests have an issue attached and the issue is not marked as closed, it requires intervention.

## Model Unit Test Pipeline
- [Issue for Individual Module tests]
- [Issue for 1L Decoder PCC]

## Model Demo Pipeline
- [Issue for Evals-batch-1]
- [Issue for Evals-batch-32]
- [Issue for 128sl-accuracy]

## Model Param Sweep Pipeline
- [Issue for All seqlen sweep]
- [Issue for Sampling features and Prefetcher]

## Model OP Device Perf Pipeline
- [Issue for OP device Perf]

## Other Pipelines
- [x] vLLM Nightly (Jobs running for various BH configurations)
- [x] Stress (Job from nightly tests on P100/P150)
- [TDB] PR/Merge Gate
```

---

## Sub-Issues

Create the following 8 sub-issues and then update the main issue with their links.

### 1. [Unit] Individual Module tests

**Title:** `[Llama-3.1-8B Blackhole] [Unit] Individual Module tests`  
**Labels:** `P1`, `models-ci`, `Tier 1 Model`  
**Type:** Task

**Content:**
```markdown
Llama-3.1-8B is missing unit tests on the Blackhole pipeline.

We should first refactor the pipeline and then re-add the tests as they were once there.
```

### 2. [Unit] 1L Decoder PCC

**Title:** `[Llama-3.1-8B Blackhole] [Unit] 1L Decoder PCC`  
**Labels:** `P1`, `models-ci`, `Tier 1 Model`  
**Type:** Task

**Content:**
```markdown
Llama-3.1-8B is missing unit tests on the Blackhole pipeline.

We should first refactor the pipeline and then re-add the tests as they were once there.
```

### 3. [Demo] Evals-batch-1

**Title:** `[Llama-3.1-8B Blackhole] [Demo] Evals-batch-1`  
**Labels:** `P1`, `models-ci`, `Tier 1 Model`

**Content:**
```markdown
Llama-3.1-8B is missing the required Evals-batch-1 (repeat batches with perf output) on the Blackhole Model Demo pipeline.

We should first refactor the pipeline and then add the test.
```

### 4. [Demo] Evals-batch-32

**Title:** `[Llama-3.1-8B Blackhole] [Demo] Evals-batch-32`  
**Labels:** `P1`, `models-ci`, `Tier 1 Model`

**Content:**
```markdown
Llama-3.1-8B is missing the required Evals-batch-32 (repeat batches with perf output) on the Blackhole Model Demo pipeline.

We should first refactor the pipeline and then add the test.
```

### 5. [Demo] 128sl-accuracy

**Title:** `[Llama-3.1-8B Blackhole] [Demo] 128sl-accuracy`  
**Labels:** `P1`, `models-ci`, `Tier 1 Model`

**Content:**
```markdown
Llama-3.1-8B has the top-1/top-5 accuracy job running, but it needs to be validated or added if not present.

Check the test output for the specific test and the error, which may not get picked up at the pipeline level.

We also need to refactor the test into the new Pipeline if not already done.
```

### 6. [Param Sweep] All seqlen sweep

**Title:** `[Llama-3.1-8B Blackhole] [Param Sweep] All seqlen sweep`  
**Labels:** `P1`, `models-ci`, `Tier 1 Model`  
**Type:** Task

**Content:**
```markdown
Llama-3.1-8B is missing the complete seqlen sweep, from 128 up to 128K.

We should first refactor the pipeline and then add the tests.
```

### 7. [Param Sweep] Sampling features and Prefetcher

**Title:** `[Llama-3.1-8B Blackhole] [Param Sweep] Sampling features and Prefetcher`  
**Labels:** `P1`, `models-ci`, `Tier 1 Model`  
**Type:** Task

**Content:**
```markdown
Llama-3.1-8B is missing the following sweep sampling param tests:
- Seed
- Logprobs
- Penalties
- n
- Prefetcher off

We should first refactor the pipeline and then add the tests.
```

### 8. OP device Perf

**Title:** `[Llama-3.1-8B Blackhole] OP device Perf`  
**Labels:** `P1`, `models-ci`, `Tier 1 Model`  
**Type:** Task

**Content:**
```markdown
Llama-3.1-8B is missing the required OP Device perf pipeline.

We should first refactor the pipeline and then add the test.
```

---

## Instructions for Creating Issues

### Step 1: Create Main Issue
1. Go to https://github.com/tenstorrent/tt-metal/issues/new
2. Use the title and content from the "Main Tracking Issue" section above
3. Apply the specified labels
4. Create the issue and note its number

### Step 2: Create All 8 Sub-Issues
For each sub-issue (1-8):
1. Go to https://github.com/tenstorrent/tt-metal/issues/new
2. Use the title and content from the respective sub-issue section
3. Apply the specified labels and type
4. Create the issue and note its number

### Step 3: Update Main Issue with Sub-Issue Links
1. Edit the main tracking issue
2. Replace the placeholder text with actual issue links:
   - `[Issue for Individual Module tests]` → `https://github.com/tenstorrent/tt-metal/issues/XXXXX`
   - Repeat for all 8 sub-issues

---

## Related Files and Resources

### Test Files
- `models/tt_transformers/demo/simple_text_demo.py` - Main demo entry point
- `models/tt_transformers/tests/test_device_perf.py` - Device performance tests
- `models/common/demos/llama31_8B_demo.py` - TTTv2 MLP1D demo

### Workflow Files
- `.github/workflows/blackhole-demo-tests-impl.yaml` - Single card demo tests
- `.github/workflows/blackhole-multi-card-demo-tests-impl.yaml` - Multi-card tests
- `.github/workflows/blackhole-nightly-tests-impl.yaml` - Nightly tests
- `.github/workflows/vllm-nightly-tests-impl.yaml` - vLLM tests

### External Resources
- [Internal Test Matrix Spreadsheet](https://docs.google.com/spreadsheets/d/1a75kfIoVXrlbri0BtwKJsFJxj6Qa8mbJENPZGgvJjPA/edit?usp=sharing)
- [Reference Issue #37410](https://github.com/tenstorrent/tt-metal/issues/37410)

---

## Summary

This document provides all necessary information to create a comprehensive tracking issue for Llama-3.1-8B missing tests on Blackhole, following the same structure as the Llama-3.3-70B Galaxy tracking issue. The main difference is the target platform (Blackhole vs Galaxy) and the test focus (single-chip + data-parallel vs multi-chip mesh).
