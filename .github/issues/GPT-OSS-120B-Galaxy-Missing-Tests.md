# [Models CI] [GPT-OSS-120B Galaxy] Missing/Failing Tests

This issue contains all the missing GPT-OSS-120B tests on WH Galaxy. Since this is a tier 1 model, all of the tests listed on this issue are required and thus need to be fixed/added.

Each missing test will have its correspondent sub-issue assigned.

Consult the [internal Matrix spreadsheet](https://docs.google.com/spreadsheets/d/1a75kfIoVXrlbri0BtwKJsFJxj6Qa8mbJENPZGgvJjPA/edit?usp=sharing) for the latest test status.

## Required tests

If the tests have an issue attached and the issue is not marked as closed, it requires intervention.

### Model Unit Test Pipeline
- [ ] [GPT-OSS-120B Galaxy] [Unit] Individual Module tests
- [ ] [GPT-OSS-120B Galaxy] [Unit] 1L Decoder PCC

### Model Demo Pipeline
- [ ] [GPT-OSS-120B Galaxy] [Demo] Evals-batch-1
- [ ] [GPT-OSS-120B Galaxy] [Demo] Evals-batch-32
- [ ] [GPT-OSS-120B Galaxy] [Demo] Long Prompts

### Model Param Sweep Pipeline
- [ ] [GPT-OSS-120B Galaxy] [Param Sweep] Batch Size Sweep
- [ ] [GPT-OSS-120B Galaxy] [Param Sweep] Sequence Length Sweep

### Model OP Device Perf Pipeline
- [ ] [GPT-OSS-120B Galaxy] [OP Perf] Performance Targets

### Other Pipelines
- [ ] vLLM Nightly
- [ ] Stress
- [ ] PR/Merge Gate

## Labels
- `LLMs on Metal`
- `models-ci`
- `Tier 1 Model`

## Notes
This tracking issue mirrors the structure of issue #37410 (Llama-3.3-70B Galaxy Missing/Failing Tests) but targets the GPT-OSS-120B model.

Current status: GPT-OSS-120B tests exist in the Galaxy demo pipeline but require expansion to match tier 1 model requirements.
