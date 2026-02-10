# [GPT-OSS-120B Galaxy] [Demo] Evals-batch-1

GPT-OSS-120B is missing the required Evals-batch-1 (repeat batches with perf output) on the Galaxy Model Demo pipeline.

## Description
This test should run evaluation workloads with batch size 1, repeating the workload multiple times to measure stable performance metrics.

## Current Status
The current Galaxy demo pipeline has basic GPT-OSS tests but lacks comprehensive evaluation metrics for batch-1 workloads.

## Action Items
- [ ] Implement Evals-batch-1 test for GPT-OSS-120B
- [ ] Ensure performance metrics are captured
- [ ] Add to Galaxy demo test pipeline
- [ ] Set appropriate performance targets

## Reference
Similar to Llama-3.3-70B issue: https://github.com/tenstorrent/tt-metal/issues/37421
