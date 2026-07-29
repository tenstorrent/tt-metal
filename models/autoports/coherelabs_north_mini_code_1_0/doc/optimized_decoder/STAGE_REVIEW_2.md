# Stage review 2

Verdict: `more-work-needed`

The fresh independent `$stage-review` xhigh subagent reviewed checkpoint
`03b1b0078f1` read-only. It inspected the implementation, tests, candidate
JSON, profiler rows, correctness/JUnit transcripts, watcher evidence, context
contract, and first-review closure.

## Required work

### P1: Equivalent authentic BFP4 expert validation

`test_optimized_real_weight_moe_decode` covers authentic eager batch-1
position-zero decode, while non-aligned prefill coverage still uses synthetic
state and the performance harness measures synthetic state. The saved
`real_validated_bfp4_expert_*.json` files therefore contain matched latency
but no authentic PCC.

Run and preserve BFP4 real-target evidence for layers 1 and 4 covering
non-aligned prefill, traced/cache-consuming decode, and batches 1 and 32.
Select BFP4 if those pass; retain BFP8 only if equivalent authentic evidence
reproduces a failure.

## P1: DRAM-sharded expert matmul attempt

The selected batch-32 dense-expert matmuls use DRAM-interleaved rank-3 expert
weights and generic matmul programs. Final profiler rows are not DRAM-sharded,
use `in0_block_w=1`, and dominate batch-32 MoE latency. Existing explicit
64/80/100-core candidates use
`MatmulMultiCoreReuseProgramConfig`, not a DRAM-sharded expert family.

Adapt expert weight packing/rank and sweep legal DRAM-sharded BFP4/LoFi
geometries at both batches, or preserve a minimal adapted repro proving that
the batched expert contract cannot use DRAM-sharded weights.

## Closed findings

The reviewer accepted the residual-reshard correction, active-expert branch
proof, non-aligned multi-user coverage, context capacity, machine-readable
correctness transcript, and watcher-clean run.

Only a later independent `clean-pass` closes the stage.
