# Stage review 1

Verdict: `more-work-needed`

The fresh independent `$stage-review` subagent inspected the live
`skillexp-work` worktree read-only. It read the stage goal and the optimize,
device-usage, AutoFix, and stage-review skills; inspected the implementation,
tests, candidate JSON, raw/filtered Tracy reports, watcher log, context probe,
README, and work log; and did not open hardware or modify files.

## Required work

### P1: Cross geometry with the selected precision policy

`geom16_b1/b32.json` and `geom32_b1/b32.json` compare geometry under
BFP8/HiFi2, while the selected dense policy is BFP8/LoFi attention plus
BFP4/LoFi MLP. Final BFP4 MLP profiler rows are marked `SLOW`, have no output
subblock, and use `in0_block_w=4/4/6`. Measure QKV, output, gate, up, and down
at batch 1 and 32 under the final policy, including compatible 12-core and
larger legal block/subblock candidates. Reproduce the winning cumulative
default with PCC and profiler evidence.

### P1: Close the residual-topology contradiction

`tracy/selected/dense_decode_b1/filtered.csv` contains four width-sharded
reshards totaling about 6.4 us even though the README says the final dense
profile has no reshard. The material matmuls run on 12 workers while the
selected residual layout uses 16 cores. Measure a coherent 12-core
residual/norm/matmul chain and other compatible layouts. Remove the reshards
or record row-specific blockers and whole-layer comparisons, then correct the
documentation.

### P1: Exercise active-expert prefill and tune the dense expert fallback

`_sparse_moe()` sends every `total_tokens >= 32` to
`_dense_expert_moe_chunk()`. The sequence-33 test labelled active-expert and
all sequence-128 MoE prefill profiles therefore execute dense all-expert
matmuls. Add branch-proof tests and measure active-expert prefill at
representative nonaligned lengths and batches. Separately tune the unavoidable
batch-32 dense path with explicit programs, DRAM-sharded candidates, wider
legal K blocks, memory placement, and output subblocks.

The review accepts the existing AutoFix evidence as a controlled model-local
API limitation for batch-32 active-expert decode. It does not require a shared
TTNN/kernel change inside this restricted stage, but that evidence does not
waive prefill coverage or tuning of the selected fallback.

### P1: Validate precision decisions on target weights

All saved performance candidates use synthetic state. BFP4 dense prefill
measured 0.461 ms versus selected 0.509 ms at batch 1, and BFP4 experts
measured 2.087 ms versus selected 3.330 ms at batch 32. Their PCC vetoes are
synthetic; the real-weight MoE test always constructs BFP8 experts. Run the
disputed policies with real checkpoint weights and target activations,
covering nonaligned prefill, traced decode, cache-consuming replay, and both
batches. Cross BFP4/BFP8 with the same winning expert geometry and decide from
real-target PCC and latency.

## Hard-check gaps

- Save machine-readable correctness output or a full pytest/JUnit transcript;
  PCC values currently appear only in prose.
- Add explicit batch-32 roofline accounting.
- Make current cumulative candidate commands reproducible; old policy JSON
  files do not all contain the current 47-field policy.

## Anomaly ledger

| Anomaly | Evidence | Resolution |
|---|---|---|
| README claims no dense reshards | Four final-profile reshard rows | more work needed |
| Sequence-33 MoE prefill called active-expert | Threshold 32 selects dense expert BMM | more work needed |
| Batch-32 decode is dense all-expert | Sparse alternatives lose and single-card fused API cannot return all routes | controlled model-local limitation |
| Faster BFP4 candidates rejected | Only synthetic-policy vetoes exist | more work needed |

Only a later independent `clean-pass` closes the stage.
