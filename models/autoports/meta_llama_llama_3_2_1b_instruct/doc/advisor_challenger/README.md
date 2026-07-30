# Advisor Challenger

Model: `meta-llama/Llama-3.2-1B-Instruct`  
Decode batch: 32  
Outcome: **no change; the frozen incumbent wins**

## Measured result

The shipped decoder's best warmed traced replay was **0.411884 ms** across five
repeats. Its repeat spread, and therefore the tie floor, was **0.004359 ms**.
The only executable cleanup needed to make the graph advisor-capturable measured
**0.411074 ms** with PCC 1.0. Its nominal 0.000810 ms gain is inside the floor,
so it is a tie and was reverted.

| Measured set | Best ms | Oracle | Verdict |
| --- | ---: | --- | --- |
| frozen incumbent | 0.411884 | batch-32 eager/replay PCC 1.0 | shipped |
| static capture cleanup | 0.411074 | batch-32 eager/replay PCC 1.0 | rejected: tie |

No cumulative or pairwise advisor combinations existed because no advisor chain
survived reconciliation.

## What the advisor found

There is one dense layer kind covering all 16 layers. It was captured once at
batch 32 with the shipped precision: BFP8 attention weights, BFP4 MLP weights,
and BF16 norm weights. The capture considered and advised DRAM sharding for all
five linears.

The incumbent op CSV already shows:

- both RMSNorms L1 width-sharded on 32 cores;
- QKV and output projections using DRAM-sharded weights;
- gate, up, and down projections using DRAM-sharded weights;
- sharded attention and MLP activation chains.

Therefore the advisor's demonstrated recall set minus the shipped graph is
empty. Its remaining differences are geometry proposals, not timing evidence.
No geometry was adopted without a measured win.

The capture reported two recoverable constraint findings:
`rotary_embedding_llama` requires matching input/output memory layouts, and
`nlp_concat_heads_decode` requires a sharded input. Neither layer kind is
terminal or uncapturable; uncapturable layer share is 0%.

## Fidelity and precision

`compute_config` and `math_fidelity` in `final_ir.mlir` were treated as traced
state, not advice. The shipped HiFi2 policy was retained. The final IR's LoFi
entries on BFP4 MLP operations were not incorporated.

## Artifacts

- `incumbent.json`: frozen pre-advisor batch-32 baseline and executed policy
- `tracy/incumbent/incumbent_ops.csv`: sole op-level ranking source
- `shard_advise/dense/report.json` and `final_ir.mlir`: one shipped-precision capture
- `reconciliation.json`: chain-level set difference and screened cleanup
- `final.json`: invariant, measured sets, and no-change selection
- `perf_harness.py`: repeatable batch-32 measurement/oracle harness
- `capture_decoder.py`: shipped-policy advisor capture target
