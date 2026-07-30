# Gemma-4 26B A4B advisor challenger

Outcome: **NO-CHANGE**. The frozen shipped decoder remains the fastest measured
correct configuration.

The batch-1 incumbent was measured three times with the decoder's own traced
decode harness. Its layer-count-weighted repeats were 1.270303, 1.277361, and
1.270569 ms. The frozen incumbent is therefore the best repeat, 1.270303 ms,
with a 0.007058 ms spread used as the tie floor.

## Capture and reconciliation

The capture target passed the shipped policy explicitly. Sliding attention
traced BF16 QKV/O weights; full attention traced BFP8 QKV/O weights; both
traced BFP8 dense and expert weights. The BF16 DRAM-sharding eligibility flag
was enabled. Each kind considered four eligible dense/attention matmuls and
the advisor recommended DRAM sharding for all four.

Most of that advice was already shipped: packed dense gate/up and dense down
are DRAM-sharded for both kinds, and full-attention O is DRAM-sharded. The
remaining material chains, ranked from the fresh incumbent CSVs, were:

| Chain | Summed decode-window share | Weighted batch-1 result | Decision |
| --- | ---: | ---: | --- |
| Sliding QKV + O | 17.366% | 1.309811 ms | rejected, +0.039508 ms |
| Full QKV | 7.346% | 1.279798 ms | rejected, +0.009495 ms |
| Both chains | pairwise/cumulative | 1.316766 ms | rejected; PCC also failed |

The pairwise set reached decode PCC 0.993277 for sliding attention and
0.983235 for full attention against the incumbent's 0.995 bar. The unchanged
incumbent passed all three real-weight cases after the traceability-only code
cleanup.

`ttnn.sparse_matmul` remains a terminal tracer boundary. Routed expert
gate/up/down occurs in all 30 layers and accounts for 18.318% of the sliding
decode window and 18.226% of the full-attention window. It is recorded as
uncapturable, not silently rejected.

## Artifacts

- `incumbent.json`: ordered freeze, repeats, noise floor, and executed policy
- `shard_advise/<kind>/report.json` and `final_ir.mlir`: one successful capture
  per layer kind
- `reconciliation.json`: shape-matched chain ranking and every measured verdict
- `final.json`: all single/pairwise measured sets and the no-change selection
- `measurements/`: immutable harness and real-weight oracle outputs
- `tracy/`: CSV-only incumbent `tt-perf-report` outputs

The only decoder source change replaces a traced
`Tensor.memory_config()` query with the identical already-declared head-split
memory configuration. It changes no shipped geometry or precision policy.
