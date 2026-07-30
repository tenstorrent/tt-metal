# Gemma 4 26B A4B advisor challenger

Result: **no performance-policy change**. The frozen incumbent remains the
winner. Sliding attention's best of three repeats is 1.269832 ms with a
1.063 µs spread; full attention's best is 1.271294 ms with a 4.278 µs spread.

The capture ran once for each layer kind at the shipped precision and batch 1.
The BF16 DRAM-sharded-matmul eligibility option was enabled. Each report
considered all four dense matmuls. The shipped math fidelities were retained
for every measurement; `compute_config` in `final_ir.mlir` was treated as
traced state, not advice.

## Reconciliation and measurements

The advisor agreed with the incumbent's packed gate/up and MLP-down DRAM
sharding for both kinds and its full-attention O sharding. Material
disagreements were measured one variable at a time:

| Challenger | Kind | Result | Decision |
| --- | --- | ---: | --- |
| QKV DRAM sharding | sliding | 1.343102 ms | reject, +73.270 µs |
| QKV DRAM sharding | full | 1.301563 ms | reject, +30.269 µs |
| O DRAM sharding | sliding | 1.272889 ms | reject, +3.057 µs and outside noise |
| coherent R11 residual/norm sharding | sliding | 1.169551 ms | reject: real-weight decode PCC 0.994795 < 0.995 |
| coherent R11 residual/norm sharding | full | 1.362680 ms | reject, +91.386 µs |

The incumbent was re-run after capture at 1.272548 ms sliding and 1.268760 ms
full. Per the frozen-best invariant, a later incumbent rerun does not replace
the original best repeat. The shipped real-weight oracle passed all three
cache/layer cases: sliding decode 0.999617 and full decode 0.999754.

`reconcile.py` was run for both layer kinds. Its legacy parser did not recognize
the current symbol-table-based `final_ir.mlir` layouts or the profiler's
microsecond `Device Time` column, so the resulting rows were normalized from
the same authoritative IR, `report.json`, and copied profiler CSVs into
`reconciliation.json`. No disagreement is rejected in prose only.

## Uncapturable work

Routed experts terminate the tracer at `ttnn.sparse_matmul`. They occur in all
30 layers: 25 sliding-attention layers and 5 full-attention layers. The capture
therefore covers the shipped attention and dense-MLP prefix and explicitly
excludes the routed-expert suffix.

No re-capture was performed because no topology rewrite was kept. The decoder
only replaces dynamic `memory_config()` queries with the already-declared
QKV-head memory config, which is runtime-equivalent and makes the shipped graph
traceable by the advisor.
