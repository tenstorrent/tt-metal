# Phi-3.5 Mini Advisor Challenger

Outcome: **no advisor change shipped**. At decode batch 32, the frozen incumbent's best repeat was
`1.100042574 ms`; the final incumbent remeasurement was `1.099775368 ms`. Their `0.000267206 ms`
difference is inside the incumbent's `0.000375528 ms` repeat spread, so the governing tie rule keeps the
incumbent.

## Batch-32 prerequisite

The supplied experiment snapshot only executed decode batch 1. Before the advisor ran, the decoder was
made batch-configurable and its Q/K/V decode sharding was generalized to a legal batch-32 `8x4` core
grid. That executable batch-32 decoder was then frozen and profiled. The advisor did not influence this
prerequisite.

The incumbent measurement used five groups of 100 traced replays:

`[1.100418102, 1.100042574, 1.100290455, 1.100304183, 1.100239046] ms`.

Policy provenance is `tracy/incumbent_decode_perf_report.csv` plus
`doc/datatype_sweep/selected_precision_config.json`, not constructor defaults.

## Capture

Phi has one layer kind: 32 dense decoder layers. It was captured once at batch 32 with BFLOAT8_B
attention weights, BFLOAT4_B MLP weights, and BFLOAT16 norms. The capture considered and advised all
four DRAM-shardable matmuls. It produced 39 ops, 36 final choices, one spill, and the known
`nlp_concat_heads_decode` unfixable warning.

The capture target asserts the TTNN weight dtypes before tracing. The pinned advisor serializes both
block-float widths as `bfp_bf8` in MLIR even though the asserted MLP tensors are BFLOAT4_B; this version
skew is recorded in `report.json`. Compute fidelity in `final_ir.mlir` was treated as traced state and
was not adopted as advice.

## Chain screening

All shares come from the frozen `996.910 us` device window:

| Chain | Summed share | Best measured candidate | Result |
| --- | ---: | ---: | --- |
| MLP DRAM-sharded family | 35.436% | 1.100430042 ms | Rejected; the shipped source already uses this family and the remeasurement did not beat the frozen best |
| RoPE L1 chain | 17.947% | 1.116339359 ms | Rejected; the full geometry is illegal at Phi's 48-wide split-half shard, and the legal L1-tail isolate was slower |
| Norm sharding | 1.412% | 1.149981376 ms | Rejected; 11 cores is illegal, and the nearest legal lower bracket (8 cores) was slower |

Exact repeats and constraint failures are retained in `screening_measurements.json`.

The mandated `scripts/reconcile.py` command was run. Its current parsers returned zero because this
MLIR uses `#ttnn_layout` references and this v2.1 report uses `Device Time`; `reconciliation.json`
contains the corrected chain parse from those same authoritative inputs.

No chain survived screening, so cumulative and pairwise combinations were empty by construction. The
only measured best set is the incumbent.

## Correctness and final selection

The shipped decoder passed the repository oracle at PCC `>= 0.995`:

- prefill PCC: `0.9999909379944513`
- decode PCC: `0.9998426082085206`

The final decoder retains the frozen policy and adds batch-32 execution support plus a tracer-safe
declared MLP memory config. No advisor geometry is enabled.
