# Advisor contribution: 24,949.218 → 22,397.946 ± 74.289 µs

At decode batch 1, shard advice removes an estimated **2,551.272 µs/model** from
the already-optimized decoder: 24,949.218 µs before and 22,397.946 µs after,
with a conservative ±74.289 µs band. The separate derived host-wall estimate is
27.634930 → 25.075706 ms (median of the composite repeats; the sum of per-kind
medians is 25.076043 ms). The shipped change is a 32-core L1-width-sharded
RMSNorm for all 48 sparse-MoE layers.

The pinned advisor was `/home/mvasiljevic/tt-mlir` at `618cd4e75d`. The config
has 49 layers: 1 dense full-attention, 36 sliding-attention MoE, and 12
full-attention MoE. Every incumbent and capture used requested, measured, and
capture batch 1. The incumbent measurements preceded all captures. Timing used
the supplied `harness_template.py` protocol through model hooks: 10 warmups,
five blocks, and 50 trace replays per block, each configuration in a fresh
process. The shipped policy and dtypes came from the exact-final optimized
decoder runs, not constructor defaults.

The frozen policy records every execution-relevant dtype and placement flag,
including `advisor_moe_norm_cores: 0`; it does not rely on the now-changed
meaning of the symbolic `default` policy name. Its provenance is the three
exact-final decode profiler CSVs plus the policy object that executed them.
Every measured candidate is constructed with `dataclasses.replace` from the
shipping `default` policy, changing only `advisor_moe_norm_cores`; the timing
JSON records the complete executed dataclass and invocation.

## Reconciliation and decision

| Kind | Device estimate | Band | Advisor-drop ceiling | Verdict |
|---|---:|---:|---:|---|
| Dense full attention ×1 | 153.390 µs | ±0.849 µs | 3.476 µs | no attributable measurable candidate |
| Sliding MoE ×36 | 18,827.352 → 16,678.504 µs | ±58.968 µs | 2,148.848 µs measured saving | ship norm32 |
| Full MoE ×12 | 5,968.476 → 5,566.052 µs | ±14.472 µs | 402.424 µs measured saving | ship norm32 |

All three tool-generated reconciliations close at 100% and are not degraded.
The pinned tracer cannot accept traced inputs at `paged_fused_update_cache`,
and `sparse_matmul` is terminal, so captures preserve the executed graph only
up to those declared boundaries. Untraced shares are 38.85% dense, 68.96%
sliding MoE, and 67.90% full MoE; advisor reach is not implied beyond them.

Dense reconciliation ranked a 1.419 µs boundary row as
`rotary_embedding -> linear`. This is a soft name/position pairing across the
declared paged-cache/SDPA capture gap. The shipped profile's row is actually the
`concat_heads -> O projection` 8-core L1 regrid, while the authoritative
`shard_advise/dense_full_attention/final_ir.mlir` has no captured or advised
edge for that region. Screening it would test a hand-authored optimization the
advisor never proposed, contaminating the contribution. The remaining dense
rows are each below the 0.849 µs floor. The MoE `ReshapeView`, retilize, and
fill-pad unresolved time lies after the declared sparse terminal and is not
silently credited to advice.

The `not_measurable` verdicts stopped all ranked conversion-chain screening.
The norm sweep is separate: `material_ops_on_le_2_cores` explicitly identified
the advisor-attributable one-core norm and required either a measured attempt
or a hard error. Measuring that mandatory material-op candidate is therefore
not a chain screen below the feasibility floor.

The advisor independently agreed with 71.968 µs/layer of dense work,
25.700 µs/layer of sliding-MoE work, and 25.492 µs/layer of full-MoE work.
`advised_boundaries.us_advisor_agrees` is 1.458 µs/model of explicit conversion
time and is reported but not screened. The one-core
`nlp_create_qkv_heads_decode` cost is likewise reported to `$optimize` because
the advisor agrees with that placement. In contrast, the advisor changes the
one-core MoE RMSNorm to a 22-core width-sharded placement, making the measured
norm direction attributable here. `model_estimate.layer_handoff` reports the dense
0.886 µs entry conversion but no redundant per-model handoff charge; it was not
screened or booked as contribution.

The advisor's 22-core norm improved both kinds but lost to the required
above-advice sweep. The exactly-dividing 32-core grid won: sliding MoE
0.578076 → 0.518386 ms and full MoE 0.553659 → 0.520124 ms. The 64-core grid
regressed. Every 32-core repeat beat every frozen-incumbent repeat, and fresh
process confirmations were 0.518440 and 0.519718 ms. Official layer-1 weights
passed the sliding oracle at PCC 0.999719. Because layer-4 checkpoint shards
were not cached, the full-attention MoE oracle transparently remapped the same
official layer-1 tensors to the layer-4 path and passed at PCC 0.999526 (bar
0.995).
The 22/64 policies and capture switch remain default-off.

Each winning kind has its own bounded one-replay op-level profile under
`profiles/<kind>_norm_32/perf_report.csv`. The timing records name these files
and record their own oracle result; profiler runs were separate from timing.
