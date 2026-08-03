# Advisor contribution at decode batch 1

## Full-model estimate: 28,207.286 ± 49.029 us before; 28,207.286 ± 49.029 us after

The measured `$shard-advise` contribution is **0.000 us/model**. No decoder change ships. The estimate is the sum of each bounded one-layer profile window times its model-config layer count: 1 dense/full/forced-RoPE, 36 sliding/RoPE/sparse-MoE, and 12 full/no-RoPE/sparse-MoE layers (49 total). The band is the three incumbent repeat spreads scaled by those counts and summed linearly.

The incumbent was frozen before all captures. `decode_batch`, `requested_decode_batch`, and `capture_batch` are 1. The advisor binary exists at the required `618cd4e75d` pin. The executed policy came from `doc/optimized_decoder/artifacts/final_decode_b1.json`: default candidate, BFP8 attention/cache/expert weights, BFP4 dense gate/up, and BFP8 dense down.

## Measurements and decisions

The dense incumbent was 203.313 us/layer (spread 0.168 us). The complete ranked chain at an exact-dividing geometry measured 234.085 us (233.966–234.382 us), and the required 110-core point above the advisor's 96-core point measured 292.203 us (291.476–292.671 us). Both lose every incumbent repeat (203.169–203.337 us). The down-only DRAM-sharded isolate also lost at 207.422 us (206.654–208.143 us). Full DRAM-sharded chains with legal 64-core and 32-core down projections each stalled before a timed repeat; both fresh-process logs and focused triage reports are retained. No dense candidate ships.

The stalled DRAM-sharded attempts left runtime state contaminated even though `tt-smi -ls --local` still enumerated all four boards. The explicit `tt-smi -r 0 1 2 3` recovery and a 1x1 mesh smoke passed. A fresh post-reset default-policy verification then completed at 203.002 us median (202.648–204.105 us), confirming the default-off knobs did not break the shipped path; this verification is not substituted for the frozen control.

Sliding sparse attention was `aggregate_only`: the advisor-attributable conversion ceiling was 1.688 us/layer versus a 0.936 us floor. The top three chains were attempted together, then isolated. The aggregate cannot execute: rotary requires sharded cos/sin, and concat requires a sharded input. The exact hard errors are retained. A fresh default-topology control measured 613.508–614.089 us, overlaps the incumbent's 613.185–614.121 us, and independently fails non-overlap.

Full/no-RoPE sparse attention was `not_measurable`: the entire 0.563 us/layer ceiling is only 0.45x its 1.263 us floor. Per the feasibility verdict it was not screened, and its contribution is recorded as zero with that arithmetic.

All accounting closes at 100%. The dense/sliding/full profile windows are 179.282/584.728/581.483 us. The sparse captures retain the shipped attention/residual prefix; the pinned tracer rejects exact route-presence `ones_like(TracedTensor)`, and `sparse_matmul` is terminal immediately afterward. Consequently 75.66% of sliding and 76.40% of full sparse windows are explicitly unreachable, covering 48 of 49 model layers' expert suffixes.

## Report-only costs

Advisor-agreed shipped boundaries total 132.628 us/model; this is real conversion time but not marginal advisor contribution and was never screened. Layer handoff costs total 39.919 us/model because sparse layers enter from DRAM and leave in L1; this is also reported upstream and never screened or attributed.

The one-core `NLPCreateQKVHeadsDecodeDeviceOperation` costs about 14.1 us/layer, but the advisor independently agrees with that placement. It belongs to a future `$optimize` pass, not this contribution measurement.

`rejected_knobs` are recorded in `final.json`; every measurement-only policy remains default-off. No oracle was needed for a rejected change. The unchanged shipped decoder retains its official real-weight correctness evidence, recorded as `oracle_weights: real`.
