# Advisor contribution: full-model decode estimate 22,392.6 → 21,107.7 µs (±22.8 µs)

At decode batch 32, the pinned shard advisor (`618cd4e75d`) contributed a measured
RoPE L1-residency chain. The full 32-layer estimate improves by 1,284.9 µs
(5.74% of the profiled full-model estimate), well outside the ±22.822 µs
uncertainty band. The shipped per-layer confirmation is 0.748458 ms versus the
frozen 0.788610 ms incumbent. Every candidate and confirmation repeat beats
every incumbent repeat.

## Control and capture

- Model config: 32 hidden layers, one meaningful kind: `dense: 32`.
- Incumbent protocol: the required template, 50 untimed warmups and five means
  of 100 traced replays in a fresh process; spread 0.713 µs.
- Executed policy: BFP4 attention/gate-up/down, BFP8 KV, LoFi, 8-core residual
  grid, block widths 12/12/6/16. It came from the executed final-policy log and
  final profiler, not constructor-default metadata.
- Capture: required capture template adapter, batch 32, after the incumbent,
  37 traced ops, one L1 spill. `paged_fused_update_cache` is terminal in the
  pinned tracer; its measured 39.422 µs (5.634%) is reported as unreachable.
- Reconciliation closes at 100% over a single 699.768 µs replay and is not
  degraded. It ranked 70.732 µs/layer of dropped conversions above the 0.433 µs
  floor (`measurable`).

## Candidate and correctness

The repeated top-ranked query/key rows are components of one maximal RoPE L1
chain. The candidate keeps cos/sin, split halves, negate/concat, multiplies,
and adds in L1 on exact batch-dividing 32-core height shards. It removes the
DRAM/retilize round trips without changing dtype, fidelity, or topology.

- First fresh process: median 0.748709 ms, repeats
  `[0.748614, 0.748709, 0.748571, 0.748724, 0.748766]`.
- Fresh confirmation: median 0.748458 ms, repeats
  `[0.748458, 0.748539, 0.748378, 0.748584, 0.748060]`.
- Real-weight oracle: recorded target batch-32 activations plus a matching
  nonzero reference-filled cache against the HF decoder-layer reference;
  PCC 0.998993 at the incumbent 0.995 bar.

The advisor's 11- and 22-core selections were not treated as recommendations.
The measured implementation uses a legal exactly-dividing 32-core geometry,
above both advised counts. Losing/out-of-scope alternatives remain default-off
and are named in `final.json.rejected_knobs`.

## Reported, not screened or attributed

- `advised_boundaries.us_advisor_agrees`: 5.129 µs/layer. This is real time,
  but the advisor endorses those conversions, so its marginal contribution is zero.
- Layer handoff: DRAM entry while output remains L1 costs 1.019 µs/layer,
  31.589 µs/model. It is upstream decoder work because the advisor was not
  asked about inter-layer boundaries.
- Fused-cache terminal share: 39.422 µs/layer, unreachable to this pinned tracer.
- Partial cache/SDPA edge “drops” are invalid soft positional pairs caused by
  that terminal capture. The IR also reports `nlp_concat_heads_decode` as
  unfixable unless its input is sharded; they were not suppressed candidates.

Artifacts include the frozen control, both candidate measurements, bounded
incumbent/candidate perf CSVs, advisor report and IR, generated reconciliation,
real-weight oracle log, and final decision JSON.
