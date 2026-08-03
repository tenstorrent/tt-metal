# Full-model estimate: 38,887.6 ± 80.9 us before; 38,095.8 ± 80.9 us after

At decode batch 1, `$shard-advise` adds a measured **791.7 us/model (2.04%)**
to the already-optimized Gemma-4 26B A4B decoder. The shipped change keeps the
concat-heads output sharded into the output projection. This is a derived
full-model estimate, not an end-to-end full-model measurement: 25
sliding-attention and 5 full-attention layer windows are
weighted using the model's explicit 30-entry `layer_types` list.

## Frozen controls and measured candidate

All `decode_batch`, `requested_decode_batch`, and `capture_batch` values are
1. The controls were frozen before capture. Sliding measured 1.341153 ms with
repeats 1.339828–1.342632 ms; full attention measured 1.539374 ms with repeats
1.538430–1.540587 ms. The advisor-directed, exactly dividing 88-core norm
candidate measured 1.346942 ms sliding and 1.549943 ms full. It was slower for
both kinds and is rejected/default-off. A 110-core point above the advised
count is illegal because hidden width 2816 cannot form tile-aligned shards.

The winning medians are 1.318449 ms sliding and 1.494548 ms full attention;
every candidate repeat beats every incumbent repeat, and fresh-process
confirmations preserve strict separation. A final override-free shipped-default
run also stays strictly separated: 1.319063 ms sliding and 1.494706 ms full.
The headline subtracts measured
per-layer deltas times 25 and 5 from the reconciliation windows. The
conservative band is linear: 70.100 + 10.785 = 80.885 us.

## Reconciliation and scope

The pinned advisor is `618cd4e75d`; it was not rebuilt. Both generated
reconciliations close at 100% and are not degraded. The sparse-MoE suffix is
tracer-terminal at `sparse_matmul`, leaving 58.68% of the sliding window and
51.42% of the full window untraced; no contribution is claimed there.

Sliding reconciliation is `aggregate_only`: its 4.789 us total attributable
conversion ceiling is 1.71x the 2.804 us control spread, while no individual
chain clears the floor. Full attention is `measurable`: 8.448 us total versus
a 2.157 us spread. The material one-core LayerNorm disagreement was measured
on the advisor's exact-dividing 88-core direction and lost for both kinds.
Post-screen verdicts live in `evidence_{sliding,full}_attention.json` and are
validated and merged by `reconcile.py --evidence`; the reconciliation outputs
are freshly tool-generated rather than hand-authored.

`advised_boundaries.us_advisor_agrees` is 6.676 us/layer sliding and 4.012
us/layer full. It is real time, reported but not screened or credited.
`model_estimate.layer_handoff` is likewise reported only: DRAM-to-L1 entry
costs 0.921 us/layer sliding and 0.904 us/layer full. The unresolved FillPad
and ReshapeView rows remain structural/shape operations in the captured IR;
they are not booked as advisor contribution.

The shipped decoder passes the real-weight HuggingFace oracle: sliding
prefill/decode PCC is 0.998651/0.999707 and full attention is
0.997775/0.999868, all above 0.995. `GEMMA4_ADVISOR_NORM_CORES` stays
default-off at 8; the sharded-SDPA extension is also rejected/default-off
because the kernel reports `Sharded output not supported for GQA`.

Candidate op-level evidence is bounded to one eager replay with the fixed
`PERF_DECODE`/`PERF_DECODE_END` signposts in `tracy/candidate_sliding_ops.csv`
and `tracy/candidate_full_ops.csv` (77 and 79 device rows). Timing decisions
remain exclusively from fresh processes using `scripts/harness_template.py`;
the eager replay is profile-only because this Tracy build does not associate
cached trace-replay device rows with host signposts.
