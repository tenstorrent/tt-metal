# B1 local findings: routing placement and source-flow controls

Balanced expert placement consistently reduced isolated Combine kernel time on both tested layers, while Dispatch changed much less. All 12 fresh-process captures passed coverage validation: two processes for each of six layer/routing cases. This advances the controlled-routing question in B1; it does not yet explain the historical full-model timing gap.

## Measurements

Each value is the mean of two process-level means, followed by the range of those two means. Within each process, discard iteration 0, then take the maximum kernel duration across eight devices for each of nine retained iterations and average. Iterations are not pooled as independent process runs. All times are milliseconds.

| Layer | Routing | Dispatch mean [run range] | Combine mean [run range] |
|---:|---|---:|---:|
| 18 | captured | 1.659080 [1.656553–1.661606] | 1.723374 [1.720989–1.725759] |
| 18 | placement_balanced | 1.624431 [1.621753–1.627108] | 1.062319 [1.059503–1.065135] |
| 18 | source_shuffled | 1.660601 [1.658265–1.662938] | 1.653402 [1.652474–1.654330] |
| 23 | captured | 1.641102 [1.637837–1.644367] | 1.477407 [1.475933–1.478881] |
| 23 | placement_balanced | 1.606123 [1.606034–1.606213] | 1.095940 [1.092841–1.099039] |
| 23 | source_shuffled | 1.626815 [1.625332–1.628298] | 1.478913 [1.474195–1.483632] |

## What changed

| Layer | Balanced versus captured: Dispatch | Balanced versus captured: Combine | Source-shuffled versus captured: Dispatch | Source-shuffled versus captured: Combine |
|---:|---:|---:|---:|---:|
| 18 | -2.09% | -38.36% | +0.09% | -4.06% |
| 23 | -2.13% | -25.82% | -0.87% | +0.10% |

Negative percentages mean shorter kernel duration. The much larger Combine response supports expert placement as a useful investigation target. It does not establish destination imbalance alone as the mechanism: placement also changes per-chip tile-rounded expert-region load, expert ordering, token destination fanout, and locality.

Source shuffling preserves expert counts, destination assignment totals, and the destination-fanout histogram, but changes source-to-destination traffic and per-expert ordering/batching. Its L18 Combine response shows aggregate destination counts alone are insufficient to predict cost. The L23 Combine response is small compared with its process-to-process range; two process repetitions do not support a fine-grained statistical claim.

## B1 questions answered and remaining

- **Controlled routing experiment: completed.** Routing and new isolated timings are paired case by case, with an explicit full-128-expert SP8/TP1 placement. Placement affects Combine cost reproducibly across these two process runs.
- **Actual routing paired with historical PP layer timings: still open.** The recovered selections came from another TP execution. These PP-placement replays are counterfactual and cannot explain the old L18/L23 timing difference directly.
- **Per-device load versus timing: evidence recorded, mechanism still open.** RUN_TIMINGS.csv/timing.json preserve device duration spread and full per-device samples; host_comparison.json and cases preserve assignment/flow/fanout metrics. Counts are not measured network bytes. Next discriminate destination load, tile padding, ordering and sourceflow, rather than attribute all effects to a single imbalance score.
- **Production optimization: not established.** This microbenchmark substitutes a layout conversion for expert FFN, omits shared-expert concurrency and downstream weighted reduction, and does not check numerical outputs. A real placement change must relocate expert weights consistently and pass numerical correctness, then matched traced TP and PP4 throughput tests.

## Validation and recovery caveats

All successful captures use the identical archived worker SHA256: `ec4b970740f4a02cd4bac913911dabf589bd8eb2b2af4c7224a02940b0709e01`. Each capture validates eight devices × ten iterations × two target operations, finite positive durations, matching START/END signposts, and per-device operation/call sequence. The 12 captures contain 1920 validated target rows; 1728 remain after discarding warm-ups.

Failed imports, interrupted/stalled attempts and firmware-initialization failures are excluded. Targeted recovery was performed between some successful repeats. Thus process repetitions also differ in recovery state. The post-recovery manager-free success does not isolate manager removal from recovery as the cause of the earlier stall. These are local eight-device Galaxy results, not LoudBox measurements.

Artifacts: [per-run table](MATRIX_RESULTS.md), [machine-readable aggregate](matrix_timings.json), [validation log](VALIDATION.md), [paired full-model capture plan](PAIRED_CAPTURE_PLAN.md). Reproduce the matrix with `python3 aggregate_timings.py`. No GitHub issue or message has been posted.
