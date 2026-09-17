# B1: paired routing and timing in one run — load, not ordering

September 17. Supersedes the provenance gap in [PAIRED_CAPTURE_PLAN.md](../2026-09-11/b1-controlled-routing/PAIRED_CAPTURE_PLAN.md):
routing and device timings now come from the same full-model execution, with verified layer labels.

## Result

Across the nine layers rank 2 owns (global L18–L26), routing imbalance predicts Combine cost:

    Combine = 0.055 + 1.370 x (destination max/mean)    r = +0.971, R^2 = 0.943, n = 9

Dispatch correlates far more weakly (r = +0.671). The intercept is ~0, so Combine cost is close to
proportional to the busiest destination chip's load. Extrapolating to perfect balance gives 1.42 ms
(-65% for L18, -37% for L23) — a straight-line guess beyond the measured range (lowest observed
imbalance 1.63), not a predicted speedup.

| layer | imbalance | busiest chip | Combine | Dispatch | FFN |
|-------|-----------|--------------|---------|----------|-----|
| L18 | 2.92 | 7488 | 4.079 ms | 2.612 ms | 2.879 ms |
| L24 | 2.80 | 7180 | 3.708 ms | 2.497 ms | 2.873 ms |
| L20 | 2.40 | 6145 | 3.502 ms | 2.377 ms | 2.817 ms |
| L21 | 2.21 | 5648 | 3.284 ms | 2.378 ms | 2.523 ms |
| L25 | 2.19 | 5608 | 2.970 ms | 2.313 ms | 2.642 ms |
| L19 | 2.16 | 5520 | 3.093 ms | 2.442 ms | 2.514 ms |
| L22 | 2.05 | 5249 | 2.804 ms | 2.268 ms | 2.665 ms |
| L23 | 1.77 | 4520 | 2.277 ms | 1.770 ms | 2.309 ms |
| L26 | 1.63 | 4176 | 2.350 ms | 2.369 ms | 2.352 ms |

Timings are per-operation maxima across the 8 devices, measured chunk only (the WARMUP pass is
excluded). They are not layer wall-clock latency.

## Ordering is not a confound

`chip = expert_id // 16` and `slot = expert_id % 16`, so chip assignment and slot position always
move together in the model — no prior measurement could separate them. Permuting expert ids WITHIN
each 16-block moves slot only, leaving per-chip load, source->destination traffic, fanout, locality
and padded totals byte-identical (asserted in `prepare_cases.py`).

| Combine, same harness | L18 | L23 |
|-----------------------|-----|-----|
| balanced placement (2026-09-11) | -38.36% | -25.82% |
| source shuffle (2026-09-11) | -4.06% | +0.10% |
| **slot permutation** | **+0.40%** | **+0.01%** |
| same-case repeat (noise floor) | +0.09% | - |

Slot permutation is ~100x below the placement effect and ~10x below the source shuffle on L18, and
at the noise floor on L23. The r=0.971 correlation is a load effect, not slot position in disguise.

## Expert popularity is stable across chunks

The run requests 20 chunks of 5,120 tokens, but the golden trace holds 56,320 = exactly 11 x 5,120,
and `_load_token_pool` TILES a short pool (`pool * -(-num_tokens // len(pool))`). Chunks 11-19 therefore
replay chunks 0-8's token ids. Only chunks 0-10 are distinct text, and the figures below use those.

Per-expert counts correlate r = 0.926–0.990 between any two distinct chunks; the 10 busiest experts
overlap 7.7–9.3 out of 10. Each layer's imbalance stays in a narrow band (L18 2.92–3.64, L23
1.17–1.77) and the ordering between layers never inverts. Over all 20 chunks the correlation is
slightly LOWER (0.918–0.988), not higher, which is the tell that the tiled chunks are not duplicates
in routing terms.

They are not, because routing depends on position as well as token: chunk 11 replays chunk 0's exact
token ids at a different KV offset and only 21.2% (L18) / 20.0% (L23) of the per-token selections
match, while per-expert counts still correlate 0.848 / 0.905. Aggregate popularity survives an almost
total change of individual selections, which points at the gate weights rather than the token stream.

A static placement is therefore worth attempting. Caveat: all chunks come from ONE document, so this
shows stability across different text within a prompt, not across unrelated prompts.

## Reproducibility

The replay harness reproduces 2026-09-11: L18 captured Combine measured 1.719 and 1.720 ms here
against 1.723 ms then (0.2%), and L23 captured Combine measured 1.477 ms against 1.477 ms. Different
checkout, rebuilt board, worker recovered from an archived `worker.py.gz`.

## Scope and limits

- One request, one chunk for the timings; nine layers is nine points, not a general law.
- Correlation, not causation. Balanced placement still moves load, tile padding, fanout and locality
  together; only ordering has been isolated and excluded.
- The replay excludes FFN, shared expert and the weighted reduction, and checks no numerical output.
  It measures dispatch/combine cost, not model throughput.
- One permutation seed, and the slot-permutation test covers only L18 and L23.
- A production change still needs expert weights actually moved, correctness checked, and matched
  traced PP4 throughput.

## Files

- `routing_counts_L18_L26_20chunk.json` — per-layer, per-chunk expert counts and destination load.
  Raw indices (16 MB) are not committed; `TT_MOE_ROUTING_CAPTURE=18,...,26` regenerates them.
- `compare_layer_timings.py` — per-layer Dispatch/Combine/FFN from an `ops_perf_results` CSV pair.
- `run_b1_replay.sh`, `compare_replay_timings.py` — controlled replay driver and its analyzer.
