# A/B compute sprint: final measured result

A improves only **0.20%**; B improves **2.41%**, with all tested output bytes
unchanged. This is a modest scheduling improvement, not a new numerical recipe.
No canonical production headers were edited.

## Decisive uninstrumented throughput

Single core, Q256/K512/D128, 16 Q repeats × 512 repeated resident K/V chunks.
Each result uses 12 warmups and 10 measured blocking trace replays, with
alternating canonical/private-disabled/candidate execution order. Q/K/V remain
raw BF16; HiFi2, BF16 DST, exp/subtraction/rounding/recurrence algorithms, CB
formats/counts, Q/K chunks and both input slots are unchanged.

| Recipe | Canonical ms | Winner ms | Less time | Canonical TF/core | Winner TF/core |
|---|---:|---:|---:|---:|---:|
| A, frozen main BF16 streaming | 275.8081 | 275.2584 | 0.1993% | 1.99325 | 1.99724 |
| B, compensated BF16 streaming | 345.5943 | 337.2816 | 2.4053% | 1.59075 | 1.62996 |

Sources: `A-final-steady-02.json`, `B-block-reuse-fence-steady-01.json`.
Both private-disabled controls are bitwise identical to canonical. Every
enabled winner is also bitwise identical, including eager/trace comparison.
An earlier independent A round measured 0.2044%, confirming its very small gain.
These are compute-resident results, not measured whole-chip application speedups.

## Retained changes

A (`SDPA_BF16_EXP_BLOCK`): process eight contiguous BF16 DST tiles with sixteen
stock 32-instruction exp replays, one address setup, and one final drain. The
stock exp macro initialization, coefficients, arithmetic and rounding remain
unchanged. Original scalar calls remain the fallback outside the eight-tile
subblock. Pack still waits for SFPU stores before consuming DST.

B enables exactly three implementation-only flags:

- `SDPA_BF16_BLOCK_STATE`: numerator DST layout groups the two high tiles, two
  low tiles and two chunk tiles. This permits three two-tile copies and two
  two-tile packs instead of six scalar copies and four scalar packs. Same
  15 SFPU arithmetic instructions; only DST offsets change. L1 layout is unchanged.
- `SDPA_BF16_CORRECTION_REUSE`: adjacent even/odd column vectors share the
  column-broadcast correction. Skip only duplicate correction loads in the
  second vector; preserve all adds, MADs, rounding, residuals and stores.
- `SDPA_BF16_CORRECTION_FENCE`: remove the balanced PACK_DONE post/wait/get
  triplet inside compensated SALAD. The helper already waits for correction-CB
  publication, which is a pack-completion release and occurs after the PV row
  being consumed. The removed fence additionally waited for the next PV row.
  All other waits, L1-acc toggles and normalization ordering remain unchanged.

Plane grouping, correction reuse and the fence argument originated with the
E/G agent and were transferred and measured independently for B. Ordering
details: `../lowp/correction_fence/ORDERING.md`. The B Q256 correction-address
reset wrapper is retained. B does **not** enable the A exp-batching experiment.

`FROZEN_MANIFEST.json` pins the numerical definitions, winner flags, sources and
benchmark contract. Experimental headers still contain disabled screening
branches; only the listed winner flags define the retained configurations.
All manifest hashes verified locally after final qualification.

## Exact correctness qualification

The standalone fullchip adapter calls the canonical rectangular device-input
adapter and redirects only the compute kernel/implementation flags. Original
reader/writer sources and all data movement parameters are unchanged. It tests
canonical, private-disabled and winner for each case, plus two trace replays.

| Winner | Q/K lengths | Cores / distinct Q jobs per core | Seed | Input distributions | Result |
|---|---|---|---:|---:|---|
| A | 2048/8192 | 2 / 4 | 1237 | 6 | All bytes equal |
| A | 1024/1024 | 2 / 2 | 20260918 | 8 | All bytes equal |
| B | 2048/1536 | 2 / 4 | 1237 | 6 | All bytes equal |
| B | 2048/8192 | 2 / 4 | 20260918 | 8 | All bytes equal |

Six-input suite: normal, growing-max, scaled-QK, outliers, common K, common V.
Eight-input suite additionally includes common Q and constant V. Growing-max
scales consecutive K chunks from 0.25 to 4; it is not repeated identical KV.
Thus each final winner passed 14 fullchip shape/distribution cases, with
42 canonical/disabled/winner records and two trace replays per record.
All fullchip comparisons are strict BF16-bit comparisons, not merely L2/PCC.

Artifacts: `A-expblock-fullchip-01.json`, `A-final-heldout-01.json`,
`B-block-reuse-fence-fullchip-02.json`, `B-final-heldout-01.json`.
Resident benchmark and profiler outputs add separate exact comparisons.

This is **not** exhaustive production SDPA qualification. Fence/layout changes
are qualified here for the fixed noncausal, non-ring D128 recipe. Causal/ring,
different head dimensions, padded tiles, masks, other chunk geometries and
concurrent production scheduling need their own integration audit and tests.

## What the profiler says

Clean captures are `profile-A-final-02` and `profile-B-final`; their companion
benchmark JSONs exist and confirm exact outputs. Counters use eight Q repeats,
512 resident K chunks, and the reported clock is 1350 MHz. The HiFi2 useful-work
roof is 2048 FLOPs/cycle/core. These diagnostic profiles are separate from the
uninstrumented throughput measurements above.

| Profile | Useful utilization, own HiFi2 roof | FPU busy | SFPU busy | Both busy | Neither busy |
|---|---:|---:|---:|---:|---:|
| A canonical | 72.144% | 77.392% | 23.622% | 21.010% | 19.996% |
| A winner | 72.372% | 77.637% | 22.212% | 19.703% | 19.853% |
| B canonical | 57.557% | 62.643% | 37.772% | 20.212% | 19.797% |
| B winner | 59.023% | 64.239% | 38.044% | 20.555% | 18.273% |

Useful utilization counts only QK/PV FLOPs. FPU-busy counters also include other
FPU work; the two definitions must not be conflated.

A removes about 2.75 million SFPU-busy cycles but only 0.59 million elapsed
cycles: much of that SFPU work was already hidden by FPU work. This explains
the tiny wall-time gain. B keeps approximately 146.08 million FPU-busy cycles,
reduces SFPU-busy cycles from 88.08M to 86.51M, and reduces neither-busy cycles
from 46.16M to 41.55M. Its improvement mainly reduces exposed synchronization
and scheduling gaps. SFPU busy **percentage** rises only because runtime falls.
Raw cycles and decoded counters: `profile_summary.json`.

## Screening decisions and limitations

| Candidate | Observation |
|---|---|
| A/B pack-width cache | No gain; slight regression; reject |
| B redundant L1-acc enable/disable toggles | No gain; reject |
| A max-stat block copy | ~0.009% change, noise-scale; reject |
| A denominator repeated L1 enable removal | Slower; reject |
| B plane grouping alone | 0.368% less time |
| B plane grouping + correction reuse | 0.865% less time |
| B plane grouping + reuse + fence removal | 2.405% less time; retained |

B still costs materially more compute time than A; this sprint does not remove
compensation's fundamental work. A's isolated 0.2% optimization may not justify
a separate production specialization unless a common batched-exp helper makes
it cheap to maintain. No broad 80%-useful-utilization claim is warranted.

Operational exclusions: one device-open Ethernet heartbeat failure required
the coordinator's reserved-board reset and fresh smoke. One premature launch
ran old CLI choices while upload was pending and failed before device open.
The first A profiler run completed kernels but failed post-run JSON generation
because Tracy supplied a relative script path; it is excluded. A clean repeat
with an absolute path succeeded. None of these failed runs is used for the
final accuracy or performance claims. All hardware runs used the global lock;
no autonomous reset or production source modifications occurred.
