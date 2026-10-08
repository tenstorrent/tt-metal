# r06-b03-a01: uneven two-wave row split (9 rows wave A / 11 rows wave B)

## Motivation
On the round root r05-b01-a01 (1.5696), the two waves are equal (10 tile-rows each). Its `waves.py` timeline at h7168
(µs from the chip's first worker start, medians of per-wave max):

| | read end | push | go | drain start -> end |
|---|---|---|---|---|
| A | 3.59 | 4.51 | 8.53 | 9.05 -> 14.21 |
| B | 6.29 | 7.24 | 10.66 | 11.31 -> 15.59 |

- The two drains overlap by ~2.9 µs (A ends 14.21, B starts 11.31). With both waves writing, each core drains at
  ~135 ns/tile instead of ~95 ns/tile alone, so the drain is aggregate-bound. The write window from A's drain start
  to B's drain end is 6.5 µs for the whole output (~345 GB/s).
- So the kernel end is about "A drain start + whole output at the aggregate write rate" (9.05 + 6.5 = 15.55, measured
  15.59). Wave B's own chain (B read end 6.29 + ~5.0 µs chain + B drain alone ~3.0 µs ≈ 14.3) is not the binding term.
- A's drain start is set by A's read end + a fixed ~5.4 µs chain (PRE tail, push, fabric AG, go, gather). A's read end
  scales with A's share of the rows, because the read is aggregate-bound (B's read still ends at 6.29, the same as
  the parent's single 20-row read).

Same picture at h3584: A drain 7.24-9.71, B drain 8.43-10.91, overlapped by 1.3 µs.

r05-b03-a01's reflection (#3) already suggested an uneven split. This node tests it, but in the other direction from
that suggestion: the model above says the first wave should be **smaller**, so its chain starts earlier and its drain
finishes before B's drain starts.

Model (h7168), with f = A's row share and the read aggregate-bound:
- A drain start ≈ f·6.3 + 5.4.
- A drain alone runs at the per-core rate (~2.7 µs for 28 tiles).
- B drain start ≈ 6.3 + 5.0 = 11.3.
- f = 0.5 (parent): A's drain ends ~12 µs and overlaps B's, so the end is 15.6.
- f = 0.45 (9/11 rows): A's drain ends ~10.95, so the drains just don't overlap. End ≈ 11.3 + B's 22 cores x 28
  tiles at ~370 GB/s ≈ 14.6 µs.
- f = 0.4 (8/12) leaves DRAM idle again and makes B bigger, giving ≈ 14.9.

So 9/11 is the model's optimum on all four shapes. h3584: A drain ~6.9-8.6, B drain starts ~8.4, end ≈ 10.3-10.6 vs
11.05.

## Mechanism
Host factory + reader + forwarder (all inside the op dir). Compute and writer kernels are untouched.
- `dit_rmsnorm_wave_a_rows(rows) = max(1, 9*rows/20)` → 9 of 20 rows in wave A, 11 in wave B. `compute_sizing`
  sizes the per-wave page region (`wave_slots`, `wave_span_bytes`) from the larger wave B (22 slots: 3456 B per wave
  region, under the 4352 B packet) and records `wave_a_slots` (18). Eligibility also requires B ≤ 2·A rows and
  2·B ≤ 32 slots.
- Worker → wave mapping: the 18 wave-A workers are spread evenly over the 40 row-major worker cores (Bresenham), so
  both waves still cover every grid position, like the parent's even/odd interleave. Slot j of a wave → row j/2 of
  that wave, column half j%2. Writer RT args (stick offset, pair offset, arrival inc) are derived from the same maps.
- Read gate: each wave-B worker is signalled by the nearest preceding wave-A worker (the first one if none
  precedes it). So each A worker ups 1 or 2 partners' start_sem at the same lead block as before
  (`kWaveSignalLeadBlocks = 2`). Reader RT args become `wave_role, num_partners, p0x, p0y, p1x, p1y`.
- Forwarder: a new CT arg `wave_a_slots`. Its worker list is compact wave-major: A's 18 slots, then B's 22. Wave 0 is
  sent once 18 arrivals are in, and wave 1 once 22 are, in the same 16-bit fields. Go release is A's 18 workers,
  then B's 22.

## Why this is not a repeat
- r05-b01-a01 (parent): equal 10/10 waves. This node changes only the row partition (and the plumbing it needs).
- r05-b01-a02 / r05-b03-a02 / r05-b01-a03 / r05-b03-a03: four equal waves. They add forwarder releases and drain
  overlap. This node keeps 2 waves and removes the drain overlap instead.
- r05-b03-a01 #3 suggested an uneven split with B smaller. This node does A smaller, from the drain-overlap analysis
  above. If it loses, the reverse is the obvious child.
- Not the gamma un-gate port (r05-b01-a03 #1 / r05-b03-a03 #1), nor the wave lead tuning (r05-b01-a01 #1): both are
  orthogonal and can be stacked later.

## Expected effect and risk
- Expected: h7168 ~14.8-15.2 (from 15.84), h6144 ~13.9-14.2 (from 14.59), h3584/h4096 −0.3..−0.6 µs. Score ~1.60-1.63.
- Risks:
  - Mapping or plumbing bugs. A missing partner signal hangs a wave-B reader. A wrong arrival threshold or release
    range hangs the forwarder or a worker's go wait. These should show up as `hang`.
  - A wrong slot offset corrupts the stat and fails PCC.
  - If B's push is gated by the streamed-gamma barrier (as in the parent), B's chain may be longer than modelled, and
    the end will be bound by B's 22-core drain.
- Judge with r05-b01-a01's `analysis/waves.py`. The target is A drain end ≲ B drain start, and B drain duration ~3 µs
  at h7168.
