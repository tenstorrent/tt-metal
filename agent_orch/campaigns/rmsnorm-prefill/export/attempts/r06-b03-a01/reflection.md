# r06-b03-a01 result: 1.5512 (ok)

## What happened vs expected
Valid on every shape on the first build and the first device run: no hang, no JIT error. PCC is 0.9999985 and max_abs
0.0205-0.0240, the parent's values. So all of this works on HW:
- the uneven split (wave A = 9 rows / 18 workers, wave B = 11 rows / 22 workers);
- the evenly-spread wave-A placement;
- the 1-2 partner start-sem gate;
- the forwarder's per-wave slot counts (compact wave-major worker list);
- the 22-slot wave regions (3456 B each, 6912 B page).

But it is not faster. µs, chip mean, parent r05-b01-a01 -> this node:

| shape | parent | this | change | min-chip parent -> this |
|---|---|---|---|---|
| h3584 | 11.19 | 11.50 | +2.8% | 9.65 -> 9.78 |
| h4096 | 12.08 | 12.34 | +2.1% | 10.59 -> 10.90 |
| h6144 | 14.59 | 14.63 | +0.3% (noise) | - -> 14.17 |
| h7168 | 15.84 | 15.77 | -0.4% (noise) | 15.71 -> 15.50 |

Score 1.5512 vs 1.5696 (-1.2%). I expected 1.60-1.63 (h7168 ~14.6-15.2). The wide shapes are flat. The narrow shapes
lost ~0.3 µs, partly launch skew (dev0..3 at h3584: 13.32/12.13/10.78/9.78), but the min-chip kernel is also
+0.13/+0.31 µs worse there.

## Why (profiler evidence)
Scripts: r05-b01-a01's `analysis/waves.py` and `analysis/fwd.py`, copied to `analysis/`. Outputs: `waves_out.txt` and
`fwd_out.txt`. The parent's numbers come from its committed outputs. µs, medians of per-wave max:

| h7168 | A read end | A push | A F_SEND#0 s (fwd clock) | A go | A drain s -> e | B read end | B go | B drain s -> e | end |
|---|---|---|---|---|---|---|---|---|---|
| parent | 3.59 | 4.51 | 5.27 | 8.53 | 9.05 -> 14.21 | 6.29 | 10.66 | 11.31 -> 15.59 | 15.74 |
| this | 3.39 | 4.24 | 4.62 | 8.43 | 8.91 -> 13.60 | 6.31 | 10.73 | 11.33 -> 15.67 | 15.81 |

The model had two premises. Both are wrong:

1. **"A's read end scales with A's share of the rows" is false.** Each worker reads exactly one half-row (28 tiles at
   h7168, 14 at h3584), whatever the wave size. So a wave's read end is per-core time, not aggregate time.
   - Fewer concurrent A readers made each one only ~6% faster: A read end 3.59 -> 3.39 at h7168, and 2.08 -> 2.00
     at h3584.
   - B's read end is B's start (A's lead signal) plus the same per-core read time. It did not move at h7168
     (6.29 -> 6.31). It got later on the narrow shapes (3.43 -> 3.58, 3.88 -> 4.04), because B now has 22 readers.
   - The waves pipeline in time, but neither wave is aggregate-bound at 18-22 cores.
2. **"A's go follows A's push" is false.** At h7168, A's push and forwarder send moved 0.27 / 0.65 µs earlier
   (F_SEND#0 s med 5.27 -> 4.62). But F_GO#0 moved only 0.14 µs (7.95 -> 7.81 on the forwarder clock), and the
   worker go only 0.10 µs (8.53 -> 8.43).
   - Even the best chip-call waits ≥1.65 µs from its send end to F_GO#0 (min F_SEND#0 e 4.59, min F_GO#0 s 6.24).
   - So wave A's release is gated by the peers' wave-A packets (cross-chip launch skew plus the ~2.3 µs fabric
     round), not by this chip's push. Starting A's local chain earlier doesn't move its drain start.
3. **The drains still overlap, and their rate didn't change.** A's drain ends 13.60 (was 14.21) and B starts 11.33,
   so they still share ~2.3 µs.
   - Per-core drain duration is unchanged: A 3.56 µs, B 3.84 µs for 28 tiles, ~130 ns/tile.
   - B's drain is the tail as before. It starts at the same 11.33 because B's go (10.73) is set by B's AG, which
     the split doesn't touch.

The "kernel end = A drain start + whole output / aggregate rate" model fits the parent, but its lever (an earlier A
drain start) is fixed by A's go, and A's go is fabric/peer bound.

## Classification
neutral to slightly negative (flawed premise; -1.2% geomean, wide shapes within noise, narrow shapes +2-3% from B's
later read end). The uneven-wave plumbing is correct and reusable:
- `dit_rmsnorm_wave_a_rows`;
- `wave_a_slots` in the sizing struct and forwarder CT args;
- the per-wave slot thresholds;
- the 1-2 partner gate (reader RT args `wave_role, num_partners, p0x, p0y, p1x, p1y`).

## What a child of this node should try next
1. **Revert the split to 10/10.** Set `dit_rmsnorm_wave_a_rows` to `rows / 2` (the rest of the plumbing is neutral)
   or start from r05-b01-a01. Don't try 8/12 or other uneven ratios: per-wave reads are per-core bound and A's go is
   peer bound, so the row partition has no lever. The reverse (B smaller) would only lengthen A's read and chain.
2. **The lever is the AG release, not the local chain.** A's go sits ~3.6-4.2 µs after A's push on every shape,
   whatever the push time (this node moved the push 0.3-0.65 µs with no effect). Candidates:
   - forwarder flush-then-inc go release (no `async_write_barrier` before the incs; r04-b03-a01 #2);
   - measure per-device F_SEND#0 -> out_ready to separate fabric latency from cross-chip launch skew.
3. **The B tail is B's AG plus a ~130 ns/tile drain.** That rate is ~35% slower than the 20-core drains of round 4
   (~95 ns/tile), even where only one wave drains. Find whether 40-core placement (rows y=2..5) changes the NoC drain
   paths. The per-core NoC0/NoC1 share (r02-b02-a01 #1) was tuned for 20 cores in rows 2-3.
4. The gamma un-gate port (r05-b03-a03 #1) is orthogonal and still the cheapest likely gain on this lineage.
