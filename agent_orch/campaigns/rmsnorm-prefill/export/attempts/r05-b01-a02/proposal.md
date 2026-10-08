# r05-b01-a02: four-wave k=4 column split: 80 workers each take a quarter tile-row, and the rows run as four 20-core all-gather waves (wave k+1 reads as wave k lands, one fabric gather + go per wave, 8-bit wave fields in a 4-wave forwarder), so three AG chains hide behind reads and the exposed drain is a quarter of the output

## Motivation
Parent r05-b01-a01 (1.5696) proved the wave pipeline on 40 half-row workers. Its `waves.py` timeline (h7168, µs):
A read end 3.59, B read 2.87 -> 6.29, A go 8.53, B go 10.66, A drain 9.05 -> 14.21, B drain 11.31 -> 15.59, end 15.74.

The kernel is now: last wave's read end + one fixed chain + last wave's drain.
- **Last wave read end ≈ total read time** (6.29 vs the root's 6.33): 20 cores per wave keep the read aggregate-bound,
  so how many waves the read is cut into doesn't change when it ends.
- **The chain** (read end -> drain start) is ~5.0 µs on every shape: PRE tail, push, forwarder send, fabric AG
  (~2.6 µs send -> go even with the chips in sync, r05-b04-a01), go -> drain start 0.5. It is fixed per wave.
- **The exposed drain is the last wave's drain**: 4.3 µs at h7168 (B 11.31 -> 15.59), and it is slowed by A's
  drain overlapping it (A ends 14.21). Alone, half the output takes ~2.8 µs at ~400 GB/s.

With two waves, only one AG chain (A's) is hidden, and the tail still carries half the output. The parent's
reflection #2 proposes more waves. With N waves of equal size the model is
`end ≈ T_read + chain + T_drain / N`, as long as consecutive waves' drains don't overlap. That holds when one wave's
drain (T_drain/N) is no longer than the spacing between wave read ends (T_read/N). Reads and drains run at about the
same aggregate rate, so this holds.

Four waves need 20 workers per wave to stay aggregate-bound (r03-b04-a02: 10-core waves were per-core capped). With
20 tile-rows, that means a k=4 column split: 80 workers, each one quarter of a tile-row, 5 rows (20 sticks) per wave.
Every campaign width is divisible by 4 tile-cols: 28/32/48/56 -> 7/8/12/14 tiles per worker.

Model at h7168: 6.3 + ~5.2 (combine of 16 partials instead of 8) + ~1.5 = ~13.0 µs vs 15.74. At h3584:
3.4 + ~4.9 + ~0.8 = ~9.1 µs vs 11.05 (per-chip max). The first wave's drain starts at ~1.6 + 5 = 6.6 µs at h7168,
after the last read ends (6.3), so reads and drains never share DRAM.

## Mechanism
Generalize the parent's two-wave / two-half code to a split factor S, with S workers per tile-row and S waves. S=4 is
chosen when rows % 4 == 0, tile-cols % 4 == 0, 4*rows+1 cores fit, and 4*rows/4 <= 32 slots per wave. Otherwise S=2
(the parent's path). All four campaign shapes take S=4.

1. **Sizing** (`compute_sizing`, types.hpp): new `col_split_factor` S. wave_slots = S * rows/S = rows (20).
   wave_span = the same tile-row-0 slot layout (3328 B for 20 slots). Page = S * span (13312 B). This is still one
   L1-interleaved page per device.
2. **Factory decomposition**: num_workers = S * rows. Worker i: wave = i % S, slot j = i / S,
   row = wave * rows/S + j / S, quarter = j % S, col_offset = quarter * W/S. A row's S quarters are slots S*p..S*p+S-1,
   contiguous 64 B face-rows that never straddle the 16-slot face boundary. stats_tiles_cols = S * ring (16). The
   existing 1/(num_tile_cols*32*stats_tiles_cols) stays 1/H_full.
3. **Reader wave chain**: wave_role becomes bit flags (1 = signal the next wave's same-slot partner, i+1;
   2 = wait on start_sem). Middle waves both wait and signal. The signal point is unchanged (kWaveSignalLeadBlocks=2):
   after block 0 lands for 7/8/12-tile rows, after block 1 for 14-tile rows (~45-60% of the wave's read landed).
4. **Writer**: arrival increment `1 << (32/S * wave)`. The post-go group read is one
   `1024 + S*64` B read per device (1280 B for S=4), landing at stride 1280 in the 64 B-page gathered CB.
5. **Compute**: partial t is the tile at 64 B page `(t / S) * stride_pages + t % S`. The combine adds 16 partials
   (8 add_tiles into DST 0) instead of 8.
6. **Forwarder** (`dit_rmsnorm_wave_forwarder.cpp`): the wave count is a CT arg. Field width is 32/S bits
   (8 for S=4: arrivals ≤ 20, out_ready ≤ 7). Same poll loop: send wave `sent` when its arrivals are complete,
   release wave `released` when all peers' wave packets have landed.

Files: device_operation_types.hpp, program_factory.cpp, reader, worker writer, compute, wave forwarder (all in the op
dir).

## Why this is not a repeat
- r05-b01-a01 / r05-b03-a01: two waves of 20 half-row workers. This keeps 20 cores per wave and doubles the wave count,
  so three of four chains hide behind reads and the exposed drain is a quarter of the output instead of half
  (overlapped with the other half).
- r03-b04-a02 (two waves of 10 full-row cores): per-core caps. Here every wave keeps 20 cores, and per-core work is a
  quarter row.
- r05-b04-a01 (two forwarder rounds on 20 workers, stock forwarder): its rounds serialized, because the stock
  forwarder couldn't send round r+1 before round r's go. Here each wave's send waits only on its own arrivals (field per
  wave), so the four AGs overlap in the fabric.
- r01-b03-a02/a03 (k=4, 80 workers, one AG): the leader-combine hop and all 80 cores draining at once. Here there is
  no leader hop (each quarter pushes its own stick), and only 20 cores read or drain at any moment.

## Expected effect and risk
Expect -1.5..-3 µs on h6144/h7168 and -1..-2 µs on h3584/h4096 (score ~1.75-1.9) if the model holds. Things that
could eat it:
- **Per-wave fixed costs**: the forwarder does 4 sends + 4 x 20 go incs serially (F_GO ~0.5 µs each). At h3584 the
  wave spacing is only ~0.85 µs, so forwarder work may stack up and delay later waves' go.
- **The 16-partial combine** (+0.1-0.2 µs per chain).
- **Dispatch**: 81 cores and more RT args may grow the host-side launch skew, which already dominates h3584's chip
  mean (dev0 13.06 vs dev3 9.65 µs in the parent).
- **Per-wave AG latency floor** (~2.6 µs): it doesn't shrink, but it is hidden for waves 1-3.

Risks:
- Hang: wave fields, chained start sems, forwarder loop. Shows as `hang`.
- Accuracy: slot offsets, group-read stride, partial indexing. Shows as `accuracy_fail` (PCC drops sharply if any
  partial is wrong).
- L1: the packet CB (2 x 13312 B on every core) and the gathered CB (~8 KB). Compile/alloc errors are iterated
  before the device run.

Judge with the parent's `waves.py`, generalized to 4 waves (wave = position in the start-sem chain): per-wave read
end, go, drain start/end. The target is the last wave's drain end ≈ its read end + chain + ~T_drain/4.
