# r05-b01-a01: k=2 column split run as two 20-core AG waves: 40 workers each take half a tile-row; wave A (rows 0-9) reads first, wave B's read starts as A's lands, and each wave gets its own fabric gather and go, so A's AG overlaps B's read and B's AG overlaps A's drain

## Motivation
On the root (r04-b04-a02) the kernel is DRAM read, then a gap with no DRAM traffic, then the DRAM write:
- The input read moves 1.15-2.29 MB/chip at ~400 GB/s: R_INPUT end max 3.4/4.0/5.7/6.3 µs.
- The drain writes the same bytes at ~400 GB/s: 3.3-5.65 µs per core.
- Between them, DRAM is idle for ~5-6 µs on every shape. That time is the PRE tail (0.8-1.1 µs at HiFi4), push
  0.25, collect 0.1, F_FABRIC ~2.5 (fabric latency plus cross-chip skew), go + stick read ~0.8-1.1, combine ~0.4,
  and the first POST block ~0.3 (my `pre.py`/`ag.py` on reports/r04-b04-a02, plus r04-b01-a03's breakdown).
- Every node since r02 has shaved pieces of that gap. None has overlapped it with DRAM traffic, which is the
  bigger prize: about half of today's kernel.

r03-b04-a02 tried overlapping with two waves of 10 full-row workers and lost 5%. The plumbing worked. The loss
came from per-core caps: 10 cores read at only 273 GB/s, and HiFi4 PRE over 56 tiles per core (~110 ns/tile) made
each wave's stat 2.4 µs later. Its reflection (#4) names the fix: "Waves become worth a retry only with more cores
than rows: e.g. a k=2 column split (40 workers, as two waves of 20)." Each wave then has today's 20 cores and
today's per-core read depth, so it should read at today's aggregate rate. Each core has half a row, so per-core PRE
and drain work is halved too.

## Mechanism
Enabled only for the plain AG RMSNorm path: one forwarder, one tile-row per worker, even num_tile_cols, broadcast
gamma or none, no bias/RoPE/per-head/per-token, resident POST, BH, ring <= 8, 2*rows+1 cores available. All four
campaign shapes qualify. Every other configuration keeps today's code path.

1. **Decomposition (factory).** num_workers = 2 * num_tile_rows (40). Worker i: wave = i % 2, j = i / 2,
   row = wave * rows_A + j / 2, half = j % 2. Each worker covers columns [half*W/2, (half+1)*W/2). The kernels'
   num_tile_cols CT arg becomes W/2, so all CBs and loops are half-width. The input row stride (full W) and the
   column offset are new reader/writer args. compute's stats_tiles_cols = 2*ring, so the combine adds 8 partial
   tiles (both halves from 4 chips), and the existing 1/(num_tile_cols*32*stats_tiles_cols) stays exactly 1/H_full.
2. **Waves (reader).** As in r03-b04-a02: a wave-B core waits on a start semaphore before its input read. Its
   wave-A partner (worker i-1) ups it once A's read is 2 blocks from landing, so B's reads queue right behind
   A's. Lookahead stays at 4 blocks.
3. **Stats page layout (sizing, writer).** Within wave w, slot j's stick sits at
   `w*span + L(j)`, with `L(j) = (j/16)*2048 + (j%16)*64`. face_00 row 0 is at that offset and face_01 row 0 at
   +1024, so each slot is a valid fp32 tile row 0, the r04-b02-a01 layout that was validated on HW. A row's two
   halves are slots 2p and 2p+1, which are 64 B apart. So **one 1152 B read per chip brings both halves' sticks**,
   4 reads after go instead of today's 8. The gathered CB uses 64 B pages, and compute addresses device d / half
   h's tile at page `d*18 + h` (the r04-b02-a01 offset-indexing trick). The page becomes 2*span = 6656 B (L1
   interleaved scratch, shape-only decision in compute_sizing so create_stats_buffer agrees). Each wave's fabric
   write is span = 3328 B, which is <= 4352 B.
4. **Forwarder (forked into the op dir as `dit_rmsnorm_wave_forwarder.cpp`).** It uses r03-b04-a02's two-event
   poll loop with 16-bit per-wave fields. Workers increment arrival by `1 << 16*wave`, and the fused fabric inc
   adds `1 << 16*wave` to out_ready. The forwarder sends wave w's region as soon as its 20 sticks are in, and
   releases wave w's 20 go-sems as soon as its 3 peer packets have landed.
5. Writer drain and gamma use the column offset. The stick push uses the new slot offset (face_01 at +1024) and
   the per-wave arrival increment. Posted drain, ack-free push and streamed gamma are unchanged.

Files: device op factory + types (sizing), reader, worker writer, compute, new wave forwarder.

## Why this is not a repeat
- r03-b04-a02 (waves of 10 full-row cores): flawed because of per-core caps. Here each wave keeps 20 cores, the
  cores have half the per-core work, and both halves of a row are in the same wave.
- r01-b03-a0x (column split k=3/4, 60-80 cores, all at once): the row leader summed follower partials before the
  push (+1-2 µs), and 80 concurrent writers collapsed the drain. Here both halves push independently (no leader
  hop), the AG carries 2 partials per row, and only 20 cores read or write DRAM at any moment.
- r01-b02-a01's column split never engaged: 40 sticks don't fit a 34-stick packet. Two waves of 20 sticks do.

## Expected effect and risk
Model (DRAM ~400 GB/s, gap ~4.5 µs from a wave's read end to its drain start): A read, B read back to back, then
A drain right as B's read is done plus 4.5 µs. Ideal ≈ A read + 4.5 + A drain + B drain:
h3584 ~9.5, h4096 ~10.5, h6144 ~12.8, h7168 ~13.7 µs, versus 12.0/13.6/16.9/17.9.
I'd be satisfied with half of that (score ~1.5-1.6).

Risks:
- Per-wave AG costs: each wave pays fabric latency, and skew is set by the latest chip. Check F_SEND/F_GO per wave.
- B's read could be slowed by A's tail. If B's chain gates the end, a child should tune the signal lead.
- 41 cores make dispatch heavier, and a shorter device op is more host-bound, so chip launch skew could eat part of
  the gain.
- Hang risks: wave fields, start sem, per-wave go. Accuracy risks: slot offsets, page-indexed tiles. Either would
  show as hang or accuracy_fail. Compile errors are iterated before the device run.
- Half-width 14 (h3584) is the first time the ragged last block (4,4,4,2) runs in this lineage. The code paths
  handle tails (block-padded pushes), but they are untested here.
