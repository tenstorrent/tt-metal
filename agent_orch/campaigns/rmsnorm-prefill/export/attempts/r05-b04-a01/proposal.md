# r05-b04-a01: two-round half-row pipeline: each of the 20 workers takes one column half of two rows, in two forwarder rounds, so round 0's all-gather overlaps round 1's read and round 0's POST/drain overlaps round 1's all-gather

## Motivation
Today every chip runs one rigid wave: all 20 workers read their full row (h7168: 0.7 -> 6.3 µs, ~400-440 GB/s,
about DRAM peak), then push their stick, then wait the whole fabric AG (push end -> go 1.9 µs on the last-launched
chip, up to 4.8 µs on dev0 because of cross-chip launch skew), then run combine + POST and drain the whole row
(go -> drain end 6.4 µs at h7168, again ~DRAM peak). DRAM sits idle for the whole AG window, and the whole drain
comes after the last go. Per-core timeline of the parent r04-b04-a02 (`analysis/tl.py`, medians, µs from first
kernel start, dev3 = the latest chip):

| shape | R_INPUT end | W_PUSH end | W_AGWAIT end | C_POST start | C_POST end | W_DRAIN end |
|---|---|---|---|---|---|---|
| h3584 | 3.12 | 3.98 | 6.72 | 7.86 | 9.80 | 10.58 |
| h7168 | 5.84 | 6.92 | 9.65 | 10.89 | 14.63 | 16.13 |

dev0 at h7168 waits until 12.20 for go and ends at 19.15 (chip mean 17.93). The post-go chain
(stick read 0.6 + combine 0.55 + POST/drain of 56 tiles ~5.2) is ~6.4 µs on every chip.

r03-b04-a02 tried to overlap the AG with DRAM traffic by splitting the 20 *cores* into two waves and lost 5%:
a wave of 10 cores could not keep DRAM busy, and per-core rates (PRE, drain) stretched each wave. Its reflection #4:
"waves become worth a retry only with more cores than rows".

## Mechanism
Split the *work*, not the cores. Each tile-row is cut into k=2 column halves, and all 20 workers stay busy in both
waves: worker w handles column half (w % 2) of row (w / 2) in round 0 and of row (w / 2 + 10) in round 1. The
existing multi-round forwarder protocol carries this unchanged (max_rounds = 2, 20 sticks per round, one packet each):

- **Round 0** = rows 0-9 (both halves), **round 1** = rows 10-19. Each round still has 20 sticks (one per worker),
  so it fits the 34-stick fabric packet and the stock forwarder (no fork, no protocol change).
- **Reader**: one trid-pipelined pass over both half-rows back to back (no drain bubble between rounds); input tile
  = row * full_cols + col_offset + c.
- **Compute** (`two_phase`): PRE0, PRE1, x*gamma 0, x*gamma 1, then COMB0 + POST0, COMB1 + POST1. Each PRE row-stat
  is a half-row partial; the combine sums ring_size * 2 = 8 gathered partial sticks (the same row's two halves from
  all 4 chips), and 1/H uses slice * 32 * 8 = the full hidden size. Everything stays HiFi4 / fp32 DST, no approx.
- **Writer**: gamma slice [col_offset, col_offset + slice); stick push to slot w of packet round r; after go(r) it
  reads its row's 2 sibling sticks from each device's page (d, f, r). Round order: push0 -> go0 -> read sticks0 ->
  push1 (only after go0, which is what keeps the stock forwarder's cumulative arrival count race-free) -> drain0
  (polling go1 between output blocks, reading sticks1 as soon as it lands) -> drain1. Output tile col = col_offset + c.
- **Factory / sizing**: `pick_col_splits` (shape-level: AG path, RMS, whole-row norm, one row per worker, even
  worker count and width, slice >= 8 tiles) is used by both `compute_sizing` (stats buffer = 2 rounds of pages) and
  `create_at`; the program only engages the split on the prescale-gamma resident path (broadcast weight read by the
  writer, no bias/rope/streaming). Slice-sized CBs; intermediate (x*gamma) holds both half-rows; the stat CBs hold 2
  rounds.

Files: `device/dit_fused_distributed_rmsnorm_program_factory.cpp`, `device/kernels/dataflow/dit_rmsnorm_fused_reader.cpp`,
`device/kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp`, `device/kernels/compute/dit_rmsnorm_fused_compute.cpp`.

## Why this is not a repeat
- r03-b04-a02 (two waves of 10 *cores*): each wave had half the readers/drainers, so phases were per-core-bound.
  Here both rounds use all 20 cores; each round moves half the bytes at the full 20-core rate.
- r01-b02-a01 / r01-b03-a0x (column split onto 40-80 *more cores*): needed > 34 sticks per packet (infeasible, a01)
  or an on-chip leader combine hop (+1-1.5 µs, b03). Here the core count and sticks/round are unchanged; the column
  halves are combined by the post-AG FPU add that already exists (8 tiles instead of 4).
- r04-b02-a0x (forwarder multicast release) changed only the release; this changes the schedule.

## Expected effect and risk
Model (h7168, dev3): read0 0.7-3.5, read1 3.5-6.3, push0 ~4.1, push1 ~6.6-8, compute busy PRE/XG until ~9.6,
COMB0/POST0 9.6-12.0 (drain0 to ~12.8), COMB1/POST1 12.0-14.4, drain1 to ~15.4 (vs 16.1). The post-go1 chain is half
the parent's (28 tiles instead of 56), so skewed chips gain most (dev0 ~19.2 -> ~16.5). Narrow shapes: go1 lands
about where today's single go lands (two serialized AGs of ~2 µs from an earlier first push), but only half a row
is left after it: h3584 ~11.2 -> ~9.6. Expect -1 to -2.5 µs per shape, score ~1.5+.

Risks:
- Compute becomes the per-core floor (PRE+XG+POST all HiFi4 for 56 tiles + 2 combines ~13.5 µs at h7168); if
  PRE/XG are slower than estimated, wide shapes gain less.
- Two AG rounds serialize (push1 waits go0); if the fabric AG is longer than ~2 µs per round, narrow shapes gain less.
- Hang risk: round/slot mapping, CB sizing (stats/gathered CBs hold 2 rounds), go/arrival counting. Accuracy risk only
  through a wrong sibling slot or col offset (PCC would collapse, not drift).
