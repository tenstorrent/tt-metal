# r05-b03-a02 result: 1.4219 (ok)

## What happened vs expected
Valid on every shape on the first device run, with no compile iteration. PCC 0.9999985 and max_abs 0.0204-0.0237 match
the parent. So these all work on HW:
- the k=4 split (80 quarter-row workers + 1 forwarder, one kernel group);
- the 4-wave forwarder with 8-bit per-wave fields;
- the 4-deep packet CB and the 3328 B per-wave fabric region;
- the tile-row-0 slot layout with one 1280 B row read per chip;
- the 16-partial combine through 64 B-page tile views;
- the chained wave gate (middle waves wait, then signal).

It is **slower** than the parent r05-b03-a01 (1.5174) on every shape. Score 1.4219 (-6.3%). µs, chip mean:

| shape | parent (2 waves) | this (4 waves) | change |
|---|---|---|---|
| h3584 | 11.47 | 13.05 | +1.58 (+13.8%) |
| h4096 | 12.76 | 13.59 | +0.83 (+6.5%) |
| h6144 | 15.25 | 15.52 | +0.27 (+1.7%) |
| h7168 | 16.02 | 16.84 | +0.82 (+5.1%) |

The loss is not launch skew. The min-chip kernel is worse too: h3584 11.94 vs 10.04, h7168 16.58 vs 15.12 (`eval/ops.csv`
per device). I predicted about -1 to -2 µs (score 1.65-1.7). The pipeline premise was never tested, because one
existing writer behaviour serialised the waves (below).

## Why (profiler evidence)
Scripts in `analysis/`, all run as `python3 X.py $DREAM_HOME/rmsnorm-prefill/reports/<node> [waves]`:
- `wavesN.py`: per-wave timeline. Waves are recovered as cores sorted by go time.
- `push.py`: per-wave R_INPUT / W_GAMMA / W_PUSH.
- `gopost.py`: the go -> POST breakdown per core.

Outputs: `waves_out.txt` (this node) and `waves_parent_out.txt`.

1. **The reads pipelined exactly as designed.** Per-wave R_INPUT end max, µs:

   | shape | W0 | W1 | W2 | W3 |
   |---|---|---|---|---|
   | h3584 | 1.35 | 2.45 | 3.52 | 4.29 |
   | h7168 | 2.08 | 3.51 | 5.19 | 6.38 |

   The total read end is the same as the parent's (4.29 vs 3.42 at h3584 is a little worse; 6.38 vs 6.41 at h7168).
   So 20-core quarter-row waves keep the aggregate DRAM rate, and the start_sem chain works.

2. **But the stick pushes were NOT staggered. They were gated by the broadcast-gamma read, so the AG waves bunched
   up.** W_PUSH end max per wave, µs:
   - h3584: 4.50 / 4.46 / 4.56 / 5.08 (read ends 1.35..4.29)
   - h7168: 2.96 / 6.55 / 6.82 / 7.41 (read ends 2.08..6.38)

   Every gated push ends right at W_GAMMA end, which is 4.35-4.62 at h3584 and 6.5-7.1 at h7168. The cause is in
   the writer's W_GAMMA loop:
   - `poll_stick()` only runs between read issues and between chunks. The chunk barrier
     `noc.async_read_barrier<TXN_ID>` blocks without polling.
   - The gamma reads are issued at ~1.4 µs, but they land only after the whole DRAM input queue drains: 80 cores'
     input reads are ahead of them.
   - With 7-8 gamma pages per core (h3584/h4096) there is a single chunk. So every wave's push waits for that one
     barrier, including wave 0, whose stat was ready at ~2.2 µs.
   - At h6144/h7168 (12/14 pages, 2 chunks), wave 0 escapes between the chunks (push 2.9 µs). Waves 1-3 still wait
     for the last chunk.

   The parent had the same gate, but it hit only wave B: its B push of 7.32 µs sits right at its W_GAMMA end of 7.24.
   Its wave A had 2 chunks everywhere, so it escaped.

   Result: the forwarder sends were bunched together.
   - h3584: F_SEND#0..3 end at 4.89 / 5.15 / 5.41 / 5.64.
   - h7168: send#0 at 3.95, sends 1-3 at 6.91 / 7.27 / 7.83.

   The gos follow ~0.6 µs apart (F_GO#0..3 = 8.23 / 9.12 / 9.71 / 10.30 at h3584), and each wave's AG still takes
   ~3 µs from push to go. So 4 waves gave no AG overlap. They only added 4 serial go releases and drains that
   started late and close together. Wave 0's chain was 6.9 µs at h3584, from read end 1.35 to drain start 8.70,
   against the model's 5.2.
3. **The gather fix works; the bigger combine eats part of it.** Per core, relative to go:

   | | gather read (go -> C_COMB start) | combine (C_COMB on TRISC_0) | go -> C_POST |
   |---|---|---|---|
   | this node | 0.63 | 0.35 (8 adds) | 1.34 |
   | parent | 1.01 | 0.20 | 1.57 |
   | r05-b01-a01 | 0.54 | 0.22 | 1.12 |

   C_COMB end -> C_POST start is a fixed ~0.36 everywhere.
4. Per-wave drain duration (median per core): 1.73 µs at h3584 and 2.4-3.4 µs at h7168. With quarter rows of 7-14
   tiles per core, that is ~250 ns/tile because consecutive waves' drains overlap (aggregate-bound), as expected.

## Classification
Repairable failure (bug: the stick push is gated by the BRISC gamma trid barrier, which completes only after every
wave's input read has drained from DRAM, so the 4 AG waves fire at the same time). The 4-wave plumbing itself is
correct and validated on HW. The overlap premise is untested, not refuted: reads did pipeline, pushes did not.

## What a child of this node should try next
1. **Repair: never block the stick push on gamma.** In `dit_rmsnorm_fused_worker_writer.cpp`, W_GAMMA loop:
   - Replace `noc.async_read_barrier<NocOptions::TXN_ID>({.trid = t})` with
     `while (!noc.is_read_trid_flushed(t)) { poll_stick(); }` (`Noc::is_read_trid_flushed`, api/dataflow/noc.h:640).
   - Or simpler and stronger: issue the gamma reads, push the stick as soon as the stat is ready, and do the
     gamma chunk barriers/pushes AFTER the push. Gamma is only needed by the x*gamma pre-pass under the AG wait.
   - Expected wave pushes ≈ read end + 0.9: h7168 ≈ 3.0 / 4.4 / 6.1 / 7.3, h3584 ≈ 2.2 / 3.3 / 4.4 / 5.2.
   - Check with `analysis/push.py`: W_PUSH end must track R_INPUT end per wave, not W_GAMMA end.
   - **The same fix applies to the 2-wave nodes (parent, r05-b01-a01):** their wave B push sits on W_GAMMA end too.
     That is a cheap, likely-positive test on the 1.57 node first.
2. **Then judge 4 vs 2 waves.** Even with staggered pushes, each wave's AG is ~3 µs push -> go (cross-chip skew).
   The forwarder releases waves in order, so 4 waves only pay if the gos come out ~R/4 apart.
   - Use `wavesN.py` to read F_GO#k spacing and the per-wave drain start.
   - If 4 waves still lose on h3584/h4096, a shape-dependent choice (2 waves for narrow, 4 for wide) is a cheap
     host-side knob in `compute_sizing`.
3. **Make gamma cheaper or earlier.** With 80 cores, gamma is 80 x 14-28 tiny reads queued behind the input stream.
   - Read gamma before the input on wave>0 cores, which wait on start_sem anyway and have idle DRAM time first.
   - Or let one core per column quarter read it and multicast it to its 20 peers.
4. **Cut the 16-partial combine (0.35 µs).** Pre-add the 4 quarter sticks of a row on BRISC/SFPU before the push?
   No: that is the leader hop r01-b03 lost on. Better: add the partials pairwise as 4 accumulating adds over 2 tile
   views each, or do the combine as SFPU row-0-only adds (VectorMode::R, 2 iterations/face) instead of full-tile
   ELWADDs. Only row 0 is real.
5. Production caveats carried over: posted drain fence (r04-b04-a01 #2). The 64 B-page overlapping tile views rely on
   the unpacker addressing tiles by fifo page size.
