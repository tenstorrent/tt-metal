# r03-b04-a03: stream gamma to compute in 8-page chunks (sticky-trid BRISC reads, bank-rotated inside each chunk) so the x*gamma pre-pass starts when PRE ends, not when the whole gamma row has landed; parent's two-wave AG reverted

## Motivation
Profiler evidence from the grandparent r03-b04-a01 (single wave, score 1.3131), using `analysis/late2.py` and
`analysis/core.py` on `reports/r03-b04-a01`:

- The BRISC gamma loop is issue-bound at about 50 ns per 32 B face-row read: 112 reads take ~5.5 us at h7168 and
  ~2.9 us at h3584. Under DM_DYNAMIC_NOC it is ~1 us slower than in r01-b04-a03, which fits the per-read L1 counter
  RMW. The writer pushes `weight_cb` only once, after ONE barrier at the end of the loop. So compute's x*gamma
  pre-pass (~60 ns/tile) cannot start until the LAST gamma page lands, even though it waits on `weight_cb`
  cumulatively per block.
- On healthy cores at h7168, gamma lands at ~6.9-7.5 us and x*gamma takes ~3.3 us. That leaves it finishing within
  0.1-0.5 us of the gathered sticks' arrival (C_COMB start). The minimum of (C_COMB start - gamma end) over the cores is
  3.3-4.6 us, against ~3.4 us of x*gamma. So x*gamma is right at the edge of the post-AG critical path.
- **Straggler loop on device 0.** In 9 of 10 measured h7168 calls on dev 0, and 8 of 10 h6144 calls, one core
  (usually (2,2)) launches ~1.4 us late. That is the cross-call coupling: it finished last in the previous call. Its
  gamma then lands ~1.3 us AFTER its PRE end (9.74 vs 8.4 us), its x*gamma runs after the AG, and its POST starts
  ~2-2.5 us later than every other core's (POST lag after go: 3.1-3.9 us vs 1.28 median). It finishes 1-2 us after
  the other 19 cores, so it is also the late starter next call. Its late stick also gates F_COLLECT (9.12 vs <=8.0 us).
  The last-vs-median drain end is 1.6 us on dev 0 and 0.4 us on the other chips (`late2.py`). r03-b02-a02 (best node)
  shows the same thing: dev 0 h7168 straggler in 9/10 calls.

## Mechanism
Writer (`dit_rmsnorm_fused_worker_writer.cpp`), W_GAMMA block only:
- Read gamma in chunks of 8 pages (= NUM_DRAM_BANKS). Tag each chunk's 16 face-row reads with a NoC1 read trid.
  The trid is set once per chunk with `noc_async_read_set_trid`; the reads themselves are plain one-packet reads, so
  there is no per-read set_trid and no outstanding-count poll. Trids 1..4 are in rotation.
- Two chunks behind the issue front, wait on that chunk's trid and `cb_weight.push_back` its pages. compute's per-block
  cumulative `wait_front` then starts x*gamma on the first pages while the rest are still in flight.
- Bank de-phasing is kept, but inside each chunk: worker w reads chunk pages in the order rotated by `tile_row_start % 8`.
  Each chunk has one page per bank, so concurrent workers still hit different banks, and a chunk lands complete before
  its push. The old order was a whole-row rotation, which can't be streamed in order.
- Restore read trid 0 after the loop. The stick-push poll inside the loop is unchanged.
- One-packet read template (`max_page_size = 64`) for the 32 B face rows. This skips the any-length split loop.

The parent's two-wave AG (r03-b04-a02, flawed at 20 cores per its reflection) is reverted: the factory, reader and
writer are restored to r03-b04-a01 and the forked wave forwarder is deleted. So this node is a01 plus the gamma
change above.

## Why this is not a repeat
- r01-b01-a02 and r01-b04-a02 interleaved gamma reads into the NCRISC input stream. Here gamma stays on BRISC/NoC1
  (r01-b04-a03's fix). Only its delivery to compute is pipelined.
- r01-b04-a04 polled for the stick push inside the gamma loop (kept). Nobody has changed WHEN compute may consume
  gamma: since r01-b04-a03 it has always been one push after one barrier.
- r03-b04-a02 (parent) staggered whole cores. This node is per-core and changes no cross-core protocol.

## Expected effect and risk
- Healthy cores: x*gamma starts at PRE end (~6.3-6.9 us at h7168) instead of at gamma end (~6.9-7.5 us). That adds
  ~0.5 us of slack and removes the small x*gamma overhang on the right-half cores at h7168. Expect h3584/h4096 about
  unchanged (gamma slack is 2-2.5 us there).
- Straggler: its x*gamma starts ~1.3 us earlier, so its POST lag should drop from ~3.5 to ~2 us, and its finish lag
  shrinks enough to weaken the self-sustaining late start. Expected dev 0 h6144/h7168 kernel -0.5..-1.5 us, roughly
  +1-3% on those shapes and +0.5-1.5% geomean. Judge with `analysis/late2.py`: dev 0 straggler calls and
  last-med drain should fall.
- Risk: the trid tag is sticky on BRISC's NoC1 read cmd buf. If it leaked into the later gathered-stick reads, those
  would still complete (the global counters are unaffected) because it's reset after the loop. A wrong chunk/page
  mapping would show as a PCC failure (gamma wrong on some tiles), not a hang. The trid barrier on a never-issued
  trid returns immediately. Accuracy should be bit-identical.
