# t263 (R1b: widen DiffVAE NA K/V L1 ring) — notes

## Finding before any device run
- The spec's levers (more ring columns via smaller CBs, K-only ring, BF8 K/V) look moot:
  - The 1-D W ring already holds gather_width-1 = 3 columns of K and V (mode=2), i.e. every reusable brick.
    More columns cannot raise the hit rate. A K-only ring (mode 1) only reuses less.
  - BF8 K/V is #253's lever (DIFFVAE_NA_BF8, ttp/t253 47e19132826): not duplicated here.
  - #260 cut DRAM K/V reads ~4x (1 new column of 4 per chunk) but NA only went 152.3 -> 121.0 ms/block.
    So the op is probably no longer DRAM-bound. Per core: ~286 work items x 14 kv chunks = ~4000 chunks
    in 121 ms ≈ 40k cycles per 8-tile kv chunk, about 2-3x a rough compute estimate.
- To settle it, commit 6a23f8dfe10 adds DIFFVAE_NA_ABLATE=reads|math. It is a hashed, timing-only
  diagnostic and the output is garbage: `reads` skips all K/V reads (gives the compute floor); `math`
  only drains the CBs (gives the reader floor).

## Device run (blx01)
- Driver: blx01 /var/tmp/fasth3/t263/drv/driver263.sh (copy in t263-drv/). Log: drv/driver.log.
  Marker: drv/driver.marker (`T263_DRIVER_DONE stage=.. rc=.. jobs=..`).
- Build: /var/tmp/fasth3/t252/b (reused, as t260 did) @6a23f8dfe10. Incremental, OK.
- Job A = def + reads (-t 480); job B = math (-t 300). One process per arm, SEEDS=0, profiled stage trees.
  Outputs: /var/tmp/fasth3/t263/out_A, out_B (stage_tree_<arm>.txt, run.log).
- DROP: job 949 (A, ours) killed by broker device recovery at ~2026-10-08 05:23:50 UTC, box g15blx01,
  chips 24-31 (bridge reset 951/952 failed exit 8). Before the kill, the def arm gave 3.115 s,
  md5 13802b012e19cf652be8d9c88cdd9316 (= job 946 ring default, so the build is valid).
  The driver reruns A once the broker is healthy twice in a row.

## Next step
On marker: read driver.log `summ` lines (neighborhood-sdpa ms per arm).
- reads ≈ def (121): the op is compute/overhead-bound, so ring widening, K-only and BF8 cannot help.
  Report that; the next lever is on the compute side (bigger kv chunk / fewer per-chunk ops).
- math ≈ def: the op is reader-bound. Then reader-side levers (fewer per-tile NOC issues: one read per ring
  entry for K too, cheaper index math) are worth a knob.
