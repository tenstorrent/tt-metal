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

## Run 2 (2026-10-08 ~10:50 UTC)
- blx01 host rebooted 09:51 UTC (second drop, t272 job 984, chips 16-23, 09:45 UTC); driver263 died with it.
  $F/t252/b was deleted, so a fresh worktree build $F/t263/b @6a23f8dfe10 (log t263/build2_*.log).
- Update from #267: cherry-picked 47e19132826 (DIFFVAE_NA_BF8) as e5f71e2f899. Found that the reader sized
  bf16 mask pages with the Q/K/V tile size, wrong for bf8 operands: fixed in 51ecd3ac3e2 (kernel only).
- Tree runs @51ecd3ac3e2 (python/kernel-only on top of the 6a23f8dfe10 C++ build).
- Jobs submitted by hand from the run (no driver): A = def + reads, B = math + bf8 (bf8 scored, HOST_SEEDS=0,1).
- Built OK (build2.rc=0), t263/b checked out @51ecd3ac3e2.
- Job A resubmitted as broker job 003 (10:55 UTC, queued behind t261 job 002).
- DROP 2: job 003 (ours, t263 A) killed by device recovery at 2026-10-08 10:59:30 UTC, g15blx01, chips 16-23
  (tray 3), 66 s in, during JIT warm-up (no NA had run). Bridge resets 005/006 failed, health-gate 007 failed.
  Config A (def+reads) has now dropped twice in a row on blx01 (949, 003): skipped there per the rules,
  though both drops look box-wide (t272 job 984 and t261 job 931 dropped the same way today).

## Next step (exact)
When t263-drv/probe.sh exits 0 (blx01 healthy, last incident >= 15 min old, no smarton job):
ssh g15blx01: check `git -C /var/tmp/fasth3/t263/b rev-parse --short=11 HEAD` = 51ecd3ac3e2 (box rebooted? dir is
on /var/tmp, survives), then from /var/tmp/fasth3/t263:
  tt-device-mcp run-bg "bash $T/drv/run263.sh 'math:DIFFVAE_NA_ABLATE=math bf8:DIFFVAE_NA_BF8=1' $T/out_B 'math bf8' 'bf8'" -w $T -e $T/drv/env.yaml -t 540
Then score: $F/t48/python_env/bin/python $F/t260/drv/cmp241.py $F/diffvae/ref $T/out_B/bf8 $T/out_B/cmp_bf8.json 0,1
Compare bf8 md5 vs def 13802b012e19cf652be8d9c88cdd9316 (must differ), decode vs def 3.113-3.115 s (946/949).
reads arm: skipped on blx01 (2 drops); follow-up on another box.

## 2026-10-08 11:19 UTC: job B submitted
blx01 broker job 020 (math + bf8), -t 540, log /var/log/tt-device-broker/2026-10-08_111857_020.log.
On wake: `ssh g15blx01 tt-device-mcp status -j 020`; then score bf8 per 'Next step (exact)' and compare.

## 2026-10-08 result: job 020 (math + bf8) completed, exit 0, 292 s
- math ablation (DIFFVAE_NA_ABLATE=math, reader only, compute drained): stage-5 NA 103.0 ms/block vs 120.8 default.
  Decode 2.959 s (output garbage by design). So reads are ~85% of NA time; compute adds only ~18 ms/block.
- bf8 (DIFFVAE_NA_BF8=1 on the ring default): NA 120.8 ms/block (= default), decode 3.166 s vs 3.113-3.115 default.
  md5 08fecde9a85471ad78fbd1d671be0629 (differs from def 13802b01..., A/B valid).
  Quality BROKEN on the ring path: PCC 0.253 / 0.271, PSNR 12.3 / 11.5 dB (seeds 0,1, cmp_bf8.json). Host noise 12.5/11.5 s.
- Conclusion: halving K/V bytes gives zero NA gain, so the reader is not DRAM-bandwidth-bound but bound by per-tile
  NOC issue / index math / latency. Ring widening, K-only ring and BF8 cannot reach the <=60 ms target. BF8 rejected
  (no gain, and broken with the ring: likely a ring-size/tile-size mismatch; not debugged, no point without gain).
- reads ablation (compute floor) never ran: config A dropped twice on blx01 (949, 003).
- Next lever: reader issue cost (one NOC read per ring entry/brick instead of per tile, cheaper index math,
  more outstanding reads). Nothing landed on t48.
