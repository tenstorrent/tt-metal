# t212 — port t11/t23/t25 DiffVAE optimizations onto t48

Branch ttp/t212-c1-port-t11-t23-t25-diffvae-optimization, base origin t48 5e4e0cd643a.
Code commits (oldest first, land these): e13d711079d 5fa467e1094 288d095c131 e0aa465bd01
382a5928005 b6521b3a7d0 a2fe56d60c1 8b1167ef43b (HEAD of code = 8b1167ef43b).
Skipped: e72eef929d0 (superseded by 55f146f4891), 7452faf3b38 (already in t48), 21ee6a13483 (notes only).

## Running on blx01 (/var/tmp/fasth3/t212)
- Build: setup212.sh -> worktree b/ at 8b1167ef43b, Release build. build.log ends "BUILD212_DONE rc=0".
  Built on blx01, not g15: g15 footprint is ~94/100 GB and a g15 build can't serve blx01 jobs.
- Driver: driver212.sh (copy in tt-project/t212/), started 2026-10-07 18:37 UTC (local log time).
  Waits for the build, makes treeA (t208 overlay of 5e4e python + 8b1167 test file), then one broker
  job per arm, -t 600: A = unported t48 build (/var/tmp/fasth3/t48), B = ported build b/.
  Both run test_diffvae_ltx.py::test_decode_wsp_timing -k s34x60, ring, 2 links, slab 78,
  dumping pixels to out_<A|B>/px.pt. Then cmp212.py -> cmp.json (PCC, PSNR, worst frame).
- Marker: driver.marker "T212_DRIVER_DONE stage=.. rc=.. jobs=..". Log: driver.log.

## Drops
- 2026-10-07 ~18:23 UTC, blx01, job 795 (t209's, not ours): chip 13 off PCIe after a timeout,
  broker HELD/degraded at 18:35 UTC. Driver waits on the health check.

## Next step on wake
1. Read driver.log, cmp.json, out_A/run.log and out_B/run.log ("[decode" ms lines, timing tree).
2. Need PCC >= 0.9999 (B vs A) and B ~5.5 s.
3. ttp checks; land code commits via a -land branch from origin t48 + ttp push --detach;
   ttp push --own --detach.
4. Delete out_*/px.pt (~900 MB each) on blx01. Keep b/ (follow-up DiffVAE tasks use it).

## Result (2026-10-07 19:14 UTC, blx01 jobs 810/812)
A unported 12009 ms, B ported 5635 ms. B vs A: PCC 0.999919, PSNR 53.7 dB (worst frame 57: 52.6 dB).
Code landed via ttp/t212-land (cherry-picks onto 5e4e0cd643a, head a40d78b8bae) with ttp push --detach.
