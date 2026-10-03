# t100 notes (halo-mode conv3d re-sweep)

## Why conv3d rose 308 -> 360 ms (t61 job 029 vs t96)
- Same blockings, same 1150 MHz clamp. 029 ran conv3d on a pre-padded zeros input (NeighborPad 119 ms + T concat
  38 ms outside the conv). t96 runs the halo-only reader (unpadded shard, T replicate, halo buffer, H/W logical masks,
  pad_offset). That moves ~52 ms into conv3d while removing ~150 ms elsewhere. 029's per-op CSV is gone.
- t96 per-layer conv3d (ms, 8-chip max): s2_res 138.5 (9x15.39), s3_res 83.3 (12x6.95), s4_res 71.3 (8x8.92),
  s1_up 23.1, s3_chg 13.6, s1_res 10.5, s0_res 9.1, s0_up 6.4, s4_out 4.5.
- Job 458 confirms per layer: s2_res table blocking 15240 us halo vs 13376 us pre-padded (+14%).

## Harness (47aecb9bdd7, 82f0ee930dd)
- run_sweep(halo=HaloSpec(...), table_key=..., max_seconds=..., near_table=True). LTX list + device test in
  models/tt_dit/tests/models/ltx/bruteforce_conv3d_sweep_ltx.py (opens (4,8), create_submesh(2,4)). CPU test
  test_conv3d_sweep_halo_cpu.py: 8 pass. hw_product=32 only (16/64 hung in wan 2x4 sweeps), max_t_block 8.
- JSON per layer: table_us (halo), table_padded_us, best_blocking/best_us, top_20, output_check
  (md5/max_abs_diff/PCC of best vs table on first and last device).

## Device run (blx03)
- Driver: /var/tmp/fasth3/t100/src/tmp/blx03/t100/driver100.sh, launched 2026-10-03 07:18 UTC. One broker job per
  layer, in order s2_res s3_res s4_res s1_up (first job 458). Marker `T100_DRIVER_DONE <stage> <rc>` in
  g14blx03:/var/tmp/fasth3/t100/driver.log; rc 9 = drop/reboot during OUR job -> stop ALL device work, report.
  Per-layer log run100_<layer>.log, results in /var/tmp/fasth3/t100/results/<layer>_*.json.
- Re-launch for missing layers: `LAYERS="..."` env; finished layers have results/<layer>_done.

## Sweep results (jobs 458/461/464/467, all clean; JSONs in tmp/t100/results/, not committed)
| layer (calls) | table us | best (any) | best keeping table C_in_block |
|---|---|---|---|
| s2_res (9) | 15240 | table is best | - |
| s3_res (12) | 6924 | table is best (6894) | - |
| s4_res (8) | 8899 [128,64,6,2,16] | 7327 [64,128,6,4,8] -17.7%, PCC 0.99993, max_abs 2 | 8170 [128,64,6,4,8] -8.2% |
| s1_up (1) | 22963 [128,64,5,4,8] | 20264 [64,256,1,2,16] -11.8%, PCC 1.0, max_abs 4 | 20915 [128,64,5,2,16] -8.9% |
Expected decode gain: exact arm ~8 ms, best arm ~15 ms (per chip, 1150 MHz clamp).
_HALO_LAST_KEYS/_FORCE_SPATIAL_KEYS in conv3d.py are not read anywhere; only _BLOCKINGS matters.

## Decode A/B (job 469, launched 07:57 UTC blx03)
- Stage branch ttp/t100-stage = this branch + t96 trace harness (057e841c056, 489e09bcd36; vae_ltx.py conflict
  resolved by keeping both exact_shard and trace_decode). Staged at /var/tmp/fasth3/t100/src.
- runab100.sh: arms table, exact, best, table2 via ab100.py (patches _BLOCKINGS in-process). Log
  /var/tmp/fasth3/t100/runab100.log: grep "AB arm=traced\|T100AB_\|CMP100\|T100_ARM". Driver marker
  "T100_DRIVER_DONE ab <rc>" in driver.log (old sweep log: driver.log.sweep). rc 9 = drop during our job -> stop all.

## Decode A/B result (job 469, clean, traced 544x960/145f, 2x4 submesh, mesh key 4,8)
| arm | traced decode s (3 runs) | min | vs table |
|---|---|---|---|
| table | 0.5367 0.5197 0.5209 | 0.5197 | - |
| exact (s4 [128,64,6,4,8], s1_up [128,64,5,2,16]) | 0.5218 0.5109 0.5062 | 0.5062 | -13.5 ms (-2.6%), md5 identical |
| best (s4 [64,128,6,4,8], s1_up [64,256,1,2,16]) | 0.5262 0.5107 0.5118 | 0.5107 | -9 ms, PSNR 52.6 dB, not identical |
| table2 | 0.5210 0.5226 0.5218 | 0.5210 | md5 identical to table |
Picked exact: bit-identical, and best is no faster. _BLOCKINGS updated for s4_res and s1_up (4,8 keys).
s2_res/s3_res unchanged (table already fastest in halo mode).

## Done
Branch has the _BLOCKINGS change + CPU test. blx03 staging (src, ab) removed; logs/results kept in
g14blx03:/var/tmp/fasth3/t100.
