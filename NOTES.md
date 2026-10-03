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

## Next
1. Read the 4 JSONs. For each layer with best_us <= 0.97 * table_us, update _BLOCKINGS key (4,8,...) in
   models/tt_dit/utils/conv3d.py. Check output_check (PCC ~1; md5 may differ when C_in_block changes).
2. Then one short decode A/B on blx03 2x4 (t97 harness pattern, LTX_CONV3D_BLOCKING_MESH=4,8): wall time and YUV
   md5/PSNR vs current table. Clean /var/tmp/fasth3/t100/src afterwards.
