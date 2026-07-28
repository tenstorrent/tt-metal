# rms_norm perf experiments (Perf tournament round 1)

One directory per floated idea. Each is a self-contained perf-lab-style A/B
micro-benchmark of ONE stage, authored by a `blocking-perf-part-optimizer`
subagent, measured on device, reporting ns + the predicate the win holds for.

`zone_report.py` reads `generated/profiler/.logs/profile_log_device.csv` (written
by `scripts/run_safe_pytest.sh --profile`) and aggregates the permanent
`MaybeDeviceZoneScope` stages per RISC per core:

    python3 tests/ttnn/unit_tests/operations/rms_norm/perf_experiments/zone_report.py [run_host_id]

## Round-1 measured breakdown (blackhole_p150b, 11x10 grid, AICLK 1349.99 MHz)

Focus shape: `(1,1,8192,1024)` BLOCK_SHARDED, shard `[1024,128]`, grid `(8,8)`,
bf16 / TILE / `fp32_dest_acc_en=False` / HiFi2 / bf16 TILE gamma.
Derived: 64 cores, `cw=8 cw1=8 cw2=1` (flat), per core 32 tile-rows x `Wt=4`,
`nw=1`, `ht_block=8`, `nh_core=4`, `fuse_sq=1`, `x_res=1`, `gamma_res=1`.

Whole op 76_112 ns; `feature_spec` reference 25_640 ns (2.97x over).

| rank | stage | critical-path ns | why it is not at a roofline |
|---|---|---|---|
| 1 | `cmp_rsqrt` real work | 29_000 (38%) | 32 fp32 tiles/core, `block_size = 1`, all 1024 lanes computed where only column 0 is consumed |
| 2 | combine round trip | 19_400 (26%) | 1 MB gathered into one root; necessary payload is 8 KB |
| 3 | `cmp_scale` unpack | 17_600 (23%) | 128 tile unpacks + per-window fp32 srcB reconfig |
| 4 | `cmp_square` | 6_600 (9%) | already 1 fused FPU op/tile at HiFi2 |
| 5 | `cmp_gamma_mul` | 3_950 (5%) | at FPU floor; its `cb_scaled` round trip is removable |
| 6 | reader/writer DM | ~130 | zero-copy sharded both sides — saturated at zero |
