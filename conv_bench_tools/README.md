# conv_bench profiling tools (GH #45995)

Reusable drivers for the conv2d 3-mode (main / helper_sbm / helper_trm) real-config baseline sweep.
Branch `wransom/conv_bench`. Assumes repo at /localdev/wransom/tt-metal and uses /tmp/cb_last.log scratch.

## Files
- `cb_warm_conv.py` — reads newest `generated/profiler/reports/*/ops_perf_results_*.csv`, prints the warm
  (2nd of run_twice) `Conv2dDeviceOperation` `DEVICE KERNEL DURATION [ns]`.
- `cb_bench.sh <reps> <mode...>` — runs each mode under `python -m tracy`, appends a row per (config,mode) to
  `conv_bench_data.csv`. Config comes from `CB_*` env (+ `MODEL`,`LABEL`). Statuses: ok / OOM / HANG / FATAL / FAIL.
- `cb_probe.sh` — fast eligibility/fit probe (no tracy, hang-guarded), prints the CONV_BENCH tuner line.
- `cb_collect_{rn50,sdxl,vu}.sh` — the WH real-config family sweeps (the conv lists + real configs per family).

## Run a family sweep
    bash conv_bench_tools/cb_collect_rn50.sh   # etc.  (long; run in background)

## BLACKHOLE continuation plan
1. Rebuild on the BH machine first: `./build_metal.sh` (profiler is ON by default; the validate relaxation +
   hang fix + real-settings rewire are all committed, so a fresh BH build includes them). Then recreate the
   /tmp scratch is unnecessary — these scripts are committed.
2. Re-run all three `cb_collect_*` drivers on BH → WH-vs-BH cross-arch migration comparison. The 10 cases that
   were "did-not-fit n150" (7 SDXL: 384<-1152 128², 1536<-1536 64², 768<-1536 64², 768<-2304 64², 384<-384 128²,
   384<-768 128², 768<-768 128²; 3 vanilla_unet: 32<-32 480x640, 128<-256 120x160, 64<-128 240x320) should be
   re-checked — BH's larger L1 may fit them (the whole point).
3. Add a VAE driver (SDXL VAE 256-512ch, 512²/1024², groups=1, bf8 weights, BS/HS) — these are heavyweight,
   eligible, and only OOM'd WH for lack of L1; BH is where they may run without DRAM slicing.
4. Note: BH DST capacity / L1-per-core differ from WH → the helper_trm relax-eligibility band shifts, but
   helper_trm stays N/A for real (tiled) convs; the migration (main vs helper_sbm) comparison is unaffected.
