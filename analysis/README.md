# SDPA single-chip bottleneck analysis (TEN-4679)

Helper tools that produced the data on the Confluence page
*SDPA Single-Chip Bottleneck Analysis* (space TA).

## Files

- `sdpa_sweep.py` — pytest driver that runs `test_sdpa_prefill::run_test_sdpa_tt`
  for the Llama 3.1 8B SDPA shape across a sequence-length sweep. Defaults to
  S ∈ {1024, 2048, 4096, 8192, 16384, 32768}, nh=32, nkv=8, BF8, causal.
- `post_process.py` — reads tracy's `profile_log_device.csv` and derives the
  five ticket answers. Recovers full 32-bit `ref_cnt` from auxiliary
  profiler-id 9091 records when present (degrades gracefully to the 24-bit
  truncated value otherwise).
- `make_charts.py` — writes the four PNGs that go on the Confluence page.
- `ten4679_results.csv` — last produced per-S summary table.

## Run

```bash
cd $TT_METAL_HOME
python -m tracy --profiler-capture-perf-counters=all \
                -p -- pytest analysis/sdpa_sweep.py::test_sweep -s

python analysis/post_process.py \
  /path/to/tracy/reports/<TS>/profile_log_device.csv \
  --seq-lens 1024 2048 4096 8192 16384 32768

python analysis/make_charts.py analysis/ten4679_results.csv -o analysis/charts
```

`--perf-counter-groups=all` is equivalent to `fpu,instrn,unpack,pack,l1_0`.
L1 bank 1 (NOC Ring 1) requires a separate run with `=…,l1_1` because the L1
banks share the same hardware counters via a 3-bit mux selector.

## Notes

- Per-core perf-counter percentages are intrinsic to the kernel, not the head
  count. The `nh / nkv` ladder in `sdpa_sweep.py` exists only to dodge OOM
  failures when profiling at the full Llama shape on smaller boxes.
- Q2 (dest contention) is derived via `MATH_INSTRN_AVAILABLE − AVAILABLE_MATH`
  (the FPU data-hazard scoreboard stall), because the dedicated
  `WAITING_FOR_SFPU_IDLE_*` counter is dormant on Black Hole for SDPA — the
  kernel uses `tile_regs_acquire/wait/release` for dest sync, not
  `STALLWAIT(WAIT_SFPU)`.
- Q3 (single-cycle-exp counterfactual) is analytical: cycles-per-`exp_tile`
  comes from the LLK source comment in
  `tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_exp.h`
  (~68 cycles for a full tile in the `<APPROX, ITERATIONS=32>` config SDPA
  uses).
- The `ref_cnt` 24-bit packing in `perf_counters.hpp` truncates the BH 32-bit
  hardware counter for kernels longer than ~12 ms. The fix (auxiliary
  profiler-id 9091 record) lives on the `perf_counters_integration` branch;
  `post_process.py` here can use those records if present.
