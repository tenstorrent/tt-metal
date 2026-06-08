# Precompile benchmark — cold vs warmup+warm, with full CPU/JIT instrumentation

A reproducible, instrumented local benchmark of the precompile system on a **fixed 75-test
layernorm suite**. It answers, in detail, where time and CPU go in each way of running a suite,
and what the precompile path actually buys end-to-end.

## The two ways of running, and the north-star metric

| method | what happens |
|---|---|
| **COLD** | the standard run: kernels JIT-compile **inline & serial** during the test run |
| **PRECOMPILE** | **WARMUP** (phase 1 = *collect* + *precompile*, hardware-free & parallel) then a **WARM** run that reuses the on-disk JIT cache |

**North-star = total end-to-end wall.** For COLD that is the run itself; for PRECOMPILE it is
`probe_real + probe_mock + warmup + warm` (the honest total, exactly as `run_safe_pytest.sh
--precompile` charges it). Speedup = `COLD / PRECOMPILE`.

## Two cache axes (the "multiple ways")

1. **JIT binary cache** (`TT_METAL_CACHE`) — the precompile target. Reset before every COLD run
   and before every WARMUP, so COLD is *truly* cold and WARMUP *truly* fills it; the WARM run
   then reuses it. This is the cold↔warm axis itself.
2. **Compiler cache** (`ccache`) — caches the C++→object step *underneath* the kernel JIT.
   **Kernel JIT only uses ccache when `TT_METAL_CCACHE_KERNEL_SUPPORT` is set** (`build.cpp:115`);
   the default/CI behaviour does **not**. So three ccache conditions are benchmarked:
   - `off` — flag unset (default/CI). ccache plays no part in kernel JIT.
   - `on/deleted` — flag set, isolated `CCACHE_DIR` emptied first → **cold compiler cache**.
   - `on/warm` — flag set, `CCACHE_DIR` reused from the preceding `deleted` run → **warm compiler cache**.

The full matrix is `{COLD, PRECOMPILE} × {off, on/deleted, on/warm}` = 6 configs, repeated
`BENCH_REPEATS` times (default 2), reported by median.

## What is measured, per phase

* **wall / user-CPU / sys-CPU / %CPU / peak-RSS** — `run_and_time.py` wraps each phase and reads
  `getrusage(RUSAGE_CHILDREN)`, which aggregates the **entire reaped descendant tree** (pytest +
  every JIT compiler it forks). This is the stand-in for `/usr/bin/time -v` (absent on this host).
* **CPU over time + compiler-vs-host split** — `cpu_sampler.py` samples the orchestrator's process
  subtree every 0.25 s from `/proc`, attributing each tick's CPU-seconds to `compiler`
  (riscv-tt-elf-g++/cc1plus/ccache/sfpi/…), `python` (pytest/ttnn), or `other`. The summarizer
  integrates this over each phase's window (from `marks.csv`) → **mean cores, peak cores, core
  utilization (of nproc), and the fraction of CPU that went to the JIT toolchain**.
* **JIT telemetry** — `build_cache_telemetry.cpp` logs `JIT cache stats: H/T hits (P%) [C cached…]`.
  `jitted = T − H` (kernels actually compiled), `H` = served from cache. Parsed from every run log.
* **WARMUP sub-split** — the plugin prints `compiled N programs in Xs`; the summarizer treats the
  last `X`s of the warmup window as the **compile** subphase (the parallel JIT burst) and the rest
  as **collect** (single-process meta shape-propagation), reporting each subphase's utilization.

## Why these particular knobs

* The suite is **75 deterministic node-ids** (`layernorm_75.txt`), a strided sample across the 259
  `test_layer_norm.py` cases for shape/kernel diversity. Fixed list ⇒ identical selection every run
  and across COLD/WARMUP/WARM (the warm collect must mirror the real run).
* Runs hold the **device flock** for the whole matrix so nothing interleaves and CPU readings are clean.
* `nproc` here is **8** (cgroup-limited), not the 32 logical host CPUs — so "utilization" is against
  8, and the system-wide `/proc/stat` figure is reported only as a contention signal.

## Run it

```bash
scripts/precompile_bench/run_bench.sh                 # 75 tests, 2 repeats -> /tmp/lnbench
BENCH_REPEATS=3 BENCH_OUT=/tmp/lnbench3 scripts/precompile_bench/run_bench.sh
scripts/precompile_bench/summarize.py /tmp/lnbench    # -> stdout + /tmp/lnbench/SUMMARY.txt
```

Smoke-test the harness on a tiny slice first:

```bash
head -3 scripts/precompile_bench/layernorm_75.txt > /tmp/s3.txt
BENCH_SEL=/tmp/s3.txt BENCH_OUT=/tmp/smoke BENCH_REPEATS=1 scripts/precompile_bench/run_bench.sh
```

## Files

| file | role |
|---|---|
| `layernorm_75.txt` | the fixed 75 node-ids (the suite) |
| `run_bench.sh` | orchestrator: the 6-config × N-repeat matrix, device lock, per-phase timing + marks |
| `run_and_time.py` | per-phase wall + whole-tree CPU + peak RSS (getrusage stand-in for `time -v`) |
| `cpu_sampler.py` | 0.25 s `/proc` subtree sampler → CPU-seconds split compiler/python/other |
| `_probe_real.py` / `_probe_mock.py` | the real-device / hardware-free build_key + fingerprint probes |
| `summarize.py` | joins phase results + marks + sampler → the report below (`SUMMARY.txt`) |

## Caveats (honest reading)

* PRECOMPILE carries fixed overhead (`probe_real + probe_mock` device opens + warmup startup).
  On a tiny suite it is **slower** than cold (overhead not amortized); the win grows with suite size.
* The meta-collect can miss a small % of programs (host-side prep it skips), so the WARM run may
  cold-compile a few kernels (hit-rate <100%) — correct, just not a full speedup.
* `compile_wall` is the plugin's own measure; the collect/compile sampler split uses it as the
  window boundary, so the split is approximate at the ~0.25 s sampling granularity.

## Results

See `RESULTS.md` (generated narrative) and `SUMMARY.txt` in the output dir for the latest numbers.
