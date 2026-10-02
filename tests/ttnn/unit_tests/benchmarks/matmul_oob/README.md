# Matmul out-of-box config benchmark

Validation for the new default matmul program config selection (#57884, `ttnn.CONFIG.matmul_auto_config_v2`,
code in `ttnn/cpp/ttnn/operations/matmul/device/config/matmul_auto_config.{hpp,cpp}`). It compares the legacy
default selection (`oob`) with the new one (`v2`) on the same inputs.

| File | Purpose |
|---|---|
| `run.py` | Runs one named suite (see Suites) |
| `run_all.sh` | Runs the complete validation below in one go |
| `cases.csv` | The 539 benchmark cases: curated shapes from issues and models, a generic grid, sharded cases, and 304 real-model calls from the model tracer (traced tier) |
| `suite.py` | Case definitions (`Case`), the curated tiers, `cases_from_csv` |
| `run_suite.py` | Runs cases in one or more modes; records the selected config, device kernel time, per-RISC times, cores and PCC |
| `summarize.py` | Per-tier speedup of one mode over another, regressions, status changes |
| `pytest_device_time.py` | pytest plugin: per-test outcome and total device kernel time |
| `compare_pytest_times.py` | Compares two `pytest_device_time` outputs (`--auto-only`: just the tests that use the default selection) |
| `pytest_auto_tests.txt` | The `pytest-auto` suite's tests |
| `sweep_configs.py` | Times alternative explicit configs for a case (investigation only) |
| `compare_runs.py` | Joins two `run_all.sh` result directories per case and per pytest test |
| `results/<arch>/` | Reference results of `run_all.sh` (same layout as a run directory) |

## Suites (`run.py`)

`run.py --suite SUITE` runs one named suite, legacy selection against `matmul_auto_config_v2`, and writes
`generated/matmul_oob/<arch>_<git rev>/<suite>/` with a `summary.txt`. Rerunning a suite skips finished parts.

| Suite | What | Time on Wormhole |
|---|---|---|
| `gist` | The #57884 gist sweeps (`gist/`): 116 2D-routed and 64 1D-routed Llama shapes, bf16, HiFi4, fp32 dest acc; mean wall time over 20 calls; PCC | about 20 min |
| `gist-fast` | The same without the PCC check | about 10 min |
| `gist-device` | The gist sweeps' 180 shapes as benchmark cases (`suite.py` tier `gist`): device kernel time and PCC, like `validation`. Use this one to compare against other suites; the gist harness's wall time includes host dispatch | about 30 min |
| `validation` | Every case in `cases.csv` (device kernel time, PCC against torch) | about 1.5 h |
| `validation-fast` | `cases_fast.csv`: 41 cases across the tiers, including block-float and fp32-accumulation cases | about 5 min |
| `pytest` | The matmul pytest directory, flag off and on (outcome and device time per test) | about 50 min |
| `pytest-fast` | Every 10th test of it (`DEVICE_TIME_SAMPLE=10`) | about 5 min |
| `pytest-auto` | The tests in `pytest_auto_tests.txt`: the matmul pytest tests in which some matmul goes through the default config selection. The others pass their own program configs, so the flag cannot change them and they only add noise | |
| `all` | `validation`, `gist-device` and `pytest` (the suites that report device kernel time) | |

```bash
tests/ttnn/unit_tests/benchmarks/matmul_oob/run.py --suite gist-fast
MM_KCAP=32 tests/ttnn/unit_tests/benchmarks/matmul_oob/run.py --suite gist-fast --out generated/matmul_oob/kcap32
```

`pytest_device_time.py` records, for each test, the last config the default selection chose during it
(`auto_config`, null when every matmul in the test passed its own program config). The `pytest` suite's summary
reports every test and then the default-selection tests only, and writes their node ids to
`pytest/auto_tests.txt`. To refresh `pytest_auto_tests.txt` after tests are added, run the `pytest` suite and copy
that file over it. A test skipped on the machine that made the list is not in it.

"Legacy" is the same build with the flag off, which is within about 1% of main on these suites. To compare
against main itself, run the gist suite on a main build: `gist/time_default.py` works there too (it only
times the default config). `gist/bh_reference_*.log` are the gist author's Blackhole 12x10 results (tt-metal
default, a searched oracle and the tt-mlir rule), for context.

## Running it on a new machine (e.g. Blackhole)

### 1. Check out and build

```bash
git fetch origin rmillerTT/mm-oob
git checkout rmillerTT/mm-oob
./build_metal.sh --release --build-tests      # profiler (Tracy) stays enabled; it is required
source python_env/bin/activate                # create it first with ./create_venv.sh if needed
```

The branch includes the Reuse factory fix from #57957 (the new selection relies on it). Device times come from
the in-process device profiler, so do not build with `--disable-profiler`.

Check that the device is idle and healthy first (`tt-smi`), and reset it after any hang (`tt-smi -r 0`).

### 2. Run everything

From the repo root:

```bash
tests/ttnn/unit_tests/benchmarks/matmul_oob/run_all.sh 2>&1 | tee generated/matmul_oob_run.log
```

This runs, in order:

1. The selector gtests (`MatmulAutoConfig.*`, device-free, seconds).
2. The benchmark suite: all 539 cases in `cases.csv`, each in both modes, 2 warmup + 5 timed calls per mode,
   PCC against torch on a NaN-filled output (so tiles the kernel never writes fail). About 1.5 hours on Wormhole.
3. The whole matmul pytest directory (`tests/ttnn/unit_tests/operations/matmul/`), once with
   `matmul_auto_config_v2` off and once on (set through `TTNN_CONFIG_OVERRIDES`), recording each test's outcome
   and device time. About 1.5 hours per pass on Wormhole.

Everything goes to `generated/matmul_oob/<arch>_<git rev>/` and is archived as `<that dir>.tar.gz` at the end.
Run just part of it with `--skip-suite` or `--skip-pytest`, or choose the directory with `--out DIR`.

If a run is interrupted (or the device hangs: reset it, then rerun the same command), rerunning continues where it
stopped: the suite resumes from its CSV and a completed pytest pass is not repeated. To force a pytest pass to
rerun, delete its `pytest <mode> done` line from `run_info.txt`.

### 3. What to send back

The archive `generated/matmul_oob/<arch>_<rev>.tar.gz`. Its contents:

| File | Contents |
|---|---|
| `run_info.txt` | Arch, git rev, step timestamps, headline numbers |
| `gtests.log` | Selector gtest output |
| `suite.csv` | One row per case and mode: config, device time, per-RISC times, cores, PCC, status |
| `suite_summary.txt` | v2 over legacy: geomean and counts per tier, status changes, largest regressions and speedups |
| `suite.log` | Raw suite output (for errors) |
| `pytest_off.jsonl`, `pytest_on.jsonl` | Per-test outcome and device time |
| `pytest_compare.txt` | Outcome changes and device-time speedup, flag on over off |
| `pytest_off.log`, `pytest_on.log` | Raw pytest output |

### 4. Compare with the Wormhole reference

```bash
python3 tests/ttnn/unit_tests/benchmarks/matmul_oob/compare_runs.py \
    tests/ttnn/unit_tests/benchmarks/matmul_oob/results/wormhole_b0 generated/matmul_oob/<arch>_<rev> \
    --out generated/matmul_oob/comparison
```

`suite_by_case.csv` has, per case, both runs' legacy and v2 times, statuses, configs and speedups, with a note
where they disagree (a regression in only one of the runs, or a v2 error or PCC failure). `pytest_by_test.csv` has
each pytest test's outcome and device time with the flag off and on in both runs. `summary.txt` has the counts.

## Reading the results

`suite_summary.txt` reports speedup = legacy time / v2 time. A case is counted as a regression below 0.95 and as
an improvement above 1.05. `status` is `ok`, `pcc_fail` (wrong output), `error` (the op raised, e.g. an invalid
config or L1 overflow) or `infeasible` (the case's L1-resident tensors don't fit this device; skipped in both
modes). A v2 `pcc_fail` or `error` where legacy is `ok` is a bug in the new selection. `fallback` = 1 on a v2 row
means the new selection did not handle the inputs and the legacy one was used.

`pytest_compare.txt` lists tests whose outcome changed with the flag on, then device-time changes for tests
above 20 us. Tests with explicit program configs are not affected by the flag, so their time changes are run-to-run
noise (±20% on tiny tests on Wormhole).

### Wormhole reference (n150, 8x8 grid)

`results/wormhole_b0/` (compressed; `compare_runs.py` reads it directly), from commit 64f6f185ae6:

| Tier | Cases | Geomean v2 / legacy | Faster >5% | Slower >5% |
|---|---|---|---|---|
| issues | 79 | 1.63 | 56 | 1 |
| models | 48 | 1.15 | 28 | 0 |
| generic | 90 | 1.62 | 48 | 1 |
| sharded | 11 | 1.14 | 1 | 0 |
| traced | 304 | 1.88 | 189 | 3 |
| all | 532 | 1.70 | 322 | 5 |

No v2 errors or PCC failures and no v2 fallbacks to the legacy selection; one case that fails with the legacy
selection (`s_o_h_8192x512x512`) works. The 5 regressions (5-18%) are understood and are not fixable with a
generic heuristic on Wormhole data:

- N of at most 8 tiles (`t_linear_3c39fac2e6` 0.82x, `t_linear_3f7267544b` 0.88x): 2D degenerates to
  one-tile-wide blocks, legacy uses 1D.
- `g_256x4096x1024_bfp8_dram` (0.93x): a small 2D block that wants a deeper K block.
- `i31743-dram.dram.l1` (0.95x): 1D vs 2D; the same shape with a DRAM output goes the other way.
- `t_linear_7e2d79e94a` (0.95x), and other decode linears with bf16 weights on 40-64 cores just under 5%:
  DRAM-bandwidth bound, a shallower K block would be better.

Matmul pytest directory, flag on vs off: 1 outcome change, `test_matmul_activation_with_sharded_input` (PCC
0.99988 against its 0.9999 threshold; v2's deeper K block, and neither selection reaches 0.9999 against an fp32
reference). Device time geomean 1.15x over 893 timed tests.

The selection has no Blackhole-specific rules yet. Blackhole results that differ from these patterns (for
example a different number of regressions from the core-count or K-depth choices) are exactly what this run is
meant to find.
