# Fabric builder perf: cleanup plan

Branch: `arajagopal/fabric-builder-perf-tests`

Goal: profile host-side fabric bring-up (cold cache vs. hot cache) and run it in CI on
`wh_llmbox_perf` (T3K). The test should follow the patterns the existing fabric perf tests
already use, so it reads like its neighbors and adding hardware or fabric configs is a
data-only change.

---

## 1. Patterns we are following

| Pattern | Existing example | What we take from it |
|---|---|---|
| pytest module wraps a C++ benchmark binary | `tests/tt_metal/microbenchmarks/ethernet/test_fabric_mux_v2_throughput.py` | `subprocess.run` the binary, binary writes JSON with a `context` block, Python owns goldens and pass/fail |
| Sweep via `@pytest.mark.parametrize`, one shared run function | `tests/tt_metal/microbenchmarks/ethernet/test_fabric_mux_bandwidth.py` | `fabric_config` is a parametrize axis |
| Golden CSV next to the test, keyed by hardware | mux v2: `fabric_mux_v2_throughput_golden_<arch>.csv`; BW test: `golden_*_summary_<arch>_<cluster>.csv` | `fabric_builder_perf_golden_<arch>_<cluster>.csv`, chosen from what the binary detects |
| Golden refresh via env var | `FABRIC_MUX_V2_THROUGHPUT_UPDATE_GOLDEN=1` | `FABRIC_BUILDER_PERF_UPDATE_GOLDEN=1` |
| Output dir under `generated/`, env override | `FABRIC_MUX_V2_THROUGHPUT_OUTPUT_DIR`, default `generated/fabric_mux_v2_throughput` | `FABRIC_BUILDER_PERF_OUTPUT_DIR`, default `generated/fabric_builder_perf` |
| Missing / stale golden rows are errors | mux v2 `validate_against_golden` (`missing-golden`, `stale-golden`) | same statuses |
| Summary CSV + text table printed and saved | mux v2 `summary_<arch>.csv` / `.txt` | same |
| Tracy capture from pytest | `tests/ttnn/tracy/test_realtime_profiler.py` `_run_under_tracy` | free port + `TRACY_PORT`, tool paths from `tools.tracy.common.PROFILER_BIN_DIR` |
| CI menu entry is a pytest command | `fabric_perf_tests.yaml` mux entries | `cmd: pytest -svv tests/tt_metal/tt_fabric/fabric_builder_perf/test_fabric_builder_perf.py` |

Consequences:

- No `config.json`. Hardware is detected at runtime (arch + cluster type), like the BW test.
- No `run.py` CLI. The pytest module replaces it.
- No `--hardware` argument. `{sku}` in the test list only matters for scheduling.

### Coding guidelines

The code should read like carefully hand-written repo code: small named functions, clear
control flow, and errors that surface where they happen. When unsure, follow an existing
pattern or ask.

**Error handling**

- No catch-all `try` blocks. Both `fabric_init_benchmark.cpp` (`try { ... } catch (const
  std::exception&)` around all of `main`) and `run.py` (`try: ... except Exception` around all of
  `main`, plus a second catch in `__main__`) do this today; both go away.
- C++: let failures propagate. Use `TT_FATAL(cond, "message {}", value)` for invariants
  (throws; an uncaught exception already exits non-zero with the message). Use `log_error` +
  `return 1` only for usage errors before any device work, as `test_tt_fabric.cpp` `main` does.
- Python: raise `AssertionError` with a specific message at the point of failure, as mux v2
  does. No `except Exception`. `try/finally` (or a context manager) is fine for cleanup that must
  happen, such as stopping `tracy-capture`, as `test_realtime_profiler.py` does.
- Catch a specific exception only when there is a specific recovery (for example, skipping a
  busy port while searching for a free one).

**C++ benchmark**

- Parse arguments with `test_args::get_command_option_and_remaining_args` and
  `test_args::validate_remaining_args` from `tests/tt_metal/test_utils/test_common.hpp`, not a
  hand-rolled loop. (`test_rw_buffer.cpp` uses these, but it also wraps them in a catch-all;
  copy the helpers, not the wrapper.)
- Log with `log_info(tt::LogTest, ...)`, not `std::cerr`.
- One job per function: `parse_args`, `count_cache_artifacts`, `open_cold`, `open_hot`,
  `write_results`. `main` reads top to bottom as the measurement sequence.
- `clang-format` (120 columns); one statement per line; no chained `if/else` on a single line.

**Python test**

- Follow `test_fabric_mux_v2_throughput.py`: module-level constants, small pure helpers,
  one `test_*` function per parametrized case, no module-level mutable state (the older mux
  test's global `SPEEDUPS` list is the pattern to avoid).
- `subprocess.run(..., check=False)` and assert on the return code with a message that points
  at the log file.
- `pathlib.Path` throughout; `csv.DictReader` / `DictWriter` for CSVs.
- Formatted by the repo's `black` config (120 columns); no semicolon-joined statements.

**Comments**

- Only where the code cannot say it: a constraint, an ordering requirement, a non-obvious
  reason. No comments restating the next line.

---

## 2. Current state

### What the branch adds

| Area | Files |
|---|---|
| Tracy zones (`FABRIC_BUILDER` category) | `tt_metal/impl/device/device_manager.cpp`, `tt_metal/impl/device/firmware/fabric_firmware_initializer.cpp`, `tt_metal/fabric/control_plane.cpp`, `tt_metal/tools/profiler/tracy_debug_categories.txt` |
| Profiled build | `.github/workflows/build-artifact.yaml` (`tracy-debug-categories` input) |
| CI wiring | `.github/workflows/tm-fabric-tests.yaml` (`build-artifact-fabric-builder`, `fabric-builder-perf-tests`), `.github/workflows/tm-fabric-builder-tests-perf-impl.yaml`, `tests/pipeline_reorg/fabric_builder_perf_tests.yaml` |
| Test | `tests/tt_metal/tt_fabric/fabric_init_perf/{fabric_init_benchmark.cpp, run.py, config.json, CMakeLists.txt}` |

Instrumented zones today:

- `DeviceManager::initialize_fabric_and_dispatch_fw`
- `FabricFirmwareInitializer::init`
- `ControlPlane::write_routing_tables_to_all_chips`
- `FabricFirmwareInitializer::compile_and_configure_fabric`

### What is broken

1. **Baselines missing.** `baselines/*.json` are deleted in the working tree, but `run.py` still
   loads `baselines/<hardware>.json`. Every run fails before measuring.
2. **Zone names do not match the code.** Tracy matches the exact string passed to
   `TTZoneScopedDN`. `config.json` uses `write_routing_tables_to_all_chips` and
   `compile_and_configure_fabric` (code has class-qualified names), and lists
   `FabricFirmwareInitializer::configure`, which has no zone.

Both go away with this plan: `run.py`, `config.json`, and `baselines/` are replaced.

### Why `run.py` is being replaced rather than trimmed

About 375 lines, mostly from the old multi-run / Galaxy / approval design: statistics over a
single sample, a zone parent graph with cycle checks, baseline approval and provenance,
CPU/binary fingerprinting, seven output files, `--mode report|enforce`, and a hand-rolled
CLI. None of that exists in the neighboring perf tests.

### `fabric_init_benchmark.cpp` (126 lines, audit mode already removed, uncommitted)

- Named "fabric init" throughout (directory, binary, CMake target, `FabricInitBenchmark::*`
  zone markers, CI paths). The rest of the branch (Tracy category, workflows, test list) says
  "fabric builder".
- Takes `--arch` / `--devices` and validates hardware itself, while hardcoding `FABRIC_2D`.
- `MeshDevice::create` is duplicated in two branches because each zone needs a literal name.
- Writes `schema_version`, `pid`, `open_elapsed_ns`, `device_ids`, `teardown_complete`,
  `build_type`, and saves the file three times.
- `#ifndef TRACY_ENABLE` branch is dead: CMake only builds the target when Tracy is on.
- All of `main` is one `try { ... } catch (const std::exception&)` that prints to `std::cerr`
  and returns 2. Checks are `throw std::runtime_error` instead of `TT_FATAL`, argument parsing
  is a hand-rolled loop, and several `if/else` chains are on single lines.

---

## 3. Target design

### Files

```
tests/tt_metal/tt_fabric/fabric_builder_perf/
  CMakeLists.txt
  fabric_builder_benchmark.cpp
  test_fabric_builder_perf.py
  fabric_builder_perf_golden_wormhole_b0_t3k.csv
```

### Benchmark binary

The binary measures and records facts. It makes no pass/fail decisions about hardware.

- CLI: `--output FILE --fabric-config NAME` (e.g. `FABRIC_2D`).
- Flow: require empty `TT_METAL_CACHE` -> wait for Tracy -> `SetFabricConfig` ->
  `open_cold()` / close -> `open_hot()` / close -> write JSON once -> reset fabric config.
- `open_cold()` and `open_hot()` are two small helpers, each with its literal zone name
  (`FabricBuilderBenchmark::cold` / `::hot`) around `MeshDevice::create`.
- Output JSON, shaped like the mux v2 benchmark output (a `context` block):

```json
{
  "context": {"arch": "wormhole_b0", "cluster_type": "t3k", "num_devices": 8, "fabric_config": "FABRIC_2D"},
  "phases": {
    "cold": {"artifacts_before": 0,   "artifacts_after": 412},
    "hot":  {"artifacts_before": 412, "artifacts_after": 412}
  }
}
```

`cluster_type` uses the same lowercase `enchantum::to_string(get_cluster_type())` string as
`ResultsManager::get_golden_csv_filename`, so golden names line up with the BW test.

### pytest module: `test_fabric_builder_perf.py`

Modeled on `test_fabric_mux_v2_throughput.py`.

```python
ZONES = [
    "DeviceManager::initialize_fabric_and_dispatch_fw",
    "FabricFirmwareInitializer::init",
    "ControlPlane::write_routing_tables_to_all_chips",
    "FabricFirmwareInitializer::compile_and_configure_fabric",
    "FabricFirmwareInitializer::configure",
]
CACHES = ["cold", "hot"]
DEFAULT_TOLERANCE_PERCENT = 10.0

@pytest.mark.parametrize("fabric_config", ["FABRIC_2D"])
def test_fabric_builder_perf(fabric_config):
    ...
```

Per case:

1. Fresh `TT_METAL_CACHE` under `<output_dir>/<fabric_config>/cache`.
2. Run the binary under `tracy-capture` (free port, `TRACY_PORT`), export with
   `tracy-csvexport -u` to `zones.csv`.
3. `extract`: for each cache marker, find each zone in `ZONES` exactly once inside it.
4. `validate`: cold `artifacts_before == 0`, hot `artifacts_before > 0`.
5. Golden path from `context.arch` + `context.cluster_type`.
6. If `FABRIC_BUILDER_PERF_UPDATE_GOLDEN=1`: write/refresh this fabric config's rows in the golden
   (keep existing per-row `tolerance_percent`, default for new rows) and pass.
7. Otherwise compare against the golden (below) and `raise AssertionError` on any error.
8. Write `summary_<arch>_<cluster>.csv` / `.txt` and print the table.

Environment:

| Variable | Purpose | Default |
|---|---|---|
| `FABRIC_BUILDER_PERF_UPDATE_GOLDEN` | `1` rewrites golden values instead of checking | unset |
| `FABRIC_BUILDER_PERF_OUTPUT_DIR` | where captures, CSVs, logs, summaries go | `$TT_METAL_HOME/generated/fabric_builder_perf` |

The test sets `CCACHE_DISABLE=1`, `TT_METAL_DEVICE_PROFILER=0`, and the fresh
`TT_METAL_CACHE` for the child process. It does not police the rest of the environment.

### Golden CSV

`fabric_builder_perf_golden_<arch>_<cluster>.csv`, next to the test:

```csv
fabric_config,cache,zone,golden_ms,tolerance_percent
FABRIC_2D,cold,DeviceManager::initialize_fabric_and_dispatch_fw,<ms>,10.0
FABRIC_2D,cold,FabricFirmwareInitializer::init,<ms>,10.0
...
FABRIC_2D,hot,FabricFirmwareInitializer::configure,<ms>,10.0
```

Key is `(fabric_config, cache, zone)`. `tolerance_percent` is per row, like the BW goldens.

### Comparison

| Situation | Status | Test result |
|---|---|---|
| `abs(measured / golden - 1) <= tolerance_percent / 100` | `pass` | pass |
| Outside tolerance (slower or faster) | `fail` | fail |
| Golden file missing | n/a | fail |
| Measured `(fabric_config, cache, zone)` has no golden row | `missing-golden` | fail |
| Golden row for this `fabric_config` not measured | `stale-golden` | fail |
| Zone missing from trace, cache check fails, binary fails | n/a | fail |

Two-sided like mux v2: a large speedup also fails, so goldens get refreshed instead of going
stale. No geomean: the zones are nested (bring-up contains init and configure), so one
regression moves several zones and a geomean would double-count it. Revisit when there are
several fabric configs (geomean over the top-level bring-up zone only).

### Extending

- New fabric config: add it to the `parametrize` list, run once with
  `FABRIC_BUILDER_PERF_UPDATE_GOLDEN=1`, commit the new rows.
- New hardware: add a `skus:` row in `fabric_builder_perf_tests.yaml`, run once with
  `FABRIC_BUILDER_PERF_UPDATE_GOLDEN=1` on that machine, commit the new golden file.
- New zone: add `TTZoneScopedDN(FABRIC_BUILDER, "...")` in the code and the name to `ZONES`,
  refresh goldens.

---

## 4. Tasks

Each task is one reviewable commit. Task 0 goes first so every later diff uses the final
names. Tasks 1 and 2 are independent.

Naming rule for this cleanup: anything this branch owns that says "fabric init" becomes
"fabric builder". Pre-existing uses of "fabric init" elsewhere in the repo (log messages,
`fabric_init.cpp`, comments) are unrelated and stay.

### Task 0: Rename fabric init -> fabric builder (no behavior change)

- **Files and renames:**

| Before | After |
|---|---|
| `tests/tt_metal/tt_fabric/fabric_init_perf/` | `tests/tt_metal/tt_fabric/fabric_builder_perf/` |
| `fabric_init_benchmark.cpp` | `fabric_builder_benchmark.cpp` |
| CMake target / binary `fabric_init_benchmark` | `fabric_builder_benchmark` |
| `add_subdirectory(fabric_init_perf)` in `tests/tt_metal/tt_fabric/CMakeLists.txt` | `add_subdirectory(fabric_builder_perf)` |
| Zone markers `FabricInitBenchmark::cold` / `::hot` | `FabricBuilderBenchmark::cold` / `::hot` |
| Error text `fabric-init benchmark failed` | `fabric-builder benchmark failed` |
| `run.py` references (binary path, marker names, report titles) | updated to match |
| `fabric_builder_perf_tests.yaml` `cmd` path and the `config.json` comment | updated to match |

- **Method:** `git mv` for the directory and source file so history follows; then edit
  references. `FABRIC_INIT_BUILD_TYPE` is not renamed; Task 2 deletes it.
- **Done when:** `rg -i 'fabric[_-]?init|FabricInit' tests/tt_metal/tt_fabric tests/pipeline_reorg .github`
  returns nothing; the target builds.

### Task 1: Instrument `FabricFirmwareInitializer::configure`

- **Files:** `tt_metal/impl/device/firmware/fabric_firmware_initializer.cpp`,
  `tt_metal/impl/device/device_manager.cpp`, `tt_metal/fabric/control_plane.cpp`
- **Change:** add `TTZoneScopedDN(FABRIC_BUILDER, "FabricFirmwareInitializer::configure");` at
  the top of `configure()`. Remove the whitespace-only lines left after the existing
  `TTZoneScopedDN` calls.
- **Done when:** a profiled build shows the zone inside
  `DeviceManager::initialize_fabric_and_dispatch_fw`, after `FabricFirmwareInitializer::init`.

### Task 2: Simplify `fabric_builder_benchmark.cpp`

- **Files:** `fabric_builder_benchmark.cpp`, `CMakeLists.txt`
- **Change:**
  - CLI becomes `--output FILE --fabric-config NAME`, parsed with the `test_args` helpers;
    parse the name into `FabricConfig`.
  - Remove the catch-all `try/catch` around `main`. Replace `throw std::runtime_error` checks
    with `TT_FATAL`; replace `std::cerr` with `log_error` / `log_info`.
  - Remove `--arch` / `--devices` and the in-binary hardware check.
  - Split into `parse_args`, `count_cache_artifacts`, `open_cold()` / `open_hot()`,
    `write_results`; remove the duplicated branch.
  - Emit the `context` + `phases` JSON from Section 3, written once at the end.
  - Remove `schema_version`, `pid`, `open_elapsed_ns`, `device_ids`, `teardown_complete`,
    `build_type` (and `FABRIC_INIT_BUILD_TYPE` in CMake), the dead `#ifndef TRACY_ENABLE`
    branch, and the ccache env check.
- **Done when:** builds with `ENABLE_TRACY`; running it by hand under `tracy-capture` on a T3K
  writes the JSON with `cluster_type: "t3k"`.
- **Note:** `run.py` is already non-functional (missing baselines) and is deleted in Task 3,
  so it is not updated here.

### Task 3: Add `test_fabric_builder_perf.py` (measure, report, update golden)

- **Files:** new `test_fabric_builder_perf.py`; delete `run.py`, `config.json`, `baselines/`.
- **Change:**
  - Path helpers (`get_tt_metal_home`, `get_benchmark_binary`, `get_output_dir`,
    `get_golden_path`, `should_update_golden`) mirroring mux v2.
  - `run_under_tracy` following `test_realtime_profiler.py` (free port, `TRACY_PORT`,
    tools from `PROFILER_BIN_DIR`, stdout to a log file).
  - `read_zones`, `extract`, `validate_cache_state`.
  - `write_golden_rows` for the update path; summary CSV/text output.
  - `@pytest.mark.parametrize("fabric_config", ["FABRIC_2D"])`.
  - Follows the Python guidelines in Section 1: no `except Exception`, failures are
    `AssertionError`s at the point of failure, `try/finally` only to stop `tracy-capture`.
- **Done when:** on a T3K, `FABRIC_BUILDER_PERF_UPDATE_GOLDEN=1 pytest -svv
  tests/tt_metal/tt_fabric/fabric_builder_perf/test_fabric_builder_perf.py` passes and writes
  `fabric_builder_perf_golden_wormhole_b0_t3k.csv` with 10 rows.

### Task 4: Golden validation

- **Files:** `test_fabric_builder_perf.py`
- **Change:** `read_golden_rows` and `validate_against_golden` with the statuses and
  two-sided tolerance from Section 3. Missing golden file is an `AssertionError`.
- **Done when:** without the env var, the test passes against a fresh golden; fails with
  `missing-golden` when a row is deleted; fails with `fail` when a `golden_ms` is edited past
  tolerance; fails when the golden file is absent.
- **Depends on:** Task 3.

### Task 5: Workflow and test-list update

- **Files:** `tests/pipeline_reorg/fabric_builder_perf_tests.yaml`,
  `.github/workflows/tm-fabric-builder-tests-perf-impl.yaml`
- **Change:**
  - `cmd: pytest -svv tests/tt_metal/tt_fabric/fabric_builder_perf/test_fabric_builder_perf.py`.
  - Header comment: describe the golden CSV and `FABRIC_BUILDER_PERF_UPDATE_GOLDEN`.
  - Resolve the `timeout` and `owner_id` TODOs.
  - Impl workflow: drop `FABRIC_BUILDER_BUILD_ARTIFACT` / `FABRIC_BUILDER_DOCKER_IMAGE`;
    publish `generated/fabric_builder_perf/summary_*.txt` to the job summary; upload
    `generated/fabric_builder_perf`.
- **Keep:** the separate test list and impl workflow. This job needs the
  `fabric-builder` profiled artifact, so it cannot share `fabric_perf_tests.yaml`, whose jobs use
  the normal artifact.
- **Depends on:** Task 3.

### Task 6: Commit the first T3K golden

- **Files:** `fabric_builder_perf_golden_wormhole_b0_t3k.csv`
- **Change:** run the update path on a `wh_llmbox_perf` machine with the profiled build. Run
  the check path 2-3 more times and set per-row `tolerance_percent` from observed variance.
- **Done when:** repeated check runs on that machine pass.
- **Depends on:** Task 4. Must land before the CI job is expected to pass.

### Task 7: CI validation run

- **Change:** dispatch `(TM-Fabric) Fabric Tests` on the branch with only the fabric builder
  perf job enabled (all HW SKU boxes, CPU-only, and T3K perf unchecked).
- **Done when:** the job passes and the summary shows all 10 rows.
- **Depends on:** Tasks 5 and 6.

---

## 5. Out of scope

- Additional hardware (Galaxy, Blackhole) and fabric configs beyond `FABRIC_2D`. The design
  supports both; add them later.
- Uploading results to the benchmark database (`upload_benchmarks`).
- Geomean check (see Comparison).
