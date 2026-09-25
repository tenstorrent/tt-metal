# Fabric initialization performance CI — initial implementation

## Status and scope

This package implements the agreed design: full Wormhole Galaxy and Blackhole Galaxy, FABRIC_2D, and **cold followed by hot initialization in the same process**. It is generated source for review and integration—not an already merged change or a hardware-qualified implementation.

Validated locally: 42 Python tests (31 extraction/comparison/metadata tests, 3 simulated CLI tests, and 8 workflow-integration tests), plus Python/JSON/YAML syntax. The simulated CLI tests use fake tools; they do **not** validate Tracy's real capture protocol or Metal hardware behavior.

Not yet validated: compiling/linking the C++ benchmark against your checkout; running real capture/export tools; GitHub Actions semantic validation; applying patches to your exact checkout; full teardown/reinitialization on either Galaxy. No performance numbers have been fabricated. Both baseline files intentionally contain null values.

The source/API reference is indexed tt-metal main inspected during generation, not a pinned checkout. Review the integration diff, then perform the single-pair smoke test below before enabling this in scheduled CI.

## What is included

- `tests/tt_metal/tt_fabric/fabric_init_perf/fabric_init_benchmark.cpp`: a standalone Metal test executable. No test fixture opens a device before the cold phase. It opens the full system mesh, validates architecture/device count, closes it, and opens/closes the same physical devices again without exiting.
- `run.py`: isolated cache handling, separate cache audit, repeated captures, host-zone CSV parsing, zone hierarchy validation, measurements, JUnit/Markdown/CSV/JSON reports, and regression comparison.
- `config.json`: hardware definitions and the five exact expected zone names.
- `baselines/wh_galaxy_perf.json` and `baselines/bh_galaxy_perf.json`: unmeasured templates.
- `fabric-init-perf-impl.yaml`: reusable workflow with two hardware jobs and a small matrix-selection job. Runner labels are looked up from the repository's existing SKU configuration.
- `fabric-init-perf.yaml`: standalone manual entry point for smoke testing; builds once and invokes the hardware workflow.
- `integration/apply_integration.py`: prints a reviewable diff by default; `--apply` registers the category/target and connects the reusable workflow under the existing fabric-perf workflow. It is intended for the inspected workflow layout and fails on missing or ambiguous anchors.
- Python tests and `VALIDATION.json`.

## Logical layout

```text
fabric-perf-tests
+-- existing microbenchmark jobs
+-- Fabric Init Tests
    +-- Select hardware
    +-- Wormhole Galaxy -- cold + hot
    +-- Blackhole Galaxy -- cold + hot
```

GitHub controls the exact visual expansion of nested reusable workflows. Zone results are in each job's report, not individual GitHub jobs. Within each hardware job:

```text
one cache-audit process: cold open -> close -> hot open -> close
three timed processes by default, each:
    new empty cache + start capture
    cold open -> close -> hot open -> close
    finish capture -> export -> validate
aggregate cold/hot separately -> compare -> upload
```

The audit has JIT telemetry enabled; timed runs explicitly disable it. A fresh cache is used for the audit and each measured pair. Every individual pair uses the same process for cold and hot. The host and operating system are not rebooted or cache-flushed between pairs.

## Integrating the package

Extract the ZIP outside your checkout. From the tt-metal checkout root, set the package path and copy only the new files:

```bash
PACKAGE=/absolute/path/to/fabric-init-ci-package
cp -a "$PACKAGE/.github/." .github/
cp -a "$PACKAGE/tests/." tests/

# PyYAML is required by the integration helper, not by the measurement runner.
# Use your existing development environment if PyYAML is already installed.
python3 "$PACKAGE/integration/apply_integration.py"
# Review the printed diff. No existing repository file was edited by this command.
python3 "$PACKAGE/integration/apply_integration.py" --apply
git diff
```

The helper makes these existing-file changes:

1. Registers `fabric-init` in `tt_metal/tools/profiler/tracy_debug_categories.txt`.
2. Adds `add_subdirectory(fabric_init_perf)` to the fabric test CMake file.
3. Adds a `tracy-debug-categories` reusable-build input, defaulting to `off`, forwards it to `build_metal.sh`, and distinguishes the instrumented tarball with a `_fabric-init` suffix. The first implementation accepts only `off` and `fabric-init`.
4. Adds manual parent-workflow inputs for opting into init tests and selecting `report` versus `enforce`. Existing push/scheduled behavior does not automatically opt into init tests.
5. Adds the nested reusable workflow call to `tm-fabric-tests-perf-impl.yaml`.

The helper does not edit or duplicate the five zones you already added. Verify the following names match your `TTZoneScopedDN(FABRIC_INIT, "...")` strings, or edit `config.json` to match. Names are part of the baseline identity:

```text
DeviceManager::initialize_fabric_and_dispatch_fw
FabricFirmwareInitializer::init
write_routing_tables_to_all_chips
compile_and_configure_fabric
FabricFirmwareInitializer::configure
```

Expected hierarchy: one outer zone, direct `init` and `configure` children, and the two sequential subzones inside `init`. The parser requires exactly one occurrence per phase, on the phase-marker thread. Other existing Tracy zones are allowed. Missing/duplicated zones fail rather than being averaged away.

The benchmark adds two phase marker zones, `FabricInitBenchmark::cold` and `FabricInitBenchmark::hot`, also under `FABRIC_INIT`. They contain only mesh-open calls; teardown is outside the markers. Your outer bring-up zone remains the reported headline metric, not the larger mesh-open marker.

## Build and first hardware smoke test

Use your existing build tree and usual options; no clean rebuild or environment recreation is required solely for these edits. Enable the category and Metal tests:

```bash
./build_metal.sh --build-type Release --build-metal-tests --build-perf-debug fabric-init
```

If your workflow builds on `/scratch` and deploys to `/data`, keep that workflow: deploy the affected Metal library, this new benchmark binary, and matching Tracy tools; preserve source edits and ensure the binary loads the matching library. Do not run against a stale production library.

Expected tools under the build directory:

```text
build/test/tt_metal/tt_fabric/fabric_init_benchmark
build/tools/profiler/bin/tracy-capture
build/tools/profiler/bin/tracy-csvexport
```

On an exclusively allocated Wormhole Galaxy, with no other Tracy-instrumented process occupying port 8086:

```bash
python3 tests/tt_metal/tt_fabric/fabric_init_perf/run.py \
  --hardware wh_galaxy_perf --mode report --pairs 1 \
  --output generated/fabric-init/wh-smoke
```

On Blackhole Galaxy, substitute `--hardware bh_galaxy_perf` and a fresh output directory. For the normal initial run, use `--pairs 3` (the default). Output directories must be empty; the runner refuses to overwrite results.

From GitHub Actions, use the standalone **(TM-Fabric) Fabric Init Perf** workflow first, in `report` mode. Once validated, use the existing fabric workflow with `run-fabric-init-perf-tests=true`; this selects the instrumented build and nested group. Both platforms use full 32-device system meshes. The exact 2D shape is observed, checked for consistency, and recorded in the baseline identity.

If your checkout's source/API or workflow structure differs, adapt the integration first. Do not bypass missing-zone, cache-audit, or teardown validation merely to make a smoke test pass.

## Cache and lifecycle contract

- `TT_METAL_CACHE` is a new empty job-owned directory before the process starts. The code also rejects preexisting compiled artifacts at cold-phase entry.
- `TT_METAL_CCACHE_KERNEL_SUPPORT` is **removed**, not set to `0`: the wrapper is presence-controlled. `CCACHE_DISABLE=1` is set, and remote compiler-cache configuration is removed from the child environment.
- Existing shared/user cache directories are never recursively cleared. Job-owned cache directories are retained for diagnostics but excluded from CI uploads.
- The cold run is allowed to reuse work compiled earlier in that same run. Cold means no preexisting compiled artifacts, not “zero hits anywhere during initialization.”
- Hot preserves Metal cache files and retained host state, while closing and destroying the first mesh. The second open must run all five zones again on the same physical devices.
- The audit requires cold compilation work, reduced hot compilation work, and either explicit cache-hit/dedup evidence or elimination of compilation work in hot. These are process-wide JIT counters, not fabric-only attribution. The audit is a separate process pair under identical cache controls, not a claim that per-zone cache hits were measured in each timed run.
- Timed runs disable extra JIT telemetry, device profiling, and streaming device profiling. Ordinary unrelated host zones already present in the build may remain active.
- Invalid hardware, partial visibility overrides, debug environments, no cache artifacts, failed close, incomplete captures, wrong zone counts, or invalid hierarchy produce a measurement failure (exit 2), including in report mode.
- The audit and extra repetitions can change OS caches and thermal state. This is a compilation-cache-cold benchmark, not a cold-boot or cold-OS-cache benchmark.

## Reports and baselines

Each hardware job produces:

```text
provenance.json                 commit/build artifact/binary identity and run setup
measurements.json              per-pair inclusive zone durations and environment identity
comparison.json / timings.csv  per-phase medians, spread, baseline, limits, and status
summary.md / junit.xml          CI-readable results
baseline-candidate.json         measured candidate; NOT automatically approved/installed
audit/metadata.json             cache evidence and cold/hot lifecycle information
pair-00/capture.tracy           one trace containing both phases
pair-00/zones.csv               unwrapped host zones
pair-00/metadata.json           one PID, physical device IDs, and completed teardowns
...                            additional pairs
```

Baseline identity includes hardware SKU, architecture, device count, observed mesh shape, FABRIC_2D, Release build, host CPU model/affinity size, zone naming contract, telemetry mode, and cache policy. Different host CPU types or allocations require calibration rather than silent comparison. The build SHA/image is recorded as provenance rather than included in strict identity, since commits are expected to change.

For each zone and phase, the implementation compares the median of independently repeated pairs. The permitted slowdown is the larger of:

- baseline duration multiplied by the approved relative percentage; and
- the approved absolute allowance in milliseconds.

A regression must exceed both allowances; equality passes. Improvements pass. Parent and child durations are not summed. Cold and hot never share a baseline or get averaged together.

Start in `report` mode. After several representative runs on stable hardware:

1. Inspect traces, cache audit, and sample spread.
2. Copy the appropriate `baseline-candidate.json` into the corresponding baseline file.
3. Choose `max_regression_percent` and `min_regression_ms` for every phase/zone using measured variability.
4. Fill `provenance.approved_by`, retain the reference commit, and review the baseline change.
5. Switch to `enforce` with at least three pairs.

No tolerance values are invented in the templates. Enforcement rejects unmeasured/unapproved baselines and incompatible identities. Report mode still fails measurement-integrity errors; valid unbaselined measurements are marked UNBASELINED, not PASS. Existing populated baselines can report regressions without failing the job until enforcement is enabled.

## Validation and first-run acceptance

Local tests:

```bash
(cd tests/tt_metal/tt_fabric/fabric_init_perf && python3 -m unittest -v test_runner.py test_cli_simulated.py)
(cd "$PACKAGE/integration" && python3 -m unittest -v test_integration.py)
```

Before rollout, confirm on **both** hardware configurations:

- The C++ target compiles/links, including access to the internal JIT telemetry API on your checkout.
- The exact full-system mesh open/close path runs successfully twice in one process.
- The five user-added zone names match and appear once in each phase.
- Real `tracy-capture` connects before cold initialization and exits/saves a readable trace.
- Real `tracy-csvexport -u` matches the expected host-zone CSV format.
- Device and cache audit evidence is valid; no hardware reset is required between phases.
- Failed measurements upload diagnostics, and real GitHub workflow validation accepts nested calls/permissions on your branch.

The integration tests exercise representative YAML fixtures, including the repository's array-based build invocation. They do not substitute for applying the patch to a real checkout. GitHub runner scheduling, firmware state, host load, and physical connectivity are not testable in this environment.

## Source references used for interface alignment

- [Current fabric perf workflow](https://github.com/tenstorrent/tt-metal/blob/main/.github/workflows/tm-fabric-tests-perf-impl.yaml)
- [Parent fabric workflow](https://github.com/tenstorrent/tt-metal/blob/main/.github/workflows/tm-fabric-tests.yaml)
- [Build workflow](https://github.com/tenstorrent/tt-metal/blob/main/.github/workflows/build-artifact.yaml)
- [Runner SKU configuration](https://github.com/tenstorrent/tt-metal/blob/main/.github/sku_config.yaml)
- [Setup action](https://github.com/tenstorrent/tt-metal/blob/main/.github/actions/setup-job/action.yml)
- [Fabric test CMake](https://github.com/tenstorrent/tt-metal/blob/main/tests/tt_metal/tt_fabric/CMakeLists.txt)
- [MeshDevice API](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/api/tt-metalium/mesh_device.hpp)
- [Cache telemetry API](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/jit_build/build_cache_telemetry.hpp) and [implementation](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/jit_build/build_cache_telemetry.cpp)
- [Tracy CSV exporter](https://github.com/tenstorrent/tracy/blob/master/csvexport/src/csvexport.cpp)
