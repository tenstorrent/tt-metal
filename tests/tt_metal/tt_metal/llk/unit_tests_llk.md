# LLK Unit Tests

`unit_tests_llk` is a C++ gtest binary that exercises the LLK (Low-Level Kernel)
compute path on real Tenstorrent hardware. It is the primary way to validate
compute-kernel changes locally and is the same binary that runs in the merge gate
and in the L2 nightly pipeline.

This document is a tutorial for developers writing or debugging LLK tests.

- **Source:** `tests/tt_metal/tt_metal/llk/test_*.cpp`
- **Build target:** `unit_tests_llk` (defined in this directory's `CMakeLists.txt`)
- **Built artifact:** `build/test/tt_metal/unit_tests_llk`
- **Fixtures:** `llk_device_fixture.hpp` (LLK-specific) plus shared fixtures in
  `tests/tt_metal/tt_metal/common/`

---

## Quick start

```bash
# Incremental build of just the LLK test binary
cmake --build build --target unit_tests_llk -- -j 8

# Run the full FD (fast dispatch) suite
unset TT_METAL_SLOW_DISPATCH_MODE
ARCH_NAME=<arch> TT_METAL_HOME=$(pwd) LD_LIBRARY_PATH=$(pwd)/build/lib \
  ./build/test/tt_metal/unit_tests_llk

# Run the full SD (slow dispatch) suite
TT_METAL_SLOW_DISPATCH_MODE=1 \
ARCH_NAME=<arch> TT_METAL_HOME=$(pwd) LD_LIBRARY_PATH=$(pwd)/build/lib \
  ./build/test/tt_metal/unit_tests_llk
```

`ARCH_NAME` matches the card you're running on (for example `wormhole_b0`,
`blackhole`, or `quasar`).

---

## Build

For a clean full build (sets up CMake, builds metal, builds tests):

```bash
./build_metal.sh --build-tests
```

For day-to-day work after editing test files, an incremental build of just the
test binary is much faster:

```bash
cmake --build build --target unit_tests_llk -- -j 8
```

---

## Dispatch modes

The same binary supports both dispatch modes; the `TT_METAL_SLOW_DISPATCH_MODE`
environment variable toggles the underlying device behavior.

| Mode | Env | What runs / what skips |
|---|---|---|
| **Fast dispatch** (FD) | `unset TT_METAL_SLOW_DISPATCH_MODE` | Dispatch-agnostic tests. SD-only fixtures auto-skip; arch-specific fixtures auto-skip on the wrong arch. |
| **Slow dispatch** (SD) | `TT_METAL_SLOW_DISPATCH_MODE=1` | Dispatch-agnostic tests plus the SD-only set. FD-only fixtures auto-skip; arch-specific fixtures auto-skip on the wrong arch. |

Each test opts into a mode via its fixture choice — see [Fixtures](#fixtures).

---

## Useful invocations

### Filter to a single test or fixture

```bash
./build/test/tt_metal/unit_tests_llk --gtest_filter='LLKMeshDeviceFixture.<TestName>'
./build/test/tt_metal/unit_tests_llk --gtest_filter='LLKBlackholeSingleCardFixture.*'
./build/test/tt_metal/unit_tests_llk --gtest_filter='-*Quasar*:*StochasticRounding*'  # exclude
```

### List tests without running

```bash
./build/test/tt_metal/unit_tests_llk --gtest_list_tests
```

### Shard a run across multiple processes

`gtest`'s built-in sharding splits the test set across N runners. The merge gate
uses this to fan out FD runs:

```bash
GTEST_TOTAL_SHARDS=4 GTEST_SHARD_INDEX=0 \
  ./build/test/tt_metal/unit_tests_llk
```

### Optional environment variables

| Env | Purpose |
|---|---|
| `TT_METAL_LLK_ASSERTS=1` | Compile and run with LLK device-side asserts enabled. The cold first run rebuilds the JIT cache (can take several minutes); warm runs are unchanged. |
| `LOGURU_LEVEL=INFO\|WARNING\|ERROR` | Verbosity of metal's runtime logs. |
| `GTEST_OUTPUT=xml:./reports/` | Emit JUnit-style XML (what CI consumes). |
| `GTEST_TOTAL_SHARDS=N`, `GTEST_SHARD_INDEX=I` | Run only ~1/N of the test cases. |

---

## Fixtures

All LLK fixtures live in `llk_device_fixture.hpp`. They share a single
`MeshDevice` per **test suite** (gtest's `SetUpTestSuite` / `TearDownTestSuite`),
so per-test setup is bounded to copying a few handles.

| Fixture | Use when… | FD | SD |
|---|---|:---:|:---:|
| `LLKMeshDeviceFixture` | The test works under both dispatch modes. (Most tests.) | yes | yes |
| `LLKMeshDeviceSingleCardFixture` | You need the canonical single-card mesh; works under both modes. | yes | yes |
| `LLKBlackholeSingleCardFixture` | Blackhole-only test. Auto-`GTEST_SKIP` on other archs. | yes (BH HW) | yes (BH HW) |
| `LLKQuasarMeshDeviceSingleCardFixture` | Quasar-only test. Auto-`GTEST_SKIP` on other archs. | yes (Quasar HW) | yes (Quasar HW) |
| `LLKMeshDeviceFixtureSlowDispatchOnly` | The test is only correct under SD (e.g. uses async write/read patterns broken under FD). | skip | yes |
| `UnitMeshCQFixture` *(from `tests/tt_metal/tt_metal/common/`)* | The test is only meaningful under FD (uses host command-queue APIs directly). | yes | skip |

> **When in doubt, start with `LLKMeshDeviceFixture`.** Only switch to a more
> restrictive fixture if you discover the test legitimately can't run under one
> of the modes — and add a code comment explaining why.
>
> The table describes fixture capability. CI can still schedule a capable fixture
> in only one dispatch mode to keep coverage balanced. Today merge gate excludes
> `LLKBlackholeSingleCardFixture` suites from the Blackhole FD shards and runs
> them in the Blackhole SD slice instead.

---

## Adding a new test

1. **Create or edit a `test_*.cpp`** in this directory. Name files by topic
   (e.g. `test_my_new_thing.cpp`).
2. **Pick a fixture** from the table above.
3. **Write the test** using the standard `EnqueueMeshWorkload` + `Finish`
   pattern — it works in both dispatch modes:

   ```cpp
   #include "llk_device_fixture.hpp"

   namespace tt::tt_metal {

   TEST_F(LLKMeshDeviceFixture, MyNewThing) {
       auto& mesh_device = *devices_[0];
       auto& cq = mesh_device.mesh_command_queue();

       Program program = CreateProgram();
       // ... CreateBuffer, CreateCircularBuffer, CreateKernel, SetRuntimeArgs ...

       distributed::MeshWorkload workload;
       auto zero_coord = distributed::MeshCoordinate(0, 0);
       auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
       workload.add_program(device_range, std::move(program));

       distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
       distributed::Finish(cq);

       // ... read back outputs and EXPECT_*
   }

   }  // namespace tt::tt_metal
   ```

   See the existing `test_*.cpp` files in this directory for full working
   examples.

4. **Register the new source file** in `sources.cmake`:

   ```cmake
   set(UNIT_TESTS_LLK_SRC
       ...
       test_my_new_thing.cpp   # add this
   )
   ```

5. **Build and run locally** in both modes:

   ```bash
   cmake --build build --target unit_tests_llk -- -j 8

   unset TT_METAL_SLOW_DISPATCH_MODE
   ARCH_NAME=<arch> TT_METAL_HOME=$(pwd) LD_LIBRARY_PATH=$(pwd)/build/lib \
     ./build/test/tt_metal/unit_tests_llk --gtest_filter='*MyNewThing*'

   TT_METAL_SLOW_DISPATCH_MODE=1 \
   ARCH_NAME=<arch> TT_METAL_HOME=$(pwd) LD_LIBRARY_PATH=$(pwd)/build/lib \
     ./build/test/tt_metal/unit_tests_llk --gtest_filter='*MyNewThing*'
   ```

6. **Push and enter the merge queue** — the merge gate detects changes under
   `tests/tt_metal/tt_metal/llk/**` and runs the test binary on the supported
   architectures in both dispatch modes automatically. The PR gate does **not**
   run the LLK unit tests (by design, to avoid re-running the same suite on every
   PR commit); they run once per queued merge.

### What if my test only works under slow dispatch?

If the test legitimately doesn't work under FD (and you can't fix the test),
use the SD-only fixture:

```cpp
TEST_F(LLKMeshDeviceFixtureSlowDispatchOnly, MySdOnlyThing) {
    // ...
}
```

Document the reason in a code comment. Under FD the test prints
`Skipping: test requires slow dispatch (TT_METAL_SLOW_DISPATCH_MODE=1)`. The
merge-gate SD jobs (and the L2 nightly) will still cover it.

### What if my test is architecture-specific?

Use `LLKBlackholeSingleCardFixture` for a Blackhole-only test, or
`LLKQuasarMeshDeviceSingleCardFixture` for a Quasar-only test. They auto-skip
on the wrong architecture and run on the right one wherever it is available.

---

## CI integration

The same binary runs in merge gate and L2 nightly with different filters:

| Job | Workflow | Trigger | Dispatch | Filter |
|---|---|---|---|---|
| `llk-fd-unit-tests-wormhole` | `merge-gate.yaml` | LLK changes when entering merge queue (or push to `main`) | FD | default `*` filter |
| `llk-fd-unit-tests-blackhole` | `merge-gate.yaml` | LLK changes when entering merge queue (or push to `main`) | FD | excludes `LLKBlackholeSingleCardFixture.*` and `*MulReduceScalarTest*`; those run in BH SD |
| `llk-sd-unit-tests-wormhole` | `merge-gate.yaml` | LLK changes when entering merge queue (or push to `main`) | SD | `LLKMeshDeviceFixtureSlowDispatchOnly.*` plus Quasar fixture filters (currently skip on WH) |
| `llk-sd-unit-tests-blackhole` | `merge-gate.yaml` | LLK changes when entering merge queue (or push to `main`) | SD | `LLKMeshDeviceFixtureSlowDispatchOnly.*`, `LLKBlackholeSingleCardFixture.*`, `*MulReduceScalarTest*`, plus Quasar fixture filters (currently skip on BH) |
| `llk-sd-unit-tests` | `tt-metal-l2-nightly.yaml` | nightly cron / `workflow_dispatch` | SD | default `*` filter |

The merge-gate FD jobs are sharded across multiple runners using
`GTEST_TOTAL_SHARDS` / `GTEST_SHARD_INDEX` and run with
`TT_METAL_LLK_ASSERTS=1`. The L2 nightly runs the full SD set across the
broader hardware pool.

The PR gate intentionally does **not** run the LLK unit tests — the suite runs
once per queued merge (via merge gate) plus nightly, rather than on every PR
commit. If you need PR-gate feedback on an LLK change, run the relevant
invocation from [Quick start](#quick-start) locally before entering the merge
queue.

Trigger gating uses `find-changed-files.sh`'s `llk-unit-tests-changed` flag,
which fires on any change under `tests/tt_metal/tt_metal/llk/**`. Changes to
the LLK engine itself (under `tt_metal/tt-llk/**`), SFPI version pins, and LLK
CI files also trigger the same jobs via the per-arch/common/SFPI/CI flags.

---

## Common gotchas

- **JIT cache at `~/.cache/tt-metal-cache/`.** Cold runs are slow; subsequent
  runs with the same kernel set are fast. Clearing the cache forces a full
  recompile on the next run.
- **`unit_tests_llk` requires real Tenstorrent hardware.** There is no
  simulator path — you'll see a UMD error if `/dev/tenstorrent` is missing.
- **Don't run `tt-smi` and `unit_tests_llk` simultaneously.** They contend for
  the device. Stop any background `tt-smi` first.

---

## Where to look when something breaks

| Symptom | Probable cause / fix |
|---|---|
| Test silently skips with *"Skipping: test requires slow dispatch"* | Fixture is `LLKMeshDeviceFixtureSlowDispatchOnly`. Either set `TT_METAL_SLOW_DISPATCH_MODE=1` or move the test to `LLKMeshDeviceFixture`. |
| Test silently skips with *"This suite can only be run with fast dispatch …"* | Fixture is (or inherits) `UnitMeshCQFixture`. Either unset `TT_METAL_SLOW_DISPATCH_MODE` or move the test to `LLKMeshDeviceFixture`. |
| Test silently skips with *"Not a Quasar device"* / on a non-Blackhole machine | Fixture is `LLKQuasar*` / `LLKBlackholeSingleCardFixture`. By design — runs only on matching hardware. |
| `Mixing fast and slow dispatch is prohibited!` | Test is calling `tt_metal::detail::LaunchProgram(...)` (slow-dispatch only) inside a process running FD. Convert to `EnqueueMeshWorkload` + `Finish`. |
| `device->close()` issues at shutdown | Don't add per-test device-open/close logic; rely on the suite-shared fixtures in `llk_device_fixture.hpp`. |
| Merge-gate `llk-fd-unit-tests-*` not triggered | Check `find-changed-files.sh` — the `llk-unit-tests-changed` case must match your file path (currently `tests/tt_metal/tt_metal/llk/**`). Note the merge gate (not the PR gate) runs these. |
