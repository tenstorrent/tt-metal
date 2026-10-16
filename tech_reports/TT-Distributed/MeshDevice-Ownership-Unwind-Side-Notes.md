# MeshDevice ownership unwind: side notes

Things found while working on [MeshDevice-Ownership-Unwind.md](MeshDevice-Ownership-Unwind.md) that are not part of its plan. Most are pre-existing and left untouched. They are recorded for whoever works in this area next.

Each entry says how sure it is: **verified** means reproduced or read in code during the work; **observed** means seen once or inferred and not chased down.

## Existing bugs and test failures

### `DeviceInitFabric.RejectsSingleHost1ChipMesh` aborts `unit_tests_device` (verified)

- **Symptom:** on a single-chip Wormhole (n150) host, running the whole `unit_tests_device` binary aborts at this test (exit 134). Nothing after it runs. The test file is `tests/tt_metal/tt_metal/device/test_device_init_and_teardown.cpp`.
- **What happens:** the test opens a 1x1 mesh with `FABRIC_1D` and expects `TT_FATAL ... requires at least 2 participating chips`. That fires. Then the completion-queue reader thread throws a second `TT_FATAL` at `tt_metal/impl/event/dispatch.cpp:254` ("Unexpected values for event in completion queue, got cmd id CQ_DISPATCH_CMD_ILLEGAL"). It is thrown on a `std::thread`, so it ends in `std::terminate`.
- **Likely cause (not confirmed):** the mesh is torn down after a failed init while the reader thread is still polling a queue that never got its event.
- **Not caused by the ownership work.** It reproduces at `b455dd57a4e` and at the commit before step 0 (`68d5754a1e4`). A `tt-smi -r` reset did not change it.
- **Caveat:** earlier unfiltered runs of the same binary passed (50 tests). Something on the host changed in between (a reboot is the leading guess).
- **Workaround:** `unit_tests_device --gtest_filter=-DeviceInitFabric.RejectsSingleHost1ChipMesh`.

### `CommandList*` tests segfault in the device profiler on close (verified)

- **Symptom:** on this single-chip Wormhole host, `distributed_unit_tests --gtest_filter='CommandList*'` (for example `CommandListTest.BuildsIndependentSnapshotsAndReplaysBlockingAndNonBlocking` and `CommandListMultiCQTest.EnforcesTheCommandQueueUsedAtBuildTime`, in `tests/tt_metal/distributed/test_mesh_command_list.cpp`) dies with a segfault, so the rest of the binary does not run.
- **Where:** a null dereference in `DeviceProfiler::getTraceIdAndCount`, reached from `MeshDevice::close()` through `ReadMeshDeviceProfilerResults`. It has nothing to do with ownership.
- **Not caused by the ownership work.** It reproduces at the commit before step 0 (`68d5754a1e4`) and at the step-5 tip.
- **Workaround:** `--gtest_filter='-*Fabric*:CommandList*'` runs the other 206 tests.
- **Output hazard:** if two runs of the binary write to the same log file, their output interleaves and the crash looks like it happened twice.

### Opening a device twice gives an unhelpful error (verified)

Opening a device while another handle already holds it surfaces as a `TT_FATAL` about a garbage `context_id`, not as "device already open". Found when a new test opened a device next to an autouse `device` fixture. Fixing the error message would save time for the next person who writes such a test.

### `unit_mesh::aggregate` fails on `create_unit_meshes` results with a misleading message (verified)

`aggregate` needs the parent's allocator. `create_unit_meshes` results have a hidden parent that was never initialized, so `aggregate` throws a "remote-only MeshDevice" message that points at the wrong problem. Only unit meshes carved from an initialized mesh with `create_submeshes(MeshShape(1, 1))` work. The ownership doc (§4.1, "Scope") explains why the two kinds of "unit mesh group" are incompatible.

### FFT cache clearing in conftests is inert (verified)

The FFT conftests look up `clear_fft_device_caches` in a place where it does not exist. It is only exposed at `ttnn._ttnn.operations.experimental.clear_fft_device_caches`. The lookup finds nothing, so the caches are never cleared between tests. Step 2 re-keyed the stream caches on `md->id()`, so this no longer causes stale-device bugs there.

### `bluestein_host.hpp` caches are keyed on device address (observed)

`stockham_host.hpp` and the twiddle caches were re-keyed on the mesh id in step 2. The Bluestein caches were not touched and still key on the raw device address. A device reopened at the same address would hit a stale entry. Same class of bug as the one step 2 fixed. Not exercised by any test seen.

## Build and tooling

### The unity-off build (`-e`) is broken (verified)

A build with unity builds off fails in `test_mcast_host_api.cpp`: `emitted_metadata` is undefined. The code only compiles because unity builds (default ON, `TT_UNITY_BUILDS`) pull in a declaration from another file. Unrelated to this work. Build without `-e`.

### Rebuild everything after a layout change (verified)

After changing a class layout or a header included widely, run a full `ninja -k 0`, not just the test you care about. A stale `test_clean_init` binary caused a deterministic `TensixReleaseOwnership.ReleaseOwnershipWithSubprocess` failure that looked like a real bug.

### Python needs `cmake --install build` (verified)

`ninja` alone does not refresh what Python loads. After building, run `cmake --install build` (fast; it only copies) so `_ttnn.so` and the libraries are current. Otherwise Python smoke tests silently run old behavior. `ninja install` is much slower.

### `validate-metalium-api` pre-commit hook (verified)

A header under `tt_metal/api/tt-metalium/` (stable tier) must not include an experimental header (anything under `experimental/`). The hook fails the commit with the offending include. This is why `MeshDeviceOptions` is a stable header and `CreateMeshDeviceOptions` is an alias of it in `metal_env.hpp`. New public headers also need an entry in `TT_METAL_PUBLIC_API` in `tt_metal/sources.cmake`.

### The clang-format hook fails the first commit (verified)

When it reformats files, the commit fails and the files change on disk. Re-stage and commit again. If you tested before that, the binaries no longer match HEAD textually (usually harmless, but rebuild before final runs).

### Running the profiler tests locally (verified)

They fail for environmental reasons unless you have: the Python env active (`source python_env/bin/activate`), `ARCH_NAME=wormhole_b0`, `TT_METAL_DEVICE_PROFILER=1`, and the programming examples built.

## Test infrastructure

### Where a new FFT test can live (verified)

Tests under the FFT directory get an autouse `device` fixture from its `conftest.py`, which opens the device. A test that opens its own device must live outside it (it is in `tests/ttnn/unit_tests/base_functionality/test_fft_device_reopen.py`).

### What is not covered by local testing (verified)

A single n150 cannot run, and local runs skipped or did not build-and-run, these:
- H2D / D2H / D2D stream services (Blackhole Galaxy only; only compiled).
- Multi-chip mesh and submesh paths (all `MeshDevice2x4`, `MeshSocketTest*`, `DispatchContextFixture` cases skip).
- H2D / D2H socket tests, DRAM-receive socket, tensor prefetcher.
- The streaming profiler.

CI is the only coverage for these.

## CI notes

- **`gh` version:** the host has gh 2.4.0. `gh run list` has no `--branch` flag; use the REST API (`gh api repos/<owner>/<repo>/actions/runs?branch=...`).
- **Standalone `single-card-profiler-tests.yaml`:** ends in `startup_failure` for everyone, not just this branch. Use the "(Profiler) Choose your pipeline" workflow instead.
- **Runner labels:** `.github/sku_config.yaml` lists them. Blackhole Galaxy is `arch-blackhole, pipeline-functional, topology-6u, in-service`.
- **Stream-service CI dispatch:** the custom-test dispatch command used for the Blackhole Galaxy stream services was a best guess. Check that run actually ran the stream-service tests before trusting a green result.
- **Helper script:** `scratch/dispatch_ci.sh <workflow> [-f key=value ...]` dispatches on branch `ruizhang/meshdevice-ownership` and prints the run URL. The branch name is hardcoded. It is untracked.

## Documentation hazard

The IDE markdown preview (`markdown-rich-preview`) stops rendering everything after an indented continuation paragraph that follows a blank line after a nested list. In the ownership doc this made the preview blank from `## Summary`. Keep continuation text unindented or part of the list item. Avoid `[[...]]` too.
