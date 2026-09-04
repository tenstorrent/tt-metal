# ltx-rt Historical Fix Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Forward-port only the audited production fixes, prove them on the local Blackhole LTX server, and push the tested commit to `origin/ltx-rt`.

**Architecture:** Keep current `ltx-rt` implementations and add four narrow deltas: shared model paths, Ring-SDPA override routing, complete AGMM cache identity, and prewarm interfaces/tests lost in a merge. Tests isolate configuration selection and cache identity on the host, then exercise the affected compiled paths through the broker and live server.

**Tech Stack:** C++20, TT-Metal/TTNN, nanobind, Python/pytest, Bash, CMake through `build_metal.sh`, tt-device-mcp, FastAPI ltx-server.

**Spec:** `docs/superpowers/specs/2026-09-03-ltx-rt-historical-fixes.md`

## Global Constraints

- Base all work on `origin/ltx-rt` at `3a41478432f`.
- Work only in `/home/smarton/.claude/worktrees/ltx-rt-fixes-2026-09-03` until deployment.
- Do not port the gate-merge stack, unvalidated tuning knobs, or historical diagnostic instrumentation.
- Do not key `fused_ternary_scalar`; current cache-hit code rewrites it as a runtime argument.
- Device work uses `user-tt-device-mcp`; query the queue first and do not set `timeout_sec`.
- Build through `/home/smarton/tt-workflows/scripts/build.sh`; enable ccache with `build_metal.sh -c`.
- Keep host compilation outside broker reservations.
- Use the three-stage prewarm flow for a cold full-pipeline build key.
- Do not force-push, reset destructively, kill foreign jobs, force-reset devices, or weaken the frozen goal check.
- Keep the model-path repair, SDPA fix, AGMM fix, and prewarm repair in separate commits.

---

### Task 1: Configure the isolated build and preserve shared model paths

**Files:**
- Create outside git: `/home/smarton/ltx-rt-integration/ttw.toml`
- Modify: `models/tt_dit/tests/models/ltx/audio_compile_bench.sh:29-30`

**Interfaces:**
- Produces: a stamped C++ build command for all later tasks.
- Produces: world-readable shared model defaults independent of a former employee's home.

- [ ] **Step 1: Create the build configuration**

Write:

```toml
cpp_build_cmd = "cd /home/smarton/.claude/worktrees/ltx-rt-fixes-2026-09-03 && ./build_metal.sh -c --build-tests"
data_build_cmd = "cd /home/smarton/.claude/worktrees/ltx-rt-fixes-2026-09-03 && bash -n tt_metal/tools/kernel_prewarm/prewarm_and_submit.sh && /home/smarton/tt-metal/python_env/bin/python -m compileall -q models/tt_dit"
tmp_dir = "tmp"
```

- [ ] **Step 2: Apply the path repair**

Use these exact defaults:

```bash
export LTX_CHECKPOINT="${LTX_CHECKPOINT:-/home/models/ltx-2.3/ltx-2.3-22b-distilled-1.1.safetensors}"
export GEMMA_PATH="${GEMMA_PATH:-/home/models/gemma-3-12b-it-qat-q4_0-unquantized/}"
```

- [ ] **Step 3: Verify paths and script syntax**

Run:

```bash
bash -n models/tt_dit/tests/models/ltx/audio_compile_bench.sh
sha256sum /home/models/ltx-2.3/ltx-2.3-22b-distilled-1.1.safetensors
test -r /home/models/gemma-3-12b-it-qat-q4_0-unquantized/config.json
```

Expected LTX digest:

```text
b33b7fe4bbfe084f484be4aaf90b0f1d95dca20d403ac4c0e037eb8c4f0af7cc
```

- [ ] **Step 4: Commit**

Commit only `audio_compile_bench.sh` with subject:

```text
ltx/test: use shared model paths in audio compile bench
```

### Task 2: Route the Ring-SDPA override into per-N configurations

**Files:**
- Modify: `models/tt_dit/models/transformers/ltx/attention_ltx.py:37-203`
- Modify: `models/tt_dit/tests/models/ltx/test_transformer_ltx.py`

**Interfaces:**
- Produces: `LTXAttention.resolve_ring_sdpa_chunks(mesh_key, override)` returning `(fallback, per_n)`.
- Preserves: unset defaults `(96,256)` for `N=9728` and `(192,512)` for `N=38912`.

- [ ] **Step 1: Write the failing no-device test**

Add:

```python
def test_ring_sdpa_chunk_override_reaches_per_n_configs():
    mesh_key = (True, 8, 4)
    fallback, per_n = attention_ltx.LTXAttention.resolve_ring_sdpa_chunks(mesh_key, None)
    assert fallback == (128, 512)
    assert per_n == {9728: (96, 256), 38912: (192, 512)}

    fallback, per_n = attention_ltx.LTXAttention.resolve_ring_sdpa_chunks(mesh_key, "128,256")
    assert fallback == (128, 256)
    assert per_n == {9728: (128, 256), 38912: (128, 256)}
    assert per_n.get(12345, fallback) == (128, 256)
```

- [ ] **Step 2: Run the test and verify RED**

Run with the worktree source first on `PYTHONPATH`:

```bash
PYTHONPATH="$PWD/ttnn:$PWD" /home/smarton/tt-metal/python_env/bin/python -m pytest \
  models/tt_dit/tests/models/ltx/test_transformer_ltx.py \
  -k ring_sdpa_chunk_override_reaches_per_n_configs -q
```

Expected: fail because `resolve_ring_sdpa_chunks` does not exist.

- [ ] **Step 3: Implement the pure resolver**

Add to `LTXAttention`:

```python
@classmethod
def resolve_ring_sdpa_chunks(cls, mesh_key, override):
    fallback = cls.sdpa_chunk_size_map.get(mesh_key, cls.default_sdpa_chunk_size)
    selected = None
    if override:
        selected = tuple(int(value) for value in override.split(","))
        if len(selected) != 2:
            raise ValueError("LTX_SDPA_RING_CHUNK must be q,k")
        fallback = selected
    per_n = {
        n: selected if selected is not None else chunk
        for (blackhole, sp, tp, n), chunk in cls.ring_sdpa_chunk_by_n.items()
        if (blackhole, sp, tp) == mesh_key
    }
    return fallback, per_n
```

Replace direct map parsing in `__init__` with:

```python
ring_sdpa_chunk_size, ring_chunks_by_n = self.resolve_ring_sdpa_chunks(
    mesh_key, os.environ.get("LTX_SDPA_RING_CHUNK")
)
```

Build `_ring_pc_by_n` from `ring_chunks_by_n.items()`.

- [ ] **Step 4: Run focused and neighboring host tests**

Run:

```bash
PYTHONPATH="$PWD/ttnn:$PWD" /home/smarton/tt-metal/python_env/bin/python -m pytest \
  models/tt_dit/tests/models/ltx/test_transformer_ltx.py \
  -k 'ring_sdpa_chunk_override_reaches_per_n_configs' -q
/home/smarton/tt-workflows/scripts/build.sh data "Ring SDPA override resolver"
```

Expected: pass.

- [ ] **Step 5: Commit**

Commit implementation and test with subject:

```text
ltx/attn: apply ring chunk override to per-N configs
```

### Task 3: Complete AGMM program-cache identity

**Files:**
- Modify: `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/device/all_gather_minimal_matmul_async_device_operation_types.hpp:101-139`
- Create: `tests/ttnn/unit_tests/gtests/ccl/test_all_gather_minimal_matmul_async_cache_identity.cpp`
- Modify: `tests/ttnn/unit_tests/gtests/sources.cmake:43-52`
- Modify: `models/tt_dit/tests/models/wan2_2/test_all_gather_minimal_matmul_async.py`

**Interfaces:**
- Produces: distinct program-cache identity for `fused_activation`, `output_dtype`, and `compute_kernel_config`.
- Preserves: cache reuse across differing runtime ternary scalar values.

- [ ] **Step 1: Add the device-free identity regression**

Create a GTest that builds otherwise-identical `AllGatherMinimalMatmulAsyncParams`, hashes reflected attributes with `ttsl::hash::hash_objects_with_default_seed`, and asserts distinct identity for:

```cpp
const auto baseline = hash_attributes(std::nullopt, std::nullopt, DeviceComputeKernelConfig{});
EXPECT_NE(
    hash_attributes(
        operations::unary::UnaryWithParam{operations::unary::UnaryOpType::EXP},
        std::nullopt,
        DeviceComputeKernelConfig{}),
    baseline);
EXPECT_NE(hash_attributes(std::nullopt, DataType::BFLOAT16, DeviceComputeKernelConfig{}), baseline);
auto changed_compute_config = DeviceComputeKernelConfig{};
changed_compute_config.math_approx_mode = !changed_compute_config.math_approx_mode;
EXPECT_NE(hash_attributes(std::nullopt, std::nullopt, changed_compute_config), baseline);
```

Construct the attributes with fixed values for every other constructor argument, including `chunks=1`, `dim=-1`, and `fuse_swiglu=false`. Add the source under `UNIT_TESTS_TTNN_CCL_SOURCES`.

- [ ] **Step 2: Add the same-process device regression**

Add a `2x4`, Ring, cluster-axis-1 test that creates one submesh and calls `run_test_linear` twice with exactly the same shape and configuration:

```python
common = dict(
    M=32,
    K=2048,
    N=2048,
    M_block_size=1,
    K_block_size=8,
    N_block_size=8,
    subblock_h=1,
    subblock_w=2,
    topology=ttnn.Topology.Ring,
    core_grid=ttnn.CoreCoord(4, 4),
    num_workers_per_link=4,
    num_links=1,
    use_non_fused=False,
    force_transpose=True,
    sp_axis=0,
    tp_axis=1,
    cluster_axis=1,
    chunks=1,
)
plain = run_test_linear(submesh, activation=None, **common)
gelu = run_test_linear(submesh, activation="gelu", **common)
for result in (plain, gelu):
    assert result[0][0][0]["pcc"] > 0.9995
    assert result[0][0][0]["relative_rmse"] < 0.02
```

Use the existing `(2,4)` `FABRIC_1D` fixture and `_create_cluster_submesh`.

- [ ] **Step 3: Build and verify the host regression is RED**

Run:

```bash
TTW_CONFIG=/home/smarton/ltx-rt-integration/ttw.toml \
TTW_STATE=/home/smarton/ltx-rt-integration/.ttw \
/home/smarton/tt-workflows/scripts/build.sh cpp "AGMM identity regression before fix"
build_Release/test/ttnn/unit_tests_ttnn_ccl \
  --gtest_filter='AllGatherMinimalMatmulAsync.CompileAffectingAttributesHaveDistinctProgramCacheIdentity'
```

Expected: GTest fails because all three changed attributes are currently omitted.

- [ ] **Step 4: Add the missing reflected attributes**

Keep names and values in the same order:

```cpp
static constexpr auto attribute_names = std::make_tuple(
    "num_links",
    "ring_size",
    "output_mem_config",
    "topology",
    "cluster_axis",
    "force_transpose",
    "num_workers_per_link",
    "num_buffers_per_channel",
    "config",
    "fused_activation",
    "output_dtype",
    "compute_kernel_config",
    "fsdp_cluster_axis",
    "fsdp_ring_size",
    "using_persistent_weight_buffer",
    "chunks",
    "dim",
    "fuse_swiglu");

auto attribute_values() const {
    return std::forward_as_tuple(
        this->num_links,
        this->ring_size,
        this->output_mem_config,
        this->topology,
        this->cluster_axis,
        this->force_transpose,
        this->num_workers_per_link,
        this->num_buffers_per_channel,
        this->config,
        this->fused_activation,
        this->output_dtype,
        this->compute_kernel_config,
        this->fsdp_cluster_axis,
        this->fsdp_ring_size,
        this->using_persistent_weight_buffer,
        this->chunks,
        this->dim,
        this->fuse_swiglu);
}
```

Add the corresponding values directly to `attribute_values()`. Do not add `fused_ternary_scalar`.

- [ ] **Step 5: Rebuild and verify GREEN**

Run the stamped C++ build again, then the same GTest. Expected: pass.

Run formatting and diff checks:

```bash
git diff --check
clang-format --dry-run --Werror \
  tests/ttnn/unit_tests/gtests/ccl/test_all_gather_minimal_matmul_async_cache_identity.cpp \
  ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/device/all_gather_minimal_matmul_async_device_operation_types.hpp
```

- [ ] **Step 6: Commit**

Commit the header and both regressions with subject:

```text
ttnn/ccl: complete AGMM program cache identity
```

### Task 4: Restore prewarm controls and stale-code coverage

**Files:**
- Modify: `tt_metal/tools/kernel_prewarm/prewarm_and_submit.sh:63-75`
- Modify: `ttnn/cpp/ttnn-nanobind/device.cpp`
- Modify: `tests/ttnn/unit_tests/base_functionality/test_device.py`
- Create: `tests/tt_metal/tt_metal/tools/test_kernel_prewarm_wrapper.py`
- Modify: `tests/tt_metal/tt_metal/api/test_offline_kernel_compile.cpp`

**Interfaces:**
- Produces Python bindings:
  `kernel_prewarm_set_capture_only(bool)`,
  `kernel_prewarm_cold_start_needed() -> bool`,
  and `kernel_prewarm_offline_compile() -> int`.
- Guarantees stage-one compound commands inherit capture-only mode.
- Restores end-to-end stale-kernel binary coverage.

- [ ] **Step 1: Write the wrapper propagation regression**

Use `tmp_path` to create:

- an env YAML with `TT_METAL_CACHE` and `TT_METAL_HOME`;
- a fake executable `build_Release/tools/kernel_prewarm`;
- a fake `tt-device-mcp` first on `PATH`.

The fake broker command records its second argument and appends one line to
`$TEST_CACHE/kernel_prewarm.manifest` for `run`; it exits zero for `run-bg`.
Invoke:

```python
subprocess.run(
    [
        str(script),
        "-e",
        str(env_file),
        "-c",
        "--",
        "cd /tmp && env",
    ],
    env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}", "TEST_CACHE": str(cache)},
    check=True,
)
assert recorded_command == "export TT_METAL_KERNEL_CAPTURE_ONLY=1; cd /tmp && env"
```

- [ ] **Step 2: Write the binding-surface regression**

Add:

```python
def test_kernel_prewarm_control_bindings_exist():
    names = (
        "kernel_prewarm_set_capture_only",
        "kernel_prewarm_cold_start_needed",
        "kernel_prewarm_offline_compile",
    )
    assert all(hasattr(ttnn._ttnn.device, name) for name in names)
```

- [ ] **Step 3: Restore stale-code tests in their existing test file**

Restore the helper kernel writer, ELF reader, and the two named
`MeshDeviceFixture` tests from first parent `8672b35d9bed` into
`test_offline_kernel_compile.cpp`, where those tests originally lived. Adapt
them to coexist with the current public offline-compile tests. Add the required
current headers:

```cpp
#include "device_fixture.hpp"
#include "impl/program/kernel_prewarm.hpp"
#include "jit_build/build.hpp"
#include "tt_metal/jit_build/build_env_manager.hpp"
```

Keep temporary kernel files unique and remove them at test end. No CMake source
registration change is needed because the file is already in the API test
target.

- [ ] **Step 4: Verify the host-visible tests are RED**

Run:

```bash
PYTHONPATH="$PWD/ttnn:$PWD" /home/smarton/tt-metal/python_env/bin/python -m pytest \
  tests/tt_metal/tt_metal/tools/test_kernel_prewarm_wrapper.py \
  tests/ttnn/unit_tests/base_functionality/test_device.py \
  -k 'kernel_prewarm' -q
```

Expected: wrapper propagation and binding-surface tests fail.

- [ ] **Step 5: Restore the wrapper export and nanobind functions**

Change stage one to:

```bash
tt-device-mcp run "export TT_METAL_KERNEL_CAPTURE_ONLY=1; $CMD" \
  -w "$WORKSPACE" -t "$TIMEOUT" ${ENV_FILE:+-e "$ENV_FILE"} || true
```

Leave the existing `TIMEOUT=1200` interface unchanged. It was not lost in the
merge and is outside this surgical forward-port. Task 5 passes its chosen
timeout explicitly when it invokes the wrapper.

Add:

```cpp
#include <tt-metalium/kernel_prewarm_control.hpp>
```

and bind the three `KernelPrewarm*` functions in `device_module` using the
docstrings from first parent `8672b35d9bed`.

- [ ] **Step 6: Build and run host gates**

Run the stamped C++ build. Then run:

```bash
PYTHONPATH="$PWD/ttnn:$PWD" /home/smarton/tt-metal/python_env/bin/python -m pytest \
  tests/tt_metal/tt_metal/tools/test_kernel_prewarm_wrapper.py \
  tests/ttnn/unit_tests/base_functionality/test_device.py \
  -k 'kernel_prewarm' -q
bash -n tt_metal/tools/kernel_prewarm/prewarm_and_submit.sh
git diff --check
```

Expected: pass, and `ttnn._ttnn.__file__` resolves inside the integration worktree.

- [ ] **Step 7: Commit**

Commit all prewarm repair files with subject:

```text
tt_metal/prewarm: restore cold-start safety interfaces
```

### Task 5: Run broker-serialized device regressions

**Files:**
- Create outside git: `/home/smarton/ltx-rt-integration/device.env.yaml`
- Update outside git: `/home/smarton/ltx-rt-integration/PROGRESS.md`

**Interfaces:**
- Consumes: built worktree and all focused tests.
- Produces: broker job IDs and complete logs tied to the candidate commit.

- [ ] **Step 1: Create the device environment**

Use shared, readable model paths and a worktree-specific cache:

```yaml
TT_METAL_HOME: "/home/smarton/.claude/worktrees/ltx-rt-fixes-2026-09-03"
PYTHONPATH: "/home/smarton/.claude/worktrees/ltx-rt-fixes-2026-09-03/ttnn:/home/smarton/.claude/worktrees/ltx-rt-fixes-2026-09-03"
PYTHON_ENV_DIR: "/home/smarton/tt-metal/python_env"
TT_METAL_CACHE: "/home/smarton/.cache/tt-metal/ltx-rt-fixes-2026-09-03"
LTX_CHECKPOINT: "/home/models/ltx-2.3/ltx-2.3-22b-distilled-1.1.safetensors"
GEMMA_PATH: "/home/models/gemma-3-12b-it-qat-q4_0-unquantized/"
HF_HOME: "/home/smarton/hf"
```

- [ ] **Step 2: Query the queue**

Call `tt_device_queue_status` before submission. If another tenant runs, queue
normally; do not bypass, kill, or reset.

- [ ] **Step 3: Run each stale-code regression separately**

Through `tt_device_job_run` with owner `[claude]smarton`, the worktree as
`workspace`, and the YAML as `env`, run:

```bash
TT_METAL_SLOW_DISPATCH_MODE=1 build_Release/test/tt_metal/unit_tests_api \
  --gtest_filter='MeshDeviceFixture.OfflinePrewarmReflectsEditedKernelBody'
```

Then run:

```bash
TT_METAL_SLOW_DISPATCH_MODE=1 build_Release/test/tt_metal/unit_tests_api \
  --gtest_filter='MeshDeviceFixture.EditedKernelBodyForcesRecompileNotStaleCacheHit'
```

Require exit code zero for both.

- [ ] **Step 4: Run the AGMM same-process regression**

Through `tt_device_job_run`, run only the new `2x4` identity test:

```bash
/home/smarton/tt-metal/python_env/bin/python -m pytest \
  models/tt_dit/tests/models/wan2_2/test_all_gather_minimal_matmul_async.py \
  -k 'cache_identity and 2x4' -s
```

Require both Torch comparisons and program execution to pass.

- [ ] **Step 5: Run a prewarmed LTX regression**

Use the worktree's `prewarm_and_submit.sh -c` with the device YAML and an
explicit 600-second timeout for this validation. Run:

```bash
tt_metal/tools/kernel_prewarm/prewarm_and_submit.sh \
  -e /home/smarton/ltx-rt-integration/device.env.yaml \
  -w /home/smarton/.claude/worktrees/ltx-rt-fixes-2026-09-03 \
  -t 600 \
  -c -- \
  "/home/smarton/tt-metal/python_env/bin/python -m pytest models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py::test_pipeline_distilled -k bh_2x4sp1tp0 -s"
```

Require stage-one capture-only, stage-two off-device compile, stage-three warm
execution, and the test's quality gate. Claim the returned stage-three broker
job in the loop registry and wait through `tt_device_job_wait`.

- [ ] **Step 6: Record evidence**

Append candidate SHA, build ID, broker job IDs, statuses, run times, and log
paths to `PROGRESS.md`. Do not mark `device_tests=PASS` until every job is
independently rechecked.

### Task 6: Review, deploy, and validate live server media

**Files:**
- Read: all changes since `3a41478432f`
- Update outside git: `/home/smarton/ltx-rt-integration/PROGRESS.md`
- Create outside git: validated media under `/home/smarton/ltx-rt-integration/media/`

**Interfaces:**
- Produces: a reviewed candidate deployed from `/home/smarton/tt-metal`.
- Produces: 720p and 1080p server-media evidence.

- [ ] **Step 1: Run task and whole-branch review**

Generate review packages from `3a41478432f` to candidate HEAD. Run the
task-scoped review loops and a final TT-aware whole-branch review. Resolve all
Critical/Important findings and re-run affected tests.

- [ ] **Step 2: Prepare the serving checkout safely**

The serving checkout has the same path repair as Task 1. Preserve it by first
committing it there or stashing only that file. Then fast-forward local
`ltx-rt` to the reviewed candidate. Do not reset.

- [ ] **Step 3: Restart and verify server identity**

Restart through `/home/smarton/loops/restart_server.sh`. Verify:

```bash
curl -fsS http://127.0.0.1:8081/health
git -C /home/smarton/tt-metal rev-parse HEAD
```

Require `status=ok`, `device_ready=true`, `worker_up=true`, and exact candidate
SHA.

- [ ] **Step 4: Mint a dedicated direct-client key**

Run:

```bash
cd /home/smarton/ltx-server
/home/smarton/tt-metal/python_env/bin/python -m ltx_server.apikeys \
  --data-dir /home/smarton/ltx-server/data mint ltx-rt-fixes-validation
```

Keep the raw key outside git.

- [ ] **Step 5: Validate 720p and 1080p through the server**

Run `tools/model_bringup/validate.py` twice against port 8081:

```bash
python tools/model_bringup/validate.py --base-url http://127.0.0.1:8081 \
  --model ltx-fast --seconds 6 --size 1280x704 \
  --output /home/smarton/ltx-rt-integration/media/ltx-fast-720p.mp4
python tools/model_bringup/validate.py --base-url http://127.0.0.1:8081 \
  --model ltx-fast --seconds 6 --size 1920x1088 \
  --output /home/smarton/ltx-rt-integration/media/ltx-fast-1080p.mp4
```

Require completed jobs, H.264 video, AAC audio, exact dimensions, duration at
least five seconds, and non-flat luma evidence.

### Task 7: Merge, push, and satisfy the frozen goal

**Files:**
- Create outside git: `/home/smarton/ltx-rt-integration/evidence.json`
- Update outside git: `/home/smarton/ltx-rt-integration/PROGRESS.md`

**Interfaces:**
- Produces: `origin/ltx-rt` at the exact tested commit.
- Produces: frozen-goal success tied to build, device, review, and media evidence.

- [ ] **Step 1: Fast-forward local `ltx-rt`**

Confirm the serving checkout is clean and already at candidate HEAD. Do not
merge if the deployed commit differs from the reviewed/tested commit.

- [ ] **Step 2: Push**

Push the tested local branch:

```bash
git push origin ltx-rt:ltx-rt
```

If the remote moved, fetch and rebase the integration commits onto the new
remote tip, rebuild, repeat affected device/server gates, and then push. Never
force.

- [ ] **Step 3: Verify remote identity**

Run:

```bash
git ls-remote origin refs/heads/ltx-rt
git -C /home/smarton/tt-metal rev-parse HEAD
```

Require equal SHAs.

- [ ] **Step 4: Write final evidence**

Generate the file from the deployed checkout:

```python
import json
import subprocess

commit = subprocess.run(
    ["git", "-C", "/home/smarton/tt-metal", "rev-parse", "HEAD"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
evidence = {
    "commit": commit,
    "prewarm_equivalent": "PASS",
    "cache_regression": "PASS",
    "host_tests": "PASS",
    "device_tests": "PASS",
    "code_review": "PASS",
    "server_outputs": [
        {
            "path": "/home/smarton/ltx-rt-integration/media/ltx-fast-720p.mp4",
            "dimensions": [1280, 704],
            "minimum_duration": 5.0,
        },
        {
            "path": "/home/smarton/ltx-rt-integration/media/ltx-fast-1080p.mp4",
            "dimensions": [1920, 1088],
            "minimum_duration": 5.0,
        },
    ],
}
with open("/home/smarton/ltx-rt-integration/evidence.json", "w") as stream:
    json.dump(evidence, stream, indent=2)
    stream.write("\n")
```

- [ ] **Step 5: Run the frozen completion gate**

Run:

```bash
/home/smarton/ltx-rt-integration/goal_check.sh
```

Expected:

```text
GOAL_OK: fixes are on pushed ltx-rt with passing host, device, review, and server-media gates
```

Create `/home/smarton/ltx-rt-integration/DONE` only after this exits zero.
