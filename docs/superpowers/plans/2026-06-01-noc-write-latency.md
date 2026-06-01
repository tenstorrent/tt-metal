# NOC Write Round-Trip Latency Tests — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two Quasar-only tests to the `data_movement` suite that measure round-trip NOC write latency (far-corner and adjacent cores) using `rdcycles` + `DEVICE_PRINT`.

**Architecture:** Sender kernel issues a 32-byte `noc_async_write`, times the barrier with `rdcycles`, then sends a plain-write flag to the receiver. Receiver kernel spins on that flag via an uncached volatile pointer and prints confirmation. Both kernels use compile-time args only and run in a single `Program` via two `experimental::quasar::CreateKernel` calls.

**Tech Stack:** C++20, tt-metal `experimental::quasar` API, `noc_async_write` / `noc_async_write_barrier`, `DEVICE_PRINT`, inline `rdcycle` RISC-V CSR, `GenericMeshDeviceFixture`.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `tests/tt_metal/tt_metal/data_movement/noc_write_latency/kernels/sender.cpp` | Issues timed write + flag signal |
| Create | `tests/tt_metal/tt_metal/data_movement/noc_write_latency/kernels/receiver.cpp` | Spins on flag, prints confirmation |
| Create | `tests/tt_metal/tt_metal/data_movement/noc_write_latency/test_noc_write_latency.cpp` | Host test: config, program setup, two GTest cases |
| Modify | `tests/tt_metal/tt_metal/data_movement/sources.cmake` | Register new test file |

---

## Task 1: Sender kernel

**Files:**
- Create: `tests/tt_metal/tt_metal/data_movement/noc_write_latency/kernels/sender.cpp`

- [ ] **Step 1: Create the sender kernel**

```cpp
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/debug/device_print.h"
#include "dev_mem_map.h"

static inline uint32_t rdcycles() {
    uint32_t c;
    asm volatile("rdcycle %0" : "=r"(c));
    return c;
}

void kernel_main() {
    // Compile-time args
    constexpr uint32_t src_l1_addr              = get_compile_time_arg_val(0);
    constexpr uint32_t flag_local_addr           = get_compile_time_arg_val(1);
    constexpr uint32_t dst_l1_data_addr          = get_compile_time_arg_val(2);
    constexpr uint32_t dst_l1_flag_addr          = get_compile_time_arg_val(3);
    constexpr uint32_t dst_noc_x                 = get_compile_time_arg_val(4);
    constexpr uint32_t dst_noc_y                 = get_compile_time_arg_val(5);
    constexpr uint32_t num_iterations            = get_compile_time_arg_val(6);
    constexpr uint32_t transaction_size_bytes    = get_compile_time_arg_val(7);

    uint64_t dst_data_noc = get_noc_addr(dst_noc_x, dst_noc_y, dst_l1_data_addr);
    uint64_t dst_flag_noc = get_noc_addr(dst_noc_x, dst_noc_y, dst_l1_flag_addr);

    for (uint32_t i = 0; i < num_iterations; i++) {
        // Write payload to local L1 bypassing cache
        *(volatile tt_l1_ptr uint32_t*)(src_l1_addr + MEM_L1_UNCACHED_BASE) = 0xDEADBEEF;

        // Issue the data write
        noc_async_write(src_l1_addr, dst_data_noc, transaction_size_bytes);

        // Time the barrier (= round-trip until write ACK received)
        uint32_t t0 = rdcycles();
        noc_async_write_barrier();
        uint32_t t1 = rdcycles();

        DEVICE_PRINT("[sender] iter %u: %u cycles\n", i, t1 - t0);

        // Signal receiver with a flag write
        *(volatile tt_l1_ptr uint32_t*)(flag_local_addr + MEM_L1_UNCACHED_BASE) = i + 1;
        noc_async_write(flag_local_addr, dst_flag_noc, sizeof(uint32_t));
        noc_async_write_barrier();
    }
}
```

- [ ] **Step 2: Verify file exists**

```bash
ls tests/tt_metal/tt_metal/data_movement/noc_write_latency/kernels/sender.cpp
```
Expected: file listed with no error.

---

## Task 2: Receiver kernel

**Files:**
- Create: `tests/tt_metal/tt_metal/data_movement/noc_write_latency/kernels/receiver.cpp`

- [ ] **Step 1: Create the receiver kernel**

```cpp
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/debug/device_print.h"
#include "dev_mem_map.h"

void kernel_main() {
    // Compile-time args
    constexpr uint32_t flag_l1_addr   = get_compile_time_arg_val(0);
    constexpr uint32_t num_iterations = get_compile_time_arg_val(1);
    constexpr uint32_t my_noc_x       = get_compile_time_arg_val(2);
    constexpr uint32_t my_noc_y       = get_compile_time_arg_val(3);

    volatile tt_l1_ptr uint32_t* flag_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flag_l1_addr + MEM_L1_UNCACHED_BASE);

    for (uint32_t i = 0; i < num_iterations; i++) {
        // Spin until sender writes flag value i+1
        while (*flag_ptr != i + 1) {}
    }

    DEVICE_PRINT("[receiver] core (%u,%u) received all %u writes\n", my_noc_x, my_noc_y, num_iterations);
}
```

- [ ] **Step 2: Verify file exists**

```bash
ls tests/tt_metal/tt_metal/data_movement/noc_write_latency/kernels/receiver.cpp
```
Expected: file listed with no error.

---

## Task 3: Host test file

**Files:**
- Create: `tests/tt_metal/tt_metal/data_movement/noc_write_latency/test_noc_write_latency.cpp`

- [ ] **Step 1: Create the host test**

```cpp
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "multi_device_fixture.hpp"
#include "dm_common.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include "impl/context/metal_context.hpp"
#include "impl/host_api/temp_quasar_api.hpp"

namespace tt::tt_metal {

using namespace std;

namespace unit_tests::dm::noc_write_latency {

struct NocWriteLatencyConfig {
    CoreCoord src_core;
    CoreCoord dst_core;
    uint32_t num_iterations       = 100;
    uint32_t transaction_size_bytes = 32;
};

bool run_noc_write_latency(
    const shared_ptr<distributed::MeshDevice>& mesh_device,
    const NocWriteLatencyConfig& cfg) {

    IDevice* device = mesh_device->get_device(0);

    // Skip on non-Quasar or grids smaller than 9x4
    if (MetalContext::instance().get_cluster().arch() != ARCH::QUASAR) {
        log_info(LogTest, "Skipping: not a Quasar device");
        return true;
    }
    auto grid = device->compute_with_storage_grid_size();
    if (grid.x < 9 || grid.y < 4) {
        log_info(LogTest, "Skipping: grid {}x{} smaller than 9x4", grid.x, grid.y);
        return true;
    }

    // Physical NOC coordinates for source and destination
    CoreCoord phys_src = device->worker_core_from_logical_core(cfg.src_core);
    CoreCoord phys_dst = device->worker_core_from_logical_core(cfg.dst_core);

    // L1 address layout (sender core)
    L1AddressInfo src_l1 = unit_tests::dm::get_l1_address_and_size(mesh_device, cfg.src_core);
    uint32_t src_l1_addr    = src_l1.base_address;          // payload source
    uint32_t flag_local_addr = src_l1.base_address + 64;    // outgoing flag (64B past payload)

    // L1 address layout (receiver core)
    L1AddressInfo dst_l1 = unit_tests::dm::get_l1_address_and_size(mesh_device, cfg.dst_core);
    uint32_t dst_l1_data_addr = dst_l1.base_address;        // incoming payload
    uint32_t dst_l1_flag_addr = dst_l1.base_address + 64;   // incoming flag

    // Zero the receiver flag in device L1 before starting
    vector<uint32_t> zero{0};
    detail::WriteToDeviceL1(device, cfg.dst_core, dst_l1_flag_addr, zero);
    MetalContext::instance().get_cluster().l1_barrier(device->id());

    Program program = CreateProgram();

    // Sender kernel
    experimental::quasar::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/data_movement/noc_write_latency/kernels/sender.cpp",
        cfg.src_core,
        experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = 1,
            .compile_args = {
                src_l1_addr,
                flag_local_addr,
                dst_l1_data_addr,
                dst_l1_flag_addr,
                (uint32_t)phys_dst.x,
                (uint32_t)phys_dst.y,
                cfg.num_iterations,
                cfg.transaction_size_bytes,
            }});

    // Receiver kernel
    experimental::quasar::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/data_movement/noc_write_latency/kernels/receiver.cpp",
        cfg.dst_core,
        experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = 1,
            .compile_args = {
                dst_l1_flag_addr,
                cfg.num_iterations,
                (uint32_t)phys_dst.x,
                (uint32_t)phys_dst.y,
            }});

    program.set_runtime_id(unit_tests::dm::runtime_host_id++);

    MetalContext::instance().get_cluster().l1_barrier(device->id());

    auto mesh_workload = distributed::MeshWorkload();
    vector<uint32_t> coord_data = {0, 0};
    auto target_devices = distributed::MeshCoordinateRange(distributed::MeshCoordinate(coord_data));
    mesh_workload.add_program(target_devices, std::move(program));

    auto& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, mesh_workload, false);
    Finish(cq);

    return true;
}

}  // namespace unit_tests::dm::noc_write_latency

TEST_F(GenericMeshDeviceFixture, NocWriteLatencyFarCorners) {
    unit_tests::dm::noc_write_latency::NocWriteLatencyConfig cfg{
        .src_core = {0, 0},
        .dst_core = {8, 3},
        .num_iterations = 100,
        .transaction_size_bytes = 32,
    };
    EXPECT_TRUE(unit_tests::dm::noc_write_latency::run_noc_write_latency(this->mesh_device_, cfg));
}

TEST_F(GenericMeshDeviceFixture, NocWriteLatencyAdjacentCores) {
    unit_tests::dm::noc_write_latency::NocWriteLatencyConfig cfg{
        .src_core = {0, 0},
        .dst_core = {1, 0},
        .num_iterations = 100,
        .transaction_size_bytes = 32,
    };
    EXPECT_TRUE(unit_tests::dm::noc_write_latency::run_noc_write_latency(this->mesh_device_, cfg));
}

}  // namespace tt::tt_metal
```

- [ ] **Step 2: Verify file exists**

```bash
ls tests/tt_metal/tt_metal/data_movement/noc_write_latency/test_noc_write_latency.cpp
```
Expected: file listed with no error.

---

## Task 4: Register in sources.cmake

**Files:**
- Modify: `tests/tt_metal/tt_metal/data_movement/sources.cmake`

- [ ] **Step 1: Add the new test file**

In `tests/tt_metal/tt_metal/data_movement/sources.cmake`, append before the closing `)`:

```cmake
    noc_write_latency/test_noc_write_latency.cpp
```

The end of the file should look like:

```cmake
    quasar_examples/quasar_idma/test_idma_example.cpp
    noc_write_latency/test_noc_write_latency.cpp
)
```

- [ ] **Step 2: Commit**

```bash
git add \
  tests/tt_metal/tt_metal/data_movement/noc_write_latency/kernels/sender.cpp \
  tests/tt_metal/tt_metal/data_movement/noc_write_latency/kernels/receiver.cpp \
  tests/tt_metal/tt_metal/data_movement/noc_write_latency/test_noc_write_latency.cpp \
  tests/tt_metal/tt_metal/data_movement/sources.cmake
git commit -m "Add NOC write round-trip latency tests for Quasar 9x4"
```

---

## Task 5: Build and verify compilation

**Files:** (no changes — build only)

- [ ] **Step 1: Activate python env and build**

```bash
source /proj_sw/user_dev/vvukomanovic/tt-metal/python_env/bin/activate && \
./build_metal.sh --build-tests 2>&1 | tail -40
```
Expected: build completes with `[N/N] Install the project...` at the end, no errors.

- [ ] **Step 2: If compilation fails due to `rdcycle` not recognized**

Replace the `rdcycles()` helper in `sender.cpp` with a `csrrw`-based read on CSR 0xC00 (the standard `mcycle` register):

```cpp
static inline uint32_t rdcycles() {
    uint32_t c;
    asm volatile("csrr %0, 0xC00" : "=r"(c));
    return c;
}
```

Rebuild after the change.

- [ ] **Step 3: If compilation fails due to `DEVICE_PRINT` not found**

Add `#include "api/debug/dprint.h"` instead of `api/debug/device_print.h` — `dprint.h` includes `device_print.h` and defines `DEVICE_PRINT` via the same header. Rebuild.

---

## Task 6: Run the tests

**Files:** (no changes — run only)

- [ ] **Step 1: Set emulator path and run far-corners test**

```bash
export TT_METAL_SIMULATOR=/proj_sw/user_dev/vvukomanovic/tt-umd-simulators/build/emu-quasar-9x4_DM && \
./build_Release/test/tt_metal/unit_tests_data_movement \
  --gtest_filter="*NocWriteLatencyFarCorners*" \
  --gtest_color=yes
```

Expected output contains lines like:
```
[sender] iter 0: NNNN cycles
[sender] iter 1: NNNN cycles
...
[receiver] core (X,Y) received all 100 writes
[ PASSED ] NocWriteLatencyFarCorners
```

- [ ] **Step 2: Run adjacent-cores test**

```bash
export TT_METAL_SIMULATOR=/proj_sw/user_dev/vvukomanovic/tt-umd-simulators/build/emu-quasar-9x4_DM && \
./build_Release/test/tt_metal/unit_tests_data_movement \
  --gtest_filter="*NocWriteLatencyAdjacentCores*" \
  --gtest_color=yes
```

Expected output same pattern, with `[receiver] core (X,Y) received all 100 writes` and `[ PASSED ]`.

- [ ] **Step 3: Commit**

If both tests pass:

```bash
git add -p  # nothing new to stage; all files already committed in Task 4
# Only commit if any fixup changes were made during build/run
```

If fixup changes were made during Tasks 5–6, stage and commit:
```bash
git add tests/tt_metal/tt_metal/data_movement/noc_write_latency/kernels/sender.cpp \
        tests/tt_metal/tt_metal/data_movement/noc_write_latency/kernels/receiver.cpp
git commit -m "Fix rdcycles/DEVICE_PRINT after build verification"
```

---

## Self-review checklist (already applied)

- Spec coverage: sender kernel ✓, receiver kernel ✓, two GTests ✓, sources.cmake ✓, uncached L1 pattern ✓, rdcycles ✓, DEVICE_PRINT ✓
- No TBDs or placeholders — every step has exact code or exact commands
- Type/name consistency: `NocWriteLatencyConfig`, `run_noc_write_latency`, `src_l1_addr`, `flag_local_addr`, `dst_l1_data_addr`, `dst_l1_flag_addr` consistent across Tasks 3, 1, and 2
