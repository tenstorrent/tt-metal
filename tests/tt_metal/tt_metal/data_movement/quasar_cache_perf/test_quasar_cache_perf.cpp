// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar DM-core cache-write performance test (test id 912).

#include <tt-logger/tt-logger.hpp>
#include "device_fixture.hpp"
#include "dm_common.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/hal.hpp>
#include <chrono>
#include <cstdint>
#include <thread>
#include "impl/context/metal_context.hpp"
#include "kernels/cache_write_probe.hpp"
#include "llrt/hal.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"

namespace tt::tt_metal {

using namespace std;
using namespace tt::test_utils;

namespace unit_tests::dm::quasar_cache_perf {

constexpr std::uint32_t QUASAR_CACHE_WRITE_TEST_ID = 912;
constexpr std::uint32_t BASE_ADDR = 100 * 1024;
// Repeat the timed write(+flush) region this many times per run and divide the
// zone duration by it, to amortize fixed per-run overhead (April used 100).
constexpr std::uint32_t NUM_ITERATIONS = 100;
// Mode 0 size 2048 has completed in about 35s. A launch still running after this has stalled.
constexpr std::chrono::seconds kWorkloadTimeout{120};

bool should_skip_test() { return std::getenv("TT_METAL_SIMULATOR") == nullptr; }

const char* cache_write_probe_phase_name(std::uint32_t phase) {
    switch (static_cast<CacheWriteProbePhase>(phase)) {
        case CacheWriteProbePhase::Entered: return "entered";
        case CacheWriteProbePhase::BeforeStores: return "before stores";
        case CacheWriteProbePhase::BeforeFlush: return "before flush";
        case CacheWriteProbePhase::AfterFlush: return "after flush";
        case CacheWriteProbePhase::Done: return "done";
    }
    return "unknown";
}

std::uint32_t cache_write_probe_l1_addr() {
    const auto& hal = MetalContext::instance().hal();
    const auto& dev_msgs_factory = hal.get_dev_msgs_factory(HalProgrammableCoreType::TENSIX);
    const auto watcher_addr = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::WATCHER);
    const auto ring_offset =
        dev_msgs_factory.offset_of<dev_msgs::watcher_msg_t>(dev_msgs::watcher_msg_t::Field::debug_ring_buf);
    return static_cast<std::uint32_t>(watcher_addr + ring_offset);
}

// One read, after the workload wait has already expired. EventQuery only looks at the host
// completion counter, so the wait itself does not touch this core's L1.
void log_cache_write_probe(distributed::MeshDevice& mesh_device, CoreCoord core, std::uint32_t probe_id) {
    std::vector<std::uint32_t> raw(sizeof(CacheWriteProbe) / sizeof(std::uint32_t));
    log_error(tt::LogTest, "QuasarCacheWrite timed out; reading watcher probe");
    slow_dispatch::ReadFromL1(
        mesh_device, core, cache_write_probe_l1_addr(), static_cast<std::uint32_t>(sizeof(CacheWriteProbe)), raw);
    const auto* probe = reinterpret_cast<const CacheWriteProbe*>(raw.data());
    if (probe->magic != kCacheWriteProbeMagic || probe->probe_id != probe_id) {
        log_error(
            tt::LogTest,
            "QuasarCacheWrite probe missing for id {} (magic {:#x} id {})",
            probe_id,
            probe->magic,
            probe->probe_id);
        return;
    }
    log_error(
        tt::LogTest,
        "QuasarCacheWrite probe mode {} size {} iters {} phase {} ({}) iteration {} flush_addr {:#x}",
        probe->mode,
        probe->size_bytes,
        probe->num_iterations,
        probe->phase,
        cache_write_probe_phase_name(probe->phase),
        probe->iteration,
        probe->flush_addr);
}

bool wait_for_workload(distributed::MeshCommandQueue& mesh_cq, distributed::MeshEvent event) {
    const auto deadline = std::chrono::steady_clock::now() + kWorkloadTimeout;
    while (!distributed::EventQuery(event)) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
    return true;
}

// Runs one write pass of `size_bytes` via `write_path` (0=uncached,1=cached+flush)
// on a single DM core, then reads back and verifies the byte pattern landed.
bool run_cache_write(
    distributed::MeshDevice& mesh_device,
    std::uint32_t size_bytes,
    std::uint32_t write_path,
    std::uint32_t num_iterations = NUM_ITERATIONS) {
    constexpr CoreCoord core = {0, 0};
    const experimental::NodeCoord node{0, 0};
    log_info(
        tt::LogTest, "QuasarCacheWrite mode {} size {} iters {}: WriteToL1", write_path, size_bytes, num_iterations);

    // Pre-fill destination with a sentinel so a no-op run fails the read-back.
    std::vector<std::uint32_t> init_data((size_bytes + 3) / 4, 0xA5A5A5A5);
    slow_dispatch::WriteToL1(mesh_device, core, BASE_ADDR, init_data);
    log_info(
        tt::LogTest,
        "QuasarCacheWrite mode {} size {} iters {}: MakeProgramFromSpec",
        write_path,
        size_bytes,
        num_iterations);

    const experimental::KernelSpecName DM_KERNEL{"cache_write_perf"};
    experimental::KernelSpec dm_kernel_spec{
        .unique_id = DM_KERNEL,
        .source = "tests/tt_metal/tt_metal/data_movement/quasar_cache_perf/kernels/cache_write_perf.cpp",
        .num_threads = 1,
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"base_addr", "size_bytes", "write_path", "num_iterations", "test_id", "probe_id"},
            },
        .hw_config = experimental::DataMovementHardwareConfig{},
    };
    experimental::WorkUnitSpec main_wu{.name = "main", .kernels = {DM_KERNEL}, .target_nodes = node};
    experimental::ProgramSpec spec{.name = "cache_write_perf", .kernels = {dm_kernel_spec}, .work_units = {main_wu}};
    Program program = experimental::MakeProgramFromSpec(mesh_device, spec);
    log_info(
        tt::LogTest,
        "QuasarCacheWrite mode {} size {} iters {}: SetProgramRunArgs",
        write_path,
        size_bytes,
        num_iterations);

    static std::uint32_t next_probe_id = 1;
    const std::uint32_t probe_id = next_probe_id++;
    experimental::ProgramRunArgs params;
    params.kernel_run_args = {experimental::ProgramRunArgs::KernelRunArgs{
        .kernel = DM_KERNEL,
        .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
            node,
            {{"base_addr", BASE_ADDR},
             {"size_bytes", size_bytes},
             {"write_path", write_path},
             {"num_iterations", num_iterations},
             {"test_id", QUASAR_CACHE_WRITE_TEST_ID},
             {"probe_id", probe_id}}),
    }};
    experimental::SetProgramRunArgs(program, params);
    log_info(
        tt::LogTest,
        "QuasarCacheWrite mode {} size {} iters {}: EnqueueMeshWorkload",
        write_path,
        size_bytes,
        num_iterations);

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device.shape());
    workload.add_program(device_range, std::move(program));
    auto& mesh_cq = mesh_device.mesh_command_queue();
    distributed::EnqueueMeshWorkload(mesh_cq, workload, /*blocking=*/false);
    const distributed::MeshEvent event = mesh_cq.enqueue_record_event_to_host();
    if (!wait_for_workload(mesh_cq, event)) {
        log_cache_write_probe(mesh_device, core, probe_id);
        return false;
    }
    log_info(
        tt::LogTest, "QuasarCacheWrite mode {} size {} iters {}: ReadFromL1", write_path, size_bytes, num_iterations);

    std::vector<std::uint32_t> out;
    slow_dispatch::ReadFromL1(mesh_device, core, BASE_ADDR, ((size_bytes + 3) / 4) * 4, out);
    const std::uint8_t* bytes = reinterpret_cast<const std::uint8_t*>(out.data());
    for (std::uint32_t i = 0; i < size_bytes; i++) {
        if (bytes[i] != 0x5A) {
            log_error(
                tt::LogTest,
                "mode {} size {} iters {} mismatch at {}: got 0x{:02x}",
                write_path,
                size_bytes,
                num_iterations,
                i,
                bytes[i]);
            return false;
        }
    }
    log_info(tt::LogTest, "QuasarCacheWrite mode {} size {} iters {}: done", write_path, size_bytes, num_iterations);
    return true;
}

}  // namespace unit_tests::dm::quasar_cache_perf

class QuasarCacheWrite : public QuasarMeshDeviceSingleCardFixture {};

namespace unit_tests::dm::quasar_cache_perf {
// Sizes are multiples of 8 (kernel does 8-byte stores, no sub-8B tail). 24 = 3x uint64
// stores, the direct comparison point against the April descriptor-write baseline.
static const std::vector<std::uint32_t> kSizesBytes = {8, 16, 24, 32, 64, 128, 256, 512, 1024, 2048};
}  // namespace unit_tests::dm::quasar_cache_perf

TEST_F(QuasarCacheWrite, SizeSweep) {
    if (unit_tests::dm::quasar_cache_perf::should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }
    bool pass = true;
    // mode: 0=uncached 1B, 1=uncached 8B, 2=cached+range flush (April), 3=cached+fast flush
    for (std::uint32_t mode : {0u, 1u, 2u, 3u}) {
        for (std::uint32_t size_bytes : unit_tests::dm::quasar_cache_perf::kSizesBytes) {
            pass &= unit_tests::dm::quasar_cache_perf::run_cache_write(this->device(), size_bytes, mode);
        }
    }
    EXPECT_TRUE(pass);
}

// One mode and one size. Iteration count is the only thing that changes, and the counts share one
// device session, so the last "done" line before a stall is the highest count that finished.
void run_iteration_sweep(distributed::MeshDevice& mesh_device, std::uint32_t size_bytes, std::uint32_t write_path) {
    if (unit_tests::dm::quasar_cache_perf::should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }
    bool pass = true;
    for (std::uint32_t num_iterations : {1u, 10u, 25u, 50u, 100u}) {
        pass &= unit_tests::dm::quasar_cache_perf::run_cache_write(mesh_device, size_bytes, write_path, num_iterations);
    }
    EXPECT_TRUE(pass);
}

TEST_F(QuasarCacheWrite, Mode0Size2048Iterations) { run_iteration_sweep(this->device(), 2048, 0); }

TEST_F(QuasarCacheWrite, Mode2Size24Iterations) { run_iteration_sweep(this->device(), 24, 2); }

}  // namespace tt::tt_metal
