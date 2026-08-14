// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "multi_device_fixture.hpp"
#include "dm_common.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-logger/tt-logger.hpp>

namespace tt::tt_metal {

using namespace tt::test_utils;

namespace unit_tests::dm::compute_ipc {

constexpr auto kAluLoopIpc = "tests/tt_metal/tt_metal/data_movement/compute_ipc/kernels/alu_loop_ipc.cpp";

// 12 independent ALU ops + addi (decrement) + bnez (branch); kept in sync with alu_loop_ipc.cpp.
constexpr uint32_t kOpsPerIter = 14;
// Instructions retired between the two rdinstret reads = mv (1) + kOpsPerIter*iterations + the
// second pass's rdcycle (1) + the closing rdinstret itself (1) -- confirmed against disassembly
// (dm2.elf) 2026-08-14: the closing rdinstret counts its own retirement inclusively, so the window
// is mv + loop + rdcycle + rdinstret, not mv + loop + rdcycle.
constexpr uint32_t kFixedInstretOverhead = 3;
// lw + add (use) + addi (decrement) + bnez (branch); kept in sync with alu_loop_ipc.cpp.
constexpr uint32_t kLoadUseOpsPerIter = 4;
constexpr uint32_t kNumResultWords = 12;
constexpr uint32_t kResultBytes = kNumResultWords * sizeof(uint32_t);
constexpr uint32_t kLoadSrcBytes = 64;  // one cache line; contents unused, only the address matters

struct AluPassResult {
    uint32_t iterations;
    uint32_t cycles;
    uint32_t instret;
};

// Mirrors run_kernel() in quasar_examples/quasar_idma/test_idma_example.cpp: DataMovementGen2Config
// on Quasar, CreateReaderGen1DataMovementConfig elsewhere, so the same kernel source and launch path
// runs unmodified on both arches.
void run_kernel(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t result_addr, uint32_t load_src_addr) {
    const experimental::KernelSpecName DM_KERNEL{"alu_loop_ipc"};
    const experimental::NodeCoord node{0, 0};

    const auto arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
    experimental::DataMovementHardwareConfig dm_hw_config = experimental::DataMovementGen2Config{};
    if (arch != tt::ARCH::QUASAR) {
        dm_hw_config = experimental::CreateReaderGen1DataMovementConfig();
    }

    experimental::KernelSpec dm_kernel_spec{
        .unique_id = DM_KERNEL,
        .source = kAluLoopIpc,
        .num_threads = 1,
        .compile_time_args = {{"result_addr", result_addr}, {"load_src_addr", load_src_addr}},
        .hw_config = dm_hw_config,
    };

    experimental::WorkUnitSpec main_wu{
        .name = "main",
        .kernels = {DM_KERNEL},
        .target_nodes = node,
    };

    experimental::ProgramSpec spec{
        .name = "alu_loop_ipc",
        .kernels = {dm_kernel_spec},
        .work_units = {main_wu},
    };
    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {experimental::ProgramRunArgs::KernelRunArgs{.kernel = DM_KERNEL}};
    experimental::SetProgramRunArgs(program, params);

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());
    workload.add_program(device_range, std::move(program));
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, workload, true);
}

// Runs the isolated ALU-only loop on a single core and logs IPC for both passes. Returns false (with
// log_error) if either the exact-instruction-count check or the pass0->pass1 linearity check fails --
// either would mean the harness (not the hardware) is measuring the wrong thing.
bool run_alu_loop_ipc_bench(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    constexpr CoreCoord core = {0, 0};

    IDevice* device = mesh_device->get_devices()[0];
    distributed::DeviceLocalBufferConfig local_buffer_config = {
        .page_size = kResultBytes, .buffer_type = tt::tt_metal::BufferType::L1};
    distributed::ReplicatedBufferConfig buffer_config = {.size = kResultBytes};
    auto result_buffer = distributed::MeshBuffer::create(buffer_config, local_buffer_config, mesh_device.get());
    const uint32_t result_base = result_buffer->address();

    // Contents are never checked -- only the address is used, as the load source for
    // run_load_use_pass()'s cached/uncached comparison.
    distributed::DeviceLocalBufferConfig load_src_local_config = {
        .page_size = kLoadSrcBytes, .buffer_type = tt::tt_metal::BufferType::L1};
    distributed::ReplicatedBufferConfig load_src_config = {.size = kLoadSrcBytes};
    auto load_src_buffer = distributed::MeshBuffer::create(load_src_config, load_src_local_config, mesh_device.get());
    const uint32_t load_src_base = load_src_buffer->address();

    std::vector<uint32_t> zero_results(kNumResultWords, 0);
    tt_metal::detail::WriteToDeviceL1(device, core, result_base, zero_results);

    run_kernel(mesh_device, result_base, load_src_base);

    std::vector<uint32_t> result_data;
    tt_metal::detail::ReadFromDeviceL1(device, core, result_base, kResultBytes, result_data);

    const AluPassResult pass0{.iterations = result_data[0], .cycles = result_data[1], .instret = result_data[2]};
    const AluPassResult pass1{.iterations = result_data[3], .cycles = result_data[4], .instret = result_data[5]};
    const AluPassResult cached_load{.iterations = result_data[6], .cycles = result_data[7], .instret = result_data[8]};
    const AluPassResult uncached_load{
        .iterations = result_data[9], .cycles = result_data[10], .instret = result_data[11]};

    bool pass = true;
    for (const AluPassResult& r : {pass0, pass1}) {
        const uint32_t expected_instret = kOpsPerIter * r.iterations + kFixedInstretOverhead;
        const double ipc = r.cycles == 0 ? 0.0 : static_cast<double>(r.instret) / r.cycles;
        log_info(
            tt::LogTest,
            "ALU_IPC iterations={} cycles={} instret={} (expected {}) ipc={:.3f}",
            r.iterations,
            r.cycles,
            r.instret,
            expected_instret,
            ipc);
        if (r.instret != expected_instret) {
            log_error(
                tt::LogTest,
                "ALU_IPC instret mismatch at iterations={}: got {}, expected {} -- harness bug, not a hardware "
                "measurement",
                r.iterations,
                r.instret,
                expected_instret);
            pass = false;
        }
    }

    // Cheap linearity check in place of a full marginal-cost sweep: pass1 runs exactly 2x pass0's
    // iterations with no other fixed cost in this loop, so cycles should double too, within slop for
    // measurement granularity.
    const double ratio = pass0.cycles == 0 ? 0.0 : static_cast<double>(pass1.cycles) / pass0.cycles;
    log_info(tt::LogTest, "ALU_IPC pass1/pass0 cycle ratio={:.4f} (expected ~2.0)", ratio);
    if (ratio < 1.9 || ratio > 2.1) {
        log_error(tt::LogTest, "ALU_IPC cycle count did not scale linearly with iterations (ratio={:.4f})", ratio);
        pass = false;
    }

    // Load-use passes: same instret-exactness check as the ALU passes, using kLoadUseOpsPerIter
    // instead of kOpsPerIter. The uncached variant only runs on Quasar (l1_uncached_addr is a no-op
    // on BH/WH, so there is no second path to compare there).
    const auto arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
    std::vector<AluPassResult> load_use_passes = {cached_load};
    if (arch == tt::ARCH::QUASAR) {
        load_use_passes.push_back(uncached_load);
    }
    for (const AluPassResult& r : load_use_passes) {
        const uint32_t expected_instret = kLoadUseOpsPerIter * r.iterations + kFixedInstretOverhead;
        const double ipc = r.cycles == 0 ? 0.0 : static_cast<double>(r.instret) / r.cycles;
        const double cyc_per_iter = r.iterations == 0 ? 0.0 : static_cast<double>(r.cycles) / r.iterations;
        log_info(
            tt::LogTest,
            "ALU_IPC load_use iterations={} cycles={} instret={} (expected {}) ipc={:.3f} cyc/iter={:.3f}",
            r.iterations,
            r.cycles,
            r.instret,
            expected_instret,
            ipc,
            cyc_per_iter);
        if (r.instret != expected_instret) {
            log_error(
                tt::LogTest,
                "ALU_IPC load_use instret mismatch at iterations={}: got {}, expected {} -- harness bug, not a "
                "hardware measurement",
                r.iterations,
                r.instret,
                expected_instret);
            pass = false;
        }
    }
    if (arch == tt::ARCH::QUASAR && cached_load.cycles != 0) {
        const double uncached_vs_cached = static_cast<double>(uncached_load.cycles) / cached_load.cycles;
        log_info(
            tt::LogTest,
            "ALU_IPC load_use uncached/cached cycle ratio={:.4f} (>1.0 means the uncached alias costs more per "
            "load-use)",
            uncached_vs_cached);
    }

    return pass;
}

}  // namespace unit_tests::dm::compute_ipc

// =============================================================================
// Test Suite: Compute IPC
// =============================================================================

class ComputeIpcOps : public MeshDevice1x1Fixture {};

TEST_F(ComputeIpcOps, AluLoopIpc) {
    // Isolated ALU-only loop, no NoC/memory traffic: measures raw core IPC (rdinstret/rdcycle) on a
    // single DM/BRISC core, independent of the memory-system effects the FD copy benchmarks can't
    // separate out on their own.
    EXPECT_TRUE(unit_tests::dm::compute_ipc::run_alu_loop_ipc_bench(get_mesh_device()));
}

}  // namespace tt::tt_metal
