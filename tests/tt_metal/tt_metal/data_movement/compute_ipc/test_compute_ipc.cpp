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
// 4x(lw+add) + addi + bnez, and the same plus fence/sd/fence; kept in sync with alu_loop_ipc.cpp.
constexpr uint32_t kMultiLoadOpsPerIter = 10;
constexpr uint32_t kInvalLoadOpsPerIter = 13;
constexpr uint32_t kInvalOnlyOpsPerIter = 5;  // fence + sd + fence + addi + bnez
constexpr uint32_t kNumResultWords = 27;
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

    // The 4-load passes read offsets 0/4/8/12 and the invalidate pass invalidates the line holding
    // the base, so all four reads must land in that same 64 B line or the measurement is meaningless
    // (it would take extra misses the real header decode wouldn't). Any 16 B-aligned base satisfies
    // this; check rather than assume the allocator's alignment.
    if (load_src_base % 64 > 48) {
        log_error(
            tt::LogTest,
            "load_src_base {:#x} straddles a 64B cache line across offsets 0-15 (base%64={}); the 4-load "
            "passes would measure extra misses",
            load_src_base,
            load_src_base % 64);
        return false;
    }

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
    const AluPassResult multi_cached{
        .iterations = result_data[12], .cycles = result_data[13], .instret = result_data[14]};
    const AluPassResult multi_uncached{
        .iterations = result_data[15], .cycles = result_data[16], .instret = result_data[17]};
    const AluPassResult inval_cached{
        .iterations = result_data[18], .cycles = result_data[19], .instret = result_data[20]};
    const AluPassResult inval_only{
        .iterations = result_data[21], .cycles = result_data[22], .instret = result_data[23]};
    const AluPassResult full_inval{
        .iterations = result_data[24], .cycles = result_data[25], .instret = result_data[26]};

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

    // 4-load passes: model one command-header field decode (4 reads inside one 64 B line). The
    // question they answer is whether "invalidate the line once, then read cached" beats "read
    // uncached", which is what FD does today for every cmd-> field access.
    struct MultiPass {
        const char* name;
        AluPassResult r;
        uint32_t ops_per_iter;
    };
    std::vector<MultiPass> multi_passes = {{"multi_cached", multi_cached, kMultiLoadOpsPerIter}};
    if (arch == tt::ARCH::QUASAR) {
        multi_passes.push_back({"multi_uncached", multi_uncached, kMultiLoadOpsPerIter});
        multi_passes.push_back({"inval_then_cached", inval_cached, kInvalLoadOpsPerIter});
        multi_passes.push_back({"inval_only", inval_only, kInvalOnlyOpsPerIter});
    }
    for (const MultiPass& p : multi_passes) {
        const uint32_t expected_instret = p.ops_per_iter * p.r.iterations + kFixedInstretOverhead;
        const double cyc_per_iter = p.r.iterations == 0 ? 0.0 : static_cast<double>(p.r.cycles) / p.r.iterations;
        log_info(
            tt::LogTest,
            "ALU_IPC {} iterations={} cycles={} instret={} (expected {}) cyc/iter={:.3f}",
            p.name,
            p.r.iterations,
            p.r.cycles,
            p.r.instret,
            expected_instret,
            cyc_per_iter);
        if (p.r.instret != expected_instret) {
            log_error(
                tt::LogTest,
                "ALU_IPC {} instret mismatch: got {}, expected {} -- harness bug, not a hardware measurement",
                p.name,
                p.r.instret,
                expected_instret);
            pass = false;
        }
    }
    if (arch == tt::ARCH::QUASAR && multi_cached.iterations != 0 && multi_uncached.iterations != 0 &&
        inval_cached.iterations != 0) {
        const double cached_per_iter = static_cast<double>(multi_cached.cycles) / multi_cached.iterations;
        const double uncached_per_iter = static_cast<double>(multi_uncached.cycles) / multi_uncached.iterations;
        const double inval_per_iter = static_cast<double>(inval_cached.cycles) / inval_cached.iterations;
        // Isolated invalidate cost: the two passes differ only by fence/sd/fence, so the delta is the
        // primitive's own cost (plus 3 instructions' worth of issue, ~3 cyc).
        const double invalidate_cost = inval_per_iter - cached_per_iter;
        log_info(
            tt::LogTest,
            "ALU_IPC DECISION: 4 reads uncached={:.1f} cyc vs invalidate+4-cached={:.1f} cyc -> {} "
            "(isolated invalidate cost ~{:.1f} cyc; break-even at {:.1f})",
            uncached_per_iter,
            inval_per_iter,
            inval_per_iter < uncached_per_iter ? "CACHED WINS" : "UNCACHED WINS",
            invalidate_cost,
            uncached_per_iter - cached_per_iter);

        // Split the invalidate+refetch cost into the primitive itself vs the cold refetch. A slow
        // primitive could be replaced with a cheaper one; a slow refetch could not.
        if (inval_only.iterations != 0) {
            const double inval_only_per_iter = static_cast<double>(inval_only.cycles) / inval_only.iterations;
            // inval_only's loop is 5 instrs vs multi_cached's 10, so compare against the invalidate
            // sequence's share rather than the raw difference.
            log_info(
                tt::LogTest,
                "ALU_IPC SPLIT: invalidate primitive alone={:.1f} cyc/iter (5 instrs, no refetch); "
                "invalidate+refetch+4-cached total={:.1f}; so the cold refetch accounts for ~{:.1f} cyc",
                inval_only_per_iter,
                inval_per_iter,
                inval_per_iter - inval_only_per_iter - cached_per_iter);
        }

        // Full-cache invalidate vs per-line, same 4 cached reads. Instret is NOT checked for this pass:
        // its completion poll makes the count vary, so it is reported as an observation instead. A count
        // near the floor (13*iters+3) means the poll cleared immediately; much larger means it spun.
        if (full_inval.iterations != 0) {
            const double full_per_iter = static_cast<double>(full_inval.cycles) / full_inval.iterations;
            const double instr_per_iter = static_cast<double>(full_inval.instret) / full_inval.iterations;
            log_info(
                tt::LogTest,
                "ALU_IPC full_inval iterations={} cycles={} instret={} ({:.1f} instr/iter -- varies with the "
                "completion poll; ~13 means it cleared immediately) cyc/iter={:.1f} vs per-line {:.1f} -> {}",
                full_inval.iterations,
                full_inval.cycles,
                full_inval.instret,
                instr_per_iter,
                full_per_iter,
                inval_per_iter,
                full_per_iter < inval_per_iter ? "FULL INVALIDATE CHEAPER" : "PER-LINE CHEAPER");
        }
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
