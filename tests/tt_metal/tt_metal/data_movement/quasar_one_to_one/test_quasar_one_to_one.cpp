// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Quasar one-to-one DM perf scaffold. The kernel currently just DPRINTs
// "Hello, World!"; perf scenarios will be added on top.

#include <cstdint>
#include <fstream>
#include "device_fixture.hpp"
#include "dm_common.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-logger/tt-logger.hpp>

namespace tt::tt_metal {

using namespace tt::test_utils;

namespace unit_tests::dm::quasar_one_to_one {

constexpr auto kHelloWorldKernel = "tests/tt_metal/tt_metal/data_movement/quasar_one_to_one/kernels/hello_world.cpp";
constexpr auto kAttWritePerfKernel =
    "tests/tt_metal/tt_metal/data_movement/quasar_one_to_one/kernels/att_write_perf.cpp";
constexpr auto kAttWriteNonPostedPerfKernel =
    "tests/tt_metal/tt_metal/data_movement/quasar_one_to_one/kernels/att_write_nonposted_perf.cpp";
constexpr auto kDivModPerfKernel = "tests/tt_metal/tt_metal/data_movement/quasar_one_to_one/kernels/div_mod_perf.cpp";

bool should_skip_test() {
    const auto arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
    if (arch != tt::ARCH::QUASAR) {
        return true;
    }
    return std::getenv("TT_METAL_SIMULATOR") == nullptr;
}

bool run_hello_world(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    constexpr const char* DM_KERNEL = "hello_world";
    const experimental::metal2_host_api::NodeCoord node{0, 0};

    experimental::metal2_host_api::KernelSpec dm_kernel_spec{
        .unique_id = DM_KERNEL,
        .source = experimental::metal2_host_api::KernelSpec::SourceFilePath{kHelloWorldKernel},
        .num_threads = 1,
        .config_spec =
            experimental::metal2_host_api::DataMovementConfiguration{
                .gen2_data_movement_config =
                    experimental::metal2_host_api::DataMovementConfiguration::Gen2DataMovementConfig{}},
    };

    experimental::metal2_host_api::WorkUnitSpec main_wu{
        .unique_id = "main",
        .kernels = {DM_KERNEL},
        .target_nodes = node,
    };

    experimental::metal2_host_api::ProgramSpec spec{
        .program_id = "quasar_one_to_one_hello",
        .kernels = {dm_kernel_spec},
        .work_units = {main_wu},
    };
    Program program = experimental::metal2_host_api::MakeProgramFromSpec(*mesh_device, spec);

    experimental::metal2_host_api::ProgramRunParams params;
    params.kernel_run_params = {{
        .kernel_spec_name = DM_KERNEL,
    }};
    experimental::metal2_host_api::SetProgramRunParameters(program, params);

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());
    workload.add_program(device_range, std::move(program));

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, workload, true);

    return true;
}

// Shared implementation used by both AttWritePerf (posted) and
// AttWriteNonPostedPerf (non-posted). The two only differ in which kernel
// source is compiled and which program label appears in logs - the host-side
// seeding, the logical->physical coord conversion, and the validity readback
// are identical.
bool run_att_write_perf_impl(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const char* kernel_path,
    const char* unique_id,
    const char* program_id,
    const char* label) {
    constexpr uint32_t kSrcAddr = 0x10000;
    constexpr uint32_t kDstAddr = 0x20000;
    constexpr uint32_t kMasterLogicalX = 0;
    constexpr uint32_t kMasterLogicalY = 0;
    constexpr uint32_t kDstLogicalX = 1;
    constexpr uint32_t kDstLogicalY = 0;
    constexpr uint32_t kPayloadBytes = 16;
    constexpr uint32_t kNumIters = 1000;

    // The kernel writes data via the NOC and configures ATT endpoint entries
    // using physical NOC coordinates, so convert from logical here.
    IDevice* device = mesh_device->get_devices()[0];
    const CoreCoord master_core{kMasterLogicalX, kMasterLogicalY};
    const CoreCoord dst_core{kDstLogicalX, kDstLogicalY};
    const CoreCoord master_phys = device->worker_core_from_logical_core(master_core);
    const CoreCoord dst_phys = device->worker_core_from_logical_core(dst_core);
    log_info(
        tt::LogTest,
        "{}: master logical ({},{}) -> physical ({},{}); dst logical ({},{}) -> physical ({},{})",
        label,
        master_core.x,
        master_core.y,
        master_phys.x,
        master_phys.y,
        dst_core.x,
        dst_core.y,
        dst_phys.x,
        dst_phys.y);

    const experimental::metal2_host_api::NodeCoord node{kMasterLogicalX, kMasterLogicalY};

    experimental::metal2_host_api::KernelSpec dm_kernel_spec{
        .unique_id = unique_id,
        .source = experimental::metal2_host_api::KernelSpec::SourceFilePath{kernel_path},
        .num_threads = 6,
        .compile_time_arg_bindings =
            {{"src_addr", kSrcAddr},
             {"dst_addr", kDstAddr},
             {"dst_x", static_cast<uint32_t>(dst_phys.x)},
             {"dst_y", static_cast<uint32_t>(dst_phys.y)},
             {"master_x", static_cast<uint32_t>(master_phys.x)},
             {"master_y", static_cast<uint32_t>(master_phys.y)},
             {"payload_bytes", kPayloadBytes},
             {"num_iters", kNumIters}},
        .config_spec =
            experimental::metal2_host_api::DataMovementConfiguration{
                .gen2_data_movement_config =
                    experimental::metal2_host_api::DataMovementConfiguration::Gen2DataMovementConfig{}},
    };

    experimental::metal2_host_api::WorkUnitSpec main_wu{
        .unique_id = "main",
        .kernels = {unique_id},
        .target_nodes = node,
    };

    experimental::metal2_host_api::ProgramSpec spec{
        .program_id = program_id,
        .kernels = {dm_kernel_spec},
        .work_units = {main_wu},
    };
    Program program = experimental::metal2_host_api::MakeProgramFromSpec(*mesh_device, spec);

    experimental::metal2_host_api::ProgramRunParams params;
    params.kernel_run_params = {{
        .kernel_spec_name = unique_id,
    }};
    experimental::metal2_host_api::SetProgramRunParameters(program, params);

    // Seed master's source L1 with a known pattern and zero the destination so
    // we can detect whether the writes (any phase) landed at all.
    const uint32_t num_words = kPayloadBytes / sizeof(uint32_t);
    std::vector<uint32_t> pattern(num_words);
    for (uint32_t i = 0; i < num_words; i++) {
        pattern[i] = 0xCAFE0000u | i;
    }
    std::vector<uint32_t> zeros(num_words, 0);
    tt_metal::detail::WriteToDeviceL1(device, master_core, kSrcAddr, pattern);
    tt_metal::detail::WriteToDeviceL1(device, dst_core, kDstAddr, zeros);
    MetalContext::instance().get_cluster().l1_barrier(device->id());

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());
    workload.add_program(device_range, std::move(program));

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, workload, true);

    // After both phases all 6 DMs have repeatedly written the pattern to
    // dst_core/kDstAddr. The last write wins, so dst should equal pattern
    // if either phase actually transferred data.
    std::vector<uint32_t> readback;
    tt_metal::detail::ReadFromDeviceL1(device, dst_core, kDstAddr, kPayloadBytes, readback);

    if (readback != pattern) {
        log_error(
            tt::LogTest,
            "{} validity check failed: destination L1 at logical ({},{})+0x{:x} did not "
            "match the seeded pattern",
            label,
            kDstLogicalX,
            kDstLogicalY,
            kDstAddr);
        log_info(tt::LogTest, "expected ({} words):", num_words);
        for (uint32_t i = 0; i < num_words; i++) {
            log_info(tt::LogTest, "  [{}] 0x{:08x}", i, pattern[i]);
        }
        log_info(tt::LogTest, "got:");
        for (uint32_t i = 0; i < readback.size(); i++) {
            log_info(tt::LogTest, "  [{}] 0x{:08x}", i, readback[i]);
        }
        return false;
    }
    return true;
}

bool run_att_write_perf(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    return run_att_write_perf_impl(
        mesh_device, kAttWritePerfKernel, "att_write_perf", "quasar_one_to_one_att_write_perf", "AttWritePerf");
}

bool run_att_write_nonposted_perf(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    return run_att_write_perf_impl(
        mesh_device,
        kAttWriteNonPostedPerfKernel,
        "att_write_nonposted_perf",
        "quasar_one_to_one_att_write_nonposted_perf",
        "AttWriteNonPostedPerf");
}

bool run_div_mod_perf(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    constexpr const char* DM_KERNEL = "div_mod_perf";
    constexpr uint32_t kNumIters = 1000;
    // Must match kResultsBase in kernels/div_mod_perf.cpp.
    constexpr uint32_t kResultsBase = 0x30000;
    // Generous - 4 KB worth of records covers ~1000 measurements at 16 bytes each.
    constexpr uint32_t kResultsBytes = 16 * 1024;
    // Record kinds; must match the kernel's constants.
    constexpr uint32_t kKindBaseline = 0;
    constexpr uint32_t kKindStaticDiv = 1;
    constexpr uint32_t kKindStaticMod = 2;
    constexpr uint32_t kKindRuntimeDiv = 3;
    constexpr uint32_t kKindRuntimeMod = 4;

    const experimental::metal2_host_api::NodeCoord node{0, 0};

    experimental::metal2_host_api::KernelSpec dm_kernel_spec{
        .unique_id = DM_KERNEL,
        .source = experimental::metal2_host_api::KernelSpec::SourceFilePath{kDivModPerfKernel},
        .num_threads = 1,
        .compile_time_arg_bindings = {{"num_iters", kNumIters}},
        .config_spec =
            experimental::metal2_host_api::DataMovementConfiguration{
                .gen2_data_movement_config =
                    experimental::metal2_host_api::DataMovementConfiguration::Gen2DataMovementConfig{}},
    };

    experimental::metal2_host_api::WorkUnitSpec main_wu{
        .unique_id = "main",
        .kernels = {DM_KERNEL},
        .target_nodes = node,
    };

    experimental::metal2_host_api::ProgramSpec spec{
        .program_id = "quasar_one_to_one_div_mod_perf",
        .kernels = {dm_kernel_spec},
        .work_units = {main_wu},
    };
    Program program = experimental::metal2_host_api::MakeProgramFromSpec(*mesh_device, spec);

    experimental::metal2_host_api::ProgramRunParams params;
    params.kernel_run_params = {{
        .kernel_spec_name = DM_KERNEL,
    }};
    experimental::metal2_host_api::SetProgramRunParameters(program, params);

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());
    workload.add_program(device_range, std::move(program));

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, workload, true);

    // Read the records buffer the kernel populated and dump it as CSV.
    IDevice* device = mesh_device->get_devices()[0];
    const CoreCoord master_core{0, 0};
    std::vector<uint32_t> raw;
    tt_metal::detail::ReadFromDeviceL1(device, master_core, kResultsBase, kResultsBytes, raw);
    if (raw.empty()) {
        log_error(tt::LogTest, "DivModPerf: read 0 words from results buffer");
        return false;
    }
    const uint32_t count = raw[0];
    log_info(tt::LogTest, "DivModPerf: {} records in L1 results buffer", count);
    if (count == 0 || count > 4096) {
        log_error(tt::LogTest, "DivModPerf: implausible record count {} - aborting CSV dump", count);
        return false;
    }
    if (1 + count * 4 > raw.size()) {
        log_error(tt::LogTest, "DivModPerf: record count {} overruns the {}-word readback", count, raw.size());
        return false;
    }

    const std::string csv_path = "/tmp/divmod_perf_results.csv";
    std::ofstream csv(csv_path);
    if (!csv.is_open()) {
        log_error(tt::LogTest, "DivModPerf: failed to open {} for write", csv_path);
        return false;
    }
    csv << "kind,op,n,d,cycles\n";
    for (uint32_t i = 0; i < count; i++) {
        const uint32_t kind = raw[1 + i * 4 + 0];
        const uint32_t n = raw[1 + i * 4 + 1];
        const uint32_t d = raw[1 + i * 4 + 2];
        const uint32_t cyc = raw[1 + i * 4 + 3];
        const char* kind_str = "UNKNOWN";
        const char* op_str = "-";
        switch (kind) {
            case kKindBaseline:
                kind_str = "BASELINE";
                op_str = "-";
                break;
            case kKindStaticDiv:
                kind_str = "STATIC";
                op_str = "DIV";
                break;
            case kKindStaticMod:
                kind_str = "STATIC";
                op_str = "MOD";
                break;
            case kKindRuntimeDiv:
                kind_str = "RUNTIME";
                op_str = "DIV";
                break;
            case kKindRuntimeMod:
                kind_str = "RUNTIME";
                op_str = "MOD";
                break;
            default: break;
        }
        csv << kind_str << "," << op_str << "," << n << "," << d << "," << cyc << "\n";
    }
    csv.close();
    log_info(tt::LogTest, "DivModPerf: wrote {} rows to {}", count, csv_path);
    return true;
}

}  // namespace unit_tests::dm::quasar_one_to_one

// =============================================================================
// Test Suite: Quasar One-to-One
// =============================================================================

class QuasarOneToOneOps : public MeshDeviceSingleCardFixture {};

TEST_F(QuasarOneToOneOps, HelloWorld) {
    if (unit_tests::dm::quasar_one_to_one::should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }
    EXPECT_TRUE(unit_tests::dm::quasar_one_to_one::run_hello_world(devices_[0]));
}

TEST_F(QuasarOneToOneOps, AttWritePerf) {
    if (unit_tests::dm::quasar_one_to_one::should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }
    // Master tile DMs issue posted NOC writes to dst tile in two phases:
    //   Phase 1: ATT-translated address (endpoint id -> physical coord)
    //   Phase 2: physical (x,y) directly
    // Kernel DPRINTs per-issue cycles for each phase; compare to estimate the
    // DM stall introduced by ATT translation in the source-side issue path.
    EXPECT_TRUE(unit_tests::dm::quasar_one_to_one::run_att_write_perf(devices_[0]));
}

TEST_F(QuasarOneToOneOps, AttWriteNonPostedPerf) {
    if (unit_tests::dm::quasar_one_to_one::should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }
    // Same shape as AttWritePerf but using non-posted writes. The drain waits
    // for the destination ack inside the measurement window, so per-issue
    // cycles include end-to-end round-trip latency. Comparing ATT-on vs
    // ATT-off here pins down whether ATT adds round-trip latency, not just
    // source-side issue stall.
    EXPECT_TRUE(unit_tests::dm::quasar_one_to_one::run_att_write_nonposted_perf(devices_[0]));
}

TEST_F(QuasarOneToOneOps, DivModPerf) {
    if (unit_tests::dm::quasar_one_to_one::should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator";
    }
    // Microbenchmark: per-op cycle cost of 32-bit integer divide and modulus
    // on a single DM (RISC-V), swept across pow2 and non-pow2 divisors with
    // both compile-time-constant and forced-runtime divisor inputs. Reports
    // baseline loop overhead so the actual div/mod cost can be derived by
    // subtracting baseline from each (op, divisor, mode) measurement.
    EXPECT_TRUE(unit_tests::dm::quasar_one_to_one::run_div_mod_perf(devices_[0]));
}

}  // namespace tt::tt_metal
