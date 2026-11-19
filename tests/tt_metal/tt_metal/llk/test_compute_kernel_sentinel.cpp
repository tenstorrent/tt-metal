// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <impl/context/metal_context.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>
#include <tt-logger/tt-logger.hpp>

#include "llk_device_fixture.hpp"

namespace tt::tt_metal {

using std::vector;
using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::compute::compute_kernel_sentinel {

struct ComputeKernelSentinelTestConfig {
    size_t num_tiles = 0;
    // Whether or not we want the result to be stored in DST in FP32 and/or
    // accumulated with previous DST value is controlled with this flag:
    bool fp32_dest_acc_en = false;
    // Whether or not to sync full/half DST between MATH and PACK:
    bool dst_full_sync_en = false;
};

/// @param test_config - Configuration of the test -- see struct
/// @return
bool single_core_compute_kernel_sentinel(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const ComputeKernelSentinelTestConfig& test_config) {
    ////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////
    constexpr uint32_t in0_id = 0;
    constexpr uint32_t in1_id = 1;
    constexpr uint32_t in2_id = 2;
    constexpr uint32_t out0_id = 16;
    constexpr uint32_t out1_id = 17;
    constexpr uint32_t TILE_DIM = 32;
    constexpr uint32_t TILE_ELEMENTS = TILE_DIM * TILE_DIM;
    constexpr uint32_t single_tile_size_fp32 = 4 * TILE_ELEMENTS;    // 4 bytes per element
    constexpr uint32_t single_tile_size_bfp16b = 2 * TILE_ELEMENTS;  // 2 bytes per element
    constexpr uint32_t single_tile_size_bfp8b = TILE_ELEMENTS + 64;  // 1 byte per element + exponent header
    uint32_t single_tile_size_out0 = test_config.fp32_dest_acc_en ? single_tile_size_fp32 : single_tile_size_bfp16b;
    const size_t dram_buffer_size_bfp16b = test_config.num_tiles * single_tile_size_bfp16b;
    const size_t dram_buffer_size_bfp8b = test_config.num_tiles * single_tile_size_bfp8b;
    const size_t dram_buffer_size_out0 = test_config.num_tiles * single_tile_size_out0;

    CoreCoord core = {0, 0};

    auto& cq = mesh_device->mesh_command_queue();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::MeshWorkload workload;
    tt_metal::Program program = tt_metal::CreateProgram(*mesh_device);
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);

    tt_metal::CircularBufferConfig l1_input0_cb_config =
        tt_metal::CircularBufferConfig(dram_buffer_size_bfp8b, {{in0_id, tt::DataFormat::Bfp8_b}})
            .set_page_size(in0_id, single_tile_size_bfp8b);
    tt_metal::CreateCircularBuffer(program_, core, l1_input0_cb_config);

    tt_metal::CircularBufferConfig l1_input1_cb_config =
        tt_metal::CircularBufferConfig(dram_buffer_size_bfp16b, {{in1_id, tt::DataFormat::Float16_b}})
            .set_page_size(in1_id, single_tile_size_bfp16b);
    tt_metal::CreateCircularBuffer(program_, core, l1_input1_cb_config);

    tt_metal::CircularBufferConfig l1_input2_cb_config =
        tt_metal::CircularBufferConfig(dram_buffer_size_bfp16b, {{in2_id, tt::DataFormat::Float16_b}})
            .set_page_size(in2_id, single_tile_size_bfp16b);
    tt_metal::CreateCircularBuffer(program_, core, l1_input2_cb_config);

    tt_metal::CircularBufferConfig l1_output0_cb_config =
        tt_metal::CircularBufferConfig(
            dram_buffer_size_out0,
            {{out0_id, (test_config.fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b)}})
            .set_page_size(out0_id, single_tile_size_out0);
    tt_metal::CreateCircularBuffer(program_, core, l1_output0_cb_config);

    tt_metal::CircularBufferConfig l1_output1_cb_config =
        tt_metal::CircularBufferConfig(dram_buffer_size_bfp8b, {{out1_id, tt::DataFormat::Bfp8_b}})
            .set_page_size(out1_id, single_tile_size_bfp8b);
    tt_metal::CreateCircularBuffer(program_, core, l1_output1_cb_config);

    vector<uint32_t> compute_kernel_args = {};
    std::map<std::string, std::string> defines;

    defines["TT_METAL_COMPUTE_KERNEL_SENTINEL_TESTING_ENABLED"] =
        "1";  // Define to enable compute kernel sentinel testing interface
    defines["FORCE_WATCHER_OFF"] = "1";
    defines["LIGHTWEIGHT_KERNEL_ASSERTS"] = "1";
    defines["TT_METAL_COMPUTE_KERNEL_SENTINEL_ENABLED"] = "1";
    defines["TT_METAL_COMPUTE_KERNEL_SENTINEL_DEFAULT_INJECTION"] = "1";
    defines["REDUCE_OP"] = "PoolType::SUM";
    defines["REDUCE_DIM"] = "ReduceDim::REDUCE_ROW";

    auto compute_kernel = tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/compute/compute_kernel_sentinel.cpp",
        core,
        tt_metal::ComputeConfig{
            .fp32_dest_acc_en = test_config.fp32_dest_acc_en,
            .dst_full_sync_en = test_config.dst_full_sync_en,
            .compile_args = compute_kernel_args,
            .defines = defines});

    SetRuntimeArgs(program_, compute_kernel, core, {});

    distributed::EnqueueMeshWorkload(cq, workload, true);

    // Always returns true. Failed lightweight asserts will hang the core, triggering triage timeout that dumps the
    // call stack. For local testing, run ./tools/triage/dump_lightweight_asserts.py when a kernel hangs to print
    // assert info and callstack.
    return true;
}
/// Quasar version over the 2.0 host API (compute_kernel_sentinel_2_0.cpp). The kernel only passes DFB ids to the
/// inits, so the data DFBs have no reader/writer; compute is both their producer and consumer. Instead of trapping
/// on the first failed check, the kernel reports every check through the `status` DFB.
bool single_core_compute_kernel_sentinel_quasar(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, std::uint32_t max_checks) {
    // PROBE: SENTINEL_PROBE_FMT=f16 swaps MxFp8P for Float16; SENTINEL_PROBE_INJECT=0 disables injection.
    const char* probe_fmt = std::getenv("SENTINEL_PROBE_FMT");
    const char* probe_inject = std::getenv("SENTINEL_PROBE_INJECT");
    const bool probe_f16 = probe_fmt != nullptr && std::string(probe_fmt) == "f16";
    const bool inject = probe_inject == nullptr || std::string(probe_inject) != "0";
    const tt::DataFormat mx_format = probe_f16 ? tt::DataFormat::Float32 : tt::DataFormat::MxFp8P;
    log_info(tt::LogTest, "PROBE fmt={} inject={}", mx_format, inject);
    const std::uint32_t status_tile_size = tt::tile_size(tt::DataFormat::Float32);
    const CoreCoord core = {0, 0};
    auto& cq = mesh_device->mesh_command_queue();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    const NodeCoord node{static_cast<std::uint32_t>(core.x), static_cast<std::uint32_t>(core.y)};

    auto status_dram = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = status_tile_size},
        distributed::DeviceLocalBufferConfig{
            .page_size = status_tile_size, .buffer_type = tt::tt_metal::BufferType::DRAM, .bottom_up = false},
        mesh_device.get());

    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};
    const std::vector<std::pair<std::string, tt::DataFormat>> data_dfbs = {
        {"in0", mx_format},
        {"in1", tt::DataFormat::Float16_b},
        {"in2", tt::DataFormat::Float16_b},
        {"out0", tt::DataFormat::Float16_b},
        {"out1", mx_format},
    };
    const DFBSpecName STATUS_DFB{"status"};

    std::vector<DataflowBufferSpec> dfb_specs;
    decltype(KernelSpec::dfb_bindings) compute_bindings;
    for (const auto& [name, format] : data_dfbs) {
        dfb_specs.push_back(DataflowBufferSpec{
            .unique_id = DFBSpecName{name},
            .entry_size = tt::tile_size(format),
            .num_entries = 1,
            .data_format_metadata = format,
        });
        for (auto endpoint : {DFBEndpointType::PRODUCER, DFBEndpointType::CONSUMER}) {
            compute_bindings.push_back(DFBBinding{
                .dfb_spec_name = DFBSpecName{name},
                .accessor_name = name,
                .endpoint_type = endpoint,
                .access_pattern = DFBAccessPattern::STRIDED,
            });
        }
    }
    dfb_specs.push_back(DataflowBufferSpec{
        .unique_id = STATUS_DFB,
        .entry_size = status_tile_size,
        .num_entries = 1,
        .data_format_metadata = tt::DataFormat::Float32,
    });
    compute_bindings.push_back(ProducerOf(STATUS_DFB, "status"));

    KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {ConsumerOf(STATUS_DFB, "in")},
        .runtime_arg_schema = {.runtime_arg_names = {"dst_addr", "bank_id", "num_tiles"}},
        .hw_config =
            DataMovementHardwareConfig{
                .config_2xx =
                    DataMovementHardwareConfig::DataMovement2XXConfig{
                        .disable_dfb_implicit_sync_for_all = true,
                    },
            },
    };
    KernelSpec::CompilerOptions::Defines compute_defines{
        {"TT_METAL_COMPUTE_KERNEL_SENTINEL_TESTING_ENABLED", "1"},
        {"TT_METAL_COMPUTE_KERNEL_SENTINEL_ENABLED", "1"},
        {"REDUCE_OP", "PoolType::SUM"},
        {"REDUCE_DIM", "ReduceDim::REDUCE_ROW"}};
    if (inject) {
        compute_defines["TT_METAL_COMPUTE_KERNEL_SENTINEL_DEFAULT_INJECTION"] = "1";
    }
    KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = "tests/tt_metal/tt_metal/test_kernels/compute/compute_kernel_sentinel_2_0.cpp",
        .num_threads = 1,
        .compiler_options = {.defines = compute_defines},
        .dfb_bindings = compute_bindings,
        .compile_time_args = {{"max_checks", max_checks}},
        .hw_config = ComputeHardwareConfig{},
    };
    ProgramSpec spec{
        .name = "compute_kernel_sentinel_quasar",
        .kernels = {writer_spec, compute_spec},
        .dataflow_buffers = dfb_specs,
        .work_units = {WorkUnitSpec{.name = "main", .kernels = {WRITER, COMPUTE}, .target_nodes = node}},
    };
    Program program(*mesh_device, spec);

    ProgramRunArgs params;
    params.kernel_run_args = {
        ProgramRunArgs::KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values = MakeRuntimeArgsForSingleNode(
                node,
                {{"dst_addr", static_cast<std::uint32_t>(status_dram->address())}, {"bank_id", 0u}, {"num_tiles", 1u}}),
        },
        ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE},
    };
    program.set_program_run_args(params);
    LaunchProgram(*mesh_device, std::move(program));

    std::vector<std::uint32_t> status;
    distributed::ReadShard(cq, status, status_dram, zero_coord, false);
    distributed::Finish(cq);

    const std::uint32_t num_checks = status.at(0);
    bool pass = num_checks > 0;
    for (std::uint32_t i = 0; i < num_checks && 1 + i < status.size(); ++i) {
        const std::uint32_t expected = (status[1 + i] >> 8) & 0xFF;
        const std::uint32_t recorded = status[1 + i] & 0xFF;
        if (expected != recorded) {
            pass = false;
            log_error(tt::LogTest, "check {}: expected reconfig bits {:#x}, recorded {:#x}", i, expected, recorded);
        }
    }
    log_info(tt::LogTest, "{} sentinel checks, {}", num_checks, pass ? "all match" : "mismatches above");
    return pass;
}
}  // namespace unit_tests::compute::compute_kernel_sentinel

////////////////////////////////////////////////////////////////////////////
//                             Test Description
// ------------------------------------------------------------------------
// These tests aim to cover usage of compute kernel sentinel API.
////////////////////////////////////////////////////////////////////////////

TEST_F(LLKMeshDeviceFixture, TensixComputeKernelSentinel) {
    if (this->arch_ == ARCH::QUASAR) {
        for (const auto& device : devices_) {
            for (std::uint32_t max_checks : {3u, 64u}) {
                log_info(tt::LogTest, "PROBE max_checks={}", max_checks);
                EXPECT_TRUE(unit_tests::compute::compute_kernel_sentinel::single_core_compute_kernel_sentinel_quasar(
                    device, max_checks));
            }
        }
        return;
    }
    unit_tests::compute::compute_kernel_sentinel::ComputeKernelSentinelTestConfig test_config = {
        .num_tiles = 1, .fp32_dest_acc_en = false, .dst_full_sync_en = false};

    for (const auto& device : devices_) {
        EXPECT_TRUE(
            unit_tests::compute::compute_kernel_sentinel::single_core_compute_kernel_sentinel(device, test_config));
    }
}

}  // namespace tt::tt_metal
