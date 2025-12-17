// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <impl/context/metal_context.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "device_fixture.hpp"

namespace tt::tt_metal {

using std::vector;
using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::compute::state_tracker {

struct StateTrackerTestConfig {
    size_t num_tiles = 0;
    // Whether or not we want the result to be stored in DST in FP32 and/or
    // accumulated with previous DST value is controlled with this flag:
    bool fp32_dest_acc_en = false;
    // Whether or not to sync full/half DST between MATH and PACK:
    bool dst_full_sync_en = false;
};

/// @param test_config - Configuration of the test -- see struct
/// @return
bool single_core_state_tracker(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const StateTrackerTestConfig& test_config) {
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
    tt_metal::Program program = tt_metal::CreateProgram();
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

    defines["TT_METAL_STATE_TRACKER_TESTING_ENABLED"] = "1";  // Define to enable state tracker testing interface
    defines["FORCE_WATCHER_OFF"] = "1";
    defines["TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS"] = "1";

    auto compute_kernel = tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/compute/state_tracker.cpp",
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
}  // namespace unit_tests::compute::state_tracker

////////////////////////////////////////////////////////////////////////////
//                             Test Description
// ------------------------------------------------------------------------
// These tests aim to cover usage of state tracker API.
////////////////////////////////////////////////////////////////////////////

TEST_F(MeshDeviceFixture, TensixComputeStateTracker) {
    unit_tests::compute::state_tracker::StateTrackerTestConfig test_config = {
        .num_tiles = 1, .fp32_dest_acc_en = false, .dst_full_sync_en = false};

    for (unsigned int id = 0; id < devices_.size(); id++) {
        EXPECT_TRUE(unit_tests::compute::state_tracker::single_core_state_tracker(devices_.at(id), test_config));
    }
}

}  // namespace tt::tt_metal
