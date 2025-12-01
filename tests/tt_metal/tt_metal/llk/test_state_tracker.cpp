// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <stdint.h>
#include <tt-metalium/bfloat8.hpp>
#include <bit>
#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <variant>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include "device_fixture.hpp"
#include "tests/tt_metal/tt_metal/debug_tools/debug_tools_fixture.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/packing.hpp"
#include <umd/device/types/arch.hpp>
#include <impl/debug/watcher_server.hpp>
#include <impl/context/metal_context.hpp>
#include "tt_metal/test_utils/bfloat_utils.hpp"

namespace tt {
namespace tt_metal {
class IDevice;
}  // namespace tt_metal
}  // namespace tt

namespace tt::tt_metal {

using std::vector;
using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::compute::state_tracker {

struct ReconfigConfig {
    size_t num_tiles = 0;
    // Number of tiles finished with single LLK API call:
    size_t ublock_size_tiles = 0;
    // Reconfig LLK API calls can either explicitly or implicitly take previous
    // CB indices; which version of the call is used is defined by this flag:
    bool explicit_reconfig = false;
    // Some reconfig calls are joined for SrcA/B; whether split or joined calls
    // are used is defined with this flag:
    bool split_src_reconfig = false;
    // This flag defines whether regular packing to L1 is used, or the one
    // where the result is accumulated with the previous value:
    bool l1_acc = false;
    // Whether or not we want the result to be stored in DST in FP32 and/or
    // accumulated with previous DST value is controlled with this flag:
    bool fp32_dest_acc_en = false;
    // Whether to test with copy_tile or copy_block_matmul_partials is contro-
    // lled with this flag:
    bool block_copy = true;
    // Whether or not to sync full/half DST between MATH and PACK:
    bool dst_full_sync_en = false;
};

using VariantVectorType = std::variant<std::vector<float>, std::vector<bfloat16>>;

/// @param test_config - Configuration of the test -- see struct
/// @return
bool single_core_state_tracker(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const ReconfigConfig& test_config) {
    ////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////
    bool pass = true;
    uint32_t in0_id = 0;
    uint32_t in1_id = 1;
    uint32_t in2_id = 2;
    uint32_t out0_id = 16;
    uint32_t out1_id = 17;
    uint32_t single_tile_size_fp32 = 4 * 32 * 32;          // Single 32x32 tile size for Float32
    uint32_t single_tile_size_bfp16b = 2 * 32 * 32;        // Single 32x32 tile size for Float16_b
    uint32_t single_tile_size_bfp8b = (1 * 32 * 32) + 64;  // Single 32x32 tile size for Bfp8_b
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

    defines["DST_ACCUM_MODE"] = "1";  // Needed always in order for reader kernel to load data from CB2
    defines["EXPLICIT_RECONFIG"] = test_config.explicit_reconfig ? "1" : "0";
    defines["SPLIT_SRC_RECONFIG"] = test_config.split_src_reconfig ? "1" : "0";
    defines["BLOCK_COPY"] = test_config.block_copy ? "1" : "0";
    defines["L1_ACC"] = test_config.l1_acc ? "1" : "0";

    auto compute_kernel = tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/compute/state_tracker.cpp",
        core,
        tt_metal::ComputeConfig{
            .fp32_dest_acc_en = test_config.fp32_dest_acc_en,
            .dst_full_sync_en = test_config.dst_full_sync_en,
            .compile_args = compute_kernel_args,
            .defines = defines});

    SetRuntimeArgs(
        program_,
        compute_kernel,
        core,
        {
            uint32_t(test_config.num_tiles),
            uint32_t(test_config.ublock_size_tiles),
        });

    // Enqueue the workload without blocking. The watcher will detect errors asynchronously.
    distributed::EnqueueMeshWorkload(cq, workload, false);

    // // Wait for either completion or watcher error detection
    // // Poll watcher status while waiting for the kernel to finish or trip an assert
    // const int max_poll_iterations = 1000;  // ~10 seconds with 10ms sleeps
    // int poll_count = 0;
    // bool watcher_error = false;

    // while (poll_count < max_poll_iterations) {
    //     // Check if watcher detected an error
    //     if (MetalContext::instance().watcher_server()->killed_due_to_error()) {
    //         watcher_error = true;
    //         log_error(
    //             LogTest, "Kernel execution stopped due to watcher detecting an error. See watcher log for details.");
    //         break;
    //     }

    //     // Small sleep to avoid busy-waiting and let watcher run
    //     std::this_thread::sleep_for(std::chrono::milliseconds(10));
    //     poll_count++;
    // }
    // log_info(LogTest, "Completed polling for watcher errors after {} iterations.", poll_count);
    // pass = !watcher_error;
    try {
        distributed::Finish(cq);
        // if (MetalContext::instance().watcher_server()->killed_due_to_error()) {
        //     log_error(
        //         LogTest, "Kernel execution stopped due to watcher detecting an error. See watcher log for details.");
        //     pass = false;
        // }
    } catch (const std::exception& e) {
        log_error(LogTest, "Kernel execution failed with exception: {}", e.what());
        pass = false;
    }
    log_info(LogTest, "HERE IT IS");
    return pass;
}
}  // namespace unit_tests::compute::state_tracker

////////////////////////////////////////////////////////////////////////////
//                             Test Description
// ------------------------------------------------------------------------
// These tests aim to cover usage of these API calls:
// - copy_tile_init
// - copy_tile_to_dst_init_short
// - copy_tile_to_dst_init_short_with_dt
// - unpack_reconfig_data_format
// - unpack_reconfig_data_format_srca
// - unpack_reconfig_data_format_srcb
// - pack_reconfig_l1_acc
////////////////////////////////////////////////////////////////////////////
TEST_F(MeshWatcherFixture, TensixComputeStateTracker) {
    auto arch = this->arch_;
    if (arch == tt::ARCH::GRAYSKULL) {
        GTEST_SKIP();
    }

    if (this->slow_dispatch_) {
        log_info(tt::LogTest, "Test requires fast dispatch - skipping in slow dispatch mode");
        GTEST_SKIP();
    }

    // Test that should pass
    unit_tests::compute::state_tracker::ReconfigConfig test_config = {
        .num_tiles = 1,
        .ublock_size_tiles = 1,
        .explicit_reconfig = true,
        .split_src_reconfig = false,
        .fp32_dest_acc_en = false,
        .block_copy = true,
        .dst_full_sync_en = false};

    tt::tt_metal::MetalContext::instance().rtoptions().set_test_mode_enabled(false);
    for (unsigned int id = 0; id < devices_.size(); id++) {
        ASSERT_TRUE(unit_tests::compute::state_tracker::single_core_state_tracker(devices_.at(id), test_config));
    }
}

}  // namespace tt::tt_metal
