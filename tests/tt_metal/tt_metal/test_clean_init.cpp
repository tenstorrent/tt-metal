// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <fmt/base.h>
#include <stdint.h>
#include <stdlib.h>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/device_pool.hpp>
#include <tt-metalium/host_api.hpp>
#include <array>
#include <exception>
#include <map>
#include <memory>
#include <variant>
#include <vector>

#include <tt-metalium/assert.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/kernel_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include "impl/context/metal_context.hpp"

namespace tt {
namespace tt_metal {
class CommandQueue;
}  // namespace tt_metal
}  // namespace tt

/*
 * Similar to loopback programming example, except run on al devices and skip device teardown to check if we can
 * recover from a "bad" state.
 */

using std::vector;
using namespace tt::tt_metal;

int main(int argc, char** argv) {
    if (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
        TT_THROW("Test not supported w/ slow dispatch, exiting");
    }

    // Any arg means that we shouldn't do teardown.
    bool skip_teardown = (argc > 1);
    if (skip_teardown) {
        log_info(tt::LogTest, "Running loopback test with no teardown, to see if we can recover next run.");
    } else {
        log_info(tt::LogTest, "Running loopback test with proper teardown");
    }

    bool pass = true;
    auto num_devices = tt::tt_metal::GetNumAvailableDevices();
    vector<chip_id_t> ids;
    for (unsigned int id = 0; id < num_devices; id++) {
        ids.push_back(id);
    }

    const auto& dispatch_core_config = tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config();
    tt::DevicePool::initialize(ids, 1, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, dispatch_core_config);
    const auto devices = tt::DevicePool::instance().get_all_active_devices();

    for (int device_id = 0; device_id < num_devices; device_id++) {
        try {
            /*
            * Silicon accelerator setup
            */
            IDevice* device = devices[device_id];

            /*
            * Setup program and command queue to execute along with its buffers and kernels to use
            */
            CommandQueue& cq = device->command_queue();
            Program program = CreateProgram();

            constexpr CoreCoord core = {0, 0};

            KernelHandle dram_copy_kernel_id = CreateKernel(
                program,
                "tt_metal/programming_examples/loopback/kernels/loopback_dram_copy.cpp",
                core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
            );

            constexpr uint32_t single_tile_size = 2 * (32 * 32);
            constexpr uint32_t num_tiles = 50;
            constexpr uint32_t dram_buffer_size = single_tile_size * num_tiles;

            tt::tt_metal::InterleavedBufferConfig dram_config{
                        .device= device,
                        .size = dram_buffer_size,
                        .page_size = dram_buffer_size,
                        .buffer_type = tt::tt_metal::BufferType::DRAM
            };
            tt::tt_metal::InterleavedBufferConfig l1_config{
                        .device= device,
                        .size = dram_buffer_size,
                        .page_size = dram_buffer_size,
                        .buffer_type = tt::tt_metal::BufferType::L1
            };

            auto l1_buffer = CreateBuffer(l1_config);

            auto input_dram_buffer = CreateBuffer(dram_config);
            const uint32_t input_dram_buffer_addr = input_dram_buffer->address();

            auto output_dram_buffer = CreateBuffer(dram_config);
            const uint32_t output_dram_buffer_addr = output_dram_buffer->address();

            /*
            * Create input data and runtime arguments, then execute
            */
            std::vector<uint32_t> input_vec = create_random_vector_of_bfloat16(
                dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
            EnqueueWriteBuffer(cq, input_dram_buffer, input_vec, false);

            const std::array<uint32_t, 8> runtime_args = {
                l1_buffer->address(),
                input_dram_buffer->address(),
                output_dram_buffer->address(),
                l1_buffer->size(),
                num_tiles};

            SetRuntimeArgs(
                program,
                dram_copy_kernel_id,
                core,
                runtime_args
            );

            EnqueueProgram(cq, program, false);
            log_info(tt::LogTest, "Started program");
            Finish(cq);
            log_info(tt::LogTest, "Finished program");

            /*
            * Validation & Teardown
            */
            std::vector<uint32_t> result_vec;
            EnqueueReadBuffer(cq, output_dram_buffer, result_vec, true);

            pass &= input_vec == result_vec;

        } catch (const std::exception &e) {
            log_error(tt::LogTest, "Test failed with exception!");
            log_error(tt::LogTest, "{}", e.what());

            throw;
        }
    }

    if (pass) {
        log_info(tt::LogTest, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }

    // Skip teardown by throwing.
    if (skip_teardown) {
        TT_THROW("Skip teardown by throwing");
    } else {
        for (auto device : devices) {
            pass &= CloseDevice(device);
        }
    }

    // Error out with non-zero return code if we don't detect a pass
    TT_FATAL(pass, "Error");

    return 0;
}
