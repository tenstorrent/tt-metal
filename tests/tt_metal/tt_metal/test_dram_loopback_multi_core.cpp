// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "common/bfloat16.hpp"
#include "tt_metal/test_utils/deprecated/tensor.hpp"

//////////////////////////////////////////////////////////////////////////////////////
// 1. Host writes data to buffer in DRAM
// 2. dram_loader_sync kernel on logical core {0, 0} BRISC copies data from buffer
//      in step 1. to buffer in L1
// 3. remote_read_remote_write_sync kernel on logical core {0, 1} NCRISC copies data
//      from L1 buffer on core {0, 0} to L1 buffer on core {0, 1}
// 4. remote_read_remote_write_sync copies data from L1 buffer to buffer in DRAM
// 5. Host reads from buffer written to in step 4.
//////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

int main(int argc, char **argv) {
    bool pass = true;

    auto slow_dispatch_mode = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    TT_FATAL(slow_dispatch_mode, "This test only supports TT_METAL_SLOW_DISPATCH_MODE");

    try {

        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id
        tt_metal::Device *device =
            tt_metal::CreateDevice(device_id);



        ////////////////////////////////////////////////////////////////////////////
        //                      Input Data Setup
        ////////////////////////////////////////////////////////////////////////////
        std::array<uint32_t, 4> shape = {1, 1, 32, 1024 * 32};

        uint32_t seed_from_systime = std::chrono::system_clock::now().time_since_epoch().count();
        Tensor<bfloat16> tensor = initialize_tensor<bfloat16>(
            shape, Initialize::RANDOM, 100, seed_from_systime);  // TODO: make randomized!
        auto golden = tensor.get_values();
        auto src_vec = pack_bfloat16_vec_into_uint32_vec(golden);

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::Program program = tt_metal::CreateProgram();

        CoreCoord loader_logical_core = {0, 0};
        CoreCoord writer_logical_core = {0, 1};
        auto loader_worker_core = device->worker_core_from_logical_core(loader_logical_core);
        auto writer_worker_core = device->worker_core_from_logical_core(writer_logical_core);

        uint32_t single_tile_size = 2 * 1024;
        uint32_t num_input_tiles = 1024 * 1;
        uint32_t num_output_tiles = num_input_tiles;
        uint32_t dram_buffer_size =
            single_tile_size * num_output_tiles;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels
        uint32_t loader_buffer_address = 500 * 1024;
        uint32_t writer_buffer_address = 500 * 1024;
        uint32_t transient_buffer_size_tiles = 4;
        uint32_t transient_buffer_size_bytes = transient_buffer_size_tiles * single_tile_size;
        std::uint32_t stream_register_address = STREAM_REG_ADDR(0, 12);

        TT_FATAL(num_output_tiles % transient_buffer_size_tiles == 0, "Error");

        tt_metal::InterleavedBufferConfig dram_config{
                                .device=device,
                                .size = dram_buffer_size,
                                .page_size = dram_buffer_size,
                                .buffer_type = tt_metal::BufferType::DRAM
                                };

        auto input_dram_buffer = CreateBuffer(dram_config);
        uint32_t dram_buffer_src_addr = input_dram_buffer->address();

        auto output_dram_buffer = CreateBuffer(dram_config);
        uint32_t dram_buffer_dst_addr = output_dram_buffer->address();

        auto input_dram_noc_xy = input_dram_buffer->noc_coordinates();
        auto output_dram_noc_xy = output_dram_buffer->noc_coordinates();

        // Loader (producer kernel) running on BRISC on logical core {0, 0}
        auto producer_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_loader_sync.cpp",
            loader_logical_core,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

        // Writer (consumer kernel) running on NCRISC on logical core {0, 1}
        auto consumer_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/remote_read_remote_write_sync.cpp",
            writer_logical_core,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////


        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        pass &= tt_metal::detail::WriteToBuffer(input_dram_buffer, src_vec);



        tt_metal::SetRuntimeArgs(
            program,
            producer_kernel,
            loader_logical_core,
            {dram_buffer_src_addr,
            (uint32_t)input_dram_noc_xy.x,
            (uint32_t)input_dram_noc_xy.y,
            loader_buffer_address,
            (uint32_t)writer_worker_core.x,
            (uint32_t)writer_worker_core.y,
            stream_register_address,
            num_output_tiles,
            transient_buffer_size_tiles,
            transient_buffer_size_bytes});

        tt_metal::SetRuntimeArgs(
            program,
            consumer_kernel,
            writer_logical_core,
            {loader_buffer_address,
            (uint32_t)loader_worker_core.x,
            (uint32_t)loader_worker_core.y,
            dram_buffer_dst_addr,
            (uint32_t)output_dram_noc_xy.x,
            (uint32_t)output_dram_noc_xy.y,
            writer_buffer_address,
            stream_register_address,
            num_output_tiles,
            transient_buffer_size_tiles,
            transient_buffer_size_bytes});


        tt_metal::detail::LaunchProgram(device, program);

        std::vector<uint32_t> result_vec;
        tt_metal::detail::ReadFromBuffer(output_dram_buffer, result_vec);
        auto dst_vec = unpack_uint32_vec_into_bfloat16_vec(result_vec);

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        pass &= dst_vec == golden;

        pass &= tt_metal::CloseDevice(device);

    } catch (const std::exception &e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }

    TT_FATAL(pass, "Error");
    return 0;
}
