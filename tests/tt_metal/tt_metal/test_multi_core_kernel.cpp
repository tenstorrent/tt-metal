// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "common/bfloat16.hpp"
#include "common/core_coord.h"
// #include "tt_gdb/tt_gdb.hpp"


//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

std::tuple<tt_metal::Program, tt_metal::KernelHandle, tt_metal::KernelHandle> create_program(
    tt_metal::Device *device,
    uint32_t single_tile_size,
    const CoreRange &all_cores,
    const std::vector<uint32_t> &eltwise_unary_args) {
    tt_metal::Program program = tt_metal::CreateProgram();

    CoreCoord start_core = all_cores.start_coord;
    CoreCoord end_core = all_cores.end_coord;
    // input CB is larger than the output CB, to test the backpressure from the output CB all the way into the input CB
    // CB_out size = 1 forces the serialization of packer and writer kernel, generating backpressure to math kernel, input CB and reader
    for (auto x = start_core.x; x <= end_core.x; x++) {
        for (auto y = start_core.y; y <= end_core.y; y++) {
            auto core = CoreCoord{x, y};
            uint32_t src0_cb_index = 0;
            uint32_t num_input_tiles = 8;
            tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src0_cb_index, single_tile_size);
            auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

            uint32_t ouput_cb_index = 16; // output operands start at index 16
            uint32_t num_output_tiles = 1;
            tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(ouput_cb_index, single_tile_size);
            auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);
        }
    }

    auto reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_4.cpp",
        all_cores,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    auto writer_kernel = tt_metal::CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        all_cores,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    auto eltwise_unary_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy.cpp",
        all_cores,
        tt_metal::ComputeConfig{.compile_args = eltwise_unary_args}
    );

    return {std::move(program), reader_kernel, writer_kernel};
}

void compile_and_configure_program(
    tt_metal::Device *device,
    tt_metal::Program &program,
    std::vector<uint32_t> &src_vec,
    tt_metal::Buffer &src_dram_buffer) {
    ////////////////////////////////////////////////////////////////////////////
    //                      Compile Application
    ////////////////////////////////////////////////////////////////////////////



    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Application
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::detail::WriteToBuffer(src_dram_buffer, src_vec);


}

void set_rt_args(tt_metal::Program &program, tt_metal::KernelHandle kernel, const CoreRange &core_range, const std::array<uint32_t, 4> &rt_args) {
    for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
        for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
            CoreCoord core = CoreCoord(x, y);
            tt_metal::SetRuntimeArgs(program, kernel, core, rt_args);
        }
    }
}

void write_same_runtime_args_to_device(
    tt_metal::Device *device,
    tt_metal::Program &program,
    tt_metal::KernelHandle reader_kernel_id,
    tt_metal::KernelHandle writer_kernel_id,
    const CoreRange &core_range,
    int32_t num_tiles,
    tt_metal::Buffer &src_dram_buffer,
    tt_metal::Buffer &dst_dram_buffer)
{
    auto dram_src_noc_xy = src_dram_buffer.noc_coordinates();
    auto dram_dst_noc_xy = dst_dram_buffer.noc_coordinates();

    const std::array unary_reader_args{
    (std::uint32_t)src_dram_buffer.address(),
    (std::uint32_t)dram_src_noc_xy.x,
    (std::uint32_t)dram_src_noc_xy.y,
    (std::uint32_t)num_tiles};

    const std::array unary_writer_args{
    (std::uint32_t)dst_dram_buffer.address(),
    (std::uint32_t)dram_dst_noc_xy.x,
    (std::uint32_t)dram_dst_noc_xy.y,
    (std::uint32_t)num_tiles};

    set_rt_args(program, reader_kernel_id, core_range, unary_reader_args);
    set_rt_args(program, writer_kernel_id, core_range, unary_writer_args);

}

void write_unique_writer_runtime_args_to_device(
    tt_metal::Device *device,
    tt_metal::Program &program,
    tt_metal::KernelHandle reader_kernel_id,
    tt_metal::KernelHandle writer_kernel_id,
    const CoreRange &core_range,
    const CoreRangeSet &core_blocks,
    int32_t num_tiles,
    tt_metal::Buffer &src_dram_buffer,
    tt_metal::Buffer &dst_dram_buffer_1,
    tt_metal::Buffer &dst_dram_buffer_2,
    tt_metal::Buffer &dst_dram_buffer_3
) {
    auto dram_src_noc_xy = src_dram_buffer.noc_coordinates();
    // All dst buffers use the same DRAM channel
    auto dram_dst_noc_xy = dst_dram_buffer_1.noc_coordinates();

    // Same readers args because all kernels read from same src
    const std::array unary_reader_args{
        (std::uint32_t)src_dram_buffer.address(),
        (std::uint32_t)dram_src_noc_xy.x,
        (std::uint32_t)dram_src_noc_xy.y,
        (std::uint32_t)num_tiles};

    const std::array unary_writer_args_1{
        dst_dram_buffer_1.address(),
        (std::uint32_t)dram_dst_noc_xy.x,
        (std::uint32_t)dram_dst_noc_xy.y,
        (std::uint32_t)num_tiles};

    const std::array unary_writer_args_2{
        dst_dram_buffer_2.address(),
        (std::uint32_t)dram_dst_noc_xy.x,
        (std::uint32_t)dram_dst_noc_xy.y,
        (std::uint32_t)num_tiles};

    const std::array unary_writer_args_3{
        dst_dram_buffer_3.address(),
        (std::uint32_t)dram_dst_noc_xy.x,
        (std::uint32_t)dram_dst_noc_xy.y,
        (std::uint32_t)num_tiles};

    set_rt_args(program, reader_kernel_id, core_range, unary_reader_args);
    int core_range_idx = 0;
    const std::array rt_args = {unary_writer_args_1, unary_writer_args_2, unary_writer_args_3};
    for (auto core_range : core_blocks.ranges()) {
        set_rt_args(program, writer_kernel_id, core_range, rt_args.at(core_range_idx++));
    }

}

bool test_multi_core_kernel_same_runtime_args(tt_metal::Device *device) {

    bool pass = true;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Buffer Setup
    ////////////////////////////////////////////////////////////////////////////
    CoreCoord start_core = {0, 0};
    CoreCoord end_core = {2, 2};

    CoreRange all_cores(start_core, end_core);

    uint32_t single_tile_size = 2 * 1024;
    int32_t num_tiles = 2048;
    uint32_t dram_buffer_size = single_tile_size * num_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    tt_metal::InterleavedBufferConfig dram_config{
                    .device=device,
                    .size = dram_buffer_size,
                    .page_size = dram_buffer_size,
                    .buffer_type = tt_metal::BufferType::DRAM
        };

    auto src_dram_buffer = CreateBuffer(dram_config);
    uint32_t dram_buffer_src_addr = src_dram_buffer->address();
    auto dst_dram_buffer = CreateBuffer(dram_config);
    uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

    ////////////////////////////////////////////////////////////////////////////
    //                  Compile Time Args Setup
    ////////////////////////////////////////////////////////////////////////////
    // Same compile time args for all cores
    vector<uint32_t> compute_kernel_args = {
        uint(num_tiles) // per_core_tile_cnt
    };

    ////////////////////////////////////////////////////////////////////////////
    //                  Compile and Execute Program
    ////////////////////////////////////////////////////////////////////////////
    auto [program, reader_kernel_id, writer_kernel_id] = create_program(device, single_tile_size, all_cores, compute_kernel_args);

    std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
        src_dram_buffer->size(), 100, std::chrono::system_clock::now().time_since_epoch().count());

    compile_and_configure_program(device, program, src_vec, *src_dram_buffer);

    write_same_runtime_args_to_device(device, program, reader_kernel_id, writer_kernel_id, all_cores, num_tiles, *src_dram_buffer, *dst_dram_buffer);

    tt_metal::detail::LaunchProgram(device, program);

    std::vector<uint32_t> result_vec;
    tt_metal::detail::ReadFromBuffer(dst_dram_buffer, result_vec);

    ////////////////////////////////////////////////////////////////////////////
    //                          Validation
    ////////////////////////////////////////////////////////////////////////////
    pass &= (src_vec == result_vec);

    DeallocateBuffer(*src_dram_buffer);
    DeallocateBuffer(*dst_dram_buffer);

    return pass;
}

bool test_multi_core_kernel_unique_runtime_args(tt_metal::Device *device) {

    bool pass = true;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Buffer Setup
    ////////////////////////////////////////////////////////////////////////////
    CoreCoord start_core = {0, 0};
    CoreCoord end_core = {1, 1};
    CoreRange start_core_range(start_core, start_core);
    CoreRange core_group({0, 1}, {1, 1});
    CoreRange single_core({1, 0}, {1, 0});
    CoreRange all_cores(start_core, end_core);
    CoreRangeSet core_blocks = CoreRangeSet(std::vector{start_core_range, single_core, core_group});

    uint32_t single_tile_size = 2 * 1024;
    int32_t num_tiles = 2048;
    uint32_t dram_buffer_size = single_tile_size * num_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

    tt_metal::InterleavedBufferConfig dram_config{
                    .device=device,
                    .size = dram_buffer_size,
                    .page_size = dram_buffer_size,
                    .buffer_type = tt_metal::BufferType::DRAM
        };

    auto src_dram_buffer = CreateBuffer(dram_config);
    uint32_t dram_buffer_src_addr = src_dram_buffer->address();
    auto dst_dram_buffer_1 = CreateBuffer(dram_config);
    uint32_t dram_buffer_dst_addr_1 = dst_dram_buffer_1->address();
    auto dst_dram_buffer_2 = CreateBuffer(dram_config);
    uint32_t dram_buffer_dst_addr_2 = dst_dram_buffer_2->address();
    auto dst_dram_buffer_3 = CreateBuffer(dram_config);
    uint32_t dram_buffer_dst_addr_3 = dst_dram_buffer_3->address();


    ////////////////////////////////////////////////////////////////////////////
    //                  Compile Time Args Setup
    ////////////////////////////////////////////////////////////////////////////
    vector<uint32_t> compute_kernel_args = {
        uint(num_tiles) // per_core_tile_cnt
    };

    ////////////////////////////////////////////////////////////////////////////
    //                  Compile and Execute Program
    ////////////////////////////////////////////////////////////////////////////
    auto [program, reader_kernel_id, writer_kernel_id] = create_program(device, single_tile_size, all_cores, compute_kernel_args);

    std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
        src_dram_buffer->size(), 100, std::chrono::system_clock::now().time_since_epoch().count());

    compile_and_configure_program(device, program, src_vec, *src_dram_buffer);

    write_unique_writer_runtime_args_to_device(
        device, program, reader_kernel_id, writer_kernel_id, all_cores, core_blocks, num_tiles, *src_dram_buffer, *dst_dram_buffer_1, *dst_dram_buffer_2, *dst_dram_buffer_3);

    tt_metal::detail::LaunchProgram(device, program);

    std::vector<uint32_t> result_vec_1;
    tt_metal::detail::ReadFromBuffer(dst_dram_buffer_1, result_vec_1);

    std::vector<uint32_t> result_vec_2;
    tt_metal::detail::ReadFromBuffer(dst_dram_buffer_2, result_vec_2);

    std::vector<uint32_t> result_vec_3;
    tt_metal::detail::ReadFromBuffer(dst_dram_buffer_3, result_vec_3);


    ////////////////////////////////////////////////////////////////////////////
    //                          Validation
    ////////////////////////////////////////////////////////////////////////////
    pass &= (src_vec == result_vec_1);
    pass &= (src_vec == result_vec_2);
    pass &= (src_vec == result_vec_3);

    DeallocateBuffer(*src_dram_buffer);
    DeallocateBuffer(*dst_dram_buffer_1);
    DeallocateBuffer(*dst_dram_buffer_2);
    DeallocateBuffer(*dst_dram_buffer_3);

    return pass;
}

int main(int argc, char **argv) {
    bool pass = true;


    auto slow_dispatch_mode = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    TT_FATAL(slow_dispatch_mode, "This test only supports TT_METAL_SLOW_DISPATCH_MODE");

    try {


        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        tt_metal::Device *device =
            tt_metal::CreateDevice(device_id);

        pass &= test_multi_core_kernel_same_runtime_args(device);

        pass &= test_multi_core_kernel_unique_runtime_args(device);

        ////////////////////////////////////////////////////////////////////////////
        //                          Teardown
        ////////////////////////////////////////////////////////////////////////////
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
