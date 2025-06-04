// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <errno.h>
#include <fmt/base.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/types.h>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/assert.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include "hostdevcommon/kernel_structs.h"
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include "test_common.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/utils.hpp>

namespace tt {
enum class ARCH;
namespace tt_metal {
class IDevice;
}  // namespace tt_metal
}  // namespace tt

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
using std::vector;
using namespace tt;

bool test_write_interleaved_sticks_and_then_read_interleaved_sticks(const tt::ARCH& arch) {
    /*
        This test just writes sticks in a interleaved fashion to DRAM and then reads back to ensure
        they were written correctly
    */
    bool pass = true;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        tt_metal::IDevice* device = tt_metal::CreateDevice(device_id);

        int num_sticks = 256;
        int num_elements_in_stick = 1024;
        int stick_size = num_elements_in_stick * 2;
        int num_elements_in_stick_as_packed_uint32 = num_elements_in_stick / 2;
        uint32_t dram_buffer_size =
            num_sticks * stick_size;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels

        std::vector<uint32_t> src_vec = create_arange_vector_of_bfloat16(dram_buffer_size, false);

        tt_metal::InterleavedBufferConfig sticks_config{
            .device = device,
            .size = dram_buffer_size,
            .page_size = (uint64_t)stick_size,
            .buffer_type = tt_metal::BufferType::DRAM};

        auto sticks_buffer = CreateBuffer(sticks_config);
        uint32_t dram_buffer_src_addr = sticks_buffer->address();

        tt_metal::detail::WriteToBuffer(sticks_buffer, src_vec);

        vector<uint32_t> dst_vec;
        tt_metal::detail::ReadFromBuffer(sticks_buffer, dst_vec);

        pass &= (src_vec == dst_vec);
        pass &= tt_metal::CloseDevice(device);
    } catch (const std::exception& e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    return pass;
}

bool interleaved_stick_reader_single_bank_tilized_writer_datacopy_test(const tt::ARCH& arch) {
    bool pass = true;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        tt_metal::IDevice* device = tt_metal::CreateDevice(device_id);

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::Program program = tt_metal::CreateProgram();

        CoreCoord core = {0, 0};

        uint32_t single_tile_size = 2 * 1024;

        int num_sticks = 256;
        int num_elements_in_stick = 1024;
        int stick_size = num_elements_in_stick * 2;
        int num_elements_in_stick_as_packed_uint32 = num_elements_in_stick / 2;

        int num_tiles_c = stick_size / 64;

        // In this test, we are reading in sticks, but tilizing on the fly
        // and then writing tiles back to DRAM
        int num_output_tiles = (num_sticks * num_elements_in_stick) / 1024;

        uint32_t dram_buffer_size =
            num_sticks * stick_size;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels
        tt_metal::InterleavedBufferConfig src_config{
            .device = device,
            .size = dram_buffer_size,
            .page_size = (uint64_t)stick_size,
            .buffer_type = tt_metal::BufferType::DRAM};

        tt_metal::InterleavedBufferConfig dst_config{
            .device = device,
            .size = dram_buffer_size,
            .page_size = dram_buffer_size,
            .buffer_type = tt_metal::BufferType::DRAM};

        auto src_dram_buffer = CreateBuffer(src_config);
        uint32_t dram_buffer_src_addr = src_dram_buffer->address();

        auto dst_dram_buffer = CreateBuffer(dst_config);
        uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

        // input CB is larger than the output CB, to test the backpressure from the output CB all the way into the input
        // CB CB_out size = 1 forces the serialization of packer and writer kernel, generating backpressure to math
        // kernel, input CB and reader
        uint32_t src0_cb_index = tt::CBIndex::c_0;
        uint32_t num_input_tiles = num_tiles_c;
        tt_metal::CircularBufferConfig cb_src0_config =
            tt_metal::CircularBufferConfig(
                num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src0_cb_index, single_tile_size);
        auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

        uint32_t ouput_cb_index = tt::CBIndex::c_16;
        tt_metal::CircularBufferConfig cb_output_config =
            tt_metal::CircularBufferConfig(single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(ouput_cb_index, single_tile_size);
        auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

        auto unary_reader_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_stick_layout_8bank.cpp",
            core,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt_metal::NOC::RISCV_1_default,
                .compile_args = {1}});

        auto unary_writer_kernel = tt_metal::CreateKernel(
            program,
            "tt_metal/kernels/dataflow/writer_unary.cpp",
            core,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

        vector<uint32_t> compute_kernel_args = {uint(num_output_tiles)};

        auto eltwise_unary_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy.cpp",
            core,
            tt_metal::ComputeConfig{.compile_args = compute_kernel_args});

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        std::vector<uint32_t> src_vec = create_arange_vector_of_bfloat16(dram_buffer_size, false);

        tt_metal::detail::WriteToBuffer(src_dram_buffer, src_vec);

        tt_metal::SetRuntimeArgs(
            program,
            unary_reader_kernel,
            core,
            {dram_buffer_src_addr, (uint32_t)num_sticks, (uint32_t)stick_size, (uint32_t)log2(stick_size)});

        tt_metal::SetRuntimeArgs(
            program,
            unary_writer_kernel,
            core,
            {dram_buffer_dst_addr,
            (uint32_t) 0,
            (uint32_t) num_output_tiles});

        CoreCoord debug_core = {1, 1};

        tt_metal::detail::LaunchProgram(device, program);

        std::vector<uint32_t> result_vec;
        tt_metal::detail::ReadFromBuffer(dst_dram_buffer, result_vec);
        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////

        pass &= (src_vec == result_vec);

        if (not pass) {
            std::cout << "GOLDEN" << std::endl;
            print_vec_of_uint32_as_packed_bfloat16(src_vec, num_output_tiles);

            std::cout << "RESULT" << std::endl;
            print_vec_of_uint32_as_packed_bfloat16(result_vec, num_output_tiles);
        }

        DeallocateBuffer(*dst_dram_buffer);
        pass &= tt_metal::CloseDevice(device);

    } catch (const std::exception& e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    return pass;
}

bool interleaved_stick_reader_interleaved_tilized_writer_datacopy_test() {
    bool pass = true;

    /*
        Placeholder to not forget to write this test
    */

    return pass;
}

bool interleaved_tilized_reader_single_bank_stick_writer_datacopy_test() {
    bool pass = true;

    /*
        Placeholder to not forget to write this test
    */

    return pass;
}

bool interleaved_tilized_reader_interleaved_stick_writer_datacopy_test(const tt::ARCH& arch) {
    bool pass = true;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        tt_metal::IDevice* device = tt_metal::CreateDevice(device_id);

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::Program program = tt_metal::CreateProgram();

        CoreCoord core = {0, 0};

        uint32_t single_tile_size = 2 * 1024;

        int num_sticks = 256;
        int num_elements_in_stick = 1024;
        int stick_size = num_elements_in_stick * 2;
        int num_elements_in_stick_as_packed_uint32 = num_elements_in_stick / 2;

        int num_tiles_c = stick_size / 64;

        // In this test, we are reading in sticks, but tilizing on the fly
        // and then writing tiles back to DRAM
        int num_output_tiles = (num_sticks * num_elements_in_stick) / 1024;

        uint32_t dram_buffer_size =
            num_sticks * stick_size;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels

        tt_metal::InterleavedBufferConfig dram_config{
            .device = device,
            .size = dram_buffer_size,
            .page_size = (uint64_t)stick_size,
            .buffer_type = tt_metal::BufferType::DRAM};
        auto src_dram_buffer = CreateBuffer(dram_config);
        uint32_t dram_buffer_src_addr = src_dram_buffer->address();

        auto dst_dram_buffer = CreateBuffer(dram_config);
        uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

        // input CB is larger than the output CB, to test the backpressure from the output CB all the way into the input
        // CB CB_out size = 1 forces the serialization of packer and writer kernel, generating backpressure to math
        // kernel, input CB and reader
        uint32_t src0_cb_index = 0;
        uint32_t num_input_tiles = num_tiles_c;
        tt_metal::CircularBufferConfig cb_src0_config =
            tt_metal::CircularBufferConfig(
                num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src0_cb_index, single_tile_size);
        auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

        uint32_t ouput_cb_index = tt::CBIndex::c_16;
        tt_metal::CircularBufferConfig cb_output_config =
            tt_metal::CircularBufferConfig(
                num_input_tiles * single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(ouput_cb_index, single_tile_size);
        auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

        auto unary_reader_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_stick_layout_8bank.cpp",
            core,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt_metal::NOC::RISCV_1_default,
                .compile_args = {1}});

        auto unary_writer_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_stick_layout_8bank.cpp",
            core,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

        vector<uint32_t> compute_kernel_args = {uint(num_output_tiles)};

        auto eltwise_unary_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy.cpp",
            core,
            tt_metal::ComputeConfig{.compile_args = compute_kernel_args});

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        std::vector<uint32_t> src_vec = create_arange_vector_of_bfloat16(dram_buffer_size, false);

        tt_metal::detail::WriteToBuffer(src_dram_buffer, src_vec);

        tt_metal::SetRuntimeArgs(
            program,
            unary_reader_kernel,
            core,
            {dram_buffer_src_addr, (uint32_t)num_sticks, (uint32_t)stick_size, (uint32_t)log2(stick_size)});

        tt_metal::SetRuntimeArgs(
            program,
            unary_writer_kernel,
            core,
            {dram_buffer_dst_addr, (uint32_t)num_sticks, (uint32_t)stick_size, (uint32_t)log2(stick_size)});

        tt_metal::detail::LaunchProgram(device, program);

        std::vector<uint32_t> result_vec;
        tt_metal::detail::ReadFromBuffer(dst_dram_buffer, result_vec);
        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////

        pass &= (src_vec == result_vec);

        if (not pass) {
            std::cout << "GOLDEN" << std::endl;
            print_vec_of_uint32_as_packed_bfloat16(src_vec, num_output_tiles);

            std::cout << "RESULT" << std::endl;
            print_vec_of_uint32_as_packed_bfloat16(result_vec, num_output_tiles);
        }

        pass &= tt_metal::CloseDevice(device);

    } catch (const std::exception& e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    return pass;
}

template <bool src_is_in_l1, bool dst_is_in_l1>
bool test_interleaved_l1_datacopy(const tt::ARCH& arch) {
    uint num_pages = 256;
    uint num_bytes_per_page = 2048;
    uint num_entries_per_page = 512;
    uint num_bytes_per_entry = 4;
    uint buffer_size = num_pages * num_bytes_per_page;

    uint num_l1_banks = 128;
    uint num_dram_banks = 8;

    bool pass = true;

    int device_id = 0;
    tt_metal::IDevice* device = tt_metal::CreateDevice(device_id);

    tt_metal::Program program = tt_metal::CreateProgram();
    CoreCoord core = {0, 0};

    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(2 * num_bytes_per_page, {{0, tt::DataFormat::Float16_b}})
            .set_page_size(0, num_bytes_per_page);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    tt_metal::CircularBufferConfig cb_output_config =
        tt_metal::CircularBufferConfig(2 * num_bytes_per_page, {{16, tt::DataFormat::Float16_b}})
            .set_page_size(16, num_bytes_per_page);
    auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    auto unary_reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_8bank.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .compile_args = {not src_is_in_l1}});

    auto unary_writer_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_8bank.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = {not dst_is_in_l1}});

    vector<uint32_t> compute_kernel_args = {num_pages};
    auto eltwise_unary_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy.cpp",
        core,
        tt_metal::ComputeConfig{.compile_args = compute_kernel_args});

    std::vector<uint32_t> host_buffer =
        create_random_vector_of_bfloat16(buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
    tt_metal::InterleavedBufferConfig l1_config{
        .device = device,
        .size = buffer_size,
        .page_size = num_bytes_per_page,
        .buffer_type = tt_metal::BufferType::L1};
    tt_metal::InterleavedBufferConfig dram_config{
        .device = device,
        .size = buffer_size,
        .page_size = num_bytes_per_page,
        .buffer_type = tt_metal::BufferType::DRAM};

    std::shared_ptr<tt_metal::Buffer> src, dst;
    if constexpr (src_is_in_l1) {
        TT_FATAL((buffer_size % num_l1_banks) == 0, "Error");

        src = CreateBuffer(l1_config);
        tt_metal::detail::WriteToBuffer(src, host_buffer);

        tt_metal::SetRuntimeArgs(program, unary_reader_kernel, core, {src->address(), 0, 0, num_pages});

    } else {
        TT_FATAL((buffer_size % num_dram_banks) == 0, "Error");

        src = CreateBuffer(dram_config);
        tt_metal::detail::WriteToBuffer(src, host_buffer);

        tt_metal::SetRuntimeArgs(program, unary_reader_kernel, core, {src->address(), 0, 0, num_pages});
    }

    std::vector<uint32_t> readback_buffer;
    if constexpr (dst_is_in_l1) {
        dst = CreateBuffer(l1_config);

        tt_metal::SetRuntimeArgs(
            program,
            unary_writer_kernel,
            core,
            {dst->address(), 0, num_pages});

        tt_metal::detail::LaunchProgram(device, program);

        tt_metal::detail::ReadFromBuffer(dst, readback_buffer);

    } else {
        dst = CreateBuffer(dram_config);

        tt_metal::SetRuntimeArgs(
            program,
            unary_writer_kernel,
            core,
            {dst->address(), 0, num_pages});

        tt_metal::detail::LaunchProgram(device, program);

        tt_metal::detail::ReadFromBuffer(dst, readback_buffer);
    }

    pass = (host_buffer == readback_buffer);

    pass &= tt_metal::CloseDevice(device);

    TT_FATAL(pass, "Error");

    return pass;
}

int main(int argc, char** argv) {
    auto slow_dispatch_mode = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    TT_FATAL(slow_dispatch_mode, "This test only supports TT_METAL_SLOW_DISPATCH_MODE");

    bool pass = true;
    ////////////////////////////////////////////////////////////////////////////
    //                      Initial Runtime Args Parse
    ////////////////////////////////////////////////////////////////////////////
    std::vector<std::string> input_args(argv, argv + argc);
    string arch_name = "";
    try {
        std::tie(arch_name, input_args) =
            test_args::get_command_option_and_remaining_args(input_args, "--arch", "grayskull");
    } catch (const std::exception& e) {
        TT_THROW("Command line arguments found exception", e.what());
    }
    const tt::ARCH arch = tt::get_arch_from_string(arch_name);

    // DRAM row/tile interleaved layout tests
    pass &= test_write_interleaved_sticks_and_then_read_interleaved_sticks(arch);
    pass &= interleaved_stick_reader_single_bank_tilized_writer_datacopy_test(arch);
    pass &= interleaved_tilized_reader_interleaved_stick_writer_datacopy_test(arch);

    // L1 tile-interleaved tests
    pass &= test_interleaved_l1_datacopy<true, true>(arch);
    pass &= test_interleaved_l1_datacopy<false, true>(arch);
    pass &= test_interleaved_l1_datacopy<true, false>(arch);
    pass &= test_interleaved_l1_datacopy<false, false>(arch);

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }
}
