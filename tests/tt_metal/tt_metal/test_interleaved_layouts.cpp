// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/command_queue_fixture.hpp"

#include <chrono>
#include <cerrno>
#include <cstdint>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/distributed.hpp>
#include <cmath>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include "hostdevcommon/kernel_structs.h"
#include <tt-metalium/kernel_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include "test_common.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>
#include "impl/data_format/bfloat16_utils.hpp"

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
using std::vector;
using namespace tt;
using namespace tt::tt_metal;

namespace {

bool test_write_interleaved_sticks_and_then_read_interleaved_sticks(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, distributed::MeshCommandQueue& cq) {
    /*
        This test just writes sticks in a interleaved fashion to DRAM and then reads back to ensure
        they were written correctly
    */
    bool pass = true;

    try {
        int num_sticks = 256;
        int num_elements_in_stick = 1024;
        int stick_size = num_elements_in_stick * 2;
        uint32_t dram_buffer_size =
            num_sticks * stick_size;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels

        std::vector<uint32_t> src_vec = create_arange_vector_of_bfloat16(dram_buffer_size, false);

        distributed::DeviceLocalBufferConfig device_local_config{
            .page_size = (uint64_t)stick_size,
            .buffer_type = tt_metal::BufferType::DRAM,
        };

        distributed::ReplicatedBufferConfig buffer_config{
            .size = dram_buffer_size,
        };

        auto sticks_buffer = distributed::MeshBuffer::create(buffer_config, device_local_config, mesh_device.get());

        distributed::EnqueueWriteMeshBuffer(cq, sticks_buffer, src_vec, false);

        vector<uint32_t> dst_vec;
        distributed::ReadShard(cq, dst_vec, sticks_buffer, distributed::MeshCoordinate(0, 0));

        pass &= (src_vec == dst_vec);
    } catch (const std::exception& e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    return pass;
}

bool interleaved_stick_reader_single_bank_tilized_writer_datacopy_test(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, distributed::MeshCommandQueue& cq) {
    bool pass = true;

    try {
        tt_metal::Program program = tt_metal::CreateProgram();

        CoreCoord core = {0, 0};

        uint32_t single_tile_size = 2 * 1024;

        int num_sticks = 256;
        int num_elements_in_stick = 1024;
        int stick_size = num_elements_in_stick * 2;

        int num_tiles_c = stick_size / 64;

        // In this test, we are reading in sticks, but tilizing on the fly
        // and then writing tiles back to DRAM
        int num_output_tiles = (num_sticks * num_elements_in_stick) / 1024;

        uint32_t dram_buffer_size =
            num_sticks * stick_size;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels

        distributed::DeviceLocalBufferConfig src_device_local_config{
            .page_size = (uint64_t)stick_size,
            .buffer_type = tt_metal::BufferType::DRAM,
        };
        distributed::ReplicatedBufferConfig src_buffer_config{
            .size = dram_buffer_size,
        };

        distributed::DeviceLocalBufferConfig dst_device_local_config{
            .page_size = dram_buffer_size,
            .buffer_type = tt_metal::BufferType::DRAM,
        };
        distributed::ReplicatedBufferConfig dst_buffer_config{
            .size = dram_buffer_size,
        };

        auto src_dram_buffer =
            distributed::MeshBuffer::create(src_buffer_config, src_device_local_config, mesh_device.get());
        uint32_t dram_buffer_src_addr = src_dram_buffer->address();

        auto dst_dram_buffer =
            distributed::MeshBuffer::create(dst_buffer_config, dst_device_local_config, mesh_device.get());
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
        tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

        uint32_t ouput_cb_index = tt::CBIndex::c_16;
        tt_metal::CircularBufferConfig cb_output_config =
            tt_metal::CircularBufferConfig(single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(ouput_cb_index, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_output_config);

        std::vector<uint32_t> reader_compile_time_args;
        tt::tt_metal::TensorAccessorArgs(src_dram_buffer).append_to(reader_compile_time_args);
        auto unary_reader_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_stick_layout_8bank.cpp",
            core,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt_metal::NOC::RISCV_1_default,
                .compile_args = reader_compile_time_args});

        auto unary_writer_kernel = tt_metal::CreateKernel(
            program,
            "tt_metal/kernels/dataflow/writer_unary.cpp",
            core,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

        vector<uint32_t> compute_kernel_args = {uint(num_output_tiles)};

        tt_metal::CreateKernel(
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

        distributed::EnqueueWriteMeshBuffer(cq, src_dram_buffer, src_vec, false);

        tt_metal::SetRuntimeArgs(
            program,
            unary_reader_kernel,
            core,
            {dram_buffer_src_addr, (uint32_t)num_sticks, (uint32_t)stick_size, (uint32_t)log2(stick_size)});

        tt_metal::SetRuntimeArgs(
            program,
            unary_writer_kernel,
            core,
            {dram_buffer_dst_addr, (uint32_t)0, (uint32_t)num_output_tiles});

        [[maybe_unused]] CoreCoord debug_core = {1, 1};

        distributed::MeshWorkload mesh_workload;
        mesh_workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
        distributed::EnqueueMeshWorkload(cq, mesh_workload, false);

        std::vector<uint32_t> result_vec;
        distributed::ReadShard(cq, result_vec, dst_dram_buffer, distributed::MeshCoordinate(0, 0));
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

    } catch (const std::exception& e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    return pass;
}

// Placeholder tests removed - were not implemented

bool interleaved_tilized_reader_interleaved_stick_writer_datacopy_test(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, distributed::MeshCommandQueue& cq) {
    bool pass = true;

    try {
        tt_metal::Program program = tt_metal::CreateProgram();

        CoreCoord core = {0, 0};

        uint32_t single_tile_size = 2 * 1024;

        int num_sticks = 256;
        int num_elements_in_stick = 1024;
        int stick_size = num_elements_in_stick * 2;

        int num_tiles_c = stick_size / 64;

        // In this test, we are reading in sticks, but tilizing on the fly
        // and then writing tiles back to DRAM
        int num_output_tiles = (num_sticks * num_elements_in_stick) / 1024;

        uint32_t dram_buffer_size =
            num_sticks * stick_size;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels

        distributed::DeviceLocalBufferConfig device_local_config{
            .page_size = (uint64_t)stick_size,
            .buffer_type = tt_metal::BufferType::DRAM,
        };
        distributed::ReplicatedBufferConfig buffer_config{
            .size = dram_buffer_size,
        };

        auto src_dram_buffer =
            distributed::MeshBuffer::create(buffer_config, device_local_config, mesh_device.get());
        uint32_t dram_buffer_src_addr = src_dram_buffer->address();

        auto dst_dram_buffer =
            distributed::MeshBuffer::create(buffer_config, device_local_config, mesh_device.get());
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
        tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

        uint32_t ouput_cb_index = tt::CBIndex::c_16;
        tt_metal::CircularBufferConfig cb_output_config =
            tt_metal::CircularBufferConfig(
                num_input_tiles * single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(ouput_cb_index, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_output_config);

        std::vector<uint32_t> reader_compile_time_args;
        tt::tt_metal::TensorAccessorArgs(src_dram_buffer).append_to(reader_compile_time_args);
        auto unary_reader_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_stick_layout_8bank.cpp",
            core,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt_metal::NOC::RISCV_1_default,
                .compile_args = reader_compile_time_args});

        std::vector<uint32_t> writer_compile_time_args;
        tt::tt_metal::TensorAccessorArgs(dst_dram_buffer).append_to(writer_compile_time_args);
        auto unary_writer_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_stick_layout_8bank.cpp",
            core,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = writer_compile_time_args});

        vector<uint32_t> compute_kernel_args = {uint(num_output_tiles)};

        tt_metal::CreateKernel(
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

        distributed::EnqueueWriteMeshBuffer(cq, src_dram_buffer, src_vec, false);

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

        distributed::MeshWorkload mesh_workload;
        mesh_workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
        distributed::EnqueueMeshWorkload(cq, mesh_workload, false);

        std::vector<uint32_t> result_vec;
        distributed::ReadShard(cq, result_vec, dst_dram_buffer, distributed::MeshCoordinate(0, 0));
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
bool test_interleaved_l1_datacopy(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, distributed::MeshCommandQueue& cq) {
    uint num_pages = 256;
    uint num_bytes_per_page = 2048;
    uint buffer_size = num_pages * num_bytes_per_page;

    uint num_l1_banks = 128;
    uint num_dram_banks = 8;

    bool pass = true;

    tt_metal::Program program = tt_metal::CreateProgram();
    CoreCoord core = {0, 0};

    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(2 * num_bytes_per_page, {{0, tt::DataFormat::Float16_b}})
            .set_page_size(0, num_bytes_per_page);
    tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    tt_metal::CircularBufferConfig cb_output_config =
        tt_metal::CircularBufferConfig(2 * num_bytes_per_page, {{16, tt::DataFormat::Float16_b}})
            .set_page_size(16, num_bytes_per_page);
    tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    // Buffers and host data

    vector<uint32_t> compute_kernel_args = {num_pages};
    tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy.cpp",
        core,
        tt_metal::ComputeConfig{.compile_args = compute_kernel_args});

    std::vector<uint32_t> host_buffer =
        create_random_vector_of_bfloat16(buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());

    distributed::DeviceLocalBufferConfig l1_local_config{
        .page_size = num_bytes_per_page,
        .buffer_type = tt_metal::BufferType::L1,
    };
    distributed::DeviceLocalBufferConfig dram_local_config{
        .page_size = num_bytes_per_page,
        .buffer_type = tt_metal::BufferType::DRAM,
    };
    distributed::ReplicatedBufferConfig buffer_config{
        .size = buffer_size,
    };

    std::shared_ptr<distributed::MeshBuffer> src, dst;
    if constexpr (src_is_in_l1) {
        TT_FATAL(
            (buffer_size % num_l1_banks) == 0,
            "Buffer size ({}) must be divisible by number of L1 banks ({})",
            buffer_size,
            num_l1_banks);

        src = distributed::MeshBuffer::create(buffer_config, l1_local_config, mesh_device.get());
        distributed::EnqueueWriteMeshBuffer(cq, src, host_buffer, false);

    } else {
        TT_FATAL(
            (buffer_size % num_dram_banks) == 0,
            "Buffer size ({}) must be divisible by number of DRAM banks ({})",
            buffer_size,
            num_dram_banks);

        src = distributed::MeshBuffer::create(buffer_config, dram_local_config, mesh_device.get());
        distributed::EnqueueWriteMeshBuffer(cq, src, host_buffer, false);
    }

    // Create destination buffer prior to kernels to build compile-time args
    if constexpr (dst_is_in_l1) {
        dst = distributed::MeshBuffer::create(buffer_config, l1_local_config, mesh_device.get());
    } else {
        dst = distributed::MeshBuffer::create(buffer_config, dram_local_config, mesh_device.get());
    }

    // Create kernels with TensorAccessorArgs compile-time arguments
    std::vector<uint32_t> reader_compile_time_args;
    tt::tt_metal::TensorAccessorArgs(src).append_to(reader_compile_time_args);
    auto unary_reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_8bank.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args});

    std::vector<uint32_t> writer_compile_time_args;
    tt::tt_metal::TensorAccessorArgs(dst).append_to(writer_compile_time_args);
    auto unary_writer_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_8bank.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args});

    // Now that kernels exist, set reader runtime args
    tt_metal::SetRuntimeArgs(program, unary_reader_kernel, core, {src->address(), 0, 0, num_pages});

    std::vector<uint32_t> readback_buffer;
    tt_metal::SetRuntimeArgs(program, unary_writer_kernel, core, {dst->address(), 0, num_pages});

    distributed::MeshWorkload mesh_workload;
    mesh_workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(cq, mesh_workload, false);

    distributed::ReadShard(cq, readback_buffer, dst, distributed::MeshCoordinate(0, 0));

    pass = (host_buffer == readback_buffer);

    TT_FATAL(pass, "Test failed - buffer comparison did not match");

    return pass;
}

}  // namespace

TEST_F(UnitMeshCQSingleCardFixture, WriteInterleavedSticksAndReadBack) {
    ASSERT_TRUE(test_write_interleaved_sticks_and_then_read_interleaved_sticks(
        devices_[0], devices_[0]->mesh_command_queue()));
}

TEST_F(UnitMeshCQSingleCardFixture, InterleavedStickReaderSingleBankTilizedWriter) {
    ASSERT_TRUE(interleaved_stick_reader_single_bank_tilized_writer_datacopy_test(
        devices_[0], devices_[0]->mesh_command_queue()));
}

TEST_F(UnitMeshCQSingleCardFixture, InterleavedTilizedReaderInterleavedStickWriter) {
    ASSERT_TRUE(interleaved_tilized_reader_interleaved_stick_writer_datacopy_test(
        devices_[0], devices_[0]->mesh_command_queue()));
}

TEST_F(UnitMeshCQSingleCardFixture, InterleavedL1DatacopyL1ToL1) {
    ASSERT_TRUE(
        (test_interleaved_l1_datacopy<true, true>(devices_[0], devices_[0]->mesh_command_queue())));
}

TEST_F(UnitMeshCQSingleCardFixture, InterleavedL1DatacopyDramToL1) {
    ASSERT_TRUE(
        (test_interleaved_l1_datacopy<false, true>(devices_[0], devices_[0]->mesh_command_queue())));
}

TEST_F(UnitMeshCQSingleCardFixture, InterleavedL1DatacopyL1ToDram) {
    ASSERT_TRUE(
        (test_interleaved_l1_datacopy<true, false>(devices_[0], devices_[0]->mesh_command_queue())));
}

TEST_F(UnitMeshCQSingleCardFixture, InterleavedL1DatacopyDramToDram) {
    ASSERT_TRUE(
        (test_interleaved_l1_datacopy<false, false>(devices_[0], devices_[0]->mesh_command_queue())));
}
