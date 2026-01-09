// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// =============================================================================
// Consolidated legacy tests migrated from standalone int main() executables.
//
// Benefits:
//   - Single executable instead of ~30+ separate ones
//   - Reduced compile/link time
//   - Automatic dispatch mode handling via GTEST_SKIP()
//   - Standard gtest filtering, shuffling, and reporting
//
// Usage:
//   TEST_F(SlowDispatchFixture, MyTest)   - only runs in slow dispatch mode
//   TEST_F(FastDispatchFixture, MyTest)   - only runs in fast dispatch mode
//   TEST_F(EitherDispatchFixture, MyTest) - runs in both modes
//   TEST_F(HostOnlyFixture, MyTest)       - no device needed, runs always
// =============================================================================

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <tuple>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/bfloat4.hpp>
#include <tt-metalium/bfloat8.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-logger/tt-logger.hpp>
#include "impl/buffers/semaphore.hpp"
#include <tt_stl/span.hpp>

#include "hostdevcommon/kernel_structs.h"
#include "test_gold_impls.hpp"
#include "tt_metal/test_utils/deprecated/tensor.hpp"
#include "tt_metal/test_utils/bfloat_utils.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::constants;

namespace {

// =============================================================================
// FIXTURE DEFINITIONS
// =============================================================================

enum class DispatchConstraint {
    SlowOnly,  // Only runs when TT_METAL_SLOW_DISPATCH_MODE is set
    FastOnly,  // Only runs when TT_METAL_SLOW_DISPATCH_MODE is NOT set
    Either     // Runs in both modes (default)
};

inline bool IsSlowDispatch() { return std::getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr; }

template <DispatchConstraint Constraint = DispatchConstraint::Either>
class LegacyDeviceFixture : public ::testing::Test {
protected:
    void SetUp() override {
        if constexpr (Constraint == DispatchConstraint::SlowOnly) {
            if (!IsSlowDispatch()) {
                GTEST_SKIP() << "Test requires slow dispatch mode (set TT_METAL_SLOW_DISPATCH_MODE=1)";
            }
        } else if constexpr (Constraint == DispatchConstraint::FastOnly) {
            if (IsSlowDispatch()) {
                GTEST_SKIP() << "Test requires fast dispatch mode (unset TT_METAL_SLOW_DISPATCH_MODE)";
            }
        }

        mesh_device_ = distributed::MeshDevice::create_unit_mesh(0);
        device_ = mesh_device_->get_devices()[0];
    }

    void TearDown() override {
        mesh_device_.reset();
        device_ = nullptr;
    }

    IDevice* device() { return device_; }
    distributed::MeshDevice* mesh_device() { return mesh_device_.get(); }
    distributed::MeshCommandQueue& command_queue() { return mesh_device_->mesh_command_queue(); }

private:
    std::shared_ptr<distributed::MeshDevice> mesh_device_;
    IDevice* device_ = nullptr;
};

using SlowDispatchFixture = LegacyDeviceFixture<DispatchConstraint::SlowOnly>;
using FastDispatchFixture = LegacyDeviceFixture<DispatchConstraint::FastOnly>;
using EitherDispatchFixture = LegacyDeviceFixture<DispatchConstraint::Either>;

class HostOnlyFixture : public ::testing::Test {};

// =============================================================================
// HOST-ONLY TESTS (no device required)
// =============================================================================

TEST_F(HostOnlyFixture, DISABLED_Bfp8Conversion) {
    uint32_t num_tiles = 1;
    int num_float_in_tile = 1024;
    int float_data_size = num_tiles * num_float_in_tile;

    std::vector<float> fp32_vec(float_data_size, 0);
    for (size_t i = 0; i < fp32_vec.size(); i++) {
        fp32_vec.at(i) = static_cast<float>(i);
    }

    std::vector<uint32_t> shape_vec = {1, 1, 32, 32};
    std::vector<float> tiled_fp32_vec = convert_layout(
        tt::stl::make_const_span(fp32_vec), shape_vec, TensorLayoutType::LIN_ROW_MAJOR, TensorLayoutType::TILED_NFACES);

    std::vector<uint32_t> packed_bfp8b_tile_vec_rm_in =
        pack_as_bfp8_tiles(tt::stl::make_const_span(fp32_vec), /*row_major_input=*/true, /*is_exp_a=*/false);
    std::vector<float> unpacked_bfp8b_tile_vec_rm_out =
        unpack_bfp8_tiles_into_float_vec(packed_bfp8b_tile_vec_rm_in, /*row_major_output*/ true, /*is_exp_a=*/false);

    std::vector<uint32_t> packed_bfp8b_tile_vec_tile_in =
        pack_as_bfp8_tiles(tt::stl::make_const_span(tiled_fp32_vec), /*row_major_input=*/false, /*is_exp_a=*/false);
    std::vector<float> unpacked_bfp8b_tile_vec_tile_out =
        unpack_bfp8_tiles_into_float_vec(packed_bfp8b_tile_vec_tile_in, /*row_major_output=*/false, /*is_exp_a=*/false);

    std::vector<float> tiled_to_rm_fp32_vec = convert_layout(
        tt::stl::make_const_span(unpacked_bfp8b_tile_vec_tile_out),
        shape_vec,
        TensorLayoutType::TILED_NFACES,
        TensorLayoutType::LIN_ROW_MAJOR);
    std::vector<float> rm_to_tiled_fp32_vec = convert_layout(
        tt::stl::make_const_span(unpacked_bfp8b_tile_vec_rm_out),
        shape_vec,
        TensorLayoutType::LIN_ROW_MAJOR,
        TensorLayoutType::TILED_NFACES);

    EXPECT_EQ(packed_bfp8b_tile_vec_rm_in, packed_bfp8b_tile_vec_tile_in);

    ASSERT_EQ(unpacked_bfp8b_tile_vec_rm_out.size(), fp32_vec.size());
    for (size_t rm_idx = 0; rm_idx < fp32_vec.size(); rm_idx++) {
        float golden = fp32_vec.at(rm_idx);
        float converted = unpacked_bfp8b_tile_vec_rm_out.at(rm_idx);
        float atol = 8.0f;
        float rtol = 0.01f;
        EXPECT_TRUE(is_close(golden, converted, rtol, atol))
            << "Mismatch at index " << rm_idx << ": golden=" << golden << ", converted=" << converted;
    }

    ASSERT_EQ(unpacked_bfp8b_tile_vec_tile_out.size(), tiled_fp32_vec.size());
    for (size_t rm_idx = 0; rm_idx < fp32_vec.size(); rm_idx++) {
        float golden = tiled_fp32_vec.at(rm_idx);
        float converted = unpacked_bfp8b_tile_vec_tile_out.at(rm_idx);
        float atol = 8.0f;
        float rtol = 0.01f;
        EXPECT_TRUE(is_close(golden, converted, rtol, atol))
            << "Mismatch at index " << rm_idx << ": golden=" << golden << ", converted=" << converted;
    }

    EXPECT_EQ(unpacked_bfp8b_tile_vec_rm_out, tiled_to_rm_fp32_vec);
    EXPECT_EQ(unpacked_bfp8b_tile_vec_tile_out, rm_to_tiled_fp32_vec);
}

TEST_F(HostOnlyFixture, DISABLED_Bfp4Conversion) {
    uint32_t num_tiles = 1;
    int num_float_in_tile = 1024;
    int float_data_size = num_tiles * num_float_in_tile;

    std::vector<float> fp32_vec(float_data_size, 0);
    for (size_t i = 0; i < fp32_vec.size(); i++) {
        fp32_vec.at(i) = static_cast<float>(i);
    }

    std::vector<uint32_t> shape_vec = {1, num_tiles, 32, 32};
    std::vector<float> tiled_fp32_vec = convert_layout(
        tt::stl::make_const_span(fp32_vec), shape_vec, TensorLayoutType::LIN_ROW_MAJOR, TensorLayoutType::TILED_NFACES);

    std::vector<uint32_t> packed_bfp4b_tile_vec_rm_in =
        pack_as_bfp4_tiles(tt::stl::make_const_span(fp32_vec), /*row_major_input=*/true, /*is_exp_a=*/false);
    std::vector<float> unpacked_bfp4b_tile_vec_rm_out =
        unpack_bfp4_tiles_into_float_vec(packed_bfp4b_tile_vec_rm_in, /*row_major_output*/ true, /*is_exp_a=*/false);

    std::vector<uint32_t> packed_bfp4b_tile_vec_tile_in =
        pack_as_bfp4_tiles(tt::stl::make_const_span(tiled_fp32_vec), /*row_major_input=*/false, /*is_exp_a=*/false);
    std::vector<float> unpacked_bfp4b_tile_vec_tile_out =
        unpack_bfp4_tiles_into_float_vec(packed_bfp4b_tile_vec_tile_in, /*row_major_output=*/false, /*is_exp_a=*/false);

    std::vector<float> tiled_to_rm_fp32_vec = convert_layout(
        tt::stl::make_const_span(unpacked_bfp4b_tile_vec_tile_out),
        shape_vec,
        TensorLayoutType::TILED_NFACES,
        TensorLayoutType::LIN_ROW_MAJOR);
    std::vector<float> rm_to_tiled_fp32_vec = convert_layout(
        tt::stl::make_const_span(unpacked_bfp4b_tile_vec_rm_out),
        shape_vec,
        TensorLayoutType::LIN_ROW_MAJOR,
        TensorLayoutType::TILED_NFACES);

    EXPECT_EQ(packed_bfp4b_tile_vec_rm_in, packed_bfp4b_tile_vec_tile_in);

    ASSERT_EQ(unpacked_bfp4b_tile_vec_rm_out.size(), fp32_vec.size());
    for (size_t rm_idx = 0; rm_idx < fp32_vec.size(); rm_idx++) {
        float golden = fp32_vec.at(rm_idx);
        float converted = unpacked_bfp4b_tile_vec_rm_out.at(rm_idx);
        float atol = 8.0f;
        float rtol = 0.15f;
        EXPECT_TRUE(is_close(golden, converted, rtol, atol))
            << "Mismatch at index " << rm_idx << ": golden=" << golden << ", converted=" << converted;
    }

    ASSERT_EQ(unpacked_bfp4b_tile_vec_tile_out.size(), tiled_fp32_vec.size());
    for (size_t rm_idx = 0; rm_idx < fp32_vec.size(); rm_idx++) {
        float golden = tiled_fp32_vec.at(rm_idx);
        float converted = unpacked_bfp4b_tile_vec_tile_out.at(rm_idx);
        float atol = 8.0f;
        float rtol = 0.15f;
        EXPECT_TRUE(is_close(golden, converted, rtol, atol))
            << "Mismatch at index " << rm_idx << ": golden=" << golden << ", converted=" << converted;
    }

    EXPECT_EQ(unpacked_bfp4b_tile_vec_rm_out, tiled_to_rm_fp32_vec);
    EXPECT_EQ(unpacked_bfp4b_tile_vec_tile_out, rm_to_tiled_fp32_vec);
}

// =============================================================================
// FAST DISPATCH TESTS
// =============================================================================

TEST_F(FastDispatchFixture, EltwiseBinaryAdd) {
    bool multibank = true;
    auto* md = mesh_device();
    auto& cq = command_queue();

    distributed::MeshWorkload mesh_workload;
    Program program = CreateProgram();
    CoreCoord core = {0, 0};

    uint32_t single_tile_size = 2 * 1024;
    uint32_t num_tiles = 2048;
    uint32_t dram_buffer_size = single_tile_size * num_tiles;
    uint32_t page_size = multibank ? single_tile_size : dram_buffer_size;

    distributed::DeviceLocalBufferConfig device_local_config{
        .page_size = page_size,
        .buffer_type = BufferType::DRAM,
    };

    distributed::ReplicatedBufferConfig buffer_config{
        .size = dram_buffer_size,
    };
    auto src0_dram_buffer = distributed::MeshBuffer::create(buffer_config, device_local_config, md);
    uint32_t dram_buffer_src0_addr = src0_dram_buffer->address();
    auto src1_dram_buffer = distributed::MeshBuffer::create(buffer_config, device_local_config, md);
    uint32_t dram_buffer_src1_addr = src1_dram_buffer->address();
    auto dst_dram_buffer = distributed::MeshBuffer::create(buffer_config, device_local_config, md);
    uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

    uint32_t src0_cb_index = CBIndex::c_0;
    uint32_t num_input_tiles = 2;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t src1_cb_index = CBIndex::c_1;
    CircularBufferConfig cb_src1_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, DataFormat::Float16_b}})
            .set_page_size(src1_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_src1_config);

    uint32_t ouput_cb_index = CBIndex::c_16;
    uint32_t num_output_tiles = 2;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_output_tiles * single_tile_size, {{ouput_cb_index, DataFormat::Float16_b}})
            .set_page_size(ouput_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_output_config);

    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(src0_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(src1_dram_buffer).append_to(reader_compile_time_args);
    auto binary_reader_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_dual_8bank.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args});

    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(dst_dram_buffer).append_to(writer_compile_time_args);
    auto unary_writer_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_8bank.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args});

    std::map<std::string, std::string> binary_defines = {
        {"ELTWISE_OP", "add_tiles"}, {"ELTWISE_OP_TYPE", "EltwiseBinaryType::ELWADD"}};
    auto eltwise_binary_kernel = CreateKernel(
        program, "tt_metal/kernels/compute/eltwise_binary.cpp", core, ComputeConfig{.defines = binary_defines});

    SetRuntimeArgs(program, eltwise_binary_kernel, core, {2048, 1});

    const std::array<uint32_t, 7> reader_args = {
        dram_buffer_src0_addr, 0, num_tiles, dram_buffer_src1_addr, 0, num_tiles, 0};
    const std::array<uint32_t, 3> writer_args = {dram_buffer_dst_addr, 0, num_tiles};

    SetRuntimeArgs(program, unary_writer_kernel, core, writer_args);
    SetRuntimeArgs(program, binary_reader_kernel, core, reader_args);

    mesh_workload.add_program(distributed::MeshCoordinateRange(md->shape()), std::move(program));

    std::vector<uint32_t> src0_vec = create_random_vector_of_bfloat16(
        dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
    distributed::EnqueueWriteMeshBuffer(cq, src0_dram_buffer, src0_vec, false);

    std::vector<uint32_t> src1_vec = create_constant_vector_of_bfloat16(dram_buffer_size, 0.0f);
    distributed::EnqueueWriteMeshBuffer(cq, src1_dram_buffer, src1_vec, false);

    distributed::EnqueueMeshWorkload(cq, mesh_workload, false);
    std::vector<uint32_t> result_vec;
    distributed::ReadShard(cq, result_vec, dst_dram_buffer, distributed::MeshCoordinate(0, 0));

    EXPECT_EQ(src0_vec, result_vec);
}

TEST_F(FastDispatchFixture, EltwiseBinarySub) {
    bool multibank = true;
    auto* md = mesh_device();
    auto& cq = command_queue();

    distributed::MeshWorkload mesh_workload;
    Program program = CreateProgram();
    CoreCoord core = {0, 0};

    uint32_t single_tile_size = 2 * 1024;
    uint32_t num_tiles = 2048;
    uint32_t dram_buffer_size = single_tile_size * num_tiles;
    uint32_t page_size = multibank ? single_tile_size : dram_buffer_size;

    distributed::DeviceLocalBufferConfig device_local_config{
        .page_size = page_size,
        .buffer_type = BufferType::DRAM,
    };

    distributed::ReplicatedBufferConfig buffer_config{
        .size = dram_buffer_size,
    };
    auto src0_dram_buffer = distributed::MeshBuffer::create(buffer_config, device_local_config, md);
    uint32_t dram_buffer_src0_addr = src0_dram_buffer->address();
    auto src1_dram_buffer = distributed::MeshBuffer::create(buffer_config, device_local_config, md);
    uint32_t dram_buffer_src1_addr = src1_dram_buffer->address();
    auto dst_dram_buffer = distributed::MeshBuffer::create(buffer_config, device_local_config, md);
    uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

    uint32_t src0_cb_index = CBIndex::c_0;
    uint32_t num_input_tiles = 2;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t src1_cb_index = CBIndex::c_1;
    CircularBufferConfig cb_src1_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, DataFormat::Float16_b}})
            .set_page_size(src1_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_src1_config);

    uint32_t ouput_cb_index = CBIndex::c_16;
    uint32_t num_output_tiles = 2;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_output_tiles * single_tile_size, {{ouput_cb_index, DataFormat::Float16_b}})
            .set_page_size(ouput_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_output_config);

    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(src0_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(src1_dram_buffer).append_to(reader_compile_time_args);
    auto binary_reader_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_dual_8bank.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args});

    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(dst_dram_buffer).append_to(writer_compile_time_args);
    auto unary_writer_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_8bank.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args});

    std::map<std::string, std::string> binary_defines = {
        {"ELTWISE_OP", "sub_tiles"}, {"ELTWISE_OP_TYPE", "EltwiseBinaryType::ELWSUB"}};
    auto eltwise_binary_kernel = CreateKernel(
        program, "tt_metal/kernels/compute/eltwise_binary.cpp", core, ComputeConfig{.defines = binary_defines});

    SetRuntimeArgs(program, eltwise_binary_kernel, core, {2048, 1});

    const std::array<uint32_t, 7> reader_args = {
        dram_buffer_src0_addr, 0, num_tiles, dram_buffer_src1_addr, 0, num_tiles, 0};
    const std::array<uint32_t, 3> writer_args = {dram_buffer_dst_addr, 0, num_tiles};

    SetRuntimeArgs(program, unary_writer_kernel, core, writer_args);
    SetRuntimeArgs(program, binary_reader_kernel, core, reader_args);

    mesh_workload.add_program(distributed::MeshCoordinateRange(md->shape()), std::move(program));

    std::vector<uint32_t> src0_vec = create_random_vector_of_bfloat16(
        dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
    distributed::EnqueueWriteMeshBuffer(cq, src0_dram_buffer, src0_vec, false);

    std::vector<uint32_t> src1_vec = create_constant_vector_of_bfloat16(dram_buffer_size, 0.0f);
    distributed::EnqueueWriteMeshBuffer(cq, src1_dram_buffer, src1_vec, false);

    distributed::EnqueueMeshWorkload(cq, mesh_workload, false);
    std::vector<uint32_t> result_vec;
    distributed::ReadShard(cq, result_vec, dst_dram_buffer, distributed::MeshCoordinate(0, 0));

    EXPECT_EQ(src0_vec, result_vec);
}

TEST_F(FastDispatchFixture, EltwiseBinaryMul) {
    bool multibank = true;
    auto* md = mesh_device();
    auto& cq = command_queue();

    distributed::MeshWorkload mesh_workload;
    Program program = CreateProgram();
    CoreCoord core = {0, 0};

    uint32_t single_tile_size = 2 * 1024;
    uint32_t num_tiles = 2048;
    uint32_t dram_buffer_size = single_tile_size * num_tiles;
    uint32_t page_size = multibank ? single_tile_size : dram_buffer_size;

    distributed::DeviceLocalBufferConfig device_local_config{
        .page_size = page_size,
        .buffer_type = BufferType::DRAM,
    };

    distributed::ReplicatedBufferConfig buffer_config{
        .size = dram_buffer_size,
    };
    auto src0_dram_buffer = distributed::MeshBuffer::create(buffer_config, device_local_config, md);
    uint32_t dram_buffer_src0_addr = src0_dram_buffer->address();
    auto src1_dram_buffer = distributed::MeshBuffer::create(buffer_config, device_local_config, md);
    uint32_t dram_buffer_src1_addr = src1_dram_buffer->address();
    auto dst_dram_buffer = distributed::MeshBuffer::create(buffer_config, device_local_config, md);
    uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

    uint32_t src0_cb_index = CBIndex::c_0;
    uint32_t num_input_tiles = 2;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t src1_cb_index = CBIndex::c_1;
    CircularBufferConfig cb_src1_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, DataFormat::Float16_b}})
            .set_page_size(src1_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_src1_config);

    uint32_t ouput_cb_index = CBIndex::c_16;
    uint32_t num_output_tiles = 2;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_output_tiles * single_tile_size, {{ouput_cb_index, DataFormat::Float16_b}})
            .set_page_size(ouput_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_output_config);

    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(src0_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(src1_dram_buffer).append_to(reader_compile_time_args);
    auto binary_reader_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_dual_8bank.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args});

    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(dst_dram_buffer).append_to(writer_compile_time_args);
    auto unary_writer_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_8bank.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args});

    std::map<std::string, std::string> binary_defines = {
        {"ELTWISE_OP", "mul_tiles"}, {"ELTWISE_OP_TYPE", "EltwiseBinaryType::ELWMUL"}};
    auto eltwise_binary_kernel = CreateKernel(
        program, "tt_metal/kernels/compute/eltwise_binary.cpp", core, ComputeConfig{.defines = binary_defines});

    SetRuntimeArgs(program, eltwise_binary_kernel, core, {2048, 1});

    const std::array<uint32_t, 7> reader_args = {
        dram_buffer_src0_addr, 0, num_tiles, dram_buffer_src1_addr, 0, num_tiles, 0};
    const std::array<uint32_t, 3> writer_args = {dram_buffer_dst_addr, 0, num_tiles};

    SetRuntimeArgs(program, unary_writer_kernel, core, writer_args);
    SetRuntimeArgs(program, binary_reader_kernel, core, reader_args);

    mesh_workload.add_program(distributed::MeshCoordinateRange(md->shape()), std::move(program));

    std::vector<uint32_t> src0_vec = create_random_vector_of_bfloat16(
        dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
    distributed::EnqueueWriteMeshBuffer(cq, src0_dram_buffer, src0_vec, false);

    std::vector<uint32_t> src1_vec = create_constant_vector_of_bfloat16(dram_buffer_size, 1.0f);
    distributed::EnqueueWriteMeshBuffer(cq, src1_dram_buffer, src1_vec, false);

    distributed::EnqueueMeshWorkload(cq, mesh_workload, false);
    std::vector<uint32_t> result_vec;
    distributed::ReadShard(cq, result_vec, dst_dram_buffer, distributed::MeshCoordinate(0, 0));

    EXPECT_EQ(src0_vec, result_vec);
}

// =============================================================================
// SLOW DISPATCH TESTS
// =============================================================================

TEST_F(SlowDispatchFixture, AddTwoInts) {
    auto* dev = device();
    uint32_t l1_unreserved_base = dev->allocator()->get_base_allocator_addr(HalMemType::L1);

    Program program = CreateProgram();
    CoreCoord core = {0, 0};
    constexpr std::array<uint32_t, 2> first_runtime_args = {101, 202};
    constexpr std::array<uint32_t, 2> second_runtime_args = {303, 606};

    KernelHandle add_two_ints_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/add_two_ints.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {l1_unreserved_base}});

    SetRuntimeArgs(program, add_two_ints_kernel, core, first_runtime_args);
    detail::LaunchProgram(dev, program);

    std::vector<uint32_t> first_kernel_result;
    detail::ReadFromDeviceL1(dev, core, l1_unreserved_base, sizeof(int), first_kernel_result);

    SetRuntimeArgs(program, add_two_ints_kernel, core, second_runtime_args);
    detail::LaunchProgram(dev, program);

    std::vector<uint32_t> second_kernel_result;
    detail::ReadFromDeviceL1(dev, core, l1_unreserved_base, sizeof(int), second_kernel_result);

    uint32_t first_expected_result = first_runtime_args[0] + first_runtime_args[1];
    uint32_t second_expected_result = second_runtime_args[0] + second_runtime_args[1];

    EXPECT_EQ(first_kernel_result[0], first_expected_result);
    EXPECT_EQ(second_kernel_result[0], second_expected_result);
}

TEST_F(SlowDispatchFixture, Datacopy) {
    auto* dev = device();

    Program program = CreateProgram();
    CoreCoord core = {0, 0};

    uint32_t single_tile_size = 2 * 1024;
    uint32_t num_tiles = 2048;
    uint32_t dram_buffer_size = single_tile_size * num_tiles;

    InterleavedBufferConfig dram_config{
        .device = dev, .size = dram_buffer_size, .page_size = dram_buffer_size, .buffer_type = BufferType::DRAM};
    auto src_dram_buffer = CreateBuffer(dram_config);
    auto dst_dram_buffer = CreateBuffer(dram_config);

    uint32_t src0_cb_index = CBIndex::c_0;
    uint32_t num_input_tiles = 8;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t output_cb_index = CBIndex::c_16;
    uint32_t num_output_tiles = 1;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_output_tiles * single_tile_size, {{output_cb_index, DataFormat::Float16_b}})
            .set_page_size(output_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_output_config);

    auto unary_reader_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_4.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    auto unary_writer_kernel = CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    std::vector<uint32_t> compute_kernel_args = {uint(num_tiles)};
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_3m.cpp",
        core,
        ComputeConfig{.compile_args = compute_kernel_args});

    std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
        dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
    detail::WriteToBuffer(src_dram_buffer, src_vec);

    SetRuntimeArgs(program, unary_reader_kernel, core, {src_dram_buffer->address(), 0u, num_tiles});
    SetRuntimeArgs(program, unary_writer_kernel, core, {dst_dram_buffer->address(), 0u, num_tiles});

    detail::LaunchProgram(dev, program);

    std::vector<uint32_t> result_vec;
    detail::ReadFromBuffer(dst_dram_buffer, result_vec);

    EXPECT_EQ(src_vec, result_vec);
}

TEST_F(SlowDispatchFixture, DatacopyBfp8b) {
    auto* dev = device();

    Program program = CreateProgram();
    CoreCoord core = {0, 0};

    uint32_t single_tile_size = tile_size(DataFormat::Bfp8_b);
    ASSERT_EQ(single_tile_size, (256 * 4) + (16 * 4));
    uint32_t num_tiles = 2048;
    uint32_t dram_buffer_size = single_tile_size * num_tiles;

    InterleavedBufferConfig dram_config{
        .device = dev, .size = dram_buffer_size, .page_size = dram_buffer_size, .buffer_type = BufferType::DRAM};
    auto src_dram_buffer = CreateBuffer(dram_config);
    auto dst_dram_buffer = CreateBuffer(dram_config);

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 1;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, DataFormat::Bfp8_b}})
            .set_page_size(src0_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t output_cb_index = CBIndex::c_16;
    uint32_t num_output_tiles = 1;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_output_tiles * single_tile_size, {{output_cb_index, DataFormat::Bfp8_b}})
            .set_page_size(output_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_output_config);

    auto unary_reader_kernel = CreateKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    auto unary_writer_kernel = CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    std::vector<uint32_t> compute_kernel_args = {uint(num_tiles)};
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_3m.cpp",
        core,
        ComputeConfig{.compile_args = compute_kernel_args});

    std::vector<uint32_t> src_vec = test_utils::create_random_vector_of_bfp8(
        dram_buffer_size, /*is_exp_a=*/false, 100, std::chrono::system_clock::now().time_since_epoch().count());
    detail::WriteToBuffer(src_dram_buffer, src_vec);

    SetRuntimeArgs(program, unary_reader_kernel, core, {src_dram_buffer->address(), 0u, num_tiles});
    SetRuntimeArgs(program, unary_writer_kernel, core, {dst_dram_buffer->address(), 0u, num_tiles});

    detail::LaunchProgram(dev, program);

    std::vector<uint32_t> result_vec;
    detail::ReadFromBuffer(dst_dram_buffer, result_vec);

    EXPECT_EQ(src_vec, result_vec);
}

TEST_F(SlowDispatchFixture, DramLoopbackSingleCore) {
    auto* dev = device();

    Program program = CreateProgram();
    CoreCoord core = {0, 0};

    uint32_t single_tile_size = 2 * 1024;
    uint32_t num_tiles = 50;
    uint32_t dram_buffer_size = single_tile_size * num_tiles;
    uint32_t l1_buffer_addr = 400 * 1024;

    InterleavedBufferConfig dram_config{
        .device = dev, .size = dram_buffer_size, .page_size = dram_buffer_size, .buffer_type = BufferType::DRAM};
    auto input_dram_buffer = CreateBuffer(dram_config);
    auto output_dram_buffer = CreateBuffer(dram_config);

    auto dram_copy_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_copy.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    std::vector<uint32_t> input_vec = create_random_vector_of_bfloat16(
        dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
    detail::WriteToBuffer(input_dram_buffer, input_vec);

    SetRuntimeArgs(
        program,
        dram_copy_kernel,
        core,
        {l1_buffer_addr, input_dram_buffer->address(), 0u, output_dram_buffer->address(), 0u, dram_buffer_size});

    detail::LaunchProgram(dev, program);

    std::vector<uint32_t> result_vec;
    detail::ReadFromBuffer(output_dram_buffer, result_vec);

    EXPECT_EQ(input_vec, result_vec);
}

TEST_F(SlowDispatchFixture, DataflowCB) {
    auto* dev = device();

    Program program = CreateProgram();
    CoreCoord core = {0, 0};

    uint32_t single_tile_size = 2 * 1024;
    uint32_t num_tiles = 2048;
    uint32_t dram_buffer_size = single_tile_size * num_tiles;

    InterleavedBufferConfig dram_config{
        .device = dev, .size = dram_buffer_size, .page_size = dram_buffer_size, .buffer_type = BufferType::DRAM};
    auto src_dram_buffer = CreateBuffer(dram_config);
    auto dst_dram_buffer = CreateBuffer(dram_config);

    int num_cbs = 1;
    int num_tiles_per_cb = num_tiles / num_cbs;
    uint32_t num_cb_tiles = 8;

    uint32_t cb0_index = 0;
    CircularBufferConfig cb0_config =
        CircularBufferConfig(num_cb_tiles * single_tile_size, {{cb0_index, DataFormat::Float16_b}})
            .set_page_size(cb0_index, single_tile_size);
    CreateCircularBuffer(program, core, cb0_config);

    uint32_t cb1_index = 8;
    CircularBufferConfig cb1_config =
        CircularBufferConfig(num_cb_tiles * single_tile_size, {{cb1_index, DataFormat::Float16_b}})
            .set_page_size(cb1_index, single_tile_size);
    CreateCircularBuffer(program, core, cb1_config);

    uint32_t cb2_index = 16;
    CircularBufferConfig cb2_config =
        CircularBufferConfig(num_cb_tiles * single_tile_size, {{cb2_index, DataFormat::Float16_b}})
            .set_page_size(cb2_index, single_tile_size);
    CreateCircularBuffer(program, core, cb2_config);

    uint32_t cb3_index = 24;
    CircularBufferConfig cb3_config =
        CircularBufferConfig(num_cb_tiles * single_tile_size, {{cb3_index, DataFormat::Float16_b}})
            .set_page_size(cb3_index, single_tile_size);
    CreateCircularBuffer(program, core, cb3_config);

    std::vector<uint32_t> reader_cb_kernel_args = {8, 2};
    std::vector<uint32_t> writer_cb_kernel_args = {8, 4};

    auto reader_cb_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_cb_test.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_cb_kernel_args});

    auto writer_cb_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_cb_test.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_cb_kernel_args});

    std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
        dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
    detail::WriteToBuffer(src_dram_buffer, src_vec);

    SetRuntimeArgs(
        program, reader_cb_kernel, core, {src_dram_buffer->address(), 0u, static_cast<uint32_t>(num_tiles_per_cb)});
    SetRuntimeArgs(
        program, writer_cb_kernel, core, {dst_dram_buffer->address(), 0u, static_cast<uint32_t>(num_tiles_per_cb)});

    detail::LaunchProgram(dev, program);

    std::vector<uint32_t> result_vec;
    detail::ReadFromBuffer(dst_dram_buffer, result_vec);

    EXPECT_EQ(src_vec, result_vec);
}

TEST_F(SlowDispatchFixture, MatmulSingleTile) {
    auto* dev = device();

    Program program = CreateProgram();
    CoreCoord core = {0, 0};

    uint32_t single_tile_size = 2 * 1024;
    uint32_t num_tiles = 1;
    uint32_t dram_buffer_size = single_tile_size * num_tiles;

    InterleavedBufferConfig dram_config{
        .device = dev, .size = dram_buffer_size, .page_size = dram_buffer_size, .buffer_type = BufferType::DRAM};
    auto src0_dram_buffer = CreateBuffer(dram_config);
    auto src1_dram_buffer = CreateBuffer(dram_config);
    auto dst_dram_buffer = CreateBuffer(dram_config);

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t src1_cb_index = 1;
    CircularBufferConfig cb_src1_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, DataFormat::Float16_b}})
            .set_page_size(src1_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_src1_config);

    uint32_t output_cb_index = CBIndex::c_16;
    uint32_t num_output_tiles = 2;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_output_tiles * single_tile_size, {{output_cb_index, DataFormat::Float16_b}})
            .set_page_size(output_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_output_config);

    auto mm_reader_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_blocked.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    auto unary_writer_kernel = CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    std::vector<uint32_t> compute_kernel_args = {1, 1, 1, 1, 1, 1, 1};
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/matmul.cpp",
        core,
        ComputeConfig{.compile_args = compute_kernel_args});

    SHAPE shape = {1, 1, 32, 32};
    tt::deprecated::Tensor<bfloat16> tensor = tt::deprecated::initialize_tensor<bfloat16>(
        shape, tt::deprecated::Initialize::RANDOM, 0, 100, std::chrono::system_clock::now().time_since_epoch().count());
    auto activations_tile_layout =
        convert_layout_tile_swizzled_to_tile_nfaces(tt::stl::make_const_span(tensor.get_values()));
    auto activations = pack_bfloat16_vec_into_uint32_vec(activations_tile_layout);
    detail::WriteToBuffer(src0_dram_buffer, activations);

    auto identity = create_identity_matrix(32, 32, 32);
    auto weights_tile_layout = convert_layout_tile_swizzled_to_tile_nfaces(tt::stl::make_const_span(identity));
    auto weights = pack_bfloat16_vec_into_uint32_vec(weights_tile_layout);
    detail::WriteToBuffer(src1_dram_buffer, weights);

    SetRuntimeArgs(
        program,
        mm_reader_kernel,
        core,
        {src0_dram_buffer->address(),
         0u,
         src1_dram_buffer->address(),
         0u,
         1u,
         1u,
         1u,
         1 * single_tile_size,
         1 * single_tile_size});

    SetRuntimeArgs(program, unary_writer_kernel, core, {dst_dram_buffer->address(), 0u, num_tiles});

    detail::LaunchProgram(dev, program);

    std::vector<uint32_t> result_vec;
    detail::ReadFromBuffer(dst_dram_buffer, result_vec);

    auto result_bfp16 = unpack_uint32_vec_into_bfloat16_vec(result_vec);
    auto result_flat_layout = convert_layout_tile_nfaces_to_tile_swizzled(tt::stl::make_const_span(result_bfp16));
    EXPECT_EQ(tensor.get_values(), result_flat_layout);
}

TEST_F(SlowDispatchFixture, MultiplePrograms) {
    auto* dev = device();

    CoreCoord core = {0, 0};
    uint32_t single_tile_size = 2 * 1024;
    uint32_t num_tiles = 1;
    uint32_t dram_buffer_size = single_tile_size * num_tiles;

    InterleavedBufferConfig dram_config{
        .device = dev, .size = dram_buffer_size, .page_size = dram_buffer_size, .buffer_type = BufferType::DRAM};

    auto src0_dram_buffer = CreateBuffer(dram_config);
    auto src1_dram_buffer = CreateBuffer(dram_config);
    auto dst_dram_buffer = CreateBuffer(dram_config);

    auto setup_program_one = [&](IDevice* dev,
                                 const CoreCoord& core,
                                 uint32_t single_tile_size) -> std::tuple<Program, KernelHandle, KernelHandle> {
        Program program = CreateProgram();

        uint32_t src0_cb_index = 0;
        uint32_t num_input_tiles = 2;
        CircularBufferConfig cb_src0_config =
            CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, DataFormat::Float16_b}})
                .set_page_size(src0_cb_index, single_tile_size);
        CreateCircularBuffer(program, core, cb_src0_config);

        uint32_t src1_cb_index = 1;
        CircularBufferConfig cb_src1_config =
            CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, DataFormat::Float16_b}})
                .set_page_size(src1_cb_index, single_tile_size);
        CreateCircularBuffer(program, core, cb_src1_config);

        uint32_t output_cb_index = CBIndex::c_16;
        uint32_t num_output_tiles = 2;
        CircularBufferConfig cb_output_config =
            CircularBufferConfig(num_output_tiles * single_tile_size, {{output_cb_index, DataFormat::Float16_b}})
                .set_page_size(output_cb_index, single_tile_size);
        CreateCircularBuffer(program, core, cb_output_config);

        auto binary_reader_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

        auto unary_writer_kernel = CreateKernel(
            program,
            "tt_metal/kernels/dataflow/writer_unary.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

        std::map<std::string, std::string> binary_defines = {
            {"ELTWISE_OP", "add_tiles"}, {"ELTWISE_OP_TYPE", "EltwiseBinaryType::ELWADD"}};
        auto eltwise_binary_kernel = CreateKernel(
            program,
            "tt_metal/kernels/compute/eltwise_binary.cpp",
            core,
            ComputeConfig{.compile_args = {}, .defines = binary_defines});

        SetRuntimeArgs(program, eltwise_binary_kernel, core, {1, 1});

        return {std::move(program), binary_reader_kernel, unary_writer_kernel};
    };

    auto setup_program_two = [&](IDevice* dev,
                                 const CoreCoord& core,
                                 uint32_t single_tile_size) -> std::tuple<Program, KernelHandle, KernelHandle> {
        Program program = CreateProgram();

        uint32_t src0_cb_index = 0;
        uint32_t num_input_tiles = 2;
        CircularBufferConfig cb_src0_config =
            CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, DataFormat::Float16_b}})
                .set_page_size(src0_cb_index, single_tile_size);
        CreateCircularBuffer(program, core, cb_src0_config);

        uint32_t src1_cb_index = 1;
        CircularBufferConfig cb_src1_config =
            CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, DataFormat::Float16_b}})
                .set_page_size(src1_cb_index, single_tile_size);
        CreateCircularBuffer(program, core, cb_src1_config);

        uint32_t output_cb_index = CBIndex::c_16;
        uint32_t num_output_tiles = 2;
        CircularBufferConfig cb_output_config =
            CircularBufferConfig(num_output_tiles * single_tile_size, {{output_cb_index, DataFormat::Float16_b}})
                .set_page_size(output_cb_index, single_tile_size);
        CreateCircularBuffer(program, core, cb_output_config);

        auto mm_reader_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_small_block.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

        auto unary_writer_kernel = CreateKernel(
            program,
            "tt_metal/kernels/dataflow/writer_unary.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

        std::vector<uint32_t> compute_kernel_args = {1, 1, 1, 1, 1, 1, 1};
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/compute/matmul.cpp",
            core,
            ComputeConfig{.compile_args = compute_kernel_args});

        return {std::move(program), mm_reader_kernel, unary_writer_kernel};
    };

    auto [program1, reader1_kernel_id, writer1_kernel_id] = setup_program_one(dev, core, single_tile_size);
    auto [program2, reader2_kernel_id, writer2_kernel_id] = setup_program_two(dev, core, single_tile_size);

    SHAPE shape = {1, 1, 32, 32};
    tt::deprecated::Tensor<bfloat16> src0_tensor = tt::deprecated::initialize_tensor<bfloat16>(
        shape, tt::deprecated::Initialize::RANDOM, 0, 100, std::chrono::system_clock::now().time_since_epoch().count());
    auto src0_activations_tile_layout =
        convert_layout_tile_swizzled_to_tile_nfaces(tt::stl::make_const_span(src0_tensor.get_values()));
    auto src0_activations = pack_bfloat16_vec_into_uint32_vec(src0_activations_tile_layout);
    detail::WriteToBuffer(src0_dram_buffer, src0_activations);

    tt::deprecated::Tensor<bfloat16> src1_tensor = tt::deprecated::initialize_tensor<bfloat16>(
        shape, tt::deprecated::Initialize::ZEROS, 0, 100, std::chrono::system_clock::now().time_since_epoch().count());
    auto src1_activations_tile_layout =
        convert_layout_tile_swizzled_to_tile_nfaces(tt::stl::make_const_span(src1_tensor.get_values()));
    auto src1_activations = pack_bfloat16_vec_into_uint32_vec(src1_activations_tile_layout);
    detail::WriteToBuffer(src1_dram_buffer, src1_activations);

    SetRuntimeArgs(
        program1,
        reader1_kernel_id,
        core,
        {src0_dram_buffer->address(), 0u, src1_dram_buffer->address(), 0u, num_tiles});
    SetRuntimeArgs(program1, writer1_kernel_id, core, {dst_dram_buffer->address(), 0u, num_tiles});

    detail::LaunchProgram(dev, program1);

    std::vector<uint32_t> intermediate_result_vec;
    detail::ReadFromBuffer(dst_dram_buffer, intermediate_result_vec);

    EXPECT_EQ(src0_activations, intermediate_result_vec) << "Eltwise binary did not run correctly!";

    auto identity = create_identity_matrix(32, 32, 32);
    auto weights_tile_layout = convert_layout_tile_swizzled_to_tile_nfaces(tt::stl::make_const_span(identity));
    auto weights = pack_bfloat16_vec_into_uint32_vec(weights_tile_layout);
    detail::WriteToBuffer(src1_dram_buffer, weights);

    SetRuntimeArgs(
        program2,
        reader2_kernel_id,
        core,
        {src0_dram_buffer->address(), 0u, src1_dram_buffer->address(), 0u, num_tiles});
    SetRuntimeArgs(program2, writer2_kernel_id, core, {dst_dram_buffer->address(), 0u, num_tiles});

    detail::LaunchProgram(dev, program2);

    std::vector<uint32_t> result_vec;
    detail::ReadFromBuffer(dst_dram_buffer, result_vec);

    EXPECT_EQ(intermediate_result_vec, result_vec);
}

TEST_F(SlowDispatchFixture, InterleavedL1Buffer) {
    auto* dev = device();

    uint32_t page_size = 2 * 1024;
    int num_pages_one = 258;
    int num_pages_two = 378;

    uint32_t buffer_size = num_pages_one * page_size;

    InterleavedBufferConfig buff_config_0{
        .device = dev, .size = buffer_size, .page_size = page_size, .buffer_type = BufferType::L1};
    auto interleaved_buffer = CreateBuffer(buff_config_0);

    std::vector<uint32_t> host_buffer =
        create_random_vector_of_bfloat16(buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());

    detail::WriteToBuffer(interleaved_buffer, host_buffer);

    std::vector<uint32_t> readback_buffer;
    detail::ReadFromBuffer(interleaved_buffer, readback_buffer);

    EXPECT_EQ(host_buffer, readback_buffer);

    uint32_t second_buffer_size = num_pages_two * page_size;

    InterleavedBufferConfig buff_config_1{
        .device = dev, .size = second_buffer_size, .page_size = page_size, .buffer_type = BufferType::L1};
    auto second_interleaved_buffer = CreateBuffer(buff_config_1);

    std::vector<uint32_t> second_host_buffer = create_random_vector_of_bfloat16(
        second_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());

    detail::WriteToBuffer(second_interleaved_buffer, second_host_buffer);

    std::vector<uint32_t> second_readback_buffer;
    detail::ReadFromBuffer(second_interleaved_buffer, second_readback_buffer);

    EXPECT_EQ(second_host_buffer, second_readback_buffer);
}

TEST_F(SlowDispatchFixture, MultiCoreKernelSameRuntimeArgs) {
    auto* dev = device();

    CoreCoord start_core = {0, 0};
    CoreCoord end_core = {2, 2};
    CoreRange all_cores(start_core, end_core);

    uint32_t single_tile_size = 2 * 1024;
    int32_t num_tiles = 2048;
    uint32_t dram_buffer_size = single_tile_size * num_tiles;

    InterleavedBufferConfig dram_config{
        .device = dev, .size = dram_buffer_size, .page_size = dram_buffer_size, .buffer_type = BufferType::DRAM};
    auto src_dram_buffer = CreateBuffer(dram_config);
    auto dst_dram_buffer = CreateBuffer(dram_config);

    std::vector<uint32_t> compute_kernel_args = {uint(num_tiles)};

    Program program = CreateProgram();

    for (auto x = start_core.x; x <= end_core.x; x++) {
        for (auto y = start_core.y; y <= end_core.y; y++) {
            auto core = CoreCoord{x, y};
            uint32_t src0_cb_index = 0;
            uint32_t num_input_tiles = 8;
            CircularBufferConfig cb_src0_config =
                CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, DataFormat::Float16_b}})
                    .set_page_size(src0_cb_index, single_tile_size);
            CreateCircularBuffer(program, core, cb_src0_config);

            uint32_t output_cb_index = CBIndex::c_16;
            uint32_t num_output_tiles = 1;
            CircularBufferConfig cb_output_config =
                CircularBufferConfig(num_output_tiles * single_tile_size, {{output_cb_index, DataFormat::Float16_b}})
                    .set_page_size(output_cb_index, single_tile_size);
            CreateCircularBuffer(program, core, cb_output_config);
        }
    }

    auto reader_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_4.cpp",
        all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    auto writer_kernel = CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy.cpp",
        all_cores,
        ComputeConfig{.compile_args = compute_kernel_args});

    std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
        src_dram_buffer->size(), 100, std::chrono::system_clock::now().time_since_epoch().count());
    detail::WriteToBuffer(src_dram_buffer, src_vec);

    for (auto x = start_core.x; x <= end_core.x; x++) {
        for (auto y = start_core.y; y <= end_core.y; y++) {
            CoreCoord core(x, y);
            SetRuntimeArgs(
                program, reader_kernel, core, {src_dram_buffer->address(), 0u, static_cast<uint32_t>(num_tiles)});
            SetRuntimeArgs(
                program, writer_kernel, core, {dst_dram_buffer->address(), 0u, static_cast<uint32_t>(num_tiles)});
        }
    }

    detail::LaunchProgram(dev, program);

    std::vector<uint32_t> result_vec;
    detail::ReadFromBuffer(dst_dram_buffer, result_vec);

    EXPECT_EQ(src_vec, result_vec);
}

TEST_F(SlowDispatchFixture, UnalignedReadWriteDRAMInterleaved) {
    auto* dev = device();

    uint32_t single_tile_size = 2 * 1024;
    uint32_t num_tiles = 2048;
    uint32_t dram_buffer_size = (single_tile_size * num_tiles) + 2;

    InterleavedBufferConfig dram_interleaved_buffer_config{
        .device = dev, .size = dram_buffer_size, .page_size = dram_buffer_size, .buffer_type = BufferType::DRAM};
    auto device_dram_interleaved_buffer = CreateBuffer(dram_interleaved_buffer_config);

    std::vector<uint8_t> src_vec_dram_interleaved_case(dram_buffer_size);
    for (auto& v : src_vec_dram_interleaved_case) {
        v = static_cast<uint8_t>(std::rand() % 256);
    }
    detail::WriteToBuffer(device_dram_interleaved_buffer, src_vec_dram_interleaved_case);
    std::vector<uint8_t> result_vec_dram_interleaved_case;
    detail::ReadFromBuffer(device_dram_interleaved_buffer, result_vec_dram_interleaved_case);
    EXPECT_EQ(src_vec_dram_interleaved_case, result_vec_dram_interleaved_case);
}

TEST_F(SlowDispatchFixture, UnalignedReadWriteDRAMSharded) {
    auto* dev = device();

    uint32_t single_tile_size = 2 * 1024;
    uint32_t num_tiles = 2048;
    uint32_t dram_buffer_size = (single_tile_size * num_tiles) + 2;

    CoreRangeSet shard_grid(std::set<CoreRange>({CoreRange(CoreCoord(0, 0), CoreCoord(0, 0))}));
    auto shard_spec = ShardSpecBuffer(
        shard_grid,
        {1, dram_buffer_size / sizeof(uint16_t)},
        ShardOrientation::ROW_MAJOR,
        {1, dram_buffer_size / sizeof(uint16_t)},
        {1, 1});
    auto device_dram_sharded_buffer = CreateBuffer(ShardedBufferConfig{
        .device = dev,
        .size = dram_buffer_size,
        .page_size = dram_buffer_size,
        .buffer_type = BufferType::DRAM,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = shard_spec});

    std::vector<uint8_t> src_vec_dram_sharded_case(dram_buffer_size);
    for (auto& v : src_vec_dram_sharded_case) {
        v = static_cast<uint8_t>(std::rand() % 256);
    }

    detail::WriteToBuffer(device_dram_sharded_buffer, src_vec_dram_sharded_case);

    std::vector<uint8_t> result_vec_dram_sharded_case;
    detail::ReadFromBuffer(device_dram_sharded_buffer, result_vec_dram_sharded_case);
    EXPECT_EQ(src_vec_dram_sharded_case, result_vec_dram_sharded_case);
}

TEST_F(SlowDispatchFixture, UnalignedReadWriteL1Interleaved) {
    auto* dev = device();

    uint32_t single_tile_size = 2 * 1024;
    uint32_t l1_buffer_size = (single_tile_size * 4) + 2;

    InterleavedBufferConfig l1_interleaved_buffer_config{
        .device = dev, .size = l1_buffer_size, .page_size = l1_buffer_size, .buffer_type = BufferType::L1};
    auto device_l1_interleaved_buffer = CreateBuffer(l1_interleaved_buffer_config);

    std::vector<uint8_t> src_vec_l1_interleaved_case(l1_buffer_size);
    for (auto& v : src_vec_l1_interleaved_case) {
        v = static_cast<uint8_t>(std::rand() % 256);
    }

    detail::WriteToBuffer(device_l1_interleaved_buffer, src_vec_l1_interleaved_case);

    std::vector<uint8_t> result_vec_l1_interleaved_case;
    detail::ReadFromBuffer(device_l1_interleaved_buffer, result_vec_l1_interleaved_case);
    EXPECT_EQ(src_vec_l1_interleaved_case, result_vec_l1_interleaved_case);
}

TEST_F(SlowDispatchFixture, UnalignedReadWriteL1Sharded) {
    auto* dev = device();

    uint32_t single_tile_size = 2 * 1024;
    uint32_t l1_buffer_size = (single_tile_size * 4) + 2;

    CoreRangeSet l1_shard_grid(std::set<CoreRange>({CoreRange(CoreCoord(0, 0), CoreCoord(0, 0))}));
    auto l1_shard_spec = ShardSpecBuffer(
        l1_shard_grid,
        {1, l1_buffer_size / sizeof(uint16_t)},
        ShardOrientation::ROW_MAJOR,
        {1, l1_buffer_size / sizeof(uint16_t)},
        {1, 1});
    auto device_l1_sharded_buffer = CreateBuffer(ShardedBufferConfig{
        .device = dev,
        .size = l1_buffer_size,
        .page_size = l1_buffer_size,
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = l1_shard_spec});

    std::vector<uint8_t> src_vec_l1_sharded_case(l1_buffer_size);
    for (auto& v : src_vec_l1_sharded_case) {
        v = static_cast<uint8_t>(std::rand() % 256);
    }

    detail::WriteToBuffer(device_l1_sharded_buffer, src_vec_l1_sharded_case);

    std::vector<uint8_t> result_vec_l1_sharded_case;
    detail::ReadFromBuffer(device_l1_sharded_buffer, result_vec_l1_sharded_case);
    EXPECT_EQ(src_vec_l1_sharded_case, result_vec_l1_sharded_case);
}

// =============================================================================
// EITHER DISPATCH TESTS (run in both slow and fast dispatch modes)
// Migrated from test_compile_program.cpp and test_compile_sets_kernel_binaries.cpp
// =============================================================================

}  // namespace

// These tests need additional internal headers
#include <filesystem>
#include <fmt/base.h>
#include <tt_stl/assert.hpp>
#include <tt-metalium/base_types.hpp>
#include "jit_build/build.hpp"
#include "tt_metal/detail/kernel_cache.hpp"
#include "tt_metal/jit_build/build_env_manager.hpp"
#include <umd/device/types/arch.hpp>
#include "impl/program/program_impl.hpp"
#include "impl/kernels/kernel.hpp"

namespace {

// ============================================================================
// test_compile_program - Tests program compilation and kernel caching
// ============================================================================

struct KernelCacheStatus {
    std::unordered_map<std::string, std::string> kernel_name_to_hash_str;
    std::unordered_map<std::string, bool> kernel_name_to_cache_hit;
};

void ClearKernelCache(const std::string& kernel_root_path) {
    std::filesystem::remove_all(kernel_root_path);
    detail::HashLookup::inst().clear();
}

std::unordered_map<std::string, std::string> get_last_program_binary_path(
    const Program& program, const std::string& kernel_root_path) {
    std::unordered_map<std::string, std::string> kernel_name_to_last_compiled_dir;
    for (size_t kernel_id = 0; kernel_id < program.impl().num_kernels(); kernel_id++) {
        auto kernel = program.impl().get_kernel(kernel_id);
        if (not std::filesystem::exists(kernel_root_path + kernel->name())) {
            continue;
        }

        std::filesystem::path kernel_path{kernel_root_path + kernel->name()};
        std::filesystem::file_time_type ftime = std::filesystem::last_write_time(*kernel_path.begin());
        std::string latest_hash;
        for (const auto& dir_entry : std::filesystem::directory_iterator{kernel_path}) {
            auto kbtime = std::filesystem::last_write_time(dir_entry.path());
            if (kbtime > ftime) {
                ftime = kbtime;
                latest_hash = dir_entry.path().filename().string();
            }
        }
        TT_FATAL(not latest_hash.empty(), "Error");
        kernel_name_to_last_compiled_dir.insert({kernel->name(), latest_hash});
    }
    return kernel_name_to_last_compiled_dir;
}

KernelCacheStatus CompileProgramTestWrapper(IDevice* device, Program& program, bool profile_kernel = false) {
    std::unordered_map<std::string, std::string> pre_compile_kernel_to_hash_str = get_last_program_binary_path(
        program,
        BuildEnvManager::get_instance().get_device_build_env(device->build_id()).build_env.get_out_kernel_root_path());

    detail::CompileProgram(device, program);

    std::unordered_map<std::string, std::string> post_compile_kernel_to_hash_str = get_last_program_binary_path(
        program,
        BuildEnvManager::get_instance().get_device_build_env(device->build_id()).build_env.get_out_kernel_root_path());

    KernelCacheStatus kernel_cache_status;
    for (const auto& [kernel_name, hash_str] : post_compile_kernel_to_hash_str) {
        if (!pre_compile_kernel_to_hash_str.contains(kernel_name)) {
            kernel_cache_status.kernel_name_to_cache_hit.insert({kernel_name, false});
        } else {
            const auto& prev_hash_str = pre_compile_kernel_to_hash_str.at(kernel_name);
            bool cache_hit = hash_str == prev_hash_str;
            kernel_cache_status.kernel_name_to_cache_hit.insert({kernel_name, cache_hit});
        }
        kernel_cache_status.kernel_name_to_hash_str.insert({kernel_name, hash_str});
    }
    return kernel_cache_status;
}

struct ProgramAttributes {
    uint32_t num_tiles = 2048;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    DataMovementProcessor reader_processor = DataMovementProcessor::RISCV_1;
    DataMovementProcessor writer_processor = DataMovementProcessor::RISCV_0;
    NOC reader_noc = NOC::RISCV_1_default;
    NOC writer_noc = NOC::RISCV_0_default;
    tt::DataFormat data_format = tt::DataFormat::Float16_b;
    uint32_t src_cb_index = tt::CBIndex::c_0;
    uint32_t output_cb_index = tt::CBIndex::c_16;
};

Program create_program(IDevice* device, const ProgramAttributes& program_attributes) {
    CoreCoord core = {0, 0};
    Program program = CreateProgram();

    uint32_t single_tile_size = 2 * 1024;

    uint32_t num_input_tiles = 8;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(
            num_input_tiles * single_tile_size, {{program_attributes.src_cb_index, program_attributes.data_format}})
            .set_page_size(program_attributes.src_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t num_output_tiles = 1;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(
            num_output_tiles * single_tile_size, {{program_attributes.output_cb_index, program_attributes.data_format}})
            .set_page_size(program_attributes.output_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_output_config);

    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_4.cpp",
        core,
        DataMovementConfig{.processor = program_attributes.reader_processor, .noc = program_attributes.reader_noc});

    CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{.processor = program_attributes.writer_processor, .noc = program_attributes.writer_noc});

    std::vector<uint32_t> compute_kernel_args = {uint(program_attributes.num_tiles)};

    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_3m.cpp",
        core,
        ComputeConfig{
            .math_fidelity = program_attributes.math_fidelity,
            .fp32_dest_acc_en = program_attributes.fp32_dest_acc_en,
            .math_approx_mode = program_attributes.math_approx_mode,
            .compile_args = compute_kernel_args});

    return program;
}

void assert_kernel_binary_path_exists(
    const Program& program, const std::string& kernel_root_path, const KernelCacheStatus& kernel_cache_status) {
    auto kernel_name_to_hash = kernel_cache_status.kernel_name_to_hash_str;
    for (size_t kernel_id = 0; kernel_id < program.impl().num_kernels(); kernel_id++) {
        auto kernel = program.impl().get_kernel(kernel_id);
        auto hash = kernel_name_to_hash.at(kernel->name());
        auto kernel_binary_path = kernel_root_path + kernel->name() + "/" + hash;
        TT_FATAL(std::filesystem::exists(kernel_binary_path), "Expected {} folder to exist!", kernel_binary_path);
    }
}

void assert_program_cache_hit_status(
    const Program& program, bool hit_expected, const KernelCacheStatus& kernel_cache_status) {
    auto kernel_name_to_cache_hit_status = kernel_cache_status.kernel_name_to_cache_hit;
    for (size_t kernel_id = 0; kernel_id < program.impl().num_kernels(); kernel_id++) {
        auto kernel = program.impl().get_kernel(kernel_id);
        auto hit_status = kernel_name_to_cache_hit_status.at(kernel->name());
        TT_FATAL(
            hit_status == hit_expected,
            "Did not get expected cache status {} for kernel {}",
            hit_expected,
            kernel->name());
    }
}

void assert_kernel_hash_matches(
    const std::unordered_map<std::string, std::string>& golden_kernel_name_to_hash,
    const KernelCacheStatus& kernel_cache_status) {
    for (const auto& [kernel_name, hash] : kernel_cache_status.kernel_name_to_hash_str) {
        const auto& expected_hash = golden_kernel_name_to_hash.at(kernel_name);
        TT_FATAL(hash == expected_hash, "Expected hash for {} {} but got {}", kernel_name, expected_hash, hash);
    }
}

void assert_hash_comparison_for_kernel_type(
    const Program& program,
    const std::unordered_map<std::string, std::string>& prev_kernel_name_to_hash,
    const std::unordered_map<HalProcessorClassType, bool>& type_to_same_hash_expected,
    const KernelCacheStatus& kernel_cache_status) {
    auto curr_kernel_name_to_hash = kernel_cache_status.kernel_name_to_hash_str;
    for (size_t kernel_id = 0; kernel_id < program.impl().num_kernels(); kernel_id++) {
        auto kernel = program.impl().get_kernel(kernel_id);
        auto prev_hash = prev_kernel_name_to_hash.at(kernel->name());
        auto curr_hash = curr_kernel_name_to_hash.at(kernel->name());
        bool same_hash_expected = type_to_same_hash_expected.at(kernel->get_kernel_processor_class());
        if (same_hash_expected) {
            TT_FATAL(prev_hash == curr_hash, "Expected same hashes for {}", kernel->name());
        } else {
            TT_FATAL(prev_hash != curr_hash, "Expected different hashes for {}", kernel->name());
        }
    }
}

void assert_cache_hit_status_for_kernel_type(
    const Program& program,
    const std::unordered_map<HalProcessorClassType, bool>& type_to_cache_hit_status,
    const KernelCacheStatus& kernel_cache_status) {
    auto kernel_name_to_cache_hit_status = kernel_cache_status.kernel_name_to_cache_hit;
    for (size_t kernel_id = 0; kernel_id < program.impl().num_kernels(); kernel_id++) {
        auto kernel = program.impl().get_kernel(kernel_id);
        bool hit_expected = type_to_cache_hit_status.at(kernel->get_kernel_processor_class());
        auto hit_status = kernel_name_to_cache_hit_status.at(kernel->name());
        TT_FATAL(
            hit_status == hit_expected,
            "Did not get expected cache status {} for kernel {}",
            hit_expected,
            kernel->name());
    }
}

std::unordered_map<std::string, std::string> compile_program_with_modified_kernel(
    IDevice* device,
    const ProgramAttributes& attributes,
    const std::unordered_map<std::string, std::string>& prev_kernel_name_to_hash,
    const std::unordered_map<HalProcessorClassType, bool>& kernel_type_to_cache_hit_status) {
    auto program = create_program(device, attributes);
    auto kernel_cache_status = CompileProgramTestWrapper(device, program);
    assert_kernel_binary_path_exists(
        program,
        BuildEnvManager::get_instance().get_device_build_env(device->build_id()).build_env.get_out_kernel_root_path(),
        kernel_cache_status);
    assert_cache_hit_status_for_kernel_type(program, kernel_type_to_cache_hit_status, kernel_cache_status);
    assert_hash_comparison_for_kernel_type(
        program, prev_kernel_name_to_hash, kernel_type_to_cache_hit_status, kernel_cache_status);
    std::unordered_map<std::string, std::string> kernel_name_to_hash = kernel_cache_status.kernel_name_to_hash_str;
    return kernel_name_to_hash;
}

TEST_F(EitherDispatchFixture, CompileProgramInLoop) {
    auto* dev = device();

    ClearKernelCache(
        BuildEnvManager::get_instance().get_device_build_env(dev->build_id()).build_env.get_out_kernel_root_path());
    ProgramAttributes default_attributes;
    auto program = create_program(dev, default_attributes);

    static constexpr int num_compiles = 10;
    std::unordered_map<std::string, std::string> kernel_name_to_hash;
    for (int compile_idx = 0; compile_idx < num_compiles; compile_idx++) {
        auto kernel_cache_status = CompileProgramTestWrapper(dev, program);
        if (compile_idx == 0) {
            assert_kernel_binary_path_exists(
                program,
                BuildEnvManager::get_instance()
                    .get_device_build_env(dev->build_id())
                    .build_env.get_out_kernel_root_path(),
                kernel_cache_status);
            assert_program_cache_hit_status(program, /*hit_expected=*/false, kernel_cache_status);
            kernel_name_to_hash = kernel_cache_status.kernel_name_to_hash_str;
        } else {
            assert_program_cache_hit_status(program, /*hit_expected=*/true, kernel_cache_status);
            assert_kernel_hash_matches(kernel_name_to_hash, kernel_cache_status);
        }
    }
}

TEST_F(EitherDispatchFixture, CompileProgramAfterCleanKernelBinaryDirectory) {
    auto* dev = device();

    ClearKernelCache(
        BuildEnvManager::get_instance().get_device_build_env(dev->build_id()).build_env.get_out_kernel_root_path());

    ProgramAttributes default_attributes;
    auto program = create_program(dev, default_attributes);

    auto kernel_cache_status = CompileProgramTestWrapper(dev, program);

    assert_kernel_binary_path_exists(
        program,
        BuildEnvManager::get_instance().get_device_build_env(dev->build_id()).build_env.get_out_kernel_root_path(),
        kernel_cache_status);
    assert_program_cache_hit_status(program, /*hit_expected=*/false, kernel_cache_status);
    std::unordered_map<std::string, std::string> kernel_name_to_hash = kernel_cache_status.kernel_name_to_hash_str;

    ClearKernelCache(
        BuildEnvManager::get_instance().get_device_build_env(dev->build_id()).build_env.get_out_kernel_root_path());
    auto second_program = create_program(dev, default_attributes);
    auto second_kernel_cache_status = CompileProgramTestWrapper(dev, second_program);
    assert_program_cache_hit_status(second_program, /*hit_expected=*/false, second_kernel_cache_status);
    assert_kernel_hash_matches(kernel_name_to_hash, second_kernel_cache_status);
}

TEST_F(EitherDispatchFixture, CompileProgramWithModifiedProgram) {
    auto* dev = device();

    const static std::unordered_map<HalProcessorClassType, bool> compute_miss_data_movement_hit = {
        {HalProcessorClassType::COMPUTE, false}, {HalProcessorClassType::DM, true}};

    const static std::unordered_map<HalProcessorClassType, bool> compute_hit_data_movement_miss = {
        {HalProcessorClassType::COMPUTE, true}, {HalProcessorClassType::DM, false}};

    const static std::unordered_map<HalProcessorClassType, bool> compute_hit_data_movement_hit = {
        {HalProcessorClassType::COMPUTE, true}, {HalProcessorClassType::DM, true}};

    const static std::unordered_map<HalProcessorClassType, bool> compute_miss_data_movement_miss = {
        {HalProcessorClassType::COMPUTE, false}, {HalProcessorClassType::DM, false}};

    ClearKernelCache(
        BuildEnvManager::get_instance().get_device_build_env(dev->build_id()).build_env.get_out_kernel_root_path());

    ProgramAttributes attributes;
    auto program = create_program(dev, attributes);
    auto kernel_cache_status = CompileProgramTestWrapper(dev, program);
    assert_kernel_binary_path_exists(
        program,
        BuildEnvManager::get_instance().get_device_build_env(dev->build_id()).build_env.get_out_kernel_root_path(),
        kernel_cache_status);
    assert_program_cache_hit_status(program, /*hit_expected=*/false, kernel_cache_status);
    std::unordered_map<std::string, std::string> kernel_name_to_hash = kernel_cache_status.kernel_name_to_hash_str;

    // Modify compute kernel compile time args - expect cache miss for compute kernel
    attributes.num_tiles = 1024;
    kernel_name_to_hash =
        compile_program_with_modified_kernel(dev, attributes, kernel_name_to_hash, compute_miss_data_movement_hit);

    // Modify compute kernel math fidelity - expect cache miss for compute kernel
    attributes.math_fidelity = MathFidelity::LoFi;
    kernel_name_to_hash =
        compile_program_with_modified_kernel(dev, attributes, kernel_name_to_hash, compute_miss_data_movement_hit);

    // Modify compute kernel fp32_dest_acc_en - expect cache miss for compute kernel
    // Grayskull does not support fp32 accumulation
    if (dev->arch() != tt::ARCH::GRAYSKULL) {
        attributes.fp32_dest_acc_en = true;
        kernel_name_to_hash =
            compile_program_with_modified_kernel(dev, attributes, kernel_name_to_hash, compute_miss_data_movement_hit);
    }

    // Modify compute kernel math_approx_mode - expect cache miss for compute kernel
    attributes.math_approx_mode = true;
    kernel_name_to_hash =
        compile_program_with_modified_kernel(dev, attributes, kernel_name_to_hash, compute_miss_data_movement_hit);

    // Modify data movement kernel noc - expect cache miss for data movement kernels
    attributes.reader_noc = NOC::RISCV_0_default;
    attributes.writer_noc = NOC::RISCV_1_default;
    kernel_name_to_hash =
        compile_program_with_modified_kernel(dev, attributes, kernel_name_to_hash, compute_hit_data_movement_miss);

    // Modify data movement kernel processor - expect cache hit
    attributes.reader_processor = DataMovementProcessor::RISCV_1;
    attributes.writer_processor = DataMovementProcessor::RISCV_0;
    kernel_name_to_hash =
        compile_program_with_modified_kernel(dev, attributes, kernel_name_to_hash, compute_hit_data_movement_hit);

    // Modify circular buffer data format - expect cache miss for all kernels
    attributes.data_format = tt::DataFormat::Bfp8_b;
    kernel_name_to_hash =
        compile_program_with_modified_kernel(dev, attributes, kernel_name_to_hash, compute_miss_data_movement_miss);

    // Modify circular buffer index - expect cache miss for all kernels
    attributes.src_cb_index = attributes.src_cb_index + 1;
    attributes.output_cb_index = attributes.output_cb_index + 1;
    kernel_name_to_hash =
        compile_program_with_modified_kernel(dev, attributes, kernel_name_to_hash, compute_miss_data_movement_miss);
}

// ============================================================================
// test_compile_sets_kernel_binaries - Tests that kernel binaries are set correctly
// ============================================================================

}  // namespace

#include <enchantum/enchantum.hpp>
#include <thread>
#include "llrt.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_memory.h"

namespace {

std::string get_latest_kernel_binary_path(const std::string& kernel_root_path, const std::shared_ptr<Kernel>& kernel) {
    TT_FATAL(kernel != nullptr, "Error");
    TT_FATAL(std::filesystem::exists(kernel_root_path + kernel->name()), "Error");

    std::filesystem::path kernel_path{kernel_root_path + kernel->name()};
    std::filesystem::file_time_type ftime = std::filesystem::last_write_time(*kernel_path.begin());
    std::string latest_hash;
    for (const auto& dir_entry : std::filesystem::directory_iterator{kernel_path}) {
        auto kbtime = std::filesystem::last_write_time(dir_entry.path());
        if (kbtime > ftime) {
            ftime = kbtime;
            latest_hash = dir_entry.path().filename().string();
        }
    }
    TT_FATAL(not latest_hash.empty(), "Error");
    return kernel->name() + "/" + latest_hash;
}

void construct_kernel_binaries_program(Program& program, IDevice* device, CoreCoord& core) {
    uint32_t single_tile_size = 2 * 1024;
    uint32_t num_tiles = 2048;
    uint32_t dram_buffer_size = single_tile_size * num_tiles;

    InterleavedBufferConfig buff_config{
        .device = device, .size = dram_buffer_size, .page_size = dram_buffer_size, .buffer_type = BufferType::DRAM};

    auto src_dram_buffer = CreateBuffer(buff_config);
    auto dst_dram_buffer = CreateBuffer(buff_config);

    uint32_t src0_cb_index = CBIndex::c_0;
    uint32_t num_input_tiles = 8;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t ouput_cb_index = CBIndex::c_16;
    uint32_t num_output_tiles = 1;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_output_tiles * single_tile_size, {{ouput_cb_index, DataFormat::Float16_b}})
            .set_page_size(ouput_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_output_config);

    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_4.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    std::vector<uint32_t> compute_kernel_args = {uint(num_tiles)};

    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_3m.cpp",
        core,
        ComputeConfig{.compile_args = compute_kernel_args});
}

TEST_F(EitherDispatchFixture, CompileSetsKernelBinaries) {
    auto* dev = device();
    CoreCoord core = {0, 0};

    Program program = CreateProgram();
    construct_kernel_binaries_program(program, dev, core);

    // Check that binary memory objects in the kernel match the ones obtained from the persistent cache
    uint32_t programmable_core_index =
        MetalContext::instance().hal().get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
    const KernelGroup* kernel_group = program.impl().kernels_on_core(core, programmable_core_index);
    ASSERT_NE(kernel_group, nullptr);

    std::shared_ptr<Kernel> compute_kernel = nullptr;
    std::shared_ptr<Kernel> riscv0_kernel = nullptr;
    std::shared_ptr<Kernel> riscv1_kernel = nullptr;
    for (auto kernel_id : kernel_group->kernel_ids) {
        auto kernel = program.impl().get_kernel(kernel_id);
        switch (kernel->get_kernel_processor_class()) {
            case HalProcessorClassType::DM:
                switch (kernel->get_kernel_processor_type(0)) {
                    case 0: riscv0_kernel = kernel; break;
                    case 1: riscv1_kernel = kernel; break;
                    default: TT_THROW("Error");
                }
                break;
            case HalProcessorClassType::COMPUTE: compute_kernel = kernel; break;
            default: TT_THROW("Error");
        }
    }
    ASSERT_NE(compute_kernel, nullptr);
    ASSERT_NE(riscv0_kernel, nullptr);
    ASSERT_NE(riscv1_kernel, nullptr);

    // Run iteration to get golden
    auto mask = BuildEnvManager::get_instance().get_device_build_env(dev->build_id()).build_key();
    detail::CompileProgram(dev, program);

    auto compute_binaries = compute_kernel->binaries(mask);
    ASSERT_EQ(compute_binaries.size(), 3u) << "Expected 3 Compute binaries!";

    auto brisc_binaries = riscv0_kernel->binaries(mask);
    ASSERT_EQ(brisc_binaries.size(), 1u) << "Expected 1 BRISC binary!";

    auto ncrisc_binaries = riscv1_kernel->binaries(mask);
    ASSERT_EQ(ncrisc_binaries.size(), 1u) << "Expected 1 NCRISC binary!";

    // Verify binaries match after recompile
    std::vector<std::string> kernel_names = {"reader_unary_push_4", "writer_unary", "eltwise_copy_3m"};
    for (const auto& kernel_name : kernel_names) {
        std::filesystem::remove_all(
            BuildEnvManager::get_instance().get_device_build_env(dev->id()).build_env.get_out_kernel_root_path() +
            kernel_name);
    }
    detail::ClearKernelCache();

    Program new_program = CreateProgram();
    construct_kernel_binaries_program(new_program, dev, core);

    uint32_t dm_class_idx = enchantum::to_underlying(HalProcessorClassType::DM);
    uint32_t compute_class_idx = enchantum::to_underlying(HalProcessorClassType::COMPUTE);

    for (int j = 0; j < 3; j++) {
        detail::CompileProgram(dev, new_program);
        const KernelGroup* new_kernel_group = new_program.impl().kernels_on_core(core, programmable_core_index);
        std::shared_ptr<Kernel> new_compute_kernel = nullptr;
        std::shared_ptr<Kernel> new_riscv0_kernel = nullptr;
        std::shared_ptr<Kernel> new_riscv1_kernel = nullptr;
        for (auto kernel_id : new_kernel_group->kernel_ids) {
            auto kernel = new_program.impl().get_kernel(kernel_id);
            switch (kernel->get_kernel_processor_class()) {
                case HalProcessorClassType::DM:
                    switch (kernel->get_kernel_processor_type(0)) {
                        case 0: new_riscv0_kernel = kernel; break;
                        case 1: new_riscv1_kernel = kernel; break;
                        default: TT_THROW("Error");
                    }
                    break;
                case HalProcessorClassType::COMPUTE: new_compute_kernel = kernel; break;
                default: TT_THROW("Error");
            }
        }
        ASSERT_NE(new_compute_kernel, nullptr);
        ASSERT_NE(new_riscv0_kernel, nullptr);
        ASSERT_NE(new_riscv1_kernel, nullptr);

        EXPECT_EQ(new_compute_kernel->binaries(mask), compute_binaries);
        EXPECT_EQ(new_riscv0_kernel->binaries(mask), brisc_binaries);
        EXPECT_EQ(new_riscv1_kernel->binaries(mask), ncrisc_binaries);

        std::string kernel_name = get_latest_kernel_binary_path(
            BuildEnvManager::get_instance().get_device_build_env(dev->build_id()).build_env.get_out_kernel_root_path(),
            new_riscv0_kernel);
        std::string brisc_hex_path =
            BuildEnvManager::get_instance()
                .get_kernel_build_state(dev->build_id(), programmable_core_index, dm_class_idx, 0)
                .get_target_out_path(kernel_name);
        const ll_api::memory& brisc_binary =
            llrt::get_risc_binary(brisc_hex_path, ll_api::memory::Loading::CONTIGUOUS_XIP);
        EXPECT_EQ(brisc_binary, *brisc_binaries.at(0))
            << "Expected saved BRISC binary to be the same as binary in persistent cache";

        kernel_name = get_latest_kernel_binary_path(
            BuildEnvManager::get_instance().get_device_build_env(dev->build_id()).build_env.get_out_kernel_root_path(),
            new_riscv1_kernel);
        std::string ncrisc_hex_path =
            BuildEnvManager::get_instance()
                .get_kernel_build_state(dev->build_id(), programmable_core_index, dm_class_idx, 1)
                .get_target_out_path(kernel_name);
        auto load_type = (dev->arch() == tt::ARCH::GRAYSKULL || dev->arch() == tt::ARCH::WORMHOLE_B0)
                             ? ll_api::memory::Loading::CONTIGUOUS
                             : ll_api::memory::Loading::CONTIGUOUS_XIP;
        const ll_api::memory& ncrisc_binary = llrt::get_risc_binary(ncrisc_hex_path, load_type);
        EXPECT_EQ(ncrisc_binary, *ncrisc_binaries.at(0))
            << "Expected saved NCRISC binary to be the same as binary in persistent cache";

        for (int trisc_id = 0; trisc_id <= 2; trisc_id++) {
            kernel_name = get_latest_kernel_binary_path(
                BuildEnvManager::get_instance()
                    .get_device_build_env(dev->build_id())
                    .build_env.get_out_kernel_root_path(),
                new_compute_kernel);
            std::string trisc_hex_path =
                BuildEnvManager::get_instance()
                    .get_kernel_build_state(dev->build_id(), programmable_core_index, compute_class_idx, trisc_id)
                    .get_target_out_path(kernel_name);
            const ll_api::memory& trisc_binary =
                llrt::get_risc_binary(trisc_hex_path, ll_api::memory::Loading::CONTIGUOUS_XIP);
            EXPECT_EQ(trisc_binary, *compute_binaries.at(trisc_id))
                << "Expected saved TRISC binary for " << trisc_id << " to be the same as binary in persistent cache";
        }
    }
}

// =============================================================================
// ADDITIONAL SLOW DISPATCH TESTS - Migrated from standalone tests
// =============================================================================

TEST_F(SlowDispatchFixture, Bmm) {
    bool pass = true;
    try {
        IDevice* dev = device();
        Program program = CreateProgram();
        CoreCoord core = {0, 0};
        uint32_t single_tile_size = 2 * 1024;
        uint32_t Mt = 4, Kt = 2, Nt = 3, B = 2;
        uint32_t num_tilesA = Mt * Kt * B;
        uint32_t num_tilesB = Kt * Nt * B;
        uint32_t num_tilesC = Mt * Nt * B;
        uint32_t bytesA = single_tile_size * num_tilesA;
        uint32_t bytesB = single_tile_size * num_tilesB;
        uint32_t bytesC = single_tile_size * num_tilesC;

        InterleavedBufferConfig src0_config{
            .device = dev, .size = bytesA, .page_size = single_tile_size, .buffer_type = BufferType::DRAM};
        auto src0_dram_buffer = CreateBuffer(src0_config);
        uint32_t dram_buffer_src0_addr = src0_dram_buffer->address();

        InterleavedBufferConfig src1_config{
            .device = dev, .size = bytesB, .page_size = single_tile_size, .buffer_type = BufferType::DRAM};
        auto src1_dram_buffer = CreateBuffer(src1_config);
        uint32_t dram_buffer_src1_addr = src1_dram_buffer->address();

        InterleavedBufferConfig dst_config{
            .device = dev, .size = bytesC, .page_size = single_tile_size, .buffer_type = BufferType::DRAM};
        auto dst_dram_buffer = CreateBuffer(dst_config);
        uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

        uint32_t src0_cb_index = 0;
        uint32_t num_input_tiles = 2;
        CircularBufferConfig cb_src0_config =
            CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src0_cb_index, single_tile_size);
        CreateCircularBuffer(program, core, cb_src0_config);

        uint32_t src1_cb_index = 1;
        CircularBufferConfig cb_src1_config =
            CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src1_cb_index, single_tile_size);
        CreateCircularBuffer(program, core, cb_src1_config);

        uint32_t ouput_cb_index = tt::CBIndex::c_16;
        uint32_t num_output_tiles = 2;
        CircularBufferConfig cb_output_config =
            CircularBufferConfig(num_output_tiles * single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(ouput_cb_index, single_tile_size);
        CreateCircularBuffer(program, core, cb_output_config);

        std::vector<uint32_t> reader_compile_time_args;
        TensorAccessorArgs(src0_dram_buffer).append_to(reader_compile_time_args);
        TensorAccessorArgs(src1_dram_buffer).append_to(reader_compile_time_args);

        std::vector<uint32_t> writer_compile_time_args;
        TensorAccessorArgs(dst_dram_buffer).append_to(writer_compile_time_args);

        auto reader = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_bmm_8bank.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .compile_args = reader_compile_time_args});

        auto writer = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_bmm_8bank.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = writer_compile_time_args});

        std::vector<uint32_t> compute_kernel_args = {B, Mt, Kt, Nt};
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/compute/bmm.cpp",
            core,
            ComputeConfig{.compile_args = compute_kernel_args});

        std::vector<uint32_t> src0_vec = create_random_vector_of_bfloat16(bytesA, 1.0f, 0x1234);
        std::vector<uint32_t> src1_vec = create_random_vector_of_bfloat16(bytesB, 1.0f, 0x1234, -0.45f);
        detail::WriteToBuffer(src0_dram_buffer, src0_vec);
        detail::WriteToBuffer(src1_dram_buffer, src1_vec);

        uint32_t do_bcast = 0;
        SetRuntimeArgs(
            program,
            reader,
            core,
            {dram_buffer_src0_addr, dram_buffer_src1_addr, Mt, Kt, Nt, Mt * Kt, Kt * Nt, B, do_bcast});
        SetRuntimeArgs(program, writer, core, {dram_buffer_dst_addr, 0, Mt, Kt, Nt, Mt * Kt, Kt * Nt, B});

        detail::LaunchProgram(dev, program);

        std::vector<uint32_t> result_vec;
        detail::ReadFromBuffer(dst_dram_buffer, result_vec);

        // Read and validate
        auto comparison_function = [](float a, float b) {
            const float rtol = 0.05f;
            const float atol = 0.05f;
            float maxabs = fmaxf(fabsf(a), fabsf(b));
            float absdiff = fabsf(a - b);
            return (absdiff <= atol) || absdiff < rtol * maxabs;
        };

        std::vector<uint32_t> shapeA = {1, B, Mt * 32, Kt * 32};
        std::vector<uint32_t> shapeB = {1, B, Kt * 32, Nt * 32};
        std::vector<uint32_t> shapeC = {1, B, Mt * 32, Nt * 32};
        auto u16_src0_vec = u16_from_u32_vector(src0_vec);
        auto u16_src1_vec = u16_from_u32_vector(src1_vec);
        std::vector<uint16_t> src0_linear = convert_layout<uint16_t>(
            u16_src0_vec, shapeA, TensorLayoutType::TILED_NFACES, TensorLayoutType::LIN_ROW_MAJOR);
        std::vector<uint16_t> src1_linear = convert_layout<uint16_t>(
            u16_src1_vec, shapeB, TensorLayoutType::TILED_NFACES, TensorLayoutType::LIN_ROW_MAJOR);
        std::vector<uint16_t> ref_bmm = gold_bmm(shapeA, src0_linear, shapeB, src1_linear);
        auto gold_4f_u32 = u32_from_u16_vector(
            convert_layout<uint16_t>(ref_bmm, shapeC, TensorLayoutType::LIN_ROW_MAJOR, TensorLayoutType::TILED_NFACES));

        int argfail = -1;
        pass &= packed_uint32_t_vector_comparison(result_vec, gold_4f_u32, comparison_function, &argfail);
    } catch (const std::exception& e) {
        pass = false;
        log_error(LogTest, "{}", e.what());
    }
    ASSERT_TRUE(pass);
}

TEST_F(SlowDispatchFixture, TransposeHC) {
    bool pass = true;
    constexpr bool multibank = true;
    try {
        IDevice* dev = device();
        Program program = CreateProgram();
        CoreCoord core = {0, 0};

        constexpr uint32_t subtile_elements = 16U;
        constexpr uint32_t subtile_line_bytes = subtile_elements * 2U;
        constexpr uint32_t tile_elements = 32U;
        constexpr uint32_t tile_size = tile_elements * tile_elements;

        const std::vector<uint32_t> shape = {2U, tile_elements * 3U, tile_elements * 5U, tile_elements * 2U};
        uint32_t num_elements = 1U;
        for (auto s : shape) {
            num_elements *= s;
        }
        const uint32_t num_tensor_tiles = num_elements / tile_size;
        const uint32_t single_tile_bytes = 2U * tile_size;
        const uint32_t dram_buffer_bytes = single_tile_bytes * num_tensor_tiles;
        const uint32_t page_size = multibank ? single_tile_bytes : dram_buffer_bytes;

        InterleavedBufferConfig dram_config{
            .device = dev, .size = dram_buffer_bytes, .page_size = page_size, .buffer_type = BufferType::DRAM};
        auto src0_dram_buffer = CreateBuffer(dram_config);
        auto dst_dram_buffer = CreateBuffer(dram_config);
        const uint32_t alignment = dst_dram_buffer->alignment();
        const bool misaligned = alignment > subtile_line_bytes;

        const uint32_t src0_cb_index = 0U;
        const uint32_t num_buffer_tiles = 2U;
        CircularBufferConfig cb_src0_config =
            CircularBufferConfig(num_buffer_tiles * single_tile_bytes, {{src0_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src0_cb_index, single_tile_bytes);
        CreateCircularBuffer(program, core, cb_src0_config);

        const uint32_t output_cb_index = tt::CBIndex::c_16;
        CircularBufferConfig cb_output_config =
            CircularBufferConfig(num_buffer_tiles * single_tile_bytes, {{output_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(output_cb_index, single_tile_bytes);
        CreateCircularBuffer(program, core, cb_output_config);

        if (misaligned) {
            const uint32_t src1_cb_index = 1U;
            CircularBufferConfig cb_src1_config =
                CircularBufferConfig(alignment, {{src1_cb_index, tt::DataFormat::Float16_b}})
                    .set_page_size(src1_cb_index, alignment);
            CreateCircularBuffer(program, core, cb_src1_config);
        }

        const uint32_t W = shape[3U], H = shape[2U], C = shape[1U], N = shape[0U];
        const uint32_t HW = H * W;
        const uint32_t CHW = C * H * W;
        std::vector<uint32_t> reader_compile_time_args;
        reader_compile_time_args.emplace_back(alignment);
        TensorAccessorArgs(src0_dram_buffer).append_to(reader_compile_time_args);
        auto reader_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/transpose_hc_8bank.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .compile_args = reader_compile_time_args});

        std::vector<uint32_t> writer_compile_time_args;
        TensorAccessorArgs(dst_dram_buffer).append_to(writer_compile_time_args);
        auto unary_writer_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_8bank.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = writer_compile_time_args});

        std::vector<uint32_t> compute_kernel_args = {num_tensor_tiles};
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy.cpp",
            core,
            ComputeConfig{.compile_args = compute_kernel_args});

        std::vector<uint32_t> src0_vec = create_random_vector_of_bfloat16(dram_buffer_bytes, 100U, 0x1234);
        auto src_4f_16 = u16_from_u32_vector(src0_vec);
        detail::WriteToBuffer(src0_dram_buffer, src0_vec);

        SetRuntimeArgs(program, reader_kernel, core, {src0_dram_buffer->address(), 0U, W, H, C, HW, N, CHW});
        SetRuntimeArgs(program, unary_writer_kernel, core, {dst_dram_buffer->address(), 0U, num_tensor_tiles});

        detail::LaunchProgram(dev, program);

        std::vector<uint32_t> result_vec;
        detail::ReadFromBuffer(dst_dram_buffer, result_vec);

        auto comparison_function = [](float a, float b) {
            const float rtol{0.001f};
            const float atol{1e-3f};
            const float maxabs{std::fmaxf(std::abs(a), std::abs(b))};
            float absdiff{std::abs(a - b)};
            return (absdiff <= atol) || (absdiff < rtol * maxabs);
        };

        std::vector<uint16_t> src_linear =
            convert_layout<uint16_t>(src_4f_16, shape, TensorLayoutType::TILED_NFACES, TensorLayoutType::LIN_ROW_MAJOR);
        std::vector<uint16_t> gold_reduced = gold_transpose_hc(src_linear, shape);
        std::vector<uint32_t> shapeR = {shape[0U], shape[2U], shape[1U], shape[3U]};
        auto gold_16_4f = convert_layout<uint16_t>(
            gold_reduced, shapeR, TensorLayoutType::LIN_ROW_MAJOR, TensorLayoutType::TILED_NFACES);
        auto gold_4f_u32 = u32_from_u16_vector(gold_16_4f);

        int argfail = -1;
        pass &= packed_uint32_t_vector_comparison(result_vec, gold_4f_u32, comparison_function, &argfail);
    } catch (const std::exception& e) {
        pass = false;
        log_error(LogTest, "{}", e.what());
    }
    ASSERT_TRUE(pass);
}

TEST_F(SlowDispatchFixture, Flatten) {
    bool pass = true;
    try {
        IDevice* dev = device();
        Program program = CreateProgram();
        CoreCoord core = {0, 0};
        uint32_t single_tile_size = 2 * 1024;
        uint32_t num_tiles_r = 5;
        uint32_t num_tiles_c = 5;
        uint32_t num_tiles = num_tiles_r * num_tiles_c;
        uint32_t num_bytes_per_tensor_row = num_tiles_c * 64;
        uint32_t dram_buffer_size = single_tile_size * num_tiles * 32;

        InterleavedBufferConfig dram_config{
            .device = dev, .size = dram_buffer_size, .page_size = dram_buffer_size, .buffer_type = BufferType::DRAM};
        auto src_dram_buffer = CreateBuffer(dram_config);
        auto dst_dram_buffer = CreateBuffer(dram_config);

        uint32_t src0_cb_index = tt::CBIndex::c_0;
        uint32_t num_input_tiles = 8;
        CircularBufferConfig cb_src0_config =
            CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src0_cb_index, single_tile_size);
        CreateCircularBuffer(program, core, cb_src0_config);

        uint32_t ouput_cb_index = tt::CBIndex::c_16;
        CircularBufferConfig cb_output_config =
            CircularBufferConfig(single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(ouput_cb_index, single_tile_size);
        CreateCircularBuffer(program, core, cb_output_config);

        auto flatten_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/flatten.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

        auto unary_writer_kernel = CreateKernel(
            program,
            "tt_metal/kernels/dataflow/writer_unary.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

        std::vector<uint32_t> compute_kernel_args = {num_tiles * 32};
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy.cpp",
            core,
            ComputeConfig{.compile_args = compute_kernel_args});

        std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
            dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());

        // Gold standard flatten
        auto gold_standard_flatten = [&](std::vector<uint32_t> src, std::vector<uint32_t> shape) {
            std::vector<uint32_t> expected_dst_vec;
            uint32_t num_tile_rows_shape = shape.at(shape.size() - 2) / 32;
            uint32_t num_tile_cols_shape = shape.at(shape.size() - 1) / 32;
            uint32_t start_dram_addr_offset_for_tensor_row = 0;
            for (uint32_t i = 0; i < num_tile_rows_shape; i++) {
                for (uint32_t j = 0; j < 32; j++) {
                    uint32_t src_addr_ = start_dram_addr_offset_for_tensor_row;
                    for (uint32_t k = 0; k < num_tile_cols_shape; k++) {
                        for (uint32_t l = 0; l < 16; l++) {
                            expected_dst_vec.push_back(src.at(src_addr_ + l));
                        }
                        for (uint32_t l = 0; l < 31 * 16; l++) {
                            expected_dst_vec.push_back(0);
                        }
                        src_addr_ += 32 * 16;
                    }
                    start_dram_addr_offset_for_tensor_row += 16;
                }
                start_dram_addr_offset_for_tensor_row += num_tile_cols_shape * 16;
            }
            return expected_dst_vec;
        };

        std::vector<uint32_t> golden = gold_standard_flatten(src_vec, {num_tiles_r * 32, num_tiles_c * 32});

        detail::WriteToBuffer(src_dram_buffer, src_vec);
        SetRuntimeArgs(
            program,
            flatten_kernel,
            core,
            {src_dram_buffer->address(), 0, num_tiles_r, num_tiles_c, num_bytes_per_tensor_row});
        SetRuntimeArgs(program, unary_writer_kernel, core, {dst_dram_buffer->address(), 0, num_tiles * 32});

        detail::LaunchProgram(dev, program);

        std::vector<uint32_t> result_vec;
        detail::ReadFromBuffer(dst_dram_buffer, result_vec);
        pass &= (golden == result_vec);
    } catch (const std::exception& e) {
        pass = false;
        log_error(LogTest, "{}", e.what());
    }
    ASSERT_TRUE(pass);
}

TEST_F(SlowDispatchFixture, DatacopyOutputInL1) {
    bool pass = true;
    try {
        IDevice* dev = device();
        Program program = CreateProgram();
        CoreCoord core = {0, 0};
        uint32_t single_tile_size = 2 * 1024;
        uint32_t num_tiles = 32;
        uint32_t buffer_size = single_tile_size * num_tiles;

        InterleavedBufferConfig dram_config{
            .device = dev, .size = buffer_size, .page_size = buffer_size, .buffer_type = BufferType::DRAM};
        InterleavedBufferConfig l1_config{
            .device = dev, .size = buffer_size, .page_size = buffer_size, .buffer_type = BufferType::L1};

        auto src_dram_buffer = CreateBuffer(dram_config);
        auto dst_l1_buffer = CreateBuffer(l1_config);
        auto l1_dst_noc_xy = dev->virtual_core_from_logical_core(
            dst_l1_buffer->allocator()->get_logical_core_from_bank_id(0), CoreType::WORKER);

        uint32_t src0_cb_index = tt::CBIndex::c_0;
        uint32_t num_input_tiles = 8;
        CircularBufferConfig cb_src0_config =
            CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src0_cb_index, single_tile_size);
        CreateCircularBuffer(program, core, cb_src0_config);

        uint32_t ouput_cb_index = tt::CBIndex::c_16;
        CircularBufferConfig cb_output_config =
            CircularBufferConfig(single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(ouput_cb_index, single_tile_size);
        CreateCircularBuffer(program, core, cb_output_config);

        auto unary_reader_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_4.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

        auto unary_writer_kernel = CreateKernel(
            program,
            "tt_metal/kernels/dataflow/writer_unary_1.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

        std::vector<uint32_t> compute_kernel_args = {num_tiles};
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_3m.cpp",
            core,
            ComputeConfig{.compile_args = compute_kernel_args});

        std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
            buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
        detail::WriteToBuffer(src_dram_buffer, src_vec);

        SetRuntimeArgs(program, unary_reader_kernel, core, {src_dram_buffer->address(), 0, num_tiles});
        SetRuntimeArgs(
            program,
            unary_writer_kernel,
            core,
            {dst_l1_buffer->address(), (uint32_t)l1_dst_noc_xy.x, (uint32_t)l1_dst_noc_xy.y, num_tiles});

        detail::LaunchProgram(dev, program);

        std::vector<uint32_t> result_vec;
        detail::ReadFromBuffer(dst_l1_buffer, result_vec);
        pass &= (src_vec == result_vec);
    } catch (const std::exception& e) {
        pass = false;
        log_error(LogTest, "{}", e.what());
    }
    ASSERT_TRUE(pass);
}

TEST_F(SlowDispatchFixture, DramCopySticksMultiCore) {
    bool pass = true;
    try {
        IDevice* dev = device();
        Program program = CreateProgram();
        auto num_cores_c = 2;
        auto num_cores_r = 2;
        CoreCoord start_core = {0, 0};
        CoreCoord end_core = {(std::size_t)start_core.x + num_cores_c - 1, (std::size_t)start_core.y + num_cores_r - 1};
        CoreRange all_cores(start_core, end_core);

        int num_sticks = 4;
        int num_elements_in_stick = 512;
        int stick_size = num_elements_in_stick * 2;
        uint32_t dram_buffer_size = num_sticks * stick_size;

        InterleavedBufferConfig dram_config{
            .device = dev, .size = dram_buffer_size, .page_size = dram_buffer_size, .buffer_type = BufferType::DRAM};
        auto src_dram_buffer = CreateBuffer(dram_config);

        uint32_t per_core_l1_size = src_dram_buffer->size() / (num_cores_r * num_cores_c);
        std::unordered_map<CoreCoord, uint32_t> core_to_l1_addr;
        for (int i = start_core.y; i < start_core.y + num_cores_r; i++) {
            for (int j = start_core.x; j < start_core.x + num_cores_c; j++) {
                CoreCoord core = {(std::size_t)j, (std::size_t)i};
                InterleavedBufferConfig l1_config{
                    .device = dev,
                    .size = per_core_l1_size,
                    .page_size = per_core_l1_size,
                    .buffer_type = BufferType::L1};
                auto l1_b0 = CreateBuffer(l1_config);
                core_to_l1_addr[core] = l1_b0->address();
            }
        }

        auto unary_reader_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_copy_sticks.cpp",
            all_cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

        std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
            dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
        detail::WriteToBuffer(src_dram_buffer, src_vec);

        uint32_t core_index = 0;
        for (int i = start_core.y; i < start_core.y + num_cores_r; i++) {
            for (int j = start_core.x; j < start_core.x + num_cores_c; j++) {
                CoreCoord core = {(std::size_t)j, (std::size_t)i};
                SetRuntimeArgs(
                    program,
                    unary_reader_kernel,
                    core,
                    {core_to_l1_addr.at(core),
                     src_dram_buffer->address() + (core_index * stick_size),
                     0,
                     1u,
                     (uint32_t)stick_size});
                core_index++;
            }
        }

        detail::LaunchProgram(dev, program);
    } catch (const std::exception& e) {
        pass = false;
        log_error(LogTest, "{}", e.what());
    }
    ASSERT_TRUE(pass);
}

TEST_F(SlowDispatchFixture, UntilizeEltwiseBinary) {
    bool pass = true;
    try {
        IDevice* dev = device();
        Program program = CreateProgram();
        CoreCoord core = {0, 0};
        bool multibank = true;
        uint32_t num_blocks = 1;
        uint32_t num_tiles_r = 2;
        uint32_t num_tiles_c = 2;
        uint32_t single_tile_size = 2 * 1024;
        uint32_t num_tiles = num_blocks * num_tiles_r * num_tiles_c;
        uint32_t dram_buffer_size = single_tile_size * num_tiles;
        uint32_t page_size = multibank ? single_tile_size : dram_buffer_size;

        InterleavedBufferConfig dram_config{
            .device = dev, .size = dram_buffer_size, .page_size = page_size, .buffer_type = BufferType::DRAM};

        auto src0_dram_buffer = CreateBuffer(dram_config);
        auto src1_dram_buffer = CreateBuffer(dram_config);
        auto dst_dram_buffer = CreateBuffer(dram_config);

        uint32_t src0_cb_index = 0;
        uint32_t num_input_tiles = num_tiles_c;
        CircularBufferConfig cb_src0_config =
            CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src0_cb_index, single_tile_size);
        CreateCircularBuffer(program, core, cb_src0_config);

        uint32_t untilized_src0_cb_index = 24;
        CircularBufferConfig cb_untilized_src0_config =
            CircularBufferConfig(
                num_input_tiles * single_tile_size, {{untilized_src0_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(untilized_src0_cb_index, single_tile_size);
        CreateCircularBuffer(program, core, cb_untilized_src0_config);

        uint32_t src1_cb_index = 1;
        CircularBufferConfig cb_src1_config =
            CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src1_cb_index, single_tile_size);
        CreateCircularBuffer(program, core, cb_src1_config);

        uint32_t ouput_cb_index = tt::CBIndex::c_16;
        CircularBufferConfig cb_output_config =
            CircularBufferConfig(num_input_tiles * single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(ouput_cb_index, single_tile_size);
        CreateCircularBuffer(program, core, cb_output_config);

        std::vector<uint32_t> reader_compile_time_args;
        TensorAccessorArgs(src0_dram_buffer).append_to(reader_compile_time_args);
        TensorAccessorArgs(src1_dram_buffer).append_to(reader_compile_time_args);
        auto binary_reader_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_dual_8bank.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .compile_args = reader_compile_time_args});

        std::vector<uint32_t> writer_compile_time_args;
        TensorAccessorArgs(dst_dram_buffer).append_to(writer_compile_time_args);
        auto unary_writer_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_8bank.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = writer_compile_time_args});

        std::vector<uint32_t> compute_kernel_args = {num_blocks, num_tiles_r, num_tiles_c};
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/compute/untilA_elwbin_3m.cpp",
            core,
            ComputeConfig{.compile_args = compute_kernel_args, .defines = {{"ELTWISE_OP", "add_tiles"}}});

        std::vector<uint32_t> src0_vec = create_random_vector_of_bfloat16(
            dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
        detail::WriteToBuffer(src0_dram_buffer, src0_vec);
        std::vector<uint32_t> src1_vec = create_constant_vector_of_bfloat16(dram_buffer_size, 0.0f);
        detail::WriteToBuffer(src1_dram_buffer, src1_vec);

        SetRuntimeArgs(
            program,
            binary_reader_kernel,
            core,
            {src0_dram_buffer->address(), 0u, num_tiles, src1_dram_buffer->address(), 0u, num_tiles, 0u});
        SetRuntimeArgs(program, unary_writer_kernel, core, {dst_dram_buffer->address(), 0u, num_tiles});

        detail::LaunchProgram(dev, program);

        std::vector<uint32_t> result_vec;
        detail::ReadFromBuffer(dst_dram_buffer, result_vec);

        // Gold standard untilize
        auto gold_standard_untilize = [](std::vector<uint32_t> src, std::vector<uint32_t> shape) {
            std::vector<uint32_t> dst_vec;
            int num_rows = shape.at(0);
            int num_cols = shape.at(1) / 2;
            int num_tile_rows = num_rows / 32;
            int num_tile_cols = num_cols / 16;
            int face_size = 16 * 8;
            int tile_size_local = face_size * 4;
            for (int t = 0; t < num_tile_rows; t++) {
                int tile_start_index = t * num_tile_cols;
                int physical_start_for_tile_row = tile_start_index * 32 * 16;
                for (int x = 0; x < 2; x++) {
                    for (int i = 0; i < 16; i++) {
                        for (int j = 0; j < num_tile_cols; j++) {
                            for (int k = 0; k < 8; k++) {
                                int idx = physical_start_for_tile_row + (i * 8) + k + (j * tile_size_local);
                                dst_vec.push_back(src.at(idx));
                            }
                            for (int k = 0; k < 8; k++) {
                                int idx = physical_start_for_tile_row + (i * 8) + k + face_size + (j * tile_size_local);
                                dst_vec.push_back(src.at(idx));
                            }
                        }
                    }
                    physical_start_for_tile_row += 2 * face_size;
                }
            }
            return dst_vec;
        };

        std::vector<uint32_t> golden = gold_standard_untilize(src0_vec, {num_tiles_r * 32, num_tiles_c * 32});
        pass &= (golden == result_vec);
    } catch (const std::exception& e) {
        pass = false;
        log_error(LogTest, "{}", e.what());
    }
    ASSERT_TRUE(pass);
}

TEST_F(SlowDispatchFixture, MatmulSingleTileBfp8b) {
    bool pass = true;
    try {
        IDevice* dev = device();
        Program program = CreateProgram();
        CoreCoord core = {0, 0};

        uint32_t single_tile_size = tt::tile_size(tt::DataFormat::Bfp8_b);
        uint32_t num_tiles = 1;
        uint32_t dram_buffer_size = single_tile_size * num_tiles;

        InterleavedBufferConfig dram_config{
            .device = dev, .size = dram_buffer_size, .page_size = dram_buffer_size, .buffer_type = BufferType::DRAM};
        auto src0_dram_buffer = CreateBuffer(dram_config);
        auto src1_dram_buffer = CreateBuffer(dram_config);
        auto dst_dram_buffer = CreateBuffer(dram_config);

        uint32_t src0_cb_index = 0;
        uint32_t num_input_tiles = 1;
        CircularBufferConfig cb_src0_config =
            CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Bfp8_b}})
                .set_page_size(src0_cb_index, single_tile_size);
        CreateCircularBuffer(program, core, cb_src0_config);

        uint32_t src1_cb_index = 1;
        CircularBufferConfig cb_src1_config =
            CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, tt::DataFormat::Bfp8_b}})
                .set_page_size(src1_cb_index, single_tile_size);
        CreateCircularBuffer(program, core, cb_src1_config);

        uint32_t ouput_cb_index = tt::CBIndex::c_16;
        CircularBufferConfig cb_output_config =
            CircularBufferConfig(num_input_tiles * single_tile_size, {{ouput_cb_index, tt::DataFormat::Bfp8_b}})
                .set_page_size(ouput_cb_index, single_tile_size);
        CreateCircularBuffer(program, core, cb_output_config);

        auto mm_reader_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_blocked.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

        auto unary_writer_kernel = CreateKernel(
            program,
            "tt_metal/kernels/dataflow/writer_unary.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

        std::vector<uint32_t> compute_kernel_args = {1, 1, 1, 1, 1, 1, 1};
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/compute/matmul.cpp",
            core,
            ComputeConfig{.compile_args = compute_kernel_args});

        std::vector<uint32_t> activations = test_utils::create_random_vector_of_bfp8(
            dram_buffer_size, false, 100, std::chrono::system_clock::now().time_since_epoch().count());
        detail::WriteToBuffer(src0_dram_buffer, activations);

        int num_float_in_tile = 32 * 32;
        std::vector<float> vec(num_float_in_tile, 0.0f);
        for (int i = 0; i < 32; i++) {
            vec.at((i * 32) + i) = 1.0f;
        }
        std::vector<uint32_t> weights = pack_as_bfp8_tiles(tt::stl::make_const_span(vec), true, false);
        detail::WriteToBuffer(src1_dram_buffer, weights);

        SetRuntimeArgs(
            program,
            mm_reader_kernel,
            core,
            {src0_dram_buffer->address(),
             0,
             src1_dram_buffer->address(),
             0,
             1,
             1,
             1,
             single_tile_size,
             single_tile_size});
        SetRuntimeArgs(program, unary_writer_kernel, core, {dst_dram_buffer->address(), 0, num_tiles});

        detail::LaunchProgram(dev, program);

        std::vector<uint32_t> result_vec;
        detail::ReadFromBuffer(dst_dram_buffer, result_vec);
        pass &= (activations == result_vec);
    } catch (const std::exception& e) {
        pass = false;
        log_error(LogTest, "{}", e.what());
    }
    ASSERT_TRUE(pass);
}

TEST_F(SlowDispatchFixture, MatmulSingleTileOutputInL1) {
    bool pass = true;
    try {
        IDevice* dev = device();
        Program program = CreateProgram();
        CoreCoord core = {0, 0};
        uint32_t single_tile_size = 2 * 1024;
        uint32_t num_tiles = 1;
        uint32_t dram_buffer_size = single_tile_size * num_tiles;

        InterleavedBufferConfig dram_config{
            .device = dev, .size = dram_buffer_size, .page_size = dram_buffer_size, .buffer_type = BufferType::DRAM};
        InterleavedBufferConfig l1_config{
            .device = dev, .size = dram_buffer_size, .page_size = dram_buffer_size, .buffer_type = BufferType::L1};

        auto src0_dram_buffer = CreateBuffer(dram_config);
        auto src1_dram_buffer = CreateBuffer(dram_config);
        auto dst_l1_buffer = CreateBuffer(l1_config);
        auto l1_dst_noc_xy = dev->virtual_core_from_logical_core(
            dst_l1_buffer->allocator()->get_logical_core_from_bank_id(0), CoreType::WORKER);

        uint32_t src0_cb_index = 0;
        CircularBufferConfig cb_src0_config =
            CircularBufferConfig(single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src0_cb_index, single_tile_size);
        CreateCircularBuffer(program, core, cb_src0_config);

        uint32_t src1_cb_index = 1;
        CircularBufferConfig cb_src1_config =
            CircularBufferConfig(single_tile_size, {{src1_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src1_cb_index, single_tile_size);
        CreateCircularBuffer(program, core, cb_src1_config);

        uint32_t ouput_cb_index = tt::CBIndex::c_16;
        CircularBufferConfig cb_output_config =
            CircularBufferConfig(single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(ouput_cb_index, single_tile_size);
        CreateCircularBuffer(program, core, cb_output_config);

        auto mm_reader_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_blocked.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

        auto unary_writer_kernel = CreateKernel(
            program,
            "tt_metal/kernels/dataflow/writer_unary_1.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

        std::vector<uint32_t> compute_kernel_args = {1, 1, 1, 1, 1, 1, 1};
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/compute/matmul.cpp",
            core,
            ComputeConfig{.compile_args = compute_kernel_args});

        SHAPE shape = {1, 1, 32, 32};
        tt::deprecated::Tensor<bfloat16> tensor = tt::deprecated::initialize_tensor<bfloat16>(
            shape,
            tt::deprecated::Initialize::RANDOM,
            0,
            100,
            std::chrono::system_clock::now().time_since_epoch().count());
        auto activations_tile_layout =
            convert_layout_tile_swizzled_to_tile_nfaces(tt::stl::make_const_span(tensor.get_values()));
        auto activations = pack_bfloat16_vec_into_uint32_vec(activations_tile_layout);
        detail::WriteToBuffer(src0_dram_buffer, activations);

        auto identity = create_identity_matrix(32, 32, 32);
        auto weights_tile_layout = convert_layout_tile_swizzled_to_tile_nfaces(tt::stl::make_const_span(identity));
        auto weights = pack_bfloat16_vec_into_uint32_vec(weights_tile_layout);
        detail::WriteToBuffer(src1_dram_buffer, weights);

        SetRuntimeArgs(
            program,
            mm_reader_kernel,
            core,
            {src0_dram_buffer->address(),
             0,
             src1_dram_buffer->address(),
             0,
             1,
             1,
             1,
             single_tile_size,
             single_tile_size});
        SetRuntimeArgs(
            program,
            unary_writer_kernel,
            core,
            {dst_l1_buffer->address(), (uint32_t)l1_dst_noc_xy.x, (uint32_t)l1_dst_noc_xy.y, num_tiles});

        detail::LaunchProgram(dev, program);

        std::vector<uint32_t> result_vec;
        detail::ReadFromBuffer(dst_l1_buffer, result_vec);
        auto result_bfp16 = unpack_uint32_vec_into_bfloat16_vec(result_vec);
        auto result_flat_layout = convert_layout_tile_nfaces_to_tile_swizzled(tt::stl::make_const_span(result_bfp16));
        pass &= (tensor.get_values() == result_flat_layout);
    } catch (const std::exception& e) {
        pass = false;
        log_error(LogTest, "{}", e.what());
    }
    ASSERT_TRUE(pass);
}

TEST_F(SlowDispatchFixture, InterleavedSticksWriteRead) {
    bool pass = true;
    try {
        IDevice* dev = device();
        int num_sticks = 256;
        int num_elements_in_stick = 1024;
        int stick_size = num_elements_in_stick * 2;
        uint32_t dram_buffer_size = num_sticks * stick_size;

        std::vector<uint32_t> src_vec = create_arange_vector_of_bfloat16(dram_buffer_size, false);

        InterleavedBufferConfig sticks_config{
            .device = dev,
            .size = dram_buffer_size,
            .page_size = (uint64_t)stick_size,
            .buffer_type = BufferType::DRAM};
        auto sticks_buffer = CreateBuffer(sticks_config);

        detail::WriteToBuffer(sticks_buffer, src_vec);
        std::vector<uint32_t> dst_vec;
        detail::ReadFromBuffer(sticks_buffer, dst_vec);
        pass &= (src_vec == dst_vec);
    } catch (const std::exception& e) {
        pass = false;
        log_error(LogTest, "{}", e.what());
    }
    ASSERT_TRUE(pass);
}

TEST_F(SlowDispatchFixture, InterleavedStickReaderTilizedWriter) {
    bool pass = true;
    try {
        IDevice* dev = device();
        Program program = CreateProgram();
        CoreCoord core = {0, 0};
        uint32_t single_tile_size = 2 * 1024;
        int num_sticks = 256;
        int num_elements_in_stick = 1024;
        int stick_size = num_elements_in_stick * 2;
        int num_tiles_c = stick_size / 64;
        int num_output_tiles = (num_sticks * num_elements_in_stick) / 1024;
        uint32_t dram_buffer_size = num_sticks * stick_size;

        InterleavedBufferConfig src_config{
            .device = dev,
            .size = dram_buffer_size,
            .page_size = (uint64_t)stick_size,
            .buffer_type = BufferType::DRAM};
        InterleavedBufferConfig dst_config{
            .device = dev, .size = dram_buffer_size, .page_size = dram_buffer_size, .buffer_type = BufferType::DRAM};
        auto src_dram_buffer = CreateBuffer(src_config);
        auto dst_dram_buffer = CreateBuffer(dst_config);

        uint32_t src0_cb_index = tt::CBIndex::c_0;
        CircularBufferConfig cb_src0_config =
            CircularBufferConfig(num_tiles_c * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src0_cb_index, single_tile_size);
        CreateCircularBuffer(program, core, cb_src0_config);

        uint32_t ouput_cb_index = tt::CBIndex::c_16;
        CircularBufferConfig cb_output_config =
            CircularBufferConfig(single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(ouput_cb_index, single_tile_size);
        CreateCircularBuffer(program, core, cb_output_config);

        std::vector<uint32_t> reader_compile_time_args;
        TensorAccessorArgs(src_dram_buffer).append_to(reader_compile_time_args);
        auto unary_reader_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_stick_layout_8bank.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .compile_args = reader_compile_time_args});

        auto unary_writer_kernel = CreateKernel(
            program,
            "tt_metal/kernels/dataflow/writer_unary.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

        std::vector<uint32_t> compute_kernel_args = {(uint32_t)num_output_tiles};
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy.cpp",
            core,
            ComputeConfig{.compile_args = compute_kernel_args});

        std::vector<uint32_t> src_vec = create_arange_vector_of_bfloat16(dram_buffer_size, false);
        detail::WriteToBuffer(src_dram_buffer, src_vec);

        SetRuntimeArgs(
            program,
            unary_reader_kernel,
            core,
            {src_dram_buffer->address(), (uint32_t)num_sticks, (uint32_t)stick_size, (uint32_t)log2(stick_size)});
        SetRuntimeArgs(program, unary_writer_kernel, core, {dst_dram_buffer->address(), 0, (uint32_t)num_output_tiles});

        detail::LaunchProgram(dev, program);

        std::vector<uint32_t> result_vec;
        detail::ReadFromBuffer(dst_dram_buffer, result_vec);
        pass &= (src_vec == result_vec);
    } catch (const std::exception& e) {
        pass = false;
        log_error(LogTest, "{}", e.what());
    }
    ASSERT_TRUE(pass);
}

TEST_F(SlowDispatchFixture, InterleavedTilizedReaderStickWriter) {
    bool pass = true;
    try {
        IDevice* dev = device();
        Program program = CreateProgram();
        CoreCoord core = {0, 0};
        uint32_t single_tile_size = 2 * 1024;
        int num_sticks = 256;
        int num_elements_in_stick = 1024;
        int stick_size = num_elements_in_stick * 2;
        int num_tiles_c = stick_size / 64;
        int num_output_tiles = (num_sticks * num_elements_in_stick) / 1024;
        uint32_t dram_buffer_size = num_sticks * stick_size;

        InterleavedBufferConfig dram_config{
            .device = dev,
            .size = dram_buffer_size,
            .page_size = (uint64_t)stick_size,
            .buffer_type = BufferType::DRAM};
        auto src_dram_buffer = CreateBuffer(dram_config);
        auto dst_dram_buffer = CreateBuffer(dram_config);

        uint32_t src0_cb_index = 0;
        CircularBufferConfig cb_src0_config =
            CircularBufferConfig(num_tiles_c * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src0_cb_index, single_tile_size);
        CreateCircularBuffer(program, core, cb_src0_config);

        uint32_t ouput_cb_index = tt::CBIndex::c_16;
        CircularBufferConfig cb_output_config =
            CircularBufferConfig(num_tiles_c * single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(ouput_cb_index, single_tile_size);
        CreateCircularBuffer(program, core, cb_output_config);

        std::vector<uint32_t> reader_compile_time_args;
        TensorAccessorArgs(src_dram_buffer).append_to(reader_compile_time_args);
        auto unary_reader_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_stick_layout_8bank.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .compile_args = reader_compile_time_args});

        std::vector<uint32_t> writer_compile_time_args;
        TensorAccessorArgs(dst_dram_buffer).append_to(writer_compile_time_args);
        auto unary_writer_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_stick_layout_8bank.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = writer_compile_time_args});

        std::vector<uint32_t> compute_kernel_args = {(uint32_t)num_output_tiles};
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy.cpp",
            core,
            ComputeConfig{.compile_args = compute_kernel_args});

        std::vector<uint32_t> src_vec = create_arange_vector_of_bfloat16(dram_buffer_size, false);
        detail::WriteToBuffer(src_dram_buffer, src_vec);

        SetRuntimeArgs(
            program,
            unary_reader_kernel,
            core,
            {src_dram_buffer->address(), (uint32_t)num_sticks, (uint32_t)stick_size, (uint32_t)log2(stick_size)});
        SetRuntimeArgs(
            program,
            unary_writer_kernel,
            core,
            {dst_dram_buffer->address(), (uint32_t)num_sticks, (uint32_t)stick_size, (uint32_t)log2(stick_size)});

        detail::LaunchProgram(dev, program);

        std::vector<uint32_t> result_vec;
        detail::ReadFromBuffer(dst_dram_buffer, result_vec);
        pass &= (src_vec == result_vec);
    } catch (const std::exception& e) {
        pass = false;
        log_error(LogTest, "{}", e.what());
    }
    ASSERT_TRUE(pass);
}

TEST_F(SlowDispatchFixture, InterleavedL1ToL1) {
    bool pass = true;
    try {
        IDevice* dev = device();
        Program program = CreateProgram();
        CoreCoord core = {0, 0};
        uint32_t num_pages = 256;
        uint32_t num_bytes_per_page = 2048;
        uint32_t buffer_size = num_pages * num_bytes_per_page;

        CircularBufferConfig cb_src0_config =
            CircularBufferConfig(2 * num_bytes_per_page, {{0, tt::DataFormat::Float16_b}})
                .set_page_size(0, num_bytes_per_page);
        CreateCircularBuffer(program, core, cb_src0_config);
        CircularBufferConfig cb_output_config =
            CircularBufferConfig(2 * num_bytes_per_page, {{16, tt::DataFormat::Float16_b}})
                .set_page_size(16, num_bytes_per_page);
        CreateCircularBuffer(program, core, cb_output_config);

        std::vector<uint32_t> compute_kernel_args = {num_pages};
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy.cpp",
            core,
            ComputeConfig{.compile_args = compute_kernel_args});

        std::vector<uint32_t> host_buffer = create_random_vector_of_bfloat16(
            buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());

        InterleavedBufferConfig l1_config{
            .device = dev, .size = buffer_size, .page_size = num_bytes_per_page, .buffer_type = BufferType::L1};
        auto src = CreateBuffer(l1_config);
        auto dst = CreateBuffer(l1_config);
        detail::WriteToBuffer(src, host_buffer);

        std::vector<uint32_t> reader_compile_time_args;
        TensorAccessorArgs(src).append_to(reader_compile_time_args);
        auto unary_reader_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_8bank.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .compile_args = reader_compile_time_args});

        std::vector<uint32_t> writer_compile_time_args;
        TensorAccessorArgs(dst).append_to(writer_compile_time_args);
        auto unary_writer_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_8bank.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = writer_compile_time_args});

        SetRuntimeArgs(program, unary_reader_kernel, core, {src->address(), 0, 0, num_pages});
        SetRuntimeArgs(program, unary_writer_kernel, core, {dst->address(), 0, num_pages});

        detail::LaunchProgram(dev, program);

        std::vector<uint32_t> readback_buffer;
        detail::ReadFromBuffer(dst, readback_buffer);
        pass = (host_buffer == readback_buffer);
    } catch (const std::exception& e) {
        pass = false;
        log_error(LogTest, "{}", e.what());
    }
    ASSERT_TRUE(pass);
}

TEST_F(SlowDispatchFixture, InterleavedDRAMToL1) {
    bool pass = true;
    try {
        IDevice* dev = device();
        Program program = CreateProgram();
        CoreCoord core = {0, 0};
        uint32_t num_pages = 256;
        uint32_t num_bytes_per_page = 2048;
        uint32_t buffer_size = num_pages * num_bytes_per_page;

        CircularBufferConfig cb_src0_config =
            CircularBufferConfig(2 * num_bytes_per_page, {{0, tt::DataFormat::Float16_b}})
                .set_page_size(0, num_bytes_per_page);
        CreateCircularBuffer(program, core, cb_src0_config);
        CircularBufferConfig cb_output_config =
            CircularBufferConfig(2 * num_bytes_per_page, {{16, tt::DataFormat::Float16_b}})
                .set_page_size(16, num_bytes_per_page);
        CreateCircularBuffer(program, core, cb_output_config);

        std::vector<uint32_t> compute_kernel_args = {num_pages};
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy.cpp",
            core,
            ComputeConfig{.compile_args = compute_kernel_args});

        std::vector<uint32_t> host_buffer = create_random_vector_of_bfloat16(
            buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());

        InterleavedBufferConfig dram_config{
            .device = dev, .size = buffer_size, .page_size = num_bytes_per_page, .buffer_type = BufferType::DRAM};
        InterleavedBufferConfig l1_config{
            .device = dev, .size = buffer_size, .page_size = num_bytes_per_page, .buffer_type = BufferType::L1};
        auto src = CreateBuffer(dram_config);
        auto dst = CreateBuffer(l1_config);
        detail::WriteToBuffer(src, host_buffer);

        std::vector<uint32_t> reader_compile_time_args;
        TensorAccessorArgs(src).append_to(reader_compile_time_args);
        auto unary_reader_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_8bank.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .compile_args = reader_compile_time_args});

        std::vector<uint32_t> writer_compile_time_args;
        TensorAccessorArgs(dst).append_to(writer_compile_time_args);
        auto unary_writer_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_8bank.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = writer_compile_time_args});

        SetRuntimeArgs(program, unary_reader_kernel, core, {src->address(), 0, 0, num_pages});
        SetRuntimeArgs(program, unary_writer_kernel, core, {dst->address(), 0, num_pages});

        detail::LaunchProgram(dev, program);

        std::vector<uint32_t> readback_buffer;
        detail::ReadFromBuffer(dst, readback_buffer);
        pass = (host_buffer == readback_buffer);
    } catch (const std::exception& e) {
        pass = false;
        log_error(LogTest, "{}", e.what());
    }
    ASSERT_TRUE(pass);
}

TEST_F(SlowDispatchFixture, InterleavedL1ToDRAM) {
    bool pass = true;
    try {
        IDevice* dev = device();
        Program program = CreateProgram();
        CoreCoord core = {0, 0};
        uint32_t num_pages = 256;
        uint32_t num_bytes_per_page = 2048;
        uint32_t buffer_size = num_pages * num_bytes_per_page;

        CircularBufferConfig cb_src0_config =
            CircularBufferConfig(2 * num_bytes_per_page, {{0, tt::DataFormat::Float16_b}})
                .set_page_size(0, num_bytes_per_page);
        CreateCircularBuffer(program, core, cb_src0_config);
        CircularBufferConfig cb_output_config =
            CircularBufferConfig(2 * num_bytes_per_page, {{16, tt::DataFormat::Float16_b}})
                .set_page_size(16, num_bytes_per_page);
        CreateCircularBuffer(program, core, cb_output_config);

        std::vector<uint32_t> compute_kernel_args = {num_pages};
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy.cpp",
            core,
            ComputeConfig{.compile_args = compute_kernel_args});

        std::vector<uint32_t> host_buffer = create_random_vector_of_bfloat16(
            buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());

        InterleavedBufferConfig l1_config{
            .device = dev, .size = buffer_size, .page_size = num_bytes_per_page, .buffer_type = BufferType::L1};
        InterleavedBufferConfig dram_config{
            .device = dev, .size = buffer_size, .page_size = num_bytes_per_page, .buffer_type = BufferType::DRAM};
        auto src = CreateBuffer(l1_config);
        auto dst = CreateBuffer(dram_config);
        detail::WriteToBuffer(src, host_buffer);

        std::vector<uint32_t> reader_compile_time_args;
        TensorAccessorArgs(src).append_to(reader_compile_time_args);
        auto unary_reader_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_8bank.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .compile_args = reader_compile_time_args});

        std::vector<uint32_t> writer_compile_time_args;
        TensorAccessorArgs(dst).append_to(writer_compile_time_args);
        auto unary_writer_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_8bank.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = writer_compile_time_args});

        SetRuntimeArgs(program, unary_reader_kernel, core, {src->address(), 0, 0, num_pages});
        SetRuntimeArgs(program, unary_writer_kernel, core, {dst->address(), 0, num_pages});

        detail::LaunchProgram(dev, program);

        std::vector<uint32_t> readback_buffer;
        detail::ReadFromBuffer(dst, readback_buffer);
        pass = (host_buffer == readback_buffer);
    } catch (const std::exception& e) {
        pass = false;
        log_error(LogTest, "{}", e.what());
    }
    ASSERT_TRUE(pass);
}

TEST_F(SlowDispatchFixture, InterleavedDRAMToDRAM) {
    bool pass = true;
    try {
        IDevice* dev = device();
        Program program = CreateProgram();
        CoreCoord core = {0, 0};
        uint32_t num_pages = 256;
        uint32_t num_bytes_per_page = 2048;
        uint32_t buffer_size = num_pages * num_bytes_per_page;

        CircularBufferConfig cb_src0_config =
            CircularBufferConfig(2 * num_bytes_per_page, {{0, tt::DataFormat::Float16_b}})
                .set_page_size(0, num_bytes_per_page);
        CreateCircularBuffer(program, core, cb_src0_config);
        CircularBufferConfig cb_output_config =
            CircularBufferConfig(2 * num_bytes_per_page, {{16, tt::DataFormat::Float16_b}})
                .set_page_size(16, num_bytes_per_page);
        CreateCircularBuffer(program, core, cb_output_config);

        std::vector<uint32_t> compute_kernel_args = {num_pages};
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy.cpp",
            core,
            ComputeConfig{.compile_args = compute_kernel_args});

        std::vector<uint32_t> host_buffer = create_random_vector_of_bfloat16(
            buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());

        InterleavedBufferConfig dram_config{
            .device = dev, .size = buffer_size, .page_size = num_bytes_per_page, .buffer_type = BufferType::DRAM};
        auto src = CreateBuffer(dram_config);
        auto dst = CreateBuffer(dram_config);
        detail::WriteToBuffer(src, host_buffer);

        std::vector<uint32_t> reader_compile_time_args;
        TensorAccessorArgs(src).append_to(reader_compile_time_args);
        auto unary_reader_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_8bank.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .compile_args = reader_compile_time_args});

        std::vector<uint32_t> writer_compile_time_args;
        TensorAccessorArgs(dst).append_to(writer_compile_time_args);
        auto unary_writer_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_8bank.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = writer_compile_time_args});

        SetRuntimeArgs(program, unary_reader_kernel, core, {src->address(), 0, 0, num_pages});
        SetRuntimeArgs(program, unary_writer_kernel, core, {dst->address(), 0, num_pages});

        detail::LaunchProgram(dev, program);

        std::vector<uint32_t> readback_buffer;
        detail::ReadFromBuffer(dst, readback_buffer);
        pass = (host_buffer == readback_buffer);
    } catch (const std::exception& e) {
        pass = false;
        log_error(LogTest, "{}", e.what());
    }
    ASSERT_TRUE(pass);
}

// Helper for MatmulLargeBlock tests
namespace {
void create_CBs_for_fused_matmul(
    Program& program,
    IDevice* dev,
    CoreCoord core,
    bool activations_rm,
    bool output_rm,
    uint32_t M,
    uint32_t N,
    uint32_t in0_block_w,
    uint32_t out_subblock_h) {
    uint32_t num_bytes_for_df = 2;
    uint32_t in0_cb = 0;
    uint32_t in1_cb = 1;
    uint32_t tilize_mode_tilized_in0_cb = 24;
    uint32_t matmul_partials_cb = 25;
    uint32_t untilize_mode_final_matmul_partials_cb = 26;
    uint32_t untilize_mode_reblock_cb = 27;
    uint32_t out0_cb = 16;
    uint32_t single_tile_size = num_bytes_for_df * 1024;
    uint32_t num_output_tiles = M * N;
    CoreRangeSet cores(std::set<CoreRange>{CoreRange(core, core)});

    uint32_t cb0_tiles = M * in0_block_w * 2;
    CircularBufferConfig cb_in0_config =
        CircularBufferConfig(cb0_tiles * single_tile_size, {{in0_cb, tt::DataFormat::Float16_b}})
            .set_page_size(in0_cb, single_tile_size);
    CreateCircularBuffer(program, core, cb_in0_config);

    uint32_t cb1_tiles = N * in0_block_w * 2;
    CircularBufferConfig cb_in1_config =
        CircularBufferConfig(cb1_tiles * single_tile_size, {{in1_cb, tt::DataFormat::Float16_b}})
            .set_page_size(in1_cb, single_tile_size);
    CreateCircularBuffer(program, core, cb_in1_config);

    std::map<uint8_t, tt::DataFormat> partials_and_out_data_format_spec = {
        {matmul_partials_cb, tt::DataFormat::Float16_b}, {out0_cb, tt::DataFormat::Float16_b}};

    if (!activations_rm && !output_rm) {
        CircularBufferConfig cb_matmul_partials_config =
            CircularBufferConfig(num_output_tiles * single_tile_size, partials_and_out_data_format_spec)
                .set_page_size(matmul_partials_cb, single_tile_size)
                .set_page_size(out0_cb, single_tile_size);
        CreateCircularBuffer(program, cores, cb_matmul_partials_config);
    } else if (!activations_rm && output_rm) {
        CircularBufferConfig cb_matmul_partials_config =
            CircularBufferConfig(num_output_tiles * single_tile_size, {{matmul_partials_cb, tt::DataFormat::Float16_b}})
                .set_page_size(matmul_partials_cb, single_tile_size);
        CreateCircularBuffer(program, core, cb_matmul_partials_config);
        CircularBufferConfig cb_final_matmul_partials_config =
            CircularBufferConfig(
                num_output_tiles * single_tile_size,
                {{untilize_mode_final_matmul_partials_cb, tt::DataFormat::Float16_b}})
                .set_page_size(untilize_mode_final_matmul_partials_cb, single_tile_size);
        CreateCircularBuffer(program, core, cb_final_matmul_partials_config);
        uint32_t reblock_cb_tiles = N;
        CircularBufferConfig cb_reblock_config =
            CircularBufferConfig(
                reblock_cb_tiles * single_tile_size, {{untilize_mode_reblock_cb, tt::DataFormat::Float16_b}})
                .set_page_size(untilize_mode_reblock_cb, single_tile_size);
        CreateCircularBuffer(program, core, cb_reblock_config);
        CircularBufferConfig cb_output_config =
            CircularBufferConfig(num_output_tiles * single_tile_size, {{out0_cb, tt::DataFormat::Float16_b}})
                .set_page_size(out0_cb, single_tile_size);
        CreateCircularBuffer(program, core, cb_output_config);
    } else if (activations_rm && !output_rm) {
        CircularBufferConfig cb_src0_tilized_config =
            CircularBufferConfig(
                cb0_tiles * single_tile_size, {{tilize_mode_tilized_in0_cb, tt::DataFormat::Float16_b}})
                .set_page_size(tilize_mode_tilized_in0_cb, single_tile_size);
        CreateCircularBuffer(program, core, cb_src0_tilized_config);
        CircularBufferConfig cb_matmul_partials_config =
            CircularBufferConfig(num_output_tiles * single_tile_size, partials_and_out_data_format_spec)
                .set_page_size(matmul_partials_cb, single_tile_size)
                .set_page_size(out0_cb, single_tile_size);
        CreateCircularBuffer(program, core, cb_matmul_partials_config);
    } else {
        CircularBufferConfig cb_src0_tilized_config =
            CircularBufferConfig(
                num_output_tiles * single_tile_size, {{tilize_mode_tilized_in0_cb, tt::DataFormat::Float16_b}})
                .set_page_size(tilize_mode_tilized_in0_cb, single_tile_size);
        CreateCircularBuffer(program, core, cb_src0_tilized_config);
        CircularBufferConfig cb_matmul_partials_config =
            CircularBufferConfig(num_output_tiles * single_tile_size, {{matmul_partials_cb, tt::DataFormat::Float16_b}})
                .set_page_size(matmul_partials_cb, single_tile_size);
        CreateCircularBuffer(program, core, cb_matmul_partials_config);
        CircularBufferConfig cb_final_matmul_partials_config =
            CircularBufferConfig(
                num_output_tiles * single_tile_size,
                {{untilize_mode_final_matmul_partials_cb, tt::DataFormat::Float16_b}})
                .set_page_size(untilize_mode_final_matmul_partials_cb, single_tile_size);
        CreateCircularBuffer(program, core, cb_final_matmul_partials_config);
        uint32_t reblock_cb_tiles = N;
        CircularBufferConfig cb_reblock_config =
            CircularBufferConfig(
                reblock_cb_tiles * single_tile_size, {{untilize_mode_reblock_cb, tt::DataFormat::Float16_b}})
                .set_page_size(untilize_mode_reblock_cb, single_tile_size);
        CreateCircularBuffer(program, core, cb_reblock_config);
        CircularBufferConfig cb_output_config =
            CircularBufferConfig(num_output_tiles * single_tile_size, {{out0_cb, tt::DataFormat::Float16_b}})
                .set_page_size(out0_cb, single_tile_size);
        CreateCircularBuffer(program, core, cb_output_config);
    }
}

bool test_matmul_large_block(IDevice* dev, bool activations_rm, bool output_rm) {
    bool pass = true;
    try {
        Program program = CreateProgram();
        CoreCoord core = {0, 0};
        uint32_t M = 8, K = 4, N = K;
        int out_subblock_h = 4, out_subblock_w = 2;
        int in0_block_w = K;
        uint32_t single_tile_size = 2 * 1024;
        uint32_t dram_buffer_size_act = single_tile_size * M * K;
        uint32_t dram_buffer_size_weights = single_tile_size * K * N;
        uint32_t dram_buffer_size_out = single_tile_size * M * N;

        InterleavedBufferConfig act_config{
            .device = dev,
            .size = dram_buffer_size_act,
            .page_size = dram_buffer_size_act,
            .buffer_type = BufferType::DRAM};
        InterleavedBufferConfig weights_config{
            .device = dev,
            .size = dram_buffer_size_weights,
            .page_size = dram_buffer_size_weights,
            .buffer_type = BufferType::DRAM};
        InterleavedBufferConfig dst_config{
            .device = dev,
            .size = dram_buffer_size_out,
            .page_size = dram_buffer_size_out,
            .buffer_type = BufferType::DRAM};
        auto src0_dram_buffer = CreateBuffer(act_config);
        auto src1_dram_buffer = CreateBuffer(weights_config);
        auto dst_dram_buffer = CreateBuffer(dst_config);

        const std::array mm_reader_rt_args{
            src0_dram_buffer->address(),
            0u,
            src1_dram_buffer->address(),
            0u,
            (uint32_t)(K / in0_block_w),
            (uint32_t)(M * in0_block_w),
            (uint32_t)(N * in0_block_w),
            (uint32_t)(M * in0_block_w * single_tile_size),
            (uint32_t)(N * in0_block_w * single_tile_size)};

        std::vector<uint32_t> writer_rt_args;
        std::string writer_kernel;
        if (output_rm) {
            writer_kernel = "tt_metal/kernels/dataflow/writer_unary.cpp";
            writer_rt_args = {dst_dram_buffer->address(), 0u, M * N};
        } else {
            writer_kernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unswizzle.cpp";
            writer_rt_args = {
                dst_dram_buffer->address(),
                0u,
                (uint32_t)out_subblock_h,
                (uint32_t)out_subblock_w,
                M / out_subblock_h,
                N / out_subblock_w,
                out_subblock_w * single_tile_size * (N / out_subblock_w),
                out_subblock_h * out_subblock_w * single_tile_size * (N / out_subblock_w),
                out_subblock_w * single_tile_size};
        }

        auto mm_reader_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_blocked.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
        auto unary_writer_kernel = CreateKernel(
            program,
            writer_kernel,
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

        int num_blocks = K / in0_block_w;
        int in0_num_subblocks = M / out_subblock_h;
        int in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;
        int in0_subblock_num_tiles = out_subblock_h * in0_block_w;
        int in1_num_subblocks = N / out_subblock_w;
        int in1_block_num_tiles = out_subblock_w * in0_block_w * in1_num_subblocks;
        int in1_per_core_w = out_subblock_w * in1_num_subblocks;
        int out_subblock_num_tiles = out_subblock_h * out_subblock_w;
        int in0_subblock_h = (in0_block_num_tiles / in0_num_subblocks) / in0_block_w;

        create_CBs_for_fused_matmul(program, dev, core, activations_rm, output_rm, M, N, in0_block_w, out_subblock_h);

        std::vector<uint32_t> compute_kernel_args = {
            (uint32_t)in0_block_w,
            (uint32_t)in0_num_subblocks,
            (uint32_t)in0_block_num_tiles,
            (uint32_t)in0_subblock_num_tiles,
            (uint32_t)in0_subblock_h,
            (uint32_t)in1_num_subblocks,
            (uint32_t)in1_block_num_tiles,
            (uint32_t)in1_per_core_w,
            (uint32_t)num_blocks,
            (uint32_t)out_subblock_h,
            (uint32_t)out_subblock_w,
            (uint32_t)out_subblock_num_tiles,
            (uint32_t)activations_rm,
            (uint32_t)output_rm};
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/compute/matmul_large_block.cpp",
            core,
            ComputeConfig{.compile_args = compute_kernel_args});

        SHAPE shape = {1, 1, M * 32, K * 32};
        tt::deprecated::Tensor<bfloat16> tensor = tt::deprecated::initialize_tensor<bfloat16>(
            shape,
            tt::deprecated::Initialize::RANDOM,
            0,
            100,
            std::chrono::system_clock::now().time_since_epoch().count());

        std::vector<uint32_t> activations;
        if (activations_rm) {
            activations = pack_bfloat16_vec_into_uint32_vec(tensor.get_values());
        } else {
            auto activations_tilized = tilize_swizzled(tensor.get_values(), M * 32, K * 32);
            auto activations_tile_layout =
                convert_layout_tile_swizzled_to_tile_nfaces(tt::stl::make_const_span(activations_tilized));
            activations = pack_bfloat16_vec_into_uint32_vec(activations_tile_layout);
        }
        detail::WriteToBuffer(src0_dram_buffer, activations);

        auto identity = create_identity_matrix(K * 32, N * 32, std::min(K, N) * 32);
        auto identity_tilized = tilize_swizzled(identity, K * 32, N * 32);
        auto weights_tile_layout =
            convert_layout_tile_swizzled_to_tile_nfaces(tt::stl::make_const_span(identity_tilized));
        auto weights = pack_bfloat16_vec_into_uint32_vec(weights_tile_layout);
        detail::WriteToBuffer(src1_dram_buffer, weights);

        SetRuntimeArgs(program, mm_reader_kernel, core, mm_reader_rt_args);
        SetRuntimeArgs(program, unary_writer_kernel, core, writer_rt_args);

        detail::LaunchProgram(dev, program);

        std::vector<uint32_t> result_vec;
        detail::ReadFromBuffer(dst_dram_buffer, result_vec);
        auto result_bfp16 = unpack_uint32_vec_into_bfloat16_vec(result_vec);

        if (output_rm) {
            pass &= (tensor.get_values() == result_bfp16);
        } else {
            auto result_flat_layout =
                convert_layout_tile_nfaces_to_tile_swizzled(tt::stl::make_const_span(result_bfp16));
            auto result_untilized = untilize_swizzled(result_flat_layout, M * 32, N * 32);
            pass &= (tensor.get_values() == result_untilized);
        }
    } catch (const std::exception& e) {
        pass = false;
        log_error(LogTest, "{}", e.what());
    }
    return pass;
}
}  // anonymous namespace

TEST_F(SlowDispatchFixture, MatmulLargeBlockTilizedTilized) {
    ASSERT_TRUE(test_matmul_large_block(device(), false, false));
}

TEST_F(SlowDispatchFixture, MatmulLargeBlockRowMajorTilized) {
    ASSERT_TRUE(test_matmul_large_block(device(), true, false));
}

TEST_F(SlowDispatchFixture, MatmulLargeBlockTilizedRowMajor) {
    ASSERT_TRUE(test_matmul_large_block(device(), false, true));
}

TEST_F(SlowDispatchFixture, MatmulLargeBlockRowMajorRowMajor) {
    ASSERT_TRUE(test_matmul_large_block(device(), true, true));
}

TEST_F(SlowDispatchFixture, GenericBinaryReaderMatmulLargeBlock) {
    bool pass = true;
    try {
        IDevice* dev = device();
        Program program = CreateProgram();
        CoreCoord core = {0, 0};
        uint32_t M = 2, K = 18, N = K;
        int out_subblock_h = 2, out_subblock_w = 3, in0_block_w = 1;
        uint32_t single_tile_size = 2 * 1024;
        uint32_t dram_buffer_size_act = single_tile_size * M * K;
        uint32_t dram_buffer_size_weights = single_tile_size * K * N;
        uint32_t dram_buffer_size_out = single_tile_size * M * N;

        InterleavedBufferConfig act_config{
            .device = dev,
            .size = dram_buffer_size_act,
            .page_size = dram_buffer_size_act,
            .buffer_type = BufferType::DRAM};
        InterleavedBufferConfig weights_config{
            .device = dev,
            .size = dram_buffer_size_weights,
            .page_size = dram_buffer_size_weights,
            .buffer_type = BufferType::DRAM};
        InterleavedBufferConfig dst_config{
            .device = dev,
            .size = dram_buffer_size_out,
            .page_size = dram_buffer_size_out,
            .buffer_type = BufferType::DRAM};
        auto src0_dram_buffer = CreateBuffer(act_config);
        auto src1_dram_buffer = CreateBuffer(weights_config);
        auto dst_dram_buffer = CreateBuffer(dst_config);

        uint32_t src0_cb_index = 0;
        uint32_t cb0_tiles = M * in0_block_w * 2;
        CircularBufferConfig cb_src0_config =
            CircularBufferConfig(cb0_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src0_cb_index, single_tile_size);
        CreateCircularBuffer(program, core, cb_src0_config);

        uint32_t src1_cb_index = 1;
        uint32_t cb1_tiles = N * in0_block_w * 2;
        CircularBufferConfig cb_src1_config =
            CircularBufferConfig(cb1_tiles * single_tile_size, {{src1_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src1_cb_index, single_tile_size);
        CreateCircularBuffer(program, core, cb_src1_config);

        uint32_t ouput_cb_index = tt::CBIndex::c_16;
        uint32_t interm0_cb_index = 24;
        uint32_t num_output_tiles = M * N;
        std::map<uint8_t, tt::DataFormat> data_format_spec = {
            {ouput_cb_index, tt::DataFormat::Float16_b}, {interm0_cb_index, tt::DataFormat::Float16_b}};
        CircularBufferConfig cb_output_config =
            CircularBufferConfig(num_output_tiles * single_tile_size, data_format_spec)
                .set_page_size(ouput_cb_index, single_tile_size)
                .set_page_size(interm0_cb_index, single_tile_size);
        CreateCircularBuffer(program, core, cb_output_config);

        // Create source addresses
        uint32_t face_width = 16, face_height = 16, num_faces = 4;
        uint32_t dram_read_size_bytes = face_width * sizeof(bfloat16);
        uint32_t num_addresses_per_tile = face_height * num_faces;
        uint32_t num_addresses = M * K * num_addresses_per_tile;
        uint32_t src0_num_tiles_per_block = M * in0_block_w;
        uint32_t src1_num_tiles_per_block = N * in0_block_w;
        std::vector<uint32_t> source_addresses;
        source_addresses.reserve(num_addresses);
        for (uint32_t i = 0; i < num_addresses; i++) {
            source_addresses.push_back(i * dram_read_size_bytes);
        }
        int num_blocks = K / in0_block_w;
        uint32_t src0_num_reads_per_block = src0_num_tiles_per_block * num_addresses_per_tile;
        uint32_t src1_num_bytes_per_block = src1_num_tiles_per_block * single_tile_size;

        InterleavedBufferConfig l1_config{
            .device = dev,
            .size = source_addresses.size() * sizeof(uint32_t),
            .page_size = source_addresses.size() * sizeof(uint32_t),
            .buffer_type = BufferType::L1};
        auto source_addresses_in_l1 = CreateBuffer(l1_config);

        std::array<uint32_t, 12> generic_binary_reader_args = {
            src0_dram_buffer->address(),
            0u,
            src1_dram_buffer->address(),
            0u,
            (uint32_t)source_addresses.size(),
            source_addresses_in_l1->address(),
            (uint32_t)num_blocks,
            src0_num_reads_per_block,
            dram_read_size_bytes,
            src1_num_bytes_per_block,
            src0_num_tiles_per_block,
            src1_num_tiles_per_block};

        auto generic_binary_reader_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/generic_binary_reader_blocked.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

        std::array<uint32_t, 9> writer_rt_args = {
            dst_dram_buffer->address(),
            0u,
            (uint32_t)out_subblock_h,
            (uint32_t)out_subblock_w,
            M / out_subblock_h,
            N / out_subblock_w,
            out_subblock_w * single_tile_size * (N / out_subblock_w),
            out_subblock_h * out_subblock_w * single_tile_size * (N / out_subblock_w),
            out_subblock_w * single_tile_size};

        auto unary_writer_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unswizzle.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

        int in0_num_subblocks = M / out_subblock_h;
        int in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;
        int in0_subblock_num_tiles = out_subblock_h * in0_block_w;
        int in1_num_subblocks = N / out_subblock_w;
        int in1_block_num_tiles = out_subblock_w * in0_block_w * in1_num_subblocks;
        int in1_per_core_w = out_subblock_w * in1_num_subblocks;
        int out_subblock_num_tiles = out_subblock_h * out_subblock_w;

        std::vector<uint32_t> compute_kernel_args = {
            (uint32_t)in0_block_w,
            (uint32_t)in0_num_subblocks,
            (uint32_t)in0_block_num_tiles,
            (uint32_t)in0_subblock_num_tiles,
            (uint32_t)in1_num_subblocks,
            (uint32_t)in1_block_num_tiles,
            (uint32_t)in1_per_core_w,
            (uint32_t)num_blocks,
            (uint32_t)out_subblock_h,
            (uint32_t)out_subblock_w,
            (uint32_t)out_subblock_num_tiles};
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/compute/matmul_large_block_zm.cpp",
            core,
            ComputeConfig{.compile_args = compute_kernel_args});

        // Transpose tiles helper
        auto transpose_tiles = [](std::vector<uint32_t> data, int row_tiles, int col_tiles, int in0_bw) {
            std::vector<uint32_t> result;
            int tile_size = 512;
            for (int c = 0; c < col_tiles; c += in0_bw) {
                for (int r = 0; r < row_tiles; r++) {
                    for (int k = 0; k < in0_bw; k++) {
                        int offset = (tile_size * col_tiles * r) + (c * tile_size) + (k * tile_size);
                        for (int i = 0; i < tile_size; i++) {
                            result.push_back(data.at(offset + i));
                        }
                    }
                }
            }
            return result;
        };

        SHAPE shape = {1, 1, M * 32, K * 32};
        tt::deprecated::Tensor<bfloat16> tensor = tt::deprecated::initialize_tensor<bfloat16>(
            shape,
            tt::deprecated::Initialize::RANDOM,
            0,
            100,
            std::chrono::system_clock::now().time_since_epoch().count());
        auto activations_tilized = tilize_swizzled(tensor.get_values(), M * 32, K * 32);
        auto activations_tile_layout =
            convert_layout_tile_swizzled_to_tile_nfaces(tt::stl::make_const_span(activations_tilized));
        auto activations = pack_bfloat16_vec_into_uint32_vec(activations_tile_layout);
        auto activations_tile_transposed = transpose_tiles(activations, M, K, in0_block_w);
        detail::WriteToBuffer(src0_dram_buffer, activations_tile_transposed);

        auto identity = create_identity_matrix(K * 32, N * 32, std::min(K, N) * 32);
        auto identity_tilized = tilize_swizzled(identity, K * 32, N * 32);
        auto weights_tile_layout =
            convert_layout_tile_swizzled_to_tile_nfaces(tt::stl::make_const_span(identity_tilized));
        auto weights = pack_bfloat16_vec_into_uint32_vec(weights_tile_layout);
        detail::WriteToBuffer(src1_dram_buffer, weights);
        detail::WriteToDeviceL1(dev, core, source_addresses_in_l1->address(), source_addresses);

        SetRuntimeArgs(program, generic_binary_reader_kernel, core, generic_binary_reader_args);
        SetRuntimeArgs(program, unary_writer_kernel, core, writer_rt_args);

        detail::LaunchProgram(dev, program);

        std::vector<uint32_t> result_vec;
        detail::ReadFromBuffer(dst_dram_buffer, result_vec);
        auto result_bfp16 = unpack_uint32_vec_into_bfloat16_vec(result_vec);
        auto result_flat_layout = convert_layout_tile_nfaces_to_tile_swizzled(tt::stl::make_const_span(result_bfp16));
        auto result_untilized = untilize_swizzled(result_flat_layout, M * 32, N * 32);
        pass &= (tensor.get_values() == result_untilized);
    } catch (const std::exception& e) {
        pass = false;
        log_error(LogTest, "{}", e.what());
    }
    ASSERT_TRUE(pass);
}

TEST_F(SlowDispatchFixture, Bcast) {
    bool pass = true;
    try {
        IDevice* dev = device();

        const char* bdim_to_log_string[] = {"", "BCAST_H", "BCAST_W", "", "BCAST_HW"};
        const char* op_id_to_op_define[] = {"add_tiles_bcast", "sub_tiles_bcast", "mul_tiles_bcast"};
        const char* op_id_to_llkop_define[] = {"ELWADD", "ELWSUB", "ELWMUL"};
        const char* bdim_to_llkdim_define[] = {
            "", "BroadcastType::ROW", "BroadcastType::COL", "", "BroadcastType::SCALAR"};
        bool multibank = true;

        auto get_reader_name = [](bool mb, BcastDim::Enum bd) -> const char* {
            TT_FATAL(mb && "Only multibank is supported correctly.", "Error");
            if (bd == BcastDim::H) {
                return "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_bcast_h_8bank.cpp";
            }
            if (bd == BcastDim::W) {
                return "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_bcast_w_8bank.cpp";
            }
            if (bd == BcastDim::HW) {
                return "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_bcast_hw_8bank.cpp";
            }
            TT_THROW("Unexpected bcast_dim!");
            return "";
        };

        auto get_compute_name = [](BcastDim::Enum bd) -> const char* {
            switch (bd) {
                case BcastDim::H: return "tests/tt_metal/tt_metal/test_kernels/compute/bcast_h.cpp";
                case BcastDim::W: return "tests/tt_metal/tt_metal/test_kernels/compute/bcast_w.cpp";
                case BcastDim::HW: return "tests/tt_metal/tt_metal/test_kernels/compute/bcast_hw.cpp";
                default: TT_THROW("Unexpected bcast_dim!");
            }
            return "";
        };

        auto bdims = BcastDim::all();
        auto ops = BcastOp::all();

        for (auto bcast_op : ops) {
            for (auto bcast_dim : bdims) {
                log_info(
                    LogTest, "Running bcast test for bdim={}, op={}", bdim_to_log_string[bcast_dim], (int)bcast_op);

                Program program = CreateProgram();
                CoreCoord core = {0, 0};

                std::vector<uint32_t> shape = {2, 4, 2 * constants::TILE_HEIGHT, 3 * constants::TILE_WIDTH};
                uint32_t W = shape[3], H = shape[2], NC = shape[1] * shape[0], N = shape[0], C = shape[1];
                uint32_t Wt = W / constants::TILE_WIDTH;
                uint32_t Ht = H / constants::TILE_HEIGHT;
                uint32_t num_tensor_tiles = NC * H * W / (32 * 32);

                uint32_t single_tile_bytes = 2 * 1024;
                uint32_t dram_buffer_bytes = single_tile_bytes * num_tensor_tiles;
                uint32_t page_size = multibank ? single_tile_bytes : dram_buffer_bytes;

                InterleavedBufferConfig buff_config{
                    .device = dev, .size = dram_buffer_bytes, .page_size = page_size, .buffer_type = BufferType::DRAM};
                auto src0_dram_buffer = CreateBuffer(buff_config);
                auto dst_dram_buffer = CreateBuffer(buff_config);

                uint32_t src0_cb_index = 0;
                uint32_t num_buffer_tiles = 2;
                CircularBufferConfig cb_src0_config =
                    CircularBufferConfig(
                        num_buffer_tiles * single_tile_bytes, {{src0_cb_index, tt::DataFormat::Float16_b}})
                        .set_page_size(src0_cb_index, single_tile_bytes);
                CreateCircularBuffer(program, core, cb_src0_config);

                uint32_t src1_cb_index = 1;
                CircularBufferConfig cb_src1_config =
                    CircularBufferConfig(
                        num_buffer_tiles * single_tile_bytes, {{src1_cb_index, tt::DataFormat::Float16_b}})
                        .set_page_size(src1_cb_index, single_tile_bytes);
                CreateCircularBuffer(program, core, cb_src1_config);

                uint32_t ouput_cb_index = tt::CBIndex::c_16;
                CircularBufferConfig cb_output_config =
                    CircularBufferConfig(
                        num_buffer_tiles * single_tile_bytes, {{ouput_cb_index, tt::DataFormat::Float16_b}})
                        .set_page_size(ouput_cb_index, single_tile_bytes);
                CreateCircularBuffer(program, core, cb_output_config);

                std::vector<uint16_t> tiled_bcast_values;
                std::vector<uint16_t> ref_bcast_values;
                float bcast_1value = 10.0f;
                unsigned num_bcast_tiles = 0;

                if (bcast_dim == BcastDim::HW) {
                    ref_bcast_values.resize(NC, 0);
                    std::vector<uint32_t> ref_bcast_shape_with_tile_padding = {
                        N, C, constants::TILE_HEIGHT, constants::TILE_WIDTH};
                    std::vector<uint16_t> ref_bcast_values_with_tile_padding;
                    ref_bcast_values_with_tile_padding.resize(NC * constants::TILE_HEIGHT * constants::TILE_WIDTH, 0);
                    for (uint32_t j = 0; j < NC; j++) {
                        auto val = std::bit_cast<uint16_t>(bfloat16(bcast_1value + (j % 7)));
                        ref_bcast_values[j] = val;
                        ref_bcast_values_with_tile_padding[j * constants::TILE_HEIGHT * constants::TILE_WIDTH] = val;
                    }
                    tiled_bcast_values = convert_layout<uint16_t>(
                        ref_bcast_values_with_tile_padding,
                        ref_bcast_shape_with_tile_padding,
                        TensorLayoutType::LIN_ROW_MAJOR,
                        TensorLayoutType::TILED_NFACES);
                    num_bcast_tiles = NC;
                } else if (bcast_dim == BcastDim::H) {
                    ref_bcast_values.resize(NC * W, 0);
                    std::vector<uint32_t> ref_bcast_shape_with_tile_padding = {N, C, constants::TILE_HEIGHT, W};
                    std::vector<uint16_t> ref_bcast_values_with_tile_padding;
                    ref_bcast_values_with_tile_padding.resize(NC * constants::TILE_HEIGHT * W, 0);
                    for (uint32_t j = 0; j < NC * W; j++) {
                        auto val = std::bit_cast<uint16_t>(bfloat16(bcast_1value + (j % 7)));
                        ref_bcast_values[j] = val;
                        ref_bcast_values_with_tile_padding[(j % W) + ((j / W) * constants::TILE_HEIGHT * W)] = val;
                    }
                    tiled_bcast_values = convert_layout<uint16_t>(
                        ref_bcast_values_with_tile_padding,
                        ref_bcast_shape_with_tile_padding,
                        TensorLayoutType::LIN_ROW_MAJOR,
                        TensorLayoutType::TILED_NFACES);
                    num_bcast_tiles = NC * Wt;
                } else if (bcast_dim == BcastDim::W) {
                    ref_bcast_values.resize(NC * H, 0);
                    std::vector<uint32_t> ref_bcast_shape_with_tile_padding = {N, C, H, constants::TILE_WIDTH};
                    std::vector<uint16_t> ref_bcast_values_with_tile_padding;
                    ref_bcast_values_with_tile_padding.resize(NC * H * constants::TILE_WIDTH, 0);
                    for (uint32_t j = 0; j < NC * H; j++) {
                        auto val = std::bit_cast<uint16_t>(bfloat16(bcast_1value + (j % 7)));
                        ref_bcast_values[j] = val;
                        ref_bcast_values_with_tile_padding[j * constants::TILE_WIDTH] = val;
                    }
                    tiled_bcast_values = convert_layout<uint16_t>(
                        ref_bcast_values_with_tile_padding,
                        ref_bcast_shape_with_tile_padding,
                        TensorLayoutType::LIN_ROW_MAJOR,
                        TensorLayoutType::TILED_NFACES);
                    num_bcast_tiles = NC * Ht;
                }

                auto bcast_tiled_u32 = u32_from_u16_vector(tiled_bcast_values);
                auto bcast_vals_nbytes = bcast_tiled_u32.size() * sizeof(bcast_tiled_u32[0]);
                uint32_t src1_page_size = multibank ? single_tile_bytes : bcast_vals_nbytes;

                InterleavedBufferConfig src1_config{
                    .device = dev,
                    .size = bcast_vals_nbytes,
                    .page_size = src1_page_size,
                    .buffer_type = BufferType::DRAM};
                auto src1_dram_buffer = CreateBuffer(src1_config);
                detail::WriteToBuffer(src1_dram_buffer, bcast_tiled_u32);

                std::vector<uint32_t> reader_compile_time_args;
                TensorAccessorArgs(src0_dram_buffer).append_to(reader_compile_time_args);
                TensorAccessorArgs(src1_dram_buffer).append_to(reader_compile_time_args);

                auto binary_reader_kernel = CreateKernel(
                    program,
                    get_reader_name(multibank, bcast_dim),
                    core,
                    DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_1,
                        .noc = NOC::RISCV_1_default,
                        .compile_args = reader_compile_time_args});

                std::vector<uint32_t> writer_compile_time_args;
                TensorAccessorArgs(dst_dram_buffer).append_to(writer_compile_time_args);
                auto unary_writer_kernel = CreateKernel(
                    program,
                    "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_8bank.cpp",
                    core,
                    DataMovementConfig{
                        .processor = DataMovementProcessor::RISCV_0,
                        .noc = NOC::RISCV_0_default,
                        .compile_args = writer_compile_time_args});

                uint32_t nc1 = 0;
                SetRuntimeArgs(
                    program,
                    binary_reader_kernel,
                    core,
                    {src0_dram_buffer->address(),
                     0u,
                     num_tensor_tiles,
                     src1_dram_buffer->address(),
                     0u,
                     num_bcast_tiles,
                     NC * Ht * Wt,
                     NC,
                     Ht,
                     Wt,
                     nc1});
                SetRuntimeArgs(program, unary_writer_kernel, core, {dst_dram_buffer->address(), 0u, num_tensor_tiles});

                std::map<std::string, std::string> compute_defines = {
                    {"BCAST_DIM", bdim_to_llkdim_define[bcast_dim]},
                    {"BCAST_OP", op_id_to_op_define[bcast_op]},
                    {"BCAST_LLKOP", op_id_to_llkop_define[bcast_op]}};
                auto eltwise_binary_kernel = CreateKernel(
                    program,
                    get_compute_name(bcast_dim),
                    core,
                    ComputeConfig{.compile_args = {}, .defines = compute_defines});
                SetRuntimeArgs(program, eltwise_binary_kernel, core, {NC, Ht, Wt});

                std::vector<uint32_t> src0_vec = create_random_vector_of_bfloat16(dram_buffer_bytes, 10.0f, 0x1234);
                detail::WriteToBuffer(src0_dram_buffer, src0_vec);
                detail::LaunchProgram(dev, program);

                std::vector<uint32_t> result_vec;
                detail::ReadFromBuffer(dst_dram_buffer, result_vec);

                auto comparison_function = [](float a, float b) {
                    const float rtol = 0.02f;
                    const float atol = 1e-3f;
                    float maxabs = fmaxf(fabsf(a), fabsf(b));
                    float absdiff = fabsf(a - b);
                    return (absdiff <= atol) || absdiff < rtol * maxabs;
                };

                auto u16_src0_vec = u16_from_u32_vector(src0_vec);
                std::vector<uint16_t> src_linear = convert_layout<uint16_t>(
                    u16_src0_vec, shape, TensorLayoutType::TILED_NFACES, TensorLayoutType::LIN_ROW_MAJOR);
                std::vector<uint16_t> gold_added =
                    gold_bcast_op(src_linear, shape, ref_bcast_values, bcast_dim, bcast_op);
                std::vector<uint32_t> shapeR{shape[0], shape[1], shape[2], shape[3]};
                auto gold_4f_u32 = u32_from_u16_vector(convert_layout<uint16_t>(
                    gold_added, shapeR, TensorLayoutType::LIN_ROW_MAJOR, TensorLayoutType::TILED_NFACES));

                int argfail = -1;
                bool sub_pass =
                    packed_uint32_t_vector_comparison(result_vec, gold_4f_u32, comparison_function, &argfail);
                if (!sub_pass) {
                    log_error(
                        LogTest,
                        "Failure at position={} for bdim={}, op={}",
                        argfail,
                        bdim_to_log_string[bcast_dim],
                        (int)bcast_op);
                }
                pass &= sub_pass;
            }
        }
    } catch (const std::exception& e) {
        pass = false;
        log_error(LogTest, "{}", e.what());
    }
    ASSERT_TRUE(pass);
}

TEST_F(SlowDispatchFixture, CoreRangeSetProgram) {
    bool pass = true;
    try {
        IDevice* dev = device();
        Program program = CreateProgram();
        CoreRange core_range_one({0, 0}, {1, 1});
        CoreRange core_range_two({2, 2}, {3, 3});
        CoreRangeSet core_ranges = CoreRangeSet(std::vector{core_range_one, core_range_two});

        uint32_t single_tile_size = 2 * 1024;
        uint32_t num_tiles = 4;
        uint32_t buffer_size = single_tile_size * num_tiles;

        InterleavedBufferConfig dram_config{
            .device = dev, .size = buffer_size, .page_size = buffer_size, .buffer_type = BufferType::DRAM};
        auto src_dram_buffer = CreateBuffer(dram_config);

        std::map<CoreCoord, std::shared_ptr<Buffer>> core_to_l1_buffer;
        for (auto core_range : core_ranges.ranges()) {
            auto start = core_range.start_coord;
            auto end = core_range.end_coord;
            for (auto x = start.x; x <= end.x; x++) {
                for (auto y = start.y; y <= end.y; y++) {
                    CoreCoord logical_core(x, y);
                    InterleavedBufferConfig l1_config{
                        .device = dev, .size = buffer_size, .page_size = buffer_size, .buffer_type = BufferType::L1};
                    auto dst_l1_buffer = CreateBuffer(l1_config);
                    core_to_l1_buffer.emplace(logical_core, dst_l1_buffer);
                }
            }
        }

        uint32_t src0_cb_index = tt::CBIndex::c_0;
        uint32_t num_input_tiles = 8;
        CircularBufferConfig cb_src0_config =
            CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src0_cb_index, single_tile_size);
        CreateCircularBuffer(program, core_ranges, cb_src0_config);

        uint32_t ouput_cb_index = tt::CBIndex::c_16;
        CircularBufferConfig cb_output_config =
            CircularBufferConfig(single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(ouput_cb_index, single_tile_size);
        CreateCircularBuffer(program, core_ranges, cb_output_config);

        auto unary_reader_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_4.cpp",
            core_ranges,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

        auto unary_writer_kernel = CreateKernel(
            program,
            "tt_metal/kernels/dataflow/writer_unary_1.cpp",
            core_ranges,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

        std::vector<uint32_t> compute_kernel_args = {num_tiles};
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_3m.cpp",
            core_ranges,
            ComputeConfig{.compile_args = compute_kernel_args});

        for (uint32_t i = 0; i < NUM_SEMAPHORES; i++) {
            CreateSemaphore(program, core_ranges, i);
        }

        detail::CompileProgram(dev, program);

        std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
            buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
        detail::WriteToBuffer(src_dram_buffer, src_vec);

        const std::array<uint32_t, 3> reader_rt_args = {src_dram_buffer->address(), 0u, num_tiles};
        for (const auto& [core, dst_l1_buffer] : core_to_l1_buffer) {
            SetRuntimeArgs(program, unary_reader_kernel, core, reader_rt_args);
            auto l1_dst_noc_xy = dev->virtual_core_from_logical_core(
                dst_l1_buffer->allocator()->get_logical_core_from_bank_id(0), CoreType::WORKER);
            SetRuntimeArgs(
                program,
                unary_writer_kernel,
                core,
                {dst_l1_buffer->address(), (uint32_t)l1_dst_noc_xy.x, (uint32_t)l1_dst_noc_xy.y, num_tiles});
        }

        detail::LaunchProgram(dev, program);

        for (const auto& [core, dst_l1_buffer] : core_to_l1_buffer) {
            std::vector<uint32_t> result_vec;
            detail::ReadFromBuffer(dst_l1_buffer, result_vec);
            pass &= (src_vec == result_vec);
        }
    } catch (const std::exception& e) {
        pass = false;
        log_error(LogTest, "{}", e.what());
    }
    ASSERT_TRUE(pass);
}

// =============================================================================
// test_clean_init - Fast dispatch loopback test (recovery test)
// =============================================================================

TEST_F(FastDispatchFixture, CleanInit) {
    bool pass = true;
    try {
        auto* md = mesh_device();
        auto& cq = command_queue();

        constexpr uint32_t single_tile_size = 2 * (32 * 32);
        constexpr uint32_t num_tiles = 50;
        constexpr uint32_t dram_buffer_size = single_tile_size * num_tiles;

        distributed::DeviceLocalBufferConfig local_l1_config{
            .page_size = dram_buffer_size, .buffer_type = BufferType::L1};
        distributed::DeviceLocalBufferConfig local_dram_config{
            .page_size = dram_buffer_size, .buffer_type = BufferType::DRAM};
        distributed::ReplicatedBufferConfig global_buffer_config{.size = dram_buffer_size};

        auto l1_buffer = distributed::MeshBuffer::create(global_buffer_config, local_l1_config, md);
        auto input_dram_buffer = distributed::MeshBuffer::create(global_buffer_config, local_dram_config, md);
        auto output_dram_buffer = distributed::MeshBuffer::create(global_buffer_config, local_dram_config, md);

        Program program = CreateProgram();
        auto mesh_workload = distributed::MeshWorkload();

        constexpr CoreCoord core = {0, 0};

        std::vector<uint32_t> compile_time_args;
        TensorAccessorArgs(*(input_dram_buffer->get_backing_buffer())).append_to(compile_time_args);
        TensorAccessorArgs(*(output_dram_buffer->get_backing_buffer())).append_to(compile_time_args);
        KernelHandle dram_copy_kernel_id = CreateKernel(
            program,
            "tt_metal/programming_examples/loopback/kernels/loopback_dram_copy.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = compile_time_args});

        std::vector<uint32_t> input_vec = create_random_vector_of_bfloat16(
            dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
        distributed::EnqueueWriteMeshBuffer(cq, input_dram_buffer, input_vec, false);

        const std::array<uint32_t, 5> runtime_args = {
            l1_buffer->address(),
            input_dram_buffer->address(),
            output_dram_buffer->address(),
            l1_buffer->size(),
            num_tiles};

        SetRuntimeArgs(program, dram_copy_kernel_id, core, runtime_args);
        mesh_workload.add_program(distributed::MeshCoordinateRange(md->shape()), std::move(program));
        distributed::EnqueueMeshWorkload(cq, mesh_workload, false);
        distributed::Finish(cq);

        std::vector<uint32_t> result_vec;
        distributed::ReadShard(cq, result_vec, output_dram_buffer, distributed::MeshCoordinate(0, 0));

        pass = (input_vec == result_vec);
    } catch (const std::exception& e) {
        pass = false;
        log_error(LogTest, "{}", e.what());
    }
    ASSERT_TRUE(pass);
}

// =============================================================================
// test_sdpa_reduce_c - SDPA reduce operations test
// =============================================================================

namespace {

std::vector<bfloat16> make_identity_scale_tile_sdpa() {
    std::vector<bfloat16> tile(constants::TILE_HEIGHT * constants::TILE_WIDTH, static_cast<bfloat16>(1.0f));
    return tile;
}

std::vector<bfloat16> make_prev_max_matrix_sdpa(
    uint32_t q_chunk_size, float min_val, float max_val, std::vector<bfloat16>& prev_max_first_col_out) {
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(min_val, max_val);

    const uint32_t rows = q_chunk_size * 32;
    const uint32_t cols = 32;
    std::vector<bfloat16> mat(rows * cols);
    prev_max_first_col_out.resize(rows);

    for (uint32_t r = 0; r < rows; ++r) {
        float v = dist(rng);
        prev_max_first_col_out[r] = static_cast<bfloat16>(v);
        for (uint32_t c = 0; c < cols; ++c) {
            mat[(r * cols) + c] = static_cast<bfloat16>(v);
        }
    }
    return mat;
}

std::vector<bfloat16> golden_reduce_c_sdpa(
    const std::vector<bfloat16>& qk_im_rm,
    const std::vector<bfloat16>& prev_max_first_col,
    uint32_t q_chunk_size,
    uint32_t k_chunk_size,
    bool do_eltwise_max) {
    const uint32_t rows = q_chunk_size * 32;
    const uint32_t cols = k_chunk_size * 32;
    const uint32_t stats_cols = 32;

    std::vector<bfloat16> out(rows * stats_cols, static_cast<bfloat16>(0.0f));

    for (uint32_t r = 0; r < rows; ++r) {
        float row_max = -std::numeric_limits<float>::infinity();
        for (uint32_t c = 0; c < cols; ++c) {
            row_max = std::max(row_max, static_cast<float>(qk_im_rm[(r * cols) + c]));
        }
        if (do_eltwise_max) {
            row_max = std::max(row_max, static_cast<float>(prev_max_first_col[r]));
        }
        out[(r * stats_cols) + 0] = static_cast<bfloat16>(row_max);
    }
    return out;
}

float compare_first_col_mse_sdpa(
    const std::vector<bfloat16>& result_rm, const std::vector<bfloat16>& golden_first_col_rm) {
    const uint32_t rows = golden_first_col_rm.size() / 32;
    float mse = 0.0f;
    for (uint32_t r = 0; r < rows; ++r) {
        float a = static_cast<float>(result_rm[(r * 32) + 0]);
        float b = static_cast<float>(golden_first_col_rm[(r * 32) + 0]);
        float d = a - b;
        mse += d * d;
    }
    mse /= rows;
    return mse;
}

bool run_sdpa_reduce_c_variant(
    IDevice* device,
    distributed::MeshCommandQueue& cq,
    uint32_t q_chunk_size,
    uint32_t k_chunk_size,
    bool fp32_dest_acc_en,
    bool do_eltwise_max,
    const std::string& kernel_path) {
    bool pass = true;

    Program program = CreateProgram();
    CoreCoord core = {0, 0};

    auto cb_df = tt::DataFormat::Float16_b;
    auto cb_tile_size = tt::tile_size(cb_df);

    uint32_t qk_im_num_tiles = q_chunk_size * k_chunk_size;
    uint32_t stats_num_tiles = q_chunk_size;

    auto qk_im_buffer_config = ShardedBufferConfig{
        .device = device,
        .size = qk_im_num_tiles * cb_tile_size,
        .page_size = cb_tile_size,
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = ShardSpecBuffer(
            CoreRangeSet(std::set<CoreRange>({CoreRange(core, core)})),
            {q_chunk_size * constants::TILE_HEIGHT, k_chunk_size * constants::TILE_WIDTH},
            ShardOrientation::ROW_MAJOR,
            {constants::TILE_HEIGHT, constants::TILE_WIDTH},
            {q_chunk_size, k_chunk_size})};

    auto stats_buffer_config = ShardedBufferConfig{
        .device = device,
        .size = stats_num_tiles * cb_tile_size,
        .page_size = cb_tile_size,
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = ShardSpecBuffer(
            CoreRangeSet(std::set<CoreRange>({CoreRange(core, core)})),
            {stats_num_tiles * constants::TILE_HEIGHT, constants::TILE_WIDTH},
            ShardOrientation::ROW_MAJOR,
            {constants::TILE_HEIGHT, constants::TILE_WIDTH},
            {stats_num_tiles, 1})};

    auto one_tile_buffer_config = ShardedBufferConfig{
        .device = device,
        .size = cb_tile_size,
        .page_size = cb_tile_size,
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = ShardSpecBuffer(
            CoreRangeSet(std::set<CoreRange>({CoreRange(core, core)})),
            {constants::TILE_HEIGHT, constants::TILE_WIDTH},
            ShardOrientation::ROW_MAJOR,
            {constants::TILE_HEIGHT, constants::TILE_WIDTH},
            {1, 1})};

    auto qk_im_buffer = CreateBuffer(qk_im_buffer_config);
    auto prev_max_buffer = CreateBuffer(stats_buffer_config);
    auto out_max_buffer = CreateBuffer(stats_buffer_config);
    auto identity_scale_buffer = CreateBuffer(one_tile_buffer_config);

    auto cb_qk_im_id = tt::CBIndex::c_0;
    auto cb_qk_im_config = CircularBufferConfig(qk_im_num_tiles * cb_tile_size, {{cb_qk_im_id, cb_df}})
                               .set_page_size(cb_qk_im_id, cb_tile_size)
                               .set_globally_allocated_address(*qk_im_buffer);
    CreateCircularBuffer(program, core, cb_qk_im_config);

    auto cb_prev_max_id = tt::CBIndex::c_1;
    auto cb_prev_max_config = CircularBufferConfig(stats_num_tiles * cb_tile_size, {{cb_prev_max_id, cb_df}})
                                  .set_page_size(cb_prev_max_id, cb_tile_size)
                                  .set_globally_allocated_address(*prev_max_buffer);
    CreateCircularBuffer(program, core, cb_prev_max_config);

    auto cb_out_max_id = tt::CBIndex::c_2;
    auto cb_out_max_config = CircularBufferConfig(stats_num_tiles * cb_tile_size, {{cb_out_max_id, cb_df}})
                                 .set_page_size(cb_out_max_id, cb_tile_size)
                                 .set_globally_allocated_address(*out_max_buffer);
    CreateCircularBuffer(program, core, cb_out_max_config);

    auto cb_identity_scale_id = tt::CBIndex::c_3;
    auto cb_identity_scale_config = CircularBufferConfig(1 * cb_tile_size, {{cb_identity_scale_id, cb_df}})
                                        .set_page_size(cb_identity_scale_id, cb_tile_size)
                                        .set_globally_allocated_address(*identity_scale_buffer);
    CreateCircularBuffer(program, core, cb_identity_scale_config);

    std::vector<uint32_t> compute_kernel_args = {
        cb_qk_im_id,
        cb_prev_max_id,
        cb_out_max_id,
        cb_identity_scale_id,
        q_chunk_size,
        k_chunk_size,
        static_cast<uint32_t>(do_eltwise_max ? 1 : 0)};

    std::map<std::string, std::string> compute_defines;
    compute_defines["SUB_EXP_GRANULARITY"] = "0";
    compute_defines["LOG2_SUB_EXP_GRANULARITY"] = "0";
    compute_defines["STATS_GRANULARITY"] = "0";
    compute_defines["LOG2_STATS_GRANULARITY"] = "0";
    compute_defines["MUL_BCAST_GRANULARITY"] = "0";
    compute_defines["LOG2_MUL_BCAST_GRANULARITY"] = "0";
    compute_defines["DHT_GRANULARITY"] = "0";
    compute_defines["LOG2_DHT_GRANULARITY"] = "0";
    compute_defines["EXP_APPROX_MODE"] = "0";
    compute_defines["REDUCE_GRANULARITY"] = "1";
    compute_defines["LOG2_REDUCE_GRANULARITY"] = "0";

    CreateKernel(
        program,
        kernel_path,
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi2,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = false,
            .compile_args = compute_kernel_args,
            .defines = compute_defines});

    SHAPE qk_im_shape = {1, 1, q_chunk_size * 32, k_chunk_size * 32};
    tt::deprecated::Tensor<bfloat16> qk_im_tensor =
        tt::deprecated::initialize_tensor<bfloat16>(qk_im_shape, tt::deprecated::Initialize::RANDOM, -50, 50, 0);

    auto qk_im_tilized = tilize_nfaces(qk_im_tensor.get_values(), q_chunk_size * 32, k_chunk_size * 32);
    auto qk_im = pack_bfloat16_vec_into_uint32_vec(qk_im_tilized);
    detail::WriteToBuffer(qk_im_buffer, qk_im);

    std::vector<bfloat16> prev_max_first_col;
    auto prev_max_rm = make_prev_max_matrix_sdpa(q_chunk_size, 25.0f, 65.0f, prev_max_first_col);
    auto prev_max_tilized = tilize_nfaces(prev_max_rm, q_chunk_size * 32, 32);
    auto prev_max_uint_vec = pack_bfloat16_vec_into_uint32_vec(prev_max_tilized);
    detail::WriteToBuffer(prev_max_buffer, prev_max_uint_vec);

    auto identity_scale_tile = make_identity_scale_tile_sdpa();
    auto identity_scale_uint_vec = pack_bfloat16_vec_into_uint32_vec(identity_scale_tile);
    detail::WriteToBuffer(identity_scale_buffer, identity_scale_uint_vec);

    distributed::MeshWorkload workload;
    distributed::MeshCoordinate zero_coord(0, 0);
    distributed::MeshCoordinateRange device_range(zero_coord, zero_coord);
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::vector<uint32_t> out_max_vec;
    detail::ReadFromBuffer(out_max_buffer, out_max_vec);
    auto out_max_bfp16 = unpack_uint32_vec_into_bfloat16_vec(out_max_vec);
    auto out_max_rm = untilize_nfaces(out_max_bfp16, q_chunk_size * 32, 32);

    auto golden_first_col_rm =
        golden_reduce_c_sdpa(qk_im_tensor.get_values(), prev_max_first_col, q_chunk_size, k_chunk_size, do_eltwise_max);

    float mse = compare_first_col_mse_sdpa(out_max_rm, golden_first_col_rm);
    if (mse > 0.0f) {
        pass = false;
    }

    return pass;
}

}  // anonymous namespace

TEST_F(FastDispatchFixture, NIGHTLY_SdpaReduceC) {
    bool pass = true;
    try {
        IDevice* dev = device();
        auto& cq = command_queue();

        // Parameters to sweep over for correctness
        std::vector<uint32_t> q_chunk_sizes = {1, 2, 4, 8};
        std::vector<uint32_t> k_chunk_sizes = {1, 2, 4, 8, 16};
        std::vector<bool> fp32_dest_acc_ens = {false, true};
        std::vector<bool> do_eltwise = {false, true};

        std::vector<std::pair<std::string, std::string>> kernel_variants = {
            {"tests/tt_metal/tt_metal/test_kernels/misc/sdpa/reduce_c/compute.cpp", "reduce_c"},
            {"tests/tt_metal/tt_metal/test_kernels/misc/sdpa/reduce_block_max_row/compute.cpp",
             "reduce_block_max_row"}};

        for (const auto& [kernel_path, kernel_name] : kernel_variants) {
            for (uint32_t q_chunk_size : q_chunk_sizes) {
                for (uint32_t k_chunk_size : k_chunk_sizes) {
                    for (bool fp32_dest_acc_en : fp32_dest_acc_ens) {
                        for (bool do_elt : do_eltwise) {
                            bool this_passed = run_sdpa_reduce_c_variant(
                                dev, cq, q_chunk_size, k_chunk_size, fp32_dest_acc_en, do_elt, kernel_path);
                            if (!this_passed) {
                                log_error(
                                    LogTest,
                                    "SdpaReduceC failed for kernel: {}, q={}, k={}, fp32={}, eltwise={}",
                                    kernel_name,
                                    q_chunk_size,
                                    k_chunk_size,
                                    fp32_dest_acc_en,
                                    do_elt);
                            }
                            pass &= this_passed;
                        }
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        pass = false;
        log_error(LogTest, "{}", e.what());
    }
    ASSERT_TRUE(pass);
}

// =============================================================================
// test_single_dm_l1_write - Requires simulator, DISABLED by default
// =============================================================================

TEST_F(SlowDispatchFixture, SingleDmL1Write) {
    // This test requires TT_METAL_SIMULATOR environment variable to be set
    if (!std::getenv("TT_METAL_SIMULATOR")) {
        GTEST_SKIP() << "This test requires simulator mode (set TT_METAL_SIMULATOR)";
    }

    bool pass = true;
    try {
        const uint32_t address = 100 * 1024;
        const uint32_t value = 0x12345678;
        const std::unordered_map<std::string, uint32_t> named_compile_time_args = {
            {"buffer_size", 1024},
            {"", 3},
            {"!@#$%^&*()", 12},
            {"very_long_parameter_name_that_someone_could_potentially_use_to_try_to_break_the_kernel", 456}};

        std::vector<uint32_t> outputs(1);
        outputs[0] = 0;

        if (std::getenv("TT_METAL_DPRINT_CORES") == nullptr) {
            log_warning(LogTest, "Set TT_METAL_DPRINT_CORES=0,0 to see Data Movement kernel output");
        }

        IDevice* dev = device();
        constexpr CoreCoord core = {0, 0};

        detail::WriteToDeviceL1(dev, core, address, outputs);

        auto& cq = command_queue();
        distributed::MeshWorkload workload;
        distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device()->shape());
        Program program = CreateProgram();

        KernelHandle data_movement_kernel_0 = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/simple_l1_write.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .named_compile_args = named_compile_time_args});

        SetRuntimeArgs(program, data_movement_kernel_0, core, {address});
        SetCommonRuntimeArgs(program, data_movement_kernel_0, {value});

        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, true);
        detail::ReadFromDeviceL1(dev, core, address, 4, outputs);

        pass = (outputs[0] == value);
        if (!pass) {
            log_error(LogTest, "Test failed! Got 0x{:x} instead of 0x{:x}", outputs[0], value);
        }
    } catch (const std::exception& e) {
        pass = false;
        log_error(LogTest, "{}", e.what());
    }
    ASSERT_TRUE(pass);
}

// =============================================================================
// test_stress_noc_mcast - NOC multicast stress test
// =============================================================================

TEST_F(SlowDispatchFixture, StressNocMcast) {
    bool pass = true;
    try {
        IDevice* dev = device();
        Program program = CreateProgram();

        // Use default parameters (minimal stress test)
        uint32_t time_secs = 1;  // Run for 1 second instead of default 10
        uint32_t tlx = 0, tly = 0;
        uint32_t width = 1, height = 1;
        uint32_t mcast_x = 1, mcast_y = 0;  // Mcast core outside the grid
        uint32_t mcast_size = 16;
        uint32_t ucast_size = 8192;
        NOC noc = NOC::NOC_0;
        const uint32_t N_RANDS = 512;
        bool rnd_delay = false;

        CoreRange workers_logical({tlx, tly}, {tlx + width - 1, tly + height - 1});
        CoreCoord mcast_logical(mcast_x, mcast_y);
        CoreCoord tl_core = dev->worker_core_from_logical_core({tlx, tly});
        CoreCoord mcast_end = dev->worker_core_from_logical_core(workers_logical.end_coord);

        bool virtualization_enabled =
            tt::tt_metal::MetalContext::instance().hal().is_coordinate_virtualization_enabled();
        uint32_t num_dests = workers_logical.size();
        CoreCoord virtual_offset =
            virtualization_enabled ? dev->worker_core_from_logical_core({0, 0}) : CoreCoord(0, 0);

        std::vector<uint32_t> compile_args = {
            false,  // is_mcast_core
            tl_core.x,
            tl_core.y,
            mcast_end.x,
            mcast_end.y,
            num_dests,
            time_secs,
            ucast_size,
            mcast_size,
            virtual_offset.x,
            virtual_offset.y,
            N_RANDS,
            rnd_delay,
            dev->allocator()->get_base_allocator_addr(HalMemType::L1),
            dev->allocator()->get_base_allocator_addr(HalMemType::L1),
        };

        KernelHandle ucast_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/stress_noc_mcast.cpp",
            workers_logical,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = noc,
                .compile_args = compile_args,
            });

        CoreCoord grid_size = dev->logical_grid_size();
        for (CoreCoord coord : workers_logical) {
            std::vector<uint32_t> runtime_args;
            for (uint32_t i = 0; i < N_RANDS / sizeof(uint32_t); i++) {
                uint32_t rnd = 0;
                for (uint32_t j = 0; j < sizeof(uint32_t); j++) {
                    uint32_t x = rand() % grid_size.x;
                    uint32_t y = rand() % grid_size.y;
                    if (!virtualization_enabled) {
                        CoreCoord physical_coord = dev->worker_core_from_logical_core(CoreCoord(x, y));
                        x = physical_coord.x;
                        y = physical_coord.y;
                    }
                    rnd = (rnd << 8) | (y << 4) | x;
                }
                runtime_args.push_back(rnd);
            }
            SetRuntimeArgs(program, ucast_kernel, coord, runtime_args);
        }

        // Add mcast kernel
        compile_args[0] = true;  // is_mcast_core
        KernelHandle mcast_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/stress_noc_mcast.cpp",
            mcast_logical,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = noc,
                .compile_args = compile_args,
            });

        std::vector<uint32_t> mcast_runtime_args;
        mcast_runtime_args.reserve(128);
        for (int i = 0; i < 128; i++) {
            mcast_runtime_args.push_back(rand());
        }
        SetRuntimeArgs(program, mcast_kernel, mcast_logical, mcast_runtime_args);

        detail::LaunchProgram(dev, program, true);
        // If we get here without hanging, the test passed
        pass = true;
    } catch (const std::exception& e) {
        pass = false;
        log_error(LogTest, "{}", e.what());
    }
    ASSERT_TRUE(pass);
}

}  // namespace
