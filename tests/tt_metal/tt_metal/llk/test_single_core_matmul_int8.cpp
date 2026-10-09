// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <fmt/base.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <functional>
#include <random>
#include <map>
#include <memory>
#include <ostream>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include "llk_device_fixture.hpp"
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/df/float32.hpp"
#include "tt_metal/test_utils/float8_utils.hpp"
#include "tt_metal/test_utils/packing.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/test_utils/stimulus.hpp"

namespace tt::tt_metal {

using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

namespace unit_tests::compute::matmul {

std::vector<int8_t> generate_uniform_int8(size_t num_elements, int8_t min, int8_t max, uint32_t seed = 0) {
    std::vector<int8_t> random_array(num_elements);

    std::mt19937 gen(seed);
    std::uniform_int_distribution<int16_t> distrib(min, max);

    for (size_t i = 0; i < num_elements; i++) {
        random_array[i] = static_cast<int8_t>(distrib(gen));
    }

    return random_array;
}

void convert_to_sign_mag(std::vector<int8_t>& vec) {
    for (signed char & i : vec) {
        int8_t temp = i;
        if (temp < 0) {
            if (temp == -128) {
                temp = -127;
            }
            temp = (~temp) + 1;
            temp = temp | 0x80;
            i = temp;
        }
    }
}

// in0 x in1 for one 32x32 int8 tile in face-major order, clamped to [-127, 127] and returned in sign-magnitude, as
// the packer writes it.
std::vector<int8_t> golden_single_tile_matmul_int8(
    const std::vector<int8_t>& input_0, const std::vector<int8_t>& input_1) {
    constexpr size_t tile_size = 32 * 32;
    std::vector<int32_t> golden_output_int32(tile_size, 0);
    for (int x = 0; x < 32; x++) {
        for (int y = 0; y < 32; y++) {
            for (int z = 0; z < 32; z++) {
                golden_output_int32[byte_tile_face_major_index(x, y)] +=
                    (int32_t)input_0[byte_tile_face_major_index(z, y)] *
                    (int32_t)input_1[byte_tile_face_major_index(x, z)];
            }
        }
    }
    std::vector<int8_t> golden_output(tile_size, 0);
    for (size_t i = 0; i < tile_size; i++) {
        golden_output[i] = static_cast<int8_t>(std::clamp(golden_output_int32[i], -127, 127));
    }
    convert_to_sign_mag(golden_output);
    return golden_output;
}

bool single_tile_matmul_int8(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    bool pass = true;

    CoreCoord core(0, 0);
    const uint32_t in0_cb_index = 0;
    const uint32_t in1_cb_index = 1;
    const uint32_t out_cb_index = 16;
    const size_t tile_size = 1 * 1 * 32 * 32;
    const size_t byte_size = 1 * tile_size;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    auto& cq = mesh_device->mesh_command_queue();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::MeshWorkload workload;

    auto local_buffer_config =
        distributed::DeviceLocalBufferConfig{.page_size = tile_size, .buffer_type = BufferType::DRAM, .bottom_up = false};
    auto distributed_buffer_config = distributed::ShardedBufferConfig{
        .global_size = tile_size,
        .global_buffer_shape = {32, 32},
        .shard_shape = {32, 32},
        .shard_orientation = ShardOrientation::ROW_MAJOR};

    workload.add_program(device_range, tt_metal::CreateProgram());
    auto& program = workload.get_programs().at(device_range);

    auto input0_dram_buffer = distributed::MeshBuffer::create(distributed_buffer_config, local_buffer_config, mesh_device.get());
    const uint32_t in0_dram_addr = input0_dram_buffer->address();
    auto input1_dram_buffer = distributed::MeshBuffer::create(distributed_buffer_config, local_buffer_config, mesh_device.get());
    const uint32_t in1_dram_addr = input1_dram_buffer->address();
    auto output_dram_buffer = distributed::MeshBuffer::create(distributed_buffer_config, local_buffer_config, mesh_device.get());
    const uint32_t out_dram_addr = output_dram_buffer->address();

    tt_metal::CircularBufferConfig l1_input0_cb_config =
        tt_metal::CircularBufferConfig(byte_size, {{in0_cb_index, tt::DataFormat::Int8}})
            .set_page_size(in0_cb_index, byte_size);
    tt_metal::CreateCircularBuffer(program, core, l1_input0_cb_config);

    tt_metal::CircularBufferConfig l1_input1_cb_config =
        tt_metal::CircularBufferConfig(byte_size, {{in1_cb_index, tt::DataFormat::Int8}})
            .set_page_size(in1_cb_index, byte_size);
    tt_metal::CreateCircularBuffer(program, core, l1_input1_cb_config);

    tt_metal::CircularBufferConfig l1_output_cb_config =
        tt_metal::CircularBufferConfig(byte_size, {{out_cb_index, tt::DataFormat::Int8}})
            .set_page_size(out_cb_index, byte_size);
    tt_metal::CreateCircularBuffer(program, core, l1_output_cb_config);

    auto reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/reader_binary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .compile_args = {in0_cb_index, in1_cb_index}});

    auto writer_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/writer_unary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = {out_cb_index}});

    tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/single_tile_compute.cpp",
        core,
        tt_metal::ComputeConfig{.fp32_dest_acc_en = true, .compile_args = {in0_cb_index, in1_cb_index, out_cb_index}});

    ////////////////////////////////////////////////////////////////////////////
    //                      Stimulus Generation
    ////////////////////////////////////////////////////////////////////////////
    int8_t min = -5, max = 5;
    std::vector<int8_t> input_0 = generate_uniform_int8(tile_size, min, max, 0 /*seed*/);
    std::vector<int8_t> input_1 = generate_uniform_int8(tile_size, min, max, 1 /*different seed*/);

    ////////////////////////////////////////////////////////////////////////////
    //                      Golden Generation
    ////////////////////////////////////////////////////////////////////////////
    std::vector<int8_t> golden_output = golden_single_tile_matmul_int8(input_0, input_1);

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////

    convert_to_sign_mag(input_0);
    distributed::EnqueueWriteMeshBuffer(cq, input0_dram_buffer, input_0);
    convert_to_sign_mag(input_1);
    distributed::EnqueueWriteMeshBuffer(cq, input1_dram_buffer, input_1);

    tt_metal::SetRuntimeArgs(
        program,
        reader_kernel,
        core,
        {
            (uint32_t)in0_dram_addr,
            (uint32_t)0,  // in_0 dram bank id
            (uint32_t)in1_dram_addr,
            (uint32_t)0,
            (uint32_t)1,  // num_tiles
        });
    tt_metal::SetRuntimeArgs(
        program,
        writer_kernel,
        core,
        {
            (uint32_t)out_dram_addr,
            (uint32_t)0,
            (uint32_t)1,  // num_tiles
        });

    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    ////////////////////////////////////////////////////////////////////////////
    //                      Comparison Checking
    ////////////////////////////////////////////////////////////////////////////
    std::vector<int8_t> dest_buffer_data;
    distributed::EnqueueReadMeshBuffer(cq, dest_buffer_data, output_dram_buffer);
    pass = dest_buffer_data == golden_output;

    for (int i = 0; i < 1024; i++) {
        if (dest_buffer_data[i] != golden_output[i]) {
            std::cout << "element " << i << ": " << (uint32_t)dest_buffer_data[i] << " vs. "
                      << (uint32_t)golden_output[i] << " \n";
        }
    }

    return pass;
}

// in0 x in1 for one 32x32 int8 tile in face-major order, unclamped, in the sign-magnitude Int32 the packer writes.
std::vector<uint32_t> golden_single_tile_matmul_int8_to_int32(
    const std::vector<int8_t>& input_0, const std::vector<int8_t>& input_1) {
    constexpr size_t tile_size = 32 * 32;
    std::vector<uint32_t> golden(tile_size, 0);
    for (int x = 0; x < 32; x++) {
        for (int y = 0; y < 32; y++) {
            int32_t acc = 0;
            for (int z = 0; z < 32; z++) {
                acc += (int32_t)input_0[byte_tile_face_major_index(z, y)] *
                       (int32_t)input_1[byte_tile_face_major_index(x, z)];
            }
            golden[byte_tile_face_major_index(x, y)] =
                acc < 0 ? (0x80000000u | static_cast<uint32_t>(-acc)) : static_cast<uint32_t>(acc);
        }
    }
    return golden;
}

// Quasar port of single_tile_matmul_int8 on the Quasar kernel API, with an Int32 output as tt-llk's Quasar int8 matmul
// tests use. Each fidelity phase multiplies int8 operands in full, so the LLK must program LoFi for them whatever
// fidelity the kernel asks for.
bool single_tile_matmul_int8_quasar(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, MathFidelity fidelity) {
    auto& cq = mesh_device->mesh_command_queue();
    const experimental::NodeCoord node{0, 0};
    constexpr uint32_t tile_size = 32 * 32;            // one byte per Int8 datum
    constexpr uint32_t out_tile_size = 4 * tile_size;  // Int32

    auto make_dram = [&](uint32_t size) {
        return distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = size},
            {.page_size = size, .buffer_type = BufferType::DRAM},
            mesh_device.get());
    };
    auto input0_dram_buffer = make_dram(tile_size);
    auto input1_dram_buffer = make_dram(tile_size);
    auto output_dram_buffer = make_dram(out_tile_size);

    const experimental::DFBSpecName IN0_DFB{"in0_dfb"};
    const experimental::DFBSpecName IN1_DFB{"in1_dfb"};
    const experimental::DFBSpecName OUT_DFB{"out_dfb"};
    const experimental::KernelSpecName READER0{"reader0"};
    const experimental::KernelSpecName READER1{"reader1"};
    const experimental::KernelSpecName WRITER{"writer"};
    const experimental::KernelSpecName COMPUTE{"compute"};

    auto make_dfb = [](const experimental::DFBSpecName& name, uint32_t size, tt::DataFormat format) {
        return experimental::DataflowBufferSpec{
            .unique_id = name, .entry_size = size, .num_entries = 1, .data_format_metadata = format};
    };
    auto make_reader = [](const experimental::KernelSpecName& name, const experimental::DFBSpecName& dfb) {
        return experimental::KernelSpec{
            .unique_id = name,
            .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_reader_unary_2_0.cpp",
            .num_threads = 1,
            .dfb_bindings = {experimental::ProducerOf(dfb, "out")},
            .runtime_arg_schema = {.runtime_arg_names = {"src_addr", "src_bank_id", "num_tiles", "dram_page_stride"}},
            .hw_config = experimental::DataMovementHardwareConfig{},
        };
    };
    experimental::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_writer_unary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ConsumerOf(OUT_DFB, "in")},
        .runtime_arg_schema = {.runtime_arg_names = {"dst_addr", "dst_bank_id", "num_tiles", "dram_page_stride"}},
        .hw_config = experimental::DataMovementHardwareConfig{},
    };
    auto binding = [](const experimental::DFBSpecName& dfb, const char* name, experimental::DFBEndpointType type) {
        return experimental::KernelSpec::DFBBinding{.dfb_spec_name = dfb, .accessor_name = name, .endpoint_type = type};
    };
    experimental::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = "tests/tt_metal/tt_metal/test_kernels/compute/matmul_single_tile_quasar.cpp",
        .num_threads = 1,
        .dfb_bindings =
            {binding(IN0_DFB, "in0", experimental::DFBEndpointType::CONSUMER),
             binding(IN1_DFB, "in1", experimental::DFBEndpointType::CONSUMER),
             binding(OUT_DFB, "out", experimental::DFBEndpointType::PRODUCER)},
        .hw_config = experimental::ComputeHardwareConfig{.fpu_math_fidelity = fidelity, .enable_32_bit_dest = true},
    };

    experimental::WorkUnitSpec wu{.name = "main", .kernels = {READER0, READER1, WRITER, COMPUTE}, .target_nodes = node};
    experimental::ProgramSpec spec{
        .name = "matmul_int8",
        .kernels = {make_reader(READER0, IN0_DFB), make_reader(READER1, IN1_DFB), writer_spec, compute_spec},
        .dataflow_buffers =
            {make_dfb(IN0_DFB, tile_size, tt::DataFormat::Int8),
             make_dfb(IN1_DFB, tile_size, tt::DataFormat::Int8),
             make_dfb(OUT_DFB, out_tile_size, tt::DataFormat::Int32)},
        .work_units = {wu},
    };
    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    int8_t min = -5, max = 5;
    std::vector<int8_t> input_0 = generate_uniform_int8(tile_size, min, max, 0 /*seed*/);
    std::vector<int8_t> input_1 = generate_uniform_int8(tile_size, min, max, 1 /*different seed*/);
    std::vector<uint32_t> golden_output = golden_single_tile_matmul_int8_to_int32(input_0, input_1);
    convert_to_sign_mag(input_0);
    convert_to_sign_mag(input_1);
    distributed::EnqueueWriteMeshBuffer(cq, input0_dram_buffer, input_0, /*blocking=*/true);
    distributed::EnqueueWriteMeshBuffer(cq, input1_dram_buffer, input_1, /*blocking=*/true);

    auto dram_args = [&](const char* addr,
                         const char* bank,
                         const std::shared_ptr<distributed::MeshBuffer>& buf,
                         uint32_t page_size) {
        return experimental::MakeRuntimeArgsForSingleNode(
            node, {{addr, buf->address()}, {bank, 0u}, {"num_tiles", 1u}, {"dram_page_stride", page_size}});
    };
    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        {.kernel = READER0, .runtime_arg_values = dram_args("src_addr", "src_bank_id", input0_dram_buffer, tile_size)},
        {.kernel = READER1, .runtime_arg_values = dram_args("src_addr", "src_bank_id", input1_dram_buffer, tile_size)},
        {.kernel = WRITER,
         .runtime_arg_values = dram_args("dst_addr", "dst_bank_id", output_dram_buffer, out_tile_size)},
    };
    experimental::SetProgramRunArgs(program, params);
    LaunchProgram(*mesh_device, std::move(program));

    std::vector<uint32_t> dest_buffer_data;
    distributed::EnqueueReadMeshBuffer(cq, dest_buffer_data, output_dram_buffer, /*blocking=*/true);
    size_t mismatches = 0;
    for (size_t i = 0; i < tile_size; i++) {
        if (dest_buffer_data[i] != golden_output[i] && mismatches++ < 8) {
            log_info(tt::LogTest, "element {}: {} vs. golden {}", i, dest_buffer_data[i], golden_output[i]);
        }
    }
    return mismatches == 0;
}

}  // namespace unit_tests::compute::matmul

TEST_F(LLKMeshDeviceFixture, TensixTestSingleCoreSingleTileComputeMatmulInt8) {
    ASSERT_TRUE(unit_tests::compute::matmul::single_tile_matmul_int8(this->devices_.at(0)));
}

class LLKQuasarMatmulInt8Fixture : public LLKQuasarMeshDeviceSingleCardFixture,
                                   public testing::WithParamInterface<MathFidelity> {};

TEST_P(LLKQuasarMatmulInt8Fixture, TensixTestSingleCoreSingleTileComputeMatmulInt8) {
    ASSERT_TRUE(unit_tests::compute::matmul::single_tile_matmul_int8_quasar(this->devices_.at(0), GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
    QuasarMatmulInt8,
    LLKQuasarMatmulInt8Fixture,
    testing::Values(MathFidelity::LoFi, MathFidelity::HiFi4),
    [](const testing::TestParamInfo<MathFidelity>& info) {
        return info.param == MathFidelity::LoFi ? "LoFi" : "HiFi4";
    });

}  // namespace tt::tt_metal
