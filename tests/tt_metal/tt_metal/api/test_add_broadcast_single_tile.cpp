// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "device_fixture.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/test_utils/packing.hpp"
#include "tests/tt_metal/test_utils/comparison.hpp"
// test_utils includes not required directly here

namespace tt::tt_metal {

namespace unit_tests::compute::binary_bcast_single_tile {

struct Config {
    size_t num_tiles = 0;  // number of A tiles
    tt::DataFormat l1_data_format = tt::DataFormat::Float16_b;
};

// Add a stream of A tiles and a single tile of B, and compute the result of adding the two (broadcasting B).
static bool run_test(const std::shared_ptr<distributed::MeshDevice>& mesh_device, const Config& cfg) {
    using namespace tt;
    using namespace tt::tt_metal;

    const uint32_t single_tile_size = sizeof(bfloat16) * 32 * 32;
    const size_t a_bytes = cfg.num_tiles * single_tile_size;
    const size_t b_bytes = single_tile_size;

    auto& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    auto zero = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero, zero);
    Program program{};
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);

    // DRAM buffers
    distributed::DeviceLocalBufferConfig dram_cfg{.page_size = single_tile_size, .buffer_type = BufferType::DRAM};
    auto a_buf = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = a_bytes}, dram_cfg, mesh_device.get());
    auto b_buf = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = b_bytes}, dram_cfg, mesh_device.get());
    auto out_buf = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = a_bytes}, dram_cfg, mesh_device.get());

    // L1 CBs
    CoreCoord core{0, 0};
    CircularBufferConfig cb_a_cfg = CircularBufferConfig(2 * single_tile_size, {{CBIndex::c_0, cfg.l1_data_format}});
    cb_a_cfg.set_tile_dims(CBIndex::c_0, tt::tt_metal::Tile({32, 32}));
    cb_a_cfg.set_page_size(CBIndex::c_0, single_tile_size);
    CreateCircularBuffer(program_, core, cb_a_cfg);

    CircularBufferConfig cb_b_cfg = CircularBufferConfig(1 * single_tile_size, {{CBIndex::c_1, cfg.l1_data_format}});
    cb_b_cfg.set_tile_dims(CBIndex::c_1, tt::tt_metal::Tile({32, 32}));
    cb_b_cfg.set_page_size(CBIndex::c_1, single_tile_size);
    CreateCircularBuffer(program_, core, cb_b_cfg);

    CircularBufferConfig cb_out_cfg = CircularBufferConfig(2 * single_tile_size, {{CBIndex::c_16, cfg.l1_data_format}});
    cb_out_cfg.set_tile_dims(CBIndex::c_16, tt::tt_metal::Tile({32, 32}));
    cb_out_cfg.set_page_size(CBIndex::c_16, single_tile_size);
    CreateCircularBuffer(program_, core, cb_out_cfg);

    // Kernels
    std::vector<uint32_t> reader_compile_args;
    TensorAccessorArgs(*a_buf).append_to(reader_compile_args);
    TensorAccessorArgs(*b_buf).append_to(reader_compile_args);
    auto reader = CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_preload_b_stream_a.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_args});

    std::vector<uint32_t> writer_compile_args;
    TensorAccessorArgs(*out_buf).append_to(writer_compile_args);
    auto writer = CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_tensor_accessor.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_args});

    auto compute = CreateKernel(
        program_, "tests/tt_metal/tt_metal/test_kernels/compute/add_a_stream_b_constant.cpp", core, ComputeConfig{});

    // Stimulus
    std::vector<uint32_t> a_packed = tt::test_utils::generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -1.0f, 1.0f, a_bytes / sizeof(bfloat16), 17);
    std::vector<uint32_t> b_packed = tt::test_utils::generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -1.0f, 1.0f, b_bytes / sizeof(bfloat16), 23);

    // Golden
    auto a_unpacked = tt::test_utils::unpack_vector<bfloat16, uint32_t>(a_packed);
    auto b_unpacked = tt::test_utils::unpack_vector<bfloat16, uint32_t>(b_packed);
    std::vector<bfloat16> golden(a_unpacked.size());
    for (size_t t = 0; t < cfg.num_tiles; ++t) {
        for (size_t i = 0; i < 32 * 32; ++i) {
            golden[t * 32 * 32 + i] =
                bfloat16(static_cast<float>(a_unpacked[t * 32 * 32 + i]) + static_cast<float>(b_unpacked[i]));
        }
    }
    auto golden_packed = tt::test_utils::pack_vector<uint32_t, bfloat16>(golden);

    // Write, run, read
    distributed::WriteShard(cq, a_buf, a_packed, zero, false);
    distributed::WriteShard(cq, b_buf, b_packed, zero, false);

    SetRuntimeArgs(
        program_, reader, core, {a_buf->address(), b_buf->address(), (uint32_t)cfg.num_tiles, single_tile_size});
    SetRuntimeArgs(program_, writer, core, {out_buf->address(), (uint32_t)cfg.num_tiles});
    SetRuntimeArgs(program_, compute, core, {(uint32_t)cfg.num_tiles});

    distributed::EnqueueMeshWorkload(cq, workload, false);

    std::vector<uint32_t> out_packed;
    distributed::ReadShard(cq, out_packed, out_buf, zero, true);

    // Compare results
    auto out_unpacked = tt::test_utils::unpack_vector<bfloat16, uint32_t>(out_packed);

    bool pcc_close =
        test_utils::is_close_vectors<bfloat16>(out_unpacked, golden, [&](float a, float b) { return is_close(a, b); });
    return pcc_close;
}

}  // namespace unit_tests::compute::binary_bcast_single_tile

TEST_F(MeshDeviceFixture, TensixBinaryBroadcastSingleTileAdd) {
    unit_tests::compute::binary_bcast_single_tile::Config cfg{
        .num_tiles = 4, .l1_data_format = tt::DataFormat::Float16_b};
    for (unsigned int id = 0; id < num_devices_; id++) {
        ASSERT_TRUE(unit_tests::compute::binary_bcast_single_tile::run_test(devices_.at(id), cfg));
    }
}

}  // namespace tt::tt_metal


