// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <algorithm>
#include <bit>
#include <cstdint>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-logger/tt-logger.hpp>
#include "llk_device_fixture.hpp"

namespace tt::tt_metal {

/*
 * rand_tile fills Float16_b tiles with draws in [from, from + scale]. Every element must be in
 * range and the draws must cover the BF16 grid of the range, whether Dest is 16-bit or 32-bit
 * (issue #14213: a store whose width ignores the Dest mode corrupts half of each tile).
 */
namespace unit_tests::compute::sfpu::rand_tile {

struct RandConfig {
    float from;
    float scale;
    bool fp32_dest_acc_en;
};

constexpr uint32_t num_tiles = 4;
constexpr uint32_t tile_elements = constants::TILE_HW;
constexpr uint32_t seed = 0x12345678;
// Draws must reach this fraction of the BF16 values in the range (MIN_COVERAGE in test_sfpu_rand.py).
constexpr double min_coverage = 0.8;

size_t bf16_values_in(float low, float high) {
    size_t count = 0;
    for (uint32_t bits = 0; bits <= 0xFFFF; ++bits) {
        const float value = std::bit_cast<float>(bits << 16);
        count += (value >= low && value <= high) ? 1 : 0;
    }
    return count;
}

std::vector<bfloat16> run_rand(const std::shared_ptr<distributed::MeshDevice>& mesh_device, const RandConfig& config) {
    auto& cq = mesh_device->mesh_command_queue();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::MeshWorkload workload;
    workload.add_program(device_range, CreateProgram());
    auto& program = workload.get_programs().at(device_range);

    constexpr CoreCoord core = {0, 0};
    constexpr uint32_t single_tile_size = tile_elements * sizeof(bfloat16);
    constexpr uint32_t dram_buffer_size = single_tile_size * num_tiles;

    distributed::ReplicatedBufferConfig global_config{.size = dram_buffer_size};
    distributed::DeviceLocalBufferConfig dram_config{.page_size = dram_buffer_size, .buffer_type = BufferType::DRAM};
    auto dst_dram_buffer = distributed::MeshBuffer::create(global_config, dram_config, mesh_device.get());

    constexpr uint32_t output_cb_index = CBIndex::c_16;
    constexpr uint32_t num_cb_tiles = 2;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_cb_tiles * single_tile_size, {{output_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(output_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_output_config);

    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/rand_sfpu.cpp",
        core,
        ComputeConfig{
            .fp32_dest_acc_en = config.fp32_dest_acc_en,
            .compile_args =
                {num_tiles, seed, std::bit_cast<uint32_t>(config.from), std::bit_cast<uint32_t>(config.scale)},
        });

    SetRuntimeArgs(program, writer_kernel_id, core, {dst_dram_buffer->address(), 0 /* dram bank id */, num_tiles});

    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::vector<uint32_t> result_vec;
    distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_dram_buffer, /*blocking=*/true);
    return unpack_uint32_vec_into_bfloat16_vec(result_vec, [](const bfloat16& a) { return a; });
}

void test_rand(const std::shared_ptr<distributed::MeshDevice>& mesh_device, const RandConfig& config) {
    const float low = config.from;
    const float high = config.from + config.scale;
    const std::vector<bfloat16> result = run_rand(mesh_device, config);
    ASSERT_EQ(result.size(), num_tiles * tile_elements);

    const size_t expected_distinct = std::min<size_t>(bf16_values_in(low, high), tile_elements);
    const double min_distinct = min_coverage * static_cast<double>(expected_distinct);
    for (uint32_t tile = 0; tile < num_tiles; ++tile) {
        std::set<float> distinct;
        uint32_t outside = 0;
        for (uint32_t i = 0; i < tile_elements; ++i) {
            const float value = static_cast<float>(result[tile * tile_elements + i]);
            if (!(value >= low && value <= high)) {
                if (outside == 0) {
                    log_error(tt::LogTest, "tile {} element {} = {} outside [{}, {}]", tile, i, value, low, high);
                }
                ++outside;
            }
            distinct.insert(value);
        }
        EXPECT_EQ(outside, 0u) << "tile " << tile << " fp32_dest_acc_en=" << config.fp32_dest_acc_en;
        EXPECT_GE(static_cast<double>(distinct.size()), min_distinct)
            << "tile " << tile << " fp32_dest_acc_en=" << config.fp32_dest_acc_en << " grid admits "
            << expected_distinct;
    }
}

}  // namespace unit_tests::compute::sfpu::rand_tile

TEST_F(LLKMeshDeviceFixture, TensixComputeRand) {
    if (this->arch_ != ARCH::WORMHOLE_B0 && this->arch_ != ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Covers the Wormhole and Blackhole rand kernels";
    }
    for (const bool fp32_dest_acc_en : {false, true}) {
        for (const auto& [from, scale] : {std::pair{2.0f, 1.0f}, std::pair{-3.0f, 2.0f}}) {
            unit_tests::compute::sfpu::rand_tile::test_rand(
                this->devices_.at(0), {.from = from, .scale = scale, .fp32_dest_acc_en = fp32_dest_acc_en});
        }
    }
}

}  // namespace tt::tt_metal
