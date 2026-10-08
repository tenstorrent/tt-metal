// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// What toy_scaled_add's cache-hit path writes. A kernel's runtime args come in two kinds: per-core args,
// one vector for every core (the work split: which tile-rows a core owns), and common args, one vector
// the kernel's cores share (buffer addresses, alpha). A hit keeps the program a miss built and rewrites
// only what can differ between calls with the same cache key, so the toy op keeps all of that in common
// args and the hit writes a fixed number of values per kernel, on one core or the whole grid.
//
// Each test builds the program a miss builds, overwrites every runtime arg (per-core and common) with a
// sentinel, runs override_runtime_arguments with new buffers and a new alpha, and checks the outcome
// value by value: every per-core arg still holds the sentinel, every common arg holds its new value, and
// on the sharded program every shard-backed circular buffer points at the new shard. The program is
// never launched, so the sentinels never reach a kernel.

#include <algorithm>
#include <bit>
#include <cstdint>
#include <optional>
#include <set>

#include <gtest/gtest.h>
#include <tt-metalium/circular_buffer.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/toy_scaled_add/device/kernels/toy_scaled_add_args.hpp"
#include "ttnn/operations/toy_scaled_add/device/toy_scaled_add_program_factory.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::operations::toy_scaled_add {
namespace {

using namespace tt::tt_metal;
using namespace ::toy_scaled_add;

constexpr uint32_t SENTINEL = 0xDEADBEEF;
constexpr uint32_t WIDTH = 256;
constexpr float MISS_ALPHA = 0.5f;
constexpr float HIT_ALPHA = 1.25f;

Tensor device_tensor(distributed::MeshDevice* device, uint32_t height, const MemoryConfig& memory_config) {
    return create_device_tensor(
        TensorSpec(
            ttnn::Shape({1, 1, height, WIDTH}),
            TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), memory_config)),
        device);
}

// One call's tensors. The second call of a test gets fresh ones, so every address differs from the first.
struct Call {
    Tensor a;
    Tensor b;
    Tensor out;
    std::optional<Tensor> gamma;
    std::optional<Tensor> no_output;  // the inputs refer to it: the factory allocates no preallocated output

    Call(distributed::MeshDevice* device, uint32_t rows, const MemoryConfig& memory_config, bool with_gamma) :
        a(device_tensor(device, rows * tt::constants::TILE_HEIGHT, memory_config)),
        b(device_tensor(device, rows * tt::constants::TILE_HEIGHT, memory_config)),
        out(device_tensor(device, rows * tt::constants::TILE_HEIGHT, memory_config)) {
        if (with_gamma) {
            gamma = device_tensor(device, 1, DRAM_MEMORY_CONFIG);
        }
    }

    ToyScaledAddInputs inputs() const { return {.a = a, .b = b, .gamma = gamma, .output = no_output}; }
    uint32_t gamma_address() const { return gamma.has_value() ? gamma->buffer()->address() : 0u; }
};

ToyScaledAddParams params(distributed::MeshDevice* device, float alpha, const MemoryConfig& memory_config) {
    return {
        .alpha = alpha,
        .output_dtype = DataType::BFLOAT16,
        .output_memory_config = memory_config,
        .compute_kernel_config = init_device_compute_kernel_config(
            device->arch(),
            std::nullopt,
            MathFidelity::HiFi4,
            /*default_approx_mode=*/false,
            /*default_fp32_acc=*/false),
    };
}

constexpr KernelHandle KERNELS[] = {kernel_index::READER, kernel_index::WRITER, kernel_index::COMPUTE};

// Overwrites every per-core and common runtime arg; returns how many cores have per-core args.
size_t poison_runtime_args(Program& program) {
    std::set<std::pair<size_t, size_t>> cores;
    for (const KernelHandle kernel : KERNELS) {
        auto& grid = GetRuntimeArgs(program, kernel);
        for (size_t x = 0; x < grid.size(); ++x) {
            for (size_t y = 0; y < grid[x].size(); ++y) {
                auto& args = grid[x][y];
                for (size_t i = 0; i < args.size(); ++i) {
                    args[i] = SENTINEL;
                }
                if (args.size() > 0) {
                    cores.emplace(x, y);
                }
            }
        }
        auto& common = GetCommonRuntimeArgs(program, kernel);
        for (size_t i = 0; i < common.size(); ++i) {
            common[i] = SENTINEL;
        }
    }
    return cores.size();
}

// Every per-core arg of every kernel still holds the sentinel: the hit wrote none of them.
void expect_per_core_args_untouched(Program& program) {
    for (const KernelHandle kernel : KERNELS) {
        auto& grid = GetRuntimeArgs(program, kernel);
        for (size_t x = 0; x < grid.size(); ++x) {
            for (size_t y = 0; y < grid[x].size(); ++y) {
                const auto& args = grid[x][y];
                for (size_t i = 0; i < args.size(); ++i) {
                    EXPECT_EQ(args[i], SENTINEL) << "kernel " << kernel << ", core (" << x << ", " << y
                                                 << "), per-core arg " << i << " was written on a cache hit";
                }
            }
        }
    }
}

class ToyScaledAddHitPath : public TTNNFixtureWithDevice {};

TEST_F(ToyScaledAddHitPath, InterleavedHitWritesOnlyCommonArgs) {
    const CoreCoord grid = device_->compute_with_storage_grid_size();
    const MemoryConfig memory_config = DRAM_MEMORY_CONFIG;
    // One tile-row per core: one core, then every core of the grid.
    for (const uint32_t rows : {1u, static_cast<uint32_t>(grid.x * grid.y)}) {
        for (const bool with_gamma : {false, true}) {
            SCOPED_TRACE(testing::Message() << rows << " tile-row(s), gamma " << with_gamma);
            Call miss(device_, rows, memory_config, with_gamma);
            Call hit(device_, rows, memory_config, with_gamma);
            Program program(InterleavedProgramFactory::create_descriptor(
                params(device_, MISS_ALPHA, memory_config), miss.inputs(), miss.out));
            EXPECT_EQ(poison_runtime_args(program), rows);

            InterleavedProgramFactory::override_runtime_arguments(
                program, params(device_, HIT_ALPHA, memory_config), hit.inputs(), hit.out);

            expect_per_core_args_untouched(program);
            const auto& reader = GetCommonRuntimeArgs(program, kernel_index::READER);
            ASSERT_EQ(reader.size(), reader_arg::COUNT);
            EXPECT_EQ(reader[reader_arg::A_ADDR], hit.a.buffer()->address());
            EXPECT_EQ(reader[reader_arg::B_ADDR], hit.b.buffer()->address());
            EXPECT_EQ(reader[reader_arg::GAMMA_ADDR], hit.gamma_address());
            const auto& writer = GetCommonRuntimeArgs(program, kernel_index::WRITER);
            ASSERT_EQ(writer.size(), writer_arg::COUNT);
            EXPECT_EQ(writer[writer_arg::OUT_ADDR], hit.out.buffer()->address());
            const auto& compute = GetCommonRuntimeArgs(program, kernel_index::COMPUTE);
            ASSERT_EQ(compute.size(), compute_arg::COUNT);
            EXPECT_EQ(compute[compute_arg::ALPHA_BITS], std::bit_cast<uint32_t>(HIT_ALPHA));
        }
    }
}

TEST_F(ToyScaledAddHitPath, HeightShardedHitWritesOnlyCommonArgsAndShardAddresses) {
    const CoreCoord grid = device_->compute_with_storage_grid_size();
    const uint32_t num_cores = std::min<uint32_t>(grid.x, 8);
    // One tile-row per core, along the grid's first row.
    const MemoryConfig memory_config{
        TensorMemoryLayout::HEIGHT_SHARDED,
        BufferType::L1,
        ShardSpec{
            CoreRangeSet{CoreRange{CoreCoord{0, 0}, CoreCoord{num_cores - 1, 0}}},
            {tt::constants::TILE_HEIGHT, WIDTH},
            ShardOrientation::ROW_MAJOR}};
    for (const bool with_gamma : {false, true}) {
        SCOPED_TRACE(testing::Message() << "gamma " << with_gamma);
        Call miss(device_, num_cores, memory_config, with_gamma);
        Call hit(device_, num_cores, memory_config, with_gamma);
        Program program(HeightShardedProgramFactory::create_descriptor(
            params(device_, MISS_ALPHA, memory_config), miss.inputs(), miss.out));
        EXPECT_EQ(poison_runtime_args(program), num_cores);

        HeightShardedProgramFactory::override_runtime_arguments(
            program, params(device_, HIT_ALPHA, memory_config), hit.inputs(), hit.out);

        expect_per_core_args_untouched(program);
        const auto& reader = GetCommonRuntimeArgs(program, kernel_index::READER);
        ASSERT_EQ(reader.size(), sharded_reader_arg::COUNT);
        EXPECT_EQ(reader[sharded_reader_arg::GAMMA_ADDR], hit.gamma_address());
        const auto& compute = GetCommonRuntimeArgs(program, kernel_index::COMPUTE);
        ASSERT_EQ(compute.size(), compute_arg::COUNT);
        EXPECT_EQ(compute[compute_arg::ALPHA_BITS], std::bit_cast<uint32_t>(HIT_ALPHA));
        // The shards are a circular buffer's memory: one address per buffer, whatever the core count.
        size_t shard_backed = 0;
        for (const auto& circular_buffer : program.circular_buffers()) {
            if (!circular_buffer->globally_allocated()) {
                continue;
            }
            ++shard_backed;
            const auto& indices = circular_buffer->buffer_indices();
            const Tensor& backing = indices.contains(cb::A) ? hit.a : indices.contains(cb::B) ? hit.b : hit.out;
            EXPECT_EQ(
                GetCircularBufferConfig(program, circular_buffer->id()).globally_allocated_address(),
                backing.buffer()->address());
        }
        EXPECT_EQ(shard_backed, 3u);
    }
}

}  // namespace
}  // namespace ttnn::operations::toy_scaled_add
