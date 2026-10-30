// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <random>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>

#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "ttnn/unit_tests/gtests/command_list_capture_test_ops.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/qkv_causal_conv1d_silu_device_operation.hpp"

namespace ttnn::experimental::test {
namespace {

using tt::constants::TILE_HEIGHT;
using tt::constants::TILE_WIDTH;
using tt::tt_metal::distributed::MeshShape;
namespace prim = ttnn::experimental::prim;

// Values on every device of a mesh tensor, in mesh order.
template <typename T>
std::vector<std::vector<T>> per_device(const Tensor& tensor) {
    std::vector<std::vector<T>> values;
    for (const auto& device_tensor : ttnn::distributed::get_device_tensors(tensor)) {
        values.push_back(device_tensor.to_vector<T>());
    }
    return values;
}

std::vector<std::vector<float>> per_device(const std::vector<Tensor>& tensors) {
    std::vector<std::vector<float>> values;
    for (const auto& tensor : tensors) {
        for (const auto& device_values : per_device<bfloat16>(tensor)) {
            values.emplace_back(device_values.begin(), device_values.end());
        }
    }
    return values;
}

class CommandListMeshTest : public tt::tt_metal::MeshDeviceFixtureBase {
protected:
    CommandListMeshTest(const MeshShape& shape, tt::tt_fabric::FabricConfig fabric_config, uint32_t sp_axis) :
        MeshDeviceFixtureBase(Config{
            .mesh_shape = shape,
            .l1_small_size = 1 << 15,
            .trace_region_size = 32 << 20,
            .fabric_config = fabric_config}),
        sp_axis_(sp_axis) {}

    ttnn::MeshDevice& mesh() const { return *mesh_device_; }

    Tensor replicated(const Tensor& host) const {
        return ttnn::distributed::distribute_tensor(
            host, *ttnn::distributed::replicate_tensor_to_mesh_mapper(mesh()), mesh());
    }

    Tensor replicated_host(const Tensor& host) const {
        return ttnn::distributed::distribute_tensor(host, *ttnn::distributed::replicate_tensor_to_mesh_mapper(mesh()));
    }

    // One patch reaches the programs on every device.
    void check_copy_rows_on_every_device() const {
        constexpr uint32_t kCapacity = 8;
        constexpr uint32_t kWidth = 2 * TILE_WIDTH;
        const tt::tt_metal::TensorSpec spec(
            ttnn::Shape{1, 1, kCapacity * TILE_HEIGHT, kWidth},
            tt::tt_metal::TensorLayout(DataType::UINT32, tt::tt_metal::PageConfig(Layout::TILE), MemoryConfig{}));
        std::vector<uint32_t> values(kCapacity * TILE_HEIGHT * kWidth);
        for (uint32_t i = 0; i < values.size(); ++i) {
            values[i] = i + 1;
        }
        const auto input = replicated(Tensor::from_vector(values, spec));
        const auto zeros = replicated_host(Tensor::from_vector(std::vector<uint32_t>(values.size(), 0), spec));

        CommandListCapture capture(mesh());
        auto out = capture.record<CopyRowsOp>({.num_tile_rows = kCapacity}, {.input = input});
        auto list = capture.build();
        for (uint32_t tile_rows : {2u, kCapacity}) {
            copy_to_device(zeros, out);
            list.set("tile_rows", tile_rows);
            list.replay(/*blocking=*/true);
            auto expected = values;
            std::fill(expected.begin() + (tile_rows * TILE_HEIGHT * kWidth), expected.end(), 0);
            for (const auto& device_values : per_device<uint32_t>(out)) {
                EXPECT_EQ(device_values, expected) << tile_rows << " tile rows";
            }
        }
    }

    // qkv_causal_conv1d_silu_with_carry's inputs on this mesh, with its persistent exchange buffer and semaphores.
    struct Exchange {
        prim::QkvCausalConv1dSiluParams params;
        prim::QkvCausalConv1dSiluInputs inputs;
        std::vector<tt::tt_metal::GlobalSemaphore> semaphores;
    };

    static constexpr uint32_t kLocalRows = 64;
    static constexpr uint32_t kHeadWidth = 64;
    static constexpr uint32_t kChannels = 3 * kHeadWidth;

    Tensor start_tensor(uint32_t start) const {
        return Tensor::from_vector(
            std::vector<uint32_t>{start},
            tt::tt_metal::TensorSpec(
                ttnn::Shape{1},
                tt::tt_metal::TensorLayout(
                    DataType::UINT32, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), MemoryConfig{})));
    }

    Exchange make_exchange(uint32_t start) const {
        const uint32_t sp_size = mesh().shape()[sp_axis_];
        std::mt19937 generator(613);
        std::normal_distribution<float> distribution;
        const auto random_host = [&](const ttnn::Shape& shape, Layout layout) {
            std::vector<bfloat16> values(shape.volume());
            for (auto& value : values) {
                value = bfloat16(distribution(generator));
            }
            return Tensor::from_vector(
                values,
                tt::tt_metal::TensorSpec(
                    shape,
                    tt::tt_metal::TensorLayout(
                        DataType::BFLOAT16, tt::tt_metal::PageConfig(layout), ttnn::DRAM_MEMORY_CONFIG)));
        };

        Exchange exchange;
        exchange.inputs.input = ttnn::distributed::distribute_tensor(
            random_host(ttnn::Shape{1, sp_size * kLocalRows, kChannels}, Layout::TILE),
            *ttnn::distributed::shard_tensor_to_mesh_mapper(mesh(), 1, sp_axis_),
            mesh());
        exchange.inputs.history =
            replicated(random_host(ttnn::Shape{1, qkv_conv::history_rows, kChannels}, Layout::ROW_MAJOR));
        exchange.inputs.tap0 = replicated(random_host(ttnn::Shape{1, 1, kChannels}, Layout::TILE));
        exchange.inputs.tap1 = replicated(random_host(ttnn::Shape{1, 1, kChannels}, Layout::TILE));
        exchange.inputs.tap2 = replicated(random_host(ttnn::Shape{1, 1, kChannels}, Layout::TILE));
        exchange.inputs.tap3 = replicated(random_host(ttnn::Shape{1, 1, kChannels}, Layout::TILE));
        exchange.inputs.actual_start = replicated(start_tensor(start));
        exchange.inputs.exchange_buffer = replicated(Tensor::from_vector(
            std::vector<bfloat16>(qkv_conv::exchange_rows * kChannels, bfloat16(0.0f)),
            tt::tt_metal::TensorSpec(
                ttnn::Shape{1, qkv_conv::exchange_rows, kChannels},
                tt::tt_metal::TensorLayout(
                    DataType::BFLOAT16, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), ttnn::DRAM_MEMORY_CONFIG))));

        const auto exchange_core = prim::qkv_causal_conv1d_silu_exchange_core(mesh().compute_with_storage_grid_size());
        exchange.params = prim::QkvCausalConv1dSiluParams{
            .sequence = kLocalRows,
            .q_width = kHeadWidth,
            .k_width = kHeadWidth,
            .v_width = kHeadWidth,
            .channel_chunk_size = kHeadWidth,
            .sequence_parallel_axis = sp_axis_,
            .input_column_offset = 0,
            .output_mem_config = ttnn::DRAM_MEMORY_CONFIG,
            .compute_kernel_config = ttnn::init_device_compute_kernel_config(
                mesh().arch(),
                std::nullopt,
                tt::tt_metal::MathFidelity::HiFi4,
                /*default_approx_mode=*/false,
                /*default_fp32_acc=*/false,
                /*default_l1_acc=*/false)};
        for (auto& address : exchange.params.exchange_semaphore_addresses) {
            exchange.semaphores.push_back(ttnn::global_semaphore::create_global_semaphore(
                &mesh(), CoreRangeSet(CoreRange(exchange_core, exchange_core)), 0, BufferType::L1_SMALL));
            address = static_cast<uint32_t>(exchange.semaphores.back().address());
        }
        return exchange;
    }

    static std::vector<std::vector<float>> launch_eager(const Exchange& exchange) {
        return per_device(
            device_operation::launch<prim::QkvCausalConv1dSiluOperation>(exchange.params, exchange.inputs));
    }

    // A fabric op with per-device programs and global semaphores whose counts carry across calls replays like its
    // eager launch: repeatedly, interleaved with eager launches, and after a rewrite of its device-side start.
    void check_fabric_exchange_replays() const {
        auto exchange = make_exchange(/*start=*/0);
        const auto reference = launch_eager(exchange);
        ASSERT_EQ(launch_eager(exchange), reference) << "back-to-back eager launches differ";

        CommandListCapture capture(mesh());
        const auto recorded = capture.record<prim::QkvCausalConv1dSiluOperation>(exchange.params, exchange.inputs);
        auto list = capture.build();
        for (int replay = 0; replay < 3; ++replay) {
            list.replay(/*blocking=*/true);
            EXPECT_EQ(per_device(recorded), reference) << "replay " << replay;
        }

        EXPECT_EQ(launch_eager(exchange), reference) << "eager launch between replays";
        list.replay(/*blocking=*/true);
        EXPECT_EQ(per_device(recorded), reference) << "replay after an eager launch";

        // Half way into the second rank, so its first rank also holds the tail of the sequence.
        copy_to_device(replicated_host(start_tensor(kLocalRows + (kLocalRows / 2))), exchange.inputs.actual_start);
        const auto moved = launch_eager(exchange);
        EXPECT_NE(moved, reference) << "the start does not change the outputs";
        list.replay(/*blocking=*/true);
        EXPECT_EQ(per_device(recorded), moved) << "replay after the start moved";
    }

    uint32_t sp_axis_;
};

// QuietBox: an SP4 ring, whose axis turns at every chip, on a 1D ring fabric.
class CommandListMesh1x4RingTest : public CommandListMeshTest {
protected:
    CommandListMesh1x4RingTest() :
        CommandListMeshTest(MeshShape(1, 4), tt::tt_fabric::FabricConfig::FABRIC_1D_RING, /*sp_axis=*/1) {}
};

// QuietBox: SP2xTP2 on a 2D fabric.
class CommandListMesh2x2Fabric2DTest : public CommandListMeshTest {
protected:
    CommandListMesh2x2Fabric2DTest() :
        CommandListMeshTest(MeshShape(2, 2), tt::tt_fabric::FabricConfig::FABRIC_2D, /*sp_axis=*/0) {}
};

TEST_F(CommandListMesh1x4RingTest, PatchReachesEveryDevice) { check_copy_rows_on_every_device(); }

TEST_F(CommandListMesh1x4RingTest, FabricExchangeReplaysLikeEagerLaunches) { check_fabric_exchange_replays(); }

TEST_F(CommandListMesh2x2Fabric2DTest, PatchReachesEveryDevice) { check_copy_rows_on_every_device(); }

TEST_F(CommandListMesh2x2Fabric2DTest, FabricExchangeReplaysLikeEagerLaunches) { check_fabric_exchange_replays(); }

}  // namespace
}  // namespace ttnn::experimental::test
