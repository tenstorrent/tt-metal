// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Tensor prefetcher requests recorded into a CommandList.
//
// The consumer is the PrefetcherPipe validator, which checks every delivered entry against the weight in DRAM and
// hangs on a mismatch. A replay that did not re-send the recorded request would leave it waiting on an empty pipe, so
// every replay finishing is what shows the request went out again.

#include <gtest/gtest.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/command_list.hpp>
#include <tt-metalium/experimental/prefetcher_pipe.hpp>
#include <tt-metalium/experimental/tensor_prefetcher.hpp>

#include "ttnn/operations/experimental/test/prefetcher_consumer/dram_prefetcher_validator.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::test::command_list_tensor_prefetcher {
namespace {

namespace metal_exp = ::tt::tt_metal::experimental;
using ::tt::constants::TILE_HEIGHT;
using ::tt::constants::TILE_WIDTH;
using ::tt::tt_metal::BufferType;
using ::tt::tt_metal::CoreCoord;
using ::tt::tt_metal::CoreRange;
using ::tt::tt_metal::CoreRangeSet;
using ::tt::tt_metal::DataType;
using ::tt::tt_metal::Layout;
using ::tt::tt_metal::MemoryConfig;
using ::tt::tt_metal::PageConfig;
using ::tt::tt_metal::ShardOrientation;
using ::tt::tt_metal::ShardSpec;
using ::tt::tt_metal::TensorLayout;
using ::tt::tt_metal::TensorMemoryLayout;
using ::tt::tt_metal::TensorSpec;

constexpr uint32_t kTilesPerReceiver = 2;
constexpr uint32_t kKBlockTiles = 2;
// Ring depth in entries. Smaller than the weight's block count, so a replay only finishes if the consumer and the
// prefetcher keep trading credits.
constexpr uint32_t kRingEntries = 4;
constexpr uint32_t kBfloat16TileBytes = TILE_HEIGHT * TILE_WIDTH * sizeof(uint16_t);
constexpr uint32_t kReplayCount = 3;
// CommandList command buffers come out of the trace region.
constexpr int kTraceRegionSize = 64 << 20;

class CommandListTensorPrefetcherFixture : public TTNNFixtureWithDevice {
protected:
    CommandListTensorPrefetcherFixture() : TTNNFixtureWithDevice(kTraceRegionSize, DEFAULT_L1_SMALL_SIZE) {}

    void SetUp() override {
        if (!check_dispatch_mode()) {
            GTEST_SKIP() << "Command lists require fast dispatch";
        }
        TTNNFixtureWithDevice::SetUp();
        if (!metal_exp::IsTensorPrefetcherSupported(*device_)) {
            GTEST_SKIP() << "Tensor prefetcher unsupported: needs Blackhole with firmware >= 19.12.0.0 and the "
                            "streaming profiler off";
        }
    }
};

// A K-row-major weight with one receiver per bank, so ring position == bank, and the pipes that deliver it.
struct PrefetchCase {
    explicit PrefetchCase(tt::tt_metal::distributed::MeshDevice* device) {
        const uint32_t num_banks = device->dram_grid_size().x;
        ring_size = num_banks;
        const uint32_t k = ring_size * kKBlockTiles * TILE_HEIGHT;
        const uint32_t n = num_banks * kTilesPerReceiver * TILE_WIDTH;

        CoreRangeSet receiver_domain;
        for (uint32_t bank = 0; bank < num_banks; ++bank) {
            const CoreRangeSet receiver(CoreRange(CoreCoord(bank, 0), CoreCoord(bank, 0)));
            bank_to_receivers.emplace_back(bank, receiver);
            receiver_domain = receiver_domain.merge(receiver);
        }

        // Pipe rings come from the persistent L1 arena, which refuses a core a live Program holds, so carve them
        // before anything runs.
        const uint32_t entry_bytes = kKBlockTiles * kTilesPerReceiver * kBfloat16TileBytes;
        space = std::make_unique<metal_exp::PrefetcherPipeSpace>(metal_exp::CreatePrefetcherPipeSpace(
            *device,
            {.num_dram_senders = 2 * num_banks,
             .receiver_domain = receiver_domain,
             .ring_size = kRingEntries * entry_bytes,
             .max_receivers_per_pipe = 1}));
        for (auto& pipe : metal_exp::CreatePrefetcherPipesForTensorPrefetcher(
                 *space, bank_to_receivers, /*support_multi_receiver_shards=*/true)) {
            pipes.push_back(std::make_shared<metal_exp::PrefetcherPipe>(std::move(pipe)));
        }

        std::mt19937 rng(0xC0FFEE);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<::bfloat16> values(static_cast<size_t>(k) * n);
        for (auto& value : values) {
            value = ::bfloat16(dist(rng));
        }
        const MemoryConfig weight_memory_config(
            TensorMemoryLayout::WIDTH_SHARDED,
            BufferType::DRAM,
            ShardSpec(
                CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(num_banks - 1, 0))),
                {k, n / num_banks},
                ShardOrientation::ROW_MAJOR));
        const TensorSpec weight_spec(
            ttnn::Shape({1, 1, k, n}),
            TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), weight_memory_config));
        weight = Tensor::from_vector(std::move(values), weight_spec).to_device(device);
    }

    std::vector<std::reference_wrapper<const metal_exp::PrefetcherPipe>> pipe_refs() const {
        std::vector<std::reference_wrapper<const metal_exp::PrefetcherPipe>> refs;
        for (const auto& pipe : pipes) {
            refs.emplace_back(*pipe);
        }
        return refs;
    }

    // One layer: the weight's K split into one block per ring position.
    std::vector<metal_exp::TensorPrefetcherInput> inputs() const { return {{weight.mesh_tensor(), ring_size}}; }

    uint32_t ring_size = 0;
    std::vector<std::pair<uint32_t, CoreRangeSet>> bank_to_receivers;
    // Must outlive the pipes carved from it.
    std::unique_ptr<metal_exp::PrefetcherPipeSpace> space;
    std::vector<std::shared_ptr<metal_exp::PrefetcherPipe>> pipes;
    Tensor weight;
};

TEST_F(CommandListTensorPrefetcherFixture, EveryReplayResendsTheRecordedRequest) {
    PrefetchCase prefetch(device_);
    metal_exp::StartTensorPrefetcher(*device_, {});
    {
        metal_exp::CommandListBuilder builder(*device_);
        builder.add_tensor_prefetcher_request(prefetch.pipe_refs(), std::nullopt, prefetch.inputs());
        auto validator = ttnn::operations::experimental::test::make_tensor_prefetcher_pipe_validator_workload(
            device_, prefetch.weight, /*num_layers=*/1, /*print_stride=*/0, prefetch.pipes, prefetch.bank_to_receivers);
        builder.add(validator);
        auto command_list = builder.build(device_->mesh_command_queue());
        builder.deallocate();

        for (uint32_t i = 0; i < kReplayCount; ++i) {
            metal_exp::EnqueueCommandList(device_->mesh_command_queue(), command_list, /*blocking=*/true);
        }
        command_list.deallocate();
    }
    metal_exp::StopTensorPrefetcher(*device_);
}

TEST_F(CommandListTensorPrefetcherFixture, RecordedRequestsDoNotOutliveThePrefetcherRun) {
    PrefetchCase prefetch(device_);
    metal_exp::StartTensorPrefetcher(*device_, {});
    metal_exp::CommandListBuilder builder(*device_);
    builder.add_tensor_prefetcher_request(prefetch.pipe_refs(), std::nullopt, prefetch.inputs());
    auto validator = ttnn::operations::experimental::test::make_tensor_prefetcher_pipe_validator_workload(
        device_, prefetch.weight, /*num_layers=*/1, /*print_stride=*/0, prefetch.pipes, prefetch.bank_to_receivers);
    builder.add(validator);
    auto command_list = builder.build(device_->mesh_command_queue());
    metal_exp::StopTensorPrefetcher(*device_);

    metal_exp::StartTensorPrefetcher(*device_, {});
    // The list's requests were serialized for the stopped run, so replay refuses before enqueueing anything.
    EXPECT_ANY_THROW(metal_exp::EnqueueCommandList(device_->mesh_command_queue(), command_list, /*blocking=*/true));
    // So does recording more next to them, until clear() drops them.
    EXPECT_ANY_THROW(builder.add_tensor_prefetcher_request(prefetch.pipe_refs(), std::nullopt, prefetch.inputs()));
    builder.clear();
    builder.add_tensor_prefetcher_request(prefetch.pipe_refs(), std::nullopt, prefetch.inputs());
    builder.add(validator);
    command_list.deallocate();
    command_list = builder.build(device_->mesh_command_queue());
    builder.deallocate();

    metal_exp::EnqueueCommandList(device_->mesh_command_queue(), command_list, /*blocking=*/true);
    command_list.deallocate();
    metal_exp::StopTensorPrefetcher(*device_);
}

}  // namespace
}  // namespace ttnn::test::command_list_tensor_prefetcher
