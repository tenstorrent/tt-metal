// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The Tensor prefetcher where fast dispatch runs it on a dispatch engine: the same Start / Queue / Stop and
// the same PrefetcherPipes as on a DRAM-core prefetcher, with a reader and a pusher on the dispatch engine
// streaming a receiver-contiguous weight to the PrefetcherPipe validator. The validator checks every entry
// against the weight in DRAM and hangs on a mismatch, so a run that finishes delivered every block intact.

#include <gtest/gtest.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/prefetcher_pipe.hpp>
#include <tt-metalium/experimental/tensor_prefetcher.hpp>

#include "ttnn/operations/experimental/test/prefetcher_consumer/dram_prefetcher_validator.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::test::tensor_prefetcher_dispatch_engine {
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
using ::tt::tt_metal::NdShardSpec;
using ::tt::tt_metal::PageConfig;
using ::tt::tt_metal::ShardDistributionStrategy;
using ::tt::tt_metal::ShardOrientation;
using ::tt::tt_metal::TensorLayout;
using ::tt::tt_metal::TensorSpec;

constexpr uint32_t kTilesPerReceiver = 2;
constexpr uint32_t kKBlockTiles = 2;
constexpr uint32_t kBfloat16TileBytes = TILE_HEIGHT * TILE_WIDTH * sizeof(uint16_t);

class DispatchEngineTensorPrefetcherFixture : public TTNNFixtureWithDevice {
protected:
    void SetUp() override {
        if (!check_dispatch_mode()) {
            GTEST_SKIP() << "The dispatch-engine Tensor prefetcher runs with fast dispatch";
        }
        TTNNFixtureWithDevice::SetUp();
        if (device_->arch() != tt::ARCH::QUASAR || !metal_exp::IsTensorPrefetcherSupported(*device_)) {
            GTEST_SKIP() << "Needs fast dispatch on a Quasar dispatch engine serving one command queue";
        }
    }
};

// A receiver-contiguous weight -- one (K, N / receivers) slab per receiver, round-robin over the DRAM banks --
// and the PrefetcherPipes that deliver it to `receivers_per_bank` receivers of every bank, `ring_entries`
// blocks deep per receiver. A dispatch engine holds one sender endpoint per bank, so each bank's
// receivers share one pipe.
struct ReceiverContiguousCase {
    ReceiverContiguousCase(
        tt::tt_metal::distributed::MeshDevice* device, uint32_t receivers_per_bank, uint32_t ring_entries) {
        const uint32_t num_banks = device->dram_grid_size().x;
        const uint32_t num_receivers = receivers_per_bank * num_banks;
        const CoreCoord grid = device->compute_with_storage_grid_size();
        TT_FATAL(grid.x * grid.y >= num_receivers, "The test needs {} worker receivers", num_receivers);
        std::vector<CoreCoord> receivers;
        for (uint32_t i = 0; i < num_receivers; ++i) {
            receivers.emplace_back(i % grid.x, i / grid.x);
        }
        // Round-robin slabs: slab s lives in bank s % num_banks, so bank b serves receivers b, b + num_banks, ...
        CoreRangeSet receiver_domain;
        for (uint32_t bank = 0; bank < num_banks; ++bank) {
            CoreRangeSet bank_receivers;
            for (uint32_t r = bank; r < num_receivers; r += num_banks) {
                bank_receivers = bank_receivers.merge(CoreRangeSet(CoreRange(receivers[r])));
            }
            bank_to_receivers.emplace_back(bank, bank_receivers);
            receiver_domain = receiver_domain.merge(bank_receivers);
        }

        // The validator delivers each receiver's slab in one K block per receiver (its ring size).
        block_count = num_receivers;
        const uint32_t k = block_count * kKBlockTiles * TILE_HEIGHT;
        const uint32_t n = num_receivers * kTilesPerReceiver * TILE_WIDTH;
        const uint32_t entry_bytes = kKBlockTiles * kTilesPerReceiver * kBfloat16TileBytes;
        space = std::make_unique<metal_exp::PrefetcherPipeSpace>(metal_exp::CreatePrefetcherPipeSpace(
            *device,
            {.num_dram_senders = num_banks,
             .receiver_domain = receiver_domain,
             .ring_size = ring_entries * entry_bytes,
             .max_receivers_per_pipe = num_receivers}));
        for (auto& pipe : metal_exp::CreatePrefetcherPipesForTensorPrefetcher(*space, bank_to_receivers)) {
            pipes.push_back(std::make_shared<metal_exp::PrefetcherPipe>(std::move(pipe)));
        }

        std::mt19937 rng(0xC0FFEE + num_receivers);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<::bfloat16> values(static_cast<size_t>(k) * n);
        for (auto& value : values) {
            value = ::bfloat16(dist(rng));
        }
        const MemoryConfig weight_memory_config(
            BufferType::DRAM,
            NdShardSpec{
                ttnn::Shape({k, n / num_receivers}),
                CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(num_banks - 1, 0))),
                ShardOrientation::ROW_MAJOR,
                ShardDistributionStrategy::ROUND_ROBIN_1D});
        const TensorSpec weight_spec(
            ttnn::Shape({1, 1, k, n}),
            TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), weight_memory_config));
        weight = Tensor::from_vector(std::move(values), weight_spec).to_device(device);
    }

    std::vector<std::reference_wrapper<const metal_exp::PrefetcherPipe>> pipe_refs() const {
        std::vector<std::reference_wrapper<const metal_exp::PrefetcherPipe>> refs;
        refs.reserve(pipes.size());
        for (const auto& pipe : pipes) {
            refs.emplace_back(*pipe);
        }
        return refs;
    }

    uint32_t block_count = 0;
    std::vector<std::pair<uint32_t, CoreRangeSet>> bank_to_receivers;
    // Must outlive the pipes carved from it.
    std::unique_ptr<metal_exp::PrefetcherPipeSpace> space;
    std::vector<std::shared_ptr<metal_exp::PrefetcherPipe>> pipes;
    Tensor weight;
};

// Queues one request for the weight `num_layers` times over.
void queue_layers(
    tt::tt_metal::distributed::MeshDevice* device, const ReceiverContiguousCase& prefetch, uint32_t num_layers) {
    const std::vector<metal_exp::TensorPrefetcherInput> inputs(
        num_layers, metal_exp::TensorPrefetcherInput{prefetch.weight.mesh_tensor(), prefetch.block_count});
    metal_exp::QueueTensorPrefetcherRequest(*device, prefetch.pipe_refs(), std::nullopt, inputs);
}

// Validates every entry of `num_layers` copies of the weight as the receivers consume them.
void validate_layers(
    tt::tt_metal::distributed::MeshDevice* device, const ReceiverContiguousCase& prefetch, uint32_t num_layers) {
    ttnn::operations::experimental::test::test_tensor_prefetcher_pipe_validator(
        device, prefetch.weight, num_layers, /*print_stride=*/0, prefetch.pipes, prefetch.bank_to_receivers);
}

// Every block of one tensor arrives, through two-entry rings.
TEST_F(DispatchEngineTensorPrefetcherFixture, DeliversReceiverContiguousWeight) {
    ReceiverContiguousCase prefetch(device_, /*receivers_per_bank=*/2, /*ring_entries=*/2);
    metal_exp::StartTensorPrefetcher(*device_, {});
    queue_layers(device_, prefetch, /*num_layers=*/1);
    validate_layers(device_, prefetch, /*num_layers=*/1);
    metal_exp::StopTensorPrefetcher(*device_);
}

// One-entry rings, several layers: every block after the first waits on the receiver's ack, which crosses to
// the dispatch engine, and the pipes' cursors wrap and persist across tensors and requests.
TEST_F(DispatchEngineTensorPrefetcherFixture, TradesCreditsAcrossLayersAndRuns) {
    ReceiverContiguousCase prefetch(device_, /*receivers_per_bank=*/2, /*ring_entries=*/1);
    for (uint32_t run = 0; run < 2; ++run) {
        metal_exp::StartTensorPrefetcher(*device_, {});
        queue_layers(device_, prefetch, /*num_layers=*/3);
        validate_layers(device_, prefetch, /*num_layers=*/3);
        metal_exp::StopTensorPrefetcher(*device_);
    }
}

// Back-to-back requests for different receiver counts: the stage ring is laid out per receiver count, so the
// second request waits for the first one's blocks to leave the stage ring before it stages into it.
TEST_F(DispatchEngineTensorPrefetcherFixture, RelaysOutStageRingForANewReceiverCount) {
    ReceiverContiguousCase wide(device_, /*receivers_per_bank=*/2, /*ring_entries=*/1);
    ReceiverContiguousCase narrow(device_, /*receivers_per_bank=*/1, /*ring_entries=*/1);
    metal_exp::StartTensorPrefetcher(*device_, {});
    queue_layers(device_, wide, /*num_layers=*/2);
    queue_layers(device_, narrow, /*num_layers=*/2);
    validate_layers(device_, wide, /*num_layers=*/2);
    validate_layers(device_, narrow, /*num_layers=*/2);
    metal_exp::StopTensorPrefetcher(*device_);
}

}  // namespace
}  // namespace ttnn::test::tensor_prefetcher_dispatch_engine
