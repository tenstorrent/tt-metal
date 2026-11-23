// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The Tensor prefetcher where fast dispatch runs it on a dispatch engine: the same Start / Queue / Stop and
// the same PrefetcherPipes as on a DRAM-core prefetcher, with a reader and a pusher on the dispatch engine
// streaming a receiver-contiguous weight to the PrefetcherPipe validator, whether the request is queued
// directly, behind a CQ fence or an op signal, or replayed from a trace or a CommandList, and the groups a
// mask selects of a grouped weight. The validator checks
// every entry against the weight in DRAM and hangs on a mismatch, so a run that finishes delivered every block
// intact.

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
#include <tt-metalium/experimental/command_list.hpp>
#include <tt-metalium/experimental/prefetcher_pipe.hpp>
#include <tt-metalium/experimental/tensor_prefetcher.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "ttnn/operations/experimental/tensor_prefetcher/tensor_prefetcher.hpp"
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
using ::tt::tt_metal::ShardSpec;
using ::tt::tt_metal::TensorLayout;
using ::tt::tt_metal::TensorMemoryLayout;
using ::tt::tt_metal::TensorSpec;

constexpr uint32_t kTilesPerReceiver = 2;
constexpr uint32_t kKBlockTiles = 2;
constexpr uint32_t kBfloat16TileBytes = TILE_HEIGHT * TILE_WIDTH * sizeof(uint16_t);
constexpr uint32_t kReplayCount = 2;
// Traces and CommandList command buffers come out of the trace region; these hold a request and one
// validator program.
constexpr int kTraceRegionSize = 4 << 20;

class DispatchEngineTensorPrefetcherFixture : public TTNNFixtureWithDevice {
protected:
    DispatchEngineTensorPrefetcherFixture() : TTNNFixtureWithDevice(kTraceRegionSize, DEFAULT_L1_SMALL_SIZE) {}

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

// How a case leaves its weight's write. to_device only enqueues it, and a prefetch request is not ordered against
// the command queue, so the reader could stage DRAM the write has not reached yet. Most tests finish the queue
// first; a test of a device-side ordering (a CQ fence, an op signal raised behind the write) leaves the write in
// flight so that the ordering is what protects the weight.
enum class WeightWrite { Finished, InFlight };

// A receiver-contiguous weight -- one (K, N / receivers) slab per receiver, round-robin over the DRAM banks --
// and the PrefetcherPipes that deliver it to `receivers_per_bank` receivers of every bank, `ring_entries`
// blocks deep per receiver. A dispatch engine holds one sender endpoint per bank, so each bank's
// receivers share one pipe. With `num_groups` above 1 the weight is grouped, (1, num_groups, K, N), and every
// receiver slab stacks one (K, N / receivers) block per group.
struct ReceiverContiguousCase {
    ReceiverContiguousCase(
        tt::tt_metal::distributed::MeshDevice* device,
        uint32_t receivers_per_bank,
        uint32_t ring_entries,
        uint32_t num_groups = 1,
        WeightWrite weight_write = WeightWrite::Finished) {
        const uint32_t num_banks = device->dram_grid_size().x;
        const uint32_t num_receivers = receivers_per_bank * num_banks;
        const CoreCoord grid = device->compute_with_storage_grid_size();
        TT_FATAL(grid.x * grid.y >= num_receivers, "The test needs {} worker receivers", num_receivers);
        std::vector<CoreCoord> receivers;
        receivers.reserve(num_receivers);
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
            device->mesh_command_queue(0),
            {.num_dram_senders = num_banks,
             .receiver_domain = receiver_domain,
             .ring_size = ring_entries * entry_bytes,
             .max_receivers_per_pipe = num_receivers}));
        for (auto& pipe : metal_exp::CreatePrefetcherPipesForTensorPrefetcher(*space, bank_to_receivers)) {
            pipes.push_back(std::make_shared<metal_exp::PrefetcherPipe>(std::move(pipe)));
        }

        std::mt19937 rng(0xC0FFEE + num_receivers);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<::bfloat16> values(static_cast<size_t>(num_groups) * k * n);
        for (auto& value : values) {
            value = ::bfloat16(dist(rng));
        }
        const MemoryConfig weight_memory_config(
            BufferType::DRAM,
            NdShardSpec{
                num_groups > 1 ? ttnn::Shape({num_groups, k, n / num_receivers}) : ttnn::Shape({k, n / num_receivers}),
                CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(num_banks - 1, 0))),
                ShardOrientation::ROW_MAJOR,
                ShardDistributionStrategy::ROUND_ROBIN_1D});
        const TensorSpec weight_spec(
            ttnn::Shape({1, num_groups, k, n}),
            TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), weight_memory_config));
        weight = Tensor::from_vector(std::move(values), weight_spec).to_device(device);
        if (weight_write == WeightWrite::Finished) {
            tt::tt_metal::distributed::Finish(device->mesh_command_queue());
        }
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

// Queues one request for the weight `num_layers` times over. It names the device's command queue, which is not
// capturing a trace, so the request is sent at once.
void queue_layers(
    tt::tt_metal::distributed::MeshDevice* device, const ReceiverContiguousCase& prefetch, uint32_t num_layers) {
    const std::vector<metal_exp::TensorPrefetcherInput> inputs(
        num_layers, metal_exp::TensorPrefetcherInput{prefetch.weight.mesh_tensor(), prefetch.block_count});
    metal_exp::QueueTensorPrefetcherRequest(
        *device, prefetch.pipe_refs(), std::nullopt, inputs, &device->mesh_command_queue());
}

// Validates every entry of `num_layers` copies of the weight as the receivers consume them: of a grouped
// weight, the `groups` listed, ascending, or every group when none are.
void validate_layers(
    tt::tt_metal::distributed::MeshDevice* device,
    const ReceiverContiguousCase& prefetch,
    uint32_t num_layers,
    const std::vector<uint32_t>& groups = {}) {
    ttnn::operations::experimental::test::test_tensor_prefetcher_pipe_validator(
        device,
        prefetch.weight,
        num_layers,
        /*print_stride=*/0,
        prefetch.pipes,
        prefetch.bank_to_receivers,
        /*streaming=*/false,
        /*rotation=*/{},
        groups);
}

// A group selector's mask on `device`: one UINT16 word per group in a single row-major page, in `memory_config`.
Tensor make_group_mask(
    tt::tt_metal::distributed::MeshDevice* device,
    const std::vector<uint16_t>& words,
    const MemoryConfig& memory_config) {
    const auto num_groups = static_cast<uint32_t>(words.size());
    const TensorSpec spec(
        ttnn::Shape({1, 1, 1, num_groups}),
        TensorLayout(DataType::UINT16, PageConfig(Layout::ROW_MAJOR), memory_config));
    return Tensor::from_vector(std::vector<uint16_t>(words), spec).to_device(device);
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

// Pipes created while the prefetcher runs are fenced against the command queue that writes their receivers'
// config pages, and an explicit fence holds the next request until the queue reaches it. Both fences are
// dispatcher writes into the dispatch engine's own L1 that every kernel waits on.
TEST_F(DispatchEngineTensorPrefetcherFixture, FencesRequestsAgainstCommandQueue) {
    metal_exp::StartTensorPrefetcher(*device_, {});
    ReceiverContiguousCase prefetch(
        device_, /*receivers_per_bank=*/2, /*ring_entries=*/2, /*num_groups=*/1, WeightWrite::InFlight);
    metal_exp::WaitForCqOnTensorPrefetcher(device_->mesh_command_queue(), std::nullopt);
    queue_layers(device_, prefetch, /*num_layers=*/1);
    validate_layers(device_, prefetch, /*num_layers=*/1);
    metal_exp::StopTensorPrefetcher(*device_);
}

// A request queued behind a wait on an op signal is served once an op raises the signal: the consumer enqueued
// after the signalling op gets every block intact only if every kernel held at the wait.
TEST_F(DispatchEngineTensorPrefetcherFixture, ServesRequestAfterOpSignal) {
    constexpr uint32_t kSignalId = 1;
    ReceiverContiguousCase prefetch(
        device_, /*receivers_per_bank=*/2, /*ring_entries=*/2, /*num_groups=*/1, WeightWrite::InFlight);
    metal_exp::StartTensorPrefetcher(*device_, {});
    for (uint32_t layer = 0; layer < 2; ++layer) {
        metal_exp::QueueTensorPrefetcherWaitForSignal(*device_, kSignalId, /*trace_capture_cq=*/nullptr);
        queue_layers(device_, prefetch, /*num_layers=*/1);
        ttnn::operations::experimental::signal_tensor_prefetcher(device_, kSignalId, CoreCoord(0, 0));
        validate_layers(device_, prefetch, /*num_layers=*/1);
    }
    metal_exp::StopTensorPrefetcher(*device_);
}

// A grouped weight stacks its groups (MoE experts) in every receiver slab. Every group arrives, in ascending
// order; with a group selector only those whose mask word is non-zero do. The reader reads the mask when it
// reaches the request -- from DRAM by bank, or from a worker core's L1 -- and the pusher sends what it read.
// A mask that selects no group sends nothing: the request between two others asks for blocks half the size,
// so had it applied its entry size to the pipes or pushed any credit, the request after it would not arrive
// intact.
TEST_F(DispatchEngineTensorPrefetcherFixture, DeliversTheGroupsAMaskSelects) {
    constexpr uint32_t kNumGroups = 4;
    ReceiverContiguousCase prefetch(device_, /*receivers_per_bank=*/2, /*ring_entries=*/2, kNumGroups);
    const Tensor dram_mask =
        make_group_mask(device_, {1, 0, 1, 1}, MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM));
    const Tensor empty_mask =
        make_group_mask(device_, {0, 0, 0, 0}, MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM));
    const CoreRangeSet mask_core(CoreRange(CoreCoord(0, 1)));
    const Tensor l1_mask = make_group_mask(
        device_,
        {0, 1, 0, 0},
        MemoryConfig(TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1, ShardSpec(mask_core, {1, kNumGroups})));
    metal_exp::StartTensorPrefetcher(*device_, {});
    // The masks were written through the command queue, which the prefetcher does not follow on its own.
    metal_exp::WaitForCqOnTensorPrefetcher(device_->mesh_command_queue(), std::nullopt);

    queue_layers(device_, prefetch, /*num_layers=*/1);
    validate_layers(device_, prefetch, /*num_layers=*/1);
    // The groups each request delivers; none for the empty mask, which has nothing to validate (the validator
    // reads an empty list as every group).
    struct Selection {
        const Tensor* mask;
        uint32_t block_count;
        std::optional<std::vector<uint32_t>> groups;
    };
    const std::vector<Selection> selections = {
        {&dram_mask, prefetch.block_count, std::vector<uint32_t>{0, 2, 3}},
        {&empty_mask, prefetch.block_count * 2, std::nullopt},
        {&l1_mask, prefetch.block_count, std::vector<uint32_t>{1}}};
    for (const auto& [mask, block_count, groups] : selections) {
        const std::vector<metal_exp::TensorPrefetcherInput> inputs = {
            {.tensor = prefetch.weight.mesh_tensor(),
             .block_count = block_count,
             .group_selector = metal_exp::TensorPrefetcherGroupSelector{mask->mesh_tensor()}}};
        metal_exp::QueueTensorPrefetcherRequest(
            *device_, prefetch.pipe_refs(), std::nullopt, inputs, &device_->mesh_command_queue());
        if (groups.has_value()) {
            validate_layers(device_, prefetch, /*num_layers=*/1, *groups);
        }
    }
    metal_exp::StopTensorPrefetcher(*device_);
}

// A request queued while the command queue captures a trace is sent on every replay of the trace.
TEST_F(DispatchEngineTensorPrefetcherFixture, SendsTracedRequestOnEveryReplay) {
    ReceiverContiguousCase prefetch(device_, /*receivers_per_bank=*/2, /*ring_entries=*/2);
    metal_exp::StartTensorPrefetcher(*device_, {});
    auto& cq = device_->mesh_command_queue();
    auto validator = ttnn::operations::experimental::test::make_tensor_prefetcher_pipe_validator_workload(
        device_, prefetch.weight, /*num_layers=*/1, /*print_stride=*/0, prefetch.pipes, prefetch.bank_to_receivers);
    // A trace cannot load the validator's binaries, so run it once first, on a request sent at once.
    queue_layers(device_, prefetch, /*num_layers=*/1);
    tt::tt_metal::distributed::EnqueueMeshWorkload(cq, validator, /*blocking=*/true);
    const auto trace_id = device_->begin_mesh_trace(cq);
    queue_layers(device_, prefetch, /*num_layers=*/1);
    tt::tt_metal::distributed::EnqueueMeshWorkload(cq, validator, /*blocking=*/false);
    device_->end_mesh_trace(cq, trace_id);
    for (uint32_t i = 0; i < kReplayCount; ++i) {
        device_->replay_mesh_trace(cq, trace_id, /*blocking=*/true);
    }
    device_->release_mesh_trace(trace_id);
    metal_exp::StopTensorPrefetcher(*device_);
}

// A request recorded into a CommandList is sent on every replay of the list.
TEST_F(DispatchEngineTensorPrefetcherFixture, SendsRecordedRequestOnEveryCommandListReplay) {
    ReceiverContiguousCase prefetch(device_, /*receivers_per_bank=*/2, /*ring_entries=*/2);
    metal_exp::StartTensorPrefetcher(*device_, {});
    {
        const std::vector<metal_exp::TensorPrefetcherInput> inputs = {
            {prefetch.weight.mesh_tensor(), prefetch.block_count}};
        metal_exp::CommandListBuilder builder(*device_);
        builder.add_tensor_prefetcher_request(prefetch.pipe_refs(), std::nullopt, inputs);
        auto validator = ttnn::operations::experimental::test::make_tensor_prefetcher_pipe_validator_workload(
            device_, prefetch.weight, /*num_layers=*/1, /*print_stride=*/0, prefetch.pipes, prefetch.bank_to_receivers);
        builder.add(device_->mesh_command_queue(), validator);
        auto command_list = builder.build(device_->mesh_command_queue());
        builder.deallocate();
        for (uint32_t i = 0; i < kReplayCount; ++i) {
            metal_exp::EnqueueCommandList(device_->mesh_command_queue(), command_list, /*blocking=*/true);
        }
        command_list.deallocate();
    }
    metal_exp::StopTensorPrefetcher(*device_);
}

}  // namespace
}  // namespace ttnn::test::tensor_prefetcher_dispatch_engine
