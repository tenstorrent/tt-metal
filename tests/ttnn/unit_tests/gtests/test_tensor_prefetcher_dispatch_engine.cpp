// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The Tensor prefetcher where fast dispatch runs it on a dispatch engine: the same Start / Queue / Stop and
// the same PrefetcherPipes as on a DRAM-core prefetcher, with a reader and a pusher on the dispatch engine
// streaming a receiver-contiguous weight to the PrefetcherPipe validator, whether the request is queued
// directly, behind a CQ fence or an op signal, or replayed from a trace or a CommandList, and the groups a
// mask selects of a grouped weight. The validator checks
// every entry against the weight in DRAM and hangs on a mismatch, so a run that finishes delivered every block
// intact. MeasuresDeliveryCycles times delivery instead, with receivers that drop each block unread.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/command_list.hpp>
#include <tt-metalium/experimental/prefetcher_pipe.hpp>
#include <tt-metalium/experimental/tensor_prefetcher.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/workload/run_args/program_run_args.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>

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

// The blocks of a receiver slab: each `k_block_tiles` tiles of K by `tiles_per_receiver` tiles of N, and
// `blocks_per_receiver` of them per slab. 0 blocks means one block per receiver, the slab the validator expects.
struct BlockShape {
    uint32_t k_block_tiles = kKBlockTiles;
    uint32_t tiles_per_receiver = kTilesPerReceiver;
    uint32_t blocks_per_receiver = 0;

    uint32_t block_bytes() const { return k_block_tiles * tiles_per_receiver * kBfloat16TileBytes; }
};

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
        WeightWrite weight_write = WeightWrite::Finished,
        const BlockShape& shape = {}) {
        const uint32_t num_banks = device->dram_grid_size().x;
        const uint32_t num_receivers = receivers_per_bank * num_banks;
        const CoreCoord grid = device->compute_with_storage_grid_size();
        TT_FATAL(grid.x * grid.y >= num_receivers, "The test needs {} worker receivers", num_receivers);
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

        block_count = shape.blocks_per_receiver != 0 ? shape.blocks_per_receiver : num_receivers;
        const uint32_t k = block_count * shape.k_block_tiles * TILE_HEIGHT;
        const uint32_t n = num_receivers * shape.tiles_per_receiver * TILE_WIDTH;
        block_bytes = shape.block_bytes();
        space = std::make_unique<metal_exp::PrefetcherPipeSpace>(metal_exp::CreatePrefetcherPipeSpace(
            device->mesh_command_queue(0),
            {.num_dram_senders = num_banks,
             .receiver_domain = receiver_domain,
             .ring_size = ring_entries * block_bytes,
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
    uint32_t block_bytes = 0;
    std::vector<CoreCoord> receivers;
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
    // The mask's L1 copy is on the last core of the compute grid, which every device has.
    const CoreCoord grid = device_->compute_with_storage_grid_size();
    const CoreRangeSet mask_core(CoreRange(CoreCoord(grid.x - 1, grid.y - 1)));
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

// The data-movement config the timing kernels below run with: RISCV_0 on NOC_0.
tt::tt_metal::DataMovementHardwareConfig riscv0_noc0_config() {
    using namespace tt::tt_metal;
    return DataMovementHardwareConfig{
        .config_1xx =
            DataMovementHardwareConfig::DataMovement1XXConfig{
                .processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0},
        .config_2xx = DataMovementHardwareConfig::DataMovement2XXConfig{}};
}

// A workload that runs `program` on every device of `device`.
tt::tt_metal::distributed::MeshWorkload single_program_workload(
    tt::tt_metal::distributed::MeshDevice* device, tt::tt_metal::Program&& program) {
    tt::tt_metal::distributed::MeshWorkload workload;
    workload.add_program(tt::tt_metal::distributed::MeshCoordinateRange(device->shape()), std::move(program));
    return workload;
}

// A receiver workload that times delivery (prefetcher_pipe_arrival_timer.cpp) on every receiver of `prefetch`:
// receiver `prefetch.receivers[r]` writes the cycle at which each of its `num_entries` entries arrived to row `r`
// of `arrivals`, a row-major UINT32 tensor in DRAM.
tt::tt_metal::distributed::MeshWorkload make_arrival_timer_workload(
    tt::tt_metal::distributed::MeshDevice* device,
    const ReceiverContiguousCase& prefetch,
    uint32_t num_entries,
    uint32_t pop_batch,
    const Tensor& arrivals) {
    using namespace tt::tt_metal;
    std::vector<metal_exp::PrefetcherPipeParamName> names;
    std::vector<metal_exp::PrefetcherPipeParameter> parameters;
    ProgramRunArgs run_args;
    for (size_t i = 0; i < prefetch.pipes.size(); ++i) {
        metal_exp::PrefetcherPipeParamName name{fmt::format("pipe_{}", i)};
        names.push_back(name);
        parameters.push_back(
            {.unique_id = name,
             .receivers = prefetch.pipes[i]->receiver_cores(),
             .ring_size = prefetch.pipes[i]->ring_size(),
             .entry_size = prefetch.block_bytes});
        run_args.advanced_options.prefetcher_pipe_args.emplace(
            name, metal_exp::PrefetcherPipeArgument{*prefetch.pipes[i]});
    }
    const TensorParamName arrivals_name{"arrivals"};
    const ScratchpadSpecName stamps_name{"stamps"};
    KernelSpec timer{
        .unique_id = KernelSpecName{"timer"},
        .source = std::filesystem::path{"tests/tt_metal/tt_metal/test_kernels/misc/prefetcher_pipe_arrival_timer.cpp"}};
    timer.tensor_bindings = {{.tensor_parameter_name = arrivals_name, .accessor_name = "arrivals"}};
    timer.scratchpad_bindings = {{.scratchpad_spec_name = stamps_name, .accessor_name = "stamps"}};
    timer.advanced_options.prefetcher_pipe_bindings = {{.pipe_parameter_names = names, .accessor_name = "in"}};
    timer.compile_time_args = {{"num_entries", num_entries}, {"pop_batch", pop_batch}};
    timer.runtime_arg_schema.runtime_arg_names = {"out_page"};
    timer.hw_config = riscv0_noc0_config();
    ProgramSpec spec{
        .name = "prefetcher_pipe_arrival_timer",
        .kernels = {std::move(timer)},
        .scratchpads =
            {{.unique_id = stamps_name,
              .size_per_node = static_cast<uint32_t>(arrivals.tensor_spec().compute_page_size_bytes())}},
        .tensor_parameters = {{.unique_id = arrivals_name, .spec = arrivals.tensor_spec()}},
        .work_units =
            {{.name = "timer_wu",
              .kernels = {KernelSpecName{"timer"}},
              .target_nodes = metal_exp::GetPrefetcherPipeReceiverCores(prefetch.pipe_refs())}},
        .advanced_options = {.prefetcher_pipe_parameters = std::move(parameters)}};
    Program program(*device, spec);

    ProgramRunArgs::KernelRunArgs kernel_args{.kernel = KernelSpecName{"timer"}};
    for (uint32_t r = 0; r < prefetch.receivers.size(); ++r) {
        AddRuntimeArgsForNode(kernel_args.runtime_arg_values, prefetch.receivers[r], {{"out_page", r}});
    }
    run_args.kernel_run_args.push_back(std::move(kernel_args));
    run_args.tensor_args.emplace(arrivals_name, TensorArgument{arrivals.mesh_tensor()});
    program.set_program_run_args(run_args);
    return single_program_workload(device, std::move(program));
}

// The fewest batches steady_cycles_per_entry measures: it needs at least one whole batch after the middle one.
constexpr uint32_t kMinTimedBatches = 3;

// Mean cycles per entry at one receiver, over the second half of its batches: `stamps` holds the cycle at which each
// batch of `pop_batch` entries (the last one possibly short) had all arrived, for `num_entries` entries, in at least
// kMinTimedBatches batches. The first batches can arrive back to back, out of the pipe ring and stage ring the
// prefetcher filled before the receiver started.
double steady_cycles_per_entry(const uint32_t* stamps, uint32_t num_entries, uint32_t pop_batch) {
    const uint32_t num_batches = tt::div_up(num_entries, pop_batch);
    const uint32_t first = num_batches / 2;
    const uint32_t last = num_batches - 1;
    // The entries of batches first + 1 through last.
    const uint32_t entries = num_entries - (first + 1) * pop_batch;
    // Unsigned, so a stamp that wrapped still subtracts to the elapsed cycles.
    return static_cast<double>(stamps[last] - stamps[first]) / entries;
}

// One case of MeasuresDeliveryCycles: the blocks each receiver gets, how many of them its pipe ring holds, how many
// receivers each DRAM bank serves, and what fraction of its ring a receiver frees at a time (1 / pop_divisor).
struct DeliveryCase {
    BlockShape shape;
    uint32_t ring_entries = 0;
    uint32_t receivers_per_bank = 1;
    uint32_t pop_divisor = 2;
};

// Each case opens its own device, so no case inherits another's pipes or dispatch-engine state.
class DispatchEngineTensorPrefetcherDeliveryFixture : public DispatchEngineTensorPrefetcherFixture,
                                                      public testing::WithParamInterface<DeliveryCase> {};

// Each tensor is at least kDeliveryMinTensorBytes and kDeliveryMinBlocks blocks per receiver, so a tensor's own setup
// does not set the pace of small blocks.
constexpr uint32_t kDeliveryMinBlocks = 8;
constexpr uint32_t kDeliveryMinTensorBytes = 256 * 1024;
constexpr uint32_t kDeliveryLayers = 8;

// Times delivery from the dispatch-engine Tensor prefetcher to its receivers. Each receiver stamps every block as it
// arrives and drops it unread, so the pace it sees is the prefetcher's alone. A stamp is the receiver's own cycle
// count, so the result is cycles per block, and a simulator's are not silicon times. Logs one line.
TEST_P(DispatchEngineTensorPrefetcherDeliveryFixture, MeasuresDeliveryCycles) {
    const auto& [shape, ring_entries, receivers_per_bank, pop_divisor] = GetParam();
    const uint32_t num_entries = shape.blocks_per_receiver * kDeliveryLayers;
    // A receiver frees a fixed part of its ring at a time, as a consumer that takes several blocks at once does, and
    // stamps each batch as it arrives, so its own per-block cost does not set the pace.
    const uint32_t pop_batch = std::max(1u, ring_entries / pop_divisor);
    const uint32_t num_batches = tt::div_up(num_entries, pop_batch);
    ASSERT_GE(num_batches, kMinTimedBatches) << "The pace is timed over the batches after the middle one";
    // A receiver's stamps fill one row-major page of the output, in whole 64 B lines.
    constexpr uint32_t kWordsPerLine = 64 / sizeof(uint32_t);
    const uint32_t row_words = tt::round_up(num_batches, kWordsPerLine);
    const CoreCoord grid = device_->compute_with_storage_grid_size();
    if (grid.x * grid.y < receivers_per_bank * device_->dram_grid_size().x) {
        GTEST_SKIP() << "Needs " << receivers_per_bank << " worker receivers per DRAM bank";
    }
    ReceiverContiguousCase prefetch(
        device_, receivers_per_bank, ring_entries, /*num_groups=*/1, WeightWrite::Finished, shape);
    const auto num_receivers = static_cast<uint32_t>(prefetch.receivers.size());
    const TensorSpec arrivals_spec(
        ttnn::Shape({num_receivers, row_words}),
        TensorLayout(
            DataType::UINT32,
            PageConfig(Layout::ROW_MAJOR),
            MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM)));
    const Tensor arrivals =
        Tensor::from_vector(std::vector<uint32_t>(num_receivers * row_words, 0), arrivals_spec).to_device(device_);
    auto timer = make_arrival_timer_workload(device_, prefetch, num_entries, pop_batch, arrivals);

    metal_exp::StartTensorPrefetcher(*device_, {});
    // The receivers go first, so that blocks reach receivers that are already waiting for them.
    tt::tt_metal::distributed::EnqueueMeshWorkload(device_->mesh_command_queue(), timer, /*blocking=*/false);
    queue_layers(device_, prefetch, kDeliveryLayers);
    tt::tt_metal::distributed::Finish(device_->mesh_command_queue());
    metal_exp::StopTensorPrefetcher(*device_);

    const std::vector<uint32_t> stamps = arrivals.cpu().to_vector<uint32_t>();
    std::vector<double> cycles_per_block;
    for (uint32_t r = 0; r < num_receivers; ++r) {
        const uint32_t* row = stamps.data() + static_cast<size_t>(r) * row_words;
        for (uint32_t b = 1; b < num_batches; ++b) {
            ASSERT_NE(row[b], row[b - 1])
                << "receiver " << r << " recorded no time between batches " << b - 1 << " and " << b;
        }
        cycles_per_block.push_back(steady_cycles_per_entry(row, num_entries, pop_batch));
    }
    const auto [fewest, most] = std::minmax_element(cycles_per_block.begin(), cycles_per_block.end());
    // Each receiver's pace comes from its own clock over its own stamps, so the total counts every receiver at the
    // slowest one's pace: summing their paces would overstate it when one receiver finishes ahead of another.
    const double total_bytes_per_cycle = num_receivers * prefetch.block_bytes / *most;
    log_info(
        tt::LogTest,
        "{}: dispatch-engine Tensor prefetcher, {} B blocks, {}-entry rings, {} receivers: {:.0f} to {:.0f} cycles "
        "per block per receiver, {:.1f} B/cycle in total at the slowest receiver's pace",
        testing::UnitTest::GetInstance()->current_test_info()->name(),
        prefetch.block_bytes,
        ring_entries,
        num_receivers,
        *fewest,
        *most,
        total_bytes_per_cycle);
}

DeliveryCase delivery_case(
    uint32_t k_block_tiles,
    uint32_t tiles_per_receiver,
    uint32_t ring_entries,
    uint32_t receivers_per_bank,
    uint32_t pop_divisor = 2) {
    return {
        .shape =
            {.k_block_tiles = k_block_tiles,
             .tiles_per_receiver = tiles_per_receiver,
             .blocks_per_receiver = std::max(
                 kDeliveryMinBlocks,
                 kDeliveryMinTensorBytes / BlockShape{k_block_tiles, tiles_per_receiver}.block_bytes())},
        .ring_entries = ring_entries,
        .receivers_per_bank = receivers_per_bank,
        .pop_divisor = pop_divisor};
}

INSTANTIATE_TEST_SUITE_P(
    BlockSizesAndRingDepths,
    DispatchEngineTensorPrefetcherDeliveryFixture,
    testing::Values(
        // Two receivers per bank, both workers of a 2x3 grid, with 256 KiB rings.
        delivery_case(1, 1, 128, 2),
        delivery_case(2, 2, 32, 2),
        delivery_case(4, 4, 8, 2),
        delivery_case(4, 8, 4, 2),
        delivery_case(8, 8, 2, 2),
        // One receiver.
        delivery_case(4, 8, 4, 1),
        // 512 KiB rings that receivers free a quarter at a time, so more of each ring is in flight.
        delivery_case(1, 1, 256, 2, 4),
        delivery_case(2, 2, 64, 2, 4),
        delivery_case(4, 4, 16, 2, 4),
        delivery_case(4, 8, 8, 2, 4)),
    [](const testing::TestParamInfo<DeliveryCase>& info) {
        const BlockShape& shape = info.param.shape;
        return fmt::format(
            "{}KiBBlocks_{}EntryRing_{}PerBank{}",
            shape.block_bytes() / 1024,
            info.param.ring_entries,
            info.param.receivers_per_bank,
            info.param.pop_divisor == 2 ? "" : fmt::format("_PopsOneIn{}", info.param.pop_divisor));
    });

// What one worker moves over the NoC on its own, the bound for any one tile's sender, the dispatch engine's
// included. Worker (0, 0) writes 4 MiB to worker (1, 0), or reads 4 MiB from DRAM bank 0, with every transfer issued
// before any is waited on, from one or several of its data-movement cores. Logs one line per case.
TEST_F(DispatchEngineTensorPrefetcherFixture, MeasuresWorkerNocBandwidth) {
    using namespace tt::tt_metal;
    constexpr uint32_t kTotalBytes = 4u << 20;
    constexpr uint32_t kScratchBytes = 512u << 10;
    constexpr uint32_t kDramBytes = 256u << 10;
    const TensorSpec dram_spec(
        ttnn::Shape({1, kDramBytes / sizeof(uint32_t)}),
        TensorLayout(
            DataType::UINT32,
            PageConfig(Layout::ROW_MAJOR),
            MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM)));
    const Tensor dram =
        Tensor::from_vector(std::vector<uint32_t>(kDramBytes / sizeof(uint32_t), 0), dram_spec).to_device(device_);
    tt::tt_metal::distributed::Finish(device_->mesh_command_queue());
    const uint32_t dram_addr = dram.mesh_buffer().address();
    const CoreCoord peer = device_->worker_core_from_logical_core(CoreCoord(1, 0));
    struct ProbeCase {
        uint32_t mode = 0;
        uint32_t tx_bytes = 0;
        uint32_t num_vc = 0;
        uint32_t threads = 0;
    };
    for (const auto& [mode, tx_bytes, num_vc, threads] :
         {ProbeCase{0, 8192, 1, 1},
          ProbeCase{0, 8192, 1, 4},
          ProbeCase{0, 8192, 4, 4},
          ProbeCase{1, 65536, 1, 1},
          ProbeCase{1, 65536, 1, 4}}) {
        const ScratchpadSpecName buf_name{"buf"};
        KernelSpec probe{
            .unique_id = KernelSpecName{"probe"},
            .source = std::filesystem::path{"tests/tt_metal/tt_metal/test_kernels/misc/noc_bandwidth_probe.cpp"},
            .num_threads = threads};
        probe.scratchpad_bindings = {{.scratchpad_spec_name = buf_name, .accessor_name = "buf"}};
        probe.compile_time_args = {
            {"total_bytes", kTotalBytes}, {"tx_bytes", tx_bytes}, {"mode", mode}, {"num_vc", num_vc}};
        probe.runtime_arg_schema.runtime_arg_names = {"role", "peer_x", "peer_y", "dram_addr", "dram_bytes"};
        probe.hw_config = riscv0_noc0_config();
        ProgramSpec spec{
            .name = "noc_bandwidth_probe",
            .kernels = {std::move(probe)},
            .scratchpads = {{.unique_id = buf_name, .size_per_node = kScratchBytes}},
            .work_units = {
                {.name = "probe_wu",
                 .kernels = {KernelSpecName{"probe"}},
                 .target_nodes = CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(1, 0)))}}};
        Program program(*device_, spec);
        ProgramRunArgs run_args;
        ProgramRunArgs::KernelRunArgs kernel_args{.kernel = KernelSpecName{"probe"}};
        for (uint32_t x = 0; x < 2; ++x) {
            AddRuntimeArgsForNode(
                kernel_args.runtime_arg_values,
                CoreCoord(x, 0),
                {{"role", x},
                 {"peer_x", static_cast<uint32_t>(peer.x)},
                 {"peer_y", static_cast<uint32_t>(peer.y)},
                 {"dram_addr", dram_addr},
                 {"dram_bytes", kDramBytes}});
        }
        run_args.kernel_run_args.push_back(std::move(kernel_args));
        program.set_program_run_args(run_args);
        auto workload = single_program_workload(device_, std::move(program));
        tt::tt_metal::distributed::EnqueueMeshWorkload(device_->mesh_command_queue(), workload, /*blocking=*/true);
        const std::vector<uint32_t> counts = dram.cpu().to_vector<uint32_t>();
        // Each DM core reports in its own 64 B.
        constexpr uint32_t kReportWords = 64 / sizeof(uint32_t);
        uint32_t slowest = 0;
        for (uint32_t t = 0; t < threads; ++t) {
            slowest = std::max(slowest, counts[t * kReportWords + 1]);
        }
        log_info(
            tt::LogTest,
            "Worker {} {} B per transfer, {} VCs, {} DM cores: slowest core {} cycles = {:.1f} B/cycle in total "
            "(core 0: issue {}, {} {}, acked {})",
            mode == 0 ? "NoC write to a worker," : "DRAM read,",
            tx_bytes,
            num_vc,
            threads,
            slowest,
            static_cast<double>(kTotalBytes) / slowest,
            counts[0],
            mode == 0 ? "left the core" : "landed",
            counts[1],
            counts[2]);
    }
}

}  // namespace
}  // namespace ttnn::test::tensor_prefetcher_dispatch_engine
