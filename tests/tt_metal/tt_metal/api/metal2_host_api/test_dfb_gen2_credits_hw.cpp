// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Gen2 (Quasar) DataflowBuffer ordering tests: a data-movement producer/consumer pair using the
// explicit DFB credit APIs, with no compute kernel and nothing from ttnn.
//
// The finding: on Quasar, a DataflowBuffer breaks when two things hold at once. The producer issues
// MORE NoC transactions than the slots it announces with push_back, AND the consumer is a
// data-movement kernel. Then the consumer is released before the producer filled the slot and the
// delivered data is wrong. Remove either condition and the buffer behaves. A deficit of transactions
// is harmless, and a COMPUTE consumer of the same buffer, fed by the same producer with the same
// surplus, is unaffected even at a surplus large enough to deadlock the data-movement case.
//
// That second half matters for scope: a compute consumer waits through the unpacker, while a
// data-movement consumer spins on the buffer's occupancy from a RISC. Only the second path is
// affected, which is why programs whose buffers all feed compute kernels never see this.
//
// Everything here passes on Wormhole hardware, so the contract, not the test, is what differs
// between the two generations.
//
// The contract under test: wait_front(n) must not return until the matching push_back(n). Two
// independent checks, because the weaker one under-reports. Per grant, the consumer samples the
// granted slot's first word the instant wait_front returns; slots carry distinct values, so grant k
// must see entry k's word. End to end, the delivered DRAM output must equal the input. A slot released
// early only corrupts the output if the consumer also loses the race against the arriving data.
//
// The two tests that matter most, because neither involves a scratchpad or a second destination and
// the only thing varied is the transaction-to-announcement ratio:
//
//   RatioTwoReadsPerAnnouncedSlot
//       Fills each slot with two half-entry reads, both into the buffer, then announces it once.
//       FAILS on Quasar. This is the headline result and the smallest repro.
//   RatioOneReadPerTwoAnnouncedSlotsCompletes
//       One double-entry read, then announces two slots. Passes on both, which is what makes the
//       defect asymmetric: a surplus of transactions breaks it, a deficit does not.
//
// The tests that narrow it to a data-movement consumer, same producer and same surplus in each pair:
//
//   ComputeConsumerTwoReadsPerAnnouncedSlot
//       The failing 2:1 ratio with a compute kernel draining the buffer. PASSES on Quasar.
//   ComputeConsumerManyReadsPerAnnouncedSlot
//       Eight reads per announced slot, a surplus of seven against a two-slot ring. Still PASSES on
//       Quasar, which rules out the compute kernel merely winning a race it could have lost.
//   DmConsumerManyReadsPerAnnouncedSlot
//       The same eight-reads load with a data-movement consumer. HANGS ON QUASAR, so the pair differs
//       only in which hardware waits.
//   ComputeConsumerOneReadPerAnnouncedSlot
//       Control for the three-kernel pipeline the compute cases need, so their passes cannot be
//       explained by the tile copy, the data format, or the extra buffer.
//
// Supporting tests, establishing that the destination is irrelevant:
//
//   ScratchpadUsePatternsThatDisturbTheBuffer
//       Eight rows varying where an extra NoC read lands: into a scratchpad by its binding, at the
//       scratchpad's own address through a plain pointer, at a scratchpad's far end, at an unrelated
//       address with a scratchpad bound, and at an unrelated address with none. All fail identically,
//       and only the row that touches memory with ordinary load and store instructions passes,
//       because that is not a NoC transaction.
//   MinimalExtraReadNoScratchpad
//       The same thing with no scratchpad anywhere in the program. Fails.
//   ScratchpadReadEveryEntryCompletes
//       One extra read per entry, so the surplus exceeds the ring depth. HANGS ON QUASAR rather than
//       corrupting, which is why it is excluded from the normal run. gtest has no per-test timeout.
//
// Controls, all passing on both generations, so that the ratio is left as the only variable that
// matters: buffer depth and entry size (ConsumerDoesNotRunAheadOfProducer), how the producer addresses
// the buffer (...RawWritePtr), core count (...TwoCores), how DRAM addresses are computed
// (...TensorAccessor), Gen2 implicit sync switched off (...NoImplicitSync, and
// RatioTwoReadsPerAnnouncedSlotNoImplicitSync, which shows the opt-out does not help), and a check that in the baseline
// the slots genuinely are filled when a grant is taken (ConsumerObservesProducerPushCountAtEachGrant).
//
// Run everything except the two hangs (14 tests):
//   ./build_Release/test/tt_metal/unit_tests_api \
//     --gtest_filter="Gen2DFBCreditsTest.*:-Gen2DFBCreditsTest.ScratchpadReadEveryEntry*:Gen2DFBCreditsTest.DmConsumerManyReads*"
// Expected: Quasar 10 pass and 4 fail (MinimalExtraReadNoScratchpad, RatioTwoReadsPerAnnouncedSlot,
// RatioTwoReadsPerAnnouncedSlotNoImplicitSync, ScratchpadUsePatternsThatDisturbTheBuffer); Wormhole
// all 14 pass.
//
// Note the single '-' in that filter. gtest treats everything after the FIRST '-' as the exclusion
// list, colon-separated, so a second '-' produces a pattern that matches nothing and the test it was
// meant to exclude runs anyway. Written with two dashes, the filter above selects 15 tests instead of
// 14 and hangs on Quasar. Check with --gtest_list_tests before trusting a filter.
//
// Run each hanging case on its own, under an external timeout:
//   timeout 120 ./build_Release/test/tt_metal/unit_tests_api \
//     --gtest_filter="Gen2DFBCreditsTest.ScratchpadReadEveryEntry*"
//   timeout 120 ./build_Release/test/tt_metal/unit_tests_api \
//     --gtest_filter="Gen2DFBCreditsTest.DmConsumerManyReads*"
// Expected: Wormhole passes both in well under a second; Quasar does not terminate on either.
//
// Requires TT_METAL_SLOW_DISPATCH_MODE=1, plus the Quasar simulator variables for a Quasar run.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include <tt-metalium/tensor/mesh_tensor.hpp>

#include "device_fixture.hpp"
#include "test_helpers.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::BindTensorParameterToKernel;
using test_helpers::MakeMinimalDFB;
using test_helpers::MakeMinimalWorkUnit;

class Gen2DFBCreditsTest : public tt::tt_metal::MeshDeviceFixture {
protected:
    void SetUp() override {
        MeshDeviceFixture::SetUp();
        if (this->IsSkipped()) {
            return;
        }
        // Runs on Gen1 as well as Gen2 on purpose: the DataflowBuffer contract these tests assert is
        // architecture-independent, so a Gen1 pass beside a Gen2 failure is the useful result.
        const auto arch = devices_.at(0)->arch();
        if (arch != tt::ARCH::QUASAR && arch != tt::ARCH::WORMHOLE_B0 && arch != tt::ARCH::BLACKHOLE) {
            GTEST_SKIP() << "Skipping: needs Quasar, Wormhole B0 or Blackhole";
        }
    }
};

// Data movement hardware config for whichever generation is under test. Gen2 assigns the RISC and
// NOC automatically and exposes only the implicit-sync opt-out; Gen1 requires both to be named, and
// the two DM kernels on a node must sit on different RISCs.
DataMovementHardwareConfig dm_hw_config(
    bool gen2, tt::tt_metal::DataMovementProcessor proc, bool disable_implicit_sync = false) {
    if (gen2) {
        return DataMovementGen2Config{.disable_dfb_implicit_sync_for_all = disable_implicit_sync};
    }
    return DataMovementGen1Config{
        .processor = proc, .noc = (proc == tt::tt_metal::DataMovementProcessor::RISCV_0) ? NOC::NOC_0 : NOC::NOC_1};
}

// The compute counterpart of dm_hw_config: the same test body has to build a compute kernel on either
// generation, and the two config types are unrelated.
ComputeHardwareConfig compute_hw_config(bool gen2) {
    if (gen2) {
        return ComputeGen2Config{};
    }
    return ComputeGen1Config{};
}

// Returns the number of leading output entries that do not match the input, which for the defect
// under test equals the shift the whole stream has taken. Returns 0 when the round trip is exact.
uint32_t leading_mismatch_entries(
    const std::vector<uint32_t>& input, const std::vector<uint32_t>& output, uint32_t words_per_entry) {
    const uint32_t num_entries = static_cast<uint32_t>(input.size()) / words_per_entry;
    for (uint32_t e = 0; e < num_entries; e++) {
        bool entry_matches = true;
        for (uint32_t w = 0; w < words_per_entry; w++) {
            if (input[e * words_per_entry + w] != output[e * words_per_entry + w]) {
                entry_matches = false;
                break;
            }
        }
        if (entry_matches) {
            return e;
        }
    }
    return num_entries;
}

// Builds and runs the DM to DFB to DM loopback, and returns the output buffer's contents.
//
// `dfb_depth` is the DFB's num_entries; with one producer and one consumer the DFB's capacity
// equals it. `disable_implicit_sync` sets Gen2Config::disable_dfb_implicit_sync_for_all on both
// kernels, which is the only field that differs between the two tests.
std::vector<uint32_t> run_dm_to_dm_loopback(
    distributed::MeshDevice& mesh_device,
    IDevice* device,
    uint32_t dfb_depth,
    bool disable_implicit_sync,
    const std::vector<uint32_t>& input_data,
    uint32_t entry_size,
    const char* producer_source = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_accessor_loopback_producer.cpp",
    bool bind_producer_scratchpad = false) {
    const uint32_t num_transfers = static_cast<uint32_t>(input_data.size() * sizeof(uint32_t)) / entry_size;
    const uint32_t total_bytes = num_transfers * entry_size;
    const NodeCoord node{0, 0};

    InterleavedBufferConfig dram_config{
        .device = device, .size = total_bytes, .page_size = total_bytes, .buffer_type = BufferType::DRAM};
    auto input_buffer = CreateBuffer(dram_config);
    auto output_buffer = CreateBuffer(dram_config);

    ProgramSpec spec;
    spec.name = "gen2_dfb_credits_loopback";

    // Gen2 selects the DM kernel's RISC automatically, so unlike the Gen1 loopback there is no
    // processor or NOC to assign; the only hardware knob is the implicit-sync opt-out.
    const bool gen2 = device->arch() == tt::ARCH::QUASAR;
    auto make_dm_kernel = [&](const char* name, const char* source, DataMovementProcessor proc) {
        KernelSpec k{
            .unique_id = KernelSpecName{name},
            .source = source,
            .num_threads = 1,
        };
        if (gen2) {
            k.hw_config = DataMovementGen2Config{.disable_dfb_implicit_sync_for_all = disable_implicit_sync};
        } else {
            k.hw_config = DataMovementGen1Config{
                .processor = proc, .noc = (proc == DataMovementProcessor::RISCV_0) ? NOC::NOC_0 : NOC::NOC_1};
        }
        k.advanced_options.num_runtime_varargs = 3;
        return k;
    };

    auto producer = make_dm_kernel("producer", producer_source, DataMovementProcessor::RISCV_0);
    auto consumer = make_dm_kernel(
        "consumer",
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_accessor_loopback_consumer.cpp",
        DataMovementProcessor::RISCV_1);

    auto dfb = MakeMinimalDFB("loopback_dfb", entry_size, dfb_depth);
    dfb.data_format_metadata = tt::DataFormat::Float16_b;
    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"loopback_dfb"}, "my_local_dfb_name"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"loopback_dfb"}, "a_dfb_named_bob"));

    if (bind_producer_scratchpad) {
        producer.scratchpad_bindings.push_back(
            ScratchpadBinding{.scratchpad_spec_name = ScratchpadSpecName{"staging"}, .accessor_name = "staging"});
        spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"staging"}, .size_per_node = 4096}};
    }

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit_0", node, {"producer", "consumer"})};

    Program program = MakeProgramFromSpec(mesh_device, spec);

    ProgramRunArgs params;
    params.kernel_run_args = {
        ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"producer"},
            .advanced_options =
                AdvancedKernelRunArgs{
                    .runtime_varargs =
                        {{node,
                          {
                              static_cast<uint32_t>(input_buffer->address()),
                              0u,  // bank_id (single-page buffer, so bank 0)
                              num_transfers,
                          }}},
                },
        },
        ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"consumer"},
            .advanced_options =
                AdvancedKernelRunArgs{
                    .runtime_varargs =
                        {{node,
                          {
                              static_cast<uint32_t>(output_buffer->address()),
                              0u,  // bank_id
                              num_transfers,
                          }}},
                },
        },
    };
    SetProgramRunArgs(program, params);

    detail::WriteToBuffer(input_buffer, input_data);

    // Pre-fill the output buffer with a sentinel so a never-written entry is distinguishable from a
    // correctly copied zero.
    std::vector<uint32_t> sentinel(input_data.size(), 0xDEADBEEFu);
    detail::WriteToBuffer(output_buffer, sentinel);

    detail::LaunchProgram(device, program);

    std::vector<uint32_t> output_data;
    detail::ReadFromBuffer(output_buffer, output_data);
    return output_data;
}

std::vector<uint32_t> make_input(uint32_t num_transfers, uint32_t entry_size) {
    std::vector<uint32_t> input(num_transfers * entry_size / sizeof(uint32_t));
    for (size_t i = 0; i < input.size(); i++) {
        // Offset by a constant so entry 0 does not begin with 0, which a never-written slot could
        // coincidentally match.
        input[i] = static_cast<uint32_t>(i) + 0x1000u;
    }
    return input;
}

constexpr uint32_t kEntrySize = 1024;
constexpr uint32_t kNumTransfers = 8;

// The baseline, and the control the rest of the file is measured against: one NoC read per announced
// entry, which is the ordinary shape of a producer. A DM consumer must not be granted an entry before
// the producer pushes one, so the round trip has to be exact whatever the DFB's depth. It is, on both
// generations. Establishing that matters, because it is what makes a surplus of transactions the only
// variable that separates this from the failing tests rather than one variable among several.
TEST_F(Gen2DFBCreditsTest, ConsumerDoesNotRunAheadOfProducer) {
    auto mesh_device = devices_.at(0);
    IDevice* device = mesh_device->get_devices()[0];

    // Entry sizes cover both a power of two and the non-power-of-two an op actually produces: an
    // embedding row of 768 bfloat16 elements is 1536 bytes, and the DFB's transaction-id and stride
    // bookkeeping is derived from entry_size.
    for (uint32_t entry_size : {1024u, 1536u}) {
        const uint32_t words_per_entry = entry_size / sizeof(uint32_t);
        const std::vector<uint32_t> input = make_input(kNumTransfers, entry_size);

        for (uint32_t depth : {2u, 4u}) {
            const std::vector<uint32_t> output =
                run_dm_to_dm_loopback(*mesh_device, device, depth, /*disable_implicit_sync=*/false, input, entry_size);
            ASSERT_EQ(output.size(), input.size()) << "entry_size=" << entry_size << " depth=" << depth;

            const uint32_t shift = leading_mismatch_entries(input, output, words_per_entry);
            EXPECT_EQ(shift, 0u) << "entry_size " << entry_size << ", DFB depth " << depth << ": the consumer copied "
                                 << "out " << shift << " entries before the producer pushed anything, so every entry "
                                 << "landed " << shift << " positions late.";
        }
    }
}

// The same loopback with the producer filling the DFB through a raw `get_write_ptr()` address instead
// of by passing the DataflowBuffer to the NoC.
//
// Ported readers overwhelmingly have this shape, because they compute a per-item address and hand it
// to a helper, so if the two forms are not interchangeable on Gen2 that matters far more widely than
// this test. On Gen1 they are equivalent.
TEST_F(Gen2DFBCreditsTest, ConsumerDoesNotRunAheadOfProducerRawWritePtr) {
    auto mesh_device = devices_.at(0);
    IDevice* device = mesh_device->get_devices()[0];

    for (uint32_t entry_size : {1024u, 1536u}) {
        const uint32_t words_per_entry = entry_size / sizeof(uint32_t);
        const std::vector<uint32_t> input = make_input(kNumTransfers, entry_size);

        for (uint32_t depth : {2u, 4u}) {
            const std::vector<uint32_t> output = run_dm_to_dm_loopback(
                *mesh_device,
                device,
                depth,
                /*disable_implicit_sync=*/false,
                input,
                entry_size,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_raw_write_ptr_producer.cpp");
            ASSERT_EQ(output.size(), input.size()) << "entry_size=" << entry_size << " depth=" << depth;

            const uint32_t shift = leading_mismatch_entries(input, output, words_per_entry);
            EXPECT_EQ(shift, 0u) << "entry_size " << entry_size << ", DFB depth " << depth
                                 << ": filling through a raw write pointer left the stream shifted by " << shift
                                 << " entries, where passing the DataflowBuffer itself does not.";
        }
    }
}

// The same loopback spread over two cores, which is how an op that splits work by row runs it.
//
// Each core gets its own DFB instance and its own tile counter, and per-core counter assignment is
// the part of the Gen2 credit machinery with known remapping hazards, so two cores is not merely
// "the same test twice".
TEST_F(Gen2DFBCreditsTest, ConsumerDoesNotRunAheadOfProducerTwoCores) {
    auto mesh_device = devices_.at(0);
    IDevice* device = mesh_device->get_devices()[0];

    const auto grid = device->compute_with_storage_grid_size();
    if (grid.x * grid.y < 2) {
        GTEST_SKIP() << "Skipping: needs at least 2 cores, grid is " << grid.x << "x" << grid.y;
    }
    // Second core along whichever axis the grid actually spans.
    const NodeCoord node_a{0, 0};
    const NodeCoord node_b = (grid.x >= 2) ? NodeCoord{1, 0} : NodeCoord{0, 1};
    const NodeRange nodes{node_a, node_b};

    constexpr uint32_t kDepth = 2;
    constexpr uint32_t kPerCore = 8;
    const uint32_t words_per_entry = kEntrySize / sizeof(uint32_t);
    // Each core round-trips its own half, so the buffers hold both halves back to back.
    const std::vector<uint32_t> input = make_input(kPerCore * 2, kEntrySize);
    const uint32_t total_bytes = kPerCore * 2 * kEntrySize;

    InterleavedBufferConfig dram_config{
        .device = device, .size = total_bytes, .page_size = total_bytes, .buffer_type = BufferType::DRAM};
    auto input_buffer = CreateBuffer(dram_config);
    auto output_buffer = CreateBuffer(dram_config);

    ProgramSpec spec;
    spec.name = "gen2_dfb_credits_two_cores";

    const bool gen2 = device->arch() == tt::ARCH::QUASAR;
    auto make_dm_kernel = [&](const char* name, const char* source, DataMovementProcessor proc) {
        KernelSpec k{
            .unique_id = KernelSpecName{name},
            .source = source,
            .num_threads = 1,
            .hw_config = dm_hw_config(gen2, proc),
        };
        k.advanced_options.num_runtime_varargs = 3;
        return k;
    };
    auto producer = make_dm_kernel(
        "producer",
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_accessor_loopback_producer.cpp",
        DataMovementProcessor::RISCV_0);
    auto consumer = make_dm_kernel(
        "consumer",
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_accessor_loopback_consumer.cpp",
        DataMovementProcessor::RISCV_1);

    auto dfb = MakeMinimalDFB("loopback_dfb", kEntrySize, kDepth);
    dfb.data_format_metadata = tt::DataFormat::Float16_b;
    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"loopback_dfb"}, "my_local_dfb_name"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"loopback_dfb"}, "a_dfb_named_bob"));

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{WorkUnitSpec{
        .name = "work_unit_0",
        .kernels = {KernelSpecName{"producer"}, KernelSpecName{"consumer"}},
        .target_nodes = nodes,
    }};

    Program program = MakeProgramFromSpec(*mesh_device, spec);

    const uint32_t in_base = static_cast<uint32_t>(input_buffer->address());
    const uint32_t out_base = static_cast<uint32_t>(output_buffer->address());
    const uint32_t half_bytes = kPerCore * kEntrySize;

    ProgramRunArgs params;
    params.kernel_run_args = {
        ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"producer"},
            .advanced_options =
                AdvancedKernelRunArgs{
                    .runtime_varargs =
                        {
                            {node_a, {in_base, 0u, kPerCore}},
                            {node_b, {in_base + half_bytes, 0u, kPerCore}},
                        },
                },
        },
        ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"consumer"},
            .advanced_options =
                AdvancedKernelRunArgs{
                    .runtime_varargs =
                        {
                            {node_a, {out_base, 0u, kPerCore}},
                            {node_b, {out_base + half_bytes, 0u, kPerCore}},
                        },
                },
        },
    };
    SetProgramRunArgs(program, params);

    detail::WriteToBuffer(input_buffer, input);
    detail::LaunchProgram(device, program);

    std::vector<uint32_t> output;
    detail::ReadFromBuffer(output_buffer, output);
    ASSERT_EQ(output.size(), input.size());

    const uint32_t shift = leading_mismatch_entries(input, output, words_per_entry);
    EXPECT_EQ(shift, 0u) << "Two cores, DFB depth " << kDepth << ": the stream came back shifted by " << shift
                         << " entries, where the single-core form of the same program does not.";
}

// The same producer / DFB / consumer pipeline, but reaching DRAM through TensorAccessor bindings with
// a per-entry page id, which is how a real op addresses its tensors.
//
// This is the last structural difference between this harness and the embedding op's reader/writer
// pair: every other variant here uses AllocatorBank against a single bank, whereas an op issues one
// transfer per page. Gen1 covers this shape already in
// ProgramSpecHWTest.TensorAccessorBindingLoopback; this is its Gen2 counterpart.
TEST_F(Gen2DFBCreditsTest, ConsumerDoesNotRunAheadOfProducerTensorAccessor) {
    auto mesh_device = devices_.at(0);
    IDevice* device = mesh_device->get_devices()[0];

    constexpr uint32_t num_pages = 16;
    constexpr uint32_t page_size = 1536;  // 768 bfloat16 elements, as an embedding row is
    constexpr uint32_t total_bytes = num_pages * page_size;

    const bool gen2 = device->arch() == tt::ARCH::QUASAR;
    const NodeCoord node{0, 0};
    auto tensor_spec = TensorSpec(
        Shape{num_pages, page_size / 2},
        TensorLayout(
            DataType::BFLOAT16,
            PageConfig(Layout::ROW_MAJOR),
            MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM}));

    MeshTensor input_tensor = MeshTensor::allocate_on_device(*mesh_device, tensor_spec);
    MeshTensor output_tensor = MeshTensor::allocate_on_device(*mesh_device, tensor_spec);

    for (uint32_t depth : {2u, 4u}) {
        ProgramSpec spec;
        spec.name = "gen2_dfb_credits_ta";

        KernelSpec producer{
            .unique_id = KernelSpecName{"producer"},
            .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/tensor_accessor_loopback_producer.cpp",
            .num_threads = 1,
            .hw_config = dm_hw_config(gen2, DataMovementProcessor::RISCV_0),
        };
        producer.advanced_options.num_runtime_varargs = 1;

        KernelSpec consumer{
            .unique_id = KernelSpecName{"consumer"},
            .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/tensor_accessor_loopback_consumer.cpp",
            .num_threads = 1,
            .hw_config = dm_hw_config(gen2, DataMovementProcessor::RISCV_1),
        };
        consumer.advanced_options.num_runtime_varargs = 1;

        auto dfb = MakeMinimalDFB("input_dfb", page_size, depth);
        dfb.data_format_metadata = tt::DataFormat::Float16_b;
        producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"input_dfb"}, "input_dfb"));
        consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"input_dfb"}, "input_dfb"));
        BindTensorParameterToKernel(producer, "input_tensor", "input_tensor");
        BindTensorParameterToKernel(consumer, "output_tensor", "output_tensor");

        spec.kernels = {producer, consumer};
        spec.dataflow_buffers = {dfb};
        spec.tensor_parameters = {
            TensorParameter{.unique_id = TensorParamName{"input_tensor"}, .spec = tensor_spec},
            TensorParameter{.unique_id = TensorParamName{"output_tensor"}, .spec = tensor_spec},
        };
        spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit_0", node, {"producer", "consumer"})};

        Program program = MakeProgramFromSpec(*mesh_device, spec);

        ProgramRunArgs params;
        params.kernel_run_args = {
            ProgramRunArgs::KernelRunArgs{
                .kernel = KernelSpecName{"producer"},
                .advanced_options = AdvancedKernelRunArgs{.runtime_varargs = {{node, {num_pages}}}},
            },
            ProgramRunArgs::KernelRunArgs{
                .kernel = KernelSpecName{"consumer"},
                .advanced_options = AdvancedKernelRunArgs{.runtime_varargs = {{node, {num_pages}}}},
            },
        };
        params.tensor_args = {
            {TensorParamName{"input_tensor"}, TensorArgument{input_tensor}},
            {TensorParamName{"output_tensor"}, TensorArgument{output_tensor}},
        };
        SetProgramRunArgs(program, params);

        std::vector<uint32_t> input_data(total_bytes / sizeof(uint32_t));
        for (size_t i = 0; i < input_data.size(); i++) {
            input_data[i] = static_cast<uint32_t>(i) + 0x1000u;
        }
        detail::WriteToBuffer(*input_tensor.mesh_buffer().get_reference_buffer(), input_data);
        std::vector<uint32_t> sentinel(input_data.size(), 0xDEADBEEFu);
        detail::WriteToBuffer(*output_tensor.mesh_buffer().get_reference_buffer(), sentinel);

        detail::LaunchProgram(device, program);

        std::vector<uint32_t> output_data;
        detail::ReadFromBuffer(*output_tensor.mesh_buffer().get_reference_buffer(), output_data);
        ASSERT_EQ(output_data.size(), input_data.size()) << "depth=" << depth;

        const uint32_t shift = leading_mismatch_entries(input_data, output_data, page_size / sizeof(uint32_t));
        EXPECT_EQ(shift, 0u) << "TensorAccessor pipeline, DFB depth " << depth << ": the stream came back shifted by "
                             << shift << " entries.";
    }
}

// Host-versus-kernel split for the one result that disagrees with the indexed_fill investigation's
// repro. That repro's equivalent case, one extra NoC read with no scratchpad, passes; the sweep below
// reports the same thing failing in mode 6. This test runs a minimal producer that transcribes their
// case directly, with none of the sweep kernel's mode switch, lambda or extra arguments. If this
// passes where mode 6 fails, the difference is in the sweep's kernel; if it fails too, the difference
// is host-side.
TEST_F(Gen2DFBCreditsTest, MinimalExtraReadNoScratchpad) {
    auto mesh_device = devices_.at(0);
    IDevice* device = mesh_device->get_devices()[0];
    const bool gen2 = device->arch() == tt::ARCH::QUASAR;

    constexpr uint32_t kEntry = 1024;
    constexpr uint32_t kEntries = 4;
    constexpr uint32_t kSamplesAddr = 100 * 1024;
    const uint32_t total_bytes = kEntries * kEntry;
    const NodeCoord node{0, 0};

    InterleavedBufferConfig cfg{
        .device = device, .size = total_bytes, .page_size = total_bytes, .buffer_type = BufferType::DRAM};
    auto in_buf = CreateBuffer(cfg);
    auto out_buf = CreateBuffer(cfg);
    InterleavedBufferConfig l1_cfg{.device = device, .size = 32, .page_size = 32, .buffer_type = BufferType::L1};
    auto plain_buf = CreateBuffer(l1_cfg);

    ProgramSpec spec;
    spec.name = "gen2_dfb_minimal_extra_read";

    KernelSpec producer{
        .unique_id = KernelSpecName{"producer"},
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_minimal_extra_read_producer.cpp",
        .num_threads = 1,
        .hw_config = dm_hw_config(gen2, DataMovementProcessor::RISCV_0),
    };
    producer.advanced_options.num_runtime_varargs = 4;

    KernelSpec consumer{
        .unique_id = KernelSpecName{"consumer"},
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_credit_probe_consumer.cpp",
        .num_threads = 1,
        .hw_config = dm_hw_config(gen2, DataMovementProcessor::RISCV_1),
    };
    consumer.advanced_options.num_runtime_varargs = 4;

    auto dfb = MakeMinimalDFB("loopback_dfb", kEntry, 2);
    dfb.data_format_metadata = tt::DataFormat::Float16_b;
    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"loopback_dfb"}, "my_local_dfb_name"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"loopback_dfb"}, "consumer_dfb"));

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit_0", node, {"producer", "consumer"})};

    Program program = MakeProgramFromSpec(*mesh_device, spec);

    ProgramRunArgs params;
    params.kernel_run_args = {
        ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"producer"},
            .advanced_options =
                AdvancedKernelRunArgs{
                    .runtime_varargs =
                        {{node,
                          {static_cast<uint32_t>(in_buf->address()),
                           0u,
                           kEntries,
                           static_cast<uint32_t>(plain_buf->address())}}},
                },
        },
        ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"consumer"},
            .advanced_options =
                AdvancedKernelRunArgs{
                    .runtime_varargs =
                        {{node, {static_cast<uint32_t>(out_buf->address()), 0u, kEntries, kSamplesAddr}}},
                },
        },
    };
    SetProgramRunArgs(program, params);

    const std::vector<uint32_t> input = make_input(kEntries, kEntry);
    std::vector<uint32_t> sentinel(input.size(), 0xDEADBEEFu);
    std::vector<uint32_t> zero_samples(kEntries, 0u);
    detail::WriteToBuffer(in_buf, input);
    detail::WriteToBuffer(out_buf, sentinel);
    detail::WriteToDeviceL1(device, node, kSamplesAddr, zero_samples);

    detail::LaunchProgram(device, program);

    std::vector<uint32_t> samples;
    detail::ReadFromDeviceL1(device, node, kSamplesAddr, kEntries * sizeof(uint32_t), samples);
    std::vector<uint32_t> output;
    detail::ReadFromBuffer(out_buf, output);

    const uint32_t words_per_entry = kEntry / sizeof(uint32_t);
    std::string trace;
    uint32_t bad = 0;
    for (uint32_t i = 0; i < kEntries; i++) {
        if (samples[i] != input[i * words_per_entry]) {
            bad++;
        }
        trace += (i ? "," : "") + std::to_string(samples[i]);
    }
    GTEST_LOG_(INFO) << "minimal extra read: grants early " << bad << "/" << kEntries << ", output "
                     << (output == input ? "exact" : "WRONG") << ", samples [" << trace << "]";
    EXPECT_EQ(bad, 0u);
    EXPECT_EQ(output, input);
}

// Does the buffer count credits per NoC transaction completion rather than per push_back?
//
// Every case here fills the buffer correctly and every transfer targets the buffer itself, so there is
// no second destination involved. Only the ratio of NoC reads to announced slots changes. If the
// credits track transactions, a ratio other than one to one breaks the buffer on its own, which would
// mean the trigger has nothing to do with scratchpads or with where an extra read lands.
//
// Two reads per announced slot should over-grant. The complementary case, one read per two announced
// slots, should under-grant and starve the consumer, so it lives in its own test below.
//
// Shared by both: `ratio_mode` picks the shape, and the consumer samples the granted entry's first word
// so an early release is caught whether or not it also corrupts the output.
uint32_t run_ratio_case(
    distributed::MeshDevice& mesh_device,
    IDevice* device,
    uint32_t ratio_mode,
    uint32_t num_entries,
    uint32_t entry_size,
    uint32_t depth,
    std::vector<uint32_t>& samples_out,
    bool& output_exact_out,
    bool disable_implicit_sync = false,
    uint32_t sub_reads = 2) {
    const bool gen2 = device->arch() == tt::ARCH::QUASAR;
    constexpr uint32_t kSamplesAddr = 100 * 1024;
    const NodeCoord node{0, 0};
    const uint32_t total_bytes = num_entries * entry_size;

    InterleavedBufferConfig cfg{
        .device = device, .size = total_bytes, .page_size = total_bytes, .buffer_type = BufferType::DRAM};
    auto in_buf = CreateBuffer(cfg);
    auto out_buf = CreateBuffer(cfg);

    ProgramSpec spec;
    spec.name = "gen2_dfb_ratio";

    KernelSpec producer{
        .unique_id = KernelSpecName{"producer"},
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_ratio_probe_producer.cpp",
        .num_threads = 1,
        .hw_config = dm_hw_config(gen2, DataMovementProcessor::RISCV_0, disable_implicit_sync),
    };
    producer.advanced_options.num_runtime_varargs = 5;

    KernelSpec consumer{
        .unique_id = KernelSpecName{"consumer"},
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_credit_probe_consumer.cpp",
        .num_threads = 1,
        .hw_config = dm_hw_config(gen2, DataMovementProcessor::RISCV_1, disable_implicit_sync),
    };
    consumer.advanced_options.num_runtime_varargs = 4;

    auto dfb = MakeMinimalDFB("loopback_dfb", entry_size, depth);
    dfb.data_format_metadata = tt::DataFormat::Float16_b;
    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"loopback_dfb"}, "my_local_dfb_name"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"loopback_dfb"}, "consumer_dfb"));

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit_0", node, {"producer", "consumer"})};

    Program program = MakeProgramFromSpec(mesh_device, spec);

    ProgramRunArgs params;
    params.kernel_run_args = {
        ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"producer"},
            .advanced_options =
                AdvancedKernelRunArgs{
                    .runtime_varargs =
                        {{node, {static_cast<uint32_t>(in_buf->address()), 0u, num_entries, ratio_mode, sub_reads}}},
                },
        },
        ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"consumer"},
            .advanced_options =
                AdvancedKernelRunArgs{
                    .runtime_varargs =
                        {{node, {static_cast<uint32_t>(out_buf->address()), 0u, num_entries, kSamplesAddr}}},
                },
        },
    };
    SetProgramRunArgs(program, params);

    const std::vector<uint32_t> input = make_input(num_entries, entry_size);
    std::vector<uint32_t> sentinel(input.size(), 0xDEADBEEFu);
    std::vector<uint32_t> zero_samples(num_entries, 0u);
    detail::WriteToBuffer(in_buf, input);
    detail::WriteToBuffer(out_buf, sentinel);
    detail::WriteToDeviceL1(device, node, kSamplesAddr, zero_samples);

    detail::LaunchProgram(device, program);

    detail::ReadFromDeviceL1(device, node, kSamplesAddr, num_entries * sizeof(uint32_t), samples_out);
    std::vector<uint32_t> output;
    detail::ReadFromBuffer(out_buf, output);
    output_exact_out = (output == input);

    const uint32_t words_per_entry = entry_size / sizeof(uint32_t);
    uint32_t bad = 0;
    for (uint32_t i = 0; i < num_entries; i++) {
        if (samples_out[i] != input[i * words_per_entry]) {
            bad++;
        }
    }
    return bad;
}

// Two half-entry reads per announced slot, both into the buffer. Nothing here is unusual except the
// ratio, so a failure means the credits are counting transactions.
TEST_F(Gen2DFBCreditsTest, RatioTwoReadsPerAnnouncedSlot) {
    auto mesh_device = devices_.at(0);
    IDevice* device = mesh_device->get_devices()[0];

    for (uint32_t ratio_mode : {0u, 1u}) {
        std::vector<uint32_t> samples;
        bool output_exact = false;
        const uint32_t bad = run_ratio_case(*mesh_device, device, ratio_mode, 4, 1024, 2, samples, output_exact);
        std::string trace;
        for (size_t i = 0; i < samples.size(); i++) {
            trace += (i ? "," : "") + std::to_string(samples[i]);
        }
        GTEST_LOG_(INFO) << (ratio_mode == 0 ? "one read per slot (control): " : "two reads per slot: ")
                         << "grants early " << bad << "/4, output " << (output_exact ? "exact" : "WRONG")
                         << ", samples [" << trace << "]";
        EXPECT_EQ(bad, 0u) << "ratio_mode " << ratio_mode;
        EXPECT_TRUE(output_exact) << "ratio_mode " << ratio_mode;
    }
}

// The failing ratio again, with Gen2 implicit sync opted out on both kernels.
//
// This is the question that decides whether the surplus is the implicit-sync mechanism posting a credit
// per NoC transaction, or something below it. It also decides whether the ops that run ResNet on Quasar
// are already protected: every one of them passes disable_dfb_implicit_sync_for_all = true, which is the
// workaround recorded in issue #50328.
TEST_F(Gen2DFBCreditsTest, RatioTwoReadsPerAnnouncedSlotNoImplicitSync) {
    auto mesh_device = devices_.at(0);
    IDevice* device = mesh_device->get_devices()[0];

    std::vector<uint32_t> samples;
    bool output_exact = false;
    const uint32_t bad =
        run_ratio_case(*mesh_device, device, 1, 4, 1024, 2, samples, output_exact, /*disable_implicit_sync=*/true);
    std::string trace;
    for (size_t i = 0; i < samples.size(); i++) {
        trace += (i ? "," : "") + std::to_string(samples[i]);
    }
    GTEST_LOG_(INFO) << "two reads per slot, implicit sync off: grants early " << bad << "/4, output "
                     << (output_exact ? "exact" : "WRONG") << ", samples [" << trace << "]";
    EXPECT_EQ(bad, 0u);
    EXPECT_TRUE(output_exact);
}

// The complementary case: one double-entry read per two announced entries, so there are FEWER NoC
// transactions than announcements. This passes on both generations, and that asymmetry is the point.
// A surplus of transactions over-grants the consumer, while a deficit costs nothing, so whatever the
// accounting error is, it does not simply track transaction count in both directions.
TEST_F(Gen2DFBCreditsTest, RatioOneReadPerTwoAnnouncedSlotsCompletes) {
    auto mesh_device = devices_.at(0);
    IDevice* device = mesh_device->get_devices()[0];

    std::vector<uint32_t> samples;
    bool output_exact = false;
    const uint32_t bad = run_ratio_case(*mesh_device, device, 2, 4, 1024, 2, samples, output_exact);
    GTEST_LOG_(INFO) << "one read per two slots: grants early " << bad << "/4, output "
                     << (output_exact ? "exact" : "WRONG");
    EXPECT_EQ(bad, 0u);
    EXPECT_TRUE(output_exact);
}

// Scopes the failing combination: which uses of a scratchpad, alongside a DataflowBuffer in the same
// kernel, disturb the buffer?
//
// The DFB half of the producer is identical and correct in every case. Only the extra work varies. A
// start delay makes the consumer reach its first wait_front before the producer starts, so the outcome
// is deterministic rather than a race; a correct buffer is unaffected by that, since it must hold the
// consumer until the entry is readable however late the producer starts.
//
// The check is per grant, not on the final output. The consumer samples the granted entry's first word
// the instant wait_front returns, and each entry carries a distinct value, so grant k must see entry
// k's word. Checking the output alone under-reports, because a wrongly released consumer still has to
// lose the ensuing race to the in-flight data before the output goes wrong.
//
// Mode 3 is the sharp one. It issues the identical transfer to the scratchpad's *own base address*,
// taken from the scratchpad itself, but routed through a plain CoreLocalMem instead of the binding. If
// mode 1 fails where mode 3 passes, the trigger is the binding rather than the address, the region, or
// proximity to the ring.
// A bfloat16 32x32 tile, which is the entry size a compute endpoint requires.
constexpr uint32_t kTileBytes = 2048;

// Tile t is filled entirely with the bfloat16 value t+1, so each tile is a distinct small integer that
// a copy reproduces bit for bit. The sequential words make_input produces would not work here: read as
// bfloat16 they include NaNs and denormals, which a copy through the destination registers is not
// required to preserve, so a mismatch would not distinguish a credit fault from ordinary rounding.
std::vector<uint32_t> make_tile_input(uint32_t num_tiles) {
    const uint32_t words_per_tile = kTileBytes / sizeof(uint32_t);
    std::vector<uint32_t> input(num_tiles * words_per_tile);
    for (uint32_t t = 0; t < num_tiles; t++) {
        const float value = static_cast<float>(t + 1);
        uint32_t float_bits = 0;
        std::memcpy(&float_bits, &value, sizeof(float_bits));
        const uint32_t half = float_bits >> 16;  // bfloat16 is the top half of a float
        const uint32_t packed = (half << 16) | half;
        for (uint32_t w = 0; w < words_per_tile; w++) {
            input[t * words_per_tile + w] = packed;
        }
    }
    return input;
}

// The same producer and the same ratio as run_ratio_case, but the buffer it fills is drained by a
// COMPUTE kernel instead of a data-movement one.
//
// Pipeline: DM producer -> compute_in_dfb -> compute tile copy -> compute_out_dfb -> DM writer -> DRAM.
//
// Only the first buffer is under test. The producer varies its transaction-to-announcement ratio on
// compute_in_dfb exactly as in the data-movement version. The second buffer and the writer exist only
// to get the compute kernel's output where the host can read it, and the writer is strictly one
// transfer per announced slot so it cannot contribute a surplus of its own.
//
// This isolates who performs the wait. A data-movement consumer spins on the buffer's occupancy from a
// RISC; a compute consumer waits through the unpacker. If the surplus only breaks the first, then the
// condition for the defect is narrower than "a surplus of transactions" alone.
uint32_t run_compute_consumer_ratio_case(
    distributed::MeshDevice& mesh_device,
    IDevice* device,
    uint32_t ratio_mode,
    uint32_t num_tiles,
    uint32_t depth,
    std::vector<uint32_t>& samples_out,
    bool& output_exact_out,
    bool disable_implicit_sync = false,
    uint32_t sub_reads = 2) {
    const bool gen2 = device->arch() == tt::ARCH::QUASAR;
    constexpr uint32_t kSamplesAddr = 100 * 1024;
    const NodeCoord node{0, 0};
    const uint32_t total_bytes = num_tiles * kTileBytes;

    InterleavedBufferConfig cfg{
        .device = device, .size = total_bytes, .page_size = total_bytes, .buffer_type = BufferType::DRAM};
    auto in_buf = CreateBuffer(cfg);
    auto out_buf = CreateBuffer(cfg);

    ProgramSpec spec;
    spec.name = "gen2_dfb_ratio_compute";

    KernelSpec producer{
        .unique_id = KernelSpecName{"producer"},
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_ratio_probe_producer.cpp",
        .num_threads = 1,
        .hw_config = dm_hw_config(gen2, DataMovementProcessor::RISCV_0, disable_implicit_sync),
    };
    producer.advanced_options.num_runtime_varargs = 5;

    KernelSpec compute{
        .unique_id = KernelSpecName{"compute"},
        .source = "tests/tt_metal/tt_metal/test_kernels/compute/dfb_tile_copy_compute.cpp",
        .num_threads = 1,
        .hw_config = compute_hw_config(gen2),
    };
    compute.advanced_options.num_runtime_varargs = 1;

    KernelSpec writer{
        .unique_id = KernelSpecName{"writer"},
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_credit_probe_consumer.cpp",
        .num_threads = 1,
        .hw_config = dm_hw_config(gen2, DataMovementProcessor::RISCV_1, disable_implicit_sync),
    };
    writer.advanced_options.num_runtime_varargs = 4;

    auto dfb_in = MakeMinimalDFB("compute_in_dfb", kTileBytes, depth);
    dfb_in.data_format_metadata = tt::DataFormat::Float16_b;
    auto dfb_out = MakeMinimalDFB("compute_out_dfb", kTileBytes, depth);
    dfb_out.data_format_metadata = tt::DataFormat::Float16_b;

    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"compute_in_dfb"}, "my_local_dfb_name"));
    compute.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"compute_in_dfb"}, "in0"));
    compute.dfb_bindings.push_back(ProducerOf(DFBSpecName{"compute_out_dfb"}, "out"));
    writer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"compute_out_dfb"}, "consumer_dfb"));

    spec.kernels = {producer, compute, writer};
    spec.dataflow_buffers = {dfb_in, dfb_out};
    spec.work_units =
        std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit_0", node, {"producer", "compute", "writer"})};

    Program program = MakeProgramFromSpec(mesh_device, spec);

    ProgramRunArgs params;
    params.kernel_run_args = {
        ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"producer"},
            .advanced_options =
                AdvancedKernelRunArgs{
                    .runtime_varargs =
                        {{node, {static_cast<uint32_t>(in_buf->address()), 0u, num_tiles, ratio_mode, sub_reads}}},
                },
        },
        ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"compute"},
            .advanced_options = AdvancedKernelRunArgs{.runtime_varargs = {{node, {num_tiles}}}},
        },
        ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"writer"},
            .advanced_options =
                AdvancedKernelRunArgs{
                    .runtime_varargs =
                        {{node, {static_cast<uint32_t>(out_buf->address()), 0u, num_tiles, kSamplesAddr}}},
                },
        },
    };
    SetProgramRunArgs(program, params);

    const std::vector<uint32_t> input = make_tile_input(num_tiles);
    std::vector<uint32_t> sentinel(input.size(), 0xDEADBEEFu);
    std::vector<uint32_t> zero_samples(num_tiles, 0u);
    detail::WriteToBuffer(in_buf, input);
    detail::WriteToBuffer(out_buf, sentinel);
    detail::WriteToDeviceL1(device, node, kSamplesAddr, zero_samples);

    detail::LaunchProgram(device, program);

    detail::ReadFromDeviceL1(device, node, kSamplesAddr, num_tiles * sizeof(uint32_t), samples_out);
    std::vector<uint32_t> output;
    detail::ReadFromBuffer(out_buf, output);
    output_exact_out = (output == input);

    const uint32_t words_per_tile = kTileBytes / sizeof(uint32_t);
    uint32_t bad = 0;
    for (uint32_t i = 0; i < num_tiles; i++) {
        if (samples_out[i] != input[i * words_per_tile]) {
            bad++;
        }
    }
    return bad;
}

// Control for the compute-consumer pipeline: one read per announced slot. Establishes that the
// three-kernel pipeline itself is sound, so a failure in the surplus case below cannot be blamed on
// the tile copy, the data format, or the extra buffer.
TEST_F(Gen2DFBCreditsTest, ComputeConsumerOneReadPerAnnouncedSlot) {
    auto mesh_device = devices_.at(0);
    IDevice* device = mesh_device->get_devices()[0];

    std::vector<uint32_t> samples;
    bool output_exact = false;
    const uint32_t bad = run_compute_consumer_ratio_case(*mesh_device, device, 0, 4, 2, samples, output_exact);
    GTEST_LOG_(INFO) << "compute consumer, one read per slot (control): tiles wrong at the writer " << bad
                     << "/4, output " << (output_exact ? "exact" : "WRONG");
    EXPECT_EQ(bad, 0u);
    EXPECT_TRUE(output_exact);
}

// The failing ratio from RatioTwoReadsPerAnnouncedSlot, with a compute kernel in the consumer position.
//
// Same producer, same two half-entry reads per announced slot, same surplus. The only change is which
// hardware waits on the buffer. If this passes while the data-movement version fails, the defect needs
// both a surplus of transactions AND a data-movement consumer, which is a much narrower condition than
// the surplus alone.
TEST_F(Gen2DFBCreditsTest, ComputeConsumerTwoReadsPerAnnouncedSlot) {
    auto mesh_device = devices_.at(0);
    IDevice* device = mesh_device->get_devices()[0];

    std::vector<uint32_t> samples;
    bool output_exact = false;
    const uint32_t bad = run_compute_consumer_ratio_case(*mesh_device, device, 1, 4, 2, samples, output_exact);
    GTEST_LOG_(INFO) << "compute consumer, two reads per slot: tiles wrong at the writer " << bad << "/4, output "
                     << (output_exact ? "exact" : "WRONG");
    EXPECT_EQ(bad, 0u);
    EXPECT_TRUE(output_exact);
}

// The surplus pushed well past the buffer's depth, still with a compute consumer.
//
// Eight reads fill each announced slot, so the surplus is seven per slot against a two-slot ring. That
// magnitude is what turns corruption into a deadlock on the data-movement path, which makes this the
// test that separates a genuinely immune consumer from one that is merely winning a race: the writer
// only sees a tile after the copy, so a small surplus could be masked by the copy's own latency, and a
// surplus this large cannot be.
TEST_F(Gen2DFBCreditsTest, ComputeConsumerManyReadsPerAnnouncedSlot) {
    auto mesh_device = devices_.at(0);
    IDevice* device = mesh_device->get_devices()[0];

    std::vector<uint32_t> samples;
    bool output_exact = false;
    const uint32_t bad = run_compute_consumer_ratio_case(
        *mesh_device, device, 3, 4, 2, samples, output_exact, /*disable_implicit_sync=*/false, /*sub_reads=*/8);
    GTEST_LOG_(INFO) << "compute consumer, eight reads per slot: tiles wrong at the writer " << bad << "/4, output "
                     << (output_exact ? "exact" : "WRONG");
    EXPECT_EQ(bad, 0u);
    EXPECT_TRUE(output_exact);
}

// The same eight-reads-per-slot load with a data-movement consumer, for the side-by-side comparison.
//
// Kept separate from the rest because a surplus this large is the regime where the data-movement path
// stops returning at all, so this test can hang and needs an external timeout.
TEST_F(Gen2DFBCreditsTest, DmConsumerManyReadsPerAnnouncedSlot) {
    auto mesh_device = devices_.at(0);
    IDevice* device = mesh_device->get_devices()[0];

    std::vector<uint32_t> samples;
    bool output_exact = false;
    const uint32_t bad = run_ratio_case(
        *mesh_device, device, 3, 4, 2048, 2, samples, output_exact, /*disable_implicit_sync=*/false, /*sub_reads=*/8);
    GTEST_LOG_(INFO) << "DM consumer, eight reads per slot: grants early " << bad << "/4, output "
                     << (output_exact ? "exact" : "WRONG");
    EXPECT_EQ(bad, 0u);
    EXPECT_TRUE(output_exact);
}

TEST_F(Gen2DFBCreditsTest, ScratchpadUsePatternsThatDisturbTheBuffer) {
    auto mesh_device = devices_.at(0);
    IDevice* device = mesh_device->get_devices()[0];
    const bool gen2 = device->arch() == tt::ARCH::QUASAR;

    constexpr uint32_t kEntry = 1024;
    constexpr uint32_t kEntries = 4;
    constexpr uint32_t kDepth = 2;
    constexpr uint32_t kPadCapacity = 4096;
    constexpr uint32_t kExtraBytes = 32;
    constexpr uint32_t kStartDelay = 4000;
    constexpr uint32_t kSamplesAddr = 100 * 1024;

    const uint32_t total_bytes = kEntries * kEntry;
    const NodeCoord node{0, 0};

    struct Case {
        const char* name;
        uint32_t mode;
        uint32_t period;  // 0 = once before the DFB loop
    };
    const Case cases[] = {
        {"mode 0: nothing besides producing", 0, 0},
        {"mode 1: NoC-read into the scratchpad binding, once before the loop", 1, 0},
        {"mode 1: NoC-read into the scratchpad binding, every 2nd entry", 1, 2},
        {"mode 2: scratchpad touched from the CPU only", 2, 0},
        {"mode 3: same address as the scratchpad, via plain CoreLocalMem", 3, 0},
        {"mode 4: scratchpad binding, far end of a large scratchpad", 4, 0},
        {"mode 5: unrelated SRAM address, scratchpad bound but untouched", 5, 0},
        {"mode 6: unrelated SRAM address, no scratchpad bound at all", 6, 0},
    };

    std::string summary;
    for (const Case& c : cases) {
        GTEST_LOG_(INFO) << "case: " << c.name;

        InterleavedBufferConfig cfg{
            .device = device, .size = total_bytes, .page_size = total_bytes, .buffer_type = BufferType::DRAM};
        auto in_buf = CreateBuffer(cfg);
        auto out_buf = CreateBuffer(cfg);
        auto extra_buf = CreateBuffer(cfg);

        // Modes 5 and 6 need a plain SRAM destination that is provably safe to write. Take it from the
        // allocator rather than hardcoding an address: a literal is a guess about the memory map, and a
        // wrong guess corrupts whatever really lives there and looks exactly like the defect.
        InterleavedBufferConfig l1_cfg{
            .device = device, .size = kExtraBytes, .page_size = kExtraBytes, .buffer_type = BufferType::L1};
        auto plain_buf = CreateBuffer(l1_cfg);
        const uint32_t plain_addr = static_cast<uint32_t>(plain_buf->address());

        ProgramSpec spec;
        spec.name = "gen2_dfb_scratchpad_scope";

        KernelSpec producer{
            .unique_id = KernelSpecName{"producer"},
            .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_scratchpad_scope_producer.cpp",
            .num_threads = 1,
            .hw_config = dm_hw_config(gen2, DataMovementProcessor::RISCV_0),
        };
        producer.advanced_options.num_runtime_varargs = 9;

        KernelSpec consumer{
            .unique_id = KernelSpecName{"consumer"},
            .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_credit_probe_consumer.cpp",
            .num_threads = 1,
            .hw_config = dm_hw_config(gen2, DataMovementProcessor::RISCV_1),
        };
        consumer.advanced_options.num_runtime_varargs = 4;

        auto dfb = MakeMinimalDFB("loopback_dfb", kEntry, kDepth);
        dfb.data_format_metadata = tt::DataFormat::Float16_b;
        producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"loopback_dfb"}, "my_local_dfb_name"));
        consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"loopback_dfb"}, "consumer_dfb"));
        // Mode 6 is the only case with no scratchpad. A declared-but-unbound ScratchpadSpec is
        // rejected at program creation, so the spec entry has to go with the binding.
        const bool bind_pad = (c.mode != 6);
        if (bind_pad) {
            producer.scratchpad_bindings.push_back(
                ScratchpadBinding{.scratchpad_spec_name = ScratchpadSpecName{"staging"}, .accessor_name = "staging"});
        } else {
            producer.compiler_options.defines.insert({"NO_SCRATCHPAD_BINDING", "1"});
        }

        spec.kernels = {producer, consumer};
        spec.dataflow_buffers = {dfb};
        if (bind_pad) {
            spec.scratchpads = {
                ScratchpadSpec{.unique_id = ScratchpadSpecName{"staging"}, .size_per_node = kPadCapacity}};
        }
        spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit_0", node, {"producer", "consumer"})};

        Program program = MakeProgramFromSpec(*mesh_device, spec);

        ProgramRunArgs params;
        params.kernel_run_args = {
            ProgramRunArgs::KernelRunArgs{
                .kernel = KernelSpecName{"producer"},
                .advanced_options =
                    AdvancedKernelRunArgs{
                        .runtime_varargs =
                            {{node,
                              {static_cast<uint32_t>(in_buf->address()),
                               0u,
                               kEntries,
                               static_cast<uint32_t>(extra_buf->address()),
                               c.mode,
                               kExtraBytes,
                               c.period,
                               kStartDelay,
                               plain_addr}}},
                    },
            },
            ProgramRunArgs::KernelRunArgs{
                .kernel = KernelSpecName{"consumer"},
                .advanced_options =
                    AdvancedKernelRunArgs{
                        .runtime_varargs =
                            {{node, {static_cast<uint32_t>(out_buf->address()), 0u, kEntries, kSamplesAddr}}},
                    },
            },
        };
        SetProgramRunArgs(program, params);

        const std::vector<uint32_t> input = make_input(kEntries, kEntry);
        detail::WriteToBuffer(in_buf, input);
        std::vector<uint32_t> extra_fill(input.size(), 0xA5A5A5A5u);
        std::vector<uint32_t> sentinel(input.size(), 0xDEADBEEFu);
        std::vector<uint32_t> zero_samples(kEntries, 0u);
        detail::WriteToBuffer(extra_buf, extra_fill);
        detail::WriteToBuffer(out_buf, sentinel);
        detail::WriteToDeviceL1(device, node, kSamplesAddr, zero_samples);

        detail::LaunchProgram(device, program);

        std::vector<uint32_t> samples;
        detail::ReadFromDeviceL1(device, node, kSamplesAddr, kEntries * sizeof(uint32_t), samples);
        // Read the delivered data too. Comparing the two tells a real early release from a probe
        // artifact: if the probe flags a grant but the output is byte-exact, suspect the probe.
        std::vector<uint32_t> output;
        detail::ReadFromBuffer(out_buf, output);
        const bool output_exact = (output == input);

        const uint32_t words_per_entry = kEntry / sizeof(uint32_t);
        uint32_t bad_grants = 0;
        std::string trace;
        for (uint32_t i = 0; i < kEntries; i++) {
            const uint32_t want = input[i * words_per_entry];
            if (samples[i] != want) {
                bad_grants++;
            }
            trace += (i ? "," : "") + std::to_string(samples[i]);
        }

        std::string line = std::string((bad_grants == 0 && output_exact) ? "pass" : "FAIL") + "  " + c.name;
        if (bad_grants != 0 || !output_exact) {
            line += "  (grants early " + std::to_string(bad_grants) + "/" + std::to_string(kEntries) + ", output " +
                    (output_exact ? "exact" : "WRONG") + ", samples [" + trace + "], want [";
            for (uint32_t i = 0; i < kEntries; i++) {
                line += (i ? "," : "") + std::to_string(input[i * words_per_entry]);
            }
            line += "])";
        }
        GTEST_LOG_(INFO) << "result: " << line;
        summary += "  " + line + "\n";
    }

    GTEST_LOG_(INFO) << "scratchpad use patterns, " << (gen2 ? "Quasar" : "Gen1") << ":\n" << summary;
    EXPECT_EQ(summary.find("FAIL"), std::string::npos) << "some pattern released the consumer early:\n" << summary;
}

// The one scratchpad pattern that hangs on Quasar instead of corrupting: a NoC read into the
// scratchpad on every entry.
//
// Kept separate from the sweep above because a hang would stop that sweep from reporting its other
// rows. It passes on Gen1 in well under a second, so a Quasar run that does not finish promptly is
// the failure. There is no per-test timeout in gtest, so run the suite under an external timeout, or
// exclude this one with:
//   --gtest_filter="Gen2DFBCreditsTest.*:-Gen2DFBCreditsTest.ScratchpadReadEveryEntry*"
TEST_F(Gen2DFBCreditsTest, ScratchpadReadEveryEntryCompletes) {
    auto mesh_device = devices_.at(0);
    IDevice* device = mesh_device->get_devices()[0];
    const bool gen2 = device->arch() == tt::ARCH::QUASAR;

    constexpr uint32_t kEntry = 1024;
    constexpr uint32_t kEntries = 8;
    constexpr uint32_t kPadCapacity = 4096;
    constexpr uint32_t kExtraBytes = 32;
    constexpr uint32_t kStartDelay = 4000;
    constexpr uint32_t kScratchpadReadMode = 1;  // NoC-read into the scratchpad through its binding
    const uint32_t total_bytes = kEntries * kEntry;
    const NodeCoord node{0, 0};

    InterleavedBufferConfig cfg{
        .device = device, .size = total_bytes, .page_size = total_bytes, .buffer_type = BufferType::DRAM};
    auto in_buf = CreateBuffer(cfg);
    auto out_buf = CreateBuffer(cfg);
    InterleavedBufferConfig pad_cfg{
        .device = device, .size = kExtraBytes, .page_size = kExtraBytes, .buffer_type = BufferType::DRAM};
    auto pad_buf = CreateBuffer(pad_cfg);

    ProgramSpec spec;
    spec.name = "gen2_dfb_scratchpad_every_entry";

    KernelSpec producer{
        .unique_id = KernelSpecName{"producer"},
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_scratchpad_scope_producer.cpp",
        .num_threads = 1,
        .hw_config = dm_hw_config(gen2, DataMovementProcessor::RISCV_0),
    };
    producer.advanced_options.num_runtime_varargs = 9;

    KernelSpec consumer{
        .unique_id = KernelSpecName{"consumer"},
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_accessor_loopback_consumer.cpp",
        .num_threads = 1,
        .hw_config = dm_hw_config(gen2, DataMovementProcessor::RISCV_1),
    };
    consumer.advanced_options.num_runtime_varargs = 3;

    auto dfb = MakeMinimalDFB("loopback_dfb", kEntry, 2);
    dfb.data_format_metadata = tt::DataFormat::Float16_b;
    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"loopback_dfb"}, "my_local_dfb_name"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"loopback_dfb"}, "a_dfb_named_bob"));
    producer.scratchpad_bindings.push_back(
        ScratchpadBinding{.scratchpad_spec_name = ScratchpadSpecName{"staging"}, .accessor_name = "staging"});

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"staging"}, .size_per_node = kPadCapacity}};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit_0", node, {"producer", "consumer"})};

    Program program = MakeProgramFromSpec(*mesh_device, spec);

    ProgramRunArgs params;
    params.kernel_run_args = {
        ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"producer"},
            .advanced_options =
                AdvancedKernelRunArgs{
                    .runtime_varargs =
                        {{node,
                          {
                              static_cast<uint32_t>(in_buf->address()),
                              0u,  // bank_id
                              kEntries,
                              static_cast<uint32_t>(pad_buf->address()),
                              kScratchpadReadMode,
                              kExtraBytes,
                              1u,  // period: read into the scratchpad on every entry
                              kStartDelay,
                              0u,  // plain SRAM address, unused by this mode
                          }}},
                },
        },
        ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"consumer"},
            .advanced_options =
                AdvancedKernelRunArgs{
                    .runtime_varargs = {{node, {static_cast<uint32_t>(out_buf->address()), 0u, kEntries}}},
                },
        },
    };
    SetProgramRunArgs(program, params);

    const std::vector<uint32_t> input = make_input(kEntries, kEntry);
    detail::WriteToBuffer(in_buf, input);
    detail::WriteToBuffer(pad_buf, std::vector<uint32_t>(kExtraBytes / sizeof(uint32_t), 0xA5A5A5A5u));
    detail::WriteToBuffer(out_buf, std::vector<uint32_t>(input.size(), 0xDEADBEEFu));

    detail::LaunchProgram(device, program);

    std::vector<uint32_t> output;
    detail::ReadFromBuffer(out_buf, output);
    EXPECT_EQ(output, input);
}

// Direct measurement of the same defect: what had the producer pushed when each grant was taken?
//
// A DFB hands the consumer entry i only after the producer has pushed it, so observation i must be
// at least i + 1. The producer publishes its count after push_back, so a store still in flight can
// make an observation low by one; the bound below allows exactly that one and no more.
//
// A leading run of zeros is the unambiguous form of the defect: the consumer was granted entries the
// producer had not pushed at all. The run length is expected to equal the DFB's capacity.
TEST_F(Gen2DFBCreditsTest, ConsumerObservesProducerPushCountAtEachGrant) {
    auto mesh_device = devices_.at(0);
    IDevice* device = mesh_device->get_devices()[0];

    constexpr uint32_t kDfbDepth = 2;
    // Host-known SRAM address for the consumer's per-grant samples.
    constexpr uint32_t kSamplesAddr = 100 * 1024;
    const bool gen2 = device->arch() == tt::ARCH::QUASAR;

    const NodeCoord node{0, 0};
    const std::vector<uint32_t> input = make_input(kNumTransfers, kEntrySize);
    const uint32_t total_bytes = kNumTransfers * kEntrySize;

    InterleavedBufferConfig dram_config{
        .device = device, .size = total_bytes, .page_size = total_bytes, .buffer_type = BufferType::DRAM};
    auto input_buffer = CreateBuffer(dram_config);
    auto output_buffer = CreateBuffer(dram_config);

    ProgramSpec spec;
    spec.name = "gen2_dfb_credit_probe";

    KernelSpec producer{
        .unique_id = KernelSpecName{"producer"},
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_credit_probe_producer.cpp",
        .num_threads = 1,
        .hw_config = dm_hw_config(gen2, DataMovementProcessor::RISCV_0),
    };
    producer.advanced_options.num_runtime_varargs = 3;

    KernelSpec consumer{
        .unique_id = KernelSpecName{"consumer"},
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_credit_probe_consumer.cpp",
        .num_threads = 1,
        .hw_config = dm_hw_config(gen2, DataMovementProcessor::RISCV_1),
    };
    consumer.advanced_options.num_runtime_varargs = 4;

    auto dfb = MakeMinimalDFB("probe_dfb", kEntrySize, kDfbDepth);
    dfb.data_format_metadata = tt::DataFormat::Float16_b;
    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"probe_dfb"}, "producer_dfb"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"probe_dfb"}, "consumer_dfb"));

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit_0", node, {"producer", "consumer"})};

    Program program = MakeProgramFromSpec(*mesh_device, spec);

    ProgramRunArgs params;
    params.kernel_run_args = {
        ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"producer"},
            .advanced_options =
                AdvancedKernelRunArgs{
                    .runtime_varargs =
                        {{node,
                          {
                              static_cast<uint32_t>(input_buffer->address()),
                              0u,  // bank_id (single-page buffer, so bank 0)
                              kNumTransfers,
                          }}},
                },
        },
        ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"consumer"},
            .advanced_options =
                AdvancedKernelRunArgs{
                    .runtime_varargs =
                        {{node,
                          {
                              static_cast<uint32_t>(output_buffer->address()),
                              0u,  // bank_id
                              kNumTransfers,
                              kSamplesAddr,
                          }}},
                },
        },
    };
    SetProgramRunArgs(program, params);

    detail::WriteToBuffer(input_buffer, input);

    // Zero the sample region so a kernel that never wrote it is not mistaken for a measurement.
    std::vector<uint32_t> zero_samples(kNumTransfers, 0u);
    detail::WriteToDeviceL1(device, node, kSamplesAddr, zero_samples);

    detail::LaunchProgram(device, program);

    std::vector<uint32_t> samples;
    detail::ReadFromDeviceL1(device, node, kSamplesAddr, kNumTransfers * sizeof(uint32_t), samples);
    ASSERT_EQ(samples.size(), static_cast<size_t>(kNumTransfers));

    // The producer stages entry i from a distinct region of the input, so entry i's first word is
    // known here. A grant is legitimate only if the entry it hands over already holds that word.
    const uint32_t words_per_entry = kEntrySize / sizeof(uint32_t);
    std::string trace;
    uint32_t stale_grants = 0;
    for (uint32_t i = 0; i < kNumTransfers; i++) {
        const uint32_t want = input[i * words_per_entry];
        if (samples[i] != want) {
            stale_grants++;
        }
        trace += (i ? ", " : "") + std::to_string(samples[i]) +
                 (samples[i] == want ? "" : "(want " + std::to_string(want) + ")");
    }

    EXPECT_EQ(stale_grants, 0u)
        << stale_grants << " of " << kNumTransfers << " grants handed over an entry the producer had not filled yet. "
        << "wait_front(n) is required to block until the matching push_back(n), so every sample must equal the word "
        << "the producer staged for that entry. DFB capacity is " << kDfbDepth << ". Samples per grant: [" << trace
        << "]";
}

// The baseline again with the Gen2 implicit-sync ISR opted out on both kernels, confirming the opt-out
// does not itself break a program that works. It is a control, not evidence about the defect: the
// baseline passes either way, so nothing here could change.
//
// The informative opt-out result is RatioTwoReadsPerAnnouncedSlotNoImplicitSync, which applies the same
// opt-out to a FAILING configuration and leaves it failing with bit-identical per-grant samples. That
// is what rules out implicit sync and separates this defect from the mechanisms in issue #50328.
//
// The opt-out is a diagnostic only, never a recommended configuration: it is a per-kernel hammer over
// every DFB the kernel binds, and ops are expected to rely on the Gen2 implicit-sync default.
TEST_F(Gen2DFBCreditsTest, ConsumerDoesNotRunAheadOfProducerNoImplicitSync) {
    auto mesh_device = devices_.at(0);
    IDevice* device = mesh_device->get_devices()[0];

    const uint32_t words_per_entry = kEntrySize / sizeof(uint32_t);
    const std::vector<uint32_t> input = make_input(kNumTransfers, kEntrySize);

    for (uint32_t depth : {2u, 4u}) {
        const std::vector<uint32_t> output =
            run_dm_to_dm_loopback(*mesh_device, device, depth, /*disable_implicit_sync=*/true, input, kEntrySize);
        ASSERT_EQ(output.size(), input.size()) << "depth=" << depth;
        EXPECT_EQ(leading_mismatch_entries(input, output, words_per_entry), 0u) << "depth=" << depth;
    }
}

}  // namespace
}  // namespace tt::tt_metal::experimental
