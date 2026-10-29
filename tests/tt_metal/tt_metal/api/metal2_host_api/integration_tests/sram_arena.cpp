// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fast-dispatch acceptance tests for lockstep and range-lockstep SRAMArena allocation overrides on WH/BH.
// These expect working arena allocation placement; unsupported construction is a test failure.

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/experimental/sram_arena/sram_arena.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/tensor/host_tensor.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>

#include "command_queue_fixture.hpp"
#include "impl/sram_arena/sram_arena_impl.hpp"
#include "impl/dispatch/slow_dispatch.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalDFB;
using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalWorkUnit;

class SRAMArenaTest : public UnitMeshCQSingleCardFixture {
protected:
    void SetUp() override {
        UnitMeshCQSingleCardFixture::SetUp();
        if (HasFatalFailure() || IsSkipped()) {
            return;
        }
        mesh_device = devices_.at(0).get();
        if (mesh_device->arch() != tt::ARCH::WORMHOLE_B0 && mesh_device->arch() != tt::ARCH::BLACKHOLE) {
            GTEST_SKIP() << "Requires Wormhole B0 or Blackhole hardware";
        }
        alignment = mesh_device->allocator()->get_alignment(BufferType::L1);
    }

    distributed::MeshDevice* mesh_device = nullptr;
    uint32_t alignment = 0;
};

using SRAMArenaRangeLockstep = SRAMArenaTest;
using SRAMArenaLockstep = SRAMArenaTest;

constexpr uint32_t kSentinel = 0xDEADBEEFu;
constexpr uint32_t kScratchPattern = 0x13570000u;
constexpr uint32_t kDFBPattern = 0x24680000u;
constexpr uint32_t kNodePatternStride = 0x10000u;
constexpr uint32_t kDFBDepth = 2;

TensorSpec ReportSpec(const NodeRangeSet& nodes, uint32_t alignment) {
    const uint32_t words = tt::round_up(3 * sizeof(uint32_t), alignment) / sizeof(uint32_t);
    return TensorSpec(
        Shape{nodes.num_cores(), words},
        TensorLayout(
            DataType::UINT32,
            PageConfig(Layout::ROW_MAJOR),
            MemoryConfig{
                TensorMemoryLayout::HEIGHT_SHARDED,
                BufferType::L1,
                ShardSpec{nodes, {1, words}, ShardOrientation::ROW_MAJOR}}));
}

TensorSpec OutputSpec(uint32_t num_nodes, uint32_t entry_size) {
    // Each row is one DFB entry; each node writes its own pair of output pages.
    return TensorSpec(
        Shape{num_nodes * kDFBDepth, entry_size / sizeof(uint32_t)},
        TensorLayout(
            DataType::UINT32,
            PageConfig(Layout::ROW_MAJOR),
            MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM}));
}

MeshTensor CreateSentinelTensor(distributed::MeshDevice& device, const TensorSpec& spec) {
    auto tensor = MeshTensor::allocate_on_device(device, spec);
    auto& cq = device.mesh_command_queue();
    auto host = HostTensor::from_vector(std::vector<uint32_t>(spec.logical_shape().volume(), kSentinel), spec);
    cq.enqueue_write_tensor(host, tensor);
    distributed::Finish(cq);
    return tensor;
}

std::vector<uint32_t> Pattern(uint32_t bytes, uint32_t seed) {
    std::vector<uint32_t> words(bytes / sizeof(uint32_t));
    std::iota(words.begin(), words.end(), seed);
    return words;
}

void PutPattern(std::vector<uint32_t>& expected, uint32_t offset, uint32_t bytes, uint32_t seed) {
    auto pattern = Pattern(bytes, seed);
    std::copy(pattern.begin(), pattern.end(), expected.begin() + offset / sizeof(uint32_t));
}

std::vector<uint32_t> ReadArena(distributed::MeshDevice& device, NodeCoord node, uint32_t base, uint32_t bytes) {
    std::vector<uint32_t> words;
    EXPECT_TRUE(slow_dispatch::ReadFromL1(device, node, base, bytes, words));
    return words;
}

void Execute(distributed::MeshDevice& device, Program program) {
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(device.shape()), std::move(program));
    distributed::EnqueueMeshWorkload(device.mesh_command_queue(), workload, /*blocking=*/true);
}

// One report row per node. Unwritten fields and alignment padding must retain their sentinel.
struct ReportRow {
    uint32_t scratchpad = kSentinel;
    uint32_t dfb_write = kSentinel;
    uint32_t dfb_read = kSentinel;
};

void ExpectReport(distributed::MeshCommandQueue& cq, const MeshTensor& report, std::initializer_list<ReportRow> rows) {
    ASSERT_EQ(rows.size(), report.logical_shape()[0]);
    const auto row_words = report.logical_shape()[1];
    std::vector<uint32_t> expected(report.logical_shape().volume(), kSentinel);
    uint32_t offset = 0;
    for (const auto& row : rows) {
        expected[offset] = row.scratchpad;
        expected[offset + 1] = row.dfb_write;
        expected[offset + 2] = row.dfb_read;
        offset += row_words;
    }
    EXPECT_EQ(cq.enqueue_read_tensor(report).to_vector<uint32_t>(), expected)
        << "Kernel-reported arena allocation addresses or untouched report fields differ";
}

KernelSpec ScratchpadWriter() {
    auto kernel = MakeMinimalGen1DMKernel("writer", DataMovementProcessor::RISCV_0);
    kernel.source = "tests/tt_metal/tt_metal/test_kernels/dataflow/sram_arena_scratchpad.cpp";
    kernel.runtime_arg_schema.runtime_arg_names = {"scratch_seed"};
    kernel.tensor_bindings = {{.tensor_parameter_name = TensorParamName{"report"}, .accessor_name = "report"}};
    kernel.scratchpad_bindings = {{.scratchpad_spec_name = ScratchpadSpecName{"pad"}, .accessor_name = "pad"}};
    return kernel;
}

enum class AllocationMode { Lockstep, RangeLockstep };

class SRAMArenaScratchpadUpdates : public SRAMArenaTest, public ::testing::WithParamInterface<AllocationMode> {
protected:
    void SetUp() override {
        SRAMArenaTest::SetUp();
        if (HasFatalFailure() || IsSkipped()) {
            return;
        }
        const NodeRangeSet nodes{NodeRange(NodeCoord{0, 0})};
        const uint32_t reservation_bytes = 4 * alignment;
        const SRAMArenaProperties properties =
            GetParam() == AllocationMode::Lockstep
                ? SRAMArenaProperties{LockstepSRAMArenaProperties{.per_bank_size = reservation_bytes}}
                : SRAMArenaProperties{RangeLockstepSRAMArenaProperties{
                      .banks_to_be_allocated_on = nodes, .per_bank_size = reservation_bytes}};
        const auto report_spec = ReportSpec(nodes, alignment);
        ProgramSpec spec;
        spec.name = "SRAMArenaScratchpadUpdates";
        spec.kernels = {ScratchpadWriter()};
        spec.advanced_options.sram_arena_parameters = {
            {.unique_id = SRAMArenaParamName{"arena"}, .properties = properties}};
        spec.scratchpads = {{.unique_id = ScratchpadSpecName{"pad"}, .size_per_node = 2 * alignment}};
        spec.scratchpads[0].advanced_options.allocation_override =
            SRAMArenaAllocation{.arena = SRAMArenaParamName{"arena"}, .offset = alignment};
        spec.tensor_parameters = {{.unique_id = TensorParamName{"report"}, .spec = report_spec}};
        spec.work_units = {MakeMinimalWorkUnit("work", nodes, {"writer"})};
        workload.add_program(
            distributed::MeshCoordinateRange(mesh_device->shape()), MakeProgramFromSpec(*mesh_device, spec));
        initial.emplace(
            std::visit([&](const auto& props) { return SRAMArena::allocate(*mesh_device, props); }, properties));
        replacement.emplace(
            std::visit([&](const auto& props) { return SRAMArena::allocate(*mesh_device, props); }, properties));
        report.emplace(CreateSentinelTensor(*mesh_device, report_spec));
        std::vector<uint32_t> sentinel(reservation_bytes / sizeof(uint32_t), kSentinel);
        ASSERT_TRUE(slow_dispatch::WriteToL1(*mesh_device, NodeCoord{0, 0}, Base(*initial), sentinel));
        ASSERT_TRUE(slow_dispatch::WriteToL1(*mesh_device, NodeCoord{0, 0}, Base(*replacement), sentinel));
    }

    void TearDown() override {
        workload = distributed::MeshWorkload{};
        report.reset();
        replacement.reset();
        initial.reset();
        SRAMArenaTest::TearDown();
    }

    static uint32_t Base(const SRAMArena& allocation) {
        return std::visit(
            [](const auto& layout) { return static_cast<uint32_t>(layout.addr); }, allocation.impl().layout);
    }

    Program& program() { return workload.get_programs().begin()->second; }

    ProgramRunArgs Args(const SRAMArena& allocation, uint32_t seed = kScratchPattern) {
        ProgramRunArgs args;
        args.advanced_options.sram_arena_args = {{SRAMArenaParamName{"arena"}, std::cref(allocation)}};
        args.tensor_args = {{TensorParamName{"report"}, TensorArgument{*report}}};
        args.kernel_run_args = {
            {.kernel = KernelSpecName{"writer"},
             .runtime_arg_values = MakeRuntimeArgsForSingleNode(NodeCoord{0, 0}, {{"scratch_seed", seed}})}};
        return args;
    }

    void Enqueue() { distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, /*blocking=*/true); }

    uint32_t ReportedAddress() {
        return mesh_device->mesh_command_queue().enqueue_read_tensor(*report).to_vector<uint32_t>().at(0);
    }

    std::vector<uint32_t> ReadReservation(const SRAMArena& allocation) {
        return ReadArena(*mesh_device, NodeCoord{0, 0}, Base(allocation), 4 * alignment);
    }

    std::vector<uint32_t> ExpectedReservation(uint32_t seed) {
        std::vector<uint32_t> expected(4 * alignment / sizeof(uint32_t), kSentinel);
        PutPattern(expected, alignment, 2 * alignment, seed);
        return expected;
    }

    std::optional<SRAMArena> initial;
    std::optional<SRAMArena> replacement;
    std::optional<MeshTensor> report;
    distributed::MeshWorkload workload;
};

TEST_P(SRAMArenaScratchpadUpdates, FullResetRebindsScratchpad) {
    // A full argument reset moves scratchpad writes to the replacement reservation.
    SetProgramRunArgs(program(), Args(*initial));
    Enqueue();
    SetProgramRunArgs(program(), Args(*replacement, kScratchPattern + kNodePatternStride));
    Enqueue();
    EXPECT_EQ(ReportedAddress(), Base(*replacement) + alignment);
    EXPECT_EQ(ReadReservation(*replacement), ExpectedReservation(kScratchPattern + kNodePatternStride));
    EXPECT_EQ(ReadReservation(*initial), ExpectedReservation(kScratchPattern));
}

TEST_P(SRAMArenaScratchpadUpdates, PartialUpdateRebindsScratchpad) {
    // An allocation-only update moves scratchpad writes without resupplying kernel arguments.
    SetProgramRunArgs(program(), Args(*initial));
    Enqueue();
    ProgramRunArgs update;
    update.advanced_options.sram_arena_args = {{SRAMArenaParamName{"arena"}, std::cref(*replacement)}};
    UpdateProgramRunArgs(program(), update);
    Enqueue();
    EXPECT_EQ(ReportedAddress(), Base(*replacement) + alignment);
    EXPECT_EQ(ReadReservation(*replacement), ExpectedReservation(kScratchPattern));
    EXPECT_EQ(ReadReservation(*initial), ExpectedReservation(kScratchPattern));
}

TEST_P(SRAMArenaScratchpadUpdates, PartialUpdateRetainsOmittedAllocation) {
    // Updating a kernel argument preserves the omitted scratchpad allocation binding.
    SetProgramRunArgs(program(), Args(*initial));
    Enqueue();
    ProgramRunArgs update;
    update.kernel_run_args = Args(*initial, kScratchPattern + kNodePatternStride).kernel_run_args;
    UpdateProgramRunArgs(program(), update);
    Enqueue();
    EXPECT_EQ(ReportedAddress(), Base(*initial) + alignment);
    EXPECT_EQ(ReadReservation(*initial), ExpectedReservation(kScratchPattern + kNodePatternStride));
}

INSTANTIATE_TEST_SUITE_P(
    AllocationModes,
    SRAMArenaScratchpadUpdates,
    ::testing::Values(AllocationMode::Lockstep, AllocationMode::RangeLockstep),
    [](const ::testing::TestParamInfo<AllocationMode>& info) {
        return info.param == AllocationMode::Lockstep ? "Lockstep" : "RangeLockstep";
    });

KernelSpec DFBProducer(bool writes_scratchpad = false) {
    auto kernel = MakeMinimalGen1DMKernel("producer", DataMovementProcessor::RISCV_0);
    kernel.source = writes_scratchpad
                        ? "tests/tt_metal/tt_metal/test_kernels/dataflow/sram_arena_scratchpad_and_dfb_producer.cpp"
                        : "tests/tt_metal/tt_metal/test_kernels/dataflow/sram_arena_dfb_producer.cpp";
    kernel.runtime_arg_schema.runtime_arg_names = {"dfb_seed"};
    if (writes_scratchpad) {
        kernel.runtime_arg_schema.runtime_arg_names.push_back("scratch_seed");
        kernel.scratchpad_bindings = {{.scratchpad_spec_name = ScratchpadSpecName{"pad"}, .accessor_name = "pad"}};
    }

    kernel.tensor_bindings = {{.tensor_parameter_name = TensorParamName{"report"}, .accessor_name = "report"}};
    kernel.dfb_bindings = {ProducerOf(DFBSpecName{"fifo"}, "fifo")};
    return kernel;
}

KernelSpec DFBConsumer() {
    auto kernel = MakeMinimalGen1DMKernel("consumer", DataMovementProcessor::RISCV_1);
    kernel.source = "tests/tt_metal/tt_metal/test_kernels/dataflow/sram_arena_dfb_consumer.cpp";
    kernel.runtime_arg_schema.runtime_arg_names = {"output_page_start"};
    kernel.tensor_bindings = {
        {.tensor_parameter_name = TensorParamName{"report"}, .accessor_name = "report"},
        {.tensor_parameter_name = TensorParamName{"output"}, .accessor_name = "output"}};
    kernel.dfb_bindings = {ConsumerOf(DFBSpecName{"fifo"}, "fifo")};
    return kernel;
}

class SRAMArenaDFBUpdates : public SRAMArenaTest, public ::testing::WithParamInterface<AllocationMode> {
protected:
    void SetUp() override {
        SRAMArenaTest::SetUp();
        if (HasFatalFailure() || IsSkipped()) {
            return;
        }
        const NodeRangeSet nodes{NodeRange(NodeCoord{0, 0})};
        fixture_entry_size = tt::round_up(1024, alignment);
        fixture_offset = alignment;
        fixture_reservation_bytes = 2 * fixture_entry_size + 2 * alignment;
        const SRAMArenaProperties properties =
            GetParam() == AllocationMode::Lockstep
                ? SRAMArenaProperties{LockstepSRAMArenaProperties{.per_bank_size = fixture_reservation_bytes}}
                : SRAMArenaProperties{RangeLockstepSRAMArenaProperties{
                      .banks_to_be_allocated_on = nodes, .per_bank_size = fixture_reservation_bytes}};
        const TensorSpec report_spec = ReportSpec(nodes, alignment);
        const TensorSpec output_spec = OutputSpec(1, fixture_entry_size);
        ProgramSpec spec;
        spec.name = "SRAMArenaDFBUpdates";
        spec.kernels = {DFBProducer(), DFBConsumer()};
        spec.advanced_options.sram_arena_parameters = {
            {.unique_id = SRAMArenaParamName{"arena"}, .properties = properties}};
        spec.dataflow_buffers = {MakeMinimalDFB("fifo", fixture_entry_size, kDFBDepth)};
        spec.dataflow_buffers[0].data_format_metadata = tt::DataFormat::Float16_b;
        spec.dataflow_buffers[0].advanced_options.allocation_override =
            SRAMArenaAllocation{.arena = SRAMArenaParamName{"arena"}, .offset = fixture_offset};
        spec.tensor_parameters = {
            {.unique_id = TensorParamName{"report"}, .spec = report_spec},
            {.unique_id = TensorParamName{"output"}, .spec = output_spec}};
        spec.work_units = {MakeMinimalWorkUnit("work", nodes, {"producer", "consumer"})};
        fixture_workload.add_program(
            distributed::MeshCoordinateRange(mesh_device->shape()), MakeProgramFromSpec(*mesh_device, spec));
        fixture_initial.emplace(
            std::visit([&](const auto& props) { return SRAMArena::allocate(*mesh_device, props); }, properties));
        fixture_replacement.emplace(
            std::visit([&](const auto& props) { return SRAMArena::allocate(*mesh_device, props); }, properties));
        fixture_report.emplace(CreateSentinelTensor(*mesh_device, report_spec));
        fixture_output.emplace(CreateSentinelTensor(*mesh_device, output_spec));
        std::vector<uint32_t> sentinel(fixture_reservation_bytes / sizeof(uint32_t), kSentinel);
        ASSERT_TRUE(slow_dispatch::WriteToL1(*mesh_device, NodeCoord{0, 0}, Base(*fixture_initial), sentinel));
        ASSERT_TRUE(slow_dispatch::WriteToL1(*mesh_device, NodeCoord{0, 0}, Base(*fixture_replacement), sentinel));
    }

    void TearDown() override {
        fixture_workload = distributed::MeshWorkload{};
        fixture_output.reset();
        fixture_report.reset();
        fixture_replacement.reset();
        fixture_initial.reset();
        SRAMArenaTest::TearDown();
    }

    static uint32_t Base(const SRAMArena& allocation) {
        return std::visit(
            [](const auto& layout) { return static_cast<uint32_t>(layout.addr); }, allocation.impl().layout);
    }

    Program& fixture_program() { return fixture_workload.get_programs().begin()->second; }

    ProgramRunArgs Args(const SRAMArena& allocation, uint32_t seed) {
        ProgramRunArgs args;
        args.advanced_options.sram_arena_args = {{SRAMArenaParamName{"arena"}, std::cref(allocation)}};
        args.tensor_args = {
            {TensorParamName{"report"}, TensorArgument{*fixture_report}},
            {TensorParamName{"output"}, TensorArgument{*fixture_output}}};
        args.kernel_run_args = {
            {.kernel = KernelSpecName{"producer"},
             .runtime_arg_values = MakeRuntimeArgsForSingleNode(NodeCoord{0, 0}, {{"dfb_seed", seed}})},
            {.kernel = KernelSpecName{"consumer"},
             .runtime_arg_values = MakeRuntimeArgsForSingleNode(NodeCoord{0, 0}, {{"output_page_start", 0u}})}};
        return args;
    }

    void Enqueue() {
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), fixture_workload, /*blocking=*/true);
    }

    std::vector<uint32_t> ReadTensor(const MeshTensor& tensor) {
        return mesh_device->mesh_command_queue().enqueue_read_tensor(tensor).to_vector<uint32_t>();
    }

    std::vector<uint32_t> ReadReservation(const SRAMArena& allocation) {
        return ReadArena(*mesh_device, NodeCoord{0, 0}, Base(allocation), fixture_reservation_bytes);
    }

    std::vector<uint32_t> ExpectedReport(uint32_t address) {
        std::vector<uint32_t> expected(fixture_report->logical_shape().volume(), kSentinel);
        expected[1] = address;
        expected[2] = address;
        return expected;
    }

    std::vector<uint32_t> ExpectedReservation(uint32_t seed) {
        std::vector<uint32_t> expected(fixture_reservation_bytes / sizeof(uint32_t), kSentinel);
        PutPattern(expected, fixture_offset, kDFBDepth * fixture_entry_size, seed);
        return expected;
    }

    uint32_t fixture_entry_size = 0;
    uint32_t fixture_offset = 0;
    uint32_t fixture_reservation_bytes = 0;
    std::optional<SRAMArena> fixture_initial;
    std::optional<SRAMArena> fixture_replacement;
    std::optional<MeshTensor> fixture_report;
    std::optional<MeshTensor> fixture_output;
    distributed::MeshWorkload fixture_workload;
};

TEST_P(SRAMArenaDFBUpdates, FullResetRebindsDFB) {
    // A full reset moves DFB transfers to the replacement reservation and preserves the first reservation.
    SetProgramRunArgs(fixture_program(), Args(*fixture_initial, kDFBPattern));
    Enqueue();
    const uint32_t next_seed = kDFBPattern + kNodePatternStride;
    SetProgramRunArgs(fixture_program(), Args(*fixture_replacement, next_seed));
    Enqueue();
    EXPECT_EQ(ReadTensor(*fixture_report), ExpectedReport(Base(*fixture_replacement) + fixture_offset));
    EXPECT_EQ(ReadTensor(*fixture_output), Pattern(kDFBDepth * fixture_entry_size, next_seed));
    EXPECT_EQ(ReadReservation(*fixture_replacement), ExpectedReservation(next_seed));
    EXPECT_EQ(ReadReservation(*fixture_initial), ExpectedReservation(kDFBPattern));
}

TEST_P(SRAMArenaDFBUpdates, PartialUpdateRebindsDFB) {
    // An allocation-only update moves DFB transfers without supplying tensor or kernel arguments again.
    SetProgramRunArgs(fixture_program(), Args(*fixture_initial, kDFBPattern));
    Enqueue();
    ProgramRunArgs update;
    update.advanced_options.sram_arena_args = {{SRAMArenaParamName{"arena"}, std::cref(*fixture_replacement)}};
    UpdateProgramRunArgs(fixture_program(), update);
    Enqueue();
    EXPECT_EQ(ReadTensor(*fixture_report), ExpectedReport(Base(*fixture_replacement) + fixture_offset));
    EXPECT_EQ(ReadTensor(*fixture_output), Pattern(kDFBDepth * fixture_entry_size, kDFBPattern));
    EXPECT_EQ(ReadReservation(*fixture_replacement), ExpectedReservation(kDFBPattern));
    EXPECT_EQ(ReadReservation(*fixture_initial), ExpectedReservation(kDFBPattern));
}

TEST_P(SRAMArenaDFBUpdates, PartialUpdateRetainsOmittedAllocation) {
    // Updating kernel arguments keeps DFB transfers in the previously supplied reservation.
    SetProgramRunArgs(fixture_program(), Args(*fixture_initial, kDFBPattern));
    Enqueue();
    const uint32_t next_seed = kDFBPattern + kNodePatternStride;
    ProgramRunArgs update;
    update.kernel_run_args = Args(*fixture_initial, next_seed).kernel_run_args;
    UpdateProgramRunArgs(fixture_program(), update);
    Enqueue();
    EXPECT_EQ(ReadTensor(*fixture_report), ExpectedReport(Base(*fixture_initial) + fixture_offset));
    EXPECT_EQ(ReadTensor(*fixture_output), Pattern(kDFBDepth * fixture_entry_size, next_seed));
    EXPECT_EQ(ReadReservation(*fixture_initial), ExpectedReservation(next_seed));
}

INSTANTIATE_TEST_SUITE_P(
    AllocationModes,
    SRAMArenaDFBUpdates,
    ::testing::Values(AllocationMode::Lockstep, AllocationMode::RangeLockstep),
    [](const ::testing::TestParamInfo<AllocationMode>& info) {
        return info.param == AllocationMode::Lockstep ? "Lockstep" : "RangeLockstep";
    });

TEST_F(SRAMArenaRangeLockstep, ScratchpadDefaultOffset) {
    // The scratchpad starts at the allocation’s base address, and its writes land there.
    const NodeRangeSet node_set{NodeRange(NodeCoord{0, 0})};
    const uint32_t scratch_bytes = 2 * alignment;
    const uint32_t offset = 0;
    const uint32_t reservation_bytes_per_bank =
        offset + scratch_bytes + alignment;  // Includes a trailing sentinel region.
    const RangeLockstepSRAMArenaProperties properties{
        .banks_to_be_allocated_on = node_set, .per_bank_size = reservation_bytes_per_bank};
    const auto report_spec = ReportSpec(node_set, alignment);

    ProgramSpec spec;
    spec.name = "SRAMArenaScratchpadDefaultOffset";
    spec.kernels = {ScratchpadWriter()};
    spec.advanced_options.sram_arena_parameters = {
        {.unique_id = SRAMArenaParamName{"arena"}, .properties = properties}};
    spec.scratchpads = {{.unique_id = ScratchpadSpecName{"pad"}, .size_per_node = scratch_bytes}};
    spec.scratchpads[0].advanced_options.allocation_override =
        SRAMArenaAllocation{.arena = SRAMArenaParamName{"arena"}};
    spec.tensor_parameters = {{.unique_id = TensorParamName{"report"}, .spec = report_spec}};
    spec.work_units = {MakeMinimalWorkUnit("work", node_set, {"writer"})};

    // Construct the program before supplying its reservation.
    auto program = MakeProgramFromSpec(*mesh_device, spec);
    auto arena = SRAMArena::allocate(*mesh_device, properties);
    const auto base = static_cast<uint32_t>(std::get<RangeLockstepSRAMArena>(arena.impl().layout).addr);
    auto report = CreateSentinelTensor(*mesh_device, report_spec);
    auto& cq = mesh_device->mesh_command_queue();
    std::vector<uint32_t> initial_arena(reservation_bytes_per_bank / sizeof(uint32_t), kSentinel);
    ASSERT_TRUE(slow_dispatch::WriteToL1(*mesh_device, NodeCoord{0, 0}, base, initial_arena));

    ProgramRunArgs args;
    args.advanced_options.sram_arena_args = {{SRAMArenaParamName{"arena"}, std::cref(arena)}};
    args.tensor_args = {{TensorParamName{"report"}, TensorArgument{report}}};
    args.kernel_run_args = {
        {.kernel = KernelSpecName{"writer"},
         .runtime_arg_values = MakeRuntimeArgsForSingleNode(NodeCoord{0, 0}, {{"scratch_seed", kScratchPattern}})}};
    SetProgramRunArgs(program, args);
    Execute(*mesh_device, std::move(program));

    auto expected_arena = initial_arena;
    PutPattern(expected_arena, offset, scratch_bytes, kScratchPattern);
    EXPECT_EQ(ReadArena(*mesh_device, NodeCoord{0, 0}, base, reservation_bytes_per_bank), expected_arena)
        << "Reserved SRAM contents or sentinel regions differ";
    ExpectReport(cq, report, {{.scratchpad = base + offset}});
}

TEST_F(SRAMArenaRangeLockstep, DFBDefaultOffset) {
    // The DFB starts at the allocation’s base address, and producer-to-consumer data transfer works.
    const NodeRangeSet node_set{NodeRange(NodeCoord{0, 0})};
    const uint32_t entry_size = tt::round_up(1024, alignment);
    const uint32_t dfb_bytes = kDFBDepth * entry_size;
    const uint32_t offset = 0;
    const uint32_t reservation_bytes_per_bank = offset + dfb_bytes + alignment;  // Includes a trailing sentinel region.
    const RangeLockstepSRAMArenaProperties properties{
        .banks_to_be_allocated_on = node_set, .per_bank_size = reservation_bytes_per_bank};
    const auto report_spec = ReportSpec(node_set, alignment);
    const auto output_spec = OutputSpec(node_set.num_cores(), entry_size);

    ProgramSpec spec;
    spec.name = "SRAMArenaDFBDefaultOffset";
    spec.kernels = {DFBProducer(), DFBConsumer()};
    spec.advanced_options.sram_arena_parameters = {
        {.unique_id = SRAMArenaParamName{"arena"}, .properties = properties}};
    spec.dataflow_buffers = {MakeMinimalDFB("fifo", entry_size, kDFBDepth)};
    spec.dataflow_buffers[0].data_format_metadata = tt::DataFormat::Float16_b;
    spec.dataflow_buffers[0].advanced_options.allocation_override =
        SRAMArenaAllocation{.arena = SRAMArenaParamName{"arena"}};
    spec.tensor_parameters = {
        {.unique_id = TensorParamName{"report"}, .spec = report_spec},
        {.unique_id = TensorParamName{"output"}, .spec = output_spec}};
    spec.work_units = {MakeMinimalWorkUnit("work", node_set, {"producer", "consumer"})};

    // Construct the program before supplying its reservation.
    auto program = MakeProgramFromSpec(*mesh_device, spec);
    auto arena = SRAMArena::allocate(*mesh_device, properties);
    const auto base = static_cast<uint32_t>(std::get<RangeLockstepSRAMArena>(arena.impl().layout).addr);
    auto report = CreateSentinelTensor(*mesh_device, report_spec);
    auto output = CreateSentinelTensor(*mesh_device, output_spec);
    auto& cq = mesh_device->mesh_command_queue();
    std::vector<uint32_t> initial_arena(reservation_bytes_per_bank / sizeof(uint32_t), kSentinel);
    ASSERT_TRUE(slow_dispatch::WriteToL1(*mesh_device, NodeCoord{0, 0}, base, initial_arena));

    ProgramRunArgs args;
    args.advanced_options.sram_arena_args = {{SRAMArenaParamName{"arena"}, std::cref(arena)}};
    args.tensor_args = {
        {TensorParamName{"report"}, TensorArgument{report}}, {TensorParamName{"output"}, TensorArgument{output}}};
    args.kernel_run_args = {
        {.kernel = KernelSpecName{"producer"},
         .runtime_arg_values = MakeRuntimeArgsForSingleNode(NodeCoord{0, 0}, {{"dfb_seed", kDFBPattern}})},
        {.kernel = KernelSpecName{"consumer"},
         .runtime_arg_values = MakeRuntimeArgsForSingleNode(NodeCoord{0, 0}, {{"output_page_start", 0u}})}};
    SetProgramRunArgs(program, args);
    Execute(*mesh_device, std::move(program));

    EXPECT_EQ(cq.enqueue_read_tensor(output).to_vector<uint32_t>(), Pattern(dfb_bytes, kDFBPattern))
        << "DFB consumer output differs";
    auto expected_arena = initial_arena;
    PutPattern(expected_arena, offset, dfb_bytes, kDFBPattern);
    EXPECT_EQ(ReadArena(*mesh_device, NodeCoord{0, 0}, base, reservation_bytes_per_bank), expected_arena)
        << "Reserved SRAM contents or sentinel regions differ";
    ExpectReport(cq, report, {{.dfb_write = base + offset, .dfb_read = base + offset}});
}

TEST_F(SRAMArenaRangeLockstep, ScratchpadNonzeroOffset) {
    // The scratchpad starts at `base + offset`, and its writes land there.
    const NodeRangeSet node_set{NodeRange(NodeCoord{0, 0})};
    const uint32_t scratch_bytes = 2 * alignment;
    const uint32_t offset = alignment;
    const uint32_t reservation_bytes_per_bank =
        offset + scratch_bytes + alignment;  // Includes a trailing sentinel region.
    const RangeLockstepSRAMArenaProperties properties{
        .banks_to_be_allocated_on = node_set, .per_bank_size = reservation_bytes_per_bank};
    const auto report_spec = ReportSpec(node_set, alignment);

    ProgramSpec spec;
    spec.name = "SRAMArenaScratchpadNonzeroOffset";
    spec.kernels = {ScratchpadWriter()};
    spec.advanced_options.sram_arena_parameters = {
        {.unique_id = SRAMArenaParamName{"arena"}, .properties = properties}};
    spec.scratchpads = {{.unique_id = ScratchpadSpecName{"pad"}, .size_per_node = scratch_bytes}};
    spec.scratchpads[0].advanced_options.allocation_override =
        SRAMArenaAllocation{.arena = SRAMArenaParamName{"arena"}, .offset = offset};
    spec.tensor_parameters = {{.unique_id = TensorParamName{"report"}, .spec = report_spec}};
    spec.work_units = {MakeMinimalWorkUnit("work", node_set, {"writer"})};

    // Construct the program before supplying its reservation.
    auto program = MakeProgramFromSpec(*mesh_device, spec);
    auto arena = SRAMArena::allocate(*mesh_device, properties);
    const auto base = static_cast<uint32_t>(std::get<RangeLockstepSRAMArena>(arena.impl().layout).addr);
    auto report = CreateSentinelTensor(*mesh_device, report_spec);
    auto& cq = mesh_device->mesh_command_queue();
    std::vector<uint32_t> initial_arena(reservation_bytes_per_bank / sizeof(uint32_t), kSentinel);
    ASSERT_TRUE(slow_dispatch::WriteToL1(*mesh_device, NodeCoord{0, 0}, base, initial_arena));

    ProgramRunArgs args;
    args.advanced_options.sram_arena_args = {{SRAMArenaParamName{"arena"}, std::cref(arena)}};
    args.tensor_args = {{TensorParamName{"report"}, TensorArgument{report}}};
    args.kernel_run_args = {
        {.kernel = KernelSpecName{"writer"},
         .runtime_arg_values = MakeRuntimeArgsForSingleNode(NodeCoord{0, 0}, {{"scratch_seed", kScratchPattern}})}};
    SetProgramRunArgs(program, args);
    Execute(*mesh_device, std::move(program));

    auto expected_arena = initial_arena;
    PutPattern(expected_arena, offset, scratch_bytes, kScratchPattern);
    EXPECT_EQ(ReadArena(*mesh_device, NodeCoord{0, 0}, base, reservation_bytes_per_bank), expected_arena)
        << "Reserved SRAM contents or sentinel regions differ";
    ExpectReport(cq, report, {{.scratchpad = base + offset}});
}

TEST_F(SRAMArenaRangeLockstep, DFBNonzeroOffset) {
    // The DFB starts at `base + offset`, and producer-to-consumer data transfer works.
    const NodeRangeSet node_set{NodeRange(NodeCoord{0, 0})};
    const uint32_t entry_size = tt::round_up(1024, alignment);
    const uint32_t dfb_bytes = kDFBDepth * entry_size;
    const uint32_t offset = alignment;
    const uint32_t reservation_bytes_per_bank = offset + dfb_bytes + alignment;  // Includes a trailing sentinel region.
    const RangeLockstepSRAMArenaProperties properties{
        .banks_to_be_allocated_on = node_set, .per_bank_size = reservation_bytes_per_bank};
    const auto report_spec = ReportSpec(node_set, alignment);
    const auto output_spec = OutputSpec(node_set.num_cores(), entry_size);

    ProgramSpec spec;
    spec.name = "SRAMArenaDFBNonzeroOffset";
    spec.kernels = {DFBProducer(), DFBConsumer()};
    spec.advanced_options.sram_arena_parameters = {
        {.unique_id = SRAMArenaParamName{"arena"}, .properties = properties}};
    spec.dataflow_buffers = {MakeMinimalDFB("fifo", entry_size, kDFBDepth)};
    spec.dataflow_buffers[0].data_format_metadata = tt::DataFormat::Float16_b;
    spec.dataflow_buffers[0].advanced_options.allocation_override =
        SRAMArenaAllocation{.arena = SRAMArenaParamName{"arena"}, .offset = offset};
    spec.tensor_parameters = {
        {.unique_id = TensorParamName{"report"}, .spec = report_spec},
        {.unique_id = TensorParamName{"output"}, .spec = output_spec}};
    spec.work_units = {MakeMinimalWorkUnit("work", node_set, {"producer", "consumer"})};

    // Construct the program before supplying its reservation.
    auto program = MakeProgramFromSpec(*mesh_device, spec);
    auto arena = SRAMArena::allocate(*mesh_device, properties);
    const auto base = static_cast<uint32_t>(std::get<RangeLockstepSRAMArena>(arena.impl().layout).addr);
    auto report = CreateSentinelTensor(*mesh_device, report_spec);
    auto output = CreateSentinelTensor(*mesh_device, output_spec);
    auto& cq = mesh_device->mesh_command_queue();
    std::vector<uint32_t> initial_arena(reservation_bytes_per_bank / sizeof(uint32_t), kSentinel);
    ASSERT_TRUE(slow_dispatch::WriteToL1(*mesh_device, NodeCoord{0, 0}, base, initial_arena));

    ProgramRunArgs args;
    args.advanced_options.sram_arena_args = {{SRAMArenaParamName{"arena"}, std::cref(arena)}};
    args.tensor_args = {
        {TensorParamName{"report"}, TensorArgument{report}}, {TensorParamName{"output"}, TensorArgument{output}}};
    args.kernel_run_args = {
        {.kernel = KernelSpecName{"producer"},
         .runtime_arg_values = MakeRuntimeArgsForSingleNode(NodeCoord{0, 0}, {{"dfb_seed", kDFBPattern}})},
        {.kernel = KernelSpecName{"consumer"},
         .runtime_arg_values = MakeRuntimeArgsForSingleNode(NodeCoord{0, 0}, {{"output_page_start", 0u}})}};
    SetProgramRunArgs(program, args);
    Execute(*mesh_device, std::move(program));

    EXPECT_EQ(cq.enqueue_read_tensor(output).to_vector<uint32_t>(), Pattern(dfb_bytes, kDFBPattern))
        << "DFB consumer output differs";
    auto expected_arena = initial_arena;
    PutPattern(expected_arena, offset, dfb_bytes, kDFBPattern);
    EXPECT_EQ(ReadArena(*mesh_device, NodeCoord{0, 0}, base, reservation_bytes_per_bank), expected_arena)
        << "Reserved SRAM contents or sentinel regions differ";
    ExpectReport(cq, report, {{.dfb_write = base + offset, .dfb_read = base + offset}});
}

TEST_F(SRAMArenaRangeLockstep, BorrowingAcrossNodes) {
    // Each selected node receives the correct arena allocation addresses and produces its own expected data.
    const std::vector<NodeCoord> nodes{{0, 0}, {1, 0}};
    const NodeRangeSet node_set{NodeRange{NodeCoord{0, 0}, NodeCoord{1, 0}}};
    const uint32_t scratch_bytes = 2 * alignment;
    const uint32_t entry_size = tt::round_up(1024, alignment);
    const uint32_t dfb_bytes = kDFBDepth * entry_size;
    const uint32_t scratch_offset = alignment;
    const uint32_t dfb_offset = 4 * alignment;
    const uint32_t reservation_bytes_per_bank =
        dfb_offset + dfb_bytes + alignment;  // Includes a trailing sentinel region.
    const RangeLockstepSRAMArenaProperties properties{
        .banks_to_be_allocated_on = node_set, .per_bank_size = reservation_bytes_per_bank};
    const auto report_spec = ReportSpec(node_set, alignment);
    const auto output_spec = OutputSpec(node_set.num_cores(), entry_size);

    ProgramSpec spec;
    spec.name = "SRAMArenaAllocationAcrossNodes";
    spec.kernels = {DFBProducer(true), DFBConsumer()};
    spec.advanced_options.sram_arena_parameters = {
        {.unique_id = SRAMArenaParamName{"arena"}, .properties = properties}};
    spec.scratchpads = {{.unique_id = ScratchpadSpecName{"pad"}, .size_per_node = scratch_bytes}};
    spec.scratchpads[0].advanced_options.allocation_override =
        SRAMArenaAllocation{.arena = SRAMArenaParamName{"arena"}, .offset = scratch_offset};
    spec.dataflow_buffers = {MakeMinimalDFB("fifo", entry_size, kDFBDepth)};
    spec.dataflow_buffers[0].data_format_metadata = tt::DataFormat::Float16_b;
    spec.dataflow_buffers[0].advanced_options.allocation_override =
        SRAMArenaAllocation{.arena = SRAMArenaParamName{"arena"}, .offset = dfb_offset};
    spec.tensor_parameters = {
        {.unique_id = TensorParamName{"report"}, .spec = report_spec},
        {.unique_id = TensorParamName{"output"}, .spec = output_spec}};
    spec.work_units = {MakeMinimalWorkUnit("work", node_set, {"producer", "consumer"})};

    // Construct the program before supplying its reservation.
    auto program = MakeProgramFromSpec(*mesh_device, spec);
    auto arena = SRAMArena::allocate(*mesh_device, properties);
    const auto base = static_cast<uint32_t>(std::get<RangeLockstepSRAMArena>(arena.impl().layout).addr);
    auto report = CreateSentinelTensor(*mesh_device, report_spec);
    auto output = CreateSentinelTensor(*mesh_device, output_spec);
    auto& cq = mesh_device->mesh_command_queue();
    std::vector<uint32_t> initial_arena(reservation_bytes_per_bank / sizeof(uint32_t), kSentinel);
    for (const auto& node : nodes) {
        ASSERT_TRUE(slow_dispatch::WriteToL1(*mesh_device, node, base, initial_arena));
    }

    ProgramRunArgs args;
    args.advanced_options.sram_arena_args = {{SRAMArenaParamName{"arena"}, std::cref(arena)}};
    args.tensor_args = {
        {TensorParamName{"report"}, TensorArgument{report}}, {TensorParamName{"output"}, TensorArgument{output}}};
    ProgramRunArgs::KernelRunArgs producer_args{.kernel = KernelSpecName{"producer"}};
    ProgramRunArgs::KernelRunArgs consumer_args{.kernel = KernelSpecName{"consumer"}};
    for (uint32_t i = 0; i < nodes.size(); ++i) {
        producer_args.runtime_arg_values["scratch_seed"][nodes[i]] = kScratchPattern + i * kNodePatternStride;
        producer_args.runtime_arg_values["dfb_seed"][nodes[i]] = kDFBPattern + i * kNodePatternStride;
        consumer_args.runtime_arg_values["output_page_start"][nodes[i]] = i * kDFBDepth;
    }
    args.kernel_run_args = {producer_args, consumer_args};
    SetProgramRunArgs(program, args);
    Execute(*mesh_device, std::move(program));

    const auto output_words = cq.enqueue_read_tensor(output).to_vector<uint32_t>();
    std::vector<uint32_t> expected_output(output_spec.logical_shape().volume(), kSentinel);
    for (uint32_t i = 0; i < nodes.size(); ++i) {
        SCOPED_TRACE("node " + std::to_string(i));
        auto expected_arena = initial_arena;
        PutPattern(expected_arena, scratch_offset, scratch_bytes, kScratchPattern + i * kNodePatternStride);
        PutPattern(expected_arena, dfb_offset, dfb_bytes, kDFBPattern + i * kNodePatternStride);
        EXPECT_EQ(ReadArena(*mesh_device, nodes[i], base, reservation_bytes_per_bank), expected_arena)
            << "Reserved SRAM contents or sentinel regions differ";
        PutPattern(expected_output, i * dfb_bytes, dfb_bytes, kDFBPattern + i * kNodePatternStride);
    }
    EXPECT_EQ(output_words, expected_output) << "DFB consumer output differs";
    ExpectReport(
        cq,
        report,
        {
            {.scratchpad = base + scratch_offset, .dfb_write = base + dfb_offset, .dfb_read = base + dfb_offset},
            {.scratchpad = base + scratch_offset, .dfb_write = base + dfb_offset, .dfb_read = base + dfb_offset},
        });
}

TEST_F(SRAMArenaRangeLockstep, SharedByScratchpadAndDFB) {
    // Both resources use their assigned regions, with correct data and no corruption between them.
    const NodeRangeSet node_set{NodeRange(NodeCoord{0, 0})};
    const uint32_t scratch_bytes = 2 * alignment;
    const uint32_t entry_size = tt::round_up(1024, alignment);
    const uint32_t dfb_bytes = kDFBDepth * entry_size;
    const uint32_t scratch_offset = alignment;
    const uint32_t dfb_offset = 4 * alignment;
    const uint32_t reservation_bytes_per_bank =
        dfb_offset + dfb_bytes + alignment;  // Includes a trailing sentinel region.
    const RangeLockstepSRAMArenaProperties properties{
        .banks_to_be_allocated_on = node_set, .per_bank_size = reservation_bytes_per_bank};
    const auto report_spec = ReportSpec(node_set, alignment);
    const auto output_spec = OutputSpec(node_set.num_cores(), entry_size);

    ProgramSpec spec;
    spec.name = "SRAMArenaSharedByScratchpadAndDFB";
    spec.kernels = {DFBProducer(true), DFBConsumer()};
    spec.advanced_options.sram_arena_parameters = {
        {.unique_id = SRAMArenaParamName{"arena"}, .properties = properties}};
    spec.scratchpads = {{.unique_id = ScratchpadSpecName{"pad"}, .size_per_node = scratch_bytes}};
    spec.scratchpads[0].advanced_options.allocation_override =
        SRAMArenaAllocation{.arena = SRAMArenaParamName{"arena"}, .offset = scratch_offset};
    spec.dataflow_buffers = {MakeMinimalDFB("fifo", entry_size, kDFBDepth)};
    spec.dataflow_buffers[0].data_format_metadata = tt::DataFormat::Float16_b;
    spec.dataflow_buffers[0].advanced_options.allocation_override =
        SRAMArenaAllocation{.arena = SRAMArenaParamName{"arena"}, .offset = dfb_offset};
    spec.tensor_parameters = {
        {.unique_id = TensorParamName{"report"}, .spec = report_spec},
        {.unique_id = TensorParamName{"output"}, .spec = output_spec}};
    spec.work_units = {MakeMinimalWorkUnit("work", node_set, {"producer", "consumer"})};

    // Construct the program before supplying its reservation.
    auto program = MakeProgramFromSpec(*mesh_device, spec);
    auto arena = SRAMArena::allocate(*mesh_device, properties);
    const auto base = static_cast<uint32_t>(std::get<RangeLockstepSRAMArena>(arena.impl().layout).addr);
    auto report = CreateSentinelTensor(*mesh_device, report_spec);
    auto output = CreateSentinelTensor(*mesh_device, output_spec);
    auto& cq = mesh_device->mesh_command_queue();
    std::vector<uint32_t> initial_arena(reservation_bytes_per_bank / sizeof(uint32_t), kSentinel);
    ASSERT_TRUE(slow_dispatch::WriteToL1(*mesh_device, NodeCoord{0, 0}, base, initial_arena));

    ProgramRunArgs args;
    args.advanced_options.sram_arena_args = {{SRAMArenaParamName{"arena"}, std::cref(arena)}};
    args.tensor_args = {
        {TensorParamName{"report"}, TensorArgument{report}}, {TensorParamName{"output"}, TensorArgument{output}}};
    args.kernel_run_args = {
        {.kernel = KernelSpecName{"producer"},
         .runtime_arg_values = MakeRuntimeArgsForSingleNode(
             NodeCoord{0, 0}, {{"dfb_seed", kDFBPattern}, {"scratch_seed", kScratchPattern}})},
        {.kernel = KernelSpecName{"consumer"},
         .runtime_arg_values = MakeRuntimeArgsForSingleNode(NodeCoord{0, 0}, {{"output_page_start", 0u}})}};
    SetProgramRunArgs(program, args);
    Execute(*mesh_device, std::move(program));

    EXPECT_EQ(cq.enqueue_read_tensor(output).to_vector<uint32_t>(), Pattern(dfb_bytes, kDFBPattern))
        << "DFB consumer output differs";
    auto expected_arena = initial_arena;
    PutPattern(expected_arena, scratch_offset, scratch_bytes, kScratchPattern);
    PutPattern(expected_arena, dfb_offset, dfb_bytes, kDFBPattern);
    EXPECT_EQ(ReadArena(*mesh_device, NodeCoord{0, 0}, base, reservation_bytes_per_bank), expected_arena)
        << "Reserved SRAM contents or sentinel regions differ";
    ExpectReport(
        cq,
        report,
        {{.scratchpad = base + scratch_offset, .dfb_write = base + dfb_offset, .dfb_read = base + dfb_offset}});
}

TEST_F(SRAMArenaLockstep, ScratchpadDefaultOffset) {
    // The scratchpad starts at the allocation’s base address, and its writes land there.
    const NodeRangeSet node_set{NodeRange(NodeCoord{0, 0})};
    const uint32_t scratch_bytes = 2 * alignment;
    const uint32_t offset = 0;
    const uint32_t reservation_bytes_per_bank =
        offset + scratch_bytes + alignment;  // Includes a trailing sentinel region.
    const LockstepSRAMArenaProperties properties{.per_bank_size = reservation_bytes_per_bank};
    const auto report_spec = ReportSpec(node_set, alignment);

    ProgramSpec spec;
    spec.name = "SRAMArenaLockstepScratchpadDefaultOffset";
    spec.kernels = {ScratchpadWriter()};
    spec.advanced_options.sram_arena_parameters = {
        {.unique_id = SRAMArenaParamName{"arena"}, .properties = properties}};
    spec.scratchpads = {{.unique_id = ScratchpadSpecName{"pad"}, .size_per_node = scratch_bytes}};
    spec.scratchpads[0].advanced_options.allocation_override =
        SRAMArenaAllocation{.arena = SRAMArenaParamName{"arena"}};
    spec.tensor_parameters = {{.unique_id = TensorParamName{"report"}, .spec = report_spec}};
    spec.work_units = {MakeMinimalWorkUnit("work", node_set, {"writer"})};

    // Construct the program before supplying its reservation.
    auto program = MakeProgramFromSpec(*mesh_device, spec);
    auto arena = SRAMArena::allocate(*mesh_device, properties);
    const auto base = static_cast<uint32_t>(std::get<LockstepSRAMArena>(arena.impl().layout).addr);
    auto report = CreateSentinelTensor(*mesh_device, report_spec);
    auto& cq = mesh_device->mesh_command_queue();
    std::vector<uint32_t> initial_arena(reservation_bytes_per_bank / sizeof(uint32_t), kSentinel);
    ASSERT_TRUE(slow_dispatch::WriteToL1(*mesh_device, NodeCoord{0, 0}, base, initial_arena));

    ProgramRunArgs args;
    args.advanced_options.sram_arena_args = {{SRAMArenaParamName{"arena"}, std::cref(arena)}};
    args.tensor_args = {{TensorParamName{"report"}, TensorArgument{report}}};
    args.kernel_run_args = {
        {.kernel = KernelSpecName{"writer"},
         .runtime_arg_values = MakeRuntimeArgsForSingleNode(NodeCoord{0, 0}, {{"scratch_seed", kScratchPattern}})}};
    SetProgramRunArgs(program, args);
    Execute(*mesh_device, std::move(program));

    auto expected_arena = initial_arena;
    PutPattern(expected_arena, offset, scratch_bytes, kScratchPattern);
    EXPECT_EQ(ReadArena(*mesh_device, NodeCoord{0, 0}, base, reservation_bytes_per_bank), expected_arena)
        << "Reserved SRAM contents or sentinel regions differ";
    ExpectReport(cq, report, {{.scratchpad = base + offset}});
}

TEST_F(SRAMArenaLockstep, DFBDefaultOffset) {
    // The DFB starts at the allocation’s base address, and producer-to-consumer data transfer works.
    const NodeRangeSet node_set{NodeRange(NodeCoord{0, 0})};
    const uint32_t entry_size = tt::round_up(1024, alignment);
    const uint32_t dfb_bytes = kDFBDepth * entry_size;
    const uint32_t offset = 0;
    const uint32_t reservation_bytes_per_bank = offset + dfb_bytes + alignment;  // Includes a trailing sentinel region.
    const LockstepSRAMArenaProperties properties{.per_bank_size = reservation_bytes_per_bank};
    const auto report_spec = ReportSpec(node_set, alignment);
    const auto output_spec = OutputSpec(node_set.num_cores(), entry_size);

    ProgramSpec spec;
    spec.name = "SRAMArenaLockstepDFBDefaultOffset";
    spec.kernels = {DFBProducer(), DFBConsumer()};
    spec.advanced_options.sram_arena_parameters = {
        {.unique_id = SRAMArenaParamName{"arena"}, .properties = properties}};
    spec.dataflow_buffers = {MakeMinimalDFB("fifo", entry_size, kDFBDepth)};
    spec.dataflow_buffers[0].data_format_metadata = tt::DataFormat::Float16_b;
    spec.dataflow_buffers[0].advanced_options.allocation_override =
        SRAMArenaAllocation{.arena = SRAMArenaParamName{"arena"}};
    spec.tensor_parameters = {
        {.unique_id = TensorParamName{"report"}, .spec = report_spec},
        {.unique_id = TensorParamName{"output"}, .spec = output_spec}};
    spec.work_units = {MakeMinimalWorkUnit("work", node_set, {"producer", "consumer"})};

    // Construct the program before supplying its reservation.
    auto program = MakeProgramFromSpec(*mesh_device, spec);
    auto arena = SRAMArena::allocate(*mesh_device, properties);
    const auto base = static_cast<uint32_t>(std::get<LockstepSRAMArena>(arena.impl().layout).addr);
    auto report = CreateSentinelTensor(*mesh_device, report_spec);
    auto output = CreateSentinelTensor(*mesh_device, output_spec);
    auto& cq = mesh_device->mesh_command_queue();
    std::vector<uint32_t> initial_arena(reservation_bytes_per_bank / sizeof(uint32_t), kSentinel);
    ASSERT_TRUE(slow_dispatch::WriteToL1(*mesh_device, NodeCoord{0, 0}, base, initial_arena));

    ProgramRunArgs args;
    args.advanced_options.sram_arena_args = {{SRAMArenaParamName{"arena"}, std::cref(arena)}};
    args.tensor_args = {
        {TensorParamName{"report"}, TensorArgument{report}}, {TensorParamName{"output"}, TensorArgument{output}}};
    args.kernel_run_args = {
        {.kernel = KernelSpecName{"producer"},
         .runtime_arg_values = MakeRuntimeArgsForSingleNode(NodeCoord{0, 0}, {{"dfb_seed", kDFBPattern}})},
        {.kernel = KernelSpecName{"consumer"},
         .runtime_arg_values = MakeRuntimeArgsForSingleNode(NodeCoord{0, 0}, {{"output_page_start", 0u}})}};
    SetProgramRunArgs(program, args);
    Execute(*mesh_device, std::move(program));

    EXPECT_EQ(cq.enqueue_read_tensor(output).to_vector<uint32_t>(), Pattern(dfb_bytes, kDFBPattern))
        << "DFB consumer output differs";
    auto expected_arena = initial_arena;
    PutPattern(expected_arena, offset, dfb_bytes, kDFBPattern);
    EXPECT_EQ(ReadArena(*mesh_device, NodeCoord{0, 0}, base, reservation_bytes_per_bank), expected_arena)
        << "Reserved SRAM contents or sentinel regions differ";
    ExpectReport(cq, report, {{.dfb_write = base + offset, .dfb_read = base + offset}});
}

TEST_F(SRAMArenaLockstep, ScratchpadNonzeroOffset) {
    // The scratchpad starts at `base + offset`, and its writes land there.
    const NodeRangeSet node_set{NodeRange(NodeCoord{0, 0})};
    const uint32_t scratch_bytes = 2 * alignment;
    const uint32_t offset = alignment;
    const uint32_t reservation_bytes_per_bank =
        offset + scratch_bytes + alignment;  // Includes a trailing sentinel region.
    const LockstepSRAMArenaProperties properties{.per_bank_size = reservation_bytes_per_bank};
    const auto report_spec = ReportSpec(node_set, alignment);

    ProgramSpec spec;
    spec.name = "SRAMArenaLockstepScratchpadNonzeroOffset";
    spec.kernels = {ScratchpadWriter()};
    spec.advanced_options.sram_arena_parameters = {
        {.unique_id = SRAMArenaParamName{"arena"}, .properties = properties}};
    spec.scratchpads = {{.unique_id = ScratchpadSpecName{"pad"}, .size_per_node = scratch_bytes}};
    spec.scratchpads[0].advanced_options.allocation_override =
        SRAMArenaAllocation{.arena = SRAMArenaParamName{"arena"}, .offset = offset};
    spec.tensor_parameters = {{.unique_id = TensorParamName{"report"}, .spec = report_spec}};
    spec.work_units = {MakeMinimalWorkUnit("work", node_set, {"writer"})};

    // Construct the program before supplying its reservation.
    auto program = MakeProgramFromSpec(*mesh_device, spec);
    auto arena = SRAMArena::allocate(*mesh_device, properties);
    const auto base = static_cast<uint32_t>(std::get<LockstepSRAMArena>(arena.impl().layout).addr);
    auto report = CreateSentinelTensor(*mesh_device, report_spec);
    auto& cq = mesh_device->mesh_command_queue();
    std::vector<uint32_t> initial_arena(reservation_bytes_per_bank / sizeof(uint32_t), kSentinel);
    ASSERT_TRUE(slow_dispatch::WriteToL1(*mesh_device, NodeCoord{0, 0}, base, initial_arena));

    ProgramRunArgs args;
    args.advanced_options.sram_arena_args = {{SRAMArenaParamName{"arena"}, std::cref(arena)}};
    args.tensor_args = {{TensorParamName{"report"}, TensorArgument{report}}};
    args.kernel_run_args = {
        {.kernel = KernelSpecName{"writer"},
         .runtime_arg_values = MakeRuntimeArgsForSingleNode(NodeCoord{0, 0}, {{"scratch_seed", kScratchPattern}})}};
    SetProgramRunArgs(program, args);
    Execute(*mesh_device, std::move(program));

    auto expected_arena = initial_arena;
    PutPattern(expected_arena, offset, scratch_bytes, kScratchPattern);
    EXPECT_EQ(ReadArena(*mesh_device, NodeCoord{0, 0}, base, reservation_bytes_per_bank), expected_arena)
        << "Reserved SRAM contents or sentinel regions differ";
    ExpectReport(cq, report, {{.scratchpad = base + offset}});
}

TEST_F(SRAMArenaLockstep, DFBNonzeroOffset) {
    // The DFB starts at `base + offset`, and producer-to-consumer data transfer works.
    const NodeRangeSet node_set{NodeRange(NodeCoord{0, 0})};
    const uint32_t entry_size = tt::round_up(1024, alignment);
    const uint32_t dfb_bytes = kDFBDepth * entry_size;
    const uint32_t offset = alignment;
    const uint32_t reservation_bytes_per_bank = offset + dfb_bytes + alignment;  // Includes a trailing sentinel region.
    const LockstepSRAMArenaProperties properties{.per_bank_size = reservation_bytes_per_bank};
    const auto report_spec = ReportSpec(node_set, alignment);
    const auto output_spec = OutputSpec(node_set.num_cores(), entry_size);

    ProgramSpec spec;
    spec.name = "SRAMArenaLockstepDFBNonzeroOffset";
    spec.kernels = {DFBProducer(), DFBConsumer()};
    spec.advanced_options.sram_arena_parameters = {
        {.unique_id = SRAMArenaParamName{"arena"}, .properties = properties}};
    spec.dataflow_buffers = {MakeMinimalDFB("fifo", entry_size, kDFBDepth)};
    spec.dataflow_buffers[0].data_format_metadata = tt::DataFormat::Float16_b;
    spec.dataflow_buffers[0].advanced_options.allocation_override =
        SRAMArenaAllocation{.arena = SRAMArenaParamName{"arena"}, .offset = offset};
    spec.tensor_parameters = {
        {.unique_id = TensorParamName{"report"}, .spec = report_spec},
        {.unique_id = TensorParamName{"output"}, .spec = output_spec}};
    spec.work_units = {MakeMinimalWorkUnit("work", node_set, {"producer", "consumer"})};

    // Construct the program before supplying its reservation.
    auto program = MakeProgramFromSpec(*mesh_device, spec);
    auto arena = SRAMArena::allocate(*mesh_device, properties);
    const auto base = static_cast<uint32_t>(std::get<LockstepSRAMArena>(arena.impl().layout).addr);
    auto report = CreateSentinelTensor(*mesh_device, report_spec);
    auto output = CreateSentinelTensor(*mesh_device, output_spec);
    auto& cq = mesh_device->mesh_command_queue();
    std::vector<uint32_t> initial_arena(reservation_bytes_per_bank / sizeof(uint32_t), kSentinel);
    ASSERT_TRUE(slow_dispatch::WriteToL1(*mesh_device, NodeCoord{0, 0}, base, initial_arena));

    ProgramRunArgs args;
    args.advanced_options.sram_arena_args = {{SRAMArenaParamName{"arena"}, std::cref(arena)}};
    args.tensor_args = {
        {TensorParamName{"report"}, TensorArgument{report}}, {TensorParamName{"output"}, TensorArgument{output}}};
    args.kernel_run_args = {
        {.kernel = KernelSpecName{"producer"},
         .runtime_arg_values = MakeRuntimeArgsForSingleNode(NodeCoord{0, 0}, {{"dfb_seed", kDFBPattern}})},
        {.kernel = KernelSpecName{"consumer"},
         .runtime_arg_values = MakeRuntimeArgsForSingleNode(NodeCoord{0, 0}, {{"output_page_start", 0u}})}};
    SetProgramRunArgs(program, args);
    Execute(*mesh_device, std::move(program));

    EXPECT_EQ(cq.enqueue_read_tensor(output).to_vector<uint32_t>(), Pattern(dfb_bytes, kDFBPattern))
        << "DFB consumer output differs";
    auto expected_arena = initial_arena;
    PutPattern(expected_arena, offset, dfb_bytes, kDFBPattern);
    EXPECT_EQ(ReadArena(*mesh_device, NodeCoord{0, 0}, base, reservation_bytes_per_bank), expected_arena)
        << "Reserved SRAM contents or sentinel regions differ";
    ExpectReport(cq, report, {{.dfb_write = base + offset, .dfb_read = base + offset}});
}

TEST_F(SRAMArenaLockstep, BorrowingAcrossNodes) {
    // Each selected node receives the correct arena allocation addresses and produces its own expected data.
    const std::vector<NodeCoord> nodes{{0, 0}, {1, 0}};
    const NodeRangeSet node_set{NodeRange{NodeCoord{0, 0}, NodeCoord{1, 0}}};
    const uint32_t scratch_bytes = 2 * alignment;
    const uint32_t entry_size = tt::round_up(1024, alignment);
    const uint32_t dfb_bytes = kDFBDepth * entry_size;
    const uint32_t scratch_offset = alignment;
    const uint32_t dfb_offset = 4 * alignment;
    const uint32_t reservation_bytes_per_bank =
        dfb_offset + dfb_bytes + alignment;  // Includes a trailing sentinel region.
    const LockstepSRAMArenaProperties properties{.per_bank_size = reservation_bytes_per_bank};
    const auto report_spec = ReportSpec(node_set, alignment);
    const auto output_spec = OutputSpec(node_set.num_cores(), entry_size);

    ProgramSpec spec;
    spec.name = "SRAMArenaLockstepAllocationAcrossNodes";
    spec.kernels = {DFBProducer(true), DFBConsumer()};
    spec.advanced_options.sram_arena_parameters = {
        {.unique_id = SRAMArenaParamName{"arena"}, .properties = properties}};
    spec.scratchpads = {{.unique_id = ScratchpadSpecName{"pad"}, .size_per_node = scratch_bytes}};
    spec.scratchpads[0].advanced_options.allocation_override =
        SRAMArenaAllocation{.arena = SRAMArenaParamName{"arena"}, .offset = scratch_offset};
    spec.dataflow_buffers = {MakeMinimalDFB("fifo", entry_size, kDFBDepth)};
    spec.dataflow_buffers[0].data_format_metadata = tt::DataFormat::Float16_b;
    spec.dataflow_buffers[0].advanced_options.allocation_override =
        SRAMArenaAllocation{.arena = SRAMArenaParamName{"arena"}, .offset = dfb_offset};
    spec.tensor_parameters = {
        {.unique_id = TensorParamName{"report"}, .spec = report_spec},
        {.unique_id = TensorParamName{"output"}, .spec = output_spec}};
    spec.work_units = {MakeMinimalWorkUnit("work", node_set, {"producer", "consumer"})};

    // Construct the program before supplying its reservation.
    auto program = MakeProgramFromSpec(*mesh_device, spec);
    auto arena = SRAMArena::allocate(*mesh_device, properties);
    const auto base = static_cast<uint32_t>(std::get<LockstepSRAMArena>(arena.impl().layout).addr);
    auto report = CreateSentinelTensor(*mesh_device, report_spec);
    auto output = CreateSentinelTensor(*mesh_device, output_spec);
    auto& cq = mesh_device->mesh_command_queue();
    std::vector<uint32_t> initial_arena(reservation_bytes_per_bank / sizeof(uint32_t), kSentinel);
    for (const auto& node : nodes) {
        ASSERT_TRUE(slow_dispatch::WriteToL1(*mesh_device, node, base, initial_arena));
    }

    ProgramRunArgs args;
    args.advanced_options.sram_arena_args = {{SRAMArenaParamName{"arena"}, std::cref(arena)}};
    args.tensor_args = {
        {TensorParamName{"report"}, TensorArgument{report}}, {TensorParamName{"output"}, TensorArgument{output}}};
    ProgramRunArgs::KernelRunArgs producer_args{.kernel = KernelSpecName{"producer"}};
    ProgramRunArgs::KernelRunArgs consumer_args{.kernel = KernelSpecName{"consumer"}};
    for (uint32_t i = 0; i < nodes.size(); ++i) {
        producer_args.runtime_arg_values["scratch_seed"][nodes[i]] = kScratchPattern + i * kNodePatternStride;
        producer_args.runtime_arg_values["dfb_seed"][nodes[i]] = kDFBPattern + i * kNodePatternStride;
        consumer_args.runtime_arg_values["output_page_start"][nodes[i]] = i * kDFBDepth;
    }
    args.kernel_run_args = {producer_args, consumer_args};
    SetProgramRunArgs(program, args);
    Execute(*mesh_device, std::move(program));

    const auto output_words = cq.enqueue_read_tensor(output).to_vector<uint32_t>();
    std::vector<uint32_t> expected_output(output_spec.logical_shape().volume(), kSentinel);
    for (uint32_t i = 0; i < nodes.size(); ++i) {
        SCOPED_TRACE("node " + std::to_string(i));
        auto expected_arena = initial_arena;
        PutPattern(expected_arena, scratch_offset, scratch_bytes, kScratchPattern + i * kNodePatternStride);
        PutPattern(expected_arena, dfb_offset, dfb_bytes, kDFBPattern + i * kNodePatternStride);
        EXPECT_EQ(ReadArena(*mesh_device, nodes[i], base, reservation_bytes_per_bank), expected_arena)
            << "Reserved SRAM contents or sentinel regions differ";
        PutPattern(expected_output, i * dfb_bytes, dfb_bytes, kDFBPattern + i * kNodePatternStride);
    }
    EXPECT_EQ(output_words, expected_output) << "DFB consumer output differs";
    ExpectReport(
        cq,
        report,
        {
            {.scratchpad = base + scratch_offset, .dfb_write = base + dfb_offset, .dfb_read = base + dfb_offset},
            {.scratchpad = base + scratch_offset, .dfb_write = base + dfb_offset, .dfb_read = base + dfb_offset},
        });
}

TEST_F(SRAMArenaLockstep, SharedByScratchpadAndDFB) {
    // Both resources use their assigned regions, with correct data and no corruption between them.
    const NodeRangeSet node_set{NodeRange(NodeCoord{0, 0})};
    const uint32_t scratch_bytes = 2 * alignment;
    const uint32_t entry_size = tt::round_up(1024, alignment);
    const uint32_t dfb_bytes = kDFBDepth * entry_size;
    const uint32_t scratch_offset = alignment;
    const uint32_t dfb_offset = 4 * alignment;
    const uint32_t reservation_bytes_per_bank =
        dfb_offset + dfb_bytes + alignment;  // Includes a trailing sentinel region.
    const LockstepSRAMArenaProperties properties{.per_bank_size = reservation_bytes_per_bank};
    const auto report_spec = ReportSpec(node_set, alignment);
    const auto output_spec = OutputSpec(node_set.num_cores(), entry_size);

    ProgramSpec spec;
    spec.name = "SRAMArenaLockstepSharedByScratchpadAndDFB";
    spec.kernels = {DFBProducer(true), DFBConsumer()};
    spec.advanced_options.sram_arena_parameters = {
        {.unique_id = SRAMArenaParamName{"arena"}, .properties = properties}};
    spec.scratchpads = {{.unique_id = ScratchpadSpecName{"pad"}, .size_per_node = scratch_bytes}};
    spec.scratchpads[0].advanced_options.allocation_override =
        SRAMArenaAllocation{.arena = SRAMArenaParamName{"arena"}, .offset = scratch_offset};
    spec.dataflow_buffers = {MakeMinimalDFB("fifo", entry_size, kDFBDepth)};
    spec.dataflow_buffers[0].data_format_metadata = tt::DataFormat::Float16_b;
    spec.dataflow_buffers[0].advanced_options.allocation_override =
        SRAMArenaAllocation{.arena = SRAMArenaParamName{"arena"}, .offset = dfb_offset};
    spec.tensor_parameters = {
        {.unique_id = TensorParamName{"report"}, .spec = report_spec},
        {.unique_id = TensorParamName{"output"}, .spec = output_spec}};
    spec.work_units = {MakeMinimalWorkUnit("work", node_set, {"producer", "consumer"})};

    // Construct the program before supplying its reservation.
    auto program = MakeProgramFromSpec(*mesh_device, spec);
    auto arena = SRAMArena::allocate(*mesh_device, properties);
    const auto base = static_cast<uint32_t>(std::get<LockstepSRAMArena>(arena.impl().layout).addr);
    auto report = CreateSentinelTensor(*mesh_device, report_spec);
    auto output = CreateSentinelTensor(*mesh_device, output_spec);
    auto& cq = mesh_device->mesh_command_queue();
    std::vector<uint32_t> initial_arena(reservation_bytes_per_bank / sizeof(uint32_t), kSentinel);
    ASSERT_TRUE(slow_dispatch::WriteToL1(*mesh_device, NodeCoord{0, 0}, base, initial_arena));

    ProgramRunArgs args;
    args.advanced_options.sram_arena_args = {{SRAMArenaParamName{"arena"}, std::cref(arena)}};
    args.tensor_args = {
        {TensorParamName{"report"}, TensorArgument{report}}, {TensorParamName{"output"}, TensorArgument{output}}};
    args.kernel_run_args = {
        {.kernel = KernelSpecName{"producer"},
         .runtime_arg_values = MakeRuntimeArgsForSingleNode(
             NodeCoord{0, 0}, {{"dfb_seed", kDFBPattern}, {"scratch_seed", kScratchPattern}})},
        {.kernel = KernelSpecName{"consumer"},
         .runtime_arg_values = MakeRuntimeArgsForSingleNode(NodeCoord{0, 0}, {{"output_page_start", 0u}})}};
    SetProgramRunArgs(program, args);
    Execute(*mesh_device, std::move(program));

    EXPECT_EQ(cq.enqueue_read_tensor(output).to_vector<uint32_t>(), Pattern(dfb_bytes, kDFBPattern))
        << "DFB consumer output differs";
    auto expected_arena = initial_arena;
    PutPattern(expected_arena, scratch_offset, scratch_bytes, kScratchPattern);
    PutPattern(expected_arena, dfb_offset, dfb_bytes, kDFBPattern);
    EXPECT_EQ(ReadArena(*mesh_device, NodeCoord{0, 0}, base, reservation_bytes_per_bank), expected_arena)
        << "Reserved SRAM contents or sentinel regions differ";
    ExpectReport(
        cq,
        report,
        {{.scratchpad = base + scratch_offset, .dfb_write = base + dfb_offset, .dfb_read = base + dfb_offset}});
}

// Test-only spec: a minimal kernel on node (0, 0) whose scratchpad "pad" is allocated from SRAMArenaParameter "arena".
ProgramSpec ArenaScratchpadOnMinimalKernelSpec() {
    ProgramSpec spec;
    spec.name = "SRAMArenaUnboundScratchpad";
    spec.kernels = {MakeMinimalGen1DMKernel("writer")};
    spec.kernels[0].scratchpad_bindings = {{.scratchpad_spec_name = ScratchpadSpecName{"pad"}, .accessor_name = "pad"}};
    spec.scratchpads = {{.unique_id = ScratchpadSpecName{"pad"}, .size_per_node = 1024}};
    spec.scratchpads[0].advanced_options.allocation_override =
        SRAMArenaAllocation{.arena = SRAMArenaParamName{"arena"}};
    spec.advanced_options.sram_arena_parameters = {
        {.unique_id = SRAMArenaParamName{"arena"}, .properties = LockstepSRAMArenaProperties{.per_bank_size = 4096}}};
    spec.work_units = {MakeMinimalWorkUnit("work", NodeCoord{0, 0}, {"writer"})};
    return spec;
}

TEST_F(SRAMArenaLockstep, EnqueueRejectsUnboundScratchpad) {
    // Fast-dispatch enqueue rejects a program whose arena scratchpad never received an SRAMArena argument.
    Program program = MakeProgramFromSpec(*mesh_device, ArenaScratchpadOnMinimalKernelSpec());
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
    EXPECT_THAT(
        [&] { distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, /*blocking=*/true); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("has no bound SRAMArena")));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
