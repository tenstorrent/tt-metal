// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fast-dispatch acceptance tests for lockstep and range-lockstep RawAllocation borrowing on WH/BH.
// These expect working borrowed placement; unsupported construction is a test failure.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/experimental/allocation/raw_allocation.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/tensor/host_tensor.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>

#include "command_queue_fixture.hpp"
#include "impl/allocation/raw_allocation_impl.hpp"
#include "impl/dispatch/slow_dispatch.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalDFB;
using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalWorkUnit;

class RawAllocationTest : public UnitMeshCQSingleCardFixture {
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

using RawAllocationRangeLockstep = RawAllocationTest;
using RawAllocationLockstep = RawAllocationTest;

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
        << "Kernel-reported borrowed addresses or untouched report fields differ";
}

KernelSpec ScratchpadWriter() {
    auto kernel = MakeMinimalGen1DMKernel("writer", DataMovementProcessor::RISCV_0);
    kernel.source = "tests/tt_metal/tt_metal/test_kernels/dataflow/raw_allocation_scratchpad.cpp";
    kernel.runtime_arg_schema.runtime_arg_names = {"scratch_seed"};
    kernel.tensor_bindings = {{.tensor_parameter_name = TensorParamName{"report"}, .accessor_name = "report"}};
    kernel.scratchpad_bindings = {{.scratchpad_spec_name = ScratchpadSpecName{"pad"}, .accessor_name = "pad"}};
    return kernel;
}

enum class AllocationMode { Lockstep, RangeLockstep };

class RawAllocationScratchpadUpdates : public RawAllocationTest, public ::testing::WithParamInterface<AllocationMode> {
protected:
    void SetUp() override {
        RawAllocationTest::SetUp();
        if (HasFatalFailure() || IsSkipped()) {
            return;
        }
        const NodeRangeSet nodes{NodeRange(NodeCoord{0, 0})};
        const uint32_t reservation_bytes = 4 * alignment;
        const RawAllocationProperties properties =
            GetParam() == AllocationMode::Lockstep
                ? RawAllocationProperties{LockstepAllocationProperties{.per_bank_size = reservation_bytes}}
                : RawAllocationProperties{RangeLockstepAllocationProperties{
                      .banks_to_be_allocated_on = nodes, .per_bank_size = reservation_bytes}};
        const auto report_spec = ReportSpec(nodes, alignment);
        ProgramSpec spec;
        spec.name = "RawAllocationScratchpadUpdates";
        spec.kernels = {ScratchpadWriter()};
        spec.advanced_options.raw_allocation_parameters = {
            {.unique_id = RawAllocationParamName{"arena"}, .properties = properties}};
        spec.scratchpads = {{.unique_id = ScratchpadSpecName{"pad"}, .size_per_node = 2 * alignment}};
        spec.scratchpads[0].advanced_options.borrowed_from = ScratchpadAdvancedOptions::RawAllocationBorrowing{
            .allocation = RawAllocationParamName{"arena"}, .offset = alignment};
        spec.tensor_parameters = {{.unique_id = TensorParamName{"report"}, .spec = report_spec}};
        spec.work_units = {MakeMinimalWorkUnit("work", nodes, {"writer"})};
        workload.add_program(
            distributed::MeshCoordinateRange(mesh_device->shape()), MakeProgramFromSpec(*mesh_device, spec));
        initial.emplace(
            std::visit([&](const auto& props) { return allocate_on_sram(*mesh_device, props); }, properties));
        replacement.emplace(
            std::visit([&](const auto& props) { return allocate_on_sram(*mesh_device, props); }, properties));
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
        RawAllocationTest::TearDown();
    }

    static uint32_t Base(const RawAllocation& allocation) {
        return std::visit(
            [](const auto& layout) { return static_cast<uint32_t>(layout.addr); }, allocation.impl().layout);
    }

    Program& program() { return workload.get_programs().begin()->second; }

    ProgramRunArgs Args(const RawAllocation& allocation, uint32_t seed = kScratchPattern) {
        ProgramRunArgs args;
        args.advanced_options.raw_allocation_args = {{RawAllocationParamName{"arena"}, std::cref(allocation)}};
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

    std::vector<uint32_t> ReadReservation(const RawAllocation& allocation) {
        return ReadArena(*mesh_device, NodeCoord{0, 0}, Base(allocation), 4 * alignment);
    }

    std::vector<uint32_t> ExpectedReservation(uint32_t seed) {
        std::vector<uint32_t> expected(4 * alignment / sizeof(uint32_t), kSentinel);
        PutPattern(expected, alignment, 2 * alignment, seed);
        return expected;
    }

    std::optional<RawAllocation> initial;
    std::optional<RawAllocation> replacement;
    std::optional<MeshTensor> report;
    distributed::MeshWorkload workload;
};

TEST_P(RawAllocationScratchpadUpdates, FullResetRebindsScratchpad) {
    // A full argument reset moves scratchpad writes to the replacement reservation.
    SetProgramRunArgs(program(), Args(*initial));
    Enqueue();
    SetProgramRunArgs(program(), Args(*replacement, kScratchPattern + kNodePatternStride));
    Enqueue();
    EXPECT_EQ(ReportedAddress(), Base(*replacement) + alignment);
    EXPECT_EQ(ReadReservation(*replacement), ExpectedReservation(kScratchPattern + kNodePatternStride));
    EXPECT_EQ(ReadReservation(*initial), ExpectedReservation(kScratchPattern));
}

TEST_P(RawAllocationScratchpadUpdates, PartialUpdateRebindsScratchpad) {
    // An allocation-only update moves scratchpad writes without resupplying kernel arguments.
    SetProgramRunArgs(program(), Args(*initial));
    Enqueue();
    ProgramRunArgs update;
    update.advanced_options.raw_allocation_args = {{RawAllocationParamName{"arena"}, std::cref(*replacement)}};
    UpdateProgramRunArgs(program(), update);
    Enqueue();
    EXPECT_EQ(ReportedAddress(), Base(*replacement) + alignment);
    EXPECT_EQ(ReadReservation(*replacement), ExpectedReservation(kScratchPattern));
    EXPECT_EQ(ReadReservation(*initial), ExpectedReservation(kScratchPattern));
}

TEST_P(RawAllocationScratchpadUpdates, PartialUpdateRetainsOmittedAllocation) {
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
    RawAllocationScratchpadUpdates,
    ::testing::Values(AllocationMode::Lockstep, AllocationMode::RangeLockstep),
    [](const ::testing::TestParamInfo<AllocationMode>& info) {
        return info.param == AllocationMode::Lockstep ? "Lockstep" : "RangeLockstep";
    });

KernelSpec DFBProducer(bool writes_scratchpad = false) {
    auto kernel = MakeMinimalGen1DMKernel("producer", DataMovementProcessor::RISCV_0);
    kernel.source = writes_scratchpad
                        ? "tests/tt_metal/tt_metal/test_kernels/dataflow/raw_allocation_scratchpad_and_dfb_producer.cpp"
                        : "tests/tt_metal/tt_metal/test_kernels/dataflow/raw_allocation_dfb_producer.cpp";
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
    kernel.source = "tests/tt_metal/tt_metal/test_kernels/dataflow/raw_allocation_dfb_consumer.cpp";
    kernel.runtime_arg_schema.runtime_arg_names = {"output_page_start"};
    kernel.tensor_bindings = {
        {.tensor_parameter_name = TensorParamName{"report"}, .accessor_name = "report"},
        {.tensor_parameter_name = TensorParamName{"output"}, .accessor_name = "output"}};
    kernel.dfb_bindings = {ConsumerOf(DFBSpecName{"fifo"}, "fifo")};
    return kernel;
}

TEST_F(RawAllocationRangeLockstep, ScratchpadDefaultOffset) {
    // The scratchpad starts at the allocation’s base address, and its writes land there.
    const NodeRangeSet node_set{NodeRange(NodeCoord{0, 0})};
    const uint32_t scratch_bytes = 2 * alignment;
    const uint32_t offset = 0;
    const uint32_t reservation_bytes_per_bank =
        offset + scratch_bytes + alignment;  // Includes a trailing sentinel region.
    const RangeLockstepAllocationProperties properties{
        .banks_to_be_allocated_on = node_set, .per_bank_size = reservation_bytes_per_bank};
    const auto report_spec = ReportSpec(node_set, alignment);

    ProgramSpec spec;
    spec.name = "RawAllocationScratchpadDefaultOffset";
    spec.kernels = {ScratchpadWriter()};
    spec.advanced_options.raw_allocation_parameters = {
        {.unique_id = RawAllocationParamName{"arena"}, .properties = properties}};
    spec.scratchpads = {{.unique_id = ScratchpadSpecName{"pad"}, .size_per_node = scratch_bytes}};
    spec.scratchpads[0].advanced_options.borrowed_from =
        ScratchpadAdvancedOptions::RawAllocationBorrowing{.allocation = RawAllocationParamName{"arena"}};
    spec.tensor_parameters = {{.unique_id = TensorParamName{"report"}, .spec = report_spec}};
    spec.work_units = {MakeMinimalWorkUnit("work", node_set, {"writer"})};

    // Construct the program before supplying its reservation.
    auto program = MakeProgramFromSpec(*mesh_device, spec);
    auto arena = allocate_on_sram(*mesh_device, properties);
    const auto base = static_cast<uint32_t>(std::get<RangeLockstepAllocation>(arena.impl().layout).addr);
    auto report = CreateSentinelTensor(*mesh_device, report_spec);
    auto& cq = mesh_device->mesh_command_queue();
    std::vector<uint32_t> initial_arena(reservation_bytes_per_bank / sizeof(uint32_t), kSentinel);
    ASSERT_TRUE(slow_dispatch::WriteToL1(*mesh_device, NodeCoord{0, 0}, base, initial_arena));

    ProgramRunArgs args;
    args.advanced_options.raw_allocation_args = {{RawAllocationParamName{"arena"}, std::cref(arena)}};
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

TEST_F(RawAllocationRangeLockstep, DFBDefaultOffset) {
    // The DFB starts at the allocation’s base address, and producer-to-consumer data transfer works.
    const NodeRangeSet node_set{NodeRange(NodeCoord{0, 0})};
    const uint32_t entry_size = tt::round_up(1024, alignment);
    const uint32_t dfb_bytes = kDFBDepth * entry_size;
    const uint32_t offset = 0;
    const uint32_t reservation_bytes_per_bank = offset + dfb_bytes + alignment;  // Includes a trailing sentinel region.
    const RangeLockstepAllocationProperties properties{
        .banks_to_be_allocated_on = node_set, .per_bank_size = reservation_bytes_per_bank};
    const auto report_spec = ReportSpec(node_set, alignment);
    const auto output_spec = OutputSpec(node_set.num_cores(), entry_size);

    ProgramSpec spec;
    spec.name = "RawAllocationDFBDefaultOffset";
    spec.kernels = {DFBProducer(), DFBConsumer()};
    spec.advanced_options.raw_allocation_parameters = {
        {.unique_id = RawAllocationParamName{"arena"}, .properties = properties}};
    spec.dataflow_buffers = {MakeMinimalDFB("fifo", entry_size, kDFBDepth)};
    spec.dataflow_buffers[0].data_format_metadata = tt::DataFormat::Float16_b;
    spec.dataflow_buffers[0].borrowed_from = RawAllocationParamName{"arena"};
    spec.tensor_parameters = {
        {.unique_id = TensorParamName{"report"}, .spec = report_spec},
        {.unique_id = TensorParamName{"output"}, .spec = output_spec}};
    spec.work_units = {MakeMinimalWorkUnit("work", node_set, {"producer", "consumer"})};

    // Construct the program before supplying its reservation.
    auto program = MakeProgramFromSpec(*mesh_device, spec);
    auto arena = allocate_on_sram(*mesh_device, properties);
    const auto base = static_cast<uint32_t>(std::get<RangeLockstepAllocation>(arena.impl().layout).addr);
    auto report = CreateSentinelTensor(*mesh_device, report_spec);
    auto output = CreateSentinelTensor(*mesh_device, output_spec);
    auto& cq = mesh_device->mesh_command_queue();
    std::vector<uint32_t> initial_arena(reservation_bytes_per_bank / sizeof(uint32_t), kSentinel);
    ASSERT_TRUE(slow_dispatch::WriteToL1(*mesh_device, NodeCoord{0, 0}, base, initial_arena));

    ProgramRunArgs args;
    args.advanced_options.raw_allocation_args = {{RawAllocationParamName{"arena"}, std::cref(arena)}};
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

TEST_F(RawAllocationRangeLockstep, ScratchpadNonzeroOffset) {
    // The scratchpad starts at `base + offset`, and its writes land there.
    const NodeRangeSet node_set{NodeRange(NodeCoord{0, 0})};
    const uint32_t scratch_bytes = 2 * alignment;
    const uint32_t offset = alignment;
    const uint32_t reservation_bytes_per_bank =
        offset + scratch_bytes + alignment;  // Includes a trailing sentinel region.
    const RangeLockstepAllocationProperties properties{
        .banks_to_be_allocated_on = node_set, .per_bank_size = reservation_bytes_per_bank};
    const auto report_spec = ReportSpec(node_set, alignment);

    ProgramSpec spec;
    spec.name = "RawAllocationScratchpadNonzeroOffset";
    spec.kernels = {ScratchpadWriter()};
    spec.advanced_options.raw_allocation_parameters = {
        {.unique_id = RawAllocationParamName{"arena"}, .properties = properties}};
    spec.scratchpads = {{.unique_id = ScratchpadSpecName{"pad"}, .size_per_node = scratch_bytes}};
    spec.scratchpads[0].advanced_options.borrowed_from = ScratchpadAdvancedOptions::RawAllocationBorrowing{
        .allocation = RawAllocationParamName{"arena"}, .offset = offset};
    spec.tensor_parameters = {{.unique_id = TensorParamName{"report"}, .spec = report_spec}};
    spec.work_units = {MakeMinimalWorkUnit("work", node_set, {"writer"})};

    // Construct the program before supplying its reservation.
    auto program = MakeProgramFromSpec(*mesh_device, spec);
    auto arena = allocate_on_sram(*mesh_device, properties);
    const auto base = static_cast<uint32_t>(std::get<RangeLockstepAllocation>(arena.impl().layout).addr);
    auto report = CreateSentinelTensor(*mesh_device, report_spec);
    auto& cq = mesh_device->mesh_command_queue();
    std::vector<uint32_t> initial_arena(reservation_bytes_per_bank / sizeof(uint32_t), kSentinel);
    ASSERT_TRUE(slow_dispatch::WriteToL1(*mesh_device, NodeCoord{0, 0}, base, initial_arena));

    ProgramRunArgs args;
    args.advanced_options.raw_allocation_args = {{RawAllocationParamName{"arena"}, std::cref(arena)}};
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

TEST_F(RawAllocationRangeLockstep, DFBNonzeroOffset) {
    // The DFB starts at `base + offset`, and producer-to-consumer data transfer works.
    const NodeRangeSet node_set{NodeRange(NodeCoord{0, 0})};
    const uint32_t entry_size = tt::round_up(1024, alignment);
    const uint32_t dfb_bytes = kDFBDepth * entry_size;
    const uint32_t offset = alignment;
    const uint32_t reservation_bytes_per_bank = offset + dfb_bytes + alignment;  // Includes a trailing sentinel region.
    const RangeLockstepAllocationProperties properties{
        .banks_to_be_allocated_on = node_set, .per_bank_size = reservation_bytes_per_bank};
    const auto report_spec = ReportSpec(node_set, alignment);
    const auto output_spec = OutputSpec(node_set.num_cores(), entry_size);

    ProgramSpec spec;
    spec.name = "RawAllocationDFBNonzeroOffset";
    spec.kernels = {DFBProducer(), DFBConsumer()};
    spec.advanced_options.raw_allocation_parameters = {
        {.unique_id = RawAllocationParamName{"arena"}, .properties = properties}};
    spec.dataflow_buffers = {MakeMinimalDFB("fifo", entry_size, kDFBDepth)};
    spec.dataflow_buffers[0].data_format_metadata = tt::DataFormat::Float16_b;
    spec.dataflow_buffers[0].borrowed_from = RawAllocationParamName{"arena"};
    spec.dataflow_buffers[0].advanced_options.borrowed_memory_offset = offset;
    spec.tensor_parameters = {
        {.unique_id = TensorParamName{"report"}, .spec = report_spec},
        {.unique_id = TensorParamName{"output"}, .spec = output_spec}};
    spec.work_units = {MakeMinimalWorkUnit("work", node_set, {"producer", "consumer"})};

    // Construct the program before supplying its reservation.
    auto program = MakeProgramFromSpec(*mesh_device, spec);
    auto arena = allocate_on_sram(*mesh_device, properties);
    const auto base = static_cast<uint32_t>(std::get<RangeLockstepAllocation>(arena.impl().layout).addr);
    auto report = CreateSentinelTensor(*mesh_device, report_spec);
    auto output = CreateSentinelTensor(*mesh_device, output_spec);
    auto& cq = mesh_device->mesh_command_queue();
    std::vector<uint32_t> initial_arena(reservation_bytes_per_bank / sizeof(uint32_t), kSentinel);
    ASSERT_TRUE(slow_dispatch::WriteToL1(*mesh_device, NodeCoord{0, 0}, base, initial_arena));

    ProgramRunArgs args;
    args.advanced_options.raw_allocation_args = {{RawAllocationParamName{"arena"}, std::cref(arena)}};
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

TEST_F(RawAllocationRangeLockstep, BorrowingAcrossNodes) {
    // Each selected node receives the correct borrowed addresses and produces its own expected data.
    const std::vector<NodeCoord> nodes{{0, 0}, {1, 0}};
    const NodeRangeSet node_set{NodeRange{NodeCoord{0, 0}, NodeCoord{1, 0}}};
    const uint32_t scratch_bytes = 2 * alignment;
    const uint32_t entry_size = tt::round_up(1024, alignment);
    const uint32_t dfb_bytes = kDFBDepth * entry_size;
    const uint32_t scratch_offset = alignment;
    const uint32_t dfb_offset = 4 * alignment;
    const uint32_t reservation_bytes_per_bank =
        dfb_offset + dfb_bytes + alignment;  // Includes a trailing sentinel region.
    const RangeLockstepAllocationProperties properties{
        .banks_to_be_allocated_on = node_set, .per_bank_size = reservation_bytes_per_bank};
    const auto report_spec = ReportSpec(node_set, alignment);
    const auto output_spec = OutputSpec(node_set.num_cores(), entry_size);

    ProgramSpec spec;
    spec.name = "RawAllocationBorrowingAcrossNodes";
    spec.kernels = {DFBProducer(true), DFBConsumer()};
    spec.advanced_options.raw_allocation_parameters = {
        {.unique_id = RawAllocationParamName{"arena"}, .properties = properties}};
    spec.scratchpads = {{.unique_id = ScratchpadSpecName{"pad"}, .size_per_node = scratch_bytes}};
    spec.scratchpads[0].advanced_options.borrowed_from = ScratchpadAdvancedOptions::RawAllocationBorrowing{
        .allocation = RawAllocationParamName{"arena"}, .offset = scratch_offset};
    spec.dataflow_buffers = {MakeMinimalDFB("fifo", entry_size, kDFBDepth)};
    spec.dataflow_buffers[0].data_format_metadata = tt::DataFormat::Float16_b;
    spec.dataflow_buffers[0].borrowed_from = RawAllocationParamName{"arena"};
    spec.dataflow_buffers[0].advanced_options.borrowed_memory_offset = dfb_offset;
    spec.tensor_parameters = {
        {.unique_id = TensorParamName{"report"}, .spec = report_spec},
        {.unique_id = TensorParamName{"output"}, .spec = output_spec}};
    spec.work_units = {MakeMinimalWorkUnit("work", node_set, {"producer", "consumer"})};

    // Construct the program before supplying its reservation.
    auto program = MakeProgramFromSpec(*mesh_device, spec);
    auto arena = allocate_on_sram(*mesh_device, properties);
    const auto base = static_cast<uint32_t>(std::get<RangeLockstepAllocation>(arena.impl().layout).addr);
    auto report = CreateSentinelTensor(*mesh_device, report_spec);
    auto output = CreateSentinelTensor(*mesh_device, output_spec);
    auto& cq = mesh_device->mesh_command_queue();
    std::vector<uint32_t> initial_arena(reservation_bytes_per_bank / sizeof(uint32_t), kSentinel);
    for (const auto& node : nodes) {
        ASSERT_TRUE(slow_dispatch::WriteToL1(*mesh_device, node, base, initial_arena));
    }

    ProgramRunArgs args;
    args.advanced_options.raw_allocation_args = {{RawAllocationParamName{"arena"}, std::cref(arena)}};
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

TEST_F(RawAllocationRangeLockstep, SharedByScratchpadAndDFB) {
    // Both resources use their assigned regions, with correct data and no corruption between them.
    const NodeRangeSet node_set{NodeRange(NodeCoord{0, 0})};
    const uint32_t scratch_bytes = 2 * alignment;
    const uint32_t entry_size = tt::round_up(1024, alignment);
    const uint32_t dfb_bytes = kDFBDepth * entry_size;
    const uint32_t scratch_offset = alignment;
    const uint32_t dfb_offset = 4 * alignment;
    const uint32_t reservation_bytes_per_bank =
        dfb_offset + dfb_bytes + alignment;  // Includes a trailing sentinel region.
    const RangeLockstepAllocationProperties properties{
        .banks_to_be_allocated_on = node_set, .per_bank_size = reservation_bytes_per_bank};
    const auto report_spec = ReportSpec(node_set, alignment);
    const auto output_spec = OutputSpec(node_set.num_cores(), entry_size);

    ProgramSpec spec;
    spec.name = "RawAllocationSharedByScratchpadAndDFB";
    spec.kernels = {DFBProducer(true), DFBConsumer()};
    spec.advanced_options.raw_allocation_parameters = {
        {.unique_id = RawAllocationParamName{"arena"}, .properties = properties}};
    spec.scratchpads = {{.unique_id = ScratchpadSpecName{"pad"}, .size_per_node = scratch_bytes}};
    spec.scratchpads[0].advanced_options.borrowed_from = ScratchpadAdvancedOptions::RawAllocationBorrowing{
        .allocation = RawAllocationParamName{"arena"}, .offset = scratch_offset};
    spec.dataflow_buffers = {MakeMinimalDFB("fifo", entry_size, kDFBDepth)};
    spec.dataflow_buffers[0].data_format_metadata = tt::DataFormat::Float16_b;
    spec.dataflow_buffers[0].borrowed_from = RawAllocationParamName{"arena"};
    spec.dataflow_buffers[0].advanced_options.borrowed_memory_offset = dfb_offset;
    spec.tensor_parameters = {
        {.unique_id = TensorParamName{"report"}, .spec = report_spec},
        {.unique_id = TensorParamName{"output"}, .spec = output_spec}};
    spec.work_units = {MakeMinimalWorkUnit("work", node_set, {"producer", "consumer"})};

    // Construct the program before supplying its reservation.
    auto program = MakeProgramFromSpec(*mesh_device, spec);
    auto arena = allocate_on_sram(*mesh_device, properties);
    const auto base = static_cast<uint32_t>(std::get<RangeLockstepAllocation>(arena.impl().layout).addr);
    auto report = CreateSentinelTensor(*mesh_device, report_spec);
    auto output = CreateSentinelTensor(*mesh_device, output_spec);
    auto& cq = mesh_device->mesh_command_queue();
    std::vector<uint32_t> initial_arena(reservation_bytes_per_bank / sizeof(uint32_t), kSentinel);
    ASSERT_TRUE(slow_dispatch::WriteToL1(*mesh_device, NodeCoord{0, 0}, base, initial_arena));

    ProgramRunArgs args;
    args.advanced_options.raw_allocation_args = {{RawAllocationParamName{"arena"}, std::cref(arena)}};
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

TEST_F(RawAllocationLockstep, ScratchpadDefaultOffset) {
    // The scratchpad starts at the allocation’s base address, and its writes land there.
    const NodeRangeSet node_set{NodeRange(NodeCoord{0, 0})};
    const uint32_t scratch_bytes = 2 * alignment;
    const uint32_t offset = 0;
    const uint32_t reservation_bytes_per_bank =
        offset + scratch_bytes + alignment;  // Includes a trailing sentinel region.
    const LockstepAllocationProperties properties{.per_bank_size = reservation_bytes_per_bank};
    const auto report_spec = ReportSpec(node_set, alignment);

    ProgramSpec spec;
    spec.name = "RawAllocationLockstepScratchpadDefaultOffset";
    spec.kernels = {ScratchpadWriter()};
    spec.advanced_options.raw_allocation_parameters = {
        {.unique_id = RawAllocationParamName{"arena"}, .properties = properties}};
    spec.scratchpads = {{.unique_id = ScratchpadSpecName{"pad"}, .size_per_node = scratch_bytes}};
    spec.scratchpads[0].advanced_options.borrowed_from =
        ScratchpadAdvancedOptions::RawAllocationBorrowing{.allocation = RawAllocationParamName{"arena"}};
    spec.tensor_parameters = {{.unique_id = TensorParamName{"report"}, .spec = report_spec}};
    spec.work_units = {MakeMinimalWorkUnit("work", node_set, {"writer"})};

    // Construct the program before supplying its reservation.
    auto program = MakeProgramFromSpec(*mesh_device, spec);
    auto arena = allocate_on_sram(*mesh_device, properties);
    const auto base = static_cast<uint32_t>(std::get<LockstepAllocation>(arena.impl().layout).addr);
    auto report = CreateSentinelTensor(*mesh_device, report_spec);
    auto& cq = mesh_device->mesh_command_queue();
    std::vector<uint32_t> initial_arena(reservation_bytes_per_bank / sizeof(uint32_t), kSentinel);
    ASSERT_TRUE(slow_dispatch::WriteToL1(*mesh_device, NodeCoord{0, 0}, base, initial_arena));

    ProgramRunArgs args;
    args.advanced_options.raw_allocation_args = {{RawAllocationParamName{"arena"}, std::cref(arena)}};
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

TEST_F(RawAllocationLockstep, DFBDefaultOffset) {
    // The DFB starts at the allocation’s base address, and producer-to-consumer data transfer works.
    const NodeRangeSet node_set{NodeRange(NodeCoord{0, 0})};
    const uint32_t entry_size = tt::round_up(1024, alignment);
    const uint32_t dfb_bytes = kDFBDepth * entry_size;
    const uint32_t offset = 0;
    const uint32_t reservation_bytes_per_bank = offset + dfb_bytes + alignment;  // Includes a trailing sentinel region.
    const LockstepAllocationProperties properties{.per_bank_size = reservation_bytes_per_bank};
    const auto report_spec = ReportSpec(node_set, alignment);
    const auto output_spec = OutputSpec(node_set.num_cores(), entry_size);

    ProgramSpec spec;
    spec.name = "RawAllocationLockstepDFBDefaultOffset";
    spec.kernels = {DFBProducer(), DFBConsumer()};
    spec.advanced_options.raw_allocation_parameters = {
        {.unique_id = RawAllocationParamName{"arena"}, .properties = properties}};
    spec.dataflow_buffers = {MakeMinimalDFB("fifo", entry_size, kDFBDepth)};
    spec.dataflow_buffers[0].data_format_metadata = tt::DataFormat::Float16_b;
    spec.dataflow_buffers[0].borrowed_from = RawAllocationParamName{"arena"};
    spec.tensor_parameters = {
        {.unique_id = TensorParamName{"report"}, .spec = report_spec},
        {.unique_id = TensorParamName{"output"}, .spec = output_spec}};
    spec.work_units = {MakeMinimalWorkUnit("work", node_set, {"producer", "consumer"})};

    // Construct the program before supplying its reservation.
    auto program = MakeProgramFromSpec(*mesh_device, spec);
    auto arena = allocate_on_sram(*mesh_device, properties);
    const auto base = static_cast<uint32_t>(std::get<LockstepAllocation>(arena.impl().layout).addr);
    auto report = CreateSentinelTensor(*mesh_device, report_spec);
    auto output = CreateSentinelTensor(*mesh_device, output_spec);
    auto& cq = mesh_device->mesh_command_queue();
    std::vector<uint32_t> initial_arena(reservation_bytes_per_bank / sizeof(uint32_t), kSentinel);
    ASSERT_TRUE(slow_dispatch::WriteToL1(*mesh_device, NodeCoord{0, 0}, base, initial_arena));

    ProgramRunArgs args;
    args.advanced_options.raw_allocation_args = {{RawAllocationParamName{"arena"}, std::cref(arena)}};
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

TEST_F(RawAllocationLockstep, ScratchpadNonzeroOffset) {
    // The scratchpad starts at `base + offset`, and its writes land there.
    const NodeRangeSet node_set{NodeRange(NodeCoord{0, 0})};
    const uint32_t scratch_bytes = 2 * alignment;
    const uint32_t offset = alignment;
    const uint32_t reservation_bytes_per_bank =
        offset + scratch_bytes + alignment;  // Includes a trailing sentinel region.
    const LockstepAllocationProperties properties{.per_bank_size = reservation_bytes_per_bank};
    const auto report_spec = ReportSpec(node_set, alignment);

    ProgramSpec spec;
    spec.name = "RawAllocationLockstepScratchpadNonzeroOffset";
    spec.kernels = {ScratchpadWriter()};
    spec.advanced_options.raw_allocation_parameters = {
        {.unique_id = RawAllocationParamName{"arena"}, .properties = properties}};
    spec.scratchpads = {{.unique_id = ScratchpadSpecName{"pad"}, .size_per_node = scratch_bytes}};
    spec.scratchpads[0].advanced_options.borrowed_from = ScratchpadAdvancedOptions::RawAllocationBorrowing{
        .allocation = RawAllocationParamName{"arena"}, .offset = offset};
    spec.tensor_parameters = {{.unique_id = TensorParamName{"report"}, .spec = report_spec}};
    spec.work_units = {MakeMinimalWorkUnit("work", node_set, {"writer"})};

    // Construct the program before supplying its reservation.
    auto program = MakeProgramFromSpec(*mesh_device, spec);
    auto arena = allocate_on_sram(*mesh_device, properties);
    const auto base = static_cast<uint32_t>(std::get<LockstepAllocation>(arena.impl().layout).addr);
    auto report = CreateSentinelTensor(*mesh_device, report_spec);
    auto& cq = mesh_device->mesh_command_queue();
    std::vector<uint32_t> initial_arena(reservation_bytes_per_bank / sizeof(uint32_t), kSentinel);
    ASSERT_TRUE(slow_dispatch::WriteToL1(*mesh_device, NodeCoord{0, 0}, base, initial_arena));

    ProgramRunArgs args;
    args.advanced_options.raw_allocation_args = {{RawAllocationParamName{"arena"}, std::cref(arena)}};
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

TEST_F(RawAllocationLockstep, DFBNonzeroOffset) {
    // The DFB starts at `base + offset`, and producer-to-consumer data transfer works.
    const NodeRangeSet node_set{NodeRange(NodeCoord{0, 0})};
    const uint32_t entry_size = tt::round_up(1024, alignment);
    const uint32_t dfb_bytes = kDFBDepth * entry_size;
    const uint32_t offset = alignment;
    const uint32_t reservation_bytes_per_bank = offset + dfb_bytes + alignment;  // Includes a trailing sentinel region.
    const LockstepAllocationProperties properties{.per_bank_size = reservation_bytes_per_bank};
    const auto report_spec = ReportSpec(node_set, alignment);
    const auto output_spec = OutputSpec(node_set.num_cores(), entry_size);

    ProgramSpec spec;
    spec.name = "RawAllocationLockstepDFBNonzeroOffset";
    spec.kernels = {DFBProducer(), DFBConsumer()};
    spec.advanced_options.raw_allocation_parameters = {
        {.unique_id = RawAllocationParamName{"arena"}, .properties = properties}};
    spec.dataflow_buffers = {MakeMinimalDFB("fifo", entry_size, kDFBDepth)};
    spec.dataflow_buffers[0].data_format_metadata = tt::DataFormat::Float16_b;
    spec.dataflow_buffers[0].borrowed_from = RawAllocationParamName{"arena"};
    spec.dataflow_buffers[0].advanced_options.borrowed_memory_offset = offset;
    spec.tensor_parameters = {
        {.unique_id = TensorParamName{"report"}, .spec = report_spec},
        {.unique_id = TensorParamName{"output"}, .spec = output_spec}};
    spec.work_units = {MakeMinimalWorkUnit("work", node_set, {"producer", "consumer"})};

    // Construct the program before supplying its reservation.
    auto program = MakeProgramFromSpec(*mesh_device, spec);
    auto arena = allocate_on_sram(*mesh_device, properties);
    const auto base = static_cast<uint32_t>(std::get<LockstepAllocation>(arena.impl().layout).addr);
    auto report = CreateSentinelTensor(*mesh_device, report_spec);
    auto output = CreateSentinelTensor(*mesh_device, output_spec);
    auto& cq = mesh_device->mesh_command_queue();
    std::vector<uint32_t> initial_arena(reservation_bytes_per_bank / sizeof(uint32_t), kSentinel);
    ASSERT_TRUE(slow_dispatch::WriteToL1(*mesh_device, NodeCoord{0, 0}, base, initial_arena));

    ProgramRunArgs args;
    args.advanced_options.raw_allocation_args = {{RawAllocationParamName{"arena"}, std::cref(arena)}};
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

TEST_F(RawAllocationLockstep, BorrowingAcrossNodes) {
    // Each selected node receives the correct borrowed addresses and produces its own expected data.
    const std::vector<NodeCoord> nodes{{0, 0}, {1, 0}};
    const NodeRangeSet node_set{NodeRange{NodeCoord{0, 0}, NodeCoord{1, 0}}};
    const uint32_t scratch_bytes = 2 * alignment;
    const uint32_t entry_size = tt::round_up(1024, alignment);
    const uint32_t dfb_bytes = kDFBDepth * entry_size;
    const uint32_t scratch_offset = alignment;
    const uint32_t dfb_offset = 4 * alignment;
    const uint32_t reservation_bytes_per_bank =
        dfb_offset + dfb_bytes + alignment;  // Includes a trailing sentinel region.
    const LockstepAllocationProperties properties{.per_bank_size = reservation_bytes_per_bank};
    const auto report_spec = ReportSpec(node_set, alignment);
    const auto output_spec = OutputSpec(node_set.num_cores(), entry_size);

    ProgramSpec spec;
    spec.name = "RawAllocationLockstepBorrowingAcrossNodes";
    spec.kernels = {DFBProducer(true), DFBConsumer()};
    spec.advanced_options.raw_allocation_parameters = {
        {.unique_id = RawAllocationParamName{"arena"}, .properties = properties}};
    spec.scratchpads = {{.unique_id = ScratchpadSpecName{"pad"}, .size_per_node = scratch_bytes}};
    spec.scratchpads[0].advanced_options.borrowed_from = ScratchpadAdvancedOptions::RawAllocationBorrowing{
        .allocation = RawAllocationParamName{"arena"}, .offset = scratch_offset};
    spec.dataflow_buffers = {MakeMinimalDFB("fifo", entry_size, kDFBDepth)};
    spec.dataflow_buffers[0].data_format_metadata = tt::DataFormat::Float16_b;
    spec.dataflow_buffers[0].borrowed_from = RawAllocationParamName{"arena"};
    spec.dataflow_buffers[0].advanced_options.borrowed_memory_offset = dfb_offset;
    spec.tensor_parameters = {
        {.unique_id = TensorParamName{"report"}, .spec = report_spec},
        {.unique_id = TensorParamName{"output"}, .spec = output_spec}};
    spec.work_units = {MakeMinimalWorkUnit("work", node_set, {"producer", "consumer"})};

    // Construct the program before supplying its reservation.
    auto program = MakeProgramFromSpec(*mesh_device, spec);
    auto arena = allocate_on_sram(*mesh_device, properties);
    const auto base = static_cast<uint32_t>(std::get<LockstepAllocation>(arena.impl().layout).addr);
    auto report = CreateSentinelTensor(*mesh_device, report_spec);
    auto output = CreateSentinelTensor(*mesh_device, output_spec);
    auto& cq = mesh_device->mesh_command_queue();
    std::vector<uint32_t> initial_arena(reservation_bytes_per_bank / sizeof(uint32_t), kSentinel);
    for (const auto& node : nodes) {
        ASSERT_TRUE(slow_dispatch::WriteToL1(*mesh_device, node, base, initial_arena));
    }

    ProgramRunArgs args;
    args.advanced_options.raw_allocation_args = {{RawAllocationParamName{"arena"}, std::cref(arena)}};
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

TEST_F(RawAllocationLockstep, SharedByScratchpadAndDFB) {
    // Both resources use their assigned regions, with correct data and no corruption between them.
    const NodeRangeSet node_set{NodeRange(NodeCoord{0, 0})};
    const uint32_t scratch_bytes = 2 * alignment;
    const uint32_t entry_size = tt::round_up(1024, alignment);
    const uint32_t dfb_bytes = kDFBDepth * entry_size;
    const uint32_t scratch_offset = alignment;
    const uint32_t dfb_offset = 4 * alignment;
    const uint32_t reservation_bytes_per_bank =
        dfb_offset + dfb_bytes + alignment;  // Includes a trailing sentinel region.
    const LockstepAllocationProperties properties{.per_bank_size = reservation_bytes_per_bank};
    const auto report_spec = ReportSpec(node_set, alignment);
    const auto output_spec = OutputSpec(node_set.num_cores(), entry_size);

    ProgramSpec spec;
    spec.name = "RawAllocationLockstepSharedByScratchpadAndDFB";
    spec.kernels = {DFBProducer(true), DFBConsumer()};
    spec.advanced_options.raw_allocation_parameters = {
        {.unique_id = RawAllocationParamName{"arena"}, .properties = properties}};
    spec.scratchpads = {{.unique_id = ScratchpadSpecName{"pad"}, .size_per_node = scratch_bytes}};
    spec.scratchpads[0].advanced_options.borrowed_from = ScratchpadAdvancedOptions::RawAllocationBorrowing{
        .allocation = RawAllocationParamName{"arena"}, .offset = scratch_offset};
    spec.dataflow_buffers = {MakeMinimalDFB("fifo", entry_size, kDFBDepth)};
    spec.dataflow_buffers[0].data_format_metadata = tt::DataFormat::Float16_b;
    spec.dataflow_buffers[0].borrowed_from = RawAllocationParamName{"arena"};
    spec.dataflow_buffers[0].advanced_options.borrowed_memory_offset = dfb_offset;
    spec.tensor_parameters = {
        {.unique_id = TensorParamName{"report"}, .spec = report_spec},
        {.unique_id = TensorParamName{"output"}, .spec = output_spec}};
    spec.work_units = {MakeMinimalWorkUnit("work", node_set, {"producer", "consumer"})};

    // Construct the program before supplying its reservation.
    auto program = MakeProgramFromSpec(*mesh_device, spec);
    auto arena = allocate_on_sram(*mesh_device, properties);
    const auto base = static_cast<uint32_t>(std::get<LockstepAllocation>(arena.impl().layout).addr);
    auto report = CreateSentinelTensor(*mesh_device, report_spec);
    auto output = CreateSentinelTensor(*mesh_device, output_spec);
    auto& cq = mesh_device->mesh_command_queue();
    std::vector<uint32_t> initial_arena(reservation_bytes_per_bank / sizeof(uint32_t), kSentinel);
    ASSERT_TRUE(slow_dispatch::WriteToL1(*mesh_device, NodeCoord{0, 0}, base, initial_arena));

    ProgramRunArgs args;
    args.advanced_options.raw_allocation_args = {{RawAllocationParamName{"arena"}, std::cref(arena)}};
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

}  // namespace
}  // namespace tt::tt_metal::experimental
