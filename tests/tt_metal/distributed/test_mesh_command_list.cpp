// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/command_list.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "tests/tt_metal/tt_metal/api/metal2_host_api/test_helpers.hpp"
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"

namespace tt::tt_metal::experimental::test {
namespace {

namespace m2 = tt::tt_metal::experimental;
using namespace tt::tt_metal::distributed;

using m2::test_helpers::BindTensorParameterToKernel;
using m2::test_helpers::MakeMinimalDFB;
using m2::test_helpers::MakeMinimalGen1DMKernel;
using m2::test_helpers::MakeMinimalWorkUnit;
using ::testing::HasSubstr;
using ::testing::ThrowsMessage;

constexpr m2::NodeCoord kNode{0, 0};
constexpr uint32_t kAddressA = 100 * 1024;
constexpr uint32_t kAddressB = kAddressA + 64;
constexpr uint32_t kAddressC = kAddressB + 64;
constexpr uint32_t kValueA = 0xCAFE0001;
constexpr uint32_t kValueB = 0xCAFE0002;
constexpr char kKernelName[] = "writer";

static_assert(!std::is_copy_constructible_v<CommandListBuilder>);
static_assert(!std::is_copy_assignable_v<CommandListBuilder>);
static_assert(std::is_nothrow_move_constructible_v<CommandListBuilder>);
static_assert(std::is_nothrow_move_assignable_v<CommandListBuilder>);
static_assert(!std::is_copy_constructible_v<CommandList>);
static_assert(!std::is_copy_assignable_v<CommandList>);
static_assert(std::is_nothrow_move_constructible_v<CommandList>);
static_assert(std::is_nothrow_move_assignable_v<CommandList>);

class CommandListTest : public MeshDeviceFixtureBase {
protected:
    CommandListTest() :
        MeshDeviceFixtureBase(Config{.mesh_shape = MeshShape{1, 1}, .num_cqs = 1, .trace_region_size = (64 << 20)}) {}

    void SetUp() override {
        MeshDeviceFixtureBase::SetUp();
        if (IsSkipped()) {
            return;
        }
        const auto arch = mesh_device_->get_devices().at(0)->arch();
        if (arch != tt::ARCH::WORMHOLE_B0 && arch != tt::ARCH::BLACKHOLE) {
            GTEST_SKIP() << "Command-list tests require Wormhole B0 or Blackhole hardware";
        }
    }
};

class CommandListMultiCQTest : public MeshDeviceFixtureBase {
protected:
    CommandListMultiCQTest() :
        MeshDeviceFixtureBase(Config{.mesh_shape = MeshShape{1, 1}, .num_cqs = 2, .trace_region_size = (64 << 20)}) {}

    void SetUp() override {
        MeshDeviceFixtureBase::SetUp();
        if (IsSkipped()) {
            return;
        }
        const auto arch = mesh_device_->get_devices().at(0)->arch();
        if (arch != tt::ARCH::WORMHOLE_B0 && arch != tt::ARCH::BLACKHOLE) {
            GTEST_SKIP() << "Command-list tests require Wormhole B0 or Blackhole hardware";
        }
    }
};

class CommandListMultiDeviceTest : public MeshDeviceFixtureBase {
protected:
    CommandListMultiDeviceTest() :
        MeshDeviceFixtureBase(Config{.mesh_shape = MeshShape{2, 4}, .num_cqs = 1, .trace_region_size = (64 << 20)}) {}

    void SetUp() override {
        MeshDeviceFixtureBase::SetUp();
        if (IsSkipped()) {
            return;
        }
        const auto arch = mesh_device_->get_devices().at(0)->arch();
        if (arch != tt::ARCH::WORMHOLE_B0 && arch != tt::ARCH::BLACKHOLE) {
            GTEST_SKIP() << "Command-list tests require Wormhole B0 or Blackhole hardware";
        }
    }
};

m2::ProgramSpec make_l1_write_spec(const std::string& name) {
    const m2::KernelSpecName kernel_name{kKernelName};
    auto kernel = MakeMinimalGen1DMKernel(kKernelName, DataMovementProcessor::RISCV_0);
    kernel.source = "tests/tt_metal/tt_metal/test_kernels/dataflow/command_list_l1_write.cpp";
    kernel.runtime_arg_schema = {.runtime_arg_names = {"address"}, .common_runtime_arg_names = {"value"}};

    return m2::ProgramSpec{
        .name = name,
        .kernels = {kernel},
        .work_units = {m2::WorkUnitSpec{.name = "main", .kernels = {kernel_name}, .target_nodes = kNode}},
    };
}

MeshWorkload make_l1_write_workload(
    MeshDevice& mesh_device, uint32_t address, uint32_t value, const std::string& name) {
    auto workload = m2::MakeMeshWorkloadFromSpec(mesh_device, make_l1_write_spec(name));
    auto& program = workload.get_programs().begin()->second;

    m2::ProgramRunArgs args;
    args.kernel_run_args = {m2::ProgramRunArgs::KernelRunArgs{
        .kernel = m2::KernelSpecName{kKernelName},
        .runtime_arg_values = m2::MakeRuntimeArgsForSingleNode(kNode, {{"address", address}}),
        .common_runtime_arg_values = {{"value", value}},
    }};
    m2::SetProgramRunArgs(program, args);
    return workload;
}

using DeviceRangeValue = std::pair<MeshCoordinateRange, uint32_t>;

MeshWorkload make_l1_write_workload(
    MeshDevice& mesh_device,
    uint32_t address,
    const std::vector<DeviceRangeValue>& device_range_values,
    const std::string& name) {
    std::unordered_map<MeshCoordinateRange, m2::ProgramSpec> specs;
    std::unordered_map<MeshCoordinateRange, uint32_t> values;
    for (size_t i = 0; i < device_range_values.size(); ++i) {
        const auto& [device_range, value] = device_range_values[i];
        specs.emplace(device_range, make_l1_write_spec(name + "_" + std::to_string(i)));
        values.emplace(device_range, value);
    }

    auto workload = m2::MakeMeshWorkloadFromSpecs(mesh_device, specs);
    for (auto& [device_range, program] : workload.get_programs()) {
        m2::ProgramRunArgs args;
        args.kernel_run_args = {m2::ProgramRunArgs::KernelRunArgs{
            .kernel = m2::KernelSpecName{kKernelName},
            .runtime_arg_values = m2::MakeRuntimeArgsForSingleNode(kNode, {{"address", address}}),
            .common_runtime_arg_values = {{"value", values.at(device_range)}},
        }};
        m2::SetProgramRunArgs(program, args);
    }
    return workload;
}

IDevice* device(const std::shared_ptr<MeshDevice>& mesh_device) { return mesh_device->get_devices().at(0); }

IDevice* device(const std::shared_ptr<MeshDevice>& mesh_device, const MeshCoordinate& device_coord) {
    return mesh_device->get_device(device_coord);
}

void write_l1(const std::shared_ptr<MeshDevice>& mesh_device, uint32_t address, uint32_t value) {
    std::vector<uint32_t> data{value};
    ::tt::tt_metal::detail::WriteToDeviceL1(device(mesh_device), kNode, address, data);
}

void write_l1(
    const std::shared_ptr<MeshDevice>& mesh_device,
    const MeshCoordinate& device_coord,
    uint32_t address,
    uint32_t value) {
    std::vector<uint32_t> data{value};
    ::tt::tt_metal::detail::WriteToDeviceL1(device(mesh_device, device_coord), kNode, address, data);
}

uint32_t read_l1(const std::shared_ptr<MeshDevice>& mesh_device, uint32_t address) {
    std::vector<uint32_t> result;
    ::tt::tt_metal::detail::ReadFromDeviceL1(device(mesh_device), kNode, address, sizeof(uint32_t), result);
    EXPECT_EQ(result.size(), 1u);
    return result.at(0);
}

uint32_t read_l1(const std::shared_ptr<MeshDevice>& mesh_device, const MeshCoordinate& device_coord, uint32_t address) {
    std::vector<uint32_t> result;
    ::tt::tt_metal::detail::ReadFromDeviceL1(
        device(mesh_device, device_coord), kNode, address, sizeof(uint32_t), result);
    EXPECT_EQ(result.size(), 1u);
    return result.at(0);
}

TEST_F(CommandListTest, BuildsIndependentSnapshotsAndReplaysBlockingAndNonBlocking) {
    // Record the first workload, build a snapshot, then append a second workload
    // and build a new independent snapshot.
    auto workload_a = make_l1_write_workload(*mesh_device_, kAddressA, kValueA, "write_a");
    auto workload_b = make_l1_write_workload(*mesh_device_, kAddressB, kValueB, "write_b");
    auto& cq = mesh_device_->mesh_command_queue(0);

    CommandListBuilder builder(*mesh_device_);
    EXPECT_EQ(&builder.device(), mesh_device_.get());
    builder.add(workload_a);
    auto list_a = builder.build(cq);

    builder.add(workload_b);
    auto list_ab = builder.build(cq);
    EXPECT_EQ(&list_a.device(), mesh_device_.get());
    EXPECT_EQ(list_a.cq_id(), cq.id());
    EXPECT_EQ(list_ab.cq_id(), cq.id());

    // The first snapshot contains only workload A.
    write_l1(mesh_device_, kAddressA, 0);
    write_l1(mesh_device_, kAddressB, 0);
    EnqueueCommandList(cq, list_a, /*blocking=*/true);
    EXPECT_EQ(read_l1(mesh_device_, kAddressA), kValueA);
    EXPECT_EQ(read_l1(mesh_device_, kAddressB), 0u);

    // The second snapshot contains both workloads and also supports non-blocking enqueue.
    write_l1(mesh_device_, kAddressA, 0);
    write_l1(mesh_device_, kAddressB, 0);
    EnqueueCommandList(cq, list_ab, /*blocking=*/false);
    Finish(cq);
    EXPECT_EQ(read_l1(mesh_device_, kAddressA), kValueA);
    EXPECT_EQ(read_l1(mesh_device_, kAddressB), kValueB);
}

TEST_F(CommandListMultiDeviceTest, ReplaysMeshWideWorkloadOnEveryDevice) {
    auto workload = make_l1_write_workload(*mesh_device_, kAddressA, kValueA, "mesh_wide");
    auto& cq = mesh_device_->mesh_command_queue(0);

    CommandListBuilder builder(*mesh_device_);
    builder.add(workload);
    auto command_list = builder.build(cq);

    const MeshCoordinateRange all_devices(mesh_device_->shape());
    for (const auto& coord : all_devices) {
        write_l1(mesh_device_, coord, kAddressA, 0);
    }
    EnqueueCommandList(cq, command_list, /*blocking=*/true);
    for (const auto& coord : all_devices) {
        EXPECT_EQ(read_l1(mesh_device_, coord, kAddressA), kValueA) << "Device coordinate: " << coord;
    }

    for (const auto& coord : all_devices) {
        write_l1(mesh_device_, coord, kAddressA, 0);
    }
    EnqueueCommandList(cq, command_list, /*blocking=*/false);
    Finish(cq);
    for (const auto& coord : all_devices) {
        EXPECT_EQ(read_l1(mesh_device_, coord, kAddressA), kValueA) << "Device coordinate: " << coord;
    }
}

TEST_F(CommandListMultiDeviceTest, ReplaysHeterogeneousAndNonConvexDeviceRanges) {
    constexpr uint32_t kTopRowValue = 0xABCD0001;
    constexpr uint32_t kBottomRowValue = 0xABCD0002;
    constexpr uint32_t kTopLeftValue = 0xABCD0003;
    constexpr uint32_t kBottomRightValue = 0xABCD0004;

    const MeshCoordinateRange all_devices(mesh_device_->shape());
    const MeshCoordinateRange top_row({0, 0}, {0, 3});
    const MeshCoordinateRange bottom_row({1, 0}, {1, 3});
    const MeshCoordinateRange top_left({0, 0}, {0, 1});
    const MeshCoordinateRange bottom_right({1, 2}, {1, 3});

    auto mesh_wide = make_l1_write_workload(*mesh_device_, kAddressA, kValueA, "heterogeneous_mesh_wide");
    auto split_rows = make_l1_write_workload(
        *mesh_device_,
        kAddressB,
        std::vector<DeviceRangeValue>{{top_row, kTopRowValue}, {bottom_row, kBottomRowValue}},
        "split_rows");
    auto non_convex = make_l1_write_workload(
        *mesh_device_,
        kAddressC,
        std::vector<DeviceRangeValue>{{top_left, kTopLeftValue}, {bottom_right, kBottomRightValue}},
        "non_convex");

    auto& cq = mesh_device_->mesh_command_queue(0);
    CommandListBuilder builder(*mesh_device_);
    builder.add(mesh_wide);
    builder.add(split_rows);
    builder.add(non_convex);
    auto command_list = builder.build(cq);

    for (const auto& coord : all_devices) {
        write_l1(mesh_device_, coord, kAddressA, 0);
        write_l1(mesh_device_, coord, kAddressB, 0);
        write_l1(mesh_device_, coord, kAddressC, 0);
    }
    EnqueueCommandList(cq, command_list, /*blocking=*/true);

    for (const auto& coord : all_devices) {
        EXPECT_EQ(read_l1(mesh_device_, coord, kAddressA), kValueA) << "Device coordinate: " << coord;

        const uint32_t expected_row_value = top_row.contains(coord) ? kTopRowValue : kBottomRowValue;
        EXPECT_EQ(read_l1(mesh_device_, coord, kAddressB), expected_row_value) << "Device coordinate: " << coord;

        uint32_t expected_non_convex_value = 0;
        if (top_left.contains(coord)) {
            expected_non_convex_value = kTopLeftValue;
        } else if (bottom_right.contains(coord)) {
            expected_non_convex_value = kBottomRightValue;
        }
        EXPECT_EQ(read_l1(mesh_device_, coord, kAddressC), expected_non_convex_value) << "Device coordinate: " << coord;
    }
}

TEST_F(CommandListTest, AllowsTemporaryTensorLifetimeDuringBuild) {
    constexpr uint32_t num_pages = 2;
    constexpr uint32_t page_size = 1024;
    constexpr uint32_t total_bytes = num_pages * page_size;

    const auto page_config = PageConfig(Layout::ROW_MAJOR);
    const auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    const auto tensor_layout = TensorLayout(DataType::BFLOAT16, page_config, memory_config);
    const auto tensor_spec = TensorSpec(Shape{num_pages, 512}, tensor_layout);

    // Allocate the persistent output before command-list recording starts.
    MeshTensor output = MeshTensor::allocate_on_device(*mesh_device_, tensor_spec);
    const std::vector<uint32_t> zeros(total_bytes / sizeof(uint32_t), 0);
    ::tt::tt_metal::detail::WriteToBuffer(*output.mesh_buffer().get_reference_buffer(), zeros);

    std::vector<uint32_t> expected(total_bytes / sizeof(uint32_t));
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = 0xA0000000u + static_cast<uint32_t>(i);
    }

    // Create a workload that copies an input tensor to the persistent output.
    auto producer = MakeMinimalGen1DMKernel("producer", DataMovementProcessor::RISCV_0);
    producer.source = "tests/tt_metal/tt_metal/test_kernels/dataflow/tensor_accessor_loopback_producer.cpp";
    producer.advanced_options.num_runtime_varargs = 1;
    auto consumer = MakeMinimalGen1DMKernel("consumer", DataMovementProcessor::RISCV_1);
    consumer.source = "tests/tt_metal/tt_metal/test_kernels/dataflow/tensor_accessor_loopback_consumer.cpp";
    consumer.advanced_options.num_runtime_varargs = 1;

    auto dfb = MakeMinimalDFB("input_dfb", page_size, /*num_entries=*/2);
    dfb.data_format_metadata = DataFormat::Float16_b;
    producer.dfb_bindings.push_back(m2::ProducerOf(m2::DFBSpecName{"input_dfb"}, "input_dfb"));
    consumer.dfb_bindings.push_back(m2::ConsumerOf(m2::DFBSpecName{"input_dfb"}, "input_dfb"));
    BindTensorParameterToKernel(producer, "input_tensor", "input_tensor");
    BindTensorParameterToKernel(consumer, "output_tensor", "output_tensor");

    m2::ProgramSpec spec{
        .name = "temporary_tensor_lifetime",
        .kernels = {producer, consumer},
        .dataflow_buffers = {dfb},
        .tensor_parameters =
            {
                m2::TensorParameter{.unique_id = m2::TensorParamName{"input_tensor"}, .spec = tensor_spec},
                m2::TensorParameter{.unique_id = m2::TensorParamName{"output_tensor"}, .spec = tensor_spec},
            },
        .work_units = {MakeMinimalWorkUnit("main", kNode, {"producer", "consumer"})},
    };
    auto workload = m2::MakeMeshWorkloadFromSpec(*mesh_device_, spec);
    auto& program = workload.get_programs().begin()->second;

    // Allocate the input during recording and bind it to the workload added to the command list.
    CommandListBuilder builder(*mesh_device_);
    {
        MeshTensor temporary = MeshTensor::allocate_on_device(*mesh_device_, tensor_spec);
        ::tt::tt_metal::detail::WriteToBuffer(*temporary.mesh_buffer().get_reference_buffer(), expected);

        m2::ProgramRunArgs args;
        args.kernel_run_args = {
            m2::ProgramRunArgs::KernelRunArgs{
                .kernel = m2::KernelSpecName{"producer"},
                .advanced_options = m2::AdvancedKernelRunArgs{.runtime_varargs = {{kNode, {num_pages}}}},
            },
            m2::ProgramRunArgs::KernelRunArgs{
                .kernel = m2::KernelSpecName{"consumer"},
                .advanced_options = m2::AdvancedKernelRunArgs{.runtime_varargs = {{kNode, {num_pages}}}},
            },
        };
        args.tensor_args = {
            {m2::TensorParamName{"input_tensor"}, m2::ProgramRunArgs::TensorArgument{temporary}},
            {m2::TensorParamName{"output_tensor"}, m2::ProgramRunArgs::TensorArgument{output}},
        };
        m2::SetProgramRunArgs(program, args);
        builder.add(workload);
    }

    // Build and replay after the recorded input tensor has been deallocated.
    // Replay intentionally uses the captured raw address after the allocation is released. The stale DRAM contents
    // remain usable until another allocation reuses or overwrites that storage; no live MeshTensor handle is required.
    auto& cq = mesh_device_->mesh_command_queue(0);
    auto command_list = builder.build(cq);
    EnqueueCommandList(cq, command_list, /*blocking=*/true);

    std::vector<uint32_t> result;
    ::tt::tt_metal::detail::ReadFromBuffer(*output.mesh_buffer().get_reference_buffer(), result);
    EXPECT_EQ(result, expected);
}

TEST_F(CommandListTest, BuilderLifecyclePreservesBuiltLists) {
    // A live builder owns the device-wide active-builder reservation.
    auto workload = make_l1_write_workload(*mesh_device_, kAddressA, kValueA, "builder_lifecycle");
    auto& cq = mesh_device_->mesh_command_queue(0);

    CommandListBuilder builder(*mesh_device_);
    EXPECT_THAT(
        [&] { CommandListBuilder second_builder(*mesh_device_); },
        ThrowsMessage<std::runtime_error>(HasSubstr("Only one CommandListBuilder may exist")));
    EXPECT_THAT(
        [&] { mesh_device_->clear_loaded_sub_device_manager(); },
        ThrowsMessage<std::runtime_error>(HasSubstr("while a CommandListBuilder is active")));

    // Clearing removes the recording, after which build rejects the empty builder.
    builder.add(workload);
    builder.clear();
    EXPECT_THAT(
        [&] { (void)builder.build(cq); },
        ThrowsMessage<std::runtime_error>(HasSubstr("Cannot build an empty CommandList")));

    builder.add(workload);
    auto command_list = builder.build(cq);

    // Moving transfers the recording and reservation; explicit deallocation invalidates
    // the destination and releases the reservation for a replacement builder.
    CommandListBuilder moved_builder(std::move(builder));
    EXPECT_THAT(
        [&] { builder.add(workload); },  // NOLINT(bugprone-use-after-move)
        ThrowsMessage<std::runtime_error>(HasSubstr("CommandListBuilder has been moved from")));
    EXPECT_EQ(&moved_builder.device(), mesh_device_.get());
    moved_builder.deallocate();
    EXPECT_THAT(
        [&] { (void)moved_builder.device(); },
        ThrowsMessage<std::runtime_error>(HasSubstr("CommandListBuilder has been deallocated")));
    EXPECT_THAT(
        [&] { moved_builder.clear(); },
        ThrowsMessage<std::runtime_error>(HasSubstr("CommandListBuilder has been deallocated")));
    EXPECT_THAT(
        [&] { moved_builder.add(workload); },
        ThrowsMessage<std::runtime_error>(HasSubstr("CommandListBuilder has been deallocated")));
    EXPECT_THAT(
        [&] { (void)moved_builder.build(cq); },
        ThrowsMessage<std::runtime_error>(HasSubstr("CommandListBuilder has been deallocated")));

    EXPECT_NO_THROW({ CommandListBuilder replacement(*mesh_device_); });

    // A built list owns its serialized commands and required kernel binaries, so it
    // remains replayable without its builder.
    write_l1(mesh_device_, kAddressA, 0);
    EnqueueCommandList(cq, command_list, /*blocking=*/true);
    EXPECT_EQ(read_l1(mesh_device_, kAddressA), kValueA);
}

TEST_F(CommandListTest, CommandListMoveAndDeallocateInvalidateTheHandle) {
    // Move the built list and verify only the destination remains usable.
    auto workload = make_l1_write_workload(*mesh_device_, kAddressA, kValueA, "list_lifecycle");
    auto& cq = mesh_device_->mesh_command_queue(0);
    CommandListBuilder builder(*mesh_device_);
    builder.add(workload);
    auto command_list = builder.build(cq);

    CommandList moved_list(std::move(command_list));
    EXPECT_THAT(
        [&] { EnqueueCommandList(cq, command_list, /*blocking=*/true); },  // NOLINT(bugprone-use-after-move)
        ThrowsMessage<std::runtime_error>(HasSubstr("CommandList has been moved from")));

    write_l1(mesh_device_, kAddressA, 0);
    EnqueueCommandList(cq, moved_list, /*blocking=*/true);
    EXPECT_EQ(read_l1(mesh_device_, kAddressA), kValueA);

    // Explicit deallocation is idempotent and invalidates every remaining operation.
    moved_list.deallocate();
    moved_list.deallocate();
    EXPECT_THAT(
        [&] { EnqueueCommandList(cq, moved_list, /*blocking=*/true); },
        ThrowsMessage<std::runtime_error>(HasSubstr("CommandList has been deallocated")));
    EXPECT_THAT(
        [&] { (void)moved_list.device(); },
        ThrowsMessage<std::runtime_error>(HasSubstr("CommandList has been deallocated")));
    EXPECT_THAT(
        [&] { (void)moved_list.cq_id(); },
        ThrowsMessage<std::runtime_error>(HasSubstr("CommandList has been deallocated")));
}

TEST_F(CommandListTest, RejectsEmptyAndLegacyWorkloadsWithoutChangingTheRecording) {
    // An empty builder and an empty workload are rejected before anything is recorded.
    auto& cq = mesh_device_->mesh_command_queue(0);
    CommandListBuilder builder(*mesh_device_);

    EXPECT_THAT(
        [&] { (void)builder.build(cq); },
        ThrowsMessage<std::runtime_error>(HasSubstr("Cannot build an empty CommandList")));

    MeshWorkload empty_workload;
    EXPECT_THAT(
        [&] { builder.add(empty_workload); },
        ThrowsMessage<std::runtime_error>(HasSubstr("Cannot prepare an empty MeshWorkload")));

    // Command lists accept only ProgramSpec-created Metal 2.0 programs.
    MeshWorkload legacy_workload;
    legacy_workload.add_program(MeshCoordinateRange(mesh_device_->shape()), CreateProgram());
    EXPECT_THAT(
        [&] { builder.add(legacy_workload); },
        ThrowsMessage<std::runtime_error>(HasSubstr("Command lists only support Metal 2.0 programs")));

    // Both failed additions are transactional: the builder is still empty.
    EXPECT_THAT(
        [&] { (void)builder.build(cq); },
        ThrowsMessage<std::runtime_error>(HasSubstr("Cannot build an empty CommandList")));
}

TEST_F(CommandListTest, ReplaysCommandListsUnderTheirRecordedSubDeviceLayouts) {
    // Define two distinct layouts that both contain the workload's target node.
    const auto worker_grid = mesh_device_->compute_with_storage_grid_size();
    const SubDevice full_grid_sub_device(
        std::array{CoreRangeSet(CoreRange({0, 0}, {worker_grid.x - 1, worker_grid.y - 1}))});
    const SubDevice single_core_sub_device(std::array{CoreRangeSet(CoreRange({0, 0}, {0, 0}))});
    const auto full_grid_manager = mesh_device_->create_sub_device_manager({full_grid_sub_device}, 3200);
    const auto single_core_manager = mesh_device_->create_sub_device_manager({single_core_sub_device}, 3200);
    auto& cq = mesh_device_->mesh_command_queue(0);

    auto workload_a = make_l1_write_workload(*mesh_device_, kAddressA, kValueA, "full_grid_layout");
    auto workload_b = make_l1_write_workload(*mesh_device_, kAddressB, kValueB, "single_core_layout");

    // Record and build one list under each active layout.
    mesh_device_->load_sub_device_manager(full_grid_manager);
    auto list_a = [&] {
        CommandListBuilder builder(*mesh_device_);
        builder.add(workload_a);
        return builder.build(cq);
    }();

    mesh_device_->load_sub_device_manager(single_core_manager);
    auto list_b = [&] {
        CommandListBuilder builder(*mesh_device_);
        builder.add(workload_b);
        return builder.build(cq);
    }();

    // The second manager is still active, so the first list cannot replay until its
    // recorded manager is restored.
    EXPECT_THAT(
        [&] { EnqueueCommandList(cq, list_a, /*blocking=*/true); },
        ThrowsMessage<std::runtime_error>(HasSubstr("active sub-device manager changed")));

    write_l1(mesh_device_, kAddressB, 0);
    EnqueueCommandList(cq, list_b, /*blocking=*/true);
    EXPECT_EQ(read_l1(mesh_device_, kAddressB), kValueB);

    mesh_device_->load_sub_device_manager(full_grid_manager);
    EXPECT_THAT(
        [&] { EnqueueCommandList(cq, list_b, /*blocking=*/true); },
        ThrowsMessage<std::runtime_error>(HasSubstr("active sub-device manager changed")));

    write_l1(mesh_device_, kAddressA, 0);
    EnqueueCommandList(cq, list_a, /*blocking=*/true);
    EXPECT_EQ(read_l1(mesh_device_, kAddressA), kValueA);

    // Both lists remain live and can be alternated as long as the matching layout
    // is loaded before each replay.
    mesh_device_->load_sub_device_manager(single_core_manager);
    write_l1(mesh_device_, kAddressB, 0);
    EnqueueCommandList(cq, list_b, /*blocking=*/true);
    EXPECT_EQ(read_l1(mesh_device_, kAddressB), kValueB);

    mesh_device_->load_sub_device_manager(full_grid_manager);
    write_l1(mesh_device_, kAddressA, 0);
    EnqueueCommandList(cq, list_a, /*blocking=*/true);
    EXPECT_EQ(read_l1(mesh_device_, kAddressA), kValueA);
}

TEST_F(CommandListMultiCQTest, EnforcesTheCommandQueueUsedAtBuildTime) {
    // Build independent command lists from one recording, one per command queue.
    auto workload = make_l1_write_workload(*mesh_device_, kAddressA, kValueA, "cq_binding");
    auto& cq0 = mesh_device_->mesh_command_queue(0);
    auto& cq1 = mesh_device_->mesh_command_queue(1);

    CommandListBuilder builder(*mesh_device_);
    builder.add(workload);
    auto list0 = builder.build(cq0);
    auto list1 = builder.build(cq1);
    EXPECT_EQ(list0.cq_id(), cq0.id());
    EXPECT_EQ(list1.cq_id(), cq1.id());

    // The enqueue helper rejects a mismatched queue and accepts the bound queue.
    EXPECT_THAT(
        [&] { EnqueueCommandList(cq1, list0, /*blocking=*/true); },
        ThrowsMessage<std::runtime_error>(HasSubstr("CommandList was built for a different command queue")));
    write_l1(mesh_device_, kAddressA, 0);
    EXPECT_NO_THROW(EnqueueCommandList(cq1, list1, /*blocking=*/true));
    EXPECT_EQ(read_l1(mesh_device_, kAddressA), kValueA);
}

}  // namespace
}  // namespace tt::tt_metal::experimental::test
