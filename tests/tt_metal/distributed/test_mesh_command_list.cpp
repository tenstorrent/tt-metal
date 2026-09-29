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

IDevice* device(const std::shared_ptr<MeshDevice>& mesh_device) { return mesh_device->get_devices().at(0); }

void write_l1(const std::shared_ptr<MeshDevice>& mesh_device, uint32_t address, uint32_t value) {
    std::vector<uint32_t> data{value};
    ::tt::tt_metal::detail::WriteToDeviceL1(device(mesh_device), kNode, address, data);
}

uint32_t read_l1(const std::shared_ptr<MeshDevice>& mesh_device, uint32_t address) {
    std::vector<uint32_t> result;
    ::tt::tt_metal::detail::ReadFromDeviceL1(device(mesh_device), kNode, address, sizeof(uint32_t), result);
    EXPECT_EQ(result.size(), 1u);
    return result.at(0);
}

TEST_F(CommandListTest, BuildsIndependentSnapshotsAndReplaysThroughBothPublicEntryPoints) {
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
    list_a.replay(/*blocking=*/true);
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

TEST_F(CommandListTest, UpdatesRuntimeAndCommonRuntimeArgumentsTransactionally) {
    // Register the kernel's address RTA and value CRTA as command-list parameters.
    auto workload = make_l1_write_workload(*mesh_device_, kAddressA, kValueA, "patch_scalars");
    auto& program = workload.get_programs().begin()->second;
    auto& cq = mesh_device_->mesh_command_queue(0);

    const CmdListRuntimeArgName address_param{"destination"};
    const CmdListCommonRuntimeArgName value_param{"payload"};

    CommandListBuilder builder(*mesh_device_);
    builder.add(
        workload,
        CmdListParameters{
            .runtime_parameters =
                {{address_param,
                  std::vector<CmdListRuntimeArgInfo>{{
                      .program = std::cref(program),
                      .kernel_name = m2::KernelSpecName{kKernelName},
                      .arg_name = "address",
                      .nodes = {kNode},
                  }}}},
            .common_runtime_parameters =
                {{value_param,
                  std::vector<CmdListCommonRuntimeArgInfo>{{
                      .program = std::cref(program),
                      .kernel_name = m2::KernelSpecName{kKernelName},
                      .arg_name = "value",
                  }}}},
        });
    auto command_list = builder.build(cq);

    // An unknown parameter rejects the entire patch, leaving both original values intact.
    EXPECT_THAT(
        [&] {
            command_list.update_args(CmdListArgPatch{
                .runtime_args = {{address_param, kAddressB}, {CmdListRuntimeArgName{"unknown"}, kAddressB}},
                .common_runtime_args = {{value_param, kValueB}},
            });
        },
        ThrowsMessage<std::runtime_error>(HasSubstr("Unknown command-list runtime parameter")));

    write_l1(mesh_device_, kAddressA, 0);
    write_l1(mesh_device_, kAddressB, 0);
    command_list.replay(/*blocking=*/true);
    EXPECT_EQ(read_l1(mesh_device_, kAddressA), kValueA);
    EXPECT_EQ(read_l1(mesh_device_, kAddressB), 0u);

    // A valid patch retargets both scalar parameters for subsequent replays.
    command_list.update_args(CmdListArgPatch{
        .runtime_args = {{address_param, kAddressB}},
        .common_runtime_args = {{value_param, kValueB}},
    });

    write_l1(mesh_device_, kAddressA, 0);
    write_l1(mesh_device_, kAddressB, 0);
    command_list.replay(/*blocking=*/true);
    EXPECT_EQ(read_l1(mesh_device_, kAddressA), 0u);
    EXPECT_EQ(read_l1(mesh_device_, kAddressB), kValueB);
}

TEST_F(CommandListTest, UpdatesTensorArgumentForSubsequentReplays) {
    // Create two compatible inputs and one output for a tensor-accessor loopback.
    constexpr uint32_t num_pages = 2;
    constexpr uint32_t page_size = 1024;
    constexpr uint32_t total_bytes = num_pages * page_size;

    const auto page_config = PageConfig(Layout::ROW_MAJOR);
    const auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    const auto tensor_layout = TensorLayout(DataType::BFLOAT16, page_config, memory_config);
    const auto tensor_spec = TensorSpec(Shape{num_pages, 512}, tensor_layout);

    MeshTensor input_a = MeshTensor::allocate_on_device(*mesh_device_, tensor_spec);
    MeshTensor input_b = MeshTensor::allocate_on_device(*mesh_device_, tensor_spec);
    MeshTensor output = MeshTensor::allocate_on_device(*mesh_device_, tensor_spec);

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

    // Bind input A when recording and expose only that input as patchable.
    m2::ProgramSpec spec{
        .name = "tensor_patch",
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
        {m2::TensorParamName{"input_tensor"}, m2::ProgramRunArgs::TensorArgument{input_a}},
        {m2::TensorParamName{"output_tensor"}, m2::ProgramRunArgs::TensorArgument{output}},
    };
    m2::SetProgramRunArgs(program, args);

    const CmdListTensorArgName input_param{"input"};

    CommandListBuilder builder(*mesh_device_);
    builder.add(
        workload,
        CmdListParameters{
            .tensor_parameters =
                {{input_param,
                  std::vector<CmdListTensorArgInfo>{{
                      .program = std::cref(program),
                      .param_name = m2::TensorParamName{"input_tensor"},
                  }}}},
        });
    auto command_list = builder.build(mesh_device_->mesh_command_queue(0));

    std::vector<uint32_t> data_a(total_bytes / sizeof(uint32_t));
    std::vector<uint32_t> data_b(total_bytes / sizeof(uint32_t));
    for (size_t i = 0; i < data_a.size(); ++i) {
        data_a[i] = static_cast<uint32_t>(i);
        data_b[i] = 0x80000000u + static_cast<uint32_t>(i);
    }
    const std::vector<uint32_t> zeros(data_a.size(), 0);
    ::tt::tt_metal::detail::WriteToBuffer(*input_a.mesh_buffer().get_reference_buffer(), data_a);
    ::tt::tt_metal::detail::WriteToBuffer(*input_b.mesh_buffer().get_reference_buffer(), data_b);
    ::tt::tt_metal::detail::WriteToBuffer(*output.mesh_buffer().get_reference_buffer(), zeros);

    // The initial replay copies input A to the output.
    command_list.replay(/*blocking=*/true);
    std::vector<uint32_t> result;
    ::tt::tt_metal::detail::ReadFromBuffer(*output.mesh_buffer().get_reference_buffer(), result);
    EXPECT_EQ(result, data_a);

    // Patch to input B and verify the next replay uses the replacement tensor.
    command_list.update_args(CmdListArgPatch{
        .tensor_args = {{input_param, m2::ProgramRunArgs::TensorArgument{input_b}}},
    });
    ::tt::tt_metal::detail::WriteToBuffer(*output.mesh_buffer().get_reference_buffer(), zeros);
    command_list.replay(/*blocking=*/true);
    result.clear();
    ::tt::tt_metal::detail::ReadFromBuffer(*output.mesh_buffer().get_reference_buffer(), result);
    EXPECT_EQ(result, data_b);
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
    auto command_list = builder.build(mesh_device_->mesh_command_queue(0));
    command_list.replay(/*blocking=*/true);

    std::vector<uint32_t> result;
    ::tt::tt_metal::detail::ReadFromBuffer(*output.mesh_buffer().get_reference_buffer(), result);
    EXPECT_EQ(result, expected);
}

TEST_F(CommandListTest, BuilderLifecyclePreservesBuiltLists) {
    // A live builder owns the device-wide builder lock.
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

    // Moving transfers the recording and lock; explicit deallocation invalidates
    // the destination and releases the lock for a replacement builder.
    CommandListBuilder moved_builder(std::move(builder));
    EXPECT_THAT(
        [&] { builder.add(workload); },
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

    // A built list owns everything it needs and remains replayable without its builder.
    write_l1(mesh_device_, kAddressA, 0);
    command_list.replay(/*blocking=*/true);
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
        [&] { command_list.replay(/*blocking=*/true); },
        ThrowsMessage<std::runtime_error>(HasSubstr("CommandList has been moved from")));

    write_l1(mesh_device_, kAddressA, 0);
    moved_list.replay(/*blocking=*/true);
    EXPECT_EQ(read_l1(mesh_device_, kAddressA), kValueA);

    // Explicit deallocation is idempotent and invalidates every remaining operation.
    moved_list.deallocate();
    moved_list.deallocate();
    EXPECT_THAT(
        [&] { moved_list.replay(/*blocking=*/true); },
        ThrowsMessage<std::runtime_error>(HasSubstr("CommandList has been deallocated")));
    EXPECT_THAT(
        [&] { (void)moved_list.device(); },
        ThrowsMessage<std::runtime_error>(HasSubstr("CommandList has been deallocated")));
    EXPECT_THAT(
        [&] { (void)moved_list.cq_id(); },
        ThrowsMessage<std::runtime_error>(HasSubstr("CommandList has been deallocated")));
    EXPECT_THAT(
        [&] { moved_list.update_args({}); },
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

TEST_F(CommandListTest, CommandListsBlockTraceCaptureUntilAllHandlesAreReleased) {
    auto workload = make_l1_write_workload(*mesh_device_, kAddressA, kValueA, "trace_exclusion");
    auto& cq = mesh_device_->mesh_command_queue(0);

    CommandListBuilder builder(*mesh_device_);
    EXPECT_THAT(
        [&] { (void)mesh_device_->begin_mesh_trace(cq); },
        ThrowsMessage<std::runtime_error>(HasSubstr("CommandListBuilder or CommandList exists")));

    builder.add(workload);
    auto list_a = builder.build(cq);
    auto list_b = builder.build(cq);
    builder.deallocate();

    EXPECT_THAT(
        [&] { (void)mesh_device_->begin_mesh_trace(cq); },
        ThrowsMessage<std::runtime_error>(HasSubstr("CommandListBuilder or CommandList exists")));
    list_a.deallocate();
    EXPECT_THAT(
        [&] { (void)mesh_device_->begin_mesh_trace(cq); },
        ThrowsMessage<std::runtime_error>(HasSubstr("CommandListBuilder or CommandList exists")));

    list_b.deallocate();
    const auto trace_id = mesh_device_->begin_mesh_trace(cq);
    EnqueueMeshWorkload(cq, workload, false);
    mesh_device_->end_mesh_trace(cq, trace_id);
    mesh_device_->release_mesh_trace(trace_id);
}

TEST_F(CommandListTest, TracesBlockBuildersAcrossSubDeviceManagerChanges) {
    const auto worker_grid = mesh_device_->compute_with_storage_grid_size();
    const SubDevice full_grid_sub_device(
        std::array{CoreRangeSet(CoreRange({0, 0}, {worker_grid.x - 1, worker_grid.y - 1}))});
    const auto trace_manager = mesh_device_->create_sub_device_manager({full_grid_sub_device}, 3200);
    auto workload = make_l1_write_workload(*mesh_device_, kAddressA, kValueA, "inactive_manager_trace");
    auto& cq = mesh_device_->mesh_command_queue(0);

    mesh_device_->load_sub_device_manager(trace_manager);
    EnqueueMeshWorkload(cq, workload, false);
    Finish(cq);

    const auto trace_id = mesh_device_->begin_mesh_trace(cq);
    EXPECT_THAT(
        [&] { CommandListBuilder builder(*mesh_device_); },
        ThrowsMessage<std::runtime_error>(HasSubstr("while a trace exists")));
    EnqueueMeshWorkload(cq, workload, false);
    mesh_device_->end_mesh_trace(cq, trace_id);

    mesh_device_->clear_loaded_sub_device_manager();
    EXPECT_THAT(
        [&] { CommandListBuilder builder(*mesh_device_); },
        ThrowsMessage<std::runtime_error>(HasSubstr("while a trace exists")));

    mesh_device_->load_sub_device_manager(trace_manager);
    mesh_device_->release_mesh_trace(trace_id);
    mesh_device_->clear_loaded_sub_device_manager();
    EXPECT_NO_THROW({ CommandListBuilder builder(*mesh_device_); });
    mesh_device_->remove_sub_device_manager(trace_manager);
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
        [&] { list_a.replay(/*blocking=*/true); },
        ThrowsMessage<std::runtime_error>(HasSubstr("active sub-device manager changed")));

    write_l1(mesh_device_, kAddressB, 0);
    list_b.replay(/*blocking=*/true);
    EXPECT_EQ(read_l1(mesh_device_, kAddressB), kValueB);

    mesh_device_->load_sub_device_manager(full_grid_manager);
    EXPECT_THAT(
        [&] { list_b.replay(/*blocking=*/true); },
        ThrowsMessage<std::runtime_error>(HasSubstr("active sub-device manager changed")));

    write_l1(mesh_device_, kAddressA, 0);
    list_a.replay(/*blocking=*/true);
    EXPECT_EQ(read_l1(mesh_device_, kAddressA), kValueA);

    // Both lists remain live and can be alternated as long as the matching layout
    // is loaded before each replay.
    mesh_device_->load_sub_device_manager(single_core_manager);
    write_l1(mesh_device_, kAddressB, 0);
    list_b.replay(/*blocking=*/true);
    EXPECT_EQ(read_l1(mesh_device_, kAddressB), kValueB);

    mesh_device_->load_sub_device_manager(full_grid_manager);
    write_l1(mesh_device_, kAddressA, 0);
    list_a.replay(/*blocking=*/true);
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
