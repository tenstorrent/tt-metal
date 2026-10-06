// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/allocation_context.hpp>
#include <tt-metalium/experimental/command_list.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "tests/tt_metal/tt_metal/api/metal2_host_api/test_helpers.hpp"
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "tt_metal/distributed/fd_mesh_command_queue.hpp"

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

m2::ProgramSpec make_l1_write_spec(const std::string& name, const m2::Nodes& target_nodes = kNode) {
    const m2::KernelSpecName kernel_name{kKernelName};
    auto kernel = MakeMinimalGen1DMKernel(kKernelName, DataMovementProcessor::RISCV_0);
    kernel.source = "tests/tt_metal/tt_metal/test_kernels/dataflow/command_list_l1_write.cpp";
    kernel.runtime_arg_schema = {.runtime_arg_names = {"address"}, .common_runtime_arg_names = {"value"}};

    return m2::ProgramSpec{
        .name = name,
        .kernels = {kernel},
        .work_units = {m2::WorkUnitSpec{.name = "main", .kernels = {kernel_name}, .target_nodes = target_nodes}},
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

CmdListRuntimeArgInfo l1_write_address_arg(const Program& program) {
    return {
        .program = std::cref(program),
        .kernel_name = m2::KernelSpecName{kKernelName},
        .arg_name = "address",
        .nodes = {kNode}};
}

CmdListCommonRuntimeArgInfo l1_write_value_arg(const Program& program) {
    return {.program = std::cref(program), .kernel_name = m2::KernelSpecName{kKernelName}, .arg_name = "value"};
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

TEST_F(CommandListTest, PatchesBufferWindowsAtTheirInterleavedAddresses) {
    // Lay buffers out like command lists: interleaved over every DRAM bank with several pages per bank.
    auto& cq = mesh_device_->mesh_command_queue(0);
    auto& fd_cq = dynamic_cast<FDMeshCommandQueue&>(cq);
    const uint32_t window_size = hal::get_dram_alignment();
    const uint32_t window_words = window_size / sizeof(uint32_t);
    const uint32_t page_size = 1024;
    const uint32_t num_banks = mesh_device_->allocator()->get_num_banks(BufferType::DRAM);
    const uint32_t num_pages = num_banks * 3;
    const MeshCoordinateRange all_devices(mesh_device_->shape());

    for (const auto buffer_type : {BufferType::TRACE, BufferType::DRAM}) {
        std::shared_ptr<MeshBuffer> buffer;
        {
            auto allocation_context = make_allocation_context_guard("trace_storage");
            buffer = MeshBuffer::create(
                ReplicatedBufferConfig{.size = num_pages * page_size},
                DeviceLocalBufferConfig{.page_size = page_size, .buffer_type = buffer_type},
                mesh_device_.get());
        }
        std::vector<uint32_t> expected(num_pages * page_size / sizeof(uint32_t));
        for (size_t i = 0; i < expected.size(); ++i) {
            expected[i] = static_cast<uint32_t>(i);
        }
        EnqueueWriteMeshBuffer(cq, buffer, expected, /*blocking=*/true);

        // Patch the first and last window of pages in the first, second, last, and wrapped-around banks.
        std::vector<uint32_t> offsets;
        for (const uint32_t page : {0u, 1u, num_banks - 1, num_banks, 2 * num_banks + 1}) {
            offsets.push_back(page * page_size);
            offsets.push_back(((page + 1) * page_size) - window_size);
        }
        std::vector<std::vector<uint32_t>> patch_data(offsets.size(), std::vector<uint32_t>(window_words));
        std::vector<MeshBufferPatch> patches;
        for (size_t i = 0; i < offsets.size(); ++i) {
            for (uint32_t word = 0; word < window_words; ++word) {
                patch_data[i][word] = 0xA5000000u | (offsets[i] + word);
            }
            std::copy(patch_data[i].begin(), patch_data[i].end(), expected.begin() + offsets[i] / sizeof(uint32_t));
            patches.push_back({.device_range = all_devices, .offset = offsets[i], .data = patch_data[i]});
        }
        fd_cq.enqueue_command_list_patch(*buffer, patches);

        std::vector<uint32_t> result;
        EnqueueReadMeshBuffer(cq, result, buffer, /*blocking=*/true);
        EXPECT_EQ(result, expected) << "buffer type " << static_cast<int>(buffer_type);
    }
}

TEST_F(CommandListTest, SplitsLargePatchesAcrossPrefetchCommands) {
    // Patch every window of a 1 MB buffer. The data alone is 8x the 128 KB limit of one prefetch command, so the
    // writes must be split across several fetch-queue entries.
    constexpr uint32_t buffer_size = 1 << 20;
    auto& cq = mesh_device_->mesh_command_queue(0);
    auto buffer = MeshBuffer::create(
        ReplicatedBufferConfig{.size = buffer_size},
        DeviceLocalBufferConfig{.page_size = 1024, .buffer_type = BufferType::DRAM},
        mesh_device_.get());

    // Start from zeros so a dropped or misplaced batch shows up in the read-back.
    std::vector<uint32_t> zeros(buffer_size / sizeof(uint32_t), 0);
    EnqueueWriteMeshBuffer(cq, buffer, zeros, /*blocking=*/true);

    // Each patch is one window, carrying its slice of the expected contents.
    std::vector<uint32_t> expected(buffer_size / sizeof(uint32_t));
    std::iota(expected.begin(), expected.end(), 0xB0000000u);
    const uint32_t window_words = hal::get_dram_alignment() / sizeof(uint32_t);
    std::vector<MeshBufferPatch> patches;
    for (uint32_t word = 0; word < expected.size(); word += window_words) {
        patches.push_back(
            {.device_range = MeshCoordinateRange(mesh_device_->shape()),
             .offset = word * static_cast<uint32_t>(sizeof(uint32_t)),
             .data = ttsl::Span<const uint32_t>(expected).subspan(word, window_words)});
    }
    dynamic_cast<FDMeshCommandQueue&>(cq).enqueue_command_list_patch(*buffer, patches);

    std::vector<uint32_t> result;
    EnqueueReadMeshBuffer(cq, result, buffer, /*blocking=*/true);
    EXPECT_EQ(result, expected);
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
            .runtime_parameters = {{address_param, std::vector<CmdListRuntimeArgInfo>{l1_write_address_arg(program)}}},
            .common_runtime_parameters =
                {{value_param, std::vector<CmdListCommonRuntimeArgInfo>{l1_write_value_arg(program)}}},
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
    EXPECT_THAT(
        [&] {
            command_list.update_args(CmdListArgPatch{
                .runtime_args = {{address_param, kAddressB}},
                .common_runtime_args = {{CmdListCommonRuntimeArgName{"unknown"}, kValueB}},
            });
        },
        ThrowsMessage<std::runtime_error>(HasSubstr("Unknown command-list common-runtime parameter")));

    // Tensor arguments are checked last, so this rejection comes after both valid scalars were already staged.
    MeshTensor unused = MeshTensor::allocate_on_device(
        *mesh_device_,
        TensorSpec(
            Shape{1, 512},
            TensorLayout(
                DataType::BFLOAT16,
                PageConfig(Layout::ROW_MAJOR),
                MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM})));
    EXPECT_THAT(
        [&] {
            command_list.update_args(CmdListArgPatch{
                .tensor_args = {{CmdListTensorArgName{"unknown"}, m2::ProgramRunArgs::TensorArgument{unused}}},
                .runtime_args = {{address_param, kAddressB}},
                .common_runtime_args = {{value_param, kValueB}},
            });
        },
        ThrowsMessage<std::runtime_error>(HasSubstr("Unknown command-list tensor parameter")));

    write_l1(mesh_device_, kAddressA, 0);
    write_l1(mesh_device_, kAddressB, 0);
    EnqueueCommandList(cq, command_list, /*blocking=*/true);
    EXPECT_EQ(read_l1(mesh_device_, kAddressA), kValueA);
    EXPECT_EQ(read_l1(mesh_device_, kAddressB), 0u);

    // A valid patch retargets both scalar parameters for subsequent replays.
    command_list.update_args(CmdListArgPatch{
        .runtime_args = {{address_param, kAddressB}},
        .common_runtime_args = {{value_param, kValueB}},
    });

    write_l1(mesh_device_, kAddressA, 0);
    write_l1(mesh_device_, kAddressB, 0);
    EnqueueCommandList(cq, command_list, /*blocking=*/true);
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
    auto& cq = mesh_device_->mesh_command_queue(0);
    auto command_list = builder.build(cq);

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
    EnqueueCommandList(cq, command_list, /*blocking=*/true);
    std::vector<uint32_t> result;
    ::tt::tt_metal::detail::ReadFromBuffer(*output.mesh_buffer().get_reference_buffer(), result);
    EXPECT_EQ(result, data_a);

    // A tensor that does not match the declared TensorSpec is rejected.
    MeshTensor mismatched =
        MeshTensor::allocate_on_device(*mesh_device_, TensorSpec(Shape{num_pages, 256}, tensor_layout));
    EXPECT_THAT(
        [&] {
            command_list.update_args(CmdListArgPatch{
                .tensor_args = {{input_param, m2::ProgramRunArgs::TensorArgument{mismatched}}},
            });
        },
        ThrowsMessage<std::runtime_error>(HasSubstr("does not match its declared TensorSpec")));

    // Patch to input B and verify the next replay uses the replacement tensor.
    command_list.update_args(CmdListArgPatch{
        .tensor_args = {{input_param, m2::ProgramRunArgs::TensorArgument{input_b}}},
    });
    ::tt::tt_metal::detail::WriteToBuffer(*output.mesh_buffer().get_reference_buffer(), zeros);
    EnqueueCommandList(cq, command_list, /*blocking=*/true);
    result.clear();
    ::tt::tt_metal::detail::ReadFromBuffer(*output.mesh_buffer().get_reference_buffer(), result);
    EXPECT_EQ(result, data_b);
}

TEST_F(CommandListTest, UpdatesTensorBindingsThatCrossPatchWindows) {
    // A dynamic-shape row-major binding is two CRTA words: the base address and the aligned page size. Each extra
    // named CRTA ahead of it moves the binding one word, so across one window's worth of layouts some binding
    // straddles a window boundary.
    const auto tensor_layout = TensorLayout(
        DataType::BFLOAT16,
        PageConfig(Layout::ROW_MAJOR),
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM});
    MeshTensor recorded = MeshTensor::allocate_on_device(*mesh_device_, TensorSpec(Shape{2, 512}, tensor_layout));
    MeshTensor patched = MeshTensor::allocate_on_device(*mesh_device_, TensorSpec(Shape{2, 256}, tensor_layout));
    const uint32_t patched_page_size = patched.mesh_buffer().get_reference_buffer()->aligned_page_size();
    const CmdListTensorArgName tensor_param{"tensor"};
    auto& cq = mesh_device_->mesh_command_queue(0);

    const uint32_t window_words = hal::get_dram_alignment() / sizeof(uint32_t);
    for (uint32_t num_leading = 0; num_leading < window_words; ++num_leading) {
        auto kernel = MakeMinimalGen1DMKernel(kKernelName, DataMovementProcessor::RISCV_0);
        kernel.source = KernelSpec::SourceCode{R"(
#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
void kernel_main() {
    TensorAccessor accessor(tensor::io);
    volatile tt_l1_ptr uint32_t* destination = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg(args::address));
    destination[0] = accessor.get_bank_base_address();
    destination[1] = accessor.get_aligned_page_size();
}
)"};
        kernel.runtime_arg_schema.runtime_arg_names = {"address"};
        m2::ProgramRunArgs::KernelRunArgs::CommonRuntimeArgValues leading_values;
        for (uint32_t i = 0; i < num_leading; ++i) {
            const std::string name = "leading" + std::to_string(i);
            kernel.runtime_arg_schema.common_runtime_arg_names.push_back(name);
            leading_values[name] = i;
        }
        BindTensorParameterToKernel(kernel, "io", "io");

        m2::ProgramSpec spec{
            .name = "window_crossing_" + std::to_string(num_leading),
            .kernels = {kernel},
            .tensor_parameters = {m2::TensorParameter{
                .unique_id = m2::TensorParamName{"io"},
                .spec = recorded.tensor_spec(),
                .relaxations = TensorSpecRelaxations{.dynamic_tensor_shape = true}}},
            .work_units = {MakeMinimalWorkUnit("main", kNode, {kKernelName})},
        };
        auto workload = m2::MakeMeshWorkloadFromSpec(*mesh_device_, spec);
        auto& program = workload.get_programs().begin()->second;
        m2::ProgramRunArgs args;
        args.kernel_run_args = {m2::ProgramRunArgs::KernelRunArgs{
            .kernel = m2::KernelSpecName{kKernelName},
            .runtime_arg_values = m2::MakeRuntimeArgsForSingleNode(kNode, {{"address", kAddressA}}),
            .common_runtime_arg_values = leading_values,
        }};
        args.tensor_args = {{m2::TensorParamName{"io"}, m2::ProgramRunArgs::TensorArgument{recorded}}};
        m2::SetProgramRunArgs(program, args);

        CommandListBuilder builder(*mesh_device_);
        builder.add(
            workload,
            CmdListParameters{
                .tensor_parameters =
                    {{tensor_param,
                      std::vector<CmdListTensorArgInfo>{{
                          .program = std::cref(program),
                          .param_name = m2::TensorParamName{"io"},
                      }}}},
            });
        auto command_list = builder.build(cq);
        command_list.update_args(
            CmdListArgPatch{.tensor_args = {{tensor_param, m2::ProgramRunArgs::TensorArgument{patched}}}});

        write_l1(mesh_device_, kAddressA, 0);
        write_l1(mesh_device_, kAddressA + sizeof(uint32_t), 0);
        EnqueueCommandList(cq, command_list, /*blocking=*/true);
        EXPECT_EQ(read_l1(mesh_device_, kAddressA), patched.address()) << num_leading << " leading CRTAs";
        EXPECT_EQ(read_l1(mesh_device_, kAddressA + sizeof(uint32_t)), patched_page_size)
            << num_leading << " leading CRTAs";
    }
}

TEST_F(CommandListTest, PatchingOneParameterKeepsEarlierPatchesInItsWindow) {
    // Three adjacent CRTAs, each its own parameter. A window holds at least 8 words, so at least two of the three
    // share a window wherever they land in the stream.
    const std::vector<std::string> crta_names = {"first", "second", "third"};
    auto kernel = MakeMinimalGen1DMKernel(kKernelName, DataMovementProcessor::RISCV_0);
    kernel.source = KernelSpec::SourceCode{R"(
#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
void kernel_main() {
    volatile tt_l1_ptr uint32_t* destination = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg(args::address));
    destination[0] = get_arg(args::first);
    destination[1] = get_arg(args::second);
    destination[2] = get_arg(args::third);
}
)"};
    kernel.runtime_arg_schema = {.runtime_arg_names = {"address"}, .common_runtime_arg_names = crta_names};
    auto workload = m2::MakeMeshWorkloadFromSpec(
        *mesh_device_,
        m2::ProgramSpec{
            .name = "window_neighbors",
            .kernels = {kernel},
            .work_units = {MakeMinimalWorkUnit("main", kNode, {kKernelName})},
        });
    auto& program = workload.get_programs().begin()->second;
    m2::ProgramRunArgs args;
    args.kernel_run_args = {m2::ProgramRunArgs::KernelRunArgs{
        .kernel = m2::KernelSpecName{kKernelName},
        .runtime_arg_values = m2::MakeRuntimeArgsForSingleNode(kNode, {{"address", kAddressA}}),
        .common_runtime_arg_values = {{"first", 0}, {"second", 0}, {"third", 0}},
    }};
    m2::SetProgramRunArgs(program, args);

    CmdListParameters parameters;
    for (const auto& name : crta_names) {
        parameters.common_runtime_parameters[CmdListCommonRuntimeArgName{name}] = {
            {.program = std::cref(program), .kernel_name = m2::KernelSpecName{kKernelName}, .arg_name = name}};
    }
    CommandListBuilder builder(*mesh_device_);
    builder.add(workload, parameters);
    auto& cq = mesh_device_->mesh_command_queue(0);
    auto command_list = builder.build(cq);

    // Patch one parameter at a time. A patch built from stale window contents would write the recorded 0 back over
    // a neighbor that an earlier patch changed.
    for (uint32_t i = 0; i < crta_names.size(); ++i) {
        command_list.update_args(
            CmdListArgPatch{.common_runtime_args = {{CmdListCommonRuntimeArgName{crta_names[i]}, kValueA + i}}});
    }

    // One replay writes all three values, showing whether every patch survived.
    for (uint32_t i = 0; i < crta_names.size(); ++i) {
        write_l1(mesh_device_, kAddressA + (i * sizeof(uint32_t)), 0);
    }
    EnqueueCommandList(cq, command_list, /*blocking=*/true);
    for (uint32_t i = 0; i < crta_names.size(); ++i) {
        EXPECT_EQ(read_l1(mesh_device_, kAddressA + (i * sizeof(uint32_t))), kValueA + i) << crta_names[i];
    }
}

TEST_F(CommandListTest, UpdatesEveryTargetOfAParameter) {
    // One workload runs on two nodes; a second, added separately, runs on one. "destination" targets the RTA on both
    // nodes of the first workload, and "payload" targets the CRTA of both workloads, from two add() calls.
    constexpr uint32_t kPatchedValue = 0xCAFE0003;
    const m2::NodeCoord other_node{1, 0};
    auto two_nodes =
        m2::MakeMeshWorkloadFromSpec(*mesh_device_, make_l1_write_spec("two_nodes", m2::NodeRange(kNode, other_node)));
    auto& two_node_program = two_nodes.get_programs().begin()->second;
    auto rtas = m2::MakeRuntimeArgsForSingleNode(kNode, {{"address", kAddressA}});
    m2::AddRuntimeArgsForNode(rtas, other_node, {{"address", kAddressA}});
    m2::ProgramRunArgs args;
    args.kernel_run_args = {m2::ProgramRunArgs::KernelRunArgs{
        .kernel = m2::KernelSpecName{kKernelName},
        .runtime_arg_values = rtas,
        .common_runtime_arg_values = {{"value", kValueA}},
    }};
    m2::SetProgramRunArgs(two_node_program, args);
    auto one_node = make_l1_write_workload(*mesh_device_, kAddressC, kValueB, "one_node");
    auto& one_node_program = one_node.get_programs().begin()->second;

    const CmdListRuntimeArgName address_param{"destination"};
    const CmdListCommonRuntimeArgName value_param{"payload"};
    auto two_node_address = l1_write_address_arg(two_node_program);
    two_node_address.nodes = {kNode, other_node};
    CommandListBuilder builder(*mesh_device_);
    builder.add(
        two_nodes,
        CmdListParameters{
            .runtime_parameters = {{address_param, std::vector<CmdListRuntimeArgInfo>{two_node_address}}},
            .common_runtime_parameters =
                {{value_param, std::vector<CmdListCommonRuntimeArgInfo>{l1_write_value_arg(two_node_program)}}},
        });
    builder.add(
        one_node,
        CmdListParameters{
            .common_runtime_parameters =
                {{value_param, std::vector<CmdListCommonRuntimeArgInfo>{l1_write_value_arg(one_node_program)}}},
        });
    auto& cq = mesh_device_->mesh_command_queue(0);
    auto command_list = builder.build(cq);

    // One patch moves both nodes' writes to kAddressB and changes the value written by both workloads.
    command_list.update_args(CmdListArgPatch{
        .runtime_args = {{address_param, kAddressB}},
        .common_runtime_args = {{value_param, kPatchedValue}},
    });

    // Both nodes write the new value to the new address. The second workload keeps its address, but takes the value.
    auto read_node = [&](const m2::NodeCoord& node, uint32_t address) {
        std::vector<uint32_t> result;
        ::tt::tt_metal::detail::ReadFromDeviceL1(device(mesh_device_), node, address, sizeof(uint32_t), result);
        return result.at(0);
    };
    std::vector<uint32_t> zero{0};
    for (const auto& node : {kNode, other_node}) {
        for (const uint32_t address : {kAddressA, kAddressB, kAddressC}) {
            ::tt::tt_metal::detail::WriteToDeviceL1(device(mesh_device_), node, address, zero);
        }
    }
    EnqueueCommandList(cq, command_list, /*blocking=*/true);
    for (const auto& node : {kNode, other_node}) {
        EXPECT_EQ(read_node(node, kAddressA), 0u) << node.str();
        EXPECT_EQ(read_node(node, kAddressB), kPatchedValue) << node.str();
    }
    EXPECT_EQ(read_node(kNode, kAddressC), kPatchedValue);
}

TEST_F(CommandListTest, OrdersNonBlockingPatchesWithReplays) {
    // Each iteration retargets the write to its own L1 slot and value without waiting. A replay that fetched its
    // commands before the preceding patch landed would leave its slot unwritten.
    constexpr uint32_t kNumIterations = 1000;
    // The 4 KB of slots would overlap kernel binaries if placed at kAddressA, so they start at the L1 allocator base.
    const auto slots_address =
        static_cast<uint32_t>(mesh_device_->allocator()->get_base_allocator_addr(HalMemType::L1));
    auto workload = make_l1_write_workload(*mesh_device_, slots_address, kValueA, "patch_ordering");
    auto& program = workload.get_programs().begin()->second;
    auto& cq = mesh_device_->mesh_command_queue(0);

    const CmdListRuntimeArgName address_param{"destination"};
    const CmdListCommonRuntimeArgName value_param{"payload"};
    CommandListBuilder builder(*mesh_device_);
    builder.add(
        workload,
        CmdListParameters{
            .runtime_parameters = {{address_param, std::vector<CmdListRuntimeArgInfo>{l1_write_address_arg(program)}}},
            .common_runtime_parameters =
                {{value_param, std::vector<CmdListCommonRuntimeArgInfo>{l1_write_value_arg(program)}}},
        });
    auto command_list = builder.build(cq);

    std::vector<uint32_t> slots(kNumIterations, 0);
    ::tt::tt_metal::detail::WriteToDeviceL1(device(mesh_device_), kNode, slots_address, slots);
    for (uint32_t i = 0; i < kNumIterations; ++i) {
        command_list.update_args(CmdListArgPatch{
            .runtime_args = {{address_param, slots_address + (i * sizeof(uint32_t))}},
            .common_runtime_args = {{value_param, kValueA + i}},
        });
        EnqueueCommandList(cq, command_list, /*blocking=*/false);
    }
    Finish(cq);

    ::tt::tt_metal::detail::ReadFromDeviceL1(
        device(mesh_device_), kNode, slots_address, kNumIterations * sizeof(uint32_t), slots);
    for (uint32_t i = 0; i < kNumIterations; ++i) {
        EXPECT_EQ(slots[i], kValueA + i) << "iteration " << i;
    }
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

TEST_F(CommandListMultiDeviceTest, UpdatesParametersAcrossHeterogeneousDeviceRanges) {
    // One parameter spans the programs of both rows; another targets only the top row.
    constexpr uint32_t kTopRowValue = 0xABCD0001;
    const MeshCoordinateRange all_devices(mesh_device_->shape());
    const MeshCoordinateRange top_row({0, 0}, {0, 3});
    const MeshCoordinateRange bottom_row({1, 0}, {1, 3});

    auto split_rows = make_l1_write_workload(
        *mesh_device_,
        kAddressA,
        std::vector<DeviceRangeValue>{{top_row, kValueA}, {bottom_row, kValueB}},
        "patch_split_rows");
    const auto& top_program = split_rows.get_programs().at(top_row);
    const auto& bottom_program = split_rows.get_programs().at(bottom_row);

    const CmdListRuntimeArgName address_param{"destination"};
    const CmdListCommonRuntimeArgName top_value_param{"top_payload"};
    auto& cq = mesh_device_->mesh_command_queue(0);
    CommandListBuilder builder(*mesh_device_);
    builder.add(
        split_rows,
        CmdListParameters{
            .runtime_parameters =
                {{address_param,
                  std::vector<CmdListRuntimeArgInfo>{
                      l1_write_address_arg(top_program), l1_write_address_arg(bottom_program)}}},
            .common_runtime_parameters =
                {{top_value_param, std::vector<CmdListCommonRuntimeArgInfo>{l1_write_value_arg(top_program)}}},
        });
    auto command_list = builder.build(cq);

    command_list.update_args(
        CmdListArgPatch{
            .runtime_args = {{address_param, kAddressB}},
            .common_runtime_args = {{top_value_param, kTopRowValue}},
        },
        /*blocking=*/true);

    for (const auto& coord : all_devices) {
        write_l1(mesh_device_, coord, kAddressA, 0);
        write_l1(mesh_device_, coord, kAddressB, 0);
    }
    EnqueueCommandList(cq, command_list, /*blocking=*/true);
    for (const auto& coord : all_devices) {
        EXPECT_EQ(read_l1(mesh_device_, coord, kAddressA), 0u) << "Device coordinate: " << coord;
        EXPECT_EQ(read_l1(mesh_device_, coord, kAddressB), top_row.contains(coord) ? kTopRowValue : kValueB)
            << "Device coordinate: " << coord;
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
    EXPECT_THAT(
        [&] { command_list.update_args({}); },  // NOLINT(bugprone-use-after-move)
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

TEST_F(CommandListTest, RejectsUnresolvableParametersWithoutChangingTheRecording) {
    // Parameters are resolved inside add(), so each mistake below is caught there, before the workload is staged.
    auto workload = make_l1_write_workload(*mesh_device_, kAddressA, kValueA, "bad_parameters");
    auto not_added = make_l1_write_workload(*mesh_device_, kAddressB, kValueB, "not_added");
    const auto& program = workload.get_programs().begin()->second;
    const CmdListRuntimeArgName address_param{"destination"};
    const CmdListCommonRuntimeArgName value_param{"payload"};
    auto& cq = mesh_device_->mesh_command_queue(0);
    CommandListBuilder builder(*mesh_device_);
    auto expect_add_throws = [&](const CmdListParameters& parameters, const std::string& message) {
        EXPECT_THAT([&] { builder.add(workload, parameters); }, ThrowsMessage<std::runtime_error>(HasSubstr(message)));
    };

    // The parameter's program must belong to the workload being added.
    expect_add_throws(
        {.runtime_parameters =
             {{address_param,
               std::vector<CmdListRuntimeArgInfo>{l1_write_address_arg(not_added.get_programs().begin()->second)}}}},
        "not in the recorded MeshWorkload");

    // The argument must be declared by the kernel's runtime argument schema.
    auto undeclared_rta = l1_write_address_arg(program);
    undeclared_rta.arg_name = "missing";
    expect_add_throws(
        {.runtime_parameters = {{address_param, std::vector<CmdListRuntimeArgInfo>{undeclared_rta}}}},
        "Runtime argument 'missing' is not declared");
    auto undeclared_crta = l1_write_value_arg(program);
    undeclared_crta.arg_name = "missing";
    expect_add_throws(
        {.common_runtime_parameters = {{value_param, std::vector<CmdListCommonRuntimeArgInfo>{undeclared_crta}}}},
        "Common runtime argument 'missing' is not declared");

    // A runtime argument has one copy per node, so the parameter must say which nodes to patch.
    auto no_nodes = l1_write_address_arg(program);
    no_nodes.nodes.clear();
    expect_add_throws(
        {.runtime_parameters = {{address_param, std::vector<CmdListRuntimeArgInfo>{no_nodes}}}},
        "must name at least one node");

    // A tensor parameter must name a TensorParameter the program declares.
    expect_add_throws(
        {.tensor_parameters =
             {{CmdListTensorArgName{"tensor"},
               std::vector<CmdListTensorArgInfo>{
                   {.program = std::cref(program), .param_name = m2::TensorParamName{"missing"}}}}}},
        "TensorParameter 'missing' is not declared");

    // None of the failed calls staged the workload.
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
