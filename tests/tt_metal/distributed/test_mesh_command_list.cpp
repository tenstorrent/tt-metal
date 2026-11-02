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
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/byte_based_tensor_transfers.hpp>
#include <tt-metalium/experimental/allocation_context.hpp>
#include <tt-metalium/experimental/allocator.hpp>
#include <tt-metalium/experimental/command_list.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <internal/mesh_device.hpp>

#include "tests/tt_metal/tt_metal/api/metal2_host_api/test_helpers.hpp"
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "tt_metal/distributed/fd_mesh_command_queue.hpp"
#include "tt_metal/impl/allocator/allocator.hpp"

namespace tt::tt_metal::experimental::test {
namespace {

namespace m2 = tt::tt_metal::experimental;
using namespace tt::tt_metal::distributed;

using m2::test_helpers::BindTensorParameterToKernel;
using m2::test_helpers::MakeMinimalDFB;
using m2::test_helpers::MakeMinimalGen1DMKernel;
using m2::test_helpers::MakeMinimalWorkUnit;
using ::testing::Contains;
using ::testing::HasSubstr;
using ::testing::Not;
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
        const auto arch = mesh_device_->arch();
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
        const auto arch = mesh_device_->arch();
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
        const auto arch = mesh_device_->arch();
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

auto* device(const std::shared_ptr<MeshDevice>& mesh_device) { return mesh_device->impl().get_devices().at(0); }

IDevice* device(const std::shared_ptr<MeshDevice>& mesh_device, const MeshCoordinate& device_coord) {
    return internal::get_device(*mesh_device, device_coord);
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

constexpr uint32_t kPageSize = 1024;

// One kPageSize row-major page per row.
TensorSpec dram_tensor_spec(uint32_t num_pages) {
    return TensorSpec(
        Shape{num_pages, kPageSize / 2},
        TensorLayout(
            DataType::BFLOAT16,
            PageConfig(Layout::ROW_MAJOR),
            MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM}));
}

// Copies TensorParameter "input_tensor" to "output_tensor" one page at a time through an L1 DFB.
m2::ProgramSpec make_loopback_spec(const std::string& name, const TensorSpec& tensor_spec) {
    auto producer = MakeMinimalGen1DMKernel("producer", DataMovementProcessor::RISCV_0);
    producer.source = "tests/tt_metal/tt_metal/test_kernels/dataflow/tensor_accessor_loopback_producer.cpp";
    producer.advanced_options.num_runtime_varargs = 1;
    auto consumer = MakeMinimalGen1DMKernel("consumer", DataMovementProcessor::RISCV_1);
    consumer.source = "tests/tt_metal/tt_metal/test_kernels/dataflow/tensor_accessor_loopback_consumer.cpp";
    consumer.advanced_options.num_runtime_varargs = 1;

    auto dfb = MakeMinimalDFB("input_dfb", kPageSize, /*num_entries=*/2);
    dfb.data_format_metadata = DataFormat::Float16_b;
    producer.dfb_bindings.push_back(m2::ProducerOf(m2::DFBSpecName{"input_dfb"}, "input_dfb"));
    consumer.dfb_bindings.push_back(m2::ConsumerOf(m2::DFBSpecName{"input_dfb"}, "input_dfb"));
    BindTensorParameterToKernel(producer, "input_tensor", "input_tensor");
    BindTensorParameterToKernel(consumer, "output_tensor", "output_tensor");

    return m2::ProgramSpec{
        .name = name,
        .kernels = {producer, consumer},
        .dataflow_buffers = {dfb},
        .tensor_parameters =
            {
                m2::TensorParameter{.unique_id = m2::TensorParamName{"input_tensor"}, .spec = tensor_spec},
                m2::TensorParameter{.unique_id = m2::TensorParamName{"output_tensor"}, .spec = tensor_spec},
            },
        .work_units = {MakeMinimalWorkUnit("main", kNode, {"producer", "consumer"})},
    };
}

m2::ProgramRunArgs loopback_run_args(const MeshTensor& input, const MeshTensor& output) {
    const uint32_t num_pages = input.logical_shape()[0];
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
        {m2::TensorParamName{"input_tensor"}, m2::ProgramRunArgs::TensorArgument{input}},
        {m2::TensorParamName{"output_tensor"}, m2::ProgramRunArgs::TensorArgument{output}},
    };
    return args;
}

MeshWorkload make_loopback_workload(
    MeshDevice& mesh_device, const MeshTensor& input, const MeshTensor& output, const std::string& name) {
    auto workload = m2::MakeMeshWorkloadFromSpec(mesh_device, make_loopback_spec(name, input.tensor_spec()));
    m2::SetProgramRunArgs(workload.get_programs().begin()->second, loopback_run_args(input, output));
    return workload;
}

CmdListParameters patchable_input(const Program& program, const CmdListTensorArgName& name) {
    return {
        .tensor_parameters = {
            {name,
             std::vector<CmdListTensorArgInfo>{
                 {.program = std::cref(program), .param_name = m2::TensorParamName{"input_tensor"}}}}}};
}

void write_tensor(const MeshTensor& tensor, const std::vector<uint32_t>& data) {
    ::tt::tt_metal::detail::WriteToBuffer(*tensor.mesh_buffer().get_reference_buffer(), data);
}

std::vector<uint32_t> read_tensor(const MeshTensor& tensor) {
    std::vector<uint32_t> result;
    ::tt::tt_metal::detail::ReadFromBuffer(*tensor.mesh_buffer().get_reference_buffer(), result);
    return result;
}

// Interleaved DRAM buffer of pages_per_bank kPageSize pages in every bank.
std::shared_ptr<MeshBuffer> allocate_dram(MeshDevice& mesh_device, uint32_t pages_per_bank) {
    const uint32_t num_banks = mesh_device.allocator()->get_num_banks(BufferType::DRAM);
    return MeshBuffer::create(
        ReplicatedBufferConfig{.size = pages_per_bank * num_banks * kPageSize},
        DeviceLocalBufferConfig{.page_size = kPageSize, .buffer_type = BufferType::DRAM},
        &mesh_device);
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
    const auto tensor_spec = dram_tensor_spec(num_pages);
    MeshTensor input_a = MeshTensor::allocate_on_device(*mesh_device_, tensor_spec);
    MeshTensor input_b = MeshTensor::allocate_on_device(*mesh_device_, tensor_spec);
    MeshTensor output = MeshTensor::allocate_on_device(*mesh_device_, tensor_spec);

    // Bind input A when recording and expose only that input as patchable.
    auto workload = make_loopback_workload(*mesh_device_, input_a, output, "tensor_patch");
    const CmdListTensorArgName input_param{"input"};
    CommandListBuilder builder(*mesh_device_);
    builder.add(workload, patchable_input(workload.get_programs().begin()->second, input_param));
    auto& cq = mesh_device_->mesh_command_queue(0);
    auto command_list = builder.build(cq);

    std::vector<uint32_t> data_a(num_pages * kPageSize / sizeof(uint32_t));
    std::vector<uint32_t> data_b(data_a.size());
    for (size_t i = 0; i < data_a.size(); ++i) {
        data_a[i] = static_cast<uint32_t>(i);
        data_b[i] = 0x80000000u + static_cast<uint32_t>(i);
    }
    const std::vector<uint32_t> zeros(data_a.size(), 0);
    write_tensor(input_a, data_a);
    write_tensor(input_b, data_b);
    write_tensor(output, zeros);

    // The initial replay copies input A to the output.
    EnqueueCommandList(cq, command_list, /*blocking=*/true);
    EXPECT_EQ(read_tensor(output), data_a);

    // A tensor that does not match the declared TensorSpec is rejected.
    MeshTensor mismatched =
        MeshTensor::allocate_on_device(*mesh_device_, TensorSpec(Shape{num_pages, 256}, tensor_spec.tensor_layout()));
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
    write_tensor(output, zeros);
    EnqueueCommandList(cq, command_list, /*blocking=*/true);
    EXPECT_EQ(read_tensor(output), data_b);
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
    // The raw base address is what this test checks: the patched CRTA word.
    destination[0] = tensor_accessor::detail::TransferAccess::bank_base(accessor);
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

TEST_F(CommandListMultiDeviceTest, SynchronizedAllocatorMergesOverlappingDramAllocations) {
    // After unit-mesh aggregation, the parent mesh's allocator takes the merged state of its submeshes' allocators,
    // and checked replay on the parent queries that merged view. Each submesh allocates independently from the same
    // base address, so their buffers can partially overlap. A replay on the parent writes a range on every device, so
    // overlapping buffers must merge into one entry that covers them all.
    auto sub0 = mesh_device_->create_submesh(MeshShape(1, 1), MeshCoordinate(0, 0));
    auto sub1 = mesh_device_->create_submesh(MeshShape(1, 1), MeshCoordinate(0, 1));

    // Submesh 0: X at [0-4). Submesh 1: a pad at [0-2), then Y at [2-6). Offsets are in pages per bank from the shared
    // base address, which is where each fresh submesh allocator places its first buffer.
    const auto x = allocate_dram(*sub0, 4);
    const auto pad = allocate_dram(*sub1, 2);
    const auto y = allocate_dram(*sub1, 4);
    const DeviceAddr base = x->address();
    ASSERT_EQ(pad->address(), base);
    ASSERT_EQ(y->address(), base + (2 * kPageSize));

    ::tt::tt_metal::experimental::synchronize_allocator_state(
        mesh_device_->allocator().get(), {sub0->allocator().get(), sub1->allocator().get()});

    // X, the pad, and Y merge into the single allocation [0-6). Allocations are disjoint, so no other one overlaps it.
    EXPECT_THAT(
        mesh_device_->allocator_impl()->get_dram_allocations(),
        Contains(std::pair<DeviceAddr, DeviceAddr>{base, base + (6 * kPageSize)}));
}

TEST_F(CommandListTest, AllowsTemporaryTensorLifetimeDuringBuild) {
    constexpr uint32_t num_pages = 2;
    constexpr uint32_t total_bytes = num_pages * kPageSize;
    const auto tensor_spec = dram_tensor_spec(num_pages);

    // Allocate the persistent output before command-list recording starts.
    MeshTensor output = MeshTensor::allocate_on_device(*mesh_device_, tensor_spec);
    const std::vector<uint32_t> zeros(total_bytes / sizeof(uint32_t), 0);
    auto& write_cq = mesh_device_->mesh_command_queue(0);
    ::tt::tt_metal::enqueue_write_tensor(write_cq, reinterpret_cast<const std::byte*>(zeros.data()), output);
    Finish(write_cq);

    std::vector<uint32_t> expected(total_bytes / sizeof(uint32_t));
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = 0xA0000000u + static_cast<uint32_t>(i);
    }

    // Create a workload that copies an input tensor to the persistent output.
    auto workload =
        m2::MakeMeshWorkloadFromSpec(*mesh_device_, make_loopback_spec("temporary_tensor_lifetime", tensor_spec));
    auto& program = workload.get_programs().begin()->second;

    // Allocate the input during recording and bind it to the workload added to the command list.
    CommandListBuilder builder(*mesh_device_);
    {
        MeshTensor temporary = MeshTensor::allocate_on_device(*mesh_device_, tensor_spec);
        ::tt::tt_metal::enqueue_write_tensor(
            mesh_device_->mesh_command_queue(0), reinterpret_cast<const std::byte*>(expected.data()), temporary);
        Finish(mesh_device_->mesh_command_queue(0));

        m2::SetProgramRunArgs(program, loopback_run_args(temporary, output));
        builder.add(workload);
    }

    // Build and replay after the recorded input tensor has been deallocated.
    // Replay intentionally uses the captured raw address after the allocation is released. The stale DRAM contents
    // remain usable until another allocation reuses or overwrites that storage; no live MeshTensor handle is required.
    // Checked replay accepts this: nothing occupies the input's freed range, and the live output exactly fills its own.
    auto& cq = mesh_device_->mesh_command_queue(0);
    auto command_list = builder.build(cq);
    EnqueueCommandList(cq, command_list, /*blocking=*/true, /*check_dram_overlap=*/true);

    std::vector<uint32_t> result(total_bytes / sizeof(uint32_t));
    ::tt::tt_metal::enqueue_read_tensor(cq, output, reinterpret_cast<std::byte*>(result.data()));
    EXPECT_EQ(result, expected);
}

TEST_F(CommandListTest, CheckedReplayAcceptsOnlyFreeOrExactRanges) {
    // Checked replay passes a recorded DRAM range that is free or exactly filled by one live allocation, warning when
    // that allocation is not the tensor bound to the range, and fails on any other overlap. One command list is built
    // over a fixed layout of DRAM slots; each overlap shape is then made by freeing slots and allocating into the hole.
    //
    // Placement is predictable because DRAM allocates bottom-up with first fit: an allocation lands at the start of the
    // lowest hole that fits it. Every address the test relies on is asserted, so a change in placement fails loudly
    // instead of quietly testing a different shape.
    const auto input_spec = dram_tensor_spec(4 * mesh_device_->allocator()->get_num_banks(BufferType::DRAM));
    auto& cq = mesh_device_->mesh_command_queue(0);

    // Layout, in pages per bank from P0:  [P0 0-2][T 2-6][P1 6-8][fence 8-10]
    // T is the recorded input, and P0 and P1 touch it on both sides. The fence is never freed, so a hole that includes
    // P1 stays bounded instead of merging with the free space above it, which keeps every hole's size exact.
    MeshTensor output = MeshTensor::allocate_on_device(*mesh_device_, input_spec);
    auto p0 = allocate_dram(*mesh_device_, 2);
    std::optional<MeshTensor> t = MeshTensor::allocate_on_device(*mesh_device_, input_spec);
    auto p1 = allocate_dram(*mesh_device_, 2);
    const auto fence = allocate_dram(*mesh_device_, 2);
    // Every slot and intruder is interleaved over all banks with the same address in each, so one per-bank offset,
    // counted in pages from P0, places it.
    const DeviceAddr base = p0->address();
    auto offset_of = [&](DeviceAddr address) { return (address - base) / kPageSize; };
    ASSERT_EQ(offset_of(t->mesh_buffer().address()), 2u);
    ASSERT_EQ(offset_of(p1->address()), 6u);
    ASSERT_EQ(offset_of(fence->address()), 8u);

    // Record and build before making any holes. Kernel binaries are allocated in DRAM while the workload is recorded,
    // and one landing in a hole would shift every placement below.
    auto workload = make_loopback_workload(*mesh_device_, *t, output, "checked_replay");
    const CmdListTensorArgName input_param{"input"};
    CommandListBuilder builder(*mesh_device_);
    builder.add(workload, patchable_input(workload.get_programs().begin()->second, input_param));
    auto command_list = builder.build(cq);
    auto replay = [&](bool check_dram_overlap) {
        EnqueueCommandList(cq, command_list, /*blocking=*/true, check_dram_overlap);
    };
    auto make_data = [&](uint32_t first) {
        std::vector<uint32_t> data(input_spec.logical_shape()[0] * kPageSize / sizeof(uint32_t));
        std::iota(data.begin(), data.end(), first);
        return data;
    };
    // The check logs to stdout. A passing checked replay warns when a range is exactly filled by an allocation other
    // than the tensor bound to it.
    constexpr const char* kNotBoundWarning = "other than the tensor bound to it";
    auto checked_replay_log = [&] {
        testing::internal::CaptureStdout();
        replay(/*check_dram_overlap=*/true);
        return testing::internal::GetCapturedStdout();
    };

    // The recorded input and output are live, so each recorded range is exactly filled by the tensor bound to it, and
    // replay passes without a warning. P0 and P1 touch the input's range on both sides; touching is not overlapping, so
    // they must not count.
    const auto data_a = make_data(0xA0000000u);
    write_tensor(*t, data_a);
    EXPECT_THAT(checked_replay_log(), Not(HasSubstr(kNotBoundWarning)));
    EXPECT_EQ(read_tensor(output), data_a);

    // An allocation outside every recorded range never matters.
    {
        const auto unrelated = allocate_dram(*mesh_device_, 2);
        replay(/*check_dram_overlap=*/true);
    }

    // Free the input and fill its slot exactly with other allocations. The check identifies a slot by its address and
    // size, so replay passes, but it warns because the allocation is not the bound tensor. A same-size replacement
    // tensor is the intended case; a plain buffer stands in for one that must not be overwritten, such as a
    // command-list command buffer. The check cannot tell the two apart, so both warn.
    t.reset();
    {
        const auto same_size_buffer = allocate_dram(*mesh_device_, 4);
        ASSERT_EQ(offset_of(same_size_buffer->address()), 2u);
        EXPECT_THAT(checked_replay_log(), HasSubstr(kNotBoundWarning));
    }
    t = MeshTensor::allocate_on_device(*mesh_device_, input_spec);
    ASSERT_EQ(offset_of(t->mesh_buffer().address()), 2u);
    const auto data_b = make_data(0xB0000000u);
    write_tensor(*t, data_b);
    EXPECT_THAT(checked_replay_log(), HasSubstr(kNotBoundWarning));
    EXPECT_EQ(read_tensor(output), data_b);

    // Every other overlap fails. Each case frees some slots, allocates intruders into the hole, and expects replay to
    // throw. It then frees the intruders and reallocates the freed slots in address order, so each lands at the start
    // of what remains of the hole and the next case starts from the original layout.
    enum class Slot { P0, T, P1 };
    struct Intruder {
        uint32_t pages_per_bank;
        DeviceAddr expected_offset;
    };
    struct OverlapCase {
        const char* name;
        std::vector<Slot> freed;
        std::vector<Intruder> intruders;
        // Indices of intruders freed again before replay.
        std::vector<size_t> released;
    };
    // Each comment gives where the intruders sit, against T at [2-6).
    const std::vector<OverlapCase> cases = {
        // [2-4): inside the range, starting with it.
        {"smaller at the same address", {Slot::T}, {{2, 2}}, {}},
        // [2-8): starts with the range and runs past its end.
        {"larger at the same address", {Slot::T, Slot::P1}, {{6, 2}}, {}},
        // [0-4): starts below the range and ends inside it.
        {"straddles the start", {Slot::P0, Slot::T}, {{4, 0}}, {}},
        // [4-8): starts inside the range and runs past its end. The filler at [2-4) only pushes it there.
        {"straddles the end", {Slot::T, Slot::P1}, {{2, 2}, {4, 4}}, {0}},
        // [0-8): covers the range and both neighbors.
        {"encloses the range", {Slot::P0, Slot::T, Slot::P1}, {{8, 0}}, {}},
        // [2-4) and [4-6): two allocations that together fill the range exactly.
        {"two allocations tile the range", {Slot::T}, {{2, 2}, {2, 4}}, {}},
    };
    auto free_slot = [&](Slot slot) {
        switch (slot) {
            case Slot::P0: p0.reset(); break;
            case Slot::T: t.reset(); break;
            case Slot::P1: p1.reset(); break;
        }
    };
    auto restore_slot = [&](Slot slot) {
        switch (slot) {
            case Slot::P0:
                p0 = allocate_dram(*mesh_device_, 2);
                EXPECT_EQ(offset_of(p0->address()), 0u);
                break;
            case Slot::T:
                t = MeshTensor::allocate_on_device(*mesh_device_, input_spec);
                EXPECT_EQ(offset_of(t->mesh_buffer().address()), 2u);
                break;
            case Slot::P1:
                p1 = allocate_dram(*mesh_device_, 2);
                EXPECT_EQ(offset_of(p1->address()), 6u);
                break;
        }
    };

    // The check runs before anything is enqueued, so every failed replay must leave these zeros in the output.
    const std::vector<uint32_t> zeros(data_a.size(), 0);
    write_tensor(output, zeros);
    for (const auto& overlap : cases) {
        SCOPED_TRACE(overlap.name);
        for (const auto slot : overlap.freed) {
            free_slot(slot);
        }
        std::vector<std::shared_ptr<MeshBuffer>> intruders;
        for (const auto& [pages_per_bank, expected_offset] : overlap.intruders) {
            intruders.push_back(allocate_dram(*mesh_device_, pages_per_bank));
            ASSERT_EQ(offset_of(intruders.back()->address()), expected_offset);
        }
        for (const size_t i : overlap.released) {
            intruders[i].reset();
        }
        EXPECT_THAT(
            [&] { replay(/*check_dram_overlap=*/true); },
            ThrowsMessage<std::runtime_error>(HasSubstr("does not exactly match")));
        intruders.clear();
        for (const auto slot : overlap.freed) {
            restore_slot(slot);
        }
    }
    EXPECT_EQ(read_tensor(output), zeros);

    // With the check disabled, replay never queries the allocator, so the same smaller allocation at the input's
    // address does not stop it.
    t.reset();
    {
        const auto intruder = allocate_dram(*mesh_device_, 2);
        ASSERT_EQ(offset_of(intruder->address()), 2u);
        EXPECT_NO_THROW(replay(/*check_dram_overlap=*/false));
    }
    restore_slot(Slot::T);

    // update_args moves the recorded input range, and the bound tensor, to the new tensor. The input's old range is no
    // longer recorded, so an overlap there passes; the new tensor fills its own range, so there is no warning; and
    // replay copies the new tensor.
    MeshTensor new_input = MeshTensor::allocate_on_device(*mesh_device_, input_spec);
    const auto data_c = make_data(0xC0000000u);
    write_tensor(new_input, data_c);
    command_list.update_args(
        CmdListArgPatch{.tensor_args = {{input_param, m2::ProgramRunArgs::TensorArgument{new_input}}}});
    t.reset();
    const auto intruder = allocate_dram(*mesh_device_, 2);
    ASSERT_EQ(offset_of(intruder->address()), 2u);
    EXPECT_THAT(checked_replay_log(), Not(HasSubstr(kNotBoundWarning)));
    EXPECT_EQ(read_tensor(output), data_c);
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
    legacy_workload.add_program(MeshCoordinateRange(mesh_device_->shape()), CreateProgram(*this->mesh_device_));
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

    // A TensorParameter that backs a borrowed-memory DFB cannot be patched.
    const auto dram_spec = dram_tensor_spec(2);
    const uint32_t num_l1_banks = mesh_device_->allocator()->get_num_banks(BufferType::L1);
    const auto scratch_spec = TensorSpec(
        Shape{2 * num_l1_banks, 512},
        TensorLayout(
            DataType::BFLOAT16,
            PageConfig(Layout::ROW_MAJOR),
            MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::L1}));
    MeshTensor input = MeshTensor::allocate_on_device(*mesh_device_, dram_spec);
    MeshTensor output = MeshTensor::allocate_on_device(*mesh_device_, dram_spec);
    MeshTensor scratch = MeshTensor::allocate_on_device(*mesh_device_, scratch_spec);

    // The loopback, with its DFB borrowing memory from an extra L1 scratch tensor.
    auto borrowed_spec = make_loopback_spec("borrowed_dfb", dram_spec);
    borrowed_spec.dataflow_buffers.at(0).borrowed_from = m2::TensorParamName{"scratch_tensor"};
    borrowed_spec.tensor_parameters.push_back(
        m2::TensorParameter{.unique_id = m2::TensorParamName{"scratch_tensor"}, .spec = scratch_spec});
    auto borrowed_workload = m2::MakeMeshWorkloadFromSpec(*mesh_device_, borrowed_spec);
    auto& borrowed_program = borrowed_workload.get_programs().begin()->second;
    auto borrowed_args = loopback_run_args(input, output);
    borrowed_args.tensor_args.emplace(
        m2::TensorParamName{"scratch_tensor"}, m2::ProgramRunArgs::TensorArgument{scratch});
    m2::SetProgramRunArgs(borrowed_program, borrowed_args);
    EXPECT_THAT(
        [&] {
            builder.add(
                borrowed_workload,
                {.tensor_parameters = {
                     {CmdListTensorArgName{"scratch"},
                      std::vector<CmdListTensorArgInfo>{
                          {.program = std::cref(borrowed_program),
                           .param_name = m2::TensorParamName{"scratch_tensor"}}}}}});
        },
        ThrowsMessage<std::runtime_error>(HasSubstr("backs borrowed-memory DFB")));

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

TEST_F(CommandListTest, TracesAndCommandListsCanRunInSequenceButNotCoexist) {
    // Traces and command lists may share a device, but never while the other is live.
    auto list_workload = make_l1_write_workload(*mesh_device_, kAddressA, kValueA, "exclusion_list");
    auto trace_workload = make_l1_write_workload(*mesh_device_, kAddressB, kValueB, "exclusion_trace");
    auto& cq = mesh_device_->mesh_command_queue(0);

    // A live builder, and then live lists on their own, block trace capture until the last list is
    // released. The remaining list is released by its destructor rather than by deallocate().
    {
        CommandListBuilder builder(*mesh_device_);
        EXPECT_THAT(
            [&] { (void)mesh_device_->begin_mesh_trace(cq); },
            ThrowsMessage<std::runtime_error>(HasSubstr("while a CommandListBuilder or CommandList exists")));
        builder.add(list_workload);
        auto first_list = builder.build(cq);
        auto second_list = builder.build(cq);
        builder.deallocate();
        first_list.deallocate();
        EXPECT_THAT(
            [&] { (void)mesh_device_->begin_mesh_trace(cq); },
            ThrowsMessage<std::runtime_error>(HasSubstr("while a CommandListBuilder or CommandList exists")));

        write_l1(mesh_device_, kAddressA, 0);
        EnqueueCommandList(cq, second_list, /*blocking=*/true);
        EXPECT_EQ(read_l1(mesh_device_, kAddressA), kValueA);
    }

    // With every command list released, a trace can be captured and replayed. The trace blocks
    // builders both during capture and after capture ends, until it is released.
    EnqueueMeshWorkload(cq, trace_workload, /*blocking=*/true);
    const auto trace_id = mesh_device_->begin_mesh_trace(cq);
    EXPECT_THAT(
        [&] { CommandListBuilder builder(*mesh_device_); },
        ThrowsMessage<std::runtime_error>(HasSubstr("while a trace exists")));
    EnqueueMeshWorkload(cq, trace_workload, /*blocking=*/false);
    mesh_device_->end_mesh_trace(cq, trace_id);
    EXPECT_THAT(
        [&] { CommandListBuilder builder(*mesh_device_); },
        ThrowsMessage<std::runtime_error>(HasSubstr("while a trace exists")));

    write_l1(mesh_device_, kAddressB, 0);
    mesh_device_->replay_mesh_trace(cq, trace_id, /*blocking=*/true);
    EXPECT_EQ(read_l1(mesh_device_, kAddressB), kValueB);
    mesh_device_->release_mesh_trace(trace_id);

    // Releasing the trace makes command lists available again.
    CommandListBuilder builder(*mesh_device_);
    builder.add(list_workload);
    auto command_list = builder.build(cq);
    write_l1(mesh_device_, kAddressA, 0);
    EnqueueCommandList(cq, command_list, /*blocking=*/true);
    EXPECT_EQ(read_l1(mesh_device_, kAddressA), kValueA);
}

TEST_F(CommandListTest, TracesOnInactiveSubDeviceManagersBlockBuilders) {
    // A trace outlives the loading of a different sub-device manager, so it must keep blocking builders.
    const auto worker_grid = mesh_device_->compute_with_storage_grid_size();
    const SubDevice full_grid_sub_device(
        std::array{CoreRangeSet(CoreRange({0, 0}, {worker_grid.x - 1, worker_grid.y - 1}))});
    const auto trace_manager = mesh_device_->create_sub_device_manager({full_grid_sub_device}, 3200);
    auto workload = make_l1_write_workload(*mesh_device_, kAddressA, kValueA, "inactive_manager_trace");
    auto& cq = mesh_device_->mesh_command_queue(0);

    mesh_device_->load_sub_device_manager(trace_manager);
    EnqueueMeshWorkload(cq, workload, /*blocking=*/true);
    const auto trace_id = mesh_device_->begin_mesh_trace(cq);
    EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    mesh_device_->end_mesh_trace(cq, trace_id);

    mesh_device_->clear_loaded_sub_device_manager();
    EXPECT_THAT(
        [&] { CommandListBuilder builder(*mesh_device_); },
        ThrowsMessage<std::runtime_error>(HasSubstr("while a trace exists")));

    // Traces are released from the active manager, so the trace's manager must be reloaded first.
    mesh_device_->load_sub_device_manager(trace_manager);
    mesh_device_->release_mesh_trace(trace_id);
    mesh_device_->clear_loaded_sub_device_manager();
    mesh_device_->remove_sub_device_manager(trace_manager);
    EXPECT_NO_THROW({ CommandListBuilder builder(*mesh_device_); });
}

}  // namespace
}  // namespace tt::tt_metal::experimental::test
