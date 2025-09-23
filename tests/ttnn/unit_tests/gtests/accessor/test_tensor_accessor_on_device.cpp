// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include <fmt/format.h>
#include <cstdint>
#include <vector>

#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tests/ttnn/unit_tests/gtests/accessor/common.hpp"

#include <tt-metalium/shape.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/api/ttnn/distributed/api.hpp"

namespace tensor_accessor_device_tests {

using namespace ttnn;
using namespace tt;
using namespace tt::tt_metal;

struct InputOutputBufferParams {
    Shape tensor_shape;
    Layout layout;
    DataType dtype;
    BufferType input_buffer_type;
    BufferType output_buffer_type;

    std::optional<NdShardSpec> input_shard_spec;
    std::optional<NdShardSpec> output_shard_spec;
    tensor_accessor::ArgsConfig crta_config;
};

template <typename T>
static void test_single_core_reshard(
    const InputOutputBufferParams& params, tt::tt_metal::distributed::MeshDevice* mesh_device) {
    MemoryConfig input_mem_config = params.input_shard_spec
                                        ? MemoryConfig(params.input_buffer_type, *params.input_shard_spec)
                                        : MemoryConfig(TensorMemoryLayout::INTERLEAVED, params.input_buffer_type);
    MemoryConfig output_mem_config = params.output_shard_spec
                                         ? MemoryConfig(params.output_buffer_type, *params.output_shard_spec)
                                         : MemoryConfig(TensorMemoryLayout::INTERLEAVED, params.output_buffer_type);

    TensorSpec input_spec(params.tensor_shape, TensorLayout(params.dtype, PageConfig(params.layout), input_mem_config));
    TensorSpec output_spec(
        params.tensor_shape, TensorLayout(params.dtype, PageConfig(params.layout), output_mem_config));

    const auto src = tt::test_utils::generate_uniform_random_vector<T>(0, UINT8_MAX, params.tensor_shape.volume());

    auto input_tensor = Tensor::from_vector(src, input_spec, mesh_device);
    auto output_tensor = Tensor::from_vector(std::vector<T>(params.tensor_shape.volume()), output_spec, mesh_device);

    auto input_buffer = input_tensor.buffer();
    auto output_buffer = output_tensor.buffer();
    auto aligned_page_size = input_buffer->aligned_page_size();
    if (output_buffer->aligned_page_size() != aligned_page_size) {
        GTEST_SKIP() << "Input and output buffers must have the same aligned page size!";
    }

    auto program = CreateProgram();

    constexpr CoreCoord grid = {0, 0};
    const auto data_format = datatype_to_dataformat_converter(params.dtype);

    constexpr auto num_tiles = 2;  // Double buffered for perf, but it doesn't really matter for this test
    CBHandle cb_in0_idx = tt::CBIndex::c_0;
    auto c_in0_config = CircularBufferConfig(aligned_page_size * num_tiles, {{cb_in0_idx, data_format}})
                            .set_page_size(cb_in0_idx, aligned_page_size);
    CreateCircularBuffer(program, grid, c_in0_config);

    const auto input_accessor_args = TensorAccessorArgs(*input_buffer, params.crta_config);
    const auto output_accessor_args = TensorAccessorArgs(*output_buffer, params.crta_config);

    std::vector<uint32_t> input_compile_time_args = input_accessor_args.get_compile_time_args();
    input_compile_time_args.push_back(cb_in0_idx);
    input_compile_time_args.push_back(aligned_page_size);

    std::vector<uint32_t> output_compile_time_args = output_accessor_args.get_compile_time_args();
    output_compile_time_args.push_back(cb_in0_idx);
    output_compile_time_args.push_back(aligned_page_size);

    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "tests/ttnn/unit_tests/gtests/accessor/kernels/reader_reshard.cpp",
        grid,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = input_compile_time_args,
        });

    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "tests/ttnn/unit_tests/gtests/accessor/kernels/writer_reshard.cpp",
        grid,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = output_compile_time_args,
        });

    std::vector<uint32_t> input_runtime_args = input_accessor_args.get_common_runtime_args();
    input_runtime_args.push_back(input_buffer->address());
    input_runtime_args.push_back(input_buffer->num_pages());
    SetCommonRuntimeArgs(program, reader_kernel_id, input_runtime_args);

    std::vector<uint32_t> output_runtime_args = output_accessor_args.get_common_runtime_args();
    output_runtime_args.push_back(output_buffer->address());
    output_runtime_args.push_back(output_buffer->num_pages());
    SetCommonRuntimeArgs(program, writer_kernel_id, output_runtime_args);

    auto mesh_workload = tt::tt_metal::distributed::CreateMeshWorkload();
    mesh_workload.add_program(tt::tt_metal::distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
    EnqueueMeshWorkload(mesh_device->mesh_command_queue(), mesh_workload, true);

    auto output_tensor_cpu = output_tensor.cpu(true);

    // Data should be only in the first shard
    Tensor output_tensor_shard0 = ttnn::distributed::get_device_tensors(output_tensor_cpu).front();
    auto output_vec = output_tensor_shard0.to_vector<T>();

    EXPECT_EQ(output_vec, src);
}

struct CopyParams {
    Shape tensor_shape;
    Layout layout;
    DataType dtype;
    BufferType buffer_type;

    std::optional<NdShardSpec> input_shard_spec;
};

template <typename T>
static void test_multi_core_copy(
    const CopyParams& params,
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const std::string& kernel_path,
    const std::map<std::string, std::string>& defines = {}) {
    MemoryConfig input_mem_config = MemoryConfig(params.buffer_type, params.input_shard_spec);
    TensorSpec input_spec(params.tensor_shape, TensorLayout(params.dtype, PageConfig(params.layout), input_mem_config));

    const auto src = tt::test_utils::generate_uniform_random_vector<T>(0, UINT8_MAX, params.tensor_shape.volume());

    auto input_tensor = Tensor::from_vector(src, input_spec, mesh_device);
    auto output_tensor = Tensor::from_vector(std::vector<T>(params.tensor_shape.volume()), input_spec, mesh_device);

    auto input_buffer = input_tensor.buffer();
    auto output_buffer = output_tensor.buffer();
    auto aligned_page_size = input_buffer->aligned_page_size();

    auto program = CreateProgram();
    auto core_groups = input_buffer->buffer_distribution_spec().value().core_groups();
    auto cores_with_data = corerange_to_cores(
        core_groups.cores_with_data, std::nullopt, params.input_shard_spec->orientation == ShardOrientation::ROW_MAJOR);
    auto num_cores = cores_with_data.size();
    const auto input_accessor_args = TensorAccessorArgs(*input_buffer);
    const auto output_accessor_args = TensorAccessorArgs(*output_buffer);

    std::vector<uint32_t> compile_time_args{aligned_page_size};
    input_accessor_args.append_to(compile_time_args);
    output_accessor_args.append_to(compile_time_args);

    KernelHandle kernel_id = CreateKernel(
        program,
        kernel_path,
        core_groups.cores_with_data,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = compile_time_args,
            .defines = defines,
        });

    for (size_t core_idx = 0; core_idx < num_cores; ++core_idx) {
        const auto& core = cores_with_data[core_idx];
        std::vector<uint32_t> runtime_args = input_accessor_args.get_common_runtime_args();
        auto n_shards = core_groups.cores_in_group_1.contains(core) ? core_groups.num_shards_per_core_in_group_1
                                                                    : core_groups.num_shards_per_core_in_group_2;
        runtime_args.push_back(input_buffer->address());
        runtime_args.push_back(output_buffer->address());
        runtime_args.push_back(core_idx);   // First shard_id = core_id
        runtime_args.push_back(num_cores);  // Stride for shard_id = num cores
        runtime_args.push_back(n_shards);   // Number of shards to copy for this core
        SetRuntimeArgs(program, kernel_id, core, runtime_args);
    }

    auto mesh_workload = tt::tt_metal::distributed::CreateMeshWorkload();
    mesh_workload.add_program(tt::tt_metal::distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
    EnqueueMeshWorkload(mesh_device->mesh_command_queue(), mesh_workload, true);

    auto output_tensor_cpu = output_tensor.cpu(true);

    // Data should be only in the first shard
    Tensor output_tensor_shard0 = ttnn::distributed::get_device_tensors(output_tensor_cpu).front();
    auto output_vec = output_tensor_shard0.to_vector<T>();

    EXPECT_EQ(output_vec, src);
}

template <typename T>
static void test_multi_core_interleaved_copy(
    const CopyParams& params,
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const CoreRangeSet& cores,
    const std::map<std::string, std::string>& defines = {}) {
    // Create interleaved memory config
    MemoryConfig mem_config = MemoryConfig(TensorMemoryLayout::INTERLEAVED, params.buffer_type);
    TensorSpec tensor_spec(params.tensor_shape, TensorLayout(params.dtype, PageConfig(params.layout), mem_config));

    const auto src = tt::test_utils::generate_uniform_random_vector<T>(0, UINT8_MAX, params.tensor_shape.volume());

    auto input_tensor = Tensor::from_vector(src, tensor_spec, mesh_device);
    auto output_tensor = Tensor::from_vector(std::vector<T>(params.tensor_shape.volume()), tensor_spec, mesh_device);

    auto input_buffer = input_tensor.buffer();
    auto output_buffer = output_tensor.buffer();
    auto aligned_page_size = input_buffer->aligned_page_size();
    if (output_buffer->aligned_page_size() != aligned_page_size) {
        GTEST_SKIP() << "Input and output buffers must have the same aligned page size!";
    }

    auto program = CreateProgram();
    const auto data_format = datatype_to_dataformat_converter(params.dtype);

    // Create circular buffer for each core
    constexpr auto num_tiles = 2;
    CBHandle cb_idx = tt::CBIndex::c_0;
    auto cb_config = CircularBufferConfig(aligned_page_size * num_tiles, {{cb_idx, data_format}})
                         .set_page_size(cb_idx, aligned_page_size);
    CreateCircularBuffer(program, cores, cb_config);

    auto cores_vec = corerange_to_cores(cores, std::nullopt, true);
    auto num_cores = cores_vec.size();
    auto total_pages = input_buffer->num_pages();

    const auto input_accessor_args = TensorAccessorArgs(*input_buffer);
    const auto output_accessor_args = TensorAccessorArgs(*output_buffer);

    // Reader compile-time args: input accessor args + cb_id + page_size
    std::vector<uint32_t> reader_compile_time_args = input_accessor_args.get_compile_time_args();
    reader_compile_time_args.push_back(cb_idx);
    reader_compile_time_args.push_back(aligned_page_size);

    // Writer compile-time args: output accessor args + cb_id + page_size
    std::vector<uint32_t> writer_compile_time_args = output_accessor_args.get_compile_time_args();
    writer_compile_time_args.push_back(cb_idx);
    writer_compile_time_args.push_back(aligned_page_size);

    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "tests/ttnn/unit_tests/gtests/accessor/kernels/reader_interleaved_multi_core.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = reader_compile_time_args,
            .defines = defines,
        });

    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "tests/ttnn/unit_tests/gtests/accessor/kernels/writer_interleaved_multi_core.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = writer_compile_time_args,
            .defines = defines,
        });

    // Set runtime args for each core
    for (uint32_t core_idx = 0; core_idx < num_cores; ++core_idx) {
        const auto& core = cores_vec[core_idx];

        // Calculate page range for this core
        uint32_t pages_per_core = total_pages / num_cores;
        uint32_t extra_pages = total_pages % num_cores;
        uint32_t start_page_id = core_idx * pages_per_core + std::min(core_idx, extra_pages);
        uint32_t end_page_id = start_page_id + pages_per_core + (core_idx < extra_pages ? 1 : 0);

        // Reader runtime args: input accessor common runtime args + input_base_address + start_page_id + end_page_id
        std::vector<uint32_t> reader_runtime_args;
        reader_runtime_args.push_back(input_buffer->address());
        reader_runtime_args.push_back(start_page_id);
        reader_runtime_args.push_back(end_page_id);
        SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);

        // Writer runtime args: output accessor common runtime args + output_base_address + start_page_id + end_page_id
        std::vector<uint32_t> writer_runtime_args;
        writer_runtime_args.push_back(output_buffer->address());
        writer_runtime_args.push_back(start_page_id);
        writer_runtime_args.push_back(end_page_id);
        SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);
    }

    auto mesh_workload = tt::tt_metal::distributed::CreateMeshWorkload();
    mesh_workload.add_program(tt::tt_metal::distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
    EnqueueMeshWorkload(mesh_device->mesh_command_queue(), mesh_workload, true);

    auto output_tensor_cpu = output_tensor.cpu(true);
    Tensor output_tensor_device = ttnn::distributed::get_device_tensors(output_tensor_cpu).front();
    auto output_vec = output_tensor_device.to_vector<T>();

    EXPECT_EQ(output_vec, src);
}

template <typename T>
static void test_single_core_copy(
    const CopyParams& params, tt::tt_metal::distributed::MeshDevice* mesh_device, bool is_interleaved = false) {
    // Create memory config based on whether it's interleaved or sharded
    MemoryConfig mem_config = is_interleaved ? MemoryConfig(TensorMemoryLayout::INTERLEAVED, params.buffer_type)
                                             : MemoryConfig(params.buffer_type, params.input_shard_spec);
    TensorSpec tensor_spec(params.tensor_shape, TensorLayout(params.dtype, PageConfig(params.layout), mem_config));

    const auto src = tt::test_utils::generate_uniform_random_vector<T>(0, UINT8_MAX, params.tensor_shape.volume());

    auto input_tensor = Tensor::from_vector(src, tensor_spec, mesh_device);
    auto output_tensor = Tensor::from_vector(std::vector<T>(params.tensor_shape.volume()), tensor_spec, mesh_device);

    auto input_buffer = input_tensor.buffer();
    auto output_buffer = output_tensor.buffer();
    auto aligned_page_size = input_buffer->aligned_page_size();
    if (output_buffer->aligned_page_size() != aligned_page_size) {
        GTEST_SKIP() << "Input and output buffers must have the same aligned page size!";
    }

    auto program = CreateProgram();

    constexpr CoreCoord grid = {0, 0};
    const auto data_format = datatype_to_dataformat_converter(params.dtype);

    constexpr auto num_tiles = 2;
    CBHandle cb_in0_idx = tt::CBIndex::c_0;
    auto c_in0_config = CircularBufferConfig(aligned_page_size * num_tiles, {{cb_in0_idx, data_format}})
                            .set_page_size(cb_in0_idx, aligned_page_size);
    CreateCircularBuffer(program, grid, c_in0_config);

    const auto input_accessor_args = TensorAccessorArgs(*input_buffer);
    const auto output_accessor_args = TensorAccessorArgs(*output_buffer);

    std::vector<uint32_t> input_compile_time_args = input_accessor_args.get_compile_time_args();
    input_compile_time_args.push_back(cb_in0_idx);
    input_compile_time_args.push_back(aligned_page_size);
    input_compile_time_args.push_back(
        input_buffer->num_pages());  // tensor volume (used for unified interface, ignored by sharded)

    std::vector<uint32_t> output_compile_time_args = output_accessor_args.get_compile_time_args();
    output_compile_time_args.push_back(cb_in0_idx);
    output_compile_time_args.push_back(aligned_page_size);
    output_compile_time_args.push_back(
        output_buffer->num_pages());  // tensor volume (used for unified interface, ignored by sharded)

    // Set up kernel defines for interleaved layout if needed
    std::map<std::string, std::string> kernel_defines;
    if (is_interleaved) {
        kernel_defines["INTERLEAVED_LAYOUT"] = "1";
    }

    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "tests/ttnn/unit_tests/gtests/accessor/kernels/reader_copy_all_pages.cpp",
        grid,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = input_compile_time_args,
            .defines = kernel_defines,
        });

    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "tests/ttnn/unit_tests/gtests/accessor/kernels/writer_copy_all_pages.cpp",
        grid,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = output_compile_time_args,
            .defines = kernel_defines,
        });

    std::vector<uint32_t> input_runtime_args{input_buffer->address()};
    SetCommonRuntimeArgs(program, reader_kernel_id, input_runtime_args);

    std::vector<uint32_t> output_runtime_args{output_buffer->address()};
    SetCommonRuntimeArgs(program, writer_kernel_id, output_runtime_args);

    auto mesh_workload = tt::tt_metal::distributed::CreateMeshWorkload();
    mesh_workload.add_program(tt::tt_metal::distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
    EnqueueMeshWorkload(mesh_device->mesh_command_queue(), mesh_workload, true);

    auto output_tensor_cpu = output_tensor.cpu(true);

    // Data should be only in the first shard
    Tensor output_tensor_device = ttnn::distributed::get_device_tensors(output_tensor_cpu).front();
    auto output_vec = output_tensor_device.to_vector<T>();

    EXPECT_EQ(output_vec, src);
}

}  // namespace tensor_accessor_device_tests

using namespace tensor_accessor_device_tests;
using namespace tt::tt_metal;

class ShardedAccessorTestsReshardOnDevice : public GenericMeshDeviceFixture,
                                            public ::testing::WithParamInterface<InputOutputBufferParams> {};

TEST_P(ShardedAccessorTestsReshardOnDevice, SingleCoreReshard) {
    const auto& params = GetParam();

    switch (params.dtype) {
        case DataType::UINT8: test_single_core_reshard<uint8_t>(params, mesh_device_.get()); break;
        case DataType::UINT16: test_single_core_reshard<uint16_t>(params, mesh_device_.get()); break;
        default: TT_THROW("Unsupported data type");
    }
}

std::vector<InputOutputBufferParams> get_sharded_accessor_test_params() {
    std::vector<InputOutputBufferParams> base_params{
        InputOutputBufferParams{
            .tensor_shape = tt::tt_metal::Shape{4, 64, 96},
            .layout = Layout::TILE,
            .dtype = DataType::UINT16,
            .input_buffer_type = BufferType::L1,
            .output_buffer_type = BufferType::L1,

            .input_shard_spec =
                NdShardSpec{
                    .shard_shape = tt::tt_metal::Shape{1, 32, 64},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {3, 3})),
                    .orientation = ShardOrientation::ROW_MAJOR,
                },
            .output_shard_spec =
                NdShardSpec{
                    .shard_shape = tt::tt_metal::Shape{2, 32, 32},
                    .grid = CoreRangeSet(CoreRange({4, 4}, {5, 5})),
                    .orientation = ShardOrientation::ROW_MAJOR,
                },
        },
        InputOutputBufferParams{
            .tensor_shape = tt::tt_metal::Shape{18, 128, 64},
            .layout = Layout::TILE,
            .dtype = DataType::UINT8,
            .input_buffer_type = BufferType::L1,
            .output_buffer_type = BufferType::L1,

            .input_shard_spec =
                NdShardSpec{
                    .shard_shape = tt::tt_metal::Shape{1, 64, 96},
                    .grid = CoreRangeSet(tt::stl::Span<const CoreRange>(
                        {CoreRange({4, 6}, {6, 6}), CoreRange({1, 1}, {1, 1}), CoreRange({0, 3}, {3, 3})})),
                    .orientation = ShardOrientation::ROW_MAJOR,
                },
            .output_shard_spec =
                NdShardSpec{
                    .shard_shape = tt::tt_metal::Shape{1, 32, 32},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {5, 5})),
                    .orientation = ShardOrientation::COL_MAJOR,
                },
        },
        InputOutputBufferParams{
            .tensor_shape = tt::tt_metal::Shape{2, 3, 256},
            .layout = Layout::ROW_MAJOR,
            .dtype = DataType::UINT8,
            .input_buffer_type = BufferType::L1,
            .output_buffer_type = BufferType::L1,

            .input_shard_spec =
                NdShardSpec{
                    .shard_shape = tt::tt_metal::Shape{2, 4, 16},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {3, 3})),
                    .orientation = ShardOrientation::ROW_MAJOR,
                },
            .output_shard_spec =
                NdShardSpec{
                    .shard_shape = tt::tt_metal::Shape{1, 3, 16},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {2, 2})),
                    .orientation = ShardOrientation::COL_MAJOR,
                },
        },
        InputOutputBufferParams{
            .tensor_shape = tt::tt_metal::Shape{3, 2, 2, 3, 4},
            .layout = Layout::ROW_MAJOR,
            .dtype = DataType::UINT8,
            .input_buffer_type = BufferType::L1,
            .output_buffer_type = BufferType::L1,

            .input_shard_spec =
                NdShardSpec{
                    .shard_shape = tt::tt_metal::Shape{1, 1, 2, 2, 4},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {1, 4})),
                    .orientation = ShardOrientation::COL_MAJOR,
                },
            .output_shard_spec =
                NdShardSpec{
                    .shard_shape = tt::tt_metal::Shape{3, 3, 1, 1, 4},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {1, 1})),
                    .orientation = ShardOrientation::ROW_MAJOR,
                },
        },
        InputOutputBufferParams{
            .tensor_shape = tt::tt_metal::Shape{5, 2, 2, 64, 96},
            .layout = Layout::TILE,
            .dtype = DataType::UINT16,
            .input_buffer_type = BufferType::L1,
            .output_buffer_type = BufferType::L1,

            .input_shard_spec =
                NdShardSpec{
                    .shard_shape = tt::tt_metal::Shape{1, 1, 2, 64, 64},
                    .grid = CoreRangeSet(
                        tt::stl::Span<const CoreRange>({CoreRange({0, 0}, {2, 0}), CoreRange({0, 1}, {1, 1})})),
                    .orientation = ShardOrientation::COL_MAJOR,
                },
            .output_shard_spec =
                NdShardSpec{
                    .shard_shape = tt::tt_metal::Shape{5, 1, 1, 96, 32},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {3, 3})),
                    .orientation = ShardOrientation::ROW_MAJOR,
                },
        },
        InputOutputBufferParams{
            .tensor_shape = tt::tt_metal::Shape{64, 64},
            .layout = Layout::TILE,
            .dtype = DataType::UINT16,
            .input_buffer_type = BufferType::DRAM,
            .output_buffer_type = BufferType::L1,

            .input_shard_spec =
                NdShardSpec{
                    .shard_shape = tt::tt_metal::Shape{32, 32},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {3, 0})),
                    .orientation = ShardOrientation::ROW_MAJOR,
                },
            .output_shard_spec =
                NdShardSpec{
                    .shard_shape = tt::tt_metal::Shape{32, 32},
                    .grid = CoreRangeSet(CoreRange({4, 4}, {5, 5})),
                    .orientation = ShardOrientation::ROW_MAJOR,
                },
        },
        InputOutputBufferParams{
            .tensor_shape = tt::tt_metal::Shape{4, 64, 96},
            .layout = Layout::TILE,
            .dtype = DataType::UINT16,
            .input_buffer_type = BufferType::DRAM,
            .output_buffer_type = BufferType::L1,

            .input_shard_spec =
                NdShardSpec{
                    .shard_shape = tt::tt_metal::Shape{1, 64, 32},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {3, 0})),
                    .orientation = ShardOrientation::ROW_MAJOR,
                },
            .output_shard_spec =
                NdShardSpec{
                    .shard_shape = tt::tt_metal::Shape{2, 32, 32},
                    .grid = CoreRangeSet(CoreRange({4, 4}, {5, 5})),
                    .orientation = ShardOrientation::ROW_MAJOR,
                },
        },
        InputOutputBufferParams{
            .tensor_shape = tt::tt_metal::Shape{18, 128, 64},
            .layout = Layout::TILE,
            .dtype = DataType::UINT8,
            .input_buffer_type = BufferType::L1,
            .output_buffer_type = BufferType::DRAM,

            .input_shard_spec =
                NdShardSpec{
                    .shard_shape = tt::tt_metal::Shape{1, 64, 96},
                    .grid = CoreRangeSet(tt::stl::Span<const CoreRange>(
                        {CoreRange({4, 6}, {6, 6}), CoreRange({1, 1}, {1, 1}), CoreRange({0, 3}, {3, 3})})),
                    .orientation = ShardOrientation::ROW_MAJOR,
                },
            .output_shard_spec =
                NdShardSpec{
                    .shard_shape = tt::tt_metal::Shape{2, 32, 32},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {0, 0})),
                    .orientation = ShardOrientation::COL_MAJOR,
                },
        },
        InputOutputBufferParams{
            .tensor_shape = tt::tt_metal::Shape{2, 3, 256},
            .layout = Layout::ROW_MAJOR,
            .dtype = DataType::UINT8,
            .input_buffer_type = BufferType::DRAM,
            .output_buffer_type = BufferType::DRAM,

            .input_shard_spec =
                NdShardSpec{
                    .shard_shape = tt::tt_metal::Shape{2, 4, 16},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {0, 0})),
                    .orientation = ShardOrientation::ROW_MAJOR,
                },
            .output_shard_spec =
                NdShardSpec{
                    .shard_shape = tt::tt_metal::Shape{1, 3, 16},
                    .grid = CoreRangeSet(CoreRange({0, 0}, {0, 0})),
                    .orientation = ShardOrientation::COL_MAJOR,
                },
        }};

    std::vector<InputOutputBufferParams> test_params;
    for (const auto& base_param : base_params) {
        // All combinations of runtime/static arguments
        auto all_args_combinations = get_all_sharded_args_configs();
        for (const auto& arg_config : all_args_combinations) {
            for (int src_interleaved = 0; src_interleaved <= 1; ++src_interleaved) {
                for (int dst_interleaved = 0; dst_interleaved <= 1; ++dst_interleaved) {
                    auto p = base_param;
                    p.crta_config = arg_config;
                    if (src_interleaved) {
                        p.input_shard_spec = std::nullopt;
                    }
                    if (dst_interleaved) {
                        p.output_shard_spec = std::nullopt;
                    }
                    test_params.push_back(p);
                }
            }
        }
    }
    return test_params;
}

INSTANTIATE_TEST_SUITE_P(
    ShardedAccessorTests, ShardedAccessorTestsReshardOnDevice, testing::ValuesIn(get_sharded_accessor_test_params()));

class ShardedAccessorTestsCopyOnDevice : public GenericMeshDeviceFixture,
                                         public ::testing::WithParamInterface<CopyParams> {};

TEST_P(ShardedAccessorTestsCopyOnDevice, MultiCoreCopyLocal) {
    const auto& params = GetParam();

    const std::string kernel_path = "tests/ttnn/unit_tests/gtests/accessor/kernels/copy_local.cpp";
    switch (params.dtype) {
        case DataType::UINT8: test_multi_core_copy<uint8_t>(params, mesh_device_.get(), kernel_path); break;
        case DataType::UINT16: test_multi_core_copy<uint16_t>(params, mesh_device_.get(), kernel_path); break;
        case DataType::BFLOAT16: test_multi_core_copy<bfloat16>(params, mesh_device_.get(), kernel_path); break;
        default: TT_THROW("Unsupported data type");
    }
}

TEST_P(ShardedAccessorTestsCopyOnDevice, MultiCoreCopyLocalShardIterator) {
    const auto& params = GetParam();

    const std::string kernel_path = "tests/ttnn/unit_tests/gtests/accessor/kernels/copy_local_shard_iterator.cpp";
    switch (params.dtype) {
        case DataType::UINT8: test_multi_core_copy<uint8_t>(params, mesh_device_.get(), kernel_path); break;
        case DataType::UINT16: test_multi_core_copy<uint16_t>(params, mesh_device_.get(), kernel_path); break;
        case DataType::BFLOAT16: test_multi_core_copy<bfloat16>(params, mesh_device_.get(), kernel_path); break;
        default: TT_THROW("Unsupported data type");
    }
}

TEST_P(ShardedAccessorTestsCopyOnDevice, MultiCoreCopyLocalShardIteratorBigStep) {
    const auto& params = GetParam();

    const std::string kernel_path = "tests/ttnn/unit_tests/gtests/accessor/kernels/copy_local_shard_iterator.cpp";
    switch (params.dtype) {
        case DataType::UINT8:
            test_multi_core_copy<uint8_t>(params, mesh_device_.get(), kernel_path, {{"BIG_STEP", "2"}});
            break;
        case DataType::UINT16:
            test_multi_core_copy<uint16_t>(params, mesh_device_.get(), kernel_path, {{"BIG_STEP", "2"}});
            break;
        case DataType::BFLOAT16:
            test_multi_core_copy<bfloat16>(params, mesh_device_.get(), kernel_path, {{"BIG_STEP", "2"}});
            break;
        default: TT_THROW("Unsupported data type");
    }
}

TEST_P(ShardedAccessorTestsCopyOnDevice, SingleCoreCopyAllPages) {
    const auto& params = GetParam();

    switch (params.dtype) {
        case DataType::UINT8: test_single_core_copy<uint8_t>(params, mesh_device_.get()); break;
        case DataType::UINT16: test_single_core_copy<uint16_t>(params, mesh_device_.get()); break;
        case DataType::BFLOAT16: test_single_core_copy<bfloat16>(params, mesh_device_.get()); break;
        default: TT_THROW("Unsupported data type");
    }
}

class InterleavedAccessorTestsCopyOnDevice : public GenericMeshDeviceFixture,
                                             public ::testing::WithParamInterface<CopyParams> {};

TEST_P(InterleavedAccessorTestsCopyOnDevice, SingleCoreCopyAllPages) {
    const auto& params = GetParam();

    switch (params.dtype) {
        case DataType::UINT8: test_single_core_copy<uint8_t>(params, mesh_device_.get(), true); break;
        case DataType::UINT16: test_single_core_copy<uint16_t>(params, mesh_device_.get(), true); break;
        case DataType::BFLOAT16: test_single_core_copy<bfloat16>(params, mesh_device_.get(), true); break;
        default: TT_THROW("Unsupported data type");
    }
}

TEST_P(InterleavedAccessorTestsCopyOnDevice, MultiCoreCopyAllPages) {
    const auto& params = GetParam();

    // Use all available cores for multi-core testing
    auto device = mesh_device_->get_devices().at(0);
    auto grid_size = device->compute_with_storage_grid_size();
    CoreRangeSet cores = CoreRangeSet(CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1}));

    switch (params.dtype) {
        case DataType::UINT8: test_multi_core_interleaved_copy<uint8_t>(params, mesh_device_.get(), cores); break;
        case DataType::UINT16: test_multi_core_interleaved_copy<uint16_t>(params, mesh_device_.get(), cores); break;
        case DataType::BFLOAT16: test_multi_core_interleaved_copy<bfloat16>(params, mesh_device_.get(), cores); break;
        default: TT_THROW("Unsupported data type");
    }
}

TEST_P(InterleavedAccessorTestsCopyOnDevice, MultiCoreCopyAllPagesBigStep) {
    const auto& params = GetParam();

    // Use all available cores for multi-core testing
    auto device = mesh_device_->get_devices().at(0);
    auto grid_size = device->compute_with_storage_grid_size();
    CoreRangeSet cores = CoreRangeSet(CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1}));

    switch (params.dtype) {
        case DataType::UINT8:
            test_multi_core_interleaved_copy<uint8_t>(params, mesh_device_.get(), cores, {{"BIG_STEP", "2"}});
            break;
        case DataType::UINT16:
            test_multi_core_interleaved_copy<uint16_t>(params, mesh_device_.get(), cores, {{"BIG_STEP", "2"}});
            break;
        case DataType::BFLOAT16:
            test_multi_core_interleaved_copy<bfloat16>(params, mesh_device_.get(), cores, {{"BIG_STEP", "2"}});
            break;
        default: TT_THROW("Unsupported data type");
    }
}

INSTANTIATE_TEST_SUITE_P(
    InterleavedAccessorTests,
    InterleavedAccessorTestsCopyOnDevice,
    testing::ValuesIn({
        // 2D cases - L1 buffer type
        CopyParams{
            .tensor_shape = tt::tt_metal::Shape{64, 128},
            .layout = Layout::TILE,
            .dtype = DataType::UINT8,
            .buffer_type = BufferType::L1,
        },
        CopyParams{
            .tensor_shape = tt::tt_metal::Shape{96, 64},
            .layout = Layout::ROW_MAJOR,
            .dtype = DataType::UINT16,
            .buffer_type = BufferType::L1,
        },
        CopyParams{
            .tensor_shape = tt::tt_metal::Shape{128, 96},
            .layout = Layout::TILE,
            .dtype = DataType::BFLOAT16,
            .buffer_type = BufferType::L1,
        },

        // 2D cases - DRAM buffer type
        CopyParams{
            .tensor_shape = tt::tt_metal::Shape{64, 128},
            .layout = Layout::TILE,
            .dtype = DataType::UINT8,
            .buffer_type = BufferType::DRAM,
        },
        CopyParams{
            .tensor_shape = tt::tt_metal::Shape{96, 64},
            .layout = Layout::ROW_MAJOR,
            .dtype = DataType::UINT16,
            .buffer_type = BufferType::DRAM,
        },
        CopyParams{
            .tensor_shape = tt::tt_metal::Shape{128, 96},
            .layout = Layout::TILE,
            .dtype = DataType::BFLOAT16,
            .buffer_type = BufferType::DRAM,
        },

        // 3D cases - L1 buffer type
        CopyParams{
            .tensor_shape = tt::tt_metal::Shape{8, 64, 64},
            .layout = Layout::TILE,
            .dtype = DataType::BFLOAT16,
            .buffer_type = BufferType::L1,
        },
        CopyParams{
            .tensor_shape = tt::tt_metal::Shape{12, 96, 32},
            .layout = Layout::ROW_MAJOR,
            .dtype = DataType::UINT8,
            .buffer_type = BufferType::L1,
        },

        // 3D cases - DRAM buffer type
        CopyParams{
            .tensor_shape = tt::tt_metal::Shape{8, 64, 64},
            .layout = Layout::TILE,
            .dtype = DataType::BFLOAT16,
            .buffer_type = BufferType::DRAM,
        },
        CopyParams{
            .tensor_shape = tt::tt_metal::Shape{12, 96, 32},
            .layout = Layout::ROW_MAJOR,
            .dtype = DataType::UINT8,
            .buffer_type = BufferType::DRAM,
        },

        // 4D cases - L1 buffer type
        CopyParams{
            .tensor_shape = tt::tt_metal::Shape{4, 6, 128},
            .layout = Layout::ROW_MAJOR,
            .dtype = DataType::BFLOAT16,
            .buffer_type = BufferType::L1,
        },
        CopyParams{
            .tensor_shape = tt::tt_metal::Shape{256, 64},
            .layout = Layout::ROW_MAJOR,
            .dtype = DataType::UINT8,
            .buffer_type = BufferType::L1,
        },

        // 4D cases - DRAM buffer type
        CopyParams{
            .tensor_shape = tt::tt_metal::Shape{4, 6, 128},
            .layout = Layout::ROW_MAJOR,
            .dtype = DataType::BFLOAT16,
            .buffer_type = BufferType::DRAM,
        },
        CopyParams{
            .tensor_shape = tt::tt_metal::Shape{256, 64},
            .layout = Layout::ROW_MAJOR,
            .dtype = DataType::UINT8,
            .buffer_type = BufferType::DRAM,
        },

        // Higher dimensional cases - L1 buffer type
        CopyParams{
            .tensor_shape = tt::tt_metal::Shape{6, 64, 128},
            .layout = Layout::TILE,
            .dtype = DataType::BFLOAT16,
            .buffer_type = BufferType::L1,
        },
        CopyParams{
            .tensor_shape = tt::tt_metal::Shape{3, 2, 2, 3, 4},
            .layout = Layout::ROW_MAJOR,
            .dtype = DataType::BFLOAT16,
            .buffer_type = BufferType::L1,
        },

        // Higher dimensional cases - DRAM buffer type
        CopyParams{
            .tensor_shape = tt::tt_metal::Shape{6, 64, 128},
            .layout = Layout::TILE,
            .dtype = DataType::BFLOAT16,
            .buffer_type = BufferType::DRAM,
        },
        CopyParams{
            .tensor_shape = tt::tt_metal::Shape{3, 2, 2, 3, 4},
            .layout = Layout::ROW_MAJOR,
            .dtype = DataType::BFLOAT16,
            .buffer_type = BufferType::DRAM,
        },
    }));

INSTANTIATE_TEST_SUITE_P(
    ShardedAccessorTests,
    ShardedAccessorTestsCopyOnDevice,
    testing::ValuesIn({// 2D cases at the beginning
                       CopyParams{
                           .tensor_shape = tt::tt_metal::Shape{64, 128},
                           .layout = Layout::TILE,
                           .dtype = DataType::UINT8,
                           .buffer_type = BufferType::L1,

                           .input_shard_spec =
                               NdShardSpec{
                                   .shard_shape = tt::tt_metal::Shape{32, 32},
                                   .grid = CoreRangeSet(CoreRange({0, 0}, {1, 1})),
                                   .orientation = ShardOrientation::ROW_MAJOR,
                               },
                       },
                       CopyParams{
                           .tensor_shape = tt::tt_metal::Shape{96, 64},
                           .layout = Layout::ROW_MAJOR,
                           .dtype = DataType::UINT16,
                           .buffer_type = BufferType::L1,

                           .input_shard_spec =
                               NdShardSpec{
                                   .shard_shape = tt::tt_metal::Shape{24, 32},
                                   .grid = CoreRangeSet(CoreRange({0, 0}, {1, 1})),
                                   .orientation = ShardOrientation::COL_MAJOR,
                               },
                       },
                       CopyParams{
                           .tensor_shape = tt::tt_metal::Shape{128, 96},
                           .layout = Layout::TILE,
                           .dtype = DataType::BFLOAT16,
                           .buffer_type = BufferType::L1,

                           .input_shard_spec =
                               NdShardSpec{
                                   .shard_shape = tt::tt_metal::Shape{32, 32},
                                   .grid = CoreRangeSet(CoreRange({0, 0}, {2, 2})),
                                   .orientation = ShardOrientation::ROW_MAJOR,
                               },
                       },

                       // 3D cases
                       CopyParams{
                           .tensor_shape = tt::tt_metal::Shape{4, 64, 96},
                           .layout = Layout::TILE,
                           .dtype = DataType::UINT16,
                           .buffer_type = BufferType::L1,

                           .input_shard_spec =
                               NdShardSpec{
                                   .shard_shape = tt::tt_metal::Shape{1, 32, 64},
                                   .grid = CoreRangeSet(CoreRange({0, 0}, {3, 3})),
                                   .orientation = ShardOrientation::ROW_MAJOR,
                               },
                       },
                       CopyParams{
                           .tensor_shape = tt::tt_metal::Shape{1, 2, 32, 32},
                           .layout = Layout::TILE,
                           .dtype = DataType::BFLOAT16,
                           .buffer_type = BufferType::L1,

                           .input_shard_spec =
                               NdShardSpec{
                                   .shard_shape = tt::tt_metal::Shape{32, 32},
                                   .grid = CoreRangeSet(CoreRange({0, 0}, {0, 1})),
                                   .orientation = ShardOrientation::ROW_MAJOR,
                                   .shard_distribution_strategy = ShardDistributionStrategy::GRID_2D,
                               },
                       },
                       CopyParams{
                           .tensor_shape = tt::tt_metal::Shape{2, 64, 128},
                           .layout = Layout::ROW_MAJOR,
                           .dtype = DataType::BFLOAT16,
                           .buffer_type = BufferType::L1,

                           .input_shard_spec =
                               NdShardSpec{
                                   .shard_shape = tt::tt_metal::Shape{32, 32},
                                   .grid = CoreRangeSet(CoreRange({0, 0}, {7, 7})),
                                   .orientation = ShardOrientation::ROW_MAJOR,
                                   .shard_distribution_strategy = ShardDistributionStrategy::GRID_2D,
                               },
                       },
                       CopyParams{
                           .tensor_shape = tt::tt_metal::Shape{18, 128, 64},
                           .layout = Layout::TILE,
                           .dtype = DataType::UINT8,
                           .buffer_type = BufferType::L1,

                           .input_shard_spec =
                               NdShardSpec{
                                   .shard_shape = tt::tt_metal::Shape{1, 64, 96},
                                   .grid = CoreRangeSet(CoreRange({0, 0}, {3, 4})),
                                   .orientation = ShardOrientation::ROW_MAJOR,
                               },
                       },
                       CopyParams{
                           .tensor_shape = tt::tt_metal::Shape{2, 3, 256},
                           .layout = Layout::ROW_MAJOR,
                           .dtype = DataType::UINT8,
                           .buffer_type = BufferType::L1,

                           .input_shard_spec =
                               NdShardSpec{
                                   .shard_shape = tt::tt_metal::Shape{2, 4, 16},
                                   .grid = CoreRangeSet(CoreRange({0, 0}, {3, 3})),
                                   .orientation = ShardOrientation::ROW_MAJOR,
                               },
                       },

                       // More L1->L1 cases
                       CopyParams{
                           .tensor_shape = tt::tt_metal::Shape{32, 192},
                           .layout = Layout::TILE,
                           .dtype = DataType::UINT16,
                           .buffer_type = BufferType::L1,

                           .input_shard_spec =
                               NdShardSpec{
                                   .shard_shape = tt::tt_metal::Shape{32, 32},
                                   .grid = CoreRangeSet(CoreRange({0, 0}, {1, 2})),
                                   .orientation = ShardOrientation::ROW_MAJOR,
                               },
                       },
                       CopyParams{
                           .tensor_shape = tt::tt_metal::Shape{8, 64, 64},
                           .layout = Layout::TILE,
                           .dtype = DataType::BFLOAT16,
                           .buffer_type = BufferType::L1,

                           .input_shard_spec =
                               NdShardSpec{
                                   .shard_shape = tt::tt_metal::Shape{2, 32, 32},
                                   .grid = CoreRangeSet(CoreRange({0, 0}, {1, 1})),
                                   .orientation = ShardOrientation::ROW_MAJOR,
                               },
                       },
                       CopyParams{
                           .tensor_shape = tt::tt_metal::Shape{12, 96, 32},
                           .layout = Layout::ROW_MAJOR,
                           .dtype = DataType::UINT8,
                           .buffer_type = BufferType::L1,

                           .input_shard_spec =
                               NdShardSpec{
                                   .shard_shape = tt::tt_metal::Shape{3, 32, 16},
                                   .grid = CoreRangeSet(CoreRange({0, 0}, {1, 1})),
                                   .orientation = ShardOrientation::COL_MAJOR,
                               },
                       },
                       CopyParams{
                           .tensor_shape = tt::tt_metal::Shape{4, 6, 128},
                           .layout = Layout::ROW_MAJOR,
                           .dtype = DataType::BFLOAT16,
                           .buffer_type = BufferType::L1,

                           .input_shard_spec =
                               NdShardSpec{
                                   .shard_shape = tt::tt_metal::Shape{2, 3, 64},
                                   .grid = CoreRangeSet(CoreRange({0, 0}, {1, 0})),
                                   .orientation = ShardOrientation::ROW_MAJOR,
                               },
                       },
                       CopyParams{
                           .tensor_shape = tt::tt_metal::Shape{256, 64},
                           .layout = Layout::ROW_MAJOR,
                           .dtype = DataType::UINT8,
                           .buffer_type = BufferType::L1,

                           .input_shard_spec =
                               NdShardSpec{
                                   .shard_shape = tt::tt_metal::Shape{64, 32},
                                   .grid = CoreRangeSet(CoreRange({0, 0}, {1, 1})),
                                   .orientation = ShardOrientation::COL_MAJOR,
                               },
                       },

                       // More BFLOAT16 cases
                       CopyParams{
                           .tensor_shape = tt::tt_metal::Shape{6, 64, 128},
                           .layout = Layout::TILE,
                           .dtype = DataType::BFLOAT16,
                           .buffer_type = BufferType::L1,

                           .input_shard_spec =
                               NdShardSpec{
                                   .shard_shape = tt::tt_metal::Shape{2, 32, 64},
                                   .grid = CoreRangeSet(CoreRange({0, 0}, {2, 1})),
                                   .orientation = ShardOrientation::ROW_MAJOR,
                               },
                       },
                       CopyParams{
                           .tensor_shape = tt::tt_metal::Shape{3, 2, 2, 3, 4},
                           .layout = Layout::ROW_MAJOR,
                           .dtype = DataType::BFLOAT16,
                           .buffer_type = BufferType::L1,

                           .input_shard_spec =
                               NdShardSpec{
                                   .shard_shape = tt::tt_metal::Shape{1, 1, 2, 2, 4},
                                   .grid = CoreRangeSet(CoreRange({0, 0}, {1, 4})),
                                   .orientation = ShardOrientation::COL_MAJOR,
                               },
                       },
                       CopyParams{
                           .tensor_shape = tt::tt_metal::Shape{5, 2, 2, 64, 96},
                           .layout = Layout::TILE,
                           .dtype = DataType::UINT16,
                           .buffer_type = BufferType::L1,

                           .input_shard_spec =
                               NdShardSpec{
                                   .shard_shape = tt::tt_metal::Shape{1, 1, 2, 64, 64},
                                   .grid = CoreRangeSet(tt::stl::Span<const CoreRange>(
                                       {CoreRange({0, 0}, {2, 0}), CoreRange({0, 1}, {1, 1})})),
                                   .orientation = ShardOrientation::COL_MAJOR,
                               },
                       }}));
