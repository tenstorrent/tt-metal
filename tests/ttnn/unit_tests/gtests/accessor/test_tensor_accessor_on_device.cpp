// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include "gmock/gmock.h"
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
    const CopyParams& params, tt::tt_metal::distributed::MeshDevice* mesh_device, const std::string& kernel_path) {
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
        default: TT_THROW("Unsupported data type");
    }
}

TEST_P(ShardedAccessorTestsCopyOnDevice, MultiCoreCopyLocalShardIterator) {
    const auto& params = GetParam();

    const std::string kernel_path = "tests/ttnn/unit_tests/gtests/accessor/kernels/copy_local_shard_iterator.cpp";
    switch (params.dtype) {
        case DataType::UINT8: test_multi_core_copy<uint8_t>(params, mesh_device_.get(), kernel_path); break;
        case DataType::UINT16: test_multi_core_copy<uint16_t>(params, mesh_device_.get(), kernel_path); break;
        default: TT_THROW("Unsupported data type");
    }
}

INSTANTIATE_TEST_SUITE_P(
    ShardedAccessorTests,
    ShardedAccessorTestsCopyOnDevice,
    testing::ValuesIn(
        {CopyParams{
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
         CopyParams{
             .tensor_shape = tt::tt_metal::Shape{3, 2, 2, 3, 4},
             .layout = Layout::ROW_MAJOR,
             .dtype = DataType::UINT8,
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
                     .grid = CoreRangeSet(
                         tt::stl::Span<const CoreRange>({CoreRange({0, 0}, {2, 0}), CoreRange({0, 1}, {1, 1})})),
                     .orientation = ShardOrientation::COL_MAJOR,
                 },
         }}));
