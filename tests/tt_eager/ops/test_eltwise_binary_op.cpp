// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/constants.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/host_buffer/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "tt_numpy/functions.hpp"

using tt::tt_metal::DataType;
using tt::tt_metal::Device;
using tt::tt_metal::Layout;
using tt::tt_metal::Tensor;
using tt::tt_metal::OwnedStorage;
using tt::tt_metal::Shape;

template <typename BinaryFunction>
Tensor host_function(const Tensor& input_tensor_a, const Tensor& input_tensor_b) {
    auto input_a_buffer = tt::tt_metal::owned_buffer::get_as<bfloat16>(input_tensor_a);
    auto input_b_buffer = tt::tt_metal::owned_buffer::get_as<bfloat16>(input_tensor_b);

    auto output_buffer = tt::tt_metal::owned_buffer::create<bfloat16>(input_tensor_a.volume());

    for (auto index = 0; index < output_buffer.size(); index++) {
        auto value = BinaryFunction{}(input_a_buffer[index].to_float(), input_b_buffer[index].to_float());
        output_buffer[index] = bfloat16(value);
    }
    return Tensor(OwnedStorage{output_buffer}, input_tensor_a.get_legacy_shape(), input_tensor_a.get_dtype(), input_tensor_a.get_layout());
}

template <auto HostFunction, typename DeviceFunction, typename... Args>
bool run_test(const Shape& shape, const DeviceFunction& device_function, Device* device, Args... args) {
    auto input_tensor_a = tt::numpy::random::random(shape, DataType::BFLOAT16);
    auto input_tensor_b = tt::numpy::random::random(shape, DataType::BFLOAT16);

    auto host_output = HostFunction(input_tensor_a, input_tensor_b);
    auto device_output = device_function(input_tensor_a.to(Layout::TILE).to(device), input_tensor_b.to(Layout::TILE).to(device)).cpu().to(Layout::ROW_MAJOR);

    return tt::numpy::allclose<bfloat16>(host_output, device_output, args...);
}

template <auto HostFunction, typename DeviceFunction, typename... Args>
bool run_test_memory_configs(const Shape& shape, const DeviceFunction& device_function, Device* device, MemoryConfig memory_config_a, MemoryConfig memory_config_b, MemoryConfig memory_config_out, Args... args) {
    auto input_tensor_a = tt::numpy::random::random(shape, DataType::BFLOAT16);
    auto input_tensor_b = tt::numpy::random::random(shape, DataType::BFLOAT16);

    auto host_output = HostFunction(input_tensor_a, input_tensor_b);
    auto device_output = device_function(input_tensor_a.to(Layout::TILE).to(device, memory_config_a), input_tensor_b.to(Layout::TILE).to(device, memory_config_b), DataType::BFLOAT16, memory_config_out).cpu().to(Layout::ROW_MAJOR);

    return tt::numpy::allclose<bfloat16>(host_output, device_output, args...);
}

int main() {
    using tt::constants::TILE_HEIGHT;
    using tt::constants::TILE_WIDTH;

    int device_id = 0;
    auto device = tt::tt_metal::CreateDevice(device_id);



    {
        Shape shape = {1, 1, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH};
        auto allclose = run_test<host_function<std::plus<float>>>(shape, ttnn::add, device);
        TT_FATAL(allclose);
    }

    {
        Shape shape = {1, 1, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH};
        auto allclose = run_test<host_function<std::minus<float>>>(shape, ttnn::subtract, device);
        TT_FATAL(allclose);
    }

    {
        Shape shape = {1, 1, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH};
        auto allclose = run_test<host_function<std::multiplies<float>>>(shape, ttnn::multiply, device, 1e-2f, 1e-3f);
        TT_FATAL(allclose);
    }

    {
        Shape shape = {1, 1, 4 * tt::constants::TILE_HEIGHT, 4 * tt::constants::TILE_WIDTH};
        MemoryConfig mem_config_height_sharded = { .memory_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
                                    .buffer_type = tt::tt_metal::BufferType::L1,
                                    .shard_spec =
                                        tt::tt_metal::ShardSpec{
                                            CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{1, 1}}}},
                                            {1 * 32, 4 * 32},
                                            ShardOrientation::ROW_MAJOR}
                                        };
        MemoryConfig mem_config_interleaved = { .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
                                .buffer_type = tt::tt_metal::BufferType::DRAM};
        auto allclose = run_test_memory_configs<host_function<std::plus<float>>>(shape, ttnn::add, device, mem_config_interleaved, mem_config_height_sharded, mem_config_interleaved);
        TT_FATAL(allclose);
        allclose = run_test_memory_configs<host_function<std::plus<float>>>(shape, ttnn::add, device, mem_config_height_sharded, mem_config_interleaved, mem_config_interleaved);
        TT_FATAL(allclose);
        allclose = run_test_memory_configs<host_function<std::plus<float>>>(shape, ttnn::add, device, mem_config_height_sharded, mem_config_height_sharded, mem_config_interleaved);
        TT_FATAL(allclose);
        allclose = run_test_memory_configs<host_function<std::plus<float>>>(shape, ttnn::add, device, mem_config_interleaved, mem_config_interleaved, mem_config_height_sharded);
        TT_FATAL(allclose);
        allclose = run_test_memory_configs<host_function<std::plus<float>>>(shape, ttnn::add, device, mem_config_interleaved, mem_config_height_sharded, mem_config_height_sharded);
        TT_FATAL(allclose);
        allclose = run_test_memory_configs<host_function<std::plus<float>>>(shape, ttnn::add, device, mem_config_height_sharded, mem_config_interleaved, mem_config_height_sharded);
        TT_FATAL(allclose);
        allclose = run_test_memory_configs<host_function<std::plus<float>>>(shape, ttnn::add, device, mem_config_height_sharded, mem_config_height_sharded, mem_config_height_sharded);
        TT_FATAL(allclose);
    }

    {
        Shape shape = {1, 1, 4 * tt::constants::TILE_HEIGHT, 4 * tt::constants::TILE_WIDTH};
        MemoryConfig mem_config_width_sharded = { .memory_layout = tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED,
                                    .buffer_type = tt::tt_metal::BufferType::L1,
                                    .shard_spec =
                                        tt::tt_metal::ShardSpec{
                                            CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{1, 1}}}},
                                            {4 * 32, 1 * 32},
                                            ShardOrientation::ROW_MAJOR}
                                        };
        MemoryConfig mem_config_interleaved = { .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
                                .buffer_type = tt::tt_metal::BufferType::DRAM};
        auto allclose = run_test_memory_configs<host_function<std::plus<float>>>(shape, ttnn::add, device, mem_config_interleaved, mem_config_width_sharded, mem_config_interleaved);
        TT_FATAL(allclose);
        allclose = run_test_memory_configs<host_function<std::plus<float>>>(shape, ttnn::add, device, mem_config_width_sharded, mem_config_interleaved, mem_config_interleaved);
        TT_FATAL(allclose);
        allclose = run_test_memory_configs<host_function<std::plus<float>>>(shape, ttnn::add, device, mem_config_width_sharded, mem_config_width_sharded, mem_config_interleaved);
        TT_FATAL(allclose);
        // allclose = run_test_memory_configs<host_function<std::plus<float>>>(shape, ttnn::add, device, mem_config_interleaved, mem_config_interleaved, mem_config_width_sharded); //needs to be enabled
        // TT_FATAL(allclose);
        allclose = run_test_memory_configs<host_function<std::plus<float>>>(shape, ttnn::add, device, mem_config_interleaved, mem_config_width_sharded, mem_config_width_sharded);
        TT_FATAL(allclose);
        allclose = run_test_memory_configs<host_function<std::plus<float>>>(shape, ttnn::add, device, mem_config_width_sharded, mem_config_interleaved, mem_config_width_sharded);
        TT_FATAL(allclose);
        allclose = run_test_memory_configs<host_function<std::plus<float>>>(shape, ttnn::add, device, mem_config_width_sharded, mem_config_width_sharded, mem_config_width_sharded);
        TT_FATAL(allclose);
    }

    {
        Shape shape = {1, 1, 4 * tt::constants::TILE_HEIGHT, 4 * tt::constants::TILE_WIDTH};
        MemoryConfig mem_config_block_sharded = { .memory_layout = tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED,
                                    .buffer_type = tt::tt_metal::BufferType::L1,
                                    .shard_spec =
                                        tt::tt_metal::ShardSpec{
                                            CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{1, 1}}}},
                                            {2 * 32, 2 * 32},
                                            ShardOrientation::ROW_MAJOR}
                                        };
        MemoryConfig mem_config_interleaved = { .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
                                .buffer_type = tt::tt_metal::BufferType::DRAM};
        auto allclose = run_test_memory_configs<host_function<std::plus<float>>>(shape, ttnn::add, device, mem_config_interleaved, mem_config_block_sharded, mem_config_interleaved);
        TT_FATAL(allclose);
        allclose = run_test_memory_configs<host_function<std::plus<float>>>(shape, ttnn::add, device, mem_config_block_sharded, mem_config_interleaved, mem_config_interleaved);
        TT_FATAL(allclose);
        allclose = run_test_memory_configs<host_function<std::plus<float>>>(shape, ttnn::add, device, mem_config_block_sharded, mem_config_block_sharded, mem_config_interleaved);
        TT_FATAL(allclose);
        // allclose = run_test_memory_configs<host_function<std::plus<float>>>(shape, ttnn::add, device, mem_config_interleaved, mem_config_interleaved, mem_config_block_sharded); //needs to be enabled
        // TT_FATAL(allclose);
        allclose = run_test_memory_configs<host_function<std::plus<float>>>(shape, ttnn::add, device, mem_config_interleaved, mem_config_block_sharded, mem_config_block_sharded);
        TT_FATAL(allclose);
        allclose = run_test_memory_configs<host_function<std::plus<float>>>(shape, ttnn::add, device, mem_config_block_sharded, mem_config_interleaved, mem_config_block_sharded);
        TT_FATAL(allclose);
        allclose = run_test_memory_configs<host_function<std::plus<float>>>(shape, ttnn::add, device, mem_config_block_sharded, mem_config_block_sharded, mem_config_block_sharded);
        TT_FATAL(allclose);
    }

    auto run_binary_ops = [&] {
        {
            Shape shape = {1, 1, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH};
            auto allclose = run_test<host_function<std::plus<float>>>(shape, ttnn::add, device);
            TT_FATAL(allclose);
        }

        {
            Shape shape = {1, 1, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH};
            auto allclose = run_test<host_function<std::minus<float>>>(shape, ttnn::subtract, device);
            TT_FATAL(allclose);
        }

        {
            Shape shape = {1, 1, tt::constants::TILE_HEIGHT * 2, tt::constants::TILE_WIDTH * 2};
            auto allclose = run_test<host_function<std::plus<float>>>(shape, ttnn::add, device);
            TT_FATAL(allclose);
        }

        {
            Shape shape = {1, 1, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH};
            auto allclose =
                run_test<host_function<std::multiplies<float>>>(shape, ttnn::multiply, device, 1e-2f, 1e-3f);
            TT_FATAL(allclose);
        }

        {
            Shape shape = {1, 1, tt::constants::TILE_HEIGHT * 4, tt::constants::TILE_WIDTH * 4};
            auto allclose = run_test<host_function<std::plus<float>>>(shape, ttnn::add, device);
            TT_FATAL(allclose);
        }
    };

    device->enable_program_cache();

    run_binary_ops();
    run_binary_ops();

    // Allocate a tensor to show that the addresses aren't cached
    auto input_tensor =
        tt::numpy::random::uniform(bfloat16(0.0f), bfloat16(0.0f), {1, 1, 32, 32}).to(Layout::TILE).to(device);

    run_binary_ops();

    TT_FATAL(device->num_program_cache_entries() == 3,
        "There are {} entries",
        device->num_program_cache_entries());

    device->disable_and_clear_program_cache();

    TT_FATAL(device->num_program_cache_entries()== 0);

    TT_FATAL(tt::tt_metal::CloseDevice(device));

    return 0;
}
