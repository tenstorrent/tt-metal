// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <tt-metalium/constants.hpp>
#include <functional>

#include <tt-metalium/assert.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/shape.hpp>
#include "ttnn/decorators.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/tensor/enum_types.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

using tt::tt_metal::DataType;
using tt::tt_metal::Layout;
using tt::tt_metal::Tensor;
using tt::tt_metal::distributed::MeshDevice;

template <typename BinaryFunction>
Tensor host_function(const Tensor& input_tensor_a, const Tensor& input_tensor_b) {
    auto input_a_buffer = tt::tt_metal::host_buffer::get_as<bfloat16>(input_tensor_a);
    auto input_b_buffer = tt::tt_metal::host_buffer::get_as<bfloat16>(input_tensor_b);

    auto output_buffer = std::vector<bfloat16>(input_tensor_a.physical_volume());

    for (auto index = 0; index < output_buffer.size(); index++) {
        auto value = BinaryFunction{}(input_a_buffer[index].to_float(), input_b_buffer[index].to_float());
        output_buffer[index] = bfloat16(value);
    }
    return Tensor(
        tt::tt_metal::HostBuffer(std::move(output_buffer)),
        input_tensor_a.logical_shape(),
        input_tensor_a.dtype(),
        input_tensor_a.layout());
}

template <auto HostFunction, typename DeviceFunction, typename... Args>
bool run_test(const ttnn::Shape& shape, const DeviceFunction& device_function, MeshDevice* device, Args... args) {
    auto input_tensor_a = ttnn::random::random(shape, DataType::BFLOAT16);
    auto input_tensor_b = ttnn::random::random(shape, DataType::BFLOAT16);

    auto host_output = HostFunction(input_tensor_a, input_tensor_b);
    auto device_output = device_function(
                             input_tensor_a.to_layout(Layout::TILE).to_device(device),
                             input_tensor_b.to_layout(Layout::TILE).to_device(device))
                             .cpu()
                             .to_layout(Layout::ROW_MAJOR);

    return ttnn::allclose<bfloat16>(host_output, device_output, args...);
}

int main() {
    using tt::constants::TILE_HEIGHT;
    using tt::constants::TILE_WIDTH;

    int device_id = 0;
    auto device_owner = MeshDevice::create_unit_mesh(device_id);
    auto device = device_owner.get();

    {
        ttnn::Shape shape({1, 1, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH});
        auto allclose = run_test<host_function<std::plus<float>>>(shape, ttnn::add, device);
        TT_FATAL(allclose, "Error");
    }

    {
        ttnn::Shape shape({1, 1, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH});
        auto allclose = run_test<host_function<std::minus<float>>>(shape, ttnn::subtract, device);
        TT_FATAL(allclose, "Error");
    }

    {
        ttnn::Shape shape({1, 1, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH});
        auto allclose = run_test<host_function<std::multiplies<float>>>(shape, ttnn::multiply, device, 1e-2f, 1e-3f);
        TT_FATAL(allclose, "Error");
    }

    auto run_binary_ops = [&] {
        {
            ttnn::Shape shape({1, 1, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH});
            auto allclose = run_test<host_function<std::plus<float>>>(shape, ttnn::add, device);
            TT_FATAL(allclose, "Error");
        }

        {
            ttnn::Shape shape({1, 1, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH});
            auto allclose = run_test<host_function<std::minus<float>>>(shape, ttnn::subtract, device);
            TT_FATAL(allclose, "Error");
        }

        {
            ttnn::Shape shape({1, 1, tt::constants::TILE_HEIGHT * 2, tt::constants::TILE_WIDTH * 2});
            auto allclose = run_test<host_function<std::plus<float>>>(shape, ttnn::add, device);
            TT_FATAL(allclose, "Error");
        }

        {
            ttnn::Shape shape({1, 1, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH});
            auto allclose =
                run_test<host_function<std::multiplies<float>>>(shape, ttnn::multiply, device, 1e-2f, 1e-3f);
            TT_FATAL(allclose, "Error");
        }

        {
            ttnn::Shape shape({1, 1, tt::constants::TILE_HEIGHT * 4, tt::constants::TILE_WIDTH * 4});
            auto allclose = run_test<host_function<std::plus<float>>>(shape, ttnn::add, device);
            TT_FATAL(allclose, "Error");
        }
    };

    run_binary_ops();
    run_binary_ops();

    // Allocate a tensor to show that the addresses aren't cached
    auto input_tensor = ttnn::random::uniform(bfloat16(0.0f), bfloat16(0.0f), ttnn::Shape({1, 1, 32, 32}))
                            .to_layout(Layout::TILE)
                            .to_device(device);

    run_binary_ops();

    TT_FATAL(device->num_program_cache_entries() == 3, "There are {} entries", device->num_program_cache_entries());

    device->disable_and_clear_program_cache();

    TT_FATAL(device->num_program_cache_entries() == 0, "Error");

    return 0;
}
