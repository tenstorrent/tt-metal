// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/reduce/reduce_op.hpp"
#include "tt_numpy/functions.hpp"

#include "tensor/tensor.hpp"
#include "common/constants.hpp"

using tt::tt_metal::Device;
using tt::tt_metal::Tensor;
using tt::tt_metal::DataType;
using tt::tt_metal::Layout;

using tt::tt_metal::ReduceOpMath;
using tt::tt_metal::ReduceOpDim;
using tt::tt_metal::Shape;

template<ReduceOpMath op_math, ReduceOpDim op_dim>
Tensor device_function(const Tensor& input_tensor, Device* device) {
    return reduce(input_tensor.to(device), op_math, op_dim).cpu();
}

template<auto DeviceFunction, typename ... Args>
void run_test(Device* device, Args ...  args) {
    Shape shape = {1, 1, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH};
    auto input_tensor = tt::numpy::random::random(shape, DataType::BFLOAT16).to(Layout::TILE);
    auto device_output = DeviceFunction(input_tensor, device);
    // What should be validated? allclose against host implementation?
}

int main () {
    using tt::constants::TILE_HEIGHT;
    using tt::constants::TILE_WIDTH;

    int device_id = 0;
    auto device = tt::tt_metal::CreateDevice(device_id);



    {
        run_test<device_function<ReduceOpMath::SUM, ReduceOpDim::H>>( device);
    }

    {
        run_test<device_function<ReduceOpMath::SUM, ReduceOpDim::W>>( device);
    }

    {
        run_test<device_function<ReduceOpMath::SUM, ReduceOpDim::HW>>( device);
    }

    {
        run_test<device_function<ReduceOpMath::MAX, ReduceOpDim::H>>( device);
    }

    {
        run_test<device_function<ReduceOpMath::MAX, ReduceOpDim::W>>( device);
    }

    {
        run_test<device_function<ReduceOpMath::MAX, ReduceOpDim::HW>>( device);
    }

    device->enable_program_cache();

    auto run_reduce_ops = [&] {
        {
            run_test<device_function<ReduceOpMath::SUM, ReduceOpDim::H>>( device);
        }

        {
            run_test<device_function<ReduceOpMath::SUM, ReduceOpDim::W>>( device);
        }

        {
            run_test<device_function<ReduceOpMath::SUM, ReduceOpDim::HW>>( device);
        }

        {
            run_test<device_function<ReduceOpMath::MAX, ReduceOpDim::H>>( device);
        }

        {
            run_test<device_function<ReduceOpMath::MAX, ReduceOpDim::W>>( device);
        }

        {
            run_test<device_function<ReduceOpMath::MAX, ReduceOpDim::HW>>( device);
        }
    };

    run_reduce_ops();
    run_reduce_ops();

    // Allocate a tensor to show that the addresses aren't cached
    auto some_tensor = tt::numpy::random::uniform(bfloat16(0.0f), bfloat16(0.0f), {1, 1, 32, 32}).to(Layout::TILE).to(device);

    run_reduce_ops();

    TT_FATAL(
        device->num_program_cache_entries() == 6,
        "There are {} entries",
        device->num_program_cache_entries());

    device->disable_and_clear_program_cache();

    TT_FATAL(device->num_program_cache_entries() == 0);

    TT_FATAL(tt::tt_metal::CloseDevice(device));

    return 0;
}
