#include "tt_dnn/op_library/operation_cache.hpp"
#include "tt_dnn/op_library/reduce/reduce_op.hpp"
#include "tt_numpy/functions.hpp"

#include "tensor/tensor.hpp"
#include "constants.hpp"

using tt::tt_metal::Host;
using tt::tt_metal::Device;
using tt::tt_metal::Tensor;
using tt::tt_metal::DataType;
using tt::tt_metal::Layout;

using tt::tt_metal::ReduceOpMath;
using tt::tt_metal::ReduceOpDim;

template<ReduceOpMath::Enum op_math, ReduceOpDim::Enum op_dim>
Tensor device_function(const Tensor& input_tensor, Host* host, Device* device) {
    return reduce(input_tensor.to(device), op_math, op_dim).to(host);
}

template<auto DeviceFunction, typename ... Args>
void run_test(Host* host, Device* device, Args ...  args) {
    std::array<uint32_t, 4> shape = {1, 1, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH};
    auto input_tensor = tt::numpy::random::random(shape, DataType::BFLOAT16).to(Layout::TILE);
    auto device_output = DeviceFunction(input_tensor, host, device);
    // What should be validated? allclose against host implementation?
}

int main () {
    using tt::constants::TILE_HEIGHT;
    using tt::constants::TILE_WIDTH;

    int pci_express_slot = 0;
    auto device = tt::tt_metal::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);
    auto host = tt::tt_metal::GetHost();

    TT_ASSERT(tt::tt_metal::InitializeDevice(device));

    {
        run_test<device_function<ReduceOpMath::SUM, ReduceOpDim::H>>(host, device);
    }

    {
        run_test<device_function<ReduceOpMath::SUM, ReduceOpDim::W>>(host, device);
    }

    {
        run_test<device_function<ReduceOpMath::SUM, ReduceOpDim::HW>>(host, device);
    }

    {
        run_test<device_function<ReduceOpMath::MAX, ReduceOpDim::H>>(host, device);
    }

    {
        run_test<device_function<ReduceOpMath::MAX, ReduceOpDim::W>>(host, device);
    }

    {
        run_test<device_function<ReduceOpMath::MAX, ReduceOpDim::HW>>(host, device);
    }

    tt::tt_metal::operation_cache::enable();

    auto run_reduce_ops = [&] {
        {
            run_test<device_function<ReduceOpMath::SUM, ReduceOpDim::H>>(host, device);
        }

        {
            run_test<device_function<ReduceOpMath::SUM, ReduceOpDim::W>>(host, device);
        }

        {
            run_test<device_function<ReduceOpMath::SUM, ReduceOpDim::HW>>(host, device);
        }

        {
            run_test<device_function<ReduceOpMath::MAX, ReduceOpDim::H>>(host, device);
        }

        {
            run_test<device_function<ReduceOpMath::MAX, ReduceOpDim::W>>(host, device);
        }

        {
            run_test<device_function<ReduceOpMath::MAX, ReduceOpDim::HW>>(host, device);
        }
    };

    run_reduce_ops();
    run_reduce_ops();

    // Allocate a tensor to show that the addresses aren't cached
    auto some_tensor = tt::numpy::random::uniform(bfloat16(0.0f), bfloat16(0.0f), {1, 1, 32, 32}).to(Layout::TILE).to(device);

    run_reduce_ops();

    TT_ASSERT(tt::tt_metal::operation_cache::num_cached_programs() == 6);

    tt::tt_metal::operation_cache::disable_and_clear();

    TT_ASSERT(tt::tt_metal::operation_cache::num_cached_programs() == 0);

    TT_ASSERT(tt::tt_metal::CloseDevice(device));

    return 0;
}
