#include "tt_metal/host_api.hpp"

#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_numpy/functions.hpp"

#include "tensor/tensor.hpp"
#include "constants.hpp"

#include <cmath>

using tt::tt_metal::Host;
using tt::tt_metal::Device;
using tt::tt_metal::Tensor;
using tt::tt_metal::DataType;
using tt::tt_metal::Layout;

namespace detail {
    float sqrt(float x) { return std::sqrt(x); }
    float exp(float x) { return std::exp(x); }
    float recip(float x) { return 1 / x; }
    float gelu(float x) { return x * (0.5 * (1 + std::erf(x / std::sqrt(2)))); }
    float relu(float x) { return std::max(0.0f, x); }
    float sigmoid(float x) { return 1 / (1 + std::exp(-x)); }
    float log(float x) { return std::log(x); }
    float tanh(float x) { return std::tanh(x); }
}

template<auto UnaryFunction>
Tensor host_function(const Tensor& input_tensor) {
    auto input_tensor_data = *reinterpret_cast<std::vector<bfloat16>*>(input_tensor.data_ptr());
    auto output_tensor_data = std::vector<bfloat16>(input_tensor_data.size(), 0.0f);

    for (auto index = 0; index < output_tensor_data.size(); index++) {
        auto value = UnaryFunction(input_tensor_data[index].to_float());
        output_tensor_data[index] = bfloat16(value);
    }

    return Tensor(output_tensor_data, input_tensor.shape(), input_tensor.dtype(), input_tensor.layout());
}

template<auto Operation>
Tensor device_function(const Tensor& input_tensor, Host* host, Device* device) {
    return Operation(input_tensor.to(device)).to(host);
}

template<auto HostFunction, auto DeviceFunction, typename ... Args>
bool run_test(Host* host, Device* device, float low, float high, Args ...  args) {
    tt::numpy::random::seed(1234);
    std::array<uint32_t, 4> shape = {1, 1, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH};
    auto input_tensor = tt::numpy::random::uniform(bfloat16(low), bfloat16(high), shape).to(Layout::TILE);

    auto host_output = HostFunction(input_tensor);
    auto device_output = DeviceFunction(input_tensor, host, device);

    return tt::numpy::allclose<bfloat16>(host_output, device_output, args...);
}

int main(int argc, char **argv) {
    bool pass = true;

    try {
        int pci_express_slot = 0;
        auto device = tt::tt_metal::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);
        auto host = tt::tt_metal::GetHost();

        pass &= tt::tt_metal::InitializeDevice(device);

        pass &= run_test<host_function<detail::sqrt>, device_function<tt::tt_metal::sqrt>>(host, device, 0.0f, 1.0f, 1e-1f, 1e-5f);
        pass &= run_test<host_function<detail::exp>, device_function<tt::tt_metal::exp>>(host, device, -1.0f, 1.0f, 1e-1f, 1e-5f);
        pass &= run_test<host_function<detail::recip>, device_function<tt::tt_metal::recip>>(host, device, 1.0f, 10.0f, 1e-1f, 1e-5f);
        pass &= run_test<host_function<detail::gelu>, device_function<tt::tt_metal::gelu>>(host, device, 1.0f, 10.0f, 1e-1f, 1e-3f);
        pass &= run_test<host_function<detail::relu>, device_function<tt::tt_metal::relu>>(host, device, -1.0f, 1.0f, 1e-1f, 1e-5f);
        pass &= run_test<host_function<detail::sigmoid>, device_function<tt::tt_metal::sigmoid>>(host, device, -1.0f, 1.0f, 1e-1f, 1e-5f);
        pass &= run_test<host_function<detail::log>, device_function<tt::tt_metal::log>>(host, device, 0.0f, 1.0f, 1e-1f, 1e-2f);
        pass &= run_test<host_function<detail::tanh>, device_function<tt::tt_metal::tanh>>(host, device, -1.0f, 1.0f, 1e-1f, 1e-5f);

        pass &= tt::tt_metal::CloseDevice(device);

    } catch (const std::exception &e) {
        pass = false;
        // Capture the exception error message
        tt::log_error(tt::LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        tt::log_error(tt::LogTest, "System error message: {}", std::strerror(errno));
    }

    if (pass) {
        tt::log_info(tt::LogTest, "Test Passed");
    } else {
        tt::log_fatal(tt::LogTest, "Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
