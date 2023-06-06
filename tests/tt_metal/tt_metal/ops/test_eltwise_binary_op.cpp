#include "tt_metal/host_api.hpp"

#include "tt_dnn/op_library/eltwise_binary/eltwise_binary_op.hpp"
#include "tt_numpy/functions.hpp"

#include "tensor/tensor.hpp"
#include "constants.hpp"

using tt::tt_metal::Host;
using tt::tt_metal::Device;
using tt::tt_metal::Tensor;
using tt::tt_metal::DataType;
using tt::tt_metal::Layout;

template<typename BinaryFunction>
Tensor host_function(const Tensor& input_tensor_a, const Tensor& input_tensor_b) {
    auto input_tensor_a_data = *reinterpret_cast<std::vector<bfloat16>*>(input_tensor_a.data_ptr());
    auto input_tensor_b_data = *reinterpret_cast<std::vector<bfloat16>*>(input_tensor_b.data_ptr());
    auto output_tensor_data = std::vector<bfloat16>(input_tensor_a_data.size(), 0.0f);

    for (auto index = 0; index < output_tensor_data.size(); index++) {
        auto value = BinaryFunction{}(input_tensor_a_data[index].to_float(), input_tensor_b_data[index].to_float());
        output_tensor_data[index] = bfloat16(value);
    }
    return Tensor(output_tensor_data, input_tensor_a.shape(), input_tensor_a.dtype(), input_tensor_a.layout());
}

template<auto Operation>
Tensor device_function(const Tensor& input_tensor_a, const Tensor& input_tensor_b, Host* host, Device* device) {
    return Operation(input_tensor_a.to(device), input_tensor_b.to(device)).to(host);
}

template<auto HostFunction, auto DeviceFunction, typename ... Args>
bool run_test(Host* host, Device* device, Args ...  args) {
    std::array<uint32_t, 4> shape = {1, 1, tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH};
    auto input_tensor_a = tt::numpy::random::random(shape, DataType::BFLOAT16).to(Layout::TILE);
    auto input_tensor_b = tt::numpy::random::random(shape, DataType::BFLOAT16).to(Layout::TILE);

    auto host_output = HostFunction(input_tensor_a, input_tensor_b);
    auto device_output = DeviceFunction(input_tensor_a, input_tensor_b, host, device);

    return tt::numpy::allclose<bfloat16>(host_output, device_output, args...);
}

int main(int argc, char **argv) {
    bool pass = true;

    try {
        int pci_express_slot = 0;
        auto device = tt::tt_metal::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);
        auto host = tt::tt_metal::GetHost();

        pass &= tt::tt_metal::InitializeDevice(device);

        pass &= run_test<host_function<std::plus<float>>, device_function<tt::tt_metal::add>>(host, device);
        pass &= run_test<host_function<std::minus<float>>, device_function<tt::tt_metal::sub>>(host, device);
        pass &= run_test<host_function<std::multiplies<float>>, device_function<tt::tt_metal::mul>>(host, device, 1e-2f, 1e-3f);

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
