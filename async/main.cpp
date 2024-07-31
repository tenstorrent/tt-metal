#include "infra.hpp"
#include "operations.hpp"

Tensor create_device_tensor(const std::vector<std::size_t>& shape, Dtype dtype, std::shared_ptr<Device> device) {
    return {shape, dtype, DeviceStorage{device}};
}

Tensor create_multi_device_tensor(
    const std::vector<std::size_t>& shape, Dtype dtype, std::vector<std::shared_ptr<Device>> devices) {
    return {shape, dtype, MultiDeviceStorage{devices}};
}

auto launch_function(auto&& description, auto&& lambda) { std::cout << description << std::endl; }

void test_regular_launch(auto& devices) {
    std::cout << std::endl << std::endl;
    std::cout << "1 Regular launch" << std::endl;

    auto device = devices.at(0);

    Tensor hidden_states = create_device_tensor({2, 8}, Dtype::bfloat16, device);
    Tensor weight_0 = create_device_tensor({8, 4}, Dtype::bfloat16, device);
    Tensor bias_0 = create_device_tensor({2, 4}, Dtype::bfloat16, device);

    Tensor multi_device_hidden_states = create_multi_device_tensor({2, 8}, Dtype::bfloat16, devices);
    Tensor multi_device_weight_0 = create_multi_device_tensor({8, 4}, Dtype::bfloat16, devices);
    Tensor multi_device_bias_0 = create_multi_device_tensor({2, 4}, Dtype::bfloat16, devices);

    {
        std::cout << std::endl << std::endl;
        std::cout << "1.1 Primitive operations (single-device tensors)" << std::endl;

        Tensor output = matmul(hidden_states, weight_0);
        output = add(output, bias_0);
        std::cout << "Shape after add: " << output << std::endl;
    }
    {
        std::cout << std::endl << std::endl;
        std::cout << "1.2 Primitive operations  (multi-device tensors)" << std::endl;

        Tensor output = matmul(multi_device_hidden_states, multi_device_weight_0);
        output = add(output, multi_device_bias_0);
        std::cout << "Shape after add: " << output << std::endl;
    }

    {
        std::cout << std::endl << std::endl;
        std::cout << "1.3 Composite operations (single-device tensors)" << std::endl;

        auto output = linear_with_relu(hidden_states, weight_0, bias_0);
        std::cout << "Shape after linear_with_relu: " << output << std::endl;
    }

    {
        std::cout << std::endl << std::endl;
        std::cout << "1.4 Composite operations (multi-device tensors)" << std::endl;

        auto output = linear_with_relu(multi_device_hidden_states, multi_device_weight_0, multi_device_bias_0);
        std::cout << "Shape after linear_with_relu: " << output << std::endl;
    }

    {
        std::cout << std::endl << std::endl;
        std::cout << "1.5 Composite operations (single-device tensors)" << std::endl;

        auto&& [matmul_output, matmul_plus_bias_output] = linear_with_intermediate(hidden_states, weight_0, bias_0);
        std::cout << "Shape after linear_with_intermedate: " << matmul_output << ", " << matmul_plus_bias_output
                  << std::endl;
    }
}

void test_async_launch(auto& devices) {
    std::cout << std::endl << std::endl;
    std::cout << "2 Async launch" << std::endl;

    auto device = devices.at(0);

    Tensor hidden_states = create_device_tensor({2, 8}, Dtype::bfloat16, device);
    Tensor weight_0 = create_device_tensor({8, 4}, Dtype::bfloat16, device);
    Tensor bias_0 = create_device_tensor({2, 4}, Dtype::bfloat16, device);

    Tensor multi_device_hidden_states = create_multi_device_tensor({2, 8}, Dtype::bfloat16, devices);
    Tensor multi_device_weight_0 = create_multi_device_tensor({8, 4}, Dtype::bfloat16, devices);
    Tensor multi_device_bias_0 = create_multi_device_tensor({2, 4}, Dtype::bfloat16, devices);

    // Create async tensors and use the same APIs as before

    {
        std::cout << std::endl << std::endl;
        std::cout << "2.1 Launch primitive operations in async mode" << std::endl;

        auto async_output = matmul(hidden_states.to_async(), weight_0);
        async_output = add(async_output, bias_0);
        std::cout << "Finished 2.1" << std::endl << std::endl;

        auto output = async_output.get();
        std::cout << "Shape after async add: " << output << std::endl;
    }

    {
        std::cout << std::endl << std::endl;
        std::cout << "2.2 Launch primitive operations in async mode (multi-device tensors)" << std::endl;

        auto async_output = matmul(multi_device_hidden_states.to_async(), multi_device_weight_0);
        async_output = add(async_output, multi_device_bias_0);
        std::cout << "Finished 2.2" << std::endl << std::endl;

        auto output = async_output.get();
        std::cout << "Shape after async add: " << output << std::endl;
    }

    {
        std::cout << std::endl << std::endl;
        std::cout << "2.3 Launch composite operations in async mode" << std::endl;

        auto async_output = linear_with_relu(hidden_states.to_async(), weight_0, bias_0);
        std::cout << "Finished 2.2" << std::endl << std::endl;

        auto output = async_output.get();
        std::cout << "Shape after async linear_with_relu: " << output << std::endl;
    }

    {
        std::cout << std::endl << std::endl;
        std::cout << "2.4 Launch composite operations in async mode (multi-device tensors)" << std::endl;

        auto async_output =
            linear_with_relu(multi_device_hidden_states.to_async(), multi_device_weight_0, multi_device_bias_0);
        std::cout << "Finished 2.4" << std::endl << std::endl;

        auto output = async_output.get();
        std::cout << "Shape after async linear_with_relu: " << output << std::endl;
    }

    {
        std::cout << std::endl << std::endl;
        std::cout << "2.5 Launch operations with more complicated outputs in async mode (single-device tensors)"
                  << std::endl;

        auto linear_with_intermediate_output = linear_with_intermediate(hidden_states.to_async(), weight_0, bias_0);
        auto relu_output = relu(get_from_async<0>(linear_with_intermediate_output));
        auto slice = get_from_async(split(relu_output), 1);
        std::cout << "Finished 2.5" << std::endl << std::endl;

        auto&& output = slice.get();
        std::cout << "Shape after async slice: " << output << std::endl;
    }

    {
        std::cout << std::endl << std::endl;
        std::cout << "2.6 Launch operations with more complicated outputs in async mode (multi-device tensors)"
                  << std::endl;

        auto linear_with_intermediate_output =
            linear_with_intermediate(multi_device_hidden_states.to_async(), multi_device_weight_0, multi_device_bias_0);
        auto relu_output = relu(get_from_async<0>(linear_with_intermediate_output));
        auto slice = get_from_async(split(relu_output), 1);
        std::cout << "Finished 2.6" << std::endl << std::endl;

        auto&& output = slice.get();
        std::cout << "Shape after async slice: " << output << std::endl;
    }
}

int main() {
    auto devices = std::vector{
        std::make_shared<Device>(), std::make_shared<Device>(), std::make_shared<Device>(), std::make_shared<Device>()};
    test_regular_launch(devices);
    test_async_launch(devices);

    return 0;
}
