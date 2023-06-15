#include "tt_metal/host_api.hpp"

#include "tt_dnn/op_library/program_cache.hpp"
#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_numpy/functions.hpp"

#include "tensor/tensor.hpp"
#include "constants.hpp"

#include <cmath>

#include <chrono>
using namespace std::chrono;

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
void run_test(Host* host, Device* device, const std::array<uint32_t, 4>& shape, float low, float high, Args ...  args) {
    auto input_tensor = tt::numpy::random::uniform(bfloat16(low), bfloat16(high), shape).to(Layout::TILE);

    auto host_output = HostFunction(input_tensor);

    auto start = high_resolution_clock::now();
    auto device_output = DeviceFunction(input_tensor, host, device);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    tt::log_info(tt::LogTest, "Operation executed in {} microseconds\n", duration.count());

    TT_ASSERT(tt::numpy::allclose<bfloat16>(host_output, device_output, args...));
}

int main(int argc, char **argv) {
    using tt::constants::TILE_HEIGHT;
    using tt::constants::TILE_WIDTH;

    int pci_express_slot = 0;
    auto device = tt::tt_metal::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);
    auto host = tt::tt_metal::GetHost();

    TT_ASSERT(tt::tt_metal::InitializeDevice(device));

    auto run_tests = [&]() {
        // Program Cache Miss
        run_test<host_function<detail::sqrt>, device_function<tt::tt_metal::sqrt>>(host, device, {1, 1, TILE_HEIGHT, TILE_WIDTH}, 0.0f, 1.0f, 1e-1f, 1e-5f);

        // Program Cache Hit
        run_test<host_function<detail::sqrt>, device_function<tt::tt_metal::sqrt>>(host, device, {1, 1, TILE_HEIGHT, TILE_WIDTH}, 0.0f, 1.0f, 1e-1f, 1e-5f);

        // Program Cache Miss
        run_test<host_function<detail::sqrt>, device_function<tt::tt_metal::sqrt>>(host, device, {1, 1, 384, 4096}, 0.0f, 1.0f, 1e-1f, 1e-5f);

        // Program Cache Miss
        run_test<host_function<detail::exp>,  device_function<tt::tt_metal::exp>>(host, device, {1, 1, TILE_HEIGHT, TILE_WIDTH}, 0.0f, 1.0f, 1e-1f, 1e-5f);

        // Program Cache Hit
        run_test<host_function<detail::sqrt>, device_function<tt::tt_metal::sqrt>>(host, device, {1, 1, TILE_HEIGHT, TILE_WIDTH}, 0.0f, 1.0f, 1e-1f, 1e-5f);

        // Allocate a tensor to show that the addresses aren't cached
        auto input_tensor = tt::numpy::random::uniform(bfloat16(0.0f), bfloat16(0.0f), {1, 1, 32, 32}).to(Layout::TILE).to(device);

        // Program Cache Hit
        run_test<host_function<detail::exp>,  device_function<tt::tt_metal::exp>>(host, device, {1, 1, TILE_HEIGHT, TILE_WIDTH}, 0.0f, 1.0f, 1e-1f, 1e-5f);

        // Program Cache Miss
        run_test<host_function<detail::gelu>, device_function<tt::tt_metal::gelu>>(host, device, {1, 1, TILE_HEIGHT, TILE_WIDTH}, 1.0f, 10.0f, 1e-1f, 1e-3f);

        // Program Cache Hit
        run_test<host_function<detail::sqrt>, device_function<tt::tt_metal::sqrt>>(host, device, {1, 1, TILE_HEIGHT, TILE_WIDTH}, 0.0f, 1.0f, 1e-1f, 1e-5f);

        // Program Cache Hit
        run_test<host_function<detail::sqrt>, device_function<tt::tt_metal::sqrt>>(host, device, {1, 1, 384, 4096}, 0.0f, 1.0f, 1e-1f, 1e-5f);
    };

    tt::tt_metal::program_cache::enable();
    run_tests();

    TT_ASSERT(tt::tt_metal::CloseDevice(device));

    TT_ASSERT(tt::tt_metal::program_cache::num_cached_programs() == 4);

    return 0;
}
