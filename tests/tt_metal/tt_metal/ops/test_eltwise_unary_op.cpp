#include <cmath>

#include "constants.hpp"
#include "tensor/tensor.hpp"
#include "tensor/host_buffer.hpp"
#include "tt_metal/host_api.hpp"

#include "tt_numpy/functions.hpp"

#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_dnn/op_library/operation.hpp"
#include "tt_dnn/op_library/program_cache.hpp"

using tt::tt_metal::DataType;
using tt::tt_metal::Device;
using tt::tt_metal::Host;
using tt::tt_metal::Layout;
using tt::tt_metal::Tensor;

namespace detail {
float sqrt(float x) { return std::sqrt(x); }
float exp(float x) { return std::exp(x); }
float recip(float x) { return 1 / x; }
float gelu(float x) { return x * (0.5 * (1 + std::erf(x / std::sqrt(2)))); }
float relu(float x) { return std::max(0.0f, x); }
float sigmoid(float x) { return 1 / (1 + std::exp(-x)); }
float log(float x) { return std::log(x); }
float tanh(float x) { return std::tanh(x); }
}  // namespace detail

template <auto UnaryFunction>
Tensor host_function(const Tensor& input_tensor) {
    auto input_buffer_view = host_buffer::view_as<bfloat16>(input_tensor);

    auto output_buffer = host_buffer::create<bfloat16>(input_tensor.volume());
    auto output_view = host_buffer::view_as<bfloat16>(output_buffer);

    for (auto index = 0; index < output_view.size(); index++) {
        auto value = UnaryFunction(input_buffer_view[index].to_float());
        output_view[index] = bfloat16(value);
    }

    return Tensor(HostStorage{output_buffer}, input_tensor.shape(), input_tensor.dtype(), input_tensor.layout());
}

template <auto Operation>
Tensor device_function(const Tensor& input_tensor, Host* host, Device* device) {
    return Operation(input_tensor.to(device)).to(host);
}

template <auto HostFunction, auto DeviceFunction, typename... Args>
bool run_test(Host* host, Device* device, const std::array<uint32_t, 4>& shape, float low, float high, Args... args) {
    auto input_tensor = tt::numpy::random::uniform(bfloat16(low), bfloat16(high), shape).to(Layout::TILE);

    auto host_output = HostFunction(input_tensor);
    auto device_output = DeviceFunction(input_tensor, host, device);

    return tt::numpy::allclose<bfloat16>(host_output, device_output, args...);
}

void test_operation_infrastructure() {
    tt::log_info(tt::LogTest, "Running {}", __func__);
    using namespace tt::tt_metal;

    auto shape = std::array<uint32_t, 4>{1, 1, TILE_HEIGHT, TILE_WIDTH};
    auto input_tensor = tt::numpy::random::uniform(bfloat16(0), bfloat16(1), shape).to(Layout::TILE);

    auto op = operation::Operation(EltwiseUnary{UnaryOpType::SQRT});

    auto program_hash = op.compute_program_hash({input_tensor});
    TT_ASSERT(
        program_hash == "tt::tt_metal::EltwiseUnary{.op_type=SQRT,.param=std::nullopt}_[1, 1, 32, 32]_BFLOAT16_TILE_nullopt",
        fmt::format("Actual value is {}", program_hash)
    );

    auto profiler_info = op.create_profiler_info({input_tensor});
    TT_ASSERT(
        profiler_info.preferred_name.value() == "tt::tt_metal::EltwiseUnary{.op_type=SQRT,.param=std::nullopt}",
        fmt::format("Actual value is {}", profiler_info.preferred_name.value())
    );
    TT_ASSERT(
        profiler_info.parallelization_strategy.value() == "SINGLE_CORE",
        fmt::format("Actual value is {}", profiler_info.parallelization_strategy.value())
    );
}

void test_numerically() {
    tt::log_info(tt::LogTest, "Running {}", __func__);

    using tt::constants::TILE_HEIGHT;
    using tt::constants::TILE_WIDTH;

    int pci_express_slot = 0;
    auto device = tt::tt_metal::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);
    auto host = tt::tt_metal::GetHost();

    TT_ASSERT(tt::tt_metal::InitializeDevice(device));

    auto shape = std::array<uint32_t, 4>{1, 1, TILE_HEIGHT, TILE_WIDTH};
    {
        auto allclose = run_test<host_function<detail::sqrt>, device_function<tt::tt_metal::sqrt>>(
            host, device, shape, 0.0f, 1.0f, 1e-1f, 1e-5f);
        TT_ASSERT(allclose);
    }
    {
        auto allclose = run_test<host_function<detail::exp>, device_function<tt::tt_metal::exp>>(
            host, device, shape, -1.0f, 1.0f, 1e-1f, 1e-5f);
        TT_ASSERT(allclose);
    }
    {
        auto allclose = run_test<host_function<detail::recip>, device_function<tt::tt_metal::recip>>(
            host, device, shape, 1.0f, 10.0f, 1e-1f, 1e-5f);
        TT_ASSERT(allclose);
    }
    {
        auto allclose = run_test<host_function<detail::gelu>, device_function<tt::tt_metal::gelu>>(
            host, device, shape, 1.0f, 10.0f, 1e-1f, 1e-3f);
        TT_ASSERT(allclose);
    }
    {
        auto allclose = run_test<host_function<detail::relu>, device_function<tt::tt_metal::relu>>(
            host, device, shape, -1.0f, 1.0f, 1e-1f, 1e-5f);
        TT_ASSERT(allclose);
    }
    {
        auto allclose = run_test<host_function<detail::sigmoid>, device_function<tt::tt_metal::sigmoid>>(
            host, device, shape, -1.0f, 1.0f, 1e-1f, 1e-5f);
        TT_ASSERT(allclose);
    }
    {
        auto allclose = run_test<host_function<detail::log>, device_function<tt::tt_metal::log>>(
            host, device, shape, 0.0f, 1.0f, 1e-1f, 1e-2f);
        TT_ASSERT(allclose);
    }
    {
        auto allclose = run_test<host_function<detail::tanh>, device_function<tt::tt_metal::tanh>>(
            host, device, shape, -1.0f, 1.0f, 1e-1f, 1e-5f);
        TT_ASSERT(allclose);
    }

    TT_ASSERT(tt::tt_metal::CloseDevice(device));
}

void test_program_cache() {
    tt::log_info(tt::LogTest, "Running {}", __func__);

    using tt::constants::TILE_HEIGHT;
    using tt::constants::TILE_WIDTH;

    int pci_express_slot = 0;
    auto device = tt::tt_metal::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);
    auto host = tt::tt_metal::GetHost();

    TT_ASSERT(tt::tt_metal::InitializeDevice(device));

    auto run_tests = [&]() {
        // Program Cache Miss
        run_test<host_function<detail::sqrt>, device_function<tt::tt_metal::sqrt>>(
            host, device, {1, 1, TILE_HEIGHT, TILE_WIDTH}, 0.0f, 1.0f, 1e-1f, 1e-5f);

        // Program Cache Hit
        run_test<host_function<detail::sqrt>, device_function<tt::tt_metal::sqrt>>(
            host, device, {1, 1, TILE_HEIGHT, TILE_WIDTH}, 0.0f, 1.0f, 1e-1f, 1e-5f);

        // Program Cache Miss
        run_test<host_function<detail::sqrt>, device_function<tt::tt_metal::sqrt>>(
            host, device, {1, 1, 384, 4096}, 0.0f, 1.0f, 1e-1f, 1e-5f);

        // Program Cache Miss
        run_test<host_function<detail::exp>, device_function<tt::tt_metal::exp>>(
            host, device, {1, 1, TILE_HEIGHT, TILE_WIDTH}, 0.0f, 1.0f, 1e-1f, 1e-5f);

        // Program Cache Hit
        run_test<host_function<detail::sqrt>, device_function<tt::tt_metal::sqrt>>(
            host, device, {1, 1, TILE_HEIGHT, TILE_WIDTH}, 0.0f, 1.0f, 1e-1f, 1e-5f);

        // Allocate a tensor to show that the addresses aren't cached
        auto input_tensor =
            tt::numpy::random::uniform(bfloat16(0.0f), bfloat16(0.0f), {1, 1, 32, 32}).to(Layout::TILE).to(device);

        // Program Cache Hit
        run_test<host_function<detail::exp>, device_function<tt::tt_metal::exp>>(
            host, device, {1, 1, TILE_HEIGHT, TILE_WIDTH}, 0.0f, 1.0f, 1e-1f, 1e-5f);

        // Program Cache Miss
        run_test<host_function<detail::gelu>, device_function<tt::tt_metal::gelu>>(
            host, device, {1, 1, TILE_HEIGHT, TILE_WIDTH}, 1.0f, 10.0f, 1e-1f, 1e-3f);

        // Program Cache Hit
        run_test<host_function<detail::sqrt>, device_function<tt::tt_metal::sqrt>>(
            host, device, {1, 1, TILE_HEIGHT, TILE_WIDTH}, 0.0f, 1.0f, 1e-1f, 1e-5f);

        // Program Cache Hit
        run_test<host_function<detail::sqrt>, device_function<tt::tt_metal::sqrt>>(
            host, device, {1, 1, 384, 4096}, 0.0f, 1.0f, 1e-1f, 1e-5f);
    };

    tt::tt_metal::program_cache::enable();
    run_tests();

    TT_ASSERT(tt::tt_metal::CloseDevice(device));

    TT_ASSERT(tt::tt_metal::program_cache::num_entries() == 4);

    tt::tt_metal::program_cache::disable_and_clear();

    TT_ASSERT(tt::tt_metal::program_cache::num_entries() == 0);
}

int main(int argc, char** argv) {
    test_operation_infrastructure();
    test_numerically();
    test_program_cache();
    return 0;
}
