#pragma once

#include <libs/tensor/tensor.hpp>
#include "tt_dnn/op_library/auto_pad.hpp"
#include "tt_dnn/op_library/operation.hpp"

#include <optional>

namespace tt::tt_metal {

namespace operation {

std::vector<Tensor> run(
    const Operation& op,
    const std::vector<std::reference_wrapper<const Tensor>>& input_tensors,
    const std::vector<std::optional<std::reference_wrapper<const Tensor>>>& optional_input_tensors = {}
);

std::vector<Tensor> generic_create_output_tensors(
    const Operation& op,
    const std::vector<std::reference_wrapper<const Tensor>> &input_tensors,
    Layout output_layout = tt::tt_metal::Layout::TILE,
    const MemoryConfig &output_mem_config = MemoryConfig{.interleaved = true}
);


template<typename ConcreteOperation>
static Tensor run_without_autopad(ConcreteOperation&& concrete_op, const Tensor &input_tensor, float pad_value = 0) {
    const Operation op = Operation(concrete_op);

    Device* device;
    if (input_tensor.on_host()) {
        device = AutoPad::GetDefaultDevice();
        TT_ASSERT(device != nullptr, "Requires setting default device if no inputs to op are on device");
    } else {
        device = input_tensor.device();
    }

    if (not input_tensor.on_host()) {
        return std::move(run(op, {std::cref(input_tensor)}).at(0));
    } else {
        auto input_tensor_on_dev = input_tensor.to(device);
        return std::move(run(op, {std::cref(input_tensor_on_dev)}).at(0));
    }
}

template<typename ConcreteOperation>
static Tensor run_with_autopad(ConcreteOperation&& concrete_op, const Tensor &input_tensor, float pad_value = 0, bool pad_c=false) {
    const Operation op = Operation(concrete_op);

    Device* device;
    if (input_tensor.on_host()) {
        device = AutoPad::GetDefaultDevice();
        TT_ASSERT(device != nullptr, "Requires setting default device if no inputs to op are on device");
    } else {
        device = input_tensor.device();
    }

    auto padded_input_shape = AutoPad::pad_to_tile_shape(input_tensor.shape(), pad_c);
    auto output_shape = op.compute_output_shapes({std::cref(input_tensor)}).at(0);
    if (AutoPad::check_input_tensor_format(input_tensor, padded_input_shape)) {
        return std::move(run(op, {std::cref(input_tensor)}).at(0));
    } else {
        const auto padded_tensor = AutoPad::format_input_tensor(input_tensor, device, padded_input_shape, pad_value);
        auto output_tensor = std::move(run(op, {std::cref(padded_tensor)}).at(0));
        AutoPad::format_output_tensor(input_tensor, output_tensor, output_shape, device);
        return output_tensor;
    }
}


template<typename ConcreteOperation>
static Tensor run_with_autopad(ConcreteOperation&& concrete_op, const Tensor &input_tensor_a, const Tensor &input_tensor_b, float pad_value = 0) {
    const Operation op = Operation(concrete_op);

    Device* device;
    if (input_tensor_a.on_host() && input_tensor_b.on_host()) {
        device = AutoPad::GetDefaultDevice();
        TT_ASSERT(device != nullptr, "Requires setting default device if no inputs to op are on device");
    } else if (!input_tensor_a.on_host()){
        device = input_tensor_a.device();
    } else {
        device = input_tensor_b.device();
    }

    auto padded_input_shape_a = AutoPad::pad_to_tile_shape(input_tensor_a.shape());
    auto padded_input_shape_b = AutoPad::pad_to_tile_shape(input_tensor_b.shape());
    auto output_shape = op.compute_output_shapes({std::cref(input_tensor_a), std::cref(input_tensor_b)}).at(0);

    auto no_pad_a = AutoPad::check_input_tensor_format(input_tensor_a, padded_input_shape_a);
    auto no_pad_b = AutoPad::check_input_tensor_format(input_tensor_b, padded_input_shape_b);
    if (no_pad_a && no_pad_b) {
        return std::move(run(op, {std::cref(input_tensor_a), std::cref(input_tensor_b)}).at(0));
    } else if (no_pad_a) {
        const auto padded_input_tensor_b = AutoPad::format_input_tensor(input_tensor_b, device, padded_input_shape_b, pad_value);
        auto output_tensor = std::move(run(op, {std::cref(input_tensor_a), std::cref(padded_input_tensor_b)}).at(0));
        AutoPad::format_output_tensor(input_tensor_a, output_tensor, output_shape, device);
        return output_tensor;
    } else if (no_pad_b) {
        const auto padded_input_tensor_a = AutoPad::format_input_tensor(input_tensor_a, device, padded_input_shape_a, pad_value);
        auto output_tensor = std::move(run(op, {std::cref(padded_input_tensor_a), std::cref(input_tensor_b)}).at(0));
        AutoPad::format_output_tensor(input_tensor_a, output_tensor, output_shape, device);
        return output_tensor;
    } else {
        const auto padded_input_tensor_a = AutoPad::format_input_tensor(input_tensor_a, device, padded_input_shape_a, pad_value);
        const auto padded_input_tensor_b = AutoPad::format_input_tensor(input_tensor_b, device, padded_input_shape_b, pad_value);
        auto output_tensor = std::move(run(op, {std::cref(padded_input_tensor_a), std::cref(padded_input_tensor_b)}).at(0));
        AutoPad::format_output_tensor(input_tensor_a, output_tensor, output_shape, device);
        return output_tensor;
    }
}

}

}
