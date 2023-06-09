#pragma once

#include <libs/tensor/tensor.hpp>
#include "tt_dnn/op_library/auto_pad.hpp"
namespace tt::tt_metal {

using Shape = std::array<uint32_t, 4>;

namespace detail {

static Device* get_device(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) {
    for (auto &input_tensor : input_tensors) {
        if (not input_tensor.get().on_host()) {
            return input_tensor.get().device();
        }
    }
    auto device = AutoPad::GetDefaultDevice();
    TT_ASSERT(device != nullptr, "Requires setting default device if no inputs to op are on device");
    return device;
}

template<typename Operation>
static std::vector<Tensor> generic_create_output_tensors(const Operation& op, const std::vector<std::reference_wrapper<const Tensor>> &input_tensors, Layout output_layout = tt::tt_metal::Layout::TILE) {
    const auto& input_tensor = input_tensors.at(0).get();
    std::vector<Tensor> output_tensors;
    for (const auto& output_shape : op.compute_output_shapes(input_tensors)) {
        output_tensors.emplace_back(tt_metal::Tensor(output_shape, input_tensor.dtype(), output_layout, input_tensor.device()));
    }
    return output_tensors;
}

template<typename Operation>
static Tensor run_without_autopad(const Operation &op, const Tensor &input_tensor) {
    return std::move(op.run({std::cref(input_tensor)}).at(0));
}

template<typename Operation>
static Tensor run_with_autopad(const Operation &op, const Tensor &input_tensor) {
    Device* device;
    if (input_tensor.on_host()) {
        device = AutoPad::GetDefaultDevice();
        TT_ASSERT(device != nullptr, "Requires setting default device if no inputs to op are on device");
    } else {
        device = input_tensor.device();
    }

    auto padded_input_shape = AutoPad::pad_to_tile_shape(input_tensor.shape());
    auto output_shape = op.compute_output_shapes({std::cref(input_tensor)}).at(0);
    if (AutoPad::check_input_tensor_format(input_tensor, padded_input_shape)) {
        return std::move(op.run({std::cref(input_tensor)}).at(0));
    } else {
        const auto padded_tensor = AutoPad::format_input_tensor(input_tensor, device, padded_input_shape, 0);
        auto output = std::move(op.run({std::cref(padded_tensor)}).at(0));
        AutoPad::format_output_tensor(input_tensor, output, output_shape, device);
        return output;
    }
}

template<typename Operation>
static Tensor run_with_autopad(const Operation &op, const Tensor &input_tensor_a, const Tensor &input_tensor_b) {
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
        return std::move(op.run({std::cref(input_tensor_a), std::cref(input_tensor_b)}).at(0));
    } else if (no_pad_a) {
        const auto padded_input_tensor_b = AutoPad::format_input_tensor(input_tensor_b, device, padded_input_shape_b, 0);
        auto output = std::move(op.run({std::cref(input_tensor_a), std::cref(padded_input_tensor_b)}).at(0));
        AutoPad::format_output_tensor(input_tensor_a, output, output_shape, device);
        return output;
    } else if (no_pad_b) {
        const auto padded_input_tensor_a = AutoPad::format_input_tensor(input_tensor_a, device, padded_input_shape_a, 0);
        auto output = std::move(op.run({std::cref(padded_input_tensor_a), std::cref(input_tensor_b)}).at(0));
        AutoPad::format_output_tensor(input_tensor_a, output, output_shape, device);
        return output;
    } else {
        const auto padded_input_tensor_a = AutoPad::format_input_tensor(input_tensor_a, device, padded_input_shape_a, 0);
        const auto padded_input_tensor_b = AutoPad::format_input_tensor(input_tensor_b, device, padded_input_shape_b, 0);
        auto output = std::move(op.run({std::cref(padded_input_tensor_a), std::cref(padded_input_tensor_b)}).at(0));
        AutoPad::format_output_tensor(input_tensor_a, output, output_shape, device);
        return output;
    }
}

}

struct Operation {

    Operation() {};
    Operation(const Operation&) = delete;
    Operation& operator=(const Operation&) = delete;
    virtual ~Operation() {}

    virtual void validate(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const = 0;
    virtual std::vector<Shape> compute_output_shapes(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const = 0;
    virtual std::vector<Tensor> create_output_tensors(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const = 0;
    virtual Program create_program(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors, std::vector<Tensor> &output_tensors) const = 0;

    std::vector<Tensor> run(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {

        auto device = detail::get_device(input_tensors);
        this->validate(input_tensors);
        auto output_shapes = this->compute_output_shapes(input_tensors);
        auto output_tensors = this->create_output_tensors(input_tensors);
        auto program = this->create_program(input_tensors, output_tensors);

        tt_metal::CompileProgram(device, program);
        tt_metal::ConfigureDeviceWithProgram(device, program);
        tt_metal::LaunchKernels(device, program);

        return output_tensors;
    }

};

}
