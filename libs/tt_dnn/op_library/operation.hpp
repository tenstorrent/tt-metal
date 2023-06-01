#pragma once

#include <libs/tensor/tensor.hpp>
#include "tt_dnn/op_library/auto_pad.hpp"

namespace tt::tt_metal {

using Shape = std::array<uint32_t, 4>;

namespace detail {

static Device* get_device(const std::vector<Tensor> &input_tensors) {
    for (auto &input_tensor : input_tensors) {
        if (not input_tensor.on_host()) {
            return input_tensor.device();
        }
    }
    auto device = AutoPad::GetDefaultDevice();
    TT_ASSERT(device != nullptr, "Requires setting default device if no inputs to op are on device");
    return device;
}

static std::vector<Tensor> pad_input_tensors(const std::vector<Tensor> &input_tensors, Device* device) {
    std::vector<Tensor> padded_input_tensors;
    for (auto& tensor : input_tensors) {
        auto padded_shape = AutoPad::pad_to_tile_shape(tensor.shape());
        auto needs_to_be_formatted = not AutoPad::check_input_tensor_format(tensor, padded_shape);
        if (needs_to_be_formatted) {
            auto padded_tensor = AutoPad::format_input_tensor(tensor, device, padded_shape, 0);
            padded_input_tensors.push_back(padded_tensor);
        } else {
            padded_input_tensors.push_back(tensor);
        }
    }
    return padded_input_tensors;
}

static void slice_output_tensors(const Tensor& input_tensor, std::vector<Tensor> &padded_output_tensors, const std::vector<Shape> &output_shapes, Device* device) {
    std::vector<Tensor> sliced_output_tensors;
    for (auto index = 0; index < padded_output_tensors.size(); index++) {
        auto& output_tensor = padded_output_tensors.at(index);
        auto output_shape = output_shapes.at(index);
        if (output_shape != output_tensor.shape()) {
            AutoPad::format_output_tensor(input_tensor, output_tensor, output_shape, device);
        }
    }
}

}

struct Operation {

    virtual std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const = 0;
    virtual std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const = 0;
    virtual Program create_program(const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const = 0;

    std::vector<Tensor> run(const std::vector<Tensor> &input_tensors) const {

        auto device = detail::get_device(input_tensors);

        auto output_shapes = this->compute_output_shapes(input_tensors);

        auto padded_input_tensors = detail::pad_input_tensors(input_tensors, device);
        auto output_tensors = this->create_output_tensors(padded_input_tensors);
        auto program = this->create_program(padded_input_tensors, output_tensors);

        tt_metal::CompileProgram(device, program);
        tt_metal::ConfigureDeviceWithProgram(device, program);
        tt_metal::LaunchKernels(device, program);

        detail::slice_output_tensors(input_tensors.at(0), output_tensors, output_shapes, device);

        return output_tensors;
    }

};

}
