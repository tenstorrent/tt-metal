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

}

struct Operation {

    Operation() {};
    Operation(const Operation&) = delete;
    Operation& operator=(const Operation&) = delete;
    virtual ~Operation() {}

    virtual std::vector<Shape> compute_output_shapes(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const = 0;
    virtual std::vector<Tensor> create_output_tensors(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const = 0;
    virtual Program create_program(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors, std::vector<Tensor> &output_tensors) const = 0;

    std::vector<Tensor> run(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {

        auto device = detail::get_device(input_tensors);
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
