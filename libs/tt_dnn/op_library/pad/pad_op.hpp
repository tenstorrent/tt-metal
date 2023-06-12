#pragma once

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/operation.hpp"

namespace tt {

namespace tt_metal {

Program pad_rm(const Tensor &input_tensor_a, Tensor& output_tensor, const std::array<uint32_t, 4> &output_tensor_shape, const std::array<uint32_t, 4> &input_tensor_start, float pad_value);
Program pad_tile(const Tensor &input_tensor_a, Tensor& output_tensor, const std::array<uint32_t, 4> &output_tensor_shape, const std::array<uint32_t, 4> &input_tensor_start, float pad_value);

struct Pad : Operation {
    const std::array<uint32_t, 4> output_tensor_shape;
    const std::array<uint32_t, 4> input_tensor_start;
    float pad_value;
    Pad(const std::array<uint32_t, 4> &output_tensor_shape, const std::array<uint32_t, 4> &input_tensor_start, float pad_value)
        : output_tensor_shape(output_tensor_shape), input_tensor_start(input_tensor_start), pad_value(pad_value) {
    }

    Pad(const Pad&) = delete;
    Pad& operator=(const Pad&) = delete;
    ~Pad() {}

    void validate(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const override;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const override;
    std::vector<Tensor> create_output_tensors(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const override;
    Program create_program(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors, std::vector<Tensor> &output_tensors) const override;
};

Tensor pad(const Tensor &input_tensor_a, const std::array<uint32_t, 4> &output_tensor_shape, const std::array<uint32_t, 4> &input_tensor_start, float pad_value);

}  // namespace tt_metal

}  // namespace tt
