#pragma once

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"

namespace tt {

namespace tt_metal {

struct Pad {
    const std::array<uint32_t, 4> output_tensor_shape;
    const std::array<uint32_t, 4> input_tensor_start;
    float pad_value;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    operation::Hash compute_program_hash(const std::vector<Tensor> &input_tensors) const;
};

Tensor pad(const Tensor &input_tensor_a, const std::array<uint32_t, 4> &output_tensor_shape, const std::array<uint32_t, 4> &input_tensor_start, float pad_value);

}  // namespace tt_metal

}  // namespace tt
