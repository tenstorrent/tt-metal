#pragma once

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"

namespace tt {

namespace tt_metal {


struct Unpad {
    const std::array<uint32_t, 4> output_tensor_start;
    const std::array<uint32_t, 4> output_tensor_end;
    const std::array<uint32_t, 4> output_tensor_shape;

    Unpad(const std::array<uint32_t, 4> &output_tensor_start, const std::array<uint32_t, 4> &output_tensor_end)
        : output_tensor_start(output_tensor_start), output_tensor_end(output_tensor_end),
        output_tensor_shape{
        output_tensor_end[0] - output_tensor_start[0] + 1,
        output_tensor_end[1] - output_tensor_start[1] + 1,
        output_tensor_end[2] - output_tensor_start[2] + 1,
        output_tensor_end[3] - output_tensor_start[3] + 1,
        } {
    }

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    operation::Hash compute_program_hash(const std::vector<Tensor> &input_tensors) const;
};

Tensor unpad(const Tensor &input_tensor_a, const std::array<uint32_t, 4> &output_tensor_start, const std::array<uint32_t, 4> &output_tensor_end);

}  // namespace tt_metal

}  // namespace tt
