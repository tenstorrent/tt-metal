#pragma once

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"

namespace tt {

namespace tt_metal {

struct Reshape {

    int N, C, H, W;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const;
    operation::Hash compute_program_hash(const std::vector<Tensor> &input_tensors) const;
};

// Tensor &a cannot be const, since in some cases we modify in place
Tensor reshape (Tensor &a, int N, int C, int H, int W);

}  // namespace tt_metal

}  // namespace tt
