#pragma once

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/operation.hpp"

namespace tt {

namespace tt_metal {

struct ConcatOpParallelizationStrategy {
    enum Enum { SINGLE_CORE = 0, MULTIPLE_CORE = 1 };
};

ConcatOpParallelizationStrategy::Enum get_parallelization_strategy(const Tensor &a, const Tensor &b);

struct Concat2 {
    unsigned int dim = 3;                                              // default
    mutable tt::tt_metal::Layout layout = tt::tt_metal::Layout::TILE;  // default
    Concat2(unsigned int _dim = 3) : dim(_dim){};
    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;
    operation::Hash compute_program_hash(const std::vector<Tensor> &input_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
    tt::tt_metal::Shape get_output_shape(const std::vector<Tensor> &input_tensors) const;
};

// Ref: https://pytorch.org/docs/stable/generated/torch.cat.html#torch.cat
// Notes: Non-empty tensors provided must have the same shape, except in the cat dimension.
Tensor concat(std::vector<Tensor> &in_t, uint32_t dim = 3);

// common interface
// NCHW = [0,1,2,3]
Tensor concat(Tensor &input_tensor_a,Tensor &input_tensor_b, uint32_t dim = 3);

}  // namespace tt_metal

}  // namespace tt
