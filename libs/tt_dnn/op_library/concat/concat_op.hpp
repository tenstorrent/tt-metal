#pragma once

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/operation.hpp"

namespace tt {

namespace tt_metal {

enum class ConcatOpParallelizationStrategy {
    SINGLE_CORE = 0, MULTIPLE_CORE = 1
};

struct Concat2 {
    uint32_t dim;
    const MemoryConfig output_mem_config;
    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;
    tt::stl::reflection::Attributes attributes() const;
    ConcatOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors);
};

// Ref: https://pytorch.org/docs/stable/generated/torch.cat.html#torch.cat
// Notes: Non-empty tensors provided must have the same shape, except in the cat dimension.
Tensor concat(std::vector<Tensor> &in_t, uint32_t dim = 3, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

// common interface
// NCHW = [0,1,2,3]
Tensor concat(Tensor &input_tensor_a,Tensor &input_tensor_b, uint32_t dim = 3, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace tt_metal

}  // namespace tt
