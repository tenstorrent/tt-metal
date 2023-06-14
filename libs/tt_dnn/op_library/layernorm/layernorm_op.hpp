#pragma once

#include "libs/tensor/tensor.hpp"

#include "tt_dnn/op_library/operation.hpp"

namespace tt { namespace tt_metal {

Tensor layernorm(const Tensor &a, float eps, bool out_dram);
Tensor layernorm_gamma(const Tensor &a, float eps, const Tensor& gamma, bool out_dram);
Tensor layernorm_gamma_beta(const Tensor &a, float eps, const Tensor& gamma, const Tensor& beta, bool out_dram);

// computes layernorm(a+b)*gamma+beta
Tensor add_layernorm_gamma_beta(const Tensor& a, const Tensor &b, float eps, const Tensor& gamma, const Tensor& beta, bool out_dram);

struct ResidualLayerNorm {
    float eps;
    bool out_dram;

    void validate(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const;
    Program create_program(
        const std::vector<std::reference_wrapper<const Tensor>>& input_tensors,
        const std::vector<std::optional<std::reference_wrapper<const Tensor>>>& optional_input_tensors,
        std::vector<Tensor> &output_tensors
    ) const;
};

} }  // namespace tt::tt_metal
