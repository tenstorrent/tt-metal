#pragma once

#include "libs/tensor/tensor.hpp"

#include "tt_dnn/op_library/run_operation.hpp"

namespace tt {

namespace tt_metal {

struct ResidualLayerNorm {
    float eps;
    MemoryConfig output_mem_config;

    void validate(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor> &output_tensors
    ) const;
    operation::Hash compute_program_hash(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;
};

inline Tensor layernorm(const Tensor &a, float eps, const MemoryConfig& mem_config) {
    return operation::run(ResidualLayerNorm{.eps=eps, .output_mem_config=mem_config}, {a}, {std::nullopt, std::nullopt, std::nullopt}).at(0);
}
inline Tensor layernorm_gamma(const Tensor &a, float eps, const Tensor& gamma, const MemoryConfig& mem_config) {
    return operation::run(ResidualLayerNorm{.eps=eps, .output_mem_config=mem_config}, {a}, {std::nullopt, gamma, std::nullopt}).at(0);
}
inline Tensor layernorm_gamma_beta(const Tensor &a, float eps, const Tensor& gamma, const Tensor& beta, const MemoryConfig& mem_config) {
    return operation::run(ResidualLayerNorm{.eps=eps, .output_mem_config=mem_config}, {a}, {std::nullopt, gamma, beta}).at(0);
}
// computes layernorm(a+b)*gamma+beta
inline Tensor add_layernorm_gamma_beta(const Tensor &a, const Tensor& b, float eps, const Tensor& gamma, const Tensor& beta, const MemoryConfig& mem_config) {
    return operation::run(ResidualLayerNorm{.eps=eps, .output_mem_config=mem_config}, {a}, {b, gamma, beta}).at(0);
}

}

}  // namespace tt::tt_metal
