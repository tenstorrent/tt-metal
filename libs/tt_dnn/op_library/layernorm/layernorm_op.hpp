#pragma once

#include <optional>

#include "libs/tensor/tensor.hpp"

#include "tt_dnn/op_library/run_operation.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

struct LayerNorm {
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


struct BertLargeLayerNorm {
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

inline Tensor layernorm(const Tensor &a, float eps, std::optional<const Tensor> gamma = std::nullopt, std::optional<const Tensor> beta = std::nullopt, const MemoryConfig& mem_config = MemoryConfig{.interleaved = true}) {
    TT_ASSERT(a.shape()[3] % TILE_WIDTH == 0, "Normalizing on last dim cannot be padded");
    if (gamma.has_value()) {
        TT_ASSERT(gamma.value().shape()[3] == a.shape()[3], "Gamma width must be equal to input width");
    }
    if (beta.has_value()) {
        TT_ASSERT(beta.value().shape()[3] == a.shape()[3], "Beta width must be equal to input width");
    }
    return operation::run_with_autoformat(LayerNorm{.eps=eps, .output_mem_config=mem_config}, {a}, {std::nullopt, gamma, beta}).at(0);
}

// computes layernorm(a+b)*gamma+beta
inline Tensor add_layernorm(const Tensor &a, const Tensor& b, float eps, std::optional<const Tensor> gamma = std::nullopt, std::optional<const Tensor> beta = std::nullopt, const MemoryConfig& mem_config = MemoryConfig{.interleaved = true}) {
    TT_ASSERT(a.shape()[3] % TILE_WIDTH == 0, "Normalizing on last dim cannot be padded");
    TT_ASSERT(a.shape() == b.shape(), "Input shapes must be equal");
    if (gamma.has_value()) {
        TT_ASSERT(gamma.value().shape()[3] == a.shape()[3], "Gamma width must be equal to input width");
    }
    if (beta.has_value()) {
        TT_ASSERT(beta.value().shape()[3] == a.shape()[3], "Beta width must be equal to input width");
    }
    return operation::run_with_autoformat(LayerNorm{.eps=eps, .output_mem_config=mem_config}, {a}, {b, gamma, beta}).at(0);
}

inline Tensor bert_large_layernorm(const Tensor &a, float eps, std::optional<const Tensor> gamma = std::nullopt, std::optional<const Tensor> beta = std::nullopt, const MemoryConfig& mem_config = MemoryConfig{.interleaved = true}) {
    return operation::run_without_autoformat(BertLargeLayerNorm{.eps=eps, .output_mem_config=mem_config}, {a}, {std::nullopt, gamma, beta}).at(0);
}

inline Tensor bert_large_add_layernorm(const Tensor &a, const Tensor& b, float eps, std::optional<const Tensor> gamma = std::nullopt, std::optional<const Tensor> beta = std::nullopt, const MemoryConfig& mem_config = MemoryConfig{.interleaved = true}) {
    return operation::run_without_autoformat(BertLargeLayerNorm{.eps=eps, .output_mem_config=mem_config}, {a}, {b, gamma, beta}).at(0);
}

}

}  // namespace tt::tt_metal
