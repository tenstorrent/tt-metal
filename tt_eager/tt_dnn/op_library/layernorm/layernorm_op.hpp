// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "tt_eager/tensor/tensor.hpp"

#include "tt_dnn/op_library/run_operation.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks layernorm_multi_core(
    const Tensor &a,
    const std::optional<const Tensor> b,
    const std::optional<const Tensor> gamma,
    const std::optional<const Tensor> beta,
    Tensor& output,
    float eps,
    bool rms_norm = false,
    MathFidelity fidelity = MathFidelity::HiFi4,
    DataType im_data_format = DataType::BFLOAT16
);

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
    tt::stl::reflection::Attributes attributes() const;
};

struct RMSNorm {
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
    tt::stl::reflection::Attributes attributes() const;
};

inline Tensor layernorm(const Tensor &a, float eps, std::optional<const Tensor> gamma = std::nullopt, std::optional<const Tensor> beta = std::nullopt, const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    TT_ASSERT(a.shape()[3] % TILE_WIDTH == 0, "Normalizing on last dim cannot be padded");

    if (gamma.has_value() and gamma.value().layout() == Layout::TILE) {
        TT_ASSERT(gamma.value().shape()[3] == a.shape()[3], "Gamma width must be equal to input width");
    }
    if (beta.has_value() and beta.value().layout() == Layout::TILE) {
        TT_ASSERT(beta.value().shape()[3] == a.shape()[3], "Beta width must be equal to input width");
    }
    return operation::run_with_autoformat(LayerNorm{.eps=eps, .output_mem_config=mem_config}, {a}, {std::nullopt, gamma, beta}).at(0);
}

// computes layernorm(a+b)*gamma+beta
inline Tensor add_layernorm(const Tensor &a, const Tensor& b, float eps, std::optional<const Tensor> gamma = std::nullopt, std::optional<const Tensor> beta = std::nullopt, const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    TT_ASSERT(a.shape()[3] % TILE_WIDTH == 0, "Normalizing on last dim cannot be padded");
    TT_ASSERT(a.shape() == b.shape(), "Input shapes must be equal");
    if (gamma.has_value() and gamma.value().layout() == Layout::TILE) {
        TT_ASSERT(gamma.value().shape()[3] == a.shape()[3], "Gamma width must be equal to input width");
    }
    if (beta.has_value() and beta.value().layout() == Layout::TILE) {
        TT_ASSERT(beta.value().shape()[3] == a.shape()[3], "Beta width must be equal to input width");
    }
    return operation::run_with_autoformat(LayerNorm{.eps=eps, .output_mem_config=mem_config}, {a}, {b, gamma, beta}).at(0);
}

inline Tensor rmsnorm(const Tensor &a, float eps, std::optional<const Tensor> gamma = std::nullopt, std::optional<const Tensor> beta = std::nullopt, const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
    TT_ASSERT(a.shape()[3] % TILE_WIDTH == 0, "Normalizing on last dim cannot be padded");
    if (gamma.has_value()) {
        TT_ASSERT(gamma.value().shape()[3] == a.shape()[3], "Gamma width must be equal to input width");
    }
    if (beta.has_value()) {
        TT_ASSERT(beta.value().shape()[3] == a.shape()[3], "Beta width must be equal to input width");
    }
    return operation::run_with_autoformat(RMSNorm{.eps=eps, .output_mem_config=mem_config}, {a}, {std::nullopt, gamma, beta}).at(0);
}

}  // namespace metal

namespace operations {

using namespace tt_metal;

namespace primary {
struct LayerNormDefaultProgramConfig{
    tt::stl::reflection::Attributes attributes() const { return {}; };
};
struct LayerNormInterleavedMultiCoreProgramConfig {
    MathFidelity math_fidelity;
    DataType im_data_format;
    DataType out_data_format;

    tt::stl::reflection::Attributes attributes() const {
        return {
            {"math_fidelity", math_fidelity},
            {"im_data_format", im_data_format},
            {"out_data_format", out_data_format}
        };
    };
};
struct LayerNormShardedMultiCoreProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    std::size_t subblock_w;
    std::size_t block_h;
    std::size_t block_w;
    MathFidelity math_fidelity;
    DataType im_data_format;
    DataType out_data_format;
    bool inplace;

    tt::stl::reflection::Attributes attributes() const {
        return {
            {"compute_with_storage_grid_size", compute_with_storage_grid_size},
            {"subblock_w", subblock_w},
            {"block_h", block_h},
            {"block_w", block_w},
            {"math_fidelity", math_fidelity},
            {"im_data_format", im_data_format},
            {"out_data_format", out_data_format},
            {"inplace", inplace},
        };
    };
};


using LayerNormProgramConfig = std::variant<
    LayerNormDefaultProgramConfig,
    LayerNormInterleavedMultiCoreProgramConfig,
    LayerNormShardedMultiCoreProgramConfig
>;

operation::ProgramWithCallbacks layernorm_multi_core_sharded(
    const Tensor &a,
    const std::optional<const Tensor> b,
    const std::optional<const Tensor> gamma,
    const std::optional<const Tensor> beta,
    Tensor& output,
    float eps,
    MathFidelity fidelity,
    DataType im_data_format,
    CoreCoord compute_grid_size,
    uint32_t subblock_wt,
    uint32_t block_ht,
    uint32_t block_wt
);

struct LayerNorm {
    float eps;
    MemoryConfig output_mem_config;
    tt::operations::primary::LayerNormProgramConfig program_config;

    void validate(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor> &output_tensors
    ) const;
    tt::stl::reflection::Attributes attributes() const;
};

inline Tensor layernorm(const Tensor &a, float eps, std::optional<const Tensor> gamma = std::nullopt, std::optional<const Tensor> beta = std::nullopt, const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, const LayerNormProgramConfig& program_config = LayerNormDefaultProgramConfig{}) {
    return operation::run(LayerNorm{.eps=eps, .output_mem_config=mem_config, .program_config=program_config}, {a}, {std::nullopt, gamma, beta}).at(0);
}

// computes layernorm(a+b)*gamma+beta
inline Tensor add_layernorm(const Tensor &a, const Tensor& b, float eps, std::optional<const Tensor> gamma = std::nullopt, std::optional<const Tensor> beta = std::nullopt, const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, const LayerNormProgramConfig& program_config = LayerNormDefaultProgramConfig{}) {
    return operation::run(LayerNorm{.eps=eps, .output_mem_config=mem_config, .program_config=program_config}, {a}, {b, gamma, beta}).at(0);
}
}  // namespace primary

}  // namespace operations

}  // namespace tt
