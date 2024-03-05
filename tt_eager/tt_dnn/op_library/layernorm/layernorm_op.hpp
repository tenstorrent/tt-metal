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

enum class LayerNormType {
    LAYERNORM, RMSNORM
};

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

operation::ProgramWithCallbacks layernorm_multi_core(
    const Tensor &a,
    const std::optional<const Tensor> b,
    const std::optional<const Tensor> gamma,
    const std::optional<const Tensor> beta,
    Tensor& output,
    LayerNormType norm_type,
    float eps,
    MathFidelity fidelity = MathFidelity::HiFi4,
    DataType im_data_format = DataType::BFLOAT16
);

operation::ProgramWithCallbacks layernorm_multi_core_sharded(
    const Tensor &a,
    const std::optional<const Tensor> b,
    const std::optional<const Tensor> gamma,
    const std::optional<const Tensor> beta,
    Tensor& output,
    LayerNormType norm_type,
    float eps,
    MathFidelity fidelity,
    DataType im_data_format,
    CoreCoord compute_grid_size,
    uint32_t subblock_wt,
    uint32_t block_ht,
    uint32_t block_wt
);

struct LayerNorm {
    LayerNormType norm_type;
    float eps;
    MemoryConfig output_mem_config;
    LayerNormProgramConfig program_config;

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

template <LayerNormType norm_type>
struct make_layernorm {
    Tensor operator()(
        const Tensor &a, float eps, std::optional<const Tensor> gamma = std::nullopt, std::optional<const Tensor> beta = std::nullopt, const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) const {
        TT_FATAL(a.get_legacy_shape()[3] % TILE_WIDTH == 0, "Normalizing on last dim cannot be padded");

        if (gamma.has_value() and gamma.value().get_layout() == Layout::TILE) {
            TT_FATAL(gamma.value().get_legacy_shape()[3] == a.get_legacy_shape()[3], "Gamma width must be equal to input width");
        }
        if (beta.has_value() and beta.value().get_layout() == Layout::TILE) {
            TT_FATAL(beta.value().get_legacy_shape()[3] == a.get_legacy_shape()[3], "Beta width must be equal to input width");
        }
        return operation::run_with_autoformat(LayerNorm{.norm_type=norm_type, .eps=eps, .output_mem_config=mem_config, .program_config=LayerNormDefaultProgramConfig()}, {a}, {std::nullopt, gamma, beta}).at(0);
    }
};

template <LayerNormType norm_type>
struct make_add_layernorm {
    Tensor operator()(
        const Tensor &a, const Tensor& b, float eps, std::optional<const Tensor> gamma = std::nullopt, std::optional<const Tensor> beta = std::nullopt, const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG) const {
        TT_FATAL(a.get_legacy_shape()[3] % TILE_WIDTH == 0, "Normalizing on last dim cannot be padded");
        TT_FATAL(a.get_legacy_shape() == b.get_legacy_shape(), "Input shapes must be equal");
        if (gamma.has_value() and gamma.value().get_layout() == Layout::TILE) {
            TT_FATAL(gamma.value().get_legacy_shape()[3] == a.get_legacy_shape()[3], "Gamma width must be equal to input width");
        }
        if (beta.has_value() and beta.value().get_layout() == Layout::TILE) {
            TT_FATAL(beta.value().get_legacy_shape()[3] == a.get_legacy_shape()[3], "Beta width must be equal to input width");
        }
        return operation::run_with_autoformat(LayerNorm{.norm_type=norm_type, .eps=eps, .output_mem_config=mem_config, .program_config=LayerNormDefaultProgramConfig()}, {a}, {b, gamma, beta}).at(0);
    }
};

constexpr auto layernorm = make_layernorm<LayerNormType::LAYERNORM>{};
constexpr auto rmsnorm = make_layernorm<LayerNormType::RMSNORM>{};

// computes layernorm(a+b)*gamma+beta
constexpr auto add_layernorm = make_add_layernorm<LayerNormType::LAYERNORM>{};
constexpr auto add_rmsnorm = make_add_layernorm<LayerNormType::RMSNORM>{};

}  // namespace metal

namespace operations {

namespace primary {

template <LayerNormType layernorm_type>
struct make_layernorm {
    Tensor operator()(
        const Tensor &a, float eps, std::optional<const Tensor> gamma = std::nullopt, std::optional<const Tensor> beta = std::nullopt, const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, const LayerNormProgramConfig& program_config = LayerNormDefaultProgramConfig{}) const {
        return operation::run(LayerNorm{.norm_type=layernorm_type, .eps=eps, .output_mem_config=mem_config, .program_config=program_config}, {a}, {std::nullopt, gamma, beta}).at(0);
    }
};

template <LayerNormType layernorm_type>
struct make_add_layernorm {
    Tensor operator()(
        const Tensor &a, const Tensor& b, float eps, std::optional<const Tensor> gamma = std::nullopt, std::optional<const Tensor> beta = std::nullopt, const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, const LayerNormProgramConfig& program_config = LayerNormDefaultProgramConfig{}) const {
        return operation::run(LayerNorm{.norm_type=layernorm_type, .eps=eps, .output_mem_config=mem_config, .program_config=program_config}, {a}, {b, gamma, beta}).at(0);
    }
};

constexpr auto layernorm = make_layernorm<LayerNormType::LAYERNORM>{};
constexpr auto rmsnorm = make_layernorm<LayerNormType::RMSNORM>{};

// computes layernorm(a+b)*gamma+beta
constexpr auto add_layernorm = make_add_layernorm<LayerNormType::LAYERNORM>{};
constexpr auto add_rmsnorm = make_add_layernorm<LayerNormType::RMSNORM>{};

}  // namespace primary

}  // namespace operations

}  // namespace tt
