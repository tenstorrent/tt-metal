// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/base_types.hpp"
#include "common/core_coord.h"
#include "tensor/types.hpp"
#include "tt_eager/tensor/tensor.hpp"

#include "tt_dnn/op_library/operation.hpp"

#include "tt_dnn/op_library/compute_kernel_config.hpp"

namespace tt {
namespace operations {
namespace primary {

using namespace tt_metal;

namespace transformers {
struct SDPADefaultProgramConfig{
    tt::stl::reflection::Attributes attributes() const { return {}; };
};

struct SDPAMultiCoreProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    std::size_t q_chunk_size;
    std::size_t k_chunk_size;

    tt::stl::reflection::Attributes attributes() const {
        return {
            {"compute_with_storage_grid_size", compute_with_storage_grid_size},
            {"q_chunk_size", q_chunk_size},
            {"k_chunk_size", k_chunk_size}
        };
    };
};

using SDPAProgramConfig = std::variant<
    SDPADefaultProgramConfig,
    SDPAMultiCoreProgramConfig
>;

}  // namespace transformers

struct ScaledDotProductAttention {
    const std::optional<float> scale;
    const MemoryConfig output_mem_config;
    const tt::operations::primary::transformers::SDPAProgramConfig program_config;
    const bool is_causal;
    const DeviceComputeKernelConfig compute_kernel_config;
    std::optional<const uint32_t> valid_seq_len;

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

operation::ProgramWithCallbacks sdpa_multi_core(
    const Tensor &input_tensor_q,
    const Tensor &input_tensor_k,
    const Tensor &input_tensor_v,
    const Tensor &output_tensor,
    const std::optional<const Tensor> causal_mask,
    std::optional<float> scale,
    bool is_causal,
    std::size_t q_chunk_size,
    std::size_t k_chunk_size,
    DeviceComputeKernelConfig compute_kernel_config,
    tt::operations::primary::transformers::SDPAProgramConfig program_config,
    std::optional<const uint32_t> valid_seq_len
);

struct ScaledDotProductAttentionDecode {
    const std::optional<float> scale;
    const MemoryConfig output_mem_config;
    const tt::operations::primary::transformers::SDPAProgramConfig program_config;
    const DeviceComputeKernelConfig compute_kernel_config;
    std::optional<const uint32_t> valid_seq_len;

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

operation::ProgramWithCallbacks sdpa_decode_multi_core(
    const Tensor &input_tensor_q,
    const Tensor &input_tensor_k,
    const Tensor &input_tensor_v,
    const Tensor &output_tensor,
    const std::optional<const Tensor> mask,
    std::optional<float> scale,
    std::size_t k_chunk_size,
    DeviceComputeKernelConfig compute_kernel_config,
    tt::operations::primary::transformers::SDPAProgramConfig program_config,
    std::optional<const uint32_t> valid_seq_len
);

namespace transformers {

Tensor scaled_dot_product_attention(Tensor& input_tensor_q, Tensor& input_tensor_k, Tensor& input_tensor_v, std::optional<const Tensor> causal_mask = std::nullopt, const bool is_causal = true, std::optional<float> scale = std::nullopt, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, const SDPAProgramConfig& program_config = SDPADefaultProgramConfig{}, std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt, std::optional<const uint32_t> valid_seq_len = std::nullopt);

Tensor scaled_dot_product_attention_decode(Tensor& input_tensor_q, Tensor& input_tensor_k, Tensor& input_tensor_v, std::optional<const Tensor> mask = std::nullopt, std::optional<float> scale = std::nullopt, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, const SDPAProgramConfig& program_config = SDPADefaultProgramConfig{}, std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt, std::optional<const uint32_t> valid_seq_len = std::nullopt);

}   // namespace transformers

}   // namespace primary
}   // namespace operations
}   // namespace tt
