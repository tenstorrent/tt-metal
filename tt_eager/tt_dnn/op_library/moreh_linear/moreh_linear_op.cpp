// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/moreh_linear/moreh_linear_op.hpp"

#include <type_traits>

#include "tt_dnn/op_library/bcast/bcast_op.hpp"
#include "tt_dnn/op_library/moreh_matmul/moreh_matmul_op.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {
namespace operations {
namespace primary {

inline bool is_shape_out_features(const Shape& bias, const Shape& weight) {
    return (bias[0] == 1 && bias[1] == 1 && bias[2] == 1 && bias[3] == weight[2]);
}

inline bool is_shape_scalar(const Shape& bias) {
    return (bias[0] == 1 && bias[1] == 1 && bias[2] == 1 && bias[3] == 1);
}

inline void moreh_linear_validate(
    const Tensor& input, const Tensor& weight, std::optional<std::reference_wrapper<const Tensor>> bias) {
    const auto& weight_shape = weight.get_legacy_shape().without_padding();
    TT_ASSERT(weight_shape[0] == 1 && weight_shape[1] == 1, "weight should be a 2D tensor");
    if (bias) {
        const auto& bias_tensor = bias->get();
        const auto& bias_shape = bias_tensor.get_legacy_shape().without_padding();
        TT_ASSERT(
            is_shape_out_features(bias_shape, weight_shape) || is_shape_scalar(bias_shape),
            "shape of bias should be [1, 1, 1, wieght_shape[2]] or [1, 1, 1, 1]");
    }
}

Tensor moreh_linear_(
    const Tensor& input,
    const Tensor& weight,
    std::optional<std::reference_wrapper<const Tensor>> bias,
    const MemoryConfig& output_mem_config) {
    moreh_linear_validate(input, weight, bias);
    Tensor mm_output = moreh_matmul(input, weight, std::nullopt, false, true, output_mem_config);
    if (bias) {
        const auto& bias_tensor = bias->get();
        const auto& bias_shape = bias_tensor.get_legacy_shape().without_padding();
        BcastOpDim bcast_dim = is_shape_scalar(bias_shape) ? BcastOpDim::HW : BcastOpDim::H;
        return tt::tt_metal::bcast(mm_output, bias_tensor, BcastOpMath::ADD, bcast_dim, output_mem_config);
    }
    return mm_output;
}

Tensor moreh_linear(
    const Tensor& input,
    const Tensor& weight,
    std::optional<std::reference_wrapper<const Tensor>> bias,
    const MemoryConfig& output_mem_config) {
    TT_ASSERT(
        input.storage_type() == StorageType::DEVICE && weight.storage_type() == StorageType::DEVICE,
        "input and weight tensors need to be on device");
    return moreh_linear_(input, weight, bias, output_mem_config);
}

}  // namespace primary

}  // namespace operations

}  // namespace tt
