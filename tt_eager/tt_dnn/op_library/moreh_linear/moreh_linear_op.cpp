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
    const Tensor& input, const Tensor& weight, const std::optional<const Tensor>& bias, const std::optional<const Tensor>& output) {
    const auto& weight_shape = weight.get_legacy_shape().without_padding();
    TT_ASSERT(weight_shape[0] == 1 && weight_shape[1] == 1, "weight should be a 2D tensor");

    if (bias.has_value()) {
        const auto& bias_tensor = bias.value();
        const auto& bias_shape = bias_tensor.get_legacy_shape().without_padding();
        TT_ASSERT(
            is_shape_out_features(bias_shape, weight_shape) || is_shape_scalar(bias_shape),
            "shape of bias should be [1, 1, 1, wieght_shape[2]] or [1, 1, 1, 1]");
    }

    if (output.has_value()) {
        const auto& output_tensor = output.value();
        const auto& output_shape = output_tensor.get_legacy_shape().without_padding();
        const auto& input_shape = input.get_legacy_shape().without_padding();
        TT_ASSERT(
            input_shape[0] == output_shape[0] &&
            input_shape[1] == output_shape[1] &&
            input_shape[2] == output_shape[2] &&
            weight_shape[2] == output_shape[3],
            "shape of output should be [input_shape[0], input_shape[1], input_shape[2], weight_shape[2]]");
    }
}

Tensor _moreh_linear(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<const Tensor>& bias,
    std::optional<Tensor> output,
    const MemoryConfig& output_mem_config) {
    moreh_linear_validate(input, weight, bias, output);
    output = moreh_matmul(input, weight, false, true, output, bias, output_mem_config);
    return output.value();
}

Tensor moreh_linear(
    const Tensor& input,
    const Tensor& weight,
    std::optional<const Tensor> bias,
    std::optional<const Tensor> output,
    const MemoryConfig& output_mem_config) {
    TT_ASSERT(
        input.storage_type() == StorageType::DEVICE && weight.storage_type() == StorageType::DEVICE,
        "input and weight tensors need to be on device");
    return _moreh_linear(input, weight, bias, output, output_mem_config);
}

}  // namespace primary

}  // namespace operations

}  // namespace tt
