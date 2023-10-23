// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/moreh_linear/moreh_linear_op.hpp"

#include <type_traits>

#include "tt_dnn/op_library/bcast/bcast_op.hpp"
#include "tt_dnn/op_library/moreh_matmul/moreh_matmul_op.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {
namespace tt_metal {

inline void moreh_linear_validate(
    const Tensor& input, const Tensor& weight, std::optional<std::reference_wrapper<const Tensor>> bias) {
    const auto& weight_wo_shape = weight.shape().without_padding();
    TT_ASSERT(weight_wo_shape[0] == 1 && weight_wo_shape[1] == 1, "weight should be a 2D tensor");
    if (bias) {
        const auto& bias_tensor = bias->get();
        const auto& bias_wo_shape = bias_tensor.shape().without_padding();
        TT_ASSERT(
            bias_wo_shape[0] == 1 && bias_wo_shape[1] == 1 && bias_wo_shape[2] == 1 &&
                bias_wo_shape[3] == weight_wo_shape[2],
            "shape of bias should be [1, 1, 1, wieght_wo_shape[2]]");
    }
}

Tensor moreh_linear_(
    const Tensor& input,
    const Tensor& weight,
    std::optional<std::reference_wrapper<const Tensor>> bias,
    const MemoryConfig& output_mem_config) {
    moreh_linear_validate(input, weight, bias);
    Tensor mm_output = tt::operations::primary::moreh_matmul(input, weight, false, true, output_mem_config);
    if (bias) {
        return bcast(mm_output, bias->get(), BcastOpMath::ADD, BcastOpDim::H, output_mem_config);
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

}  // namespace tt_metal
}  // namespace tt
