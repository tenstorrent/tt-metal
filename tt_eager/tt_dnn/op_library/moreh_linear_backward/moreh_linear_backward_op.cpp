// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/moreh_linear_backward/moreh_linear_backward_op.hpp"

#include <type_traits>

#include "tt_dnn/op_library/moreh_matmul/moreh_matmul_op.hpp"
#include "tt_dnn/op_library/moreh_sum/moreh_sum_op.hpp"
#include "tt_dnn/op_library/transpose/transpose_op.hpp"
#include "tt_eager/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {
namespace operations {
namespace primary {

// TODO: Move bias backward code
////////////////////////////////////////////////////////////////////////////
//                         MorehBiasBackward
////////////////////////////////////////////////////////////////////////////
void MorehBiasBackward::validate(const std::vector<Tensor>& inputs) const {
    const auto& bias_grad = inputs.at(1);
    TT_ASSERT(is_scalar(bias_grad) || is_1d_tensor(bias_grad), "bias_grad tensor should be 1d or scalar");
}

std::vector<Shape> MorehBiasBackward::compute_output_shapes(const std::vector<Tensor>& inputs) const {
    // Inplace
    return {};
}

std::vector<Tensor> MorehBiasBackward::create_output_tensors(const std::vector<Tensor>& inputs) const {
    // Inplace
    return {};
}

operation::ProgramWithCallbacks MorehBiasBackward::create_program(
    const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) const {
    const auto& output_grad = inputs.at(0);
    const auto& bias_grad = inputs.at(1);
    return is_scalar(bias_grad) ? (moreh_bias_backward_single_core_hw(output_grad, bias_grad))
                                : (moreh_bias_backward_multi_core_h(output_grad, bias_grad));
}

stl::reflection::Attributes MorehBiasBackward::attributes() const { return {}; }

inline void moreh_linear_backward_validate(
    const Tensor& output_grad,
    const Tensor& input,
    const Tensor& weight,
    std::optional<std::reference_wrapper<const Tensor>> input_grad,
    std::optional<std::reference_wrapper<const Tensor>> weight_grad,
    std::optional<std::reference_wrapper<const Tensor>> bias_grad) {
    if (input_grad) {
        const auto& input_grad_tensor = input_grad->get();
        TT_ASSERT(is_same_shape(input, input_grad_tensor), "both tensors should be the same shape");
    }

    if (weight_grad) {
        const auto& weight_grad_tensor = weight_grad->get();
        TT_ASSERT(is_same_shape(weight, weight_grad_tensor), "both tensors should be the same shape");
    }

    if (bias_grad) {
        const auto& bias_grad_tensor = bias_grad->get();
        TT_ASSERT(
            is_scalar(bias_grad_tensor) || is_1d_tensor(bias_grad_tensor), "bias_grad tensor should be 1d or scalar");
    }
}

[[maybe_unused]] std::vector<std::variant<tt_metal::Tensor, char*>> moreh_linear_backward(
    const Tensor& output_grad,
    const Tensor& input,
    const Tensor& weight,
    std::optional<std::reference_wrapper<const Tensor>> input_grad,
    std::optional<std::reference_wrapper<const Tensor>> weight_grad,
    std::optional<std::reference_wrapper<const Tensor>> bias_grad,
    const MemoryConfig& output_mem_config) {
    TT_ASSERT(
        output_grad.storage_type() == StorageType::DEVICE && input.storage_type() == StorageType::DEVICE &&
            weight.storage_type() == StorageType::DEVICE,
        "input and weight tensors need to be on device");
    std::vector<std::variant<Tensor, char*>> outputs;
    outputs.reserve(3);

    moreh_linear_backward_validate(output_grad, input, weight, input_grad, weight_grad, bias_grad);
    if (input_grad) {
        outputs.push_back(moreh_matmul(output_grad, weight, input_grad->get(), false, false, output_mem_config));
    } else {
        outputs.push_back(nullptr);
    }

    if (weight_grad) {
        // TODO: Add output transpose and remove transpose wh
        const auto& temp_weight_grad = moreh_matmul(input, output_grad, std::nullopt, true, false, output_mem_config);
        const auto& transposed_weight_grad = transpose(temp_weight_grad, 2, 3);
        // transpose op does not set the transposed shape.
        auto transposed_weight_grad_shape = weight_grad->get().get_legacy_shape();
        transposed_weight_grad_shape[0] = temp_weight_grad.get_legacy_shape()[0];
        transposed_weight_grad_shape[1] = temp_weight_grad.get_legacy_shape()[1];
        const auto& reshaped_weight_grad = transposed_weight_grad.reshape(transposed_weight_grad_shape);
        std::vector<int64_t> dims {0, 1};
        moreh_sum(reshaped_weight_grad, weight_grad->get(), dims);
        outputs.push_back(weight_grad->get());
    } else {
        outputs.push_back(nullptr);
    }

    if (bias_grad) {
        operation::run(MorehBiasBackward{}, {output_grad, bias_grad->get()});
        outputs.push_back(bias_grad->get());
    } else {
        outputs.push_back(nullptr);
    }

    return outputs;
}

}  // namespace primary

}  // namespace operations

}  // namespace tt
