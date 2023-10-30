// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/moreh_matmul_backward/moreh_matmul_backward_op.hpp"

#include "tt_dnn/op_library/moreh_dot_backward/moreh_dot_backward_op.hpp"
#include "tt_dnn/op_library/moreh_matmul/moreh_matmul_op.hpp"
#include "tt_eager/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {
using namespace constants;
using namespace tt_metal;
namespace operations {
namespace primary {

////////////////////////////////////////////////////////////////////////////
//                         Util
////////////////////////////////////////////////////////////////////////////
inline bool is_dot_backward(const Tensor& output_grad, const Tensor& input, const Tensor& other) {
    return is_scalar(output_grad) && is_1d_tensor(input) && is_1d_tensor(other) && is_same_shape(input, other);
}

////////////////////////////////////////////////////////////////////////////
//                         MorehSum
////////////////////////////////////////////////////////////////////////////
void MorehSum::validate(const std::vector<Tensor>& inputs) const {
    const auto& src = inputs.at(0);
    const auto& dst = inputs.at(1);
    const auto& src_shape = src.shape();
    const auto& dst_shape = dst.shape();

    TT_ASSERT(src_shape[2] == dst_shape[2] && src_shape[3] == dst_shape[3]);
    TT_ASSERT(src_shape[0] >= dst_shape[0]);
    TT_ASSERT(src_shape[1] >= dst_shape[1]);
}

std::vector<Tensor> MorehSum::create_output_tensors(const std::vector<Tensor>& inputs) const {
    // Inplace
    return {};
}

std::vector<Shape> MorehSum::compute_output_shapes(const std::vector<Tensor>& inputs) const {
    // Inplace
    return {};
}

operation::ProgramWithCallbacks MorehSum::create_program(
    const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) const {
    const auto& src = inputs.at(0);
    const auto& dst = inputs.at(1);
    return moreh_sum_multi_core(src, dst);
}

tt::stl::reflection::Attributes MorehSum::attributes() const { return {}; }

////////////////////////////////////////////////////////////////////////////
//                         moreh_matmul_backward
////////////////////////////////////////////////////////////////////////////
[[maybe_unused]] std::vector<std::variant<Tensor, char*>> moreh_matmul_backward_(
    const Tensor& output_grad,
    const Tensor& input,
    const Tensor& other,
    std::optional<std::reference_wrapper<const Tensor>> input_grad,
    std::optional<std::reference_wrapper<const Tensor>> other_grad,
    const MemoryConfig& output_mem_config) {
    std::vector<std::variant<Tensor, char*>> outputs;
    outputs.reserve(2);

    if (input_grad) {
        const auto& input_grad_tensor = input_grad->get();
        if (is_same_batch_shape(output_grad, input_grad_tensor)) {
            const auto& input_grad_shape = input_grad_tensor.shape().without_padding();
            const auto& output_grad_shape = output_grad.shape().without_padding();
            tt::operations::primary::moreh_matmul(
                output_grad, other, input_grad_tensor, false, true, output_mem_config);
        } else {
            const auto& input_shape = input.shape().without_padding();
            const auto& temp_input_grad =
                tt::operations::primary::moreh_matmul(output_grad, other, std::nullopt, false, true, output_mem_config);
            operation::run(MorehSum{}, {temp_input_grad, input_grad_tensor});
        }
        outputs.push_back(input_grad_tensor);
    } else {
        outputs.push_back(nullptr);
    }

    if (other_grad) {
        const auto& other_grad_tensor = other_grad->get();
        if (is_same_batch_shape(output_grad, other_grad_tensor)) {
            tt::operations::primary::moreh_matmul(
                input, output_grad, other_grad_tensor, true, false, output_mem_config);
        } else {
            const auto& temp_other_grad =
                tt::operations::primary::moreh_matmul(input, output_grad, std::nullopt, true, false, output_mem_config);
            operation::run(MorehSum{}, {temp_other_grad, other_grad_tensor});
        }
        outputs.push_back(other_grad_tensor);
    } else {
        outputs.push_back(nullptr);
    }

    return outputs;
}

[[maybe_unused]] std::vector<std::variant<Tensor, char*>> moreh_matmul_backward(
    const Tensor& output_grad,
    const Tensor& input,
    const Tensor& other,
    std::optional<std::reference_wrapper<const Tensor>> input_grad,
    std::optional<std::reference_wrapper<const Tensor>> other_grad,
    const MemoryConfig& output_mem_config) {
    if (is_dot_backward(output_grad, input, other)) {
        return moreh_dot_backward(output_grad, input, other, input_grad, other_grad, output_mem_config);
    } else {
        return moreh_matmul_backward_(output_grad, input, other, input_grad, other_grad, output_mem_config);
    }
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
