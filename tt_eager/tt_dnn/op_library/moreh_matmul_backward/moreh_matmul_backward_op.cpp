// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/moreh_matmul_backward/moreh_matmul_backward_op.hpp"

#include "tt_dnn/op_library/moreh_dot_backward/moreh_dot_backward_op.hpp"
#include "tt_dnn/op_library/moreh_matmul/moreh_matmul_op.hpp"
#include "tt_dnn/op_library/moreh_sum/moreh_sum_op.hpp"
#include "tt_eager/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {
using namespace constants;
namespace operations {
namespace primary {

////////////////////////////////////////////////////////////////////////////
//                         Util
////////////////////////////////////////////////////////////////////////////

namespace {
inline bool is_dot_backward(const Tensor& output_grad, const Tensor& input, const Tensor& other) {
    // TODO: non-4d support for dot backward.
    if (output_grad.get_legacy_shape().rank() != 4 || input.get_legacy_shape().rank() != 4 || other.get_legacy_shape().rank() != 4) {
        return false;
    }
    return is_scalar(output_grad) && is_1d_tensor(input) && is_1d_tensor(other) && is_same_shape(input, other);
}

inline bool is_same_batch_dim(const Tensor &tensor_a, const Tensor &tensor_b) {
    // check batch dims
    const auto &a_shape = tensor_a.get_legacy_shape();
    const auto &b_shape = tensor_b.get_legacy_shape();
    std::vector<uint32_t> a_dim(tt::tt_metal::MAX_NUM_DIMENSIONS, 1);
    std::vector<uint32_t> b_dim(tt::tt_metal::MAX_NUM_DIMENSIONS, 1);
    get_tensor_dim(a_dim, a_shape);
    get_tensor_dim(b_dim, b_shape);
    for (auto i = 2; i < tt::tt_metal::MAX_NUM_DIMENSIONS; ++i) {
        if (a_dim[i] != b_dim[i]) {
            log_debug(LogOp, "{}:{} {} a_dim {} - b_dim {}", __func__, __LINE__, i, a_dim[i], b_dim[i]);
            return false;
        }
    }
    log_debug(LogOp, "{}:{} batch dims are the same.", __func__, __LINE__);
    return true;
}

}

////////////////////////////////////////////////////////////////////////////
//                         moreh_matmul_backward
////////////////////////////////////////////////////////////////////////////
std::vector<std::optional<Tensor>> moreh_matmul_backward_(
    const Tensor& output_grad,
    const Tensor& input,
    const Tensor& other,
    const std::vector<bool>& are_required_outputs,
    const std::optional<const Tensor> &input_grad,
    const std::optional<const Tensor> &other_grad,
    const MemoryConfig& output_mem_config) {
    std::vector<std::optional<Tensor>> outputs(2);
    outputs.reserve(2);

    auto find_reduce_dim = [](const Shape& a_shape, const Shape& b_shape) -> std::vector<int64_t> {

        std::vector<uint32_t> a_dim(tt::tt_metal::MAX_NUM_DIMENSIONS, 1);
        std::vector<uint32_t> b_dim(tt::tt_metal::MAX_NUM_DIMENSIONS, 1);
        get_tensor_dim(a_dim, a_shape);
        get_tensor_dim(b_dim, b_shape);
        int32_t rank = std::max(a_shape.rank(), b_shape.rank());
        log_debug(LogOp, "find_reduce_dim :{} rank {} a {} b {}", __LINE__, rank, a_shape.rank(), b_shape.rank());
        std::vector<int64_t> dims;
        // batch dims
        for (int i = 0; i < rank - 2; ++i) {
            int idx = rank - 1 - i;
            TT_ASSERT(idx >= 0);
            if (a_dim[idx] != b_dim[idx]) {
                dims.push_back(i);
                log_debug(LogOp, "find_reduce_dim :{} push {} dim", __LINE__, i);
            }
        }
        return dims;
    };

    const bool input_requires_grad = are_required_outputs.at(0);
    const bool other_requires_grad = are_required_outputs.at(1);

    if (input_requires_grad) {
        TT_ASSERT(input_grad.has_value());
        const auto& input_grad_tensor = input_grad.value();
        if (is_same_batch_dim(output_grad, input_grad_tensor)) {
            const auto& input_grad_shape = input_grad_tensor.get_legacy_shape().without_padding();
            const auto& output_grad_shape = output_grad.get_legacy_shape().without_padding();
            moreh_matmul(output_grad, other, false, true, input_grad_tensor, std::nullopt, output_mem_config);
        } else {
            const auto& input_shape = input.get_legacy_shape().without_padding();
            const auto& temp_input_grad =
                moreh_matmul(output_grad, other, false, true, std::nullopt, std::nullopt, output_mem_config);
            auto reduce_dims = find_reduce_dim(temp_input_grad.get_legacy_shape(), input_grad_tensor.get_legacy_shape());
            moreh_sum(temp_input_grad, reduce_dims, input_grad_tensor);
        }
        outputs[0] = input_grad_tensor;
    }

    if (other_requires_grad) {
        TT_ASSERT(other_grad.has_value());
        const auto& other_grad_tensor = other_grad.value();
        if (is_same_batch_dim(output_grad, other_grad_tensor)) {
            moreh_matmul(input, output_grad, true, false, other_grad_tensor, std::nullopt, output_mem_config);
        } else {
            const auto& temp_other_grad =
                moreh_matmul(input, output_grad, true, false, std::nullopt, std::nullopt, output_mem_config);
            auto reduce_dims = find_reduce_dim(temp_other_grad.get_legacy_shape(), other_grad_tensor.get_legacy_shape());
            moreh_sum(temp_other_grad, reduce_dims, other_grad_tensor);
        }
        outputs[1] = other_grad_tensor;
    }

    return outputs;
}

std::vector<std::optional<Tensor>> moreh_matmul_backward(
    const Tensor& output_grad,
    const Tensor& input,
    const Tensor& other,
    const std::vector<bool>& are_required_outputs,
    std::optional<const Tensor> input_grad,
    std::optional<const Tensor> other_grad,
    const MemoryConfig& output_mem_config) {
    if (is_dot_backward(output_grad, input, other)) {
        // TODO(seunghwan100): Add the argument "are_required_outputs" to moreh_dot_backward.
        return moreh_dot_backward(output_grad, input, other, input_grad, other_grad, output_mem_config);
    } else {
        return moreh_matmul_backward_(
            output_grad, input, other, are_required_outputs, input_grad, other_grad, output_mem_config);
    }
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
