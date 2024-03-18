// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/moreh_bmm_backward/moreh_bmm_backward_op.hpp"

#include "tt_dnn/op_library/moreh_matmul/moreh_matmul_op.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {
namespace operations {
namespace primary {

inline void moreh_bmm_backward_validate(
    const Tensor &output_grad,
    const Tensor &input,
    const Tensor &mat2,
    std::optional<Tensor> input_grad,
    std::optional<Tensor> mat2_grad,
    std::optional<Tensor> output_tensor) {
    const auto &input_shape = input.get_legacy_shape().without_padding();
    const auto &mat2_shape = mat2.get_legacy_shape().without_padding();
    const auto &output_grad_shape = output_grad.get_legacy_shape().without_padding();
    TT_ASSERT(
        output_grad.storage_type() == StorageType::DEVICE && input.storage_type() == StorageType::DEVICE &&
            mat2.storage_type() == StorageType::DEVICE,
        "input tensors need to be on device");

    TT_ASSERT(input_shape[0] == 1, "input must be a 3D tensor");
    TT_ASSERT(mat2_shape[0] == 1, "mat2 must be a 3D tensor");
    TT_ASSERT(output_grad_shape[0] == 1, "output_grad must be a 3D tensor");
    TT_ASSERT(
        output_grad_shape[1] == input_shape[1] && output_grad_shape[2] == input_shape[2] &&
            output_grad_shape[3] == mat2_shape[3],
        "check output_grad shape");

    if (input_grad.has_value()) {
        const auto &input_grad_tensor = input_grad.value();
        TT_ASSERT(
            input_grad_tensor.get_legacy_shape().without_padding() == input_shape,
            "shape of input_grad should be the same as shape of input");
    }

    if (mat2_grad.has_value()) {
        const auto &mat2_grad_tensor = mat2_grad.value();
        TT_ASSERT(
            mat2_grad_tensor.get_legacy_shape().without_padding() == mat2_shape,
            "shape of mat2_grad should be the same as shape of mat2");
    }
}

[[maybe_unused]] std::vector<std::variant<Tensor, char *>> moreh_bmm_backward(
    const Tensor &output_grad,
    const Tensor &input,
    const Tensor &mat2,
    std::optional<Tensor> input_grad,
    std::optional<Tensor> mat2_grad,
    std::optional<Tensor> output_tensor,
    const MemoryConfig &output_mem_config) {
    std::vector<std::variant<Tensor, char *>> outputs;
    outputs.reserve(2);

    moreh_bmm_backward_validate(output_grad, input, mat2, input_grad, mat2_grad, output_tensor);

    if (input_grad.has_value()) {
        outputs.push_back(tt::operations::primary::moreh_matmul(
            output_grad, mat2, input_grad.value(), false, true, output_mem_config));
    } else {
        outputs.push_back(nullptr);
    }

    if (mat2_grad.has_value()) {
        outputs.push_back(tt::operations::primary::moreh_matmul(
            input, output_grad, mat2_grad.value(), true, false, output_mem_config));
    } else {
        outputs.push_back(nullptr);
    }

    if (output_tensor.has_value()) {
        outputs.push_back(output_tensor.value());
    } else {
        outputs.push_back(nullptr);
    }
    return outputs;
}

}  // namespace primary

}  // namespace operations

}  // namespace tt
