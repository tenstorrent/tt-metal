// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/moreh_bmm/moreh_bmm_op.hpp"
#include "tt_dnn/op_library/moreh_matmul/moreh_matmul_op.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {
namespace tt_metal {

inline void moreh_bmm_backward_validate(const Tensor &output_grad, const Tensor &input, const Tensor &mat2) {
    const auto &input_shape = input.shape().without_padding();
    const auto &mat2_shape = mat2.shape().without_padding();
    const auto &output_grad_shape = output_grad.shape().without_padding();
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
}

std::vector<Tensor> moreh_bmm_backward(
    const Tensor &output_grad, const Tensor &input, const Tensor &mat2, const MemoryConfig &output_mem_config) {
    std::vector<Tensor> outputs;
    outputs.reserve(2);

    moreh_bmm_backward_validate(output_grad, input, mat2);

    // input_grad
    outputs.push_back(tt::operations::primary::moreh_matmul(output_grad, mat2, false, true, output_mem_config));
    // mat2_grad
    outputs.push_back(tt::operations::primary::moreh_matmul(input, output_grad, true, false, output_mem_config));

    return outputs;
}

}  // namespace tt_metal
}  // namespace tt
