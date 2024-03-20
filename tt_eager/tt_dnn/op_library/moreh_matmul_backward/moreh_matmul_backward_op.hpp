/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"

namespace tt {

namespace operations {

namespace primary {

using namespace tt_metal;

/*
 * GENERAL matmul_backward
 */
[[maybe_unused]] std::vector<std::variant<Tensor, char *>> moreh_matmul_backward(
    const Tensor &output_grad,
    const Tensor &input,
    const Tensor &other,
    std::optional<const Tensor> input_grad = std::nullopt,
    std::optional<const Tensor> other_grad = std::nullopt,
    std::optional<const Tensor> output_tensor = std::nullopt,
    const MemoryConfig &mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace primary

}  // namespace operations

}  // namespace tt
