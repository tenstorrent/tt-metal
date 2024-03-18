/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/operation.hpp"

namespace tt {
namespace operations {
namespace primary {

using namespace tt_metal;

[[maybe_unused]] std::vector<std::variant<Tensor, char *>> moreh_bmm_backward(
    const Tensor &output_grad,
    const Tensor &input,
    const Tensor &mat2,
    std::optional<Tensor> input_grad = std::nullopt,
    std::optional<Tensor> mat2_grad = std::nullopt,
    std::optional<Tensor> output_tensor = std::nullopt,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace primary

}  // namespace operations

}  // namespace tt
