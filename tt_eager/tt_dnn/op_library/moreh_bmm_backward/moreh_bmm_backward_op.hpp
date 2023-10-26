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

[[maybe_unused]] std::vector<std::variant<tt_metal::Tensor, char *>> moreh_bmm_backward(
    const tt_metal::Tensor &output_grad,
    const tt_metal::Tensor &input,
    const tt_metal::Tensor &mat2,
    std::optional<std::reference_wrapper<const tt_metal::Tensor>> input_grad = std::nullopt,
    std::optional<std::reference_wrapper<const tt_metal::Tensor>> mat2_grad = std::nullopt,
    const tt_metal::MemoryConfig &output_mem_config = tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace primary

}  // namespace operations

}  // namespace tt
