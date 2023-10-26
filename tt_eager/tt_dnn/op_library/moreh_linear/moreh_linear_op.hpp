/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <functional>

#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/operation.hpp"

namespace tt {
namespace operations {
namespace primary {

tt_metal::Tensor moreh_linear(
    const tt_metal::Tensor& input,
    const tt_metal::Tensor& weight,
    std::optional<std::reference_wrapper<const tt_metal::Tensor>> bias = std::nullopt,
    const tt_metal::MemoryConfig& output_mem_config = tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace primary

}  // namespace operations

}  // namespace tt
