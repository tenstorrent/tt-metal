// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "tt_eager/tensor/tensor.hpp"
#include "tt_dnn/op_library/run_operation.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

/**
Ref: https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html
>>> input = torch.randn(20, 6, 10, 10)
>>> # Separate 6 channels into 3 groups
>>> m = nn.GroupNorm(3, 6)
>>> # Separate 6 channels into 6 groups (equivalent with InstanceNorm)
>>> m = nn.GroupNorm(6, 6)
>>> # Put all 6 channels into a single group (equivalent with LayerNorm)
>>> m = nn.GroupNorm(1, 6)
>>> # Activating the module
>>> output = m(input)
*/

// FIXME: special case for group_size = 1 is only supported at this time
Tensor groupnorm(
    const Tensor& a,
    uint32_t group_size,
    float eps,
    std::optional<const Tensor> gamma = std::nullopt,
    std::optional<const Tensor> beta = std::nullopt,
    const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace tt_metal
}  // namespace tt
