// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Match tensor_shape.h's gate so production kernel builds do not see this table.
#if defined(ENABLE_LLK_ASSERT) || defined(DEBUG_PRINT_ENABLED)

#include "tensor_shape_coverage.h"

namespace ckernel::coverage
{

// No pack TensorShape coverage probes are currently defined.

constexpr bool is_tensor_shape_covered(const TensorShapeFunctionCoverage, const TensorShape&)
{
    return false;
}

} // namespace ckernel::coverage

#endif // defined(ENABLE_LLK_ASSERT) || defined(DEBUG_PRINT_ENABLED)
