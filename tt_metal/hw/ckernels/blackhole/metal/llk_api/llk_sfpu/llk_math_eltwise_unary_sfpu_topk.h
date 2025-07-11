// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_topk.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"

namespace ckernel {

// New LLK SFPU APIs

SFPU_INIT_ONLY_WITH_TYPE(topk, topk_local_sort, sfpu::topk_init)

SFPU_TOPK_LOCAL_SORT_KERNEL(topk_local_sort)

SFPU_TOPK_MERGE_KERNEL(topk_merge)

SFPU_TOPK_REBUILD_KERNEL(topk_rebuild)

}  // namespace ckernel
