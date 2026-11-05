// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/sentinel/compute_kernel_sentinel.h"
#include "api/compute/tile_move_copy.h"
#ifdef TRISC_MATH
#include "llk_math_unary_datacopy_api.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_A_api.h"
#endif

// The eltwise-unary short init is the copy short init: compute_kernel_hw_startup(icb, ocb) once as the
// first Compute API call in MAIN, then copy_init(icb) (tile_move_copy.h) before the datacopy that feeds
// the SFPU op. This header keeps those includes for the SFPU op headers that include it.
