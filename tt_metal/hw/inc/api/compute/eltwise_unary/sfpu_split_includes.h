// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if SFPU_OP_HARDSIGMOID_INCLUDE
#include "api/compute/eltwise_unary/hardsigmoid.h"
#endif

#if SFPU_OP_HARDTANH_INCLUDE
#include "api/compute/eltwise_unary/hardtanh.h"
#endif

#if SFPU_OP_HARDSWISH_INCLUDE
#include "api/compute/eltwise_unary/hardswish.h"
#endif

#if SFPU_OP_SOFTSHRINK_INCLUDE
#include "api/compute/eltwise_unary/softshrink.h"
#endif

#if SFPU_OP_SWISH_INCLUDE
#include "api/compute/eltwise_unary/swish.h"
#endif

#if SFPU_OP_FRAC_INCLUDE
#include "api/compute/eltwise_unary/frac.h"
#endif

#if SFPU_OP_ATANH_INCLUDE
#include "api/compute/eltwise_unary/atanh.h"
#endif
