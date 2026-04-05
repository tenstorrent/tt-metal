// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if SFPU_OP_COSH_INCLUDE
#include "api/compute/eltwise_unary/cosh.h"
#endif

#if SFPU_OP_CBRT_INCLUDE
#include "api/compute/eltwise_unary/cbrt.h"
#endif

#if SFPU_OP_HARDSIGMOID_INCLUDE
#include "api/compute/eltwise_unary/hardsigmoid.h"
#endif

#if SFPU_OP_SELU_INCLUDE
#include "api/compute/eltwise_unary/selu.h"
#endif

#if SFPU_OP_HARDTANH_INCLUDE
#include "api/compute/eltwise_unary/hardtanh.h"
#endif

#if SFPU_OP_SOFTSIGN_INCLUDE
#include "api/compute/eltwise_unary/softsign.h"
#endif

#if SFPU_OP_LGAMMA_INCLUDE
#include "api/compute/eltwise_unary/lgamma.h"
#endif

#if SFPU_OP_RPOW_INCLUDE
#include "api/compute/eltwise_unary/rpow.h"
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
