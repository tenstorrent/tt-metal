// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if SFPU_OP_COSH_INCLUDE
#include "api/compute/eltwise_unary/cosh.h"
#if SFPU_OP_CBRT_INCLUDE
#include "api/compute/eltwise_unary/cbrt.h"
#if SFPU_OP_SELU_INCLUDE
#include "api/compute/eltwise_unary/selu.h"
#endif
