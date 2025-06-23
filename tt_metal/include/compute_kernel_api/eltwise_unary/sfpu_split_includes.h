// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if SFPU_OP_ISINF_ISNAN_INCLUDE
#include "compute_kernel_api/eltwise_unary/isinf_isnan.h"
#endif

#if SFPU_OP_ERF_ERFC_INCLUDE
#include "compute_kernel_api/eltwise_unary/erf_erfc.h"
#endif

#if SFPU_OP_LOGICAL_NOT_NOTI_INCLUDE
#include "compute_kernel_api/eltwise_unary/logical_not_noti.h"
#endif

#if SFPU_OP_EXP_INCLUDE
#include "compute_kernel_api/eltwise_unary/exp.h"
#endif

#if SFPU_OP_GELU_INCLUDE
#include "compute_kernel_api/eltwise_unary/gelu.h"
#endif

#if SFPU_OP_SQRT_INCLUDE
#include "compute_kernel_api/eltwise_unary/sqrt.h"
#endif

#if SFPU_OP_RECIP_INCLUDE
#include "compute_kernel_api/eltwise_unary/recip.h"
#endif

#if SFPU_OP_RELU_FAMILY_INCLUDE
#include "compute_kernel_api/eltwise_unary/relu.h"
#endif

#if SFPU_OP_ELU_INCLUDE
#include "compute_kernel_api/eltwise_unary/elu.h"
#endif

#if SFPU_OP_I0_INCLUDE
#include "compute_kernel_api/eltwise_unary/i0.h"
#endif

#if SFPU_OP_I1_INCLUDE
#include "compute_kernel_api/eltwise_unary/i1.h"
#endif

#if SFPU_OP_ERFINV_INCLUDE
#include "compute_kernel_api/eltwise_unary/erfinv.h"
#endif

#if SFPU_OP_NEG_INCLUDE
#include "compute_kernel_api/eltwise_unary/negative.h"
#endif

#if SFPU_OP_TRIG_FAMILY_INCLUDE
#include "compute_kernel_api/eltwise_unary/trigonometry.h"
#endif

#if SFPU_OP_REVERSE_FAMILY_INCLUDE
#include "compute_kernel_api/eltwise_unary/reverseops.h"
#endif

#if SFPU_OP_IDENTITY_INCLUDE
#include "compute_kernel_api/eltwise_unary/identity.h"
#endif

#if SFPU_OP_TYPECAST_INCLUDE
#include "compute_kernel_api/eltwise_unary/typecast.h"
#endif

#if SFPU_OP_BITWISE_XOR_INCLUDE
#include "compute_kernel_api/eltwise_unary/bitwise_xor.h"
#endif

#if SFPU_OP_BITWISE_NOT_INCLUDE
#include "compute_kernel_api/eltwise_unary/bitwise_not.h"
#endif

#if SFPU_OP_RIGHT_SHIFT_INCLUDE
#include "compute_kernel_api/eltwise_unary/right_shift.h"
#endif

#if SFPU_OP_BITWISE_AND_INCLUDE
#include "compute_kernel_api/eltwise_unary/bitwise_and.h"
#endif

#if SFPU_OP_BITWISE_OR_INCLUDE
#include "compute_kernel_api/eltwise_unary/bitwise_or.h"
#endif

#if SFPU_OP_ROUND_FAMILY_INCLUDE
#include "compute_kernel_api/eltwise_unary/rounding.h"
#endif

#if SFPU_OP_LEFT_SHIFT_INCLUDE
#include "compute_kernel_api/eltwise_unary/left_shift.h"
#endif

#if SFPU_OP_REMAINDER_INCLUDE
#include "compute_kernel_api/eltwise_unary/remainder.h"
#endif

#if SFPU_OP_FMOD_INCLUDE
#include "compute_kernel_api/eltwise_unary/fmod.h"
#endif

#if SFPU_OP_BINOP_WITH_SCALAR_INCLUDE
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#endif

#if SFPU_OP_SOFTPLUS_INCLUDE
#include "compute_kernel_api/eltwise_unary/softplus.h"
#endif

#if SFPU_OP_PRELU_INCLUDE
#include "compute_kernel_api/eltwise_unary/prelu.h"
#endif

#if SFPU_OP_DROPOUT_INCLUDE
#include "compute_kernel_api/eltwise_unary/dropout.h"
#endif

#if SFPU_OP_FILL_INCLUDE
#include "compute_kernel_api/eltwise_unary/fill.h"
#endif

#if SFPU_OP_LOG1P_INCLUDE
#include "compute_kernel_api/eltwise_unary/log1p.h"
#endif

#if SFPU_OP_UNARY_COMP_INCLUDE
#include "compute_kernel_api/eltwise_unary/comp.h"
#endif

#if SFPU_OP_COMPUTE_KERNEL_API_INCLUDE
#include "compute_kernel_api.h"
#endif
