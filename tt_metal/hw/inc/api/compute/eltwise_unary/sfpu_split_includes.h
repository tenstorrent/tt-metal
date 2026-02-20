// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if SFPU_OP_ISINF_ISNAN_INCLUDE
#include "api/compute/eltwise_unary/isinf_isnan.h"
#endif

#if SFPU_OP_ERF_ERFC_INCLUDE
#include "api/compute/eltwise_unary/erf_erfc.h"
#endif

#if SFPU_OP_LOGICAL_NOT_INCLUDE
#include "api/compute/eltwise_unary/logical_not.h"
#endif

#if SFPU_OP_EXP_INCLUDE
#include "api/compute/eltwise_unary/exp.h"
#endif

#if SFPU_OP_GELU_INCLUDE
#include "api/compute/eltwise_unary/gelu.h"
#endif

#if SFPU_OP_SQRT_INCLUDE
#include "api/compute/eltwise_unary/sqrt.h"
#endif

#if SFPU_OP_RSQRT_INCLUDE
#include "api/compute/eltwise_unary/rsqrt.h"
#endif

#if SFPU_OP_RECIP_INCLUDE
#include "api/compute/eltwise_unary/recip.h"
#endif

#if SFPU_OP_RELU_FAMILY_INCLUDE
#include "api/compute/eltwise_unary/relu.h"
#endif

#if SFPU_OP_ELU_INCLUDE
#include "api/compute/eltwise_unary/elu.h"
#endif

#if SFPU_OP_I0_INCLUDE
#include "api/compute/eltwise_unary/i0.h"
#endif

#if SFPU_OP_I1_INCLUDE
#include "api/compute/eltwise_unary/i1.h"
#endif

#if SFPU_OP_ERFINV_INCLUDE
#include "api/compute/eltwise_unary/erfinv.h"
#endif

#if SFPU_OP_NEG_INCLUDE
#include "api/compute/eltwise_unary/negative.h"
#endif

#if SFPU_OP_TRIG_FAMILY_INCLUDE
#include "api/compute/eltwise_unary/trigonometry.h"
#endif

#if SFPU_OP_RSUB_INCLUDE
#include "api/compute/eltwise_unary/rsub.h"
#endif

#if SFPU_OP_IDENTITY_INCLUDE
#include "api/compute/eltwise_unary/identity.h"
#endif

#if SFPU_OP_TYPECAST_INCLUDE
#include "api/compute/eltwise_unary/typecast.h"
#endif

#if SFPU_OP_BITWISE_XOR_INCLUDE
#include "api/compute/eltwise_unary/bitwise_xor.h"
#endif

#if SFPU_OP_BITWISE_NOT_INCLUDE
#include "api/compute/eltwise_unary/bitwise_not.h"
#endif

#if SFPU_OP_RIGHT_SHIFT_INCLUDE
#include "api/compute/eltwise_unary/right_shift.h"
#endif

#if SFPU_OP_BITWISE_AND_INCLUDE
#include "api/compute/eltwise_unary/bitwise_and.h"
#endif

#if SFPU_OP_BITWISE_OR_INCLUDE
#include "api/compute/eltwise_unary/bitwise_or.h"
#endif

#if SFPU_OP_ROUND_FAMILY_INCLUDE
#include "api/compute/eltwise_unary/rounding.h"
#endif

#if SFPU_OP_LEFT_SHIFT_INCLUDE
#include "api/compute/eltwise_unary/left_shift.h"
#endif

#if SFPU_OP_REMAINDER_INCLUDE
#include "api/compute/eltwise_unary/remainder.h"
#endif

#if SFPU_OP_FMOD_INCLUDE
#include "api/compute/eltwise_unary/fmod.h"
#endif

#if SFPU_OP_BINOP_WITH_SCALAR_INCLUDE
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#endif

#if SFPU_OP_SOFTPLUS_INCLUDE
#include "api/compute/eltwise_unary/softplus.h"
#endif

#if SFPU_OP_LOGSIGMOID_INCLUDE
#include "api/compute/logsigmoid.h"
#endif

#if SFPU_OP_SELU_INCLUDE
#include "api/compute/eltwise_unary/selu.h"
#endif

#if SFPU_OP_PRELU_INCLUDE
#include "api/compute/eltwise_unary/prelu.h"
#endif

#if SFPU_OP_DROPOUT_INCLUDE
#include "api/compute/eltwise_unary/dropout.h"
#endif

#if SFPU_OP_FILL_INCLUDE
#include "api/compute/eltwise_unary/fill.h"
#endif

#if SFPU_OP_LOG1P_INCLUDE
#include "api/compute/eltwise_unary/log1p.h"
#endif

#if SFPU_OP_UNARY_COMP_INCLUDE
#include "api/compute/eltwise_unary/comp.h"
#endif

#if SFPU_OP_ACTIVATIONS_INCLUDE
#include "api/compute/eltwise_unary/activations.h"
#endif

#if SFPU_OP_THRESHOLD_INCLUDE
#include "api/compute/eltwise_unary/threshold.h"
#endif

#if SFPU_OP_WHERE_INCLUDE
#include "api/compute/eltwise_unary/where.h"
#endif

#if SFPU_OP_CLAMP_INCLUDE
#include "api/compute/eltwise_unary/clamp.h"
#endif

#if SFPU_OP_HARDTANH_INCLUDE
#include "api/compute/eltwise_unary/hardtanh.h"
#endif

#if SFPU_OP_RPOW_INCLUDE
#include "api/compute/eltwise_unary/rpow.h"
#endif

#if SFPU_OP_HARDMISH_INCLUDE
#include "api/compute/eltwise_unary/hardmish.h"
#endif

#if SFPU_OP_COMPUTE_KERNEL_API_INCLUDE
#include "api/compute/compute_kernel_api.h"
#endif
