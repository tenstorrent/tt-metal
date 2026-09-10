// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef SFPU_OP_ISINF_ISNAN_INCLUDE
#include "api/compute/eltwise_unary/isinf_isnan.h"
#endif

#ifdef SFPU_OP_ERF_ERFC_INCLUDE
#include "api/compute/eltwise_unary/erf_erfc.h"
#endif

#ifdef SFPU_OP_LOGICAL_NOT_INCLUDE
#include "api/compute/eltwise_unary/logical_not.h"
#endif

#ifdef SFPU_OP_EXP_INCLUDE
#include "api/compute/eltwise_unary/exp.h"
#endif

#ifdef SFPU_OP_GELU_INCLUDE
#include "api/compute/eltwise_unary/gelu.h"
#endif

#ifdef SFPU_OP_SQRT_INCLUDE
#include "api/compute/eltwise_unary/sqrt.h"
#endif

#ifdef SFPU_OP_RSQRT_INCLUDE
#include "api/compute/eltwise_unary/rsqrt.h"
#endif

#ifdef SFPU_OP_RECIP_INCLUDE
#include "api/compute/eltwise_unary/recip.h"
#endif

#ifdef SFPU_OP_CBRT_INCLUDE
#include "api/compute/eltwise_unary/cbrt.h"
#endif

#ifdef SFPU_OP_RELU_FAMILY_INCLUDE
#include "api/compute/eltwise_unary/relu.h"
#endif

#ifdef SFPU_OP_ELU_INCLUDE
#include "api/compute/eltwise_unary/elu.h"
#endif

#ifdef SFPU_OP_I0_INCLUDE
#include "api/compute/eltwise_unary/i0.h"
#endif

#ifdef SFPU_OP_I1_INCLUDE
#include "api/compute/eltwise_unary/i1.h"
#endif

#ifdef SFPU_OP_ERFINV_INCLUDE
#include "api/compute/eltwise_unary/erfinv.h"
#endif

#ifdef SFPU_OP_NEG_INCLUDE
#include "api/compute/eltwise_unary/negative.h"
#endif

#ifdef SFPU_OP_TRIG_FAMILY_INCLUDE
#include "api/compute/eltwise_unary/trigonometry.h"
#endif

#ifdef SFPU_OP_RSUB_INCLUDE
#include "api/compute/eltwise_unary/rsub.h"
#endif

#ifdef SFPU_OP_IDENTITY_INCLUDE
#include "api/compute/eltwise_unary/identity.h"
#endif

#ifdef SFPU_OP_TYPECAST_INCLUDE
#include "api/compute/eltwise_unary/typecast.h"
#endif

#ifdef SFPU_OP_BITWISE_INCLUDE
#include "api/compute/eltwise_unary/bitwise.h"
#endif

#ifdef SFPU_OP_BITWISE_NOT_INCLUDE
#include "api/compute/eltwise_unary/bitwise_not.h"
#endif

#ifdef SFPU_OP_SHIFT_INCLUDE
#include "api/compute/eltwise_unary/shift.h"
#endif

#ifdef SFPU_OP_ROUND_FAMILY_INCLUDE
#include "api/compute/eltwise_unary/rounding.h"
#endif

#ifdef SFPU_OP_REMAINDER_INCLUDE
#include "api/compute/eltwise_unary/remainder.h"
#endif

#ifdef SFPU_OP_FMOD_INCLUDE
#include "api/compute/eltwise_unary/fmod.h"
#endif

#ifdef SFPU_OP_BINOP_WITH_SCALAR_INCLUDE
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#endif

#ifdef SFPU_OP_SOFTPLUS_INCLUDE
#include "api/compute/eltwise_unary/softplus.h"
#endif

#ifdef SFPU_OP_XIELU_INCLUDE
#include "api/compute/eltwise_unary/xielu.h"
#endif

#ifdef SFPU_OP_LOGSIGMOID_INCLUDE
#include "api/compute/logsigmoid.h"
#endif

#ifdef SFPU_OP_SELU_INCLUDE
#include "api/compute/eltwise_unary/selu.h"
#endif

#ifdef SFPU_OP_PRELU_INCLUDE
#include "api/compute/eltwise_unary/prelu.h"
#endif

#ifdef SFPU_OP_DROPOUT_INCLUDE
#include "api/compute/eltwise_unary/dropout.h"
#endif

#ifdef SFPU_OP_FILL_INCLUDE
#include "api/compute/eltwise_unary/fill.h"
#endif

#ifdef SFPU_OP_LOG1P_INCLUDE
#include "api/compute/eltwise_unary/log1p.h"
#endif

#ifdef SFPU_OP_UNARY_COMP_INCLUDE
#include "api/compute/eltwise_unary/comp.h"
#endif

#ifdef SFPU_OP_ACTIVATIONS_INCLUDE
#include "api/compute/eltwise_unary/activations.h"
#endif

#ifdef SFPU_OP_THRESHOLD_INCLUDE
#include "api/compute/eltwise_unary/threshold.h"
#endif

#ifdef SFPU_OP_WHERE_INCLUDE
#include "api/compute/eltwise_unary/where.h"
#endif

#ifdef SFPU_OP_MAC_INCLUDE
#include "api/compute/eltwise_unary/mac.h"
#endif

#ifdef SFPU_OP_CLAMP_INCLUDE
#include "api/compute/eltwise_unary/clamp.h"
#endif

#ifdef SFPU_OP_HARDTANH_INCLUDE
#include "api/compute/eltwise_unary/hardtanh.h"
#endif

#ifdef SFPU_OP_RPOW_INCLUDE
#include "api/compute/eltwise_unary/rpow.h"
#endif

#ifdef SFPU_OP_HARDMISH_INCLUDE
#include "api/compute/eltwise_unary/hardmish.h"
#endif

#ifdef SFPU_OP_SOFTCAP_INCLUDE
#include "api/compute/eltwise_unary/softcap.h"
#endif

#ifdef SFPU_OP_LGAMMA_INCLUDE
#include "api/compute/eltwise_unary/lgamma.h"
#endif

#ifdef SFPU_OP_DIGAMMA_INCLUDE
#include "api/compute/eltwise_unary/digamma.h"
#endif

#ifdef SFPU_OP_TANHSHRINK_INCLUDE
#include "api/compute/eltwise_unary/tanhshrink.h"
#endif

#ifdef SFPU_OP_POLYGAMMA_INCLUDE
#include "api/compute/eltwise_unary/polygamma.h"
#endif

#ifdef SFPU_OP_MISH_INCLUDE
#include "api/compute/eltwise_unary/mish.h"
#endif

#ifdef SFPU_OP_COMPUTE_KERNEL_API_INCLUDE
#include "api/compute/compute_kernel_api.h"
#endif

#ifdef SFPU_OP_BINARY_DIV_INCLUDE
#include "api/compute/eltwise_binary_sfpu.h"
#endif

#ifdef SFPU_OP_BINARY_ATAN2_INCLUDE
#include "api/compute/atan2.h"
#endif

#ifdef SFPU_OP_BINARY_ADD_INT_INCLUDE
#include "api/compute/add_int_sfpu.h"
#endif

#ifdef SFPU_OP_BINARY_MUL_INT_INCLUDE
#include "api/compute/mul_int_sfpu.h"
#endif

#ifdef SFPU_OP_BINARY_GT_INT_INCLUDE
#include "api/compute/binary_comp.h"
#endif

#ifdef SFPU_OP_BINARY_MAX_MIN_INCLUDE
#include "api/compute/binary_max_min.h"
#endif
