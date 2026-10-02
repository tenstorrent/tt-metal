// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
// RUN: %split-file %s %t
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/reserved-immediate.cpp 2>&1 | FileCheck %s --check-prefix=RESERVED_IMMEDIATE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/immediate-too-wide.cpp 2>&1 | FileCheck %s --check-prefix=IMMEDIATE_TOO_WIDE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/runtime-gpr-immediate-too-wide.cpp 2>&1 | FileCheck %s --check-prefix=RUNTIME_GPR_IMMEDIATE_TOO_WIDE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/destination-out-of-range.cpp 2>&1 | FileCheck %s --check-prefix=DESTINATION_OUT_OF_RANGE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/lhs-out-of-range.cpp 2>&1 | FileCheck %s --check-prefix=LHS_OUT_OF_RANGE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/rhs-out-of-range.cpp 2>&1 | FileCheck %s --check-prefix=RHS_OUT_OF_RANGE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/runtime-mixed-gpr-out-of-range.cpp 2>&1 | FileCheck %s --check-prefix=RUNTIME_MIXED_GPR_OUT_OF_RANGE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/set-out-of-range.cpp 2>&1 | FileCheck %s --check-prefix=SET_OUT_OF_RANGE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/half-too-wide.cpp 2>&1 | FileCheck %s --check-prefix=HALF_TOO_WIDE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/half-operation-too-wide.cpp 2>&1 | FileCheck %s --check-prefix=HALF_OPERATION_TOO_WIDE
// clang-format on

//--- reserved-immediate.cpp
#include "hal/gpr_ops.h"

namespace gpr_ops = hal::gpr_ops;

constexpr auto reserved = gpr_ops::immediate<0xffffffffu>();
// RESERVED_IMMEDIATE: error: static assertion failed: Immediate value is reserved by hal::gpr_ops::immediate()

//--- immediate-too-wide.cpp
#include "hal/gpr_ops.h"

namespace gpr_ops = hal::gpr_ops;

void immediate_too_wide()
{
    gpr_ops::add(hal::gpr<1>(), hal::gpr<2>(), gpr_ops::immediate<64>());
}

// IMMEDIATE_TOO_WIDE: error: static assertion failed: Scalar-unit immediate must fit in six bits

//--- runtime-gpr-immediate-too-wide.cpp
#include <cstdint>

#include "hal/gpr_ops.h"

namespace gpr_ops = hal::gpr_ops;

void runtime_gpr_immediate_too_wide(const std::uint32_t index)
{
    gpr_ops::bit_and(hal::gpr(index), hal::gpr<2>(), gpr_ops::immediate<64>());
}

// RUNTIME_GPR_IMMEDIATE_TOO_WIDE: error: static assertion failed: Scalar-unit immediate must fit in six bits

//--- destination-out-of-range.cpp
#include "hal/gpr_ops.h"

namespace gpr_ops = hal::gpr_ops;

void destination_out_of_range()
{
    gpr_ops::subtract(hal::gpr<64>(), hal::gpr<2>(), hal::gpr<3>());
}

// DESTINATION_OUT_OF_RANGE: error: static assertion failed: Scalar-unit destination GPR index must be in [0, 63]

//--- lhs-out-of-range.cpp
#include "hal/gpr_ops.h"

namespace gpr_ops = hal::gpr_ops;

void lhs_out_of_range()
{
    gpr_ops::shift_left(hal::gpr<1>(), hal::gpr<64>(), gpr_ops::immediate<1>());
}

// LHS_OUT_OF_RANGE: error: static assertion failed: Scalar-unit lhs GPR index must be in [0, 63]

//--- rhs-out-of-range.cpp
#include "hal/gpr_ops.h"

namespace gpr_ops = hal::gpr_ops;

void rhs_out_of_range()
{
    gpr_ops::compare<gpr_ops::Compare::LessThan>(hal::gpr<1>(), hal::gpr<2>(), hal::gpr<64>());
}

// RHS_OUT_OF_RANGE: error: static assertion failed: Scalar-unit rhs GPR index must be in [0, 63]

//--- runtime-mixed-gpr-out-of-range.cpp
#include <cstdint>

#include "hal/gpr_ops.h"

namespace gpr_ops = hal::gpr_ops;

void runtime_mixed_gpr_out_of_range(const std::uint32_t value)
{
    gpr_ops::add(hal::gpr<64>(), hal::gpr<2>(), gpr_ops::immediate(value));
}

// RUNTIME_MIXED_GPR_OUT_OF_RANGE: error: static assertion failed: Scalar-unit GPR index must be in [0, 63]

//--- set-out-of-range.cpp
#include "hal/gpr_ops.h"

namespace gpr_ops = hal::gpr_ops;

void set_out_of_range()
{
    gpr_ops::set<1>(hal::gpr<64>());
}

// SET_OUT_OF_RANGE: error: static assertion failed: Scalar-unit GPR index must be in [0, 63]

//--- half-too-wide.cpp
#include "hal/gpr_ops.h"

namespace gpr_ops = hal::gpr_ops;

void half_too_wide()
{
    gpr_ops::set_high<0x10000>(hal::gpr<1>());
}

// HALF_TOO_WIDE: error: static assertion failed: GPR half value must fit in 16 bits

//--- half-operation-too-wide.cpp
#include "hal/gpr_ops.h"

namespace gpr_ops = hal::gpr_ops;

constexpr auto too_wide = gpr_ops::set_low_operation<0x10000>(hal::gpr<1>());
// HALF_OPERATION_TOO_WIDE: error: static assertion failed: GPR half value must fit in 16 bits
