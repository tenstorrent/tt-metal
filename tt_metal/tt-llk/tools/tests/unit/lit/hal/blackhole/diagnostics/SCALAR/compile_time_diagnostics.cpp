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
#include "hal/scalar.h"

namespace scalar = hal::scalar;

constexpr auto reserved = scalar::immediate<0xffffffffu>();
// RESERVED_IMMEDIATE: error: static assertion failed: Immediate value is reserved by hal::scalar::immediate()

//--- immediate-too-wide.cpp
#include "hal/scalar.h"

namespace scalar = hal::scalar;

void immediate_too_wide()
{
    scalar::add(hal::gpr<1>(), hal::gpr<2>(), scalar::immediate<64>());
}

// IMMEDIATE_TOO_WIDE: error: static assertion failed: Scalar-unit immediate must fit in six bits

//--- runtime-gpr-immediate-too-wide.cpp
#include <cstdint>

#include "hal/scalar.h"

namespace scalar = hal::scalar;

void runtime_gpr_immediate_too_wide(const std::uint32_t index)
{
    scalar::bit_and(hal::gpr(index), hal::gpr<2>(), scalar::immediate<64>());
}

// RUNTIME_GPR_IMMEDIATE_TOO_WIDE: error: static assertion failed: Scalar-unit immediate must fit in six bits

//--- destination-out-of-range.cpp
#include "hal/scalar.h"

namespace scalar = hal::scalar;

void destination_out_of_range()
{
    scalar::subtract(hal::gpr<64>(), hal::gpr<2>(), hal::gpr<3>());
}

// DESTINATION_OUT_OF_RANGE: error: static assertion failed: Scalar-unit destination GPR index must be in [0, 63]

//--- lhs-out-of-range.cpp
#include "hal/scalar.h"

namespace scalar = hal::scalar;

void lhs_out_of_range()
{
    scalar::shift_left(hal::gpr<1>(), hal::gpr<64>(), scalar::immediate<1>());
}

// LHS_OUT_OF_RANGE: error: static assertion failed: Scalar-unit lhs GPR index must be in [0, 63]

//--- rhs-out-of-range.cpp
#include "hal/scalar.h"

namespace scalar = hal::scalar;

void rhs_out_of_range()
{
    scalar::compare<scalar::Compare::LessThan>(hal::gpr<1>(), hal::gpr<2>(), hal::gpr<64>());
}

// RHS_OUT_OF_RANGE: error: static assertion failed: Scalar-unit rhs GPR index must be in [0, 63]

//--- runtime-mixed-gpr-out-of-range.cpp
#include <cstdint>

#include "hal/scalar.h"

namespace scalar = hal::scalar;

void runtime_mixed_gpr_out_of_range(const std::uint32_t value)
{
    scalar::add(hal::gpr<64>(), hal::gpr<2>(), scalar::immediate(value));
}

// RUNTIME_MIXED_GPR_OUT_OF_RANGE: error: static assertion failed: Scalar-unit GPR index must be in [0, 63]

//--- set-out-of-range.cpp
#include "hal/scalar.h"

namespace scalar = hal::scalar;

void set_out_of_range()
{
    scalar::set<1>(hal::gpr<64>());
}

// SET_OUT_OF_RANGE: error: static assertion failed: Scalar-unit GPR index must be in [0, 63]

//--- half-too-wide.cpp
#include "hal/scalar.h"

namespace scalar = hal::scalar;

void half_too_wide()
{
    scalar::set_high<0x10000>(hal::gpr<1>());
}

// HALF_TOO_WIDE: error: static assertion failed: GPR half value must fit in 16 bits

//--- half-operation-too-wide.cpp
#include "hal/scalar.h"

namespace scalar = hal::scalar;

constexpr auto too_wide = scalar::SetLow {hal::gpr<1>(), scalar::immediate<0x10000>()}.operation();
// HALF_OPERATION_TOO_WIDE: error: static assertion failed: GPR half value must fit in 16 bits
