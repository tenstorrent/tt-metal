// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
// RUN: %split-file %s %t
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/empty-selection.cpp 2>&1 | FileCheck %s --check-prefix=EMPTY_SELECTION
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/transpose-both.cpp 2>&1 | FileCheck %s --check-prefix=TRANSPOSE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/shift-columns-srcb.cpp 2>&1 | FileCheck %s --check-prefix=SHIFT_COLUMNS
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/shift-row-srca.cpp 2>&1 | FileCheck %s --check-prefix=SHIFT_ROW
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/shift-row-runtime-srca.cpp 2>&1 | FileCheck %s --check-prefix=SHIFT_ROW
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/shift-row-out-of-range.cpp 2>&1 | FileCheck %s --check-prefix=SHIFT_ROW_RANGE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/shift-row-address-mode.cpp 2>&1 | FileCheck %s --check-prefix=ADDRESS_MODE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/shift-row-runtime-address-mode.cpp 2>&1 | FileCheck %s --check-prefix=ADDRESS_MODE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/rarefy-srca.cpp 2>&1 | FileCheck %s --check-prefix=RAREFY
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/mask-srcb.cpp 2>&1 | FileCheck %s --check-prefix=MASK_TARGET
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/mask-horizontal-width.cpp 2>&1 | FileCheck %s --check-prefix=MASK_HORIZONTAL
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/mask-thread1-width.cpp 2>&1 | FileCheck %s --check-prefix=MASK_HORIZONTAL
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/mask-halo.cpp 2>&1 | FileCheck %s --check-prefix=MASK_HALO
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/mask-vertical-width.cpp 2>&1 | FileCheck %s --check-prefix=MASK_VERTICAL
// clang-format on

// Match an actual compiler error, not the echoed static_assert source text.
// EMPTY_SELECTION: error: static assertion failed: source selection must name SrcA, SrcB, or both
// TRANSPOSE: error: static assertion failed: transpose targets exactly one source register
// SHIFT_COLUMNS: error: static assertion failed: the combined column shift targets SrcA
// SHIFT_ROW: error: static assertion failed: the row shift targets SrcB
// SHIFT_ROW_RANGE: error: static assertion failed: SrcB row must be in [0, 63]
// ADDRESS_MODE: error: static assertion failed: Blackhole address mode must be in [0, 7]
// RAREFY: error: static assertion failed: rarefication targets SrcB
// MASK_TARGET: error: static assertion failed: the right-shift masks target SrcA
// MASK_HORIZONTAL: error: static assertion failed: horizontal right-shift register mask is 16 bits
// MASK_HALO: error: static assertion failed: halo mask is one bit
// MASK_VERTICAL: error: static assertion failed: vertical right-shift register mask is 20 bits

//--- empty-selection.cpp
#include "hal/src.h"

void f()
{
    hal::src<hal::SrcA & hal::SrcB>.fill();
}

//--- transpose-both.cpp
#include "hal/src.h"

void f()
{
    hal::src<hal::BothSources>.transpose();
}

//--- shift-columns-srcb.cpp
#include "hal/src.h"

void f()
{
    hal::src<hal::SrcB>.shift_columns<hal::src_ops::ShiftDirection::TowardColumn0>();
}

//--- shift-row-srca.cpp
#include "hal/src.h"

void f()
{
    hal::src<hal::SrcA>.shift_row<0>();
}

//--- shift-row-runtime-srca.cpp
#include "hal/src.h"

void f(unsigned row)
{
    hal::src<hal::SrcA>.shift_row(row);
}

//--- shift-row-out-of-range.cpp
#include "hal/src.h"

void f()
{
    hal::src<hal::SrcB>.shift_row<64>();
}

//--- shift-row-address-mode.cpp
#include "hal/src.h"

void f()
{
    hal::src<hal::SrcB>.shift_row<0, hal::src_ops::ShiftFill::Zero, 8>();
}

//--- shift-row-runtime-address-mode.cpp
#include "hal/src.h"

void f(unsigned row)
{
    hal::src<hal::SrcB>.shift_row<hal::src_ops::ShiftFill::Zero, 8>(row);
}

//--- rarefy-srca.cpp
#include "hal/src.h"

void f()
{
    hal::src<hal::SrcA>.rarefy();
}

//--- mask-srcb.cpp
#include "hal/src.h"

void f()
{
    hal::src<hal::SrcB>.set_right_shift_mask_vertical<1>();
}

//--- mask-horizontal-width.cpp
#include "hal/src.h"

void f()
{
    hal::src<hal::SrcA>.set_right_shift_mask_horizontal<0x10000>();
}

//--- mask-thread1-width.cpp
#include "hal/src.h"

void f()
{
    hal::src<hal::SrcA>.set_right_shift_mask_horizontal_thread1<0x10000>();
}

//--- mask-halo.cpp
#include "hal/src.h"

void f()
{
    hal::src<hal::SrcA>.set_right_shift_mask_horizontal_thread0<1, 2>();
}

//--- mask-vertical-width.cpp
#include "hal/src.h"

void f()
{
    hal::src<hal::SrcA>.set_right_shift_mask_vertical<0x100000>();
}
