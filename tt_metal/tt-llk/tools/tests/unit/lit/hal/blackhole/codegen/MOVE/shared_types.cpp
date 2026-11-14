// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %split-file %s %t
// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -UENABLE_LLK_ASSERT -fno-ipa-icf -c %t/codegen.cpp -o %t/codegen.o
// RUN: %{blackhole_objdump} -drz %t/codegen.o > %t/codegen.dump
// RUN: %{blackhole_compare_codegen} %t/codegen.dump
// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -UENABLE_LLK_ASSERT -c %t/compile_time.cpp -o %t/compile_time.o
// RUN: %{blackhole_objdump} -d %t/compile_time.o | FileCheck %s
// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -DENABLE_LLK_ASSERT -c %t/asserts.cpp -o %t/assert.o
// RUN: %{blackhole_objdump} -d %t/assert.o | FileCheck %s --check-prefix=ASSERT

// Compare each runtime HAL function with its adjacent reference_ function.
// Disable identical-code folding so both bodies are emitted independently.
// codegen.cpp is compiled without LLK_ASSERT so runtime paths contain only the
// computed value; asserts.cpp checks the runtime operand asserts.

//--- common.h
#pragma once

#include <cstdint>

#include "hal/move.h"

namespace move = hal::move;

//--- codegen.cpp
#include "common.h"

// A Dst tile spans 4 faces of 16 rows.

extern "C" __attribute__((noinline, used)) std::uint32_t tile_row_runtime(const std::uint32_t tile_index, const std::uint32_t face, const std::uint32_t row)
{
    return move::dst_layout::tile_row(tile_index, face, row);
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_tile_row_runtime(
    const std::uint32_t tile_index, const std::uint32_t face, const std::uint32_t row)
{
    return tile_index * 64u + face * 16u + row;
}

// A counter-relative row is the ten-bit two's-complement offset.

extern "C" __attribute__((noinline, used)) std::uint32_t relative_row_runtime(const std::int32_t offset)
{
    return move::relative_row(offset);
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_relative_row_runtime(const std::int32_t offset)
{
    return static_cast<std::uint32_t>(offset) & 0x3ffu;
}

//--- compile_time.cpp
#include "common.h"

extern "C" __attribute__((noinline, used)) std::uint32_t constant_tile_row()
{
    return move::dst_layout::tile_row(3, 2, 5);
}

// CHECK-LABEL: <constant_tile_row>:
// CHECK-NEXT: li a0,229{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) std::uint32_t constant_relative_row()
{
    return move::relative_row<-1>();
}

// CHECK-LABEL: <constant_relative_row>:
// CHECK-NEXT: li a0,1023{{$}}
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) std::uint32_t constant_faces()
{
    return move::rows::faces<4>().value;
}

// CHECK-LABEL: <constant_faces>:
// CHECK-NEXT: li a0,64{{$}}
// CHECK-NEXT: ret

//--- asserts.cpp
#include "common.h"

// Out-of-range offsets must trap before the row is computed; the signed
// ten-bit boundaries must not trap.

extern "C" std::uint32_t reject_relative_row_above_range()
{
    return move::relative_row(512);
}

// ASSERT-LABEL: <reject_relative_row_above_range>:
// ASSERT-NOT: li a0
// ASSERT: ebreak
// ASSERT: ret

extern "C" std::uint32_t reject_relative_row_below_range()
{
    return move::relative_row(-513);
}

// ASSERT-LABEL: <reject_relative_row_below_range>:
// ASSERT-NOT: li a0
// ASSERT: ebreak
// ASSERT: ret

extern "C" std::uint32_t accept_relative_row_boundaries()
{
    return move::relative_row(511) + move::relative_row(-512);
}

// ASSERT-LABEL: <accept_relative_row_boundaries>:
// ASSERT-NOT: ebreak
// ASSERT: ret
