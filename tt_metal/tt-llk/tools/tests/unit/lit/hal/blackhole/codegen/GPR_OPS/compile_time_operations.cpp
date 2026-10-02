// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -c %s -o %t.o
// RUN: %{blackhole_objdump} -d %t.o | FileCheck %s

#include <cstdint>

#include "hal/gpr_ops.h"

namespace gpr_ops = hal::gpr_ops;

// Objdump splits the 16-bit SETDMAREG payload: 0x5678 prints as 1,5752 and 0x1234 as 0,4660.
// GPR N owns 16-bit halves 2N (low) and 2N+1 (high).
extern "C" __attribute__((noinline, used)) void set_word()
{
    gpr_ops::set<0x12345678>(hal::gpr<5>());
}

// CHECK-LABEL: <set_word>:
// CHECK-NEXT: ttsetdmareg 1,5752,0,10
// CHECK-NEXT: ttsetdmareg 0,4660,0,11
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void set_halves()
{
    gpr_ops::set_low<0x5678>(hal::gpr<63>());
    gpr_ops::set_high<0x1234>(hal::gpr<63>());
}

// CHECK-LABEL: <set_halves>:
// CHECK-NEXT: ttsetdmareg 1,5752,0,126
// CHECK-NEXT: ttsetdmareg 0,4660,0,127
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void add_gpr_and_immediate()
{
    gpr_ops::add(hal::gpr<1>(), hal::gpr<2>(), hal::gpr<3>());
    gpr_ops::add(hal::gpr<1>(), hal::gpr<2>(), gpr_ops::immediate<63>());
}

// CHECK-LABEL: <add_gpr_and_immediate>:
// CHECK-NEXT: ttadddmareg 0,1,3,2
// CHECK-NEXT: ttadddmareg 1,1,63,2
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void subtract_gpr_and_immediate()
{
    gpr_ops::subtract(hal::gpr<4>(), hal::gpr<5>(), hal::gpr<6>());
    gpr_ops::subtract(hal::gpr<4>(), hal::gpr<5>(), gpr_ops::immediate<7>());
}

// CHECK-LABEL: <subtract_gpr_and_immediate>:
// CHECK-NEXT: ttsubdmareg 0,4,6,5
// CHECK-NEXT: ttsubdmareg 1,4,7,5
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void multiply_low_u16_gpr_and_immediate()
{
    gpr_ops::multiply_low_u16(hal::gpr<7>(), hal::gpr<8>(), hal::gpr<9>());
    gpr_ops::multiply_low_u16(hal::gpr<7>(), hal::gpr<8>(), gpr_ops::immediate<10>());
}

// CHECK-LABEL: <multiply_low_u16_gpr_and_immediate>:
// CHECK-NEXT: ttmuldmareg 0,7,9,8
// CHECK-NEXT: ttmuldmareg 1,7,10,8
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void bitwise_gpr_and_immediate()
{
    gpr_ops::bit_and(hal::gpr<10>(), hal::gpr<11>(), hal::gpr<12>());
    gpr_ops::bit_and(hal::gpr<10>(), hal::gpr<11>(), gpr_ops::immediate<1>());
    gpr_ops::bit_or(hal::gpr<13>(), hal::gpr<14>(), hal::gpr<15>());
    gpr_ops::bit_or(hal::gpr<13>(), hal::gpr<14>(), gpr_ops::immediate<2>());
    gpr_ops::bit_xor(hal::gpr<16>(), hal::gpr<17>(), hal::gpr<18>());
    gpr_ops::bit_xor(hal::gpr<16>(), hal::gpr<17>(), gpr_ops::immediate<3>());
}

// CHECK-LABEL: <bitwise_gpr_and_immediate>:
// CHECK-NEXT: ttbitwopdmareg 0,0,10,12,11
// CHECK-NEXT: ttbitwopdmareg 1,0,10,1,11
// CHECK-NEXT: ttbitwopdmareg 0,1,13,15,14
// CHECK-NEXT: ttbitwopdmareg 1,1,13,2,14
// CHECK-NEXT: ttbitwopdmareg 0,2,16,18,17
// CHECK-NEXT: ttbitwopdmareg 1,2,16,3,17
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void shift_gpr_and_immediate()
{
    gpr_ops::shift_left(hal::gpr<19>(), hal::gpr<20>(), hal::gpr<21>());
    gpr_ops::shift_left(hal::gpr<19>(), hal::gpr<20>(), gpr_ops::immediate<4>());
    gpr_ops::shift_right(hal::gpr<22>(), hal::gpr<23>(), hal::gpr<24>());
    gpr_ops::shift_right(hal::gpr<22>(), hal::gpr<23>(), gpr_ops::immediate<5>());
}

// CHECK-LABEL: <shift_gpr_and_immediate>:
// CHECK-NEXT: ttshiftdmareg 0,0,19,21,20
// CHECK-NEXT: ttshiftdmareg 1,0,19,4,20
// CHECK-NEXT: ttshiftdmareg 0,1,22,24,23
// CHECK-NEXT: ttshiftdmareg 1,1,22,5,23
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void compare_modes()
{
    gpr_ops::compare<gpr_ops::Compare::GreaterThan>(hal::gpr<25>(), hal::gpr<26>(), hal::gpr<27>());
    gpr_ops::compare<gpr_ops::Compare::GreaterThan>(hal::gpr<25>(), hal::gpr<26>(), gpr_ops::immediate<6>());
    gpr_ops::compare<gpr_ops::Compare::LessThan>(hal::gpr<28>(), hal::gpr<29>(), hal::gpr<30>());
    gpr_ops::compare<gpr_ops::Compare::LessThan>(hal::gpr<28>(), hal::gpr<29>(), gpr_ops::immediate<7>());
    gpr_ops::compare<gpr_ops::Compare::Equal>(hal::gpr<31>(), hal::gpr<32>(), hal::gpr<33>());
    gpr_ops::compare<gpr_ops::Compare::Equal>(hal::gpr<31>(), hal::gpr<32>(), gpr_ops::immediate<8>());
}

// CHECK-LABEL: <compare_modes>:
// CHECK-NEXT: ttcmpdmareg 0,0,25,27,26
// CHECK-NEXT: ttcmpdmareg 1,0,25,6,26
// CHECK-NEXT: ttcmpdmareg 0,1,28,30,29
// CHECK-NEXT: ttcmpdmareg 1,1,28,7,29
// CHECK-NEXT: ttcmpdmareg 0,2,31,33,32
// CHECK-NEXT: ttcmpdmareg 1,2,31,8,32
// CHECK-NEXT: ret

// The *_operation encoders return the same word the issuing calls emit.
extern "C" __attribute__((noinline, used)) void issue_encoded_operations()
{
    constexpr std::uint32_t low      = gpr_ops::set_low_operation<0x5678>(hal::gpr<63>());
    constexpr std::uint32_t high     = gpr_ops::set_high_operation<0x1234>(hal::gpr<63>());
    constexpr std::uint32_t add      = gpr_ops::add_operation(hal::gpr<1>(), hal::gpr<2>(), hal::gpr<3>());
    constexpr std::uint32_t subtract = gpr_ops::subtract_operation(hal::gpr<4>(), hal::gpr<5>(), gpr_ops::immediate<63>());
    constexpr std::uint32_t multiply = gpr_ops::multiply_low_u16_operation(hal::gpr<7>(), hal::gpr<8>(), hal::gpr<9>());
    constexpr std::uint32_t bit_and  = gpr_ops::bit_and_operation(hal::gpr<10>(), hal::gpr<11>(), gpr_ops::immediate<1>());
    constexpr std::uint32_t bit_or   = gpr_ops::bit_or_operation(hal::gpr<13>(), hal::gpr<14>(), hal::gpr<15>());
    constexpr std::uint32_t bit_xor  = gpr_ops::bit_xor_operation(hal::gpr<16>(), hal::gpr<17>(), gpr_ops::immediate<3>());
    constexpr std::uint32_t left     = gpr_ops::shift_left_operation(hal::gpr<19>(), hal::gpr<20>(), hal::gpr<21>());
    constexpr std::uint32_t right    = gpr_ops::shift_right_operation(hal::gpr<22>(), hal::gpr<23>(), gpr_ops::immediate<5>());
    constexpr std::uint32_t greater  = gpr_ops::compare_operation<gpr_ops::Compare::GreaterThan>(hal::gpr<25>(), hal::gpr<26>(), hal::gpr<27>());
    constexpr std::uint32_t less     = gpr_ops::compare_operation<gpr_ops::Compare::LessThan>(hal::gpr<28>(), hal::gpr<29>(), gpr_ops::immediate<7>());
    constexpr std::uint32_t equal    = gpr_ops::compare_operation<gpr_ops::Compare::Equal>(hal::gpr<31>(), hal::gpr<32>(), hal::gpr<33>());
    TTI_INSN(low);
    TTI_INSN(high);
    TTI_INSN(add);
    TTI_INSN(subtract);
    TTI_INSN(multiply);
    TTI_INSN(bit_and);
    TTI_INSN(bit_or);
    TTI_INSN(bit_xor);
    TTI_INSN(left);
    TTI_INSN(right);
    TTI_INSN(greater);
    TTI_INSN(less);
    TTI_INSN(equal);
}

// CHECK-LABEL: <issue_encoded_operations>:
// CHECK-NEXT: ttsetdmareg 1,5752,0,126
// CHECK-NEXT: ttsetdmareg 0,4660,0,127
// CHECK-NEXT: ttadddmareg 0,1,3,2
// CHECK-NEXT: ttsubdmareg 1,4,63,5
// CHECK-NEXT: ttmuldmareg 0,7,9,8
// CHECK-NEXT: ttbitwopdmareg 1,0,10,1,11
// CHECK-NEXT: ttbitwopdmareg 0,1,13,15,14
// CHECK-NEXT: ttbitwopdmareg 1,2,16,3,17
// CHECK-NEXT: ttshiftdmareg 0,0,19,21,20
// CHECK-NEXT: ttshiftdmareg 1,1,22,5,23
// CHECK-NEXT: ttcmpdmareg 0,0,25,27,26
// CHECK-NEXT: ttcmpdmareg 1,1,28,7,29
// CHECK-NEXT: ttcmpdmareg 0,2,31,33,32
// CHECK-NEXT: ret
