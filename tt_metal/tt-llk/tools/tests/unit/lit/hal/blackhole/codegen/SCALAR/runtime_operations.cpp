// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -UENABLE_LLK_ASSERT -fno-ipa-icf -c %s -o %t.o
// RUN: %{blackhole_objdump} -drz %t.o > %t.dump
// RUN: %{blackhole_compare_codegen} %t.dump

// Compare each HAL function with its adjacent reference_ function. Disable
// identical-code folding so both bodies are emitted independently.

#include <cstdint>

#include "hal/scalar.h"

namespace scalar = hal::scalar;

extern "C" __attribute__((noinline, used)) void set_runtime_word(const std::uint32_t value)
{
    scalar::set(hal::gpr<5>(), value);
}

extern "C" __attribute__((noinline, used)) void reference_set_runtime_word(const std::uint32_t value)
{
    TT_SETDMAREG(0 /*Payload_SigSelSize*/, value & 0xffffu, 0 /*SetSignalsMode*/, 10 /*RegIndex16b*/);
    TT_SETDMAREG(0 /*Payload_SigSelSize*/, value >> 16, 0 /*SetSignalsMode*/, 11 /*RegIndex16b*/);
}

extern "C" __attribute__((noinline, used)) void set_runtime_low(const std::uint32_t value)
{
    scalar::set_low(hal::gpr<5>(), value);
}

extern "C" __attribute__((noinline, used)) void reference_set_runtime_low(const std::uint32_t value)
{
    TT_SETDMAREG(0 /*Payload_SigSelSize*/, value, 0 /*SetSignalsMode*/, 10 /*RegIndex16b*/);
}

extern "C" __attribute__((noinline, used)) void set_runtime_high(const std::uint32_t value)
{
    scalar::set_high(hal::gpr<5>(), value);
}

extern "C" __attribute__((noinline, used)) void reference_set_runtime_high(const std::uint32_t value)
{
    TT_SETDMAREG(0 /*Payload_SigSelSize*/, value, 0 /*SetSignalsMode*/, 11 /*RegIndex16b*/);
}

extern "C" __attribute__((noinline, used)) void add_runtime_immediate(const std::uint32_t value)
{
    scalar::add(hal::gpr<1>(), hal::gpr<2>(), scalar::immediate(value));
}

extern "C" __attribute__((noinline, used)) void reference_add_runtime_immediate(const std::uint32_t value)
{
    TT_ADDDMAREG(1 /*OpBisConst*/, 1 /*ResultRegIndex*/, value, 2 /*OpARegIndex*/);
}

extern "C" __attribute__((noinline, used)) void add_runtime_gpr(const std::uint32_t index)
{
    scalar::add(hal::gpr<1>(), hal::gpr<2>(), hal::gpr(index));
}

extern "C" __attribute__((noinline, used)) void reference_add_runtime_gpr(const std::uint32_t index)
{
    TT_ADDDMAREG(0 /*OpBisConst*/, 1 /*ResultRegIndex*/, index, 2 /*OpARegIndex*/);
}

extern "C" __attribute__((noinline, used)) void compare_runtime_destination(const std::uint32_t index)
{
    scalar::compare<scalar::Compare::Equal>(hal::gpr(index), hal::gpr<2>(), scalar::immediate<3>());
}

extern "C" __attribute__((noinline, used)) void reference_compare_runtime_destination(const std::uint32_t index)
{
    TT_CMPDMAREG(1 /*OpBisConst*/, 2 /*OpSel*/, index, 3 /*OpBRegIndex*/, 2 /*OpARegIndex*/);
}
