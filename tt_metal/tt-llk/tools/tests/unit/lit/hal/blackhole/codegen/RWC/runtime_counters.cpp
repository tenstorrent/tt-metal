// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -UENABLE_LLK_ASSERT -fno-ipa-icf -c %s -o %t.o
// RUN: %{blackhole_objdump} -drz %t.o > %t.dump
// RUN: %{blackhole_compare_codegen} %t.dump

// Compare each HAL function with its adjacent reference_ function, including
// instruction bytes and relocations. Both bodies must be emitted independently.

#include <cstdint>

#include "hal/counters/rwc.h"

namespace rwc = hal::counters::rwc;

extern "C" __attribute__((noinline, used)) void runtime_next_face(std::uint32_t b, std::uint32_t d)
{
    const auto face = rwc::MathCounter {.src_b = rwc::Set(b), .dest = rwc::AdvanceAndReset(d), .flip = rwc::Src::A | rwc::Src::B};
    rwc::set(face);
}

extern "C" __attribute__((noinline, used)) void reference_runtime_next_face(std::uint32_t b, std::uint32_t d)
{
    const auto src_b = b;
    const auto dest  = d;
    TT_SETRWC(ckernel::p_setrwc::CLR_AB, ckernel::p_setrwc::CR_D, dest, src_b, 0, ckernel::p_setrwc::SET_BD);
}

extern "C" __attribute__((noinline, used)) void runtime_increment_dest(std::uint32_t d)
{
    const auto rows = rwc::MathCounter {.dest = rwc::Increment(d)};
    rwc::set(rows);
}

extern "C" __attribute__((noinline, used)) void reference_runtime_increment_dest(std::uint32_t d)
{
    TT_INCRWC(0, d, 0, 0);
}

extern "C" __attribute__((noinline, used)) void runtime_advance_and_reset_dest(std::uint32_t d)
{
    const auto base = rwc::MathCounter {.dest = rwc::AdvanceAndReset(d)};
    rwc::set(base);
}

extern "C" __attribute__((noinline, used)) void reference_runtime_advance_and_reset_dest(std::uint32_t d)
{
    TT_INCRWC(ckernel::p_setrwc::CR_D, d, 0, 0);
}

extern "C" __attribute__((noinline, used)) void runtime_increment_and_save_dest(std::uint32_t d)
{
    const auto save = rwc::MathCounter {.dest = rwc::IncrementAndSave(d)};
    rwc::set(save);
}

extern "C" __attribute__((noinline, used)) void reference_runtime_increment_and_save_dest(std::uint32_t d)
{
    TT_SETRWC(ckernel::p_setrwc::CLR_NONE, ckernel::p_setrwc::C_TO_CR_MODE, d, 0, 0, ckernel::p_setrwc::SET_D);
}

extern "C" __attribute__((noinline, used)) void runtime_set_and_increment(std::uint32_t a, std::uint32_t d)
{
    const auto mixed = rwc::MathCounter {.src_a = rwc::Set(a), .dest = rwc::Increment(d)};
    rwc::set(mixed);
}

extern "C" __attribute__((noinline, used)) void reference_runtime_set_and_increment(std::uint32_t a, std::uint32_t d)
{
    TT_SETRWC(ckernel::p_setrwc::CLR_NONE, 0, 0, 0, a, ckernel::p_setrwc::SET_A);
    TT_INCRWC(0, d, 0, 0);
}

extern "C" __attribute__((noinline, used)) std::uint32_t runtime_encoded_increment_src_a(std::uint32_t a)
{
    const auto counter = rwc::MathCounter {.src_a = rwc::Increment(a)};
    return counter.operation();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_runtime_encoded_increment_src_a(std::uint32_t a)
{
    return TT_OP_INCRWC(0, 0, 0, a);
}
