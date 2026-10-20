// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -UENABLE_LLK_ASSERT -fno-ipa-icf -c %s -o %t.o
// RUN: %{blackhole_objdump} -drz %t.o > %t.dump
// RUN: %{blackhole_compare_codegen} %t.dump

// hal::counters::set dispatches on the descriptor type; each reference_ body calls the
// family-specific set directly, so the generated code must be identical.

#include <cstdint>

#include "hal/tensix_counters.h"

namespace counters = hal::counters;
namespace adc      = hal::counters::adc;
namespace rwc      = hal::counters::rwc;

inline constexpr auto start = adc::AddrCounter {.channel0 = {.x = 1, .y = 2}}.for_clients<adc::Client::Unpacker0>();
inline constexpr auto face  = rwc::MathCounter {.src_b = rwc::Set(0), .dest = rwc::AdvanceAndReset(8)};

extern "C" __attribute__((noinline, used)) void shared_compile_time()
{
    counters::set<start>();
    counters::set<face>();
}

extern "C" __attribute__((noinline, used)) void reference_shared_compile_time()
{
    adc::set<start>();
    rwc::set<face>();
}

extern "C" __attribute__((noinline, used)) void shared_runtime(std::uint32_t x, std::uint32_t d)
{
    counters::set(adc::AddrCounter {.channel0 = {.x = x}}.for_clients<adc::Client::Packers>());
    counters::set(rwc::MathCounter {.dest = rwc::Increment(d)});
}

extern "C" __attribute__((noinline, used)) void reference_shared_runtime(std::uint32_t x, std::uint32_t d)
{
    adc::set(adc::AddrCounter {.channel0 = {.x = x}}.for_clients<adc::Client::Packers>());
    rwc::set(rwc::MathCounter {.dest = rwc::Increment(d)});
}
