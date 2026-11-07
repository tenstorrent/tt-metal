// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_unpack_thread} -UENABLE_LLK_ASSERT -fno-ipa-icf -c %s -o %t.o
// RUN: %{blackhole_objdump} -drz %t.o > %t.dump
// RUN: %{blackhole_compare_codegen} %t.dump

// Compare each HAL function with its adjacent reference_ function, including
// instruction bytes and relocations. Both bodies must be emitted independently.

#include <cstdint>

#include "hal/tensix_counters.h"

namespace counters = hal::counters;
namespace adc      = hal::counters::adc;

extern "C" __attribute__((noinline, used)) void runtime_set_xy(std::uint32_t x, std::uint32_t y)
{
    const auto xy_counters = adc::AddrCounter {.channel0 = {.x = x, .y = y}}.for_clients<adc::Client::Unpacker0>();
    counters::set(xy_counters);
}

extern "C" __attribute__((noinline, used)) void reference_runtime_set_xy(std::uint32_t x, std::uint32_t y)
{
    const auto ch0_x = x;
    const auto ch0_y = y;
    TT_SETADCXY(0b001 /*UNP0*/, 0 /*Ch1_Y*/, 0 /*Ch1_X*/, ch0_y, ch0_x, 0b0011 /*BitMask*/);
}

extern "C" __attribute__((noinline, used)) void runtime_set_single_counter(std::uint32_t z)
{
    const auto z_counter = adc::AddrCounter {.channel1 = {.z = z}}.for_clients<adc::Client::Unpacker1>();
    counters::set(z_counter);
}

extern "C" __attribute__((noinline, used)) void reference_runtime_set_single_counter(std::uint32_t z)
{
    TT_SETADC(0b010 /*UNP1*/, 1 /*ChannelIndex*/, 2 /*SET_Z*/, z);
}

extern "C" __attribute__((noinline, used)) void runtime_set_x_both_channels(std::uint32_t x0, std::uint32_t x1)
{
    const auto x_counters = adc::AddrCounter {.channel0 = {.x = x1}, .channel1 = {.x = x0}}.for_clients<adc::Client::Packers>();
    counters::set(x_counters);
}

extern "C" __attribute__((noinline, used)) void reference_runtime_set_x_both_channels(std::uint32_t x0, std::uint32_t x1)
{
    const auto ch0_x = x1;
    const auto ch1_x = x0;
    TT_SETADCXX(0b100 /*PAC*/, ch1_x, ch0_x);
}

extern "C" __attribute__((noinline, used)) void runtime_set_zw(std::uint32_t z, std::uint32_t w)
{
    const auto zw_counters = adc::AddrCounter {.channel1 = {.z = z, .w = w}}.for_clients<adc::Client::Packers>();
    counters::set(zw_counters);
}

extern "C" __attribute__((noinline, used)) void reference_runtime_set_zw(std::uint32_t z, std::uint32_t w)
{
    const auto ch1_z = z;
    const auto ch1_w = w;
    TT_SETADCZW(0b100 /*PAC*/, ch1_w, ch1_z, 0 /*Ch0_W*/, 0 /*Ch0_Z*/, 0b1100 /*BitMask*/);
}

extern "C" __attribute__((noinline, used)) void runtime_increment_zw(std::uint32_t z, std::uint32_t w)
{
    const auto zw_increment = adc::AddrCounter {.action = adc::Action::Increment, .channel0 = {.z = z, .w = w}}.for_clients<adc::Client::Packers>();
    counters::set(zw_increment);
}

extern "C" __attribute__((noinline, used)) void reference_runtime_increment_zw(std::uint32_t z, std::uint32_t w)
{
    const auto ch0_z = z;
    const auto ch0_w = w;
    TT_INCADCZW(0b100 /*PAC*/, 0 /*Ch1_W*/, 0 /*Ch1_Z*/, ch0_w, ch0_z);
}

extern "C" __attribute__((noinline, used)) void runtime_advance_and_reset_xy(std::uint32_t x, std::uint32_t y)
{
    const auto xy_reload = adc::AddrCounter {.action = adc::Action::AdvanceAndReset, .channel0 = {.x = x, .y = y}}.for_clients<adc::Client::Unpacker0>();
    counters::set(xy_reload);
}

extern "C" __attribute__((noinline, used)) void reference_runtime_advance_and_reset_xy(std::uint32_t x, std::uint32_t y)
{
    const auto ch0_x = x;
    const auto ch0_y = y;
    TT_ADDRCRXY(0b001 /*UNP0*/, 0 /*Ch1_Y*/, 0 /*Ch1_X*/, ch0_y, ch0_x, 0b0011 /*BitMask*/);
}

extern "C" __attribute__((noinline, used)) std::uint32_t runtime_encoded_set_x(std::uint32_t x)
{
    const auto x_counter = adc::AddrCounter {.channel0 = {.x = x}}.for_clients<adc::Client::Unpacker0>();
    return x_counter.operation();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_runtime_encoded_set_x(std::uint32_t x)
{
    return TT_OP_SETADC(0b001 /*UNP0*/, 0 /*ChannelIndex*/, 0 /*SET_X*/, x);
}
