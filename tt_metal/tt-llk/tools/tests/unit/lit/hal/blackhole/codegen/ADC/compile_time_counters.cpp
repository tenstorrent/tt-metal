// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_unpack_thread} -c %s -o %t.o
// RUN: %{blackhole_objdump} -d %t.o | FileCheck %s

#include "hal/counters/adc.h"

namespace adc = hal::counters::adc;

extern "C" __attribute__((noinline, used)) void set_one_counter_one_channel()
{
    constexpr auto y_counter = adc::AddrCounter {.channel1 = {.y = 5}}.for_clients<adc::Client::Unpacker0>();
    adc::set<y_counter>();
}

// CHECK-LABEL: <set_one_counter_one_channel>:
// CHECK-NEXT: ttsetadc 1,1,1,5
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void set_x_both_channels()
{
    constexpr auto x_counters = adc::AddrCounter {.channel0 = {.x = 4}, .channel1 = {.x = 3}}.for_clients<adc::Client::Packers>();
    adc::set<x_counters>();
}

// CHECK-LABEL: <set_x_both_channels>:
// CHECK-NEXT: ttsetadcxx 4,3,4
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void set_x_channel0_y_channel1()
{
    constexpr auto xy_counters = adc::AddrCounter {.channel0 = {.x = 6}, .channel1 = {.y = 7}}.for_clients<adc::Client::Unpacker1>();
    adc::set<xy_counters>();
}

// Write mask 0b1001 selects only Channel0 X and Channel1 Y.
// CHECK-LABEL: <set_x_channel0_y_channel1>:
// CHECK-NEXT: ttsetadcxy 2,7,0,0,6,9
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void set_all_counters_two_clients()
{
    constexpr auto channel0_counters =
        adc::AddrCounter {.channel0 = {.x = 1, .y = 2, .z = 3, .w = 4}}.for_clients<adc::Client::Unpacker0, adc::Client::Unpacker1>();
    adc::set<channel0_counters>();
}

// CHECK-LABEL: <set_all_counters_two_clients>:
// CHECK-NEXT: ttsetadcxy 3,0,0,2,1,3
// CHECK-NEXT: ttsetadczw 3,0,0,4,3,3
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void set_z_and_w_single_counters()
{
    constexpr auto w_counter  = adc::AddrCounter {.channel1 = {.w = 9}}.for_clients<adc::Client::Packers>();
    constexpr auto z_counters = adc::AddrCounter {.channel0 = {.z = 1}, .channel1 = {.z = 2}}.for_clients<adc::Client::Packers>();
    adc::set<w_counter>();
    adc::set<z_counters>();
}

// CHECK-LABEL: <set_z_and_w_single_counters>:
// CHECK-NEXT: ttsetadc 4,1,3,9
// CHECK-NEXT: ttsetadczw 4,0,2,0,1,5
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void increment_xy_and_zw()
{
    constexpr auto counter_increments =
        adc::AddrCounter {.action = adc::Action::Increment, .channel0 = {.x = 1, .y = 2}, .channel1 = {.z = 3, .w = 4}}.for_clients<adc::Client::Packers>();
    adc::set<counter_increments>();
}

// CHECK-LABEL: <increment_xy_and_zw>:
// CHECK-NEXT: ttincadcxy 4,0,0,2,1
// CHECK-NEXT: ttincadczw 4,4,3,0,0
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void advance_and_reset_xy_and_zw()
{
    constexpr auto counter_reload = adc::AddrCounter {.action = adc::Action::AdvanceAndReset, .channel0 = {.x = 1, .z = 2}}.for_clients<adc::Client::Packers>();
    adc::set<counter_reload>();
}

// CHECK-LABEL: <advance_and_reset_xy_and_zw>:
// CHECK-NEXT: ttaddrcrxy 4,0,0,0,1,1
// CHECK-NEXT: ttaddrcrzw 4,0,0,0,2,1
// CHECK-NEXT: ret

constexpr auto set_xy =
    adc::AddrCounter {.channel0 = {.x = 7, .y = 6}, .channel1 = {.x = 5, .y = 3}}.for_clients<adc::Client::Unpacker0, adc::Client::Packers>();

constexpr auto increment_xy = adc::AddrCounter {.action = adc::Action::Increment, .channel0 = set_xy.channel0, .channel1 = set_xy.channel1}
                                  .for_clients<adc::Client::Unpacker0, adc::Client::Packers>();

constexpr auto reload_xy = adc::AddrCounter {.action = adc::Action::AdvanceAndReset, .channel0 = set_xy.channel0, .channel1 = set_xy.channel1}
                               .for_clients<adc::Client::Unpacker0, adc::Client::Packers>();

constexpr auto set_zw = adc::AddrCounter {.channel0 = {.z = 0, .w = 7}, .channel1 = {.w = 6}}.for_clients<adc::Client::Unpacker1>();

constexpr auto increment_zw =
    adc::AddrCounter {.action = adc::Action::Increment, .channel0 = set_zw.channel0, .channel1 = set_zw.channel1}.for_clients<adc::Client::Unpacker1>();

constexpr auto reload_zw =
    adc::AddrCounter {.action = adc::Action::AdvanceAndReset, .channel0 = set_zw.channel0, .channel1 = set_zw.channel1}.for_clients<adc::Client::Unpacker1>();

// operation() returns the encoded word; issuing it shows which instruction it encodes.
extern "C" __attribute__((noinline, used)) void issue_encoded_operations()
{
    TTI_INSN(set_xy.operation());
    TTI_INSN(increment_xy.operation());
    TTI_INSN(reload_xy.operation());
    TTI_INSN(set_zw.operation());
    TTI_INSN(increment_zw.operation());
    TTI_INSN(reload_zw.operation());
}

// CHECK-LABEL: <issue_encoded_operations>:
// CHECK-NEXT: ttsetadcxy 5,3,5,6,7,15
// CHECK-NEXT: ttincadcxy 5,3,5,6,7
// CHECK-NEXT: ttaddrcrxy 5,3,5,6,7,15
// CHECK-NEXT: ttsetadczw 2,6,0,7,0,11
// CHECK-NEXT: ttincadczw 2,6,0,7,0
// CHECK-NEXT: ttaddrcrzw 2,6,0,7,0,11
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void single_x_and_z()
{
    constexpr auto xz_counters = adc::AddrCounter {.channel0 = {.x = 1, .z = 2}}.for_clients<adc::Client::Unpacker0>();
    adc::set<xz_counters>();
}

// Only one X counter is selected, so SETADCXX would change an unselected counter.
// CHECK-LABEL: <single_x_and_z>:
// CHECK-NEXT: ttsetadc 1,0,0,1
// CHECK-NEXT: ttsetadc 1,0,2,2
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void both_x_channels_and_z()
{
    constexpr auto xz_counters = adc::AddrCounter {.channel0 = {.x = 10, .z = 2}, .channel1 = {.x = 20}}.for_clients<adc::Client::Unpacker0>();
    adc::set<xz_counters>();
}

// SETADCXX takes X1 before X0. Z still needs its own instruction.
// CHECK-LABEL: <both_x_channels_and_z>:
// CHECK-NEXT: ttsetadcxx 1,20,10
// CHECK-NEXT: ttsetadc 1,0,2,2
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void prefer_xx_when_instruction_counts_tie()
{
    constexpr auto xy_counters = adc::AddrCounter {.channel0 = {.x = 8, .y = 2}, .channel1 = {.x = 3}}.for_clients<adc::Client::Unpacker0>();
    adc::set<xy_counters>();
}

// Both SETADC(X0) + SETADCXY(Y0,X1) and SETADCXX(X0,X1) + SETADC(Y0) need two
// instructions. Prefer the dedicated instruction for the two X counters.
// CHECK-LABEL: <prefer_xx_when_instruction_counts_tie>:
// CHECK-NEXT: ttsetadcxx 1,3,8
// CHECK-NEXT: ttsetadc 1,0,1,2
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void keep_packed_xy_when_shorter()
{
    constexpr auto xy_counters = adc::AddrCounter {.channel0 = {.x = 1, .y = 2}, .channel1 = {.x = 3}}.for_clients<adc::Client::Unpacker0>();
    adc::set<xy_counters>();
}

// SETADCXX would need a second instruction for Y0; the packed instruction covers all three.
// CHECK-LABEL: <keep_packed_xy_when_shorter>:
// CHECK-NEXT: ttsetadcxy 1,0,3,2,1,7
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void x_exceeds_dual_x_width()
{
    constexpr auto xy_counters = adc::AddrCounter {.channel0 = {.x = 1024, .y = 2}, .channel1 = {.x = 3}}.for_clients<adc::Client::Unpacker0>();
    adc::set<xy_counters>();
}

// X0 does not fit SETADCXX's ten-bit value, so keep its full-width SETADC write.
// CHECK-LABEL: <x_exceeds_dual_x_width>:
// CHECK-NEXT: ttsetadc 1,0,0,1024
// CHECK-NEXT: ttsetadcxy 1,0,3,2,0,6
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void packed_xy_and_single_z()
{
    constexpr auto xyz_counters = adc::AddrCounter {.channel0 = {.x = 7, .y = 2, .z = 2}, .channel1 = {.x = 7}}.for_clients<adc::Client::Unpacker0>();
    adc::set<xyz_counters>();
}

// Both X values fit the packed instruction, so updating Y needs no extra instruction.
// CHECK-LABEL: <packed_xy_and_single_z>:
// CHECK-NEXT: ttsetadcxy 1,0,7,2,7,7
// CHECK-NEXT: ttsetadc 1,0,2,2
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void dual_x_and_single_y_z()
{
    constexpr auto xyz_counters = adc::AddrCounter {.channel0 = {.x = 60, .y = 2, .z = 2}, .channel1 = {.x = 1023}}.for_clients<adc::Client::Unpacker0>();
    adc::set<xyz_counters>();
}

// Both X values fit SETADCXX; Y and Z each need their own instruction.
// CHECK-LABEL: <dual_x_and_single_y_z>:
// CHECK-NEXT: ttsetadcxx 1,1023,60
// CHECK-NEXT: ttsetadc 1,0,1,2
// CHECK-NEXT: ttsetadc 1,0,2,2
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void wide_x_values_and_single_y_z()
{
    constexpr auto xyz_counters = adc::AddrCounter {.channel0 = {.x = 60, .y = 2, .z = 2}, .channel1 = {.x = 60000}}.for_clients<adc::Client::Unpacker0>();
    adc::set<xyz_counters>();
}

// X1 exceeds SETADCXX's range and neither X fits SETADCXY. Four writes are necessary.
// CHECK-LABEL: <wide_x_values_and_single_y_z>:
// CHECK-NEXT: ttsetadc 1,0,0,60
// CHECK-NEXT: ttsetadc 1,1,0,60000
// CHECK-NEXT: ttsetadc 1,0,1,2
// CHECK-NEXT: ttsetadc 1,0,2,2
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void wide_x1_and_packed_x0_y()
{
    constexpr auto xyz_counters = adc::AddrCounter {.channel0 = {.x = 7, .y = 2, .z = 2}, .channel1 = {.x = 60000}}.for_clients<adc::Client::Unpacker0>();
    adc::set<xyz_counters>();
}

// X1 needs SETADC, but X0 and Y0 can still share a packed instruction.
// CHECK-LABEL: <wide_x1_and_packed_x0_y>:
// CHECK-NEXT: ttsetadc 1,1,0,60000
// CHECK-NEXT: ttsetadcxy 1,0,0,2,7,3
// CHECK-NEXT: ttsetadc 1,0,2,2
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void dual_x_and_packed_y_channels()
{
    constexpr auto xy_counters = adc::AddrCounter {.channel0 = {.x = 1023, .y = 0}, .channel1 = {.x = 1023, .y = 7}}.for_clients<adc::Client::Unpacker0>();
    adc::set<xy_counters>();
}

// Two full ten-bit X values share SETADCXX. Both Y values, including explicit zero, share SETADCXY.
// CHECK-LABEL: <dual_x_and_packed_y_channels>:
// CHECK-NEXT: ttsetadcxx 1,1023,1023
// CHECK-NEXT: ttsetadcxy 1,7,0,0,0,10
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void mixed_widths_all_counters()
{
    constexpr auto counters = adc::AddrCounter {.channel0 = {.x = 0, .y = 8, .z = 7, .w = 1024}, .channel1 = {.x = 1023, .y = 7, .z = 65535, .w = 0}}
                                  .for_clients<adc::Client::Unpacker0, adc::Client::Packers>();
    adc::set<counters>();
}

// Both families mix widths. Prefer XX on the XY tie and preserve explicitly selected zeroes.
// CHECK-LABEL: <mixed_widths_all_counters>:
// CHECK-NEXT: ttsetadcxx 5,1023,0
// CHECK-NEXT: ttsetadc 5,0,1,8
// CHECK-NEXT: ttsetadc 5,1,1,7
// CHECK-NEXT: ttsetadc 5,0,3,1024
// CHECK-NEXT: ttsetadc 5,1,2,65535
// CHECK-NEXT: ttsetadczw 5,0,0,0,7,9
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void wide_values_all_counters()
{
    constexpr auto counters =
        adc::AddrCounter {
            .channel0 = {.x = 65535, .y = 65534, .z = 65533, .w = 65532},
            .channel1 = {.x = 65531, .y = 65530, .z = 65529, .w = 65528},
        }
            .for_clients<adc::Client::Unpacker0>();
    adc::set<counters>();
}

// Every value needs the full sixteen-bit SETADC field, reaching the eight-instruction maximum.
// CHECK-LABEL: <wide_values_all_counters>:
// CHECK-NEXT: ttsetadc 1,0,0,65535
// CHECK-NEXT: ttsetadc 1,0,1,65534
// CHECK-NEXT: ttsetadc 1,1,0,65531
// CHECK-NEXT: ttsetadc 1,1,1,65530
// CHECK-NEXT: ttsetadc 1,0,2,65533
// CHECK-NEXT: ttsetadc 1,0,3,65532
// CHECK-NEXT: ttsetadc 1,1,2,65529
// CHECK-NEXT: ttsetadc 1,1,3,65528
// CHECK-NEXT: ret
