// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_unpack_thread} -c %s -o %t.o
// RUN: %{blackhole_objdump} -d %t.o | FileCheck %s

#include "hal/address_counters.h"

using hal::AddressChannel;
using hal::AddressCounterClient;

extern "C" __attribute__((noinline, used)) void set_one_counter_one_channel()
{
    hal::address_counters.client<AddressCounterClient::Unpacker0>().channel<AddressChannel::Channel1>().Y<5>().apply();
}

// CHECK-LABEL: <set_one_counter_one_channel>:
// CHECK-NEXT: ttsetadc 1,1,1,5
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void set_x_both_channels()
{
    hal::address_counters.client<AddressCounterClient::Packers>().channel<AddressChannel::Channel0>().X<3>().channel<AddressChannel::Channel1>().X<4>().apply();
}

// CHECK-LABEL: <set_x_both_channels>:
// CHECK-NEXT: ttsetadcxx 4,3,4
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void set_x_channel0_y_channel1()
{
    hal::address_counters.client<AddressCounterClient::Unpacker1>()
        .channel<AddressChannel::Channel0>()
        .X<6>()
        .channel<AddressChannel::Channel1>()
        .Y<7>()
        .apply();
}

// Write mask 0b1001 selects only Channel0 X and Channel1 Y.
// CHECK-LABEL: <set_x_channel0_y_channel1>:
// CHECK-NEXT: ttsetadcxy 2,7,0,0,6,9
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void set_all_counters_two_clients()
{
    hal::address_counters.client<AddressCounterClient::Unpacker0, AddressCounterClient::Unpacker1>()
        .channel<AddressChannel::Channel0>()
        .X<1>()
        .Y<2>()
        .Z<3>()
        .W<4>()
        .apply();
}

// CHECK-LABEL: <set_all_counters_two_clients>:
// CHECK-NEXT: ttsetadcxy 3,0,0,2,1,3
// CHECK-NEXT: ttsetadczw 3,0,0,4,3,3
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void set_z_and_w_single_counters()
{
    hal::address_counters.client<AddressCounterClient::Packers>().channel<AddressChannel::Channel1>().W<9>().apply();
    hal::address_counters.client<AddressCounterClient::Packers>().channel<AddressChannel::Channel0>().Z<1>().channel<AddressChannel::Channel1>().Z<2>().apply();
}

// CHECK-LABEL: <set_z_and_w_single_counters>:
// CHECK-NEXT: ttsetadc 4,1,3,9
// CHECK-NEXT: ttsetadczw 4,0,2,0,1,5
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void increment_xy_and_zw()
{
    hal::address_counters.client<AddressCounterClient::Packers>()
        .channel<AddressChannel::Channel0>()
        .X<1>()
        .Y<2>()
        .channel<AddressChannel::Channel1>()
        .Z<3>()
        .W<4>()
        .increment();
}

// CHECK-LABEL: <increment_xy_and_zw>:
// CHECK-NEXT: ttincadcxy 4,0,0,2,1
// CHECK-NEXT: ttincadczw 4,4,3,0,0
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void advance_and_reset_xy_and_zw()
{
    hal::address_counters.client<AddressCounterClient::Packers>().channel<AddressChannel::Channel0>().X<1>().Z<2>().advance_and_reset();
}

// CHECK-LABEL: <advance_and_reset_xy_and_zw>:
// CHECK-NEXT: ttaddrcrxy 4,0,0,0,1,1
// CHECK-NEXT: ttaddrcrzw 4,0,0,0,2,1
// CHECK-NEXT: ret

constexpr auto both_channels_xy = hal::address_counters.client<AddressCounterClient::Unpacker0, AddressCounterClient::Packers>()
                                      .channel<AddressChannel::Channel0>()
                                      .X<7>()
                                      .Y<6>()
                                      .channel<AddressChannel::Channel1>()
                                      .X<5>()
                                      .Y<3>();

constexpr auto both_channels_zw = hal::address_counters.client<AddressCounterClient::Unpacker1>()
                                      .channel<AddressChannel::Channel0>()
                                      .Z<0>()
                                      .W<7>()
                                      .channel<AddressChannel::Channel1>()
                                      .W<6>();

// get_operation() returns the encoded word; issuing it shows which instruction it encodes.
extern "C" __attribute__((noinline, used)) void issue_encoded_operations()
{
    TTI_INSN(both_channels_xy.get_operation<hal::GetOpType::SETTER>());
    TTI_INSN(both_channels_xy.get_operation<hal::GetOpType::INCREMENT>());
    TTI_INSN(both_channels_xy.get_operation<hal::GetOpType::ADVANCE_AND_RESET>());
    TTI_INSN(both_channels_zw.get_operation<hal::GetOpType::SETTER>());
    TTI_INSN(both_channels_zw.get_operation<hal::GetOpType::INCREMENT>());
    TTI_INSN(both_channels_zw.get_operation<hal::GetOpType::ADVANCE_AND_RESET>());
}

// CHECK-LABEL: <issue_encoded_operations>:
// CHECK-NEXT: ttsetadcxy 5,3,5,6,7,15
// CHECK-NEXT: ttincadcxy 5,3,5,6,7
// CHECK-NEXT: ttaddrcrxy 5,3,5,6,7,15
// CHECK-NEXT: ttsetadczw 2,6,0,7,0,11
// CHECK-NEXT: ttincadczw 2,6,0,7,0
// CHECK-NEXT: ttaddrcrzw 2,6,0,7,0,11
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void reg2flop_current_thread_counter()
{
    hal::address_counters.client<AddressCounterClient::Unpacker0>().channel<AddressChannel::Channel0>().X().from_gpr(hal::gpr<5>()).apply();
}

// CHECK-LABEL: <reg2flop_current_thread_counter>:
// CHECK-NEXT: ttreg2flop 1,2,0,0,0,5
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void reg2flop_other_thread_carry_halfword()
{
    hal::address_counters.client<AddressCounterClient::Packers>()
        .channel<AddressChannel::Channel1>()
        .W<hal::AddressCounterValue::Carry>()
        .thread<hal::AddressCounterThread::T2>()
        .from_gpr<hal::AddressCounterGprWidth::Bits16, hal::GprByteOffset::Byte2>(hal::gpr<60>())
        .apply();
}

// Flop index 55 = Channel1 (32) | Packers (16) | Carry (4) | W (3).
// CHECK-LABEL: <reg2flop_other_thread_carry_halfword>:
// CHECK-NEXT: ttreg2flop 2,3,2,2,55,60
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void reg2flop_byte_from_encoded_operation()
{
    constexpr auto byte_to_y = hal::address_counters.client<AddressCounterClient::Unpacker1>()
                                   .channel<AddressChannel::Channel0>()
                                   .Y()
                                   .from_gpr<hal::AddressCounterGprWidth::Bits8, hal::GprByteOffset::Byte3>(hal::gpr<17>());
    TTI_INSN(byte_to_y.get_operation());
}

// CHECK-LABEL: <reg2flop_byte_from_encoded_operation>:
// CHECK-NEXT: ttreg2flop 3,2,3,0,9,17
// CHECK-NEXT: ret
