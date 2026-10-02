// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_unpack_thread} -c %s -o %t.o
// RUN: %{blackhole_objdump} -dr %t.o | FileCheck %s --enable-var-scope

#include <cstdint>

#include "hal/address_counters.h"

using hal::AddressChannel;
using hal::AddressCounterClient;

// Runtime counter values are shifted into an operation word whose constant part
// (opcode, client mask, write mask) is built with lui/addi, then pushed through
// the instruction buffer.

extern "C" __attribute__((noinline, used)) void runtime_set_xy(std::uint32_t x, std::uint32_t y)
{
    hal::runtime_address_counters.client<AddressCounterClient::Unpacker0>().channel<AddressChannel::Channel0>().X(x).Y(y).apply();
}

// SETADCXY, client 1, write mask 0b0011; Channel0 X at bit 6 and Y at bit 9.
// CHECK-LABEL: <runtime_set_xy>:
// CHECK-DAG: lui [[OP:a[0-7]]],0x51200
// CHECK-DAG: addi [[OP]],[[OP]],3
// CHECK-DAG: slli a1,a1,0x9
// CHECK-DAG: slli a0,a0,0x6
// CHECK-DAG: R_RISCV_HI20 __instrn_buffer
// CHECK: sw {{a[0-7]}},0({{a[0-7]}})
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void runtime_set_single_counter(std::uint32_t z)
{
    hal::runtime_address_counters.client<AddressCounterClient::Unpacker1>().channel<AddressChannel::Channel1>().Z(z).apply();
}

// SETADC, client 2, Channel1, counter Z; the value occupies the low bits.
// CHECK-LABEL: <runtime_set_single_counter>:
// CHECK-DAG: lui [[OP:a[0-7]]],0x50580
// CHECK-DAG: add a0,a0,[[OP]]
// CHECK-DAG: R_RISCV_HI20 __instrn_buffer
// CHECK: sw a0,0({{a[0-7]}})
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void runtime_set_x_both_channels(std::uint32_t x0, std::uint32_t x1)
{
    hal::runtime_address_counters.client<AddressCounterClient::Packers>()
        .channel<AddressChannel::Channel0>()
        .X(x0)
        .channel<AddressChannel::Channel1>()
        .X(x1)
        .apply();
}

// SETADCXX, client 4; the Channel0 value is placed in the x_end2 field (bit 10).
// CHECK-LABEL: <runtime_set_x_both_channels>:
// CHECK-DAG: lui [[OP:a[0-7]]],0x5e800
// CHECK-DAG: add a1,a1,[[OP]]
// CHECK-DAG: slli a0,a0,0xa
// CHECK-DAG: R_RISCV_HI20 __instrn_buffer
// CHECK: sw {{a[0-7]}},0({{a[0-7]}})
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void runtime_set_zw(std::uint32_t z, std::uint32_t w)
{
    hal::runtime_address_counters.client<AddressCounterClient::Packers>().channel<AddressChannel::Channel1>().Z(z).W(w).apply();
}

// SETADCZW, client 4, write mask 0b1100; Channel1 Z at bit 12 and W at bit 15.
// CHECK-LABEL: <runtime_set_zw>:
// CHECK-DAG: lui [[OP:a[0-7]]],0x54800
// CHECK-DAG: addi [[OP]],[[OP]],12
// CHECK-DAG: slli a1,a1,0xf
// CHECK-DAG: slli a0,a0,0xc
// CHECK-DAG: R_RISCV_HI20 __instrn_buffer
// CHECK: sw {{a[0-7]}},0({{a[0-7]}})
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void runtime_increment_zw(std::uint32_t z, std::uint32_t w)
{
    hal::runtime_address_counters.client<AddressCounterClient::Packers>().channel<AddressChannel::Channel0>().Z(z).W(w).increment();
}

// INCADCZW, client 4; Channel0 Z at bit 6 and W at bit 9.
// CHECK-LABEL: <runtime_increment_zw>:
// CHECK-DAG: lui [[OP:a[0-7]]],0x55800
// CHECK-DAG: slli a1,a1,0x9
// CHECK-DAG: slli a0,a0,0x6
// CHECK-DAG: R_RISCV_HI20 __instrn_buffer
// CHECK: sw {{a[0-7]}},0({{a[0-7]}})
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void runtime_advance_and_reset_xy(std::uint32_t x, std::uint32_t y)
{
    hal::runtime_address_counters.client<AddressCounterClient::Unpacker0>().channel<AddressChannel::Channel0>().X(x & 0x7u).Y(y & 0x7u).advance_and_reset();
}

// ADDRCRXY, client 1, write mask 0b0011.
// CHECK-LABEL: <runtime_advance_and_reset_xy>:
// CHECK-DAG: lui [[OP:a[0-7]]],0x53200
// CHECK-DAG: addi [[OP]],[[OP]],3
// CHECK-DAG: R_RISCV_HI20 __instrn_buffer
// CHECK: sw {{a[0-7]}},0({{a[0-7]}})
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) std::uint32_t runtime_encoded_set_x(std::uint32_t x)
{
    return hal::runtime_address_counters.client<AddressCounterClient::Unpacker0>()
        .channel<AddressChannel::Channel0>()
        .X(x)
        .get_operation<hal::GetOpType::SETTER>();
}

// get_operation() returns the SETADC word without issuing it.
// CHECK-LABEL: <runtime_encoded_set_x>:
// CHECK-NEXT: lui [[OP:a[0-7]]],0x50200
// CHECK-NEXT: add a0,a0,[[OP]]
// CHECK-NEXT: ret
