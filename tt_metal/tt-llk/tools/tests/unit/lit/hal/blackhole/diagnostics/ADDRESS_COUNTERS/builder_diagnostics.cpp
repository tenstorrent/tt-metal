// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
// RUN: %split-file %s %t
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/no-client.cpp 2>&1 | FileCheck %s --check-prefix=NO_CLIENT
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/no-channel.cpp 2>&1 | FileCheck %s --check-prefix=NO_CHANNEL
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/channel-twice.cpp 2>&1 | FileCheck %s --check-prefix=CHANNEL_TWICE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/operation-two-families.cpp 2>&1 | FileCheck %s --check-prefix=OPERATION_TWO_FAMILIES
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/advance-out-of-range.cpp 2>&1 | FileCheck %s --check-prefix=ADVANCE_OUT_OF_RANGE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/gpr-after-constant.cpp 2>&1 | FileCheck %s --check-prefix=GPR_AFTER_CONSTANT
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/gpr-two-clients.cpp 2>&1 | FileCheck %s --check-prefix=GPR_TWO_CLIENTS
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/gpr-two-channels.cpp 2>&1 | FileCheck %s --check-prefix=GPR_TWO_CHANNELS
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/gpr-thread-twice.cpp 2>&1 | FileCheck %s --check-prefix=GPR_THREAD_TWICE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/gpr-thread-current.cpp 2>&1 | FileCheck %s --check-prefix=GPR_THREAD_CURRENT
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/gpr-runtime-index.cpp 2>&1 | FileCheck %s --check-prefix=GPR_RUNTIME_INDEX
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/gpr-index-range.cpp 2>&1 | FileCheck %s --check-prefix=GPR_INDEX_RANGE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/gpr-byte-offset.cpp 2>&1 | FileCheck %s --check-prefix=GPR_BYTE_OFFSET
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/runtime-two-families.cpp 2>&1 | FileCheck %s --check-prefix=RUNTIME_TWO_FAMILIES
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/runtime-increment-two-families.cpp 2>&1 | FileCheck %s --check-prefix=RUNTIME_INCREMENT_TWO_FAMILIES
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_unpack_thread} %t/runtime-no-client.cpp 2>&1 | FileCheck %s --check-prefix=RUNTIME_NO_CLIENT
// clang-format on

//--- no-client.cpp
#include "hal/address_counters.h"

using hal::AddressChannel;
using hal::AddressCounterClient;

void no_client()
{
    hal::address_counters.channel<AddressChannel::Channel0>().X<1>().apply();
}

// NO_CLIENT: error: static assertion failed: no client selected — call client<...>() first

//--- no-channel.cpp
#include "hal/address_counters.h"

using hal::AddressChannel;
using hal::AddressCounterClient;

void no_channel()
{
    hal::address_counters.client<AddressCounterClient::Unpacker0>().increment();
}

// NO_CHANNEL: error: static assertion failed: no channel selected — call channel<...>() first

//--- channel-twice.cpp
#include "hal/address_counters.h"

using hal::AddressChannel;
using hal::AddressCounterClient;

constexpr auto channel_twice =
    hal::address_counters.client<AddressCounterClient::Packers>().channel<AddressChannel::Channel0>().X<1>().channel<AddressChannel::Channel0>();
// CHANNEL_TWICE: error: static assertion failed: channel already selected — channel<...>() called twice for the same channel

//--- operation-two-families.cpp
#include "hal/address_counters.h"

using hal::AddressChannel;
using hal::AddressCounterClient;

constexpr auto two_families =
    hal::address_counters.client<AddressCounterClient::Packers>().channel<AddressChannel::Channel0>().X<1>().W<2>().get_operation<hal::GetOpType::SETTER>();
// OPERATION_TWO_FAMILIES: error: static assertion failed: get_operation() must encode exactly one instruction (select X/Y or Z/W, not both)

//--- advance-out-of-range.cpp
#include "hal/address_counters.h"

using hal::AddressChannel;
using hal::AddressCounterClient;

constexpr auto advance_out_of_range = hal::address_counters.client<AddressCounterClient::Unpacker0>()
                                          .channel<AddressChannel::Channel0>()
                                          .X<8>()
                                          .get_operation<hal::GetOpType::ADVANCE_AND_RESET>();
// ADVANCE_OUT_OF_RANGE: error: static assertion failed: ADDRCRXY increments exceed their encoded field widths

//--- gpr-after-constant.cpp
#include "hal/address_counters.h"

using hal::AddressChannel;
using hal::AddressCounterClient;

constexpr auto gpr_after_constant =
    hal::address_counters.client<AddressCounterClient::Unpacker0>().channel<AddressChannel::Channel0>().Y<1>().X().from_gpr(hal::gpr<4>());
// GPR_AFTER_CONSTANT: error: static assertion failed: REG2FLOP cannot share a builder with pending constant assignments;

//--- gpr-two-clients.cpp
#include "hal/address_counters.h"

using hal::AddressChannel;
using hal::AddressCounterClient;

constexpr auto gpr_two_clients =
    hal::address_counters.client<AddressCounterClient::Unpacker0, AddressCounterClient::Unpacker1>().channel<AddressChannel::Channel0>().X().from_gpr(
        hal::gpr<4>());
// GPR_TWO_CLIENTS: error: static assertion failed: REG2FLOP accepts exactly one address-counter client

//--- gpr-two-channels.cpp
#include "hal/address_counters.h"

using hal::AddressChannel;
using hal::AddressCounterClient;

constexpr auto gpr_two_channels =
    hal::address_counters.client<AddressCounterClient::Unpacker0>().channel<AddressChannel::Channel0>().channel<AddressChannel::Channel1>().X().from_gpr(
        hal::gpr<4>());
// GPR_TWO_CHANNELS: error: static assertion failed: REG2FLOP cannot follow a multi-channel selection

//--- gpr-thread-twice.cpp
#include "hal/address_counters.h"

using hal::AddressChannel;
using hal::AddressCounterClient;

constexpr auto gpr_thread_twice = hal::address_counters.client<AddressCounterClient::Packers>()
                                      .channel<AddressChannel::Channel0>()
                                      .X()
                                      .thread<hal::AddressCounterThread::T0>()
                                      .thread<hal::AddressCounterThread::T1>();
// GPR_THREAD_TWICE: error: static assertion failed: REG2FLOP destination thread already selected

//--- gpr-thread-current.cpp
#include "hal/address_counters.h"

using hal::AddressChannel;
using hal::AddressCounterClient;

constexpr auto gpr_thread_current =
    hal::address_counters.client<AddressCounterClient::Packers>().channel<AddressChannel::Channel0>().X().thread<hal::AddressCounterThread::Current>();
// GPR_THREAD_CURRENT: error: static assertion failed: REG2FLOP destination thread must be T0, T1, or T2

//--- gpr-runtime-index.cpp
#include <cstdint>

#include "hal/address_counters.h"

using hal::AddressChannel;
using hal::AddressCounterClient;

void gpr_runtime_index(const std::uint32_t index)
{
    hal::address_counters.client<AddressCounterClient::Unpacker0>().channel<AddressChannel::Channel0>().X().from_gpr(hal::gpr(index)).apply();
}

// GPR_RUNTIME_INDEX: error: static assertion failed: address-counter REG2FLOP requires hal::gpr<Index>()

//--- gpr-index-range.cpp
#include "hal/address_counters.h"

using hal::AddressChannel;
using hal::AddressCounterClient;

constexpr auto gpr_index_range =
    hal::address_counters.client<AddressCounterClient::Unpacker0>().channel<AddressChannel::Channel0>().X().from_gpr(hal::gpr<64>());
// GPR_INDEX_RANGE: error: static assertion failed: REG2FLOP source GPR index must be in [0, 63]

//--- gpr-byte-offset.cpp
#include "hal/address_counters.h"

using hal::AddressChannel;
using hal::AddressCounterClient;

constexpr auto gpr_byte_offset = hal::address_counters.client<AddressCounterClient::Unpacker0>()
                                     .channel<AddressChannel::Channel0>()
                                     .X()
                                     .from_gpr<hal::AddressCounterGprWidth::Bits16, hal::GprByteOffset::Byte1>(hal::gpr<4>());
// GPR_BYTE_OFFSET: error: static assertion failed: invalid REG2FLOP byte offset for the selected GPR width

//--- runtime-two-families.cpp
#include <cstdint>

#include "hal/address_counters.h"

using hal::AddressChannel;
using hal::AddressCounterClient;

void runtime_two_families(const std::uint32_t x, const std::uint32_t z)
{
    hal::runtime_address_counters.client<AddressCounterClient::Unpacker0>().channel<AddressChannel::Channel0>().X(x).Z(z).apply();
}

// RUNTIME_TWO_FAMILIES: error: static assertion failed: runtime apply() must encode exactly one TT instruction (select X/Y or Z/W, not both)

//--- runtime-increment-two-families.cpp
#include <cstdint>

#include "hal/address_counters.h"

using hal::AddressChannel;
using hal::AddressCounterClient;

void runtime_increment_two_families(const std::uint32_t y, const std::uint32_t w)
{
    hal::runtime_address_counters.client<AddressCounterClient::Packers>().channel<AddressChannel::Channel1>().Y(y).W(w).increment();
}

// RUNTIME_INCREMENT_TWO_FAMILIES: error: static assertion failed: runtime increment() must encode exactly one TT instruction (select X/Y or Z/W, not both)

//--- runtime-no-client.cpp
#include <cstdint>

#include "hal/address_counters.h"

using hal::AddressChannel;
using hal::AddressCounterClient;

void runtime_no_client(const std::uint32_t x)
{
    hal::runtime_address_counters.channel<AddressChannel::Channel0>().X(x).apply();
}

// RUNTIME_NO_CLIENT: error: static assertion failed: no client selected — call client<...>() first
