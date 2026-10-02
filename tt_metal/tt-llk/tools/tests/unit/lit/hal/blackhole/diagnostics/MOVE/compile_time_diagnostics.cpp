// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
// RUN: %split-file %s %t
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/faces-zero.cpp 2>&1 | FileCheck %s --check-prefix=FACES_POSITIVE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/faces-overflow.cpp 2>&1 | FileCheck %s --check-prefix=FACES_FIT
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/relative-row-low.cpp 2>&1 | FileCheck %s --check-prefix=RELATIVE_ROW
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/relative-row-high.cpp 2>&1 | FileCheck %s --check-prefix=RELATIVE_ROW
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/srca-default.cpp 2>&1 | FileCheck %s --check-prefix=SRCA
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/srca-rows.cpp 2>&1 | FileCheck %s --check-prefix=SRCA
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/srca-source-span.cpp 2>&1 | FileCheck %s --check-prefix=SRCA
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/srca-destination-span.cpp 2>&1 | FileCheck %s --check-prefix=SRCA
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/srca-relative-field.cpp 2>&1 | FileCheck %s --check-prefix=SRCA
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/srca-address-mode.cpp 2>&1 | FileCheck %s --check-prefix=SRCA
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/srca-final-mode.cpp 2>&1 | FileCheck %s --check-prefix=SRCA
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/srca-expansion.cpp 2>&1 | FileCheck %s --check-prefix=SRCA
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/srcb-row-broadcast-one.cpp 2>&1 | FileCheck %s --check-prefix=SRCB
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/srcb-expansion.cpp 2>&1 | FileCheck %s --check-prefix=SRCB
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/srcb-broadcast-value.cpp 2>&1 | FileCheck %s --check-prefix=SRCB
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/dst-destination-span.cpp 2>&1 | FileCheck %s --check-prefix=DST
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/dst-source-span.cpp 2>&1 | FileCheck %s --check-prefix=DST
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/loadind-alignment.cpp 2>&1 | FileCheck %s --check-prefix=LOADIND
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/loadind-gpr.cpp 2>&1 | FileCheck %s --check-prefix=LOADIND
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/storeind-alignment.cpp 2>&1 | FileCheck %s --check-prefix=STOREIND
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/srca-store-alignment.cpp 2>&1 | FileCheck %s --check-prefix=STORE_SRCA
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/srcb-store-offset.cpp 2>&1 | FileCheck %s --check-prefix=STORE_SRCB
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/mmio-write-range.cpp 2>&1 | FileCheck %s --check-prefix=MMIO_WRITE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/mmio-write-addressing.cpp 2>&1 | FileCheck %s --check-prefix=MMIO_WRITE
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/mmio-read-indirect.cpp 2>&1 | FileCheck %s --check-prefix=MMIO_READ
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/xmov-overlap.cpp 2>&1 | FileCheck %s --check-prefix=XMOV
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/xmov-l1-end.cpp 2>&1 | FileCheck %s --check-prefix=XMOV
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/xmov-size.cpp 2>&1 | FileCheck %s --check-prefix=XMOV
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/constant-invalid-operation.cpp 2>&1 | FileCheck %s --check-prefix=CONSTANT_ENCODING
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/constant-multi-instruction.cpp 2>&1 | FileCheck %s --check-prefix=CONSTANT_ENCODING
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/constant-completion.cpp 2>&1 | FileCheck %s --check-prefix=CONSTANT_ENCODING
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/constant-relative-row.cpp 2>&1 | FileCheck %s --check-prefix=CONSTANT_ENCODING
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/xmov-operation.cpp 2>&1 | FileCheck %s --check-prefix=XMOV_OPERATION
// RUN: not %{blackhole_tensix_diagnose} %{blackhole_math_thread} %t/unsupported-pair.cpp 2>&1 | FileCheck %s --check-prefix=UNSUPPORTED_PAIR
// clang-format on

// Match an actual compiler error, not the echoed static_assert source text.
// An invalid descriptor encoded in a constant expression is rejected by a trap.
// FACES_POSITIVE: error: static assertion failed: face count must be positive
// FACES_FIT: error: static assertion failed: row count does not fit RowCount
// RELATIVE_ROW: error: static assertion failed: counter-relative row offset must fit ten signed bits
// SRCA: error: static assertion failed: invalid SrcA-to-Dst transfer descriptor
// SRCB: error: static assertion failed: invalid SrcB-to-Dst transfer descriptor
// DST: error: static assertion failed: invalid Dst-to-SrcB transfer descriptor
// LOADIND: error: static assertion failed: invalid L1-to-GPR transfer descriptor
// STOREIND: error: static assertion failed: invalid GPR-to-L1 transfer descriptor
// STORE_SRCA: error: static assertion failed: invalid GPR-to-SrcA transfer descriptor
// STORE_SRCB: error: static assertion failed: invalid GPR-to-SrcB transfer descriptor
// MMIO_WRITE: error: static assertion failed: invalid GPR-to-MMIO transfer descriptor
// MMIO_READ: error: static assertion failed: invalid MMIO-to-GPR transfer descriptor
// XMOV: error: static assertion failed: invalid L1-to-L1 XMOV transfer descriptor
// CONSTANT_ENCODING: error: '__builtin_trap()' is not a constant expression
// XMOV_OPERATION: error: use of deleted function {{.*}}get_operation
// UNSUPPORTED_PAIR: error: variable {{.*}}hal::move::Transfer<hal::move::Zero, hal::move::Dst> transfer' has initializer but incomplete type

//--- common.h
#pragma once

#include <cstdint>

#include "hal/move.h"

namespace move = hal::move;

using SrcAToDst = move::Transfer<move::SrcA, move::Dst>;
using SrcBToDst = move::Transfer<move::SrcB, move::Dst>;
using DstToSrcB = move::Transfer<move::Dst, move::SrcB>;

//--- faces-zero.cpp
#include "common.h"

constexpr move::RowCount rows = move::rows::faces<0>();

//--- faces-overflow.cpp
#include "common.h"

constexpr move::RowCount rows = move::rows::faces<4096>();

//--- relative-row-low.cpp
#include "common.h"

constexpr std::uint32_t row = move::relative_row<-513>();

//--- relative-row-high.cpp
#include "common.h"

constexpr std::uint32_t row = move::relative_row<512>();

//--- srca-default.cpp
#include "common.h"

void probe()
{
    move::run<SrcAToDst {}>();
}

//--- srca-rows.cpp
#include "common.h"

void probe()
{
    move::run<SrcAToDst {.source_row = 0, .destination_row = 0, .address_mode = 0, .number_of_rows = {2}}>();
}

//--- srca-source-span.cpp
#include "common.h"

void probe()
{
    move::run<SrcAToDst {.source_row = 60, .destination_row = 0, .address_mode = 0, .number_of_rows = move::rows::Eight}>();
}

//--- srca-destination-span.cpp
#include "common.h"

void probe()
{
    move::run<SrcAToDst {.source_row = 0, .destination_row = 1020, .address_mode = 0, .number_of_rows = move::rows::Eight}>();
}

//--- srca-relative-field.cpp
#include "common.h"

void probe()
{
    move::run<SrcAToDst {
        .source_row             = 0,
        .destination_row        = 1024,
        .destination_addressing = move::RowAddressing::CounterRelative,
        .address_mode           = 0,
        .number_of_rows         = move::rows::One,
    }>();
}

//--- srca-address-mode.cpp
#include "common.h"

void probe()
{
    move::run<SrcAToDst {.source_row = 0, .destination_row = 0, .address_mode = 8, .number_of_rows = move::rows::One}>();
}

//--- srca-final-mode.cpp
#include "common.h"

void probe()
{
    move::run<SrcAToDst {.source_row = 0, .destination_row = 0, .address_mode = 0, .number_of_rows = move::rows::Eight, .final_address_mode = 1}>();
}

//--- srca-expansion.cpp
#include "common.h"

void probe()
{
    move::run<SrcAToDst {.source_row = 0, .destination_row = 0, .address_mode = 0, .number_of_rows = move::rows::faces<5>()}>();
}

//--- srcb-row-broadcast-one.cpp
#include "common.h"

void probe()
{
    move::run<SrcBToDst {.source_row = 0, .destination_row = 0, .address_mode = 0, .number_of_rows = move::rows::One, .broadcast = move::Broadcast::Row}>();
}

//--- srcb-expansion.cpp
#include "common.h"

void probe()
{
    move::run<SrcBToDst {.source_row = 0, .destination_row = 0, .address_mode = 0, .number_of_rows = move::rows::faces<3>()}>();
}

//--- srcb-broadcast-value.cpp
#include "common.h"

void probe()
{
    move::run<SrcBToDst {
        .source_row      = 0,
        .destination_row = 0,
        .address_mode    = 0,
        .number_of_rows  = move::rows::One,
        .broadcast       = static_cast<move::Broadcast>(4),
    }>();
}

//--- dst-destination-span.cpp
#include "common.h"

void probe()
{
    move::run<DstToSrcB {.source_row = 0, .destination_row = 60, .address_mode = 0, .number_of_rows = move::rows::Eight}>();
}

//--- dst-source-span.cpp
#include "common.h"

void probe()
{
    move::run<DstToSrcB {.source_row = 1020, .destination_row = 0, .address_mode = 0, .number_of_rows = move::rows::Face}>();
}

//--- loadind-alignment.cpp
#include "common.h"

void probe()
{
    move::run<move::Transfer<move::L1, move::Gpr> {
        .source      = {.base = hal::gpr<0>(), .offset = {.gpr = hal::gpr<1>()}},
        .destination = hal::gpr<2>(),
        .size        = move::ScalarSize::Bytes16,
    }>();
}

//--- loadind-gpr.cpp
#include "common.h"

void probe()
{
    move::run<move::Transfer<move::L1, move::Gpr> {
        .source      = {.base = hal::gpr<64>(), .offset = {.gpr = hal::gpr<1>()}},
        .destination = hal::gpr<2>(),
        .size        = move::ScalarSize::Bytes4,
    }>();
}

//--- storeind-alignment.cpp
#include "common.h"

void probe()
{
    move::run<move::Transfer<move::Gpr, move::L1> {
        .source      = hal::gpr<6>(),
        .destination = {.base = hal::gpr<0>(), .offset = {.gpr = hal::gpr<1>()}},
    }>();
}

//--- srca-store-alignment.cpp
#include "common.h"

void probe()
{
    move::run<move::Transfer<move::Gpr, move::SrcA> {
        .source      = hal::gpr<1>(),
        .destination = {.base = hal::gpr<0>(), .offset = {.gpr = hal::gpr<1>()}},
    }>();
}

//--- srcb-store-offset.cpp
#include "common.h"

void probe()
{
    move::run<move::Transfer<move::Gpr, move::SrcB> {
        .source      = hal::gpr<4>(),
        .destination = {.base = hal::gpr<0>()},
    }>();
}

//--- mmio-write-range.cpp
#include "common.h"

void probe()
{
    move::run<move::Transfer<move::Gpr, move::Mmio> {
        .source      = hal::gpr<0>(),
        .destination = {.addressing = move::MmioAddressing::Immediate, .byte_address = 0xffb10ffcu},
    }>();
}

//--- mmio-write-addressing.cpp
#include "common.h"

void probe()
{
    move::run<move::Transfer<move::Gpr, move::Mmio> {.source = hal::gpr<0>()}>();
}

//--- mmio-read-indirect.cpp
#include "common.h"

void probe()
{
    move::run<move::Transfer<move::Mmio, move::Gpr> {
        .source      = {.addressing = move::MmioAddressing::Indirect, .base = hal::gpr<0>(), .offset = {.gpr = hal::gpr<1>()}},
        .destination = hal::gpr<2>(),
    }>();
}

//--- xmov-overlap.cpp
#include "common.h"

void probe()
{
    move::run<move::Transfer<move::L1, move::L1> {.source = {.block = 0x100}, .destination = {.block = 0x10f}, .size = {.value = 16}}>();
}

//--- xmov-l1-end.cpp
#include "common.h"

void probe()
{
    move::run<move::Transfer<move::L1, move::L1> {.source = {.block = MEM_L1_SIZE / 16u - 1u}, .destination = {.block = 0}, .size = {.value = 2}}>();
}

//--- xmov-size.cpp
#include "common.h"

void probe()
{
    move::run<move::Transfer<move::L1, move::L1> {.source = {.block = 0}, .destination = {.block = 0x10000}, .size = {.value = 0x10000}}>();
}

//--- constant-invalid-operation.cpp
#include "common.h"

constexpr std::uint32_t operation = SrcBToDst {.source_row = 64, .destination_row = 0, .address_mode = 0, .number_of_rows = move::rows::One}.get_operation();

//--- constant-multi-instruction.cpp
#include "common.h"

constexpr std::uint32_t operation = DstToSrcB {.source_row = 0, .destination_row = 0, .address_mode = 0, .number_of_rows = move::rows::Eight}.get_operation();

//--- constant-completion.cpp
#include "common.h"

constexpr std::uint32_t operation =
    move::Transfer<move::Mmio, move::Gpr> {
        .source      = {.addressing = move::MmioAddressing::Immediate, .byte_address = 0xffb11000u},
        .destination = hal::gpr<0>(),
        .completion  = move::Completion::Wait,
    }
        .get_operation();

//--- constant-relative-row.cpp
#include "common.h"

constexpr std::uint32_t row = move::relative_row(-513);

//--- xmov-operation.cpp
#include "common.h"

constexpr std::uint32_t operation =
    move::Transfer<move::L1, move::L1> {.source = {.block = 0}, .destination = {.block = 1}, .size = {.value = 1}}.get_operation();

//--- unsupported-pair.cpp
#include "common.h"

constexpr move::Transfer<move::Zero, move::Dst> transfer {};
