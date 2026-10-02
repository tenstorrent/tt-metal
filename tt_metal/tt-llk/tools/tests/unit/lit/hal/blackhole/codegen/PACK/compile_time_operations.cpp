// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_pack_thread} -c %s -o %t.o
// RUN: %{blackhole_objdump} -d %t.o | FileCheck %s --enable-var-scope

#include <cstdint>

#include "hal/pack.h"

namespace pack = hal::pack;

extern "C" __attribute__((noinline, used)) void transfer_defaults()
{
    pack::run<pack::DataTransfer {}>();
}

// CHECK-LABEL: <transfer_defaults>:
// CHECK-NEXT: ttpacr 0,0,0,0,0,0,0,0,0,0,0,0
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void transfer_every_field()
{
    pack::run<pack::DataTransfer {
        .address_modifier      = 3,
        .context               = pack::ContextControl::HardwareCounterNoAdvance,
        .configuration_context = 3,
        .counter_context       = 2,
        .override_thread_id    = true,
        .interfaces            = 0xf,
        .datum_override        = pack::DatumOverride::Zero,
        .dest_access           = pack::DestAccess::Strided,
        .padding               = pack::RowPadding::FinalOnly,
        .alignment             = pack::PaddingAlignment::To16Datums,
        .concatenation         = pack::Concatenation::Append,
        .boundary              = pack::TileBoundary::Last,
    }>();
}

// CHECK-LABEL: <transfer_every_field>:
// CHECK-NEXT: ttpacr 3,7,1,3,2,1,15,1,1,3,0,1
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void transfer_context_controls()
{
    pack::run<pack::DataTransfer {.context = pack::ContextControl::HardwareCounter, .interfaces = 0b0101}>();
    pack::run<pack::DataTransfer {.context = pack::ContextControl::HardwareCounterReset}>();
    pack::run<pack::DataTransfer {.padding = pack::RowPadding::AllTransfers, .boundary = pack::TileBoundary::Last}>();
    pack::run<pack::DataTransfer {.padding = pack::RowPadding::NonConcatenated, .alignment = pack::PaddingAlignment::To16Datums}>();
}

// CHECK-LABEL: <transfer_context_controls>:
// CHECK-NEXT: ttpacr 0,0,0,0,0,0,5,0,0,1,0,0
// CHECK-NEXT: ttpacr 0,0,0,0,0,0,0,0,0,2,0,0
// CHECK-NEXT: ttpacr 0,1,0,0,0,0,0,0,0,0,0,1
// CHECK-NEXT: ttpacr 0,6,0,0,0,0,0,0,0,0,0,0
// CHECK-NEXT: ret

// The value is staged in two 16-bit halves before the write consumes it.
extern "C" __attribute__((noinline, used)) void register_write()
{
    pack::run<pack::RegisterWriteValue {.value = 0x1234}>();
    pack::run<pack::RegisterWriteValue {.high_half = true, .value = 0xabcd}>();
    pack::run<pack::RegisterWrite {.address_slot = 3, .stream_id = 63}>();
}

// CHECK-LABEL: <register_write>:
// CHECK-NEXT: 29024681 ttpacrsetreg
// CHECK-NEXT: 291579b1 ttpacrsetreg
// CHECK-NEXT: ttpacrsetreg 1,0,0,0,3,63,1,0
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void edge_window()
{
    pack::run<pack::EdgeWindow {.x_start = 1, .x_end = 2, .y_start = 3, .y_end = 15}>();
}

// CHECK-LABEL: <edge_window>:
// CHECK-NEXT: ttsetpkedgof 15,3,2,1
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void flush_and_clear()
{
    pack::flush_write_aligners();
    pack::clear_exponent_histogram();
}

// CHECK-LABEL: <flush_and_clear>:
// CHECK-NEXT: ttpacr 0,0,0,0,0,0,0,0,0,0,1,0
// CHECK-NEXT: ttclrexphist
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void store_encoded_operations(std::uint32_t* words)
{
    constexpr std::uint32_t transfer = pack::DataTransfer {.interfaces = 0b0011, .boundary = pack::TileBoundary::Last}.get_operation();
    constexpr std::uint32_t write    = pack::RegisterWrite {.address_slot = 1}.get_operation();
    constexpr std::uint32_t value    = pack::RegisterWriteValue {.value = 0xffff}.get_operation();
    constexpr std::uint32_t window   = pack::EdgeWindow {.y_end = 1}.get_operation();
    words[0]                         = transfer;
    words[1]                         = write;
    words[2]                         = value;
    words[3]                         = window;
    words[4]                         = pack::flush_write_aligners_operation();
    words[5]                         = pack::clear_exponent_histogram_operation();
}

// CHECK-LABEL: <store_encoded_operations>:
// CHECK-DAG: lui [[PACR:a[0-7]]],0x41000
// CHECK-DAG: addi [[TRANSFER:a[0-7]]],[[PACR]],769
// CHECK-DAG: lui [[WRITE:a[0-7]]],0x4a800
// CHECK-DAG: addi [[WRITE]],[[WRITE]],258
// CHECK-DAG: lui [[VALUE:a[0-7]]],0x4a480
// CHECK-DAG: addi [[VALUE]],[[VALUE]],-8
// CHECK-DAG: lui [[WINDOW:a[0-7]]],0x1d001
// CHECK-DAG: addi [[FLUSH:a[0-7]]],[[PACR]],2
// CHECK-DAG: lui [[CLEAR:a[0-7]]],0x21000
// CHECK-DAG: sw [[TRANSFER]],0(a0)
// CHECK-DAG: sw [[WRITE]],4(a0)
// CHECK-DAG: sw [[VALUE]],8(a0)
// CHECK-DAG: sw [[WINDOW]],12(a0)
// CHECK-DAG: sw [[FLUSH]],16(a0)
// CHECK-DAG: sw [[CLEAR]],20(a0)
// CHECK: ret
