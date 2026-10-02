// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_pack_thread} -c %s -o %t.o
// RUN: %{blackhole_objdump} -d %t.o | FileCheck %s

#include "hal/utils/address_modifier.h"

using hal::cfg::Sec;
using hal::pack::AddressModifier;
using hal::pack::configure_address_modifiers;

// Each selection is one SETC16 to its thread-CFG word. Bit layout: source row
// step 3:0, clear 5; destination row step 9:6, clear 11; source face step 12,
// clear 13; destination face step 14, clear 15.

extern "C" __attribute__((noinline, used)) void configure_default_modifier()
{
    configure_address_modifiers<AddressModifier {}>();
}

// CHECK-LABEL: <configure_default_modifier>:
// CHECK-NEXT: ttsetc16 37,0
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void configure_two_modifiers()
{
    constexpr AddressModifier rows {
        .selection = Sec::S1, .source_row = {.step = 3}, .destination_row = {.step = 5, .clear = true}, .source_face = {.clear = true}};
    constexpr AddressModifier faces {.selection = Sec::S2, .source_row = {.step = 15, .clear = true}, .destination_face = {.step = 1, .clear = true}};

    configure_address_modifiers<rows, faces>();
}

// 0x2943 and 0xc02f.
// CHECK-LABEL: <configure_two_modifiers>:
// CHECK-NEXT: ttsetc16 38,10563
// CHECK-NEXT: ttsetc16 39,49199
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void configure_all_selections()
{
    constexpr AddressModifier s3 {.selection = Sec::S3, .source_face = {.step = 1}};
    constexpr AddressModifier s0 {.selection = Sec::S0, .destination_row = {.step = 15}};
    constexpr AddressModifier s2 {.selection = Sec::S2, .destination_row = {.clear = true}};
    constexpr AddressModifier s1 {.selection = Sec::S1, .source_row = {.clear = true}};

    configure_address_modifiers<s3, s0, s2, s1>();
}

// CHECK-LABEL: <configure_all_selections>:
// CHECK-NEXT: ttsetc16 40,4096
// CHECK-NEXT: ttsetc16 37,960
// CHECK-NEXT: ttsetc16 39,2048
// CHECK-NEXT: ttsetc16 38,32
// CHECK-NEXT: ret
