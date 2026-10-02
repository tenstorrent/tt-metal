// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -c %s -o %t.o
// RUN: %{blackhole_objdump} -d %t.o | FileCheck %s

#include <cstdint>

#include "hal/src.h"

extern "C" __attribute__((noinline, used)) std::uint32_t srca_bits()
{
    return hal::source_bits(hal::SrcA);
}

// CHECK-LABEL: <srca_bits>:
// CHECK-NEXT: li a0,1
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) std::uint32_t srcb_bits()
{
    return hal::source_bits(hal::SrcB);
}

// CHECK-LABEL: <srcb_bits>:
// CHECK-NEXT: li a0,2
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) std::uint32_t both_sources_bits()
{
    return hal::source_bits(hal::BothSources);
}

// CHECK-LABEL: <both_sources_bits>:
// CHECK-NEXT: li a0,3
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void fill_union_of_sources()
{
    hal::src<hal::SrcA | hal::SrcB>.fill();
}

// CHECK-LABEL: <fill_union_of_sources>:
// CHECK-NEXT: ttzerosrc 0,1,0,3
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void release_intersection_of_sources()
{
    hal::src<hal::BothSources & hal::SrcB>.release();
}

// CHECK-LABEL: <release_intersection_of_sources>:
// CHECK-NEXT: ttcleardvalid 2,0
// CHECK-NEXT: ret
