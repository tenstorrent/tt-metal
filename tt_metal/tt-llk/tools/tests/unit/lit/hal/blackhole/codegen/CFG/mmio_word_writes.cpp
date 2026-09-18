// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_pack_thread} -S %s -o %t.s
// RUN: FileCheck %s --enable-var-scope < %t.s

#include <cstdint>

#include "hal/cfg.h"

namespace cfg = hal::cfg;

// The three complete-word stores from configure_pack() on Blackhole (one packer).
// At -O3, independent writes must share one bank lookup and emit no array storage,
// calls, or register read-modify-write operations, matching the former batch API.
extern "C" __attribute__((noinline, used)) void write_pack_words_constant()
{
    constexpr std::uint32_t counters[] = {0x100}; // pack_reads_per_xy_plane = 1; other bits zero.
    constexpr std::uint32_t edges[]    = {0xffff};
    constexpr std::uint32_t mappings[] = {0};

    cfg::write<cfg::Access::MMIO, cfg::PackCounters::pack_per_xy_plane, cfg::Sec::S0, 1>(counters);
    cfg::write<cfg::Access::MMIO, cfg::PckEdgeOffsetSec0::mask, cfg::Sec::S0, 1>(edges);
    cfg::write<cfg::Access::MMIO, cfg::TileRowSetMapping[0][0], cfg::Sec::S0, 1>(mappings);
}

// CHECK-LABEL: write_pack_words_constant:
// CHECK-NEXT: lui [[STATE:a[0-7]]],%hi(_ZN7ckernel12cfg_state_idE)
// CHECK-NEXT: lw [[ID:a[0-7]]],%lo(_ZN7ckernel12cfg_state_idE)([[STATE]])
// CHECK-NEXT: li [[BASE:a[0-7]]],-1114112
// CHECK-NEXT: addi [[BANK:a[0-7]]],[[BASE]],896
// CHECK-NEXT: bne [[ID]],zero,[[SELECTED:\.L[0-9]+]]
// CHECK-NEXT: mv [[BANK]],[[BASE]]
// CHECK-NEXT: [[SELECTED]]:
// CHECK-NEXT: li [[COUNTERS:a[0-7]]],256
// CHECK-NEXT: li [[EDGES:a[0-7]]],65536
// CHECK-NEXT: sw [[COUNTERS]],112([[BANK]])
// CHECK-NEXT: addi [[EDGES]],[[EDGES]],-1
// CHECK-NEXT: sw [[EDGES]],96([[BANK]])
// CHECK-NEXT: sw zero,80([[BANK]])
// CHECK-NEXT: ret
// CHECK-NEXT: .size write_pack_words_constant,

// Runtime values must also pass directly to the three stores without a batch.
extern "C" __attribute__((noinline, used)) void write_pack_words_runtime(std::uint32_t counter, std::uint32_t edge, std::uint32_t mapping)
{
    const std::uint32_t counters[] = {counter};
    const std::uint32_t edges[]    = {edge};
    const std::uint32_t mappings[] = {mapping};

    cfg::write<cfg::Access::MMIO, cfg::PackCounters::pack_per_xy_plane, cfg::Sec::S0, 1>(counters);
    cfg::write<cfg::Access::MMIO, cfg::PckEdgeOffsetSec0::mask, cfg::Sec::S0, 1>(edges);
    cfg::write<cfg::Access::MMIO, cfg::TileRowSetMapping[0][0], cfg::Sec::S0, 1>(mappings);
}

// CHECK-LABEL: write_pack_words_runtime:
// CHECK-NEXT: lui [[STATE:a[0-7]]],%hi(_ZN7ckernel12cfg_state_idE)
// CHECK-NEXT: lw [[ID:a[0-7]]],%lo(_ZN7ckernel12cfg_state_idE)([[STATE]])
// CHECK-NEXT: li [[BASE:a[0-7]]],-1114112
// CHECK-NEXT: addi [[BANK:a[0-7]]],[[BASE]],896
// CHECK-NEXT: bne [[ID]],zero,[[SELECTED:\.L[0-9]+]]
// CHECK-NEXT: mv [[BANK]],[[BASE]]
// CHECK-NEXT: [[SELECTED]]:
// CHECK-NEXT: sw a0,112([[BANK]])
// CHECK-NEXT: sw a1,96([[BANK]])
// CHECK-NEXT: sw a2,80([[BANK]])
// CHECK-NEXT: ret
// CHECK-NEXT: .size write_pack_words_runtime,
