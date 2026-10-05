// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_pack_thread} -S %s -o %t.s
// RUN: FileCheck %s --enable-var-scope < %t.s

#include <array>
#include <cstdint>

#include "hal/cfg.h"

namespace cfg = hal::cfg;

// Every MMIO write first selects the state bank: bank 1 (base + 896 bytes)
// unless cfg_state_id is zero. Each byte offset below is 4 * word.

extern "C" __attribute__((noinline, used)) void write_constant_group()
{
    cfg::write<cfg::Access::MMIO>(
        cfg::set<cfg::AluFormatSpecReg::SrcA_val, cfg::Sec::S0, 5>(),
        cfg::set<cfg::DestOffset::Enable, cfg::Sec::S0, 1>(),
        cfg::set<cfg::AluFormatSpecReg::SrcB_val, cfg::Sec::S0, 7>());
}

// Word 0 merges both fields into one read-modify-write (clear mask 0x1ef, data
// 229) before word 5, which only needs its bit set.
// CHECK-LABEL: {{^}}write_constant_group:
// CHECK-NEXT: lui [[STATE:a[0-7]]],%hi(_ZN7ckernel12cfg_state_idE)
// CHECK-NEXT: lw [[ID:a[0-7]]],%lo(_ZN7ckernel12cfg_state_idE)([[STATE]])
// CHECK-NEXT: li [[BASE:a[0-7]]],-1114112
// CHECK-NEXT: addi [[BANK:a[0-7]]],[[BASE]],896
// CHECK-NEXT: bne [[ID]],zero,[[SELECTED:\.L[0-9]+]]
// CHECK-NEXT: mv [[BANK]],[[BASE]]
// CHECK-NEXT: [[SELECTED]]:
// CHECK-NEXT: lw [[W0:a[0-7]]],0([[BANK]])
// CHECK-NEXT: andi [[W0]],[[W0]],-496
// CHECK-NEXT: ori [[W0]],[[W0]],229
// CHECK-NEXT: sw [[W0]],0([[BANK]])
// CHECK-NEXT: lw [[W5:a[0-7]]],20([[BANK]])
// CHECK-NEXT: ori [[W5]],[[W5]],1
// CHECK-NEXT: sw [[W5]],20([[BANK]])
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_constant_full_word()
{
    cfg::write<cfg::Access::MMIO>(cfg::set<cfg::PrngSeed::Seed_Val, cfg::Sec::S0, 0x12345678>());
}

// A full-word assignment stores word 186 without reading it.
// CHECK-LABEL: {{^}}write_constant_full_word:
// CHECK: %lo(_ZN7ckernel12cfg_state_idE)
// CHECK: [[SELECTED:\.L[0-9]+]]:
// CHECK-NEXT: li [[VALUE:a[0-7]]],305418240
// CHECK-NEXT: addi [[VALUE]],[[VALUE]],1656
// CHECK-NEXT: sw [[VALUE]],744({{a[0-7]}})
// CHECK-NEXT: ret

// The three complete-word stores from configure_pack() on Blackhole (one packer).
// At -O3, independent writes must share one bank lookup and emit no array storage,
// calls, or register read-modify-write operations, matching the former batch API.
extern "C" __attribute__((noinline, used)) void write_pack_words_constant()
{
    constexpr std::array<std::uint32_t, 1> counters = {0x100}; // pack_reads_per_xy_plane = 1; other bits zero.
    constexpr std::array<std::uint32_t, 1> edges    = {0xffff};
    constexpr std::array<std::uint32_t, 1> mappings = {0};

    cfg::write<cfg::Access::MMIO, cfg::PackCounters::pack_per_xy_plane, cfg::Sec::S0, 1>(counters);
    cfg::write<cfg::Access::MMIO, cfg::PckEdgeOffsetSec0::mask, cfg::Sec::S0, 1>(edges);
    cfg::write<cfg::Access::MMIO, cfg::TileRowSetMapping[0][0], cfg::Sec::S0, 1>(mappings);
}

// CHECK-LABEL: {{^}}write_pack_words_constant:
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
