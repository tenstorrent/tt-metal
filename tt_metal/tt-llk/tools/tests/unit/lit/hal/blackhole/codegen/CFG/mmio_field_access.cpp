// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_unpack_thread} -S %s -o %t.t0.s
// RUN: FileCheck %s --enable-var-scope --check-prefixes=CHECK,T0 < %t.t0.s
// RUN: %{blackhole_tensix_compile} %{blackhole_pack_thread} -S %s -o %t.t2.s
// RUN: FileCheck %s --enable-var-scope --check-prefixes=CHECK,T2 < %t.t2.s

#include <array>
#include <cstdint>

#include "hal/cfg.h"

namespace cfg = hal::cfg;

// State words are addressed from the bank selected by cfg_state_id (see
// mmio_word_writes.cpp), so each byte offset below is 4 * word. Thread words
// are read through the debug CREG_READ (88) and CREG_RDDATA (120) registers
// with index 672 + 68 * thread + word.

inline constexpr cfg::Field state_first {cfg::RegisterScope::State, 32, 0, 0, 0, 32, 1, 0};

// The same 64-bit field spans words 64-66 in S0 (bit 16) and 67-68 in S1 (bit 0).
inline constexpr cfg::Field state_sectioned_wide {cfg::RegisterScope::State, 32, 64, 0, 16, 64, 2, 80};

extern "C" __attribute__((noinline, used)) void write_field_group(std::uint32_t format)
{
    cfg::write<cfg::Access::MMIO>(
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.InDataFormat, cfg::Sec::S0>(format),
        cfg::set<cfg::Thcon[cfg::Reg0].TileDescriptor.Uncompressed, cfg::Sec::S0, 1>());
}

// Both fields of word 64 merge into one read-modify-write.
// CHECK-LABEL: {{^}}write_field_group:
// CHECK: %lo(_ZN7ckernel12cfg_state_idE)
// CHECK-DAG: lw [[OLD:a[0-7]]],256([[BANK:a[0-7]]])
// CHECK-DAG: andi a0,a0,15
// CHECK: andi [[OLD]],[[OLD]],-32
// CHECK: or [[DATA:a[0-7]]],{{a[0-7]}},{{a[0-7]}}
// CHECK: ori [[DATA]],[[DATA]],16
// CHECK: sw [[DATA]],256([[BANK]])
// CHECK-NOT: sw
// CHECK: ret

extern "C" __attribute__((noinline, used)) void write_last_state_words(const std::array<std::uint32_t, 2>& values)
{
    cfg::write<cfg::Access::MMIO, cfg::ChickenBits::sfpu_scbd_disable, cfg::Sec::S0, 2>(values);
}

// Words 222-223 end exactly at the bank boundary.
// CHECK-LABEL: {{^}}write_last_state_words:
// CHECK: %lo(_ZN7ckernel12cfg_state_idE)
// CHECK: sw {{a[0-7]}},888([[BANK:a[0-7]]])
// CHECK: sw {{a[0-7]}},892([[BANK]])
// CHECK-NOT: sw
// CHECK: ret

extern "C" __attribute__((noinline, used)) void write_full_state_bank(const std::array<std::uint32_t, 224>& values)
{
    cfg::write<cfg::Access::MMIO, state_first, cfg::Sec::S0, 224>(values);
}

// CHECK-LABEL: {{^}}write_full_state_bank:
// CHECK: %lo(_ZN7ckernel12cfg_state_idE)
// CHECK: addi [[END:a[0-7]]],a0,896
// CHECK: [[LOOP:\.L[0-9]+]]:
// CHECK: sw {{a[0-7]}},0({{a[0-7]}})
// CHECK: bne a0,[[END]],[[LOOP]]
// CHECK-NOT: sw
// CHECK: ret

extern "C" __attribute__((noinline, used)) std::uint32_t read_last_state_word()
{
    return cfg::read_word<cfg::Access::MMIO, cfg::PackGlobalCfgCtl::pack_disable_fast_tile_end_drain, cfg::Sec::S0, 183>();
}

// CHECK-LABEL: {{^}}read_last_state_word:
// CHECK: %lo(_ZN7ckernel12cfg_state_idE)
// CHECK: lw a0,892({{a[0-7]}})
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) std::uint32_t read_last_thread_word()
{
    return cfg::read_word<cfg::Access::MMIO, cfg::SrcASet::Base, cfg::Sec::S0, 62>();
}

// Word 67 is the last word of the current thread's bank.
// CHECK-LABEL: {{^}}read_last_thread_word:
// CHECK: li [[DEBUG:a[0-7]]],-5169152
// T0: li [[INDEX:a[0-7]]],739
// T2: li [[INDEX:a[0-7]]],875
// CHECK: sw [[INDEX]],88([[DEBUG]])
// CHECK: lw a0,120({{a[0-7]}})
// CHECK-NEXT: zext.h a0,a0
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) std::uint32_t read_other_thread_field()
{
    return cfg::read<cfg::Access::MMIO, cfg::SrcASet::Base, cfg::Sec::S0, cfg::ThreadTarget::T2>();
}

// ThreadTarget::T2 selects thread 2's bank from any thread.
// CHECK-LABEL: {{^}}read_other_thread_field:
// CHECK: li [[DEBUG:a[0-7]]],-5169152
// CHECK: li [[INDEX:a[0-7]]],813
// CHECK: sw [[INDEX]],88([[DEBUG]])
// CHECK: lw a0,120({{a[0-7]}})
// CHECK-NEXT: andi a0,a0,3
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_field_group_words(const std::array<std::uint32_t, 4>& values)
{
    cfg::write<cfg::Access::MMIO, cfg::Thcon[cfg::Reg0].TileDescriptor, cfg::Sec::S1, 4>(values);
}

// A field group anchors whole-word access on its Raw field (S1 at word 112).
// CHECK-LABEL: {{^}}write_field_group_words:
// CHECK: %lo(_ZN7ckernel12cfg_state_idE)
// CHECK: sw {{a[0-7]}},448([[BANK:a[0-7]]])
// CHECK: sw {{a[0-7]}},452([[BANK]])
// CHECK: sw {{a[0-7]}},456([[BANK]])
// CHECK: sw {{a[0-7]}},460([[BANK]])
// CHECK-NOT: sw
// CHECK: ret

extern "C" __attribute__((noinline, used)) std::uint32_t read_field_group_word()
{
    return cfg::read_word<cfg::Access::MMIO, cfg::Thcon[cfg::Reg0].TileDescriptor, cfg::Sec::S0, 3>();
}

// CHECK-LABEL: {{^}}read_field_group_word:
// CHECK: %lo(_ZN7ckernel12cfg_state_idE)
// CHECK: lw a0,268({{a[0-7]}})
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_shifted_wide_field_words(const std::array<std::uint32_t, 3>& values)
{
    cfg::write<cfg::Access::MMIO, state_sectioned_wide, cfg::Sec::S0, 3>(values);
}

// Whole-word access covers every word the selected section occupies.
// CHECK-LABEL: {{^}}write_shifted_wide_field_words:
// CHECK: %lo(_ZN7ckernel12cfg_state_idE)
// CHECK: sw {{a[0-7]}},256([[BANK:a[0-7]]])
// CHECK: sw {{a[0-7]}},260([[BANK]])
// CHECK: sw {{a[0-7]}},264([[BANK]])
// CHECK-NOT: sw
// CHECK: ret

extern "C" __attribute__((noinline, used)) std::uint32_t read_shifted_wide_field_word()
{
    return cfg::read_word<cfg::Access::MMIO, state_sectioned_wide, cfg::Sec::S0, 2>();
}

// CHECK-LABEL: {{^}}read_shifted_wide_field_word:
// CHECK: %lo(_ZN7ckernel12cfg_state_idE)
// CHECK: lw a0,264({{a[0-7]}})
// CHECK-NEXT: ret

extern "C" __attribute__((noinline, used)) void write_aligned_section_words(const std::array<std::uint32_t, 2>& values)
{
    cfg::write<cfg::Access::MMIO, state_sectioned_wide, cfg::Sec::S1, 2>(values);
}

// CHECK-LABEL: {{^}}write_aligned_section_words:
// CHECK: %lo(_ZN7ckernel12cfg_state_idE)
// CHECK: sw {{a[0-7]}},268([[BANK:a[0-7]]])
// CHECK: sw {{a[0-7]}},272([[BANK]])
// CHECK-NOT: sw
// CHECK: ret

extern "C" __attribute__((noinline, used)) std::uint32_t read_aligned_section_word()
{
    return cfg::read_word<cfg::Access::MMIO, state_sectioned_wide, cfg::Sec::S1, 1>();
}

// CHECK-LABEL: {{^}}read_aligned_section_word:
// CHECK: %lo(_ZN7ckernel12cfg_state_idE)
// CHECK: lw a0,272({{a[0-7]}})
// CHECK-NEXT: ret
