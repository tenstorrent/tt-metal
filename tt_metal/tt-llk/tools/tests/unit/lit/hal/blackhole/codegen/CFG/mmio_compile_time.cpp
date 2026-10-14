// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_pack_thread} -fno-ipa-icf -c %s -o %t.o
// RUN: %{blackhole_objdump} -drz %t.o > %t.dump
// RUN: %{blackhole_compare_codegen} %t.dump

// Each HAL function is compared with its adjacent reference_ function.
// Disable identical-code folding so both bodies must be emitted independently.

#include <array>
#include <cstdint>

#include "hal/cfg.h"

namespace cfg = hal::cfg;

// Independent hardware-layout constant for the reference MMIO operations.
inline constexpr std::uint32_t StateBankWords = 224;

inline volatile std::uint32_t tt_reg_ptr* reference_cfg_bank()
{
    const std::uint32_t bank_offset = (ckernel::cfg_state_id == 0) ? 0u : StateBankWords;
    return reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(TENSIX_CFG_BASE) + bank_offset;
}

// Merge both fields in word 0 before setting the bit in word 5.
extern "C" __attribute__((noinline, used)) void write_constant_group()
{
    cfg::write<cfg::Access::MMIO>(
        cfg::set<cfg::AluFormatSpecReg::SrcA_val, cfg::Sec::S0, 5>(),
        cfg::set<cfg::DestOffset::Enable, cfg::Sec::S0, 1>(),
        cfg::set<cfg::AluFormatSpecReg::SrcB_val, cfg::Sec::S0, 7>());
}

extern "C" __attribute__((noinline, used)) void reference_write_constant_group()
{
    auto* bank = reference_cfg_bank();
    bank[0]    = (bank[0] & ~0x1efu) | 229u;
    bank[5] |= 1u;
}

// A full-word assignment stores word 186 without reading it.
extern "C" __attribute__((noinline, used)) void write_constant_full_word()
{
    cfg::write<cfg::Access::MMIO>(cfg::set<cfg::PrngSeed::Seed_Val, cfg::Sec::S0, 0x12345678>());
}

extern "C" __attribute__((noinline, used)) void reference_write_constant_full_word()
{
    reference_cfg_bank()[186] = 0x12345678u;
}

// The three complete-word stores from configure_pack() on Blackhole (one packer).
// Independent writes share one bank lookup and need no array storage or reads.
extern "C" __attribute__((noinline, used)) void write_pack_words_constant()
{
    constexpr std::array<std::uint32_t, 1> counters = {0x100}; // pack_reads_per_xy_plane = 1; other bits zero.
    constexpr std::array<std::uint32_t, 1> edges    = {0xffff};
    constexpr std::array<std::uint32_t, 1> mappings = {0};

    cfg::write<cfg::Access::MMIO, cfg::PackCounters::pack_per_xy_plane, cfg::Sec::S0, 1>(counters);
    cfg::write<cfg::Access::MMIO, cfg::PckEdgeOffsetSec0::mask, cfg::Sec::S0, 1>(edges);
    cfg::write<cfg::Access::MMIO, cfg::TileRowSetMapping[0][0], cfg::Sec::S0, 1>(mappings);
}

extern "C" __attribute__((noinline, used)) void reference_write_pack_words_constant()
{
    auto* bank = reference_cfg_bank();
    bank[28]   = 0x100u;
    bank[24]   = 0xffffu;
    bank[20]   = 0u;
}
