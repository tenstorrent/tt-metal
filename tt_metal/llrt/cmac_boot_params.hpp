// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Boot-parameter block written to ethernet-core L1 before releasing
// erisc_cmac_simple (budabackend) from reset.
//
// The erisc_cmac_simple firmware reads a 256-byte (64-word) block from
// L1 address 0x1000 immediately after reset.  Field layout matches
// boot_params_t in:
//   budabackend/src/firmware/riscv/targets/erisc_cmac_simple/src/api/eth_cmac_init.h
//
// Only the three fields relevant to external-link bringup are written;
// all others are zero-initialised (safe / no-op for the firmware).

#pragma once

#include <array>
#include <cstdint>
#include <cstddef>

namespace llrt {

struct CmacBootParams {
    // L1 address where the firmware reads the boot-parameter block.
    static constexpr uint64_t kL1Address = 0x1000;

    // Size of the boot-parameter block in bytes (64 x uint32_t = 256 B).
    static constexpr size_t kSizeBytes = 256;
    static constexpr size_t kSizeWords = kSizeBytes / sizeof(uint32_t);  // 64

    // Build a zero-initialised boot-parameter word array and fill the
    // three fields relevant to external-link bringup:
    //   aiclk_ps        – word[18] — AI-clock period in picoseconds
    //                     (~833 @ 1200 MHz, ~714 @ 1400 MHz).
    //   rs_fec_enabled  – word[3]  — 1 = RS-FEC on, 0 = off.
    //   tx_rate_cycles  – word[61] — inter-burst gap in AI-clock cycles;
    //                     0 = full line rate.
    static std::array<uint32_t, kSizeWords> build(uint32_t aiclk_ps, bool rs_fec_enabled, uint32_t tx_rate_cycles);
};

}  // namespace llrt
