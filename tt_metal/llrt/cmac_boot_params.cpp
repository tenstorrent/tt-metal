// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Implementation of CmacBootParams::build().
//
// Field positions match boot_params_t in budabackend:
//   budabackend/src/firmware/riscv/targets/erisc_cmac_simple/src/api/eth_cmac_init.h
//
// Any word not assigned here is zero-initialised (safe / no-op for the firmware).

#include "llrt/cmac_boot_params.hpp"

namespace llrt {

std::array<uint32_t, CmacBootParams::kSizeWords> CmacBootParams::build(
    uint32_t aiclk_ps, bool rs_fec_enabled, uint32_t tx_rate_cycles) {
    std::array<uint32_t, kSizeWords> words{};  // zero-init

    // word[3]: rs_fec_en — 1 = RS-FEC on (clause 91/108), 0 = off.
    words[3] = rs_fec_enabled ? 1u : 0u;

    // word[18]: aiclk_ps — AI-clock period in picoseconds.
    // Firmware uses this for link-timing windows (PCS lock timeout, etc.).
    // Typical: ~833 ps @ 1200 MHz, ~714 ps @ 1400 MHz.
    words[18] = aiclk_ps;

    // word[61]: tx_rate_cycles — minimum inter-burst gap in AI-clock cycles.
    // 0 = full line rate (no artificial throttling).
    words[61] = tx_rate_cycles;

    return words;
}

}  // namespace llrt
