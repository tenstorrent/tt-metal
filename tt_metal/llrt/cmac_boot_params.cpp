// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Implementation of CmacBootParams::build().
//
// The erisc_cmac_simple ELF (from budabackend) reads the 64-word block at
// L1 0x1000 to configure CMAC/PCS hardware before bringing the external
// link up.  Field positions below are derived from:
//   - the erisc_cmac_simple interface description (external; not in tree)
//   - the budabackend boot_params layout used by WH CMAC firmware
//   - cross-referencing serdes_results_t.target_speed (word[3] of that
//     struct is "target speed from the boot params") in
//     tt_metal/hw/inc/internal/tt-1xx/blackhole/eth_fw_api.h
//
// Any word not assigned here is zero-initialised (safe / no-op for the
// firmware).

#include "llrt/cmac_boot_params.hpp"

namespace llrt {

std::array<uint32_t, CmacBootParams::kSizeWords> CmacBootParams::build(
    uint32_t aiclk_ps, bool rs_fec_enabled, uint32_t tx_rate_cycles) {
    std::array<uint32_t, kSizeWords> words{};  // zero-init

    // word[0]: boot-params version / magic marker.
    // Value 0x00000001 signals "version 1" to the firmware; keeps the
    // block distinguishable from an uninitialised (all-zero) L1.
    words[0] = 0x00000001u;

    // words[1-2]: reserved (firmware uses these for internal flags in
    // later protocol revisions; zero is safe for the simple CMAC path).

    // word[3]: AI-clock period in picoseconds.
    // The firmware converts this to a cycle count to derive link-timing
    // windows (e.g. PCS lock timeout, RS-FEC window size).  Typical
    // values: 833 ps ≈ 1200 MHz, 714 ps ≈ 1400 MHz.
    words[3] = aiclk_ps;

    // words[4-17]: reserved.

    // word[18]: RS-FEC enable flag (primary register word).
    // 1 = RS-FEC enabled (clause 91/108 FEC on the 100G CMAC);
    // 0 = FEC disabled (or BASE-R FEC, depending on PHY configuration).
    words[18] = rs_fec_enabled ? 1u : 0u;

    // word[19]: RS-FEC secondary configuration word.
    // Reserved for FEC mode/polynomial overrides; zero = use defaults.
    words[19] = 0u;

    // words[20-27]: reserved.

    // word[28]: TX rate control — minimum inter-packet gap in AI-clock
    // cycles.  0 = full line rate (no artificial throttling).
    words[28] = tx_rate_cycles;

    // words[29-42]: reserved.

    // word[43]: flags / secondary configuration word (placeholder).
    // Reserved for future use by the firmware (checksum, feature bits,
    // etc.).  Zero is safe.
    words[43] = 0u;

    // words[44-63]: reserved / unused.

    return words;
}

}  // namespace llrt
