// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Boot-parameter block written to ethernet-core L1 before releasing
// erisc_cmac_simple (budabackend) from reset.
//
// The erisc_cmac_simple firmware reads a 256-byte (64-word) block from
// L1 address 0x1000 immediately after reset.  It uses the fields to
// configure AICLK-based timing, RS-FEC, and TX rate before bringing
// the CMAC/PCS link up toward an external FPGA.
//
// Field assignments are based on the budabackend erisc_cmac_simple
// interface (not available in this tree).  Indices that have not been
// confirmed by firmware source are zero-initialised and commented as
// "reserved / unused".

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
    // known fields.
    //
    // Parameters:
    //   aiclk_ps        – AI-clock period in picoseconds
    //                     (e.g. ~833 for 1200 MHz, ~714 for 1400 MHz).
    //                     Used by the firmware to derive timing windows.
    //   rs_fec_enabled  – Enable Reed-Solomon FEC on the CMAC MAC/PCS.
    //                     true = RS-FEC on (default), false = off.
    //   tx_rate_cycles  – Minimum inter-packet gap in AI-clock cycles.
    //                     0 = full line rate (default).
    static std::array<uint32_t, kSizeWords> build(uint32_t aiclk_ps, bool rs_fec_enabled, uint32_t tx_rate_cycles);
};

}  // namespace llrt
