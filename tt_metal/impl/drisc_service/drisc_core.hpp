// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host access to one Blackhole DRISC through a UMD cluster: its L1, its NIU modes and its reset. Knows nothing about
// what runs on the core, and needs no metal device.

#pragma once

#include <array>
#include <cstdint>

#include <tt-metalium/vocabulary/device_types.hpp>
#include <tt-umd/types/core_coordinates.hpp>

#include "internal/tt-1xx/blackhole/noc/noc_parameters.h"

namespace tt::umd {
class Cluster;
}  // namespace tt::umd

namespace ll_api {
class ElfFile;
}

namespace tt::tt_metal::internal::drisc_service {

// One DRISC. Holds no lock and no state of the core: every method acts on the hardware directly. Copyable.
class DriscCore {
public:
    // `translated` names the DRISC in TRANSLATED coords. Throws when `chip` is not an MMIO-attached Blackhole chip.
    DriscCore(tt::umd::Cluster& cluster, ChipId chip, const tt::umd::CoreCoord& translated);

    // Puts the core in reset. Safe on a core in any state, running or hung. Leaves the NIU modes as they are: the
    // service kernel sets them at its boot (abi::ServiceHeader::noc2axi_niu_mask).
    void force_stop() const;

    // Loads a firmware-style ELF (linked with firmware_drisc.ld) as tt-metal loads DRISC firmware: text to its address,
    // the local-memory data image to MEM_DRISC_INIT_LOCAL_L1_BASE_SCRATCH, and the reset PC to MEM_DRISC_FIRMWARE_BASE.
    // Call with the core in reset (force_stop). Throws, writing nothing, when a segment lies outside its window.
    void load(const ll_api::ElfFile& image) const;

    // Flushes L1 writes and releases the core from reset.
    void start() const;

    // DRISC L1 at `addr`, over the NoC, through UMD's register window (uncached, strict ordering). A caller whose
    // write tells the running kernel to act on earlier writes puts tt_driver_atomics::sfence() before it, as
    // write_launch_msg_to_core does between launch and go.
    void write_l1(uint32_t addr, const void* data, uint32_t size) const;
    void read_l1(uint32_t addr, void* data, uint32_t size) const;
    uint32_t read_l1_u32(uint32_t addr) const;

    // NIU_CFG_0 of NIU0 and NIU1. Bit 15 set means NOC2AXI, clear means stream mode.
    std::array<uint32_t, NUM_NOCS> read_niu_cfg() const;

private:
    tt::umd::Cluster* cluster_;
    ChipId chip_;
    tt::umd::CoreCoord translated_;
};

}  // namespace tt::tt_metal::internal::drisc_service
