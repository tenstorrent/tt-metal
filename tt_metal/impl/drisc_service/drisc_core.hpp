// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host access to one Blackhole DRISC through a UMD cluster: its L1, its NIU modes, its reset, and an L1 write the KMD
// makes when an fd closes. Knows nothing about what runs on the core, and needs no metal device.

#pragma once

#include <array>
#include <cstdint>
#include <memory>

#include <tt-metalium/device_types.hpp>
#include <tt-umd/types/core_coordinates.hpp>
#include <tt-umd/types/io_window_config.hpp>
#include <tt-umd/utils/noc_cleanup_write.hpp>

namespace tt::umd {
class Cluster;
class IoWindow;
}  // namespace tt::umd

namespace ll_api {
class ElfFile;
}

namespace tt::tt_metal::drisc_service {

// One DRISC. Holds no lock and no state of the core: every method acts on the hardware directly. Copyable.
class DriscCore {
public:
    // `translated` names the DRISC in TRANSLATED coords. Throws when `chip` is not an MMIO-attached Blackhole chip.
    DriscCore(tt::umd::Cluster& cluster, ChipId chip, const tt::umd::CoreCoord& translated);

    // Puts the core in reset, then sets each NIU's NIU_CFG_0 bit 15 (NOC2AXI) to bit N of `noc2axi_mask` for NIU N,
    // the state DRISC firmware leaves it in. The service's only writer of NIU modes: its kernel never changes them.
    // Safe on a core in any state, running or hung.
    void force_stop(uint8_t noc2axi_mask) const;

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
    void write_l1_u32(uint32_t addr, uint32_t value) const;

    // NIU_CFG_0 of NIU0 and NIU1. Bit 15 set means NOC2AXI, clear means stream mode.
    std::array<uint32_t, 2> read_niu_cfg() const;

    // A window onto this DRISC's L1 that stays mapped while it lives, for writes at L1 addresses below `size`: WC
    // combines a write into one PCIe burst, UC sends each word on its own. Null when the device has no TLB to spare
    // (it has 202 for every process to share).
    std::unique_ptr<tt::umd::IoWindow> map_l1(tt::umd::HostMemoryCaching caching, uint32_t size) const;

    tt::umd::Cluster& cluster() const { return *cluster_; }
    ChipId chip() const { return chip_; }
    const tt::umd::CoreCoord& translated() const { return translated_; }

private:
    tt::umd::Cluster* cluster_;
    ChipId chip_;
    tt::umd::CoreCoord translated_;
};

// The KMD writes `value` to the DRISC's L1 at `addr` when this object is destroyed, or when the process dies without
// destroying it (tt::umd::NocCleanupWrite, whose fork caveat applies). The process's other fds may close first, so host
// memory they pinned can be gone before the core sees the write. Not copyable or movable.
class ExitWrite {
public:
    // Throws whatever tt::umd::NocCleanupWrite throws.
    ExitWrite(const DriscCore& core, uint32_t addr, uint32_t value);

private:
    tt::umd::NocCleanupWrite cleanup_;
};

}  // namespace tt::tt_metal::drisc_service
