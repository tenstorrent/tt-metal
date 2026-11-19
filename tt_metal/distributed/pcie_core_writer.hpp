// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>

namespace tt::umd {
class Cluster;
class IoWindow;
struct CoreCoord;
}  // namespace tt::umd

namespace tt::tt_metal::distributed {

// A write-only window into core's address space from addr, or null when no TLB is free or the window ends before
// addr + size. It posts its writes on a static write VC. A Strict window would wait for each write's ack, one write
// per round trip (about 0.4 us per 128 B page into a DRAM core; the strict MMIO rate into Tensix L1, #64622). Posted
// writes are ordered only by the VC: a packet on a dynamic VC may switch VCs at each router (and a DRAM core gives
// each VC its own AXI ID), so a later write could overtake an earlier one; on one static VC, writes to the core arrive
// in issue order.
std::unique_ptr<tt::umd::IoWindow> create_posted_window(
    tt::umd::Cluster& cluster, int chip_id, const tt::umd::CoreCoord& core, uint64_t addr, uint64_t size);

/**
 * Lightweight UMD-only device access for cross-process socket connectors.
 *
 * Opens the UMD driver for the local host without going through MetalContext
 * or tt-metal's Cluster (no ethernet firmware init, no RISC reset).
 *
 * A single `tt::umd::Cluster` is shared across all PCIeCoreWriters in this process to
 * avoid the file-descriptor exhaustion of building one cluster per chip.
 */
class PCIeCoreWriter {
public:
    PCIeCoreWriter(uint32_t device_id, uint32_t virtual_core_x, uint32_t virtual_core_y);
    ~PCIeCoreWriter();

    PCIeCoreWriter(const PCIeCoreWriter&) = delete;
    PCIeCoreWriter& operator=(const PCIeCoreWriter&) = delete;
    PCIeCoreWriter(PCIeCoreWriter&&) noexcept;
    PCIeCoreWriter& operator=(PCIeCoreWriter&&) noexcept;

    // Writes through the driver (cluster->write_to_device), which may reconfigure a window per write.
    std::function<void(void*, uint32_t, uint64_t)> get_pcie_writer() const;

    // Writes through a posted, write-only window on one static VC when the chip is a Blackhole, a TLB is free and
    // the window reaches window_end (an address on the core); otherwise the same writer as get_pcie_writer(). Writes
    // to one core on one VC arrive in issue order, which is what an H2D socket's bytes_sent update relies on.
    std::function<void(void*, uint32_t, uint64_t)> get_posted_pcie_writer(uint64_t window_end);

    // True after get_posted_pcie_writer() got its window.
    bool writes_through_window() const { return window_ != nullptr; }

private:
    static std::shared_ptr<tt::umd::Cluster> get_or_create_cluster();

    static std::mutex cluster_mutex_;
    static std::shared_ptr<tt::umd::Cluster> shared_cluster_;

    // Shared so the cluster outlives window_, even past static destruction.
    std::shared_ptr<tt::umd::Cluster> cluster_;
    uint32_t device_id_ = 0;
    uint32_t virtual_core_x_ = 0;
    uint32_t virtual_core_y_ = 0;
    std::unique_ptr<tt::umd::IoWindow> window_;
};

}  // namespace tt::tt_metal::distributed
