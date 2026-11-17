// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <memory>
#include <set>
#include <thread>
#include <utility>
#include <vector>

#include <internal/l1_ecc_halt.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-umd/types/cluster_descriptor_types.hpp>

namespace tt::umd {
class DeviceEventReader;
}  // namespace tt::umd

// Blackhole silicon: BRISC halts on an L1 DBE and writes l1_ecc_halt_slot_t::record. Waits detect halts through
// L1EccHaltWatch.
namespace tt::tt_metal::internal {

// A Tensix BRISC still inside an interrupt handler from an earlier process, and the IRQ it is in (BRISC_INT_NO).
struct LatchedBrisc {
    CoreCoord core;
    uint32_t int_no;
};

// Returns every such BRISC on the chip and logs the ones with a halt record. Call before the halt
// records are cleared at init. Runs even when the feature is off.
std::vector<LatchedBrisc> find_latched_briscs(MetalContext& ctx, ChipId device_id);

// Points each latched BRISC's interrupt at code in its L1 ECC halt slot that returns from the handler, so it leaves the
// handler once released from reset. Call while the chip's cores are held in reset, so none can run before both are in
// place.
void release_latched_briscs(MetalContext& ctx, ChipId device_id, const std::vector<LatchedBrisc>& latched);

// Throws if a released BRISC is still inside an interrupt handler. Call when firmware init fails.
void check_briscs_released(MetalContext& ctx, ChipId device_id, const std::vector<LatchedBrisc>& latched);

// Zeroes the halt record on every Tensix core of the chip. Call before the chip's cores are reset at init, since
// the waits during that reset read the records.
void clear_l1_ecc_halt_records(MetalContext& ctx, ChipId device_id);

// Logs the halt record of every Tensix core that halted.
void report_l1_ecc_at_close(MetalContext& ctx, ChipId device_id);

// Reads the chips' device events on a background thread. When ARC forwards a Tensix halt record, or events may have
// been lost, it makes the next L1EccHaltWatch::poll() read the halt records at once instead of after the wait's 10 s
// window. Does nothing where L1EccHaltWatch is off, and skips chips without a PCIe device or whose KMD or ARC firmware
// does not publish events; those keep only the 10 s window.
class L1EccHaltEventListener {
public:
    L1EccHaltEventListener(MetalContext& ctx, const std::set<ChipId>& device_ids);
    ~L1EccHaltEventListener();

    L1EccHaltEventListener(const L1EccHaltEventListener&) = delete;
    L1EccHaltEventListener& operator=(const L1EccHaltEventListener&) = delete;
    L1EccHaltEventListener(L1EccHaltEventListener&&) = delete;
    L1EccHaltEventListener& operator=(L1EccHaltEventListener&&) = delete;

private:
    void run();

    MetalContext& ctx_;
    std::vector<std::pair<ChipId, std::unique_ptr<umd::DeviceEventReader>>> readers_;
    std::atomic<bool> stop_{false};
    std::thread thread_;
};

}  // namespace tt::tt_metal::internal
