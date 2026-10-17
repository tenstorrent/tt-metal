// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// H2E: RX host -> ERISC -> T6, H2DLeg's shape indexed by arena. Where H2D is a device PULL, this
// is a host PUSH over MMIO -- the destination is the router's receiver channel in L1, not a socket.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "tt_metal/distributed/erisc_bridge_tasks.hpp"

namespace tt::tt_metal::distributed {
class MeshDevice;
}

namespace tt::tt_metal::experimental {

class H2ELeg {
public:
    struct Config {
        uint32_t page_bytes = 0;
        uint32_t link_idx = 0;
        // The resolved eth channel, as on E2HLeg. link_idx indexes the ACTIVE channels, a
        // different list from the one a caller resolved the bridged link from.
        static constexpr uint32_t kUnsetChan = 0xFFFFFFFFu;
        uint32_t eth_chan = kUnsetChan;
        // The chip that owns that channel, as a mesh coordinate (see E2HLeg::Config).
        uint32_t mesh_row = 0;
        uint32_t mesh_col = 0;
        // Slack against the router's decrement being a proxy for "slot reusable" rather than a
        // guarantee: its wr_sent_counter also governs reuse.
        uint32_t credit_margin = 2;
        // Frames this receiver channel took since fabric init: the router's slot cursor survives the leg.
        uint64_t frames_before = 0;
        // A task from the RX arena already holds [header | payload] as the channel expects, so
        // publish() writes those bytes unchanged. False makes it synthesise a header.
        bool frames_carry_header = true;
        // Only read when synthesising; a forwarded bridge frame names its own destination.
        uint32_t dst_noc_x = 0;
        uint32_t dst_noc_y = 0;
        uint32_t dst_l1_addr = 0;
        bool collect_timing = false;
        // First frame to time: warmup is a cold socket, cold TLB and first-touch faults. Excluded
        // here, not by slicing later -- a dropped stamp means sample index does not track frame.
        uint64_t timing_from_frame = 0;
        uint8_t* alias_region_base = nullptr;
    };

    static std::unique_ptr<H2ELeg> create(
        const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh, const Config& cfg, std::string& err);

    ~H2ELeg();

    H2ELeg(const H2ELeg&) = delete;
    H2ELeg& operator=(const H2ELeg&) = delete;

    // ONE task, false when credit is short -- the caller re-offers rather than this blocking
    // inside a poll loop. ZERO REBUILD with frames_carry_header: the bytes go to L1 as they are.
    bool publish(const BridgeDeliverTask& task);

    // Frames the router took, and the only path that refreshes occupancy. The stream accumulator
    // is a RUNNING TOTAL: difference it against create()'s baseline, never read it raw (§7.2.9).
    uint64_t drained(uint32_t arena);

    // The health counters, exposed because a benchmark that cannot read them prints literals --
    // credit_stalls=0 was once a hardcoded 0 while the leg had been counting correctly.
    std::uint64_t credit_stalls() const;  // times publish() was refused for want of a slot
    std::uint64_t oversize() const;       // frames refused for exceeding the receiver slot
    // Frames whose sending router named a different far-ring slot than the one H2E injected
    // into: the two ends have drifted. Counted, not refused.
    std::uint64_t slot_mismatch() const;

    // inject -> consumed, one sample per frame the router took. One clock stamps both ends, so
    // unlike H2H's this is a true one-way span. Empty unless Config::collect_timing was set.
    const std::vector<double>& latency_us() const;

    // Set when a frame is refused for good (empty or oversize), so it is not mistaken for backpressure.
    std::string first_error() const;

private:
    H2ELeg();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tt::tt_metal::experimental
