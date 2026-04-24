// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Host-side handle for an external CMAC port after erisc_cmac_simple has
// brought the PCS/FEC link up.  ExternalIfaceSender tracks the L1 outbound
// ring buffer on the erisc core and provides a send() API that enqueues raw
// TT-link frames (EtherType 0x1AF4) for transmission via the CMAC TX path.
//
// Usage:
//   1. Wait for is_link_up() to return true (PCS lock).
//   2. Call send() with a fully-formed TT-link frame.  Returns false if the
//      eight-slot ring is full; caller may retry.
//
// Ring-buffer layout (shared with erisc_cmac_gw kernel):
//   L1[0x2000 + slot*2048]  SlotHeader { uint32 size; uint32 flags; }
//   L1[0x2000 + slot*2048 + 8 .. slot*2048+2047]  frame payload
//
// The host writes the payload then sets size != 0.  The device clears size to
// 0 once the frame has been handed off to CMAC TX.

#include <cstdint>
#include <span>

#include <umd/device/types/cluster_descriptor_types.hpp>  // ChipId
#include <tt-metalium/core_coord.hpp>                     // CoreCoord

namespace tt::llrt {

class ExternalIfaceSender {
public:
    // Construct a sender for the erisc core at |virtual_eth_core| on |chip_id|.
    // The core must already be running erisc_cmac_simple (or erisc_cmac_gw).
    ExternalIfaceSender(ChipId chip_id, CoreCoord virtual_eth_core);

    // Write a raw TT-link frame (caller owns buf) into the erisc core's
    // outbound ring buffer at L1.  Returns false if the ring is full.
    bool send(std::span<const uint8_t> buf);

    // True if erisc_cmac_simple reports PCS lock (link is up).
    bool is_link_up() const;

    ChipId chip_id() const { return chip_id_; }
    CoreCoord virtual_core() const { return virtual_core_; }

private:
    ChipId chip_id_;
    CoreCoord virtual_core_;

    // L1 ring buffer state — head ptr written by host, tail ptr read from L1.
    uint32_t ring_head_{0};

    // Ring buffer constants (must match erisc_cmac_gw kernel).
    static constexpr uint32_t kRingBase = 0x2000;  // after boot-params block
    static constexpr uint32_t kRingSlots = 8;
    static constexpr uint32_t kSlotBytes = 2048;  // enough for one TT-link frame
};

}  // namespace tt::llrt
