// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Where the bridge may live on an ERISC, read from the live FabricEriscDatamoverConfig.
// 0x15b40 is not free: the HAL calls it UNRESERVED, but fabric's allocator starts there.
#pragma once

#include <cstdint>
#include <string>

// Here, not under distributed/: the router builder needs the identical answer when it bakes
// the address into the kernel, and it cannot include anything from there.
#include "tt_metal/fabric/erisc_bridge_block.hpp"

namespace tt::tt_fabric {
class ControlPlane;
}  // namespace tt::tt_fabric

namespace tt::tt_fabric::erisc_bridge {

struct BridgePlacement {
    bool ok = false;
    std::uint32_t block_addr = 0;
    std::uint32_t block_bytes = 0;
    std::uint32_t status_addr = 0;

    // Per sender channel: a bridged link has no wire, so every serviced channel pushes to the
    // host. One socket each -- two channels over one config buffer share a single cursor.
    std::uint32_t socket_config_addr(std::uint32_t ch) const { return block_addr + bridge_socket_config_offset(ch); }
    std::uint32_t desc_scratch_addr(std::uint32_t ch) const { return block_addr + bridge_desc_scratch_offset(ch); }
    static constexpr std::uint32_t channels() { return kBridgeChannels; }

    // The evidence behind the verdict, so a caller prints why rather than asserting blindly.
    std::uint32_t l1_base = 0;      // where fabric's allocator starts
    std::uint32_t l1_top = 0;       // max_l1_loading_size
    std::uint32_t buffers_end = 0;  // one past the highest byte any channel buffer occupies
    std::uint32_t free_tail = 0;    // l1_top - buffers_end
    std::string why;                // set when !ok
};

// The live EDM config, via the chain needing fabric's INTERNAL headers. Kept here so those
// includes live in one TU instead of spreading to every caller.
const FabricEriscDatamoverConfig& router_config(const ControlPlane& cp);

// Reads a live config; does not own it and must not outlive it.
class EriscBridgePlacement {
public:
    explicit EriscBridgePlacement(const FabricEriscDatamoverConfig& cfg) : cfg_(cfg) {}

    // Carves the block from the TOP of L1, downward, and refuses if it would touch a buffer.
    BridgePlacement place(std::uint32_t want_bytes = kBridgeBlockBytes) const;

    // The per-channel allocation map, for when place() refuses and the reason matters.
    std::string describe() const;

    // Frames the router can send before it halts PERMANENTLY: without loop C the far-receiver
    // completion never returns. 0 means the allocator was unreadable -- unknown, not unlimited.
    std::uint32_t credit_ceiling_frames() const;

    // Fabric's own per-packet channel slot. A packet exceeding this is rejected upstream of the
    // bridge, which looks like a bridge fault and is not one.
    std::uint32_t channel_slot_bytes() const;

    // The RECEIVER channel H2E lands frames in -- the buffer the remote ERISC would have
    // written over the cable. 0 when the allocator cannot be read.
    std::uint32_t receiver_channel_base(std::uint32_t vc = 0, std::uint32_t chan = 0) const;
    std::uint32_t receiver_channel_slots(std::uint32_t vc = 0, std::uint32_t chan = 0) const;

    // Sweep bounds: builder_config is internal, so callers cannot name MAX_NUM_VCS themselves.
    // Pair with receiver_channel_base() != 0 to skip combinations this build did not configure.
    static std::uint32_t max_vcs();
    static std::uint32_t max_receiver_channels();

private:
    const FabricEriscDatamoverConfig& cfg_;
};

}  // namespace tt::tt_fabric::erisc_bridge
