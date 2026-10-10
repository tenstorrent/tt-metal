// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/erisc_bridge_placement.hpp"

#include <algorithm>
#include <memory>
#include <sstream>

#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include "tt_metal/fabric/builder/fabric_builder_config.hpp"
#include "tt_metal/fabric/builder/fabric_static_sized_channels_allocator.hpp"
#include "tt_metal/fabric/builder/fabric_stream_assignment.hpp"
#include "tt_metal/fabric/erisc_datamover_builder.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/fabric_context.hpp"

namespace tt::tt_fabric::erisc_bridge {

const FabricEriscDatamoverConfig& router_config(const ControlPlane& cp) {
    return cp.get_fabric_context().get_builder_context().get_fabric_router_config();
}

std::uint32_t receiver_pkts_sent_stream(const ControlPlane& cp, std::uint32_t mesh_id, std::uint32_t channel) {
    return cp.get_fabric_context()
        .get_builder_context()
        .get_stream_assignment(MeshId{mesh_id})
        .id(StreamRole::RECEIVER_PKTS_SENT, channel, 0);
}

BridgePlacement EriscBridgePlacement::place(std::uint32_t want_bytes) const {
    BridgePlacement p;
    p.block_bytes = want_bytes;
    p.l1_top = static_cast<std::uint32_t>(cfg_.max_l1_loading_size);
    p.l1_base = static_cast<std::uint32_t>(cfg_.edm_status_address);  // the control block we can name

    const std::uint64_t end = bridge_highest_buffer_end(cfg_);
    if (end == 0) {
        p.why = "channel allocator is not FabricStaticSizedChannelsAllocator -- cannot bound the buffers";
        return p;
    }
    p.buffers_end = static_cast<std::uint32_t>(end);
    if (p.l1_top <= p.buffers_end) {
        p.why = "fabric's buffers reach the top of L1 -- there is no free tail to carve";
        return p;
    }
    p.free_tail = p.l1_top - p.buffers_end;

    // Downward from the top, then aligned down: the block must not cross l1_top either.
    // Same function the router builder used to bake the address into the kernel.
    const std::uint32_t addr = bridge_block_addr(cfg_, want_bytes);
    if (addr == 0) {
        std::ostringstream os;
        os << "free tail is " << p.free_tail << " B, need " << want_bytes << " B plus alignment";
        p.why = os.str();
        return p;
    }

    p.block_addr = addr;
    p.status_addr = addr + kStatusOffset;
    p.ok = true;
    return p;
}

std::uint32_t EriscBridgePlacement::credit_ceiling_frames() const {
    const auto alloc = std::dynamic_pointer_cast<FabricStaticSizedChannelsAllocator>(cfg_.channel_allocator);
    if (alloc == nullptr) {
        return 0;
    }
    // The receiver channel the far router would land in: its slot count is what the local
    // sender's num_free_slots is primed to, and the only credit it ever gets.
    std::uint32_t slots = 0;
    for (std::size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        for (std::size_t c = 0; c < cfg_.num_used_receiver_channels_per_vc[vc]; ++c) {
            slots = std::max(slots, static_cast<std::uint32_t>(alloc->get_receiver_channel_number_of_slots(vc, c)));
        }
    }
    return slots;
}

std::uint32_t EriscBridgePlacement::channel_slot_bytes() const {
    return static_cast<std::uint32_t>(cfg_.channel_buffer_size_bytes);
}

std::uint32_t EriscBridgePlacement::max_vcs() { return static_cast<std::uint32_t>(builder_config::MAX_NUM_VCS); }

std::uint32_t EriscBridgePlacement::max_receiver_channels() {
    return static_cast<std::uint32_t>(builder_config::num_max_receiver_channels);
}

std::uint32_t EriscBridgePlacement::receiver_channel_base(std::uint32_t vc, std::uint32_t chan) const {
    const auto alloc = std::dynamic_pointer_cast<FabricStaticSizedChannelsAllocator>(cfg_.channel_allocator);
    if (alloc == nullptr || vc >= builder_config::MAX_NUM_VCS || chan >= cfg_.num_used_receiver_channels_per_vc[vc]) {
        return 0;
    }
    return static_cast<std::uint32_t>(alloc->get_receiver_channel_base_address(vc, chan));
}

std::uint32_t EriscBridgePlacement::receiver_channel_slots(std::uint32_t vc, std::uint32_t chan) const {
    const auto alloc = std::dynamic_pointer_cast<FabricStaticSizedChannelsAllocator>(cfg_.channel_allocator);
    if (alloc == nullptr || vc >= builder_config::MAX_NUM_VCS || chan >= cfg_.num_used_receiver_channels_per_vc[vc]) {
        return 0;
    }
    return static_cast<std::uint32_t>(alloc->get_receiver_channel_number_of_slots(vc, chan));
}

std::string EriscBridgePlacement::describe() const {
    std::ostringstream os;
    os << "edm_status=0x" << std::hex << cfg_.edm_status_address << " l1_top=0x" << cfg_.max_l1_loading_size << std::dec
       << " slot=" << cfg_.channel_buffer_size_bytes << " B";
    const auto alloc = std::dynamic_pointer_cast<FabricStaticSizedChannelsAllocator>(cfg_.channel_allocator);
    if (alloc == nullptr) {
        os << "\n    (allocator is not FabricStaticSizedChannelsAllocator -- no per-channel map)";
        return os.str();
    }
    for (std::size_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        for (std::size_t c = 0; c < cfg_.num_used_sender_channels_per_vc[vc]; ++c) {
            os << "\n    vc" << vc << " snd" << c << " base=0x" << std::hex
               << alloc->get_sender_channel_base_address(vc, c) << std::dec
               << " slots=" << alloc->get_sender_channel_number_of_slots(vc, c);
        }
        for (std::size_t c = 0; c < cfg_.num_used_receiver_channels_per_vc[vc]; ++c) {
            os << "\n    vc" << vc << " rcv" << c << " base=0x" << std::hex
               << alloc->get_receiver_channel_base_address(vc, c) << std::dec
               << " slots=" << alloc->get_receiver_channel_number_of_slots(vc, c);
        }
    }
    return os.str();
}

}  // namespace tt::tt_fabric::erisc_bridge
