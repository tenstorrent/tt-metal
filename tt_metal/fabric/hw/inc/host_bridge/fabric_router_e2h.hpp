// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// The E2H side of the router, kept out of fabric_erisc_router.cpp so its hooks stay call sites.
// The unit is a LINK: a bridged router has no wire, so every channel it services pushes to host.
#pragma once

#include "hostdevcommon/erisc_bridge_layout.h"
#include "tt_metal/fabric/hw/inc/host_bridge/erisc_host_bridge_sender.hpp"

// From fabric_erisc_router_ct_args.hpp: named CT args are read there and exposed as constants.
// Asks whether THIS ROUTER is bridged, not whether this channel is.
constexpr bool enable_e2h = enable_e2h_bridge;

// The speedy VC0 path has its own send/receive code with none of these hooks, so a bridged router
// running it would put frames on the wire. The builder disables it; this catches one that didn't.
static_assert(!enable_e2h || !enable_speedy_vc0, "the E2H host bridge requires ENABLE_SPEEDY_VC0=0");

// Every channel this risc services. is_sender_channel_serviced is per-risc, so in 2-erisc mode
// the two harts take disjoint channels and never share a socket cursor.
constexpr bool sender_channel_uses_e2h(size_t sender_channel) {
    return enable_e2h && is_sender_channel_serviced[sender_channel];
}

// Retire locally ONLY where nothing downstream can: with a far receiver its real completion
// returns the credit, and retiring here as well double-counts and pins num_free_slots.
constexpr bool sender_channel_self_retires_e2h(size_t sender_channel) {
    return sender_channel_uses_e2h(sender_channel) && e2h_bridge_no_far_receiver;
}

// Block layout from hostdevcommon: the host derives these same offsets from the same constants,
// so the address the router writes and the address the host reads cannot drift apart.
constexpr uint32_t e2h_block_addr = e2h_bridge_block_addr;
constexpr uint32_t e2h_packet_capacity = e2h_bridge_capacity;
static_assert(
    !enable_e2h || e2h_block_addr % kBridgeBlockAlign == 0, "D2HSocket requires a 64 B aligned config buffer");

constexpr uint32_t e2h_socket_config_addr(size_t ch) {
    return e2h_block_addr + bridge_socket_config_offset(static_cast<uint32_t>(ch));
}
constexpr uint32_t e2h_desc_scratch_addr(size_t ch) {
    return e2h_block_addr + bridge_desc_scratch_offset(static_cast<uint32_t>(ch));
}

// Sized by MAX_NUM_SENDER_CHANNELS unconditionally, as every other per-channel array here is.
// Nothing in fabric sizes an array by a feature flag; the gating is if constexpr, not allocation.
static EriscHostBridgeSender e2h_senders[MAX_NUM_SENDER_CHANNELS];
static bool e2h_opened[MAX_NUM_SENDER_CHANNELS];

// The router is otherwise opaque: a kernel that never ran, a socket that never opened and a gate
// that never let go all look alike. One block per router, counters summed across its channels.
constexpr uint32_t e2h_status_addr = e2h_block_addr + kStatusOffset;
FORCE_INLINE volatile tt_l1_ptr BridgeStatus* e2h_status() {
    return reinterpret_cast<volatile tt_l1_ptr BridgeStatus*>(e2h_status_addr);
}

// Lazy, because the host builds the socket after this router runs. Re-openable, because ERISC L1
// persists across processes and a stale config would otherwise be latched as valid.
FORCE_INLINE bool e2h_ensure_open(size_t ch) {
    // Never read a config the host is mid-way through writing: a half-written address faults the PCIe write.
    invalidate_l1_cache();
    if (e2h_status()->host_armed != kBridgeHostArmed) {
        e2h_opened[ch] = false;
        return false;
    }
    const uint32_t cfg_addr = e2h_socket_config_addr(ch);
    auto probe = create_sender_socket_interface(cfg_addr);
    if (!probe.is_d2h || probe.downstream_fifo_total_size < bridge_socket_page_bytes(e2h_packet_capacity)) {
        return false;
    }
    // Re-open when the host reset the config, not only when the ring moved: the address is
    // usually identical across runs, and bytes_sent going backwards is the only signal.
    const bool same_ring = e2h_senders[ch].socket.downstream_fifo_addr == probe.downstream_fifo_addr &&
                           e2h_senders[ch].socket.downstream_bytes_sent_addr == probe.downstream_bytes_sent_addr;
    const bool not_rewound = probe.bytes_sent >= e2h_senders[ch].socket.bytes_sent;
    if (e2h_opened[ch] && same_ring && not_rewound) {
        return true;
    }
    e2h_status()->open_tries++;
    e2h_senders[ch] = erisc_host_bridge_open(cfg_addr, e2h_packet_capacity, e2h_desc_scratch_addr(ch));
    e2h_opened[ch] = true;
    e2h_status()->opened++;
    return true;
}

// Second credit: far-ring room is not enough, this channel's D2H socket needs a slot too.
// Non-blocking -- this runs on the router's own core, so a spin is a wedge.
FORCE_INLINE bool e2h_can_send(
    size_t ch, bool can_send, bool receiver_has_space, bool has_unsent_packet, uint32_t free_slots) {
    // Recorded BEFORE the bridge's own gate, so the three counters are mutually exclusive and
    // a stalled run names its own cause instead of leaving three live hypotheses.
    e2h_status()->free_slots = free_slots;
    if (!has_unsent_packet) {
        e2h_status()->blocked_nodata++;
    } else if (!receiver_has_space) {
        e2h_status()->blocked_rx++;
    }
    if (!can_send) {
        // Probing the socket costs an L1 read per call and nothing downstream can use the answer
        // while the router itself is not ready to send.
        return false;
    }
    const bool ready = e2h_ensure_open(ch) && erisc_host_bridge_slot_free(e2h_senders[ch], 1);
    if (!ready) {
        e2h_status()->declined++;
    }
    return ready;
}

// The far ring's write cursor, serialised: the bridge cannot write far L1, so the address becomes
// an index the RX host resolves. Unconditional -- can_send already proved a slot. Guard last.
template <typename OutboundPtrs>
FORCE_INLINE void e2h_write_frame(size_t ch, const OutboundPtrs& p, uint32_t src_addr, uint32_t payload_size_bytes) {
    const uint32_t slot_idx =
        (p.remote_receiver_channel_address_ptr - p.remote_receiver_channel_address_base) / p.slot_size_bytes;
    erisc_host_bridge_write_frame(e2h_senders[ch], src_addr, payload_size_bytes, static_cast<uint16_t>(slot_idx));
    e2h_status()->frames++;
}

// The stamp proves THIS binary is on the core. Inside the guard, because a stock build must write
// nothing: block address 0 would land the write wherever 0 + offset points.
FORCE_INLINE void e2h_init() {
    if constexpr (enable_e2h) {
        e2h_status()->build_stamp = kBridgeBuildStamp;
        for (size_t ch = 0; ch < MAX_NUM_SENDER_CHANNELS; ++ch) {
            e2h_opened[ch] = false;
        }
        // Magic last, so a host reading a half-written block sees nothing rather than zeros.
        e2h_status()->open_tries = 0;
        e2h_status()->opened = 0;
        e2h_status()->frames = 0;
        e2h_status()->declined = 0;
        e2h_status()->blocked_rx = 0;
        e2h_status()->blocked_nodata = 0;
        e2h_status()->free_slots = 0;
        e2h_status()->host_armed = 0;
        e2h_status()->magic = kBridgeStatusMagic;
    }
}
