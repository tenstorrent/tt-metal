// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Device-only header: FabricDatapathUsageL1Ptr wraps an L1 pointer to
// FabricDatapathUsageL1Results and provides the recording API.
// The core data structure is defined in fabric_trimming_types.hpp (host+device).

#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_trimming_types.hpp"
#include "tt_metal/fabric/fabric_edm_packet_header.hpp"  // for NocSendType

#include "internal/risc_attribs.h"

#include <limits>

namespace tt::tt_fabric {

// ============================================================================
// FabricDatapathUsageL1Ptr — compile-time L1 pointer wrapper
// ============================================================================

// Primary template — enabled implementation
template <bool ENABLED, size_t L1_ADDR, size_t NUM_VC = 2, size_t MAX_NUM_SENDER_CHANNELS = 9>
struct FabricDatapathUsageL1Ptr {
    using ResultsType = FabricDatapathUsageL1Results<true, NUM_VC, MAX_NUM_SENDER_CHANNELS>;
    FORCE_INLINE ResultsType* get() const { return reinterpret_cast<ResultsType*>(L1_ADDR); }

    FORCE_INLINE void reset() const {
        auto* r = get();
        r->sender_channel_min_packet_size_seen_bytes_by_vc.fill(std::numeric_limits<uint16_t>::max());
        r->sender_channel_max_packet_size_seen_bytes_by_vc.fill(0);
        r->sender_channel_used_bitfield_by_vc = 0;
        r->sender_channel_forwarded_to_bitfield_by_vc.fill(0);
        r->receiver_channel_data_forwarded_bitfield_by_vc = 0;
        r->used_noc_send_type_by_vc_bitfield.fill(0);
    }

    FORCE_INLINE void set_sender_channel_used(size_t sender_channel_id) const {
        get()->sender_channel_used_bitfield_by_vc |= (1 << sender_channel_id);
    }

    FORCE_INLINE void set_receiver_channel_data_forwarded(size_t receiver_channel_id) const {
        get()->receiver_channel_data_forwarded_bitfield_by_vc |= (1 << receiver_channel_id);
    }

    FORCE_INLINE void update_sender_channel_packet_size(size_t sender_channel_id, uint16_t packet_size_bytes) const {
        auto* r = get();
        if (packet_size_bytes < r->sender_channel_min_packet_size_seen_bytes_by_vc[sender_channel_id]) {
            r->sender_channel_min_packet_size_seen_bytes_by_vc[sender_channel_id] = packet_size_bytes;
        }
        if (packet_size_bytes > r->sender_channel_max_packet_size_seen_bytes_by_vc[sender_channel_id]) {
            r->sender_channel_max_packet_size_seen_bytes_by_vc[sender_channel_id] = packet_size_bytes;
        }
    }

    FORCE_INLINE void set_sender_channel_forwarded_to(size_t vc_id, size_t sender_channel_id) const {
        get()->sender_channel_forwarded_to_bitfield_by_vc[vc_id] |= (1 << sender_channel_id);
    }

    FORCE_INLINE void merge_sender_channel_forwarded_to(size_t vc_id, uint16_t mask) const {
        get()->sender_channel_forwarded_to_bitfield_by_vc[vc_id] |= mask;
    }

    FORCE_INLINE void set_noc_send_type_used(size_t vc_id, NocSendType noc_send_type) const {
        get()->used_noc_send_type_by_vc_bitfield[vc_id] |= (1 << static_cast<uint8_t>(noc_send_type));
    }
};

// Disabled specialization — all no-ops
template <size_t L1_ADDR, size_t NUM_VC, size_t MAX_NUM_SENDER_CHANNELS>
struct FabricDatapathUsageL1Ptr<false, L1_ADDR, NUM_VC, MAX_NUM_SENDER_CHANNELS> {
    FORCE_INLINE void reset() const {}
    FORCE_INLINE void set_sender_channel_used(size_t) const {}
    FORCE_INLINE void set_receiver_channel_data_forwarded(size_t) const {}
    FORCE_INLINE void update_sender_channel_packet_size(size_t, uint16_t) const {}
    FORCE_INLINE void set_sender_channel_forwarded_to(size_t, size_t) const {}
    FORCE_INLINE void merge_sender_channel_forwarded_to(size_t, uint16_t) const {}
    FORCE_INLINE void set_noc_send_type_used(size_t, NocSendType) const {}
};

}  // namespace tt::tt_fabric
