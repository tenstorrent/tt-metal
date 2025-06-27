// SPDX-FileCopyrightText: 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstring>  // for std::memcpy
#include <variant>  // Added include
#include <limits>
#include <algorithm>

struct alignas(uint64_t) KernelProfilerNocEventMetadata {
    enum class NocType : unsigned char { UNDEF = 0, NOC_0 = 1, NOC_1 = 2 };
    using NocVirtualChannel = int8_t;
    static constexpr uint32_t PAYLOAD_CHUNK_SIZE = 32;

    // New struct for local NOC events
    struct LocalNocEvent {
        int8_t dst_x;
        int8_t dst_y;
        int8_t mcast_end_dst_x;
        int8_t mcast_end_dst_y;
        NocType noc_type : 4;
        NocVirtualChannel noc_vc : 4;
        uint8_t payload_chunks;

        void setNumBytes(uint32_t num_bytes) {
            uint32_t bytes_rounded_up = (num_bytes + PAYLOAD_CHUNK_SIZE - 1) / PAYLOAD_CHUNK_SIZE;
            payload_chunks = std::min(uint32_t(std::numeric_limits<uint8_t>::max()), bytes_rounded_up);
        }
        uint32_t getNumBytes() const { return payload_chunks * PAYLOAD_CHUNK_SIZE; }
    };

    // represents a fabric NOC event
    enum class FabricPacketType : unsigned char { REGULAR, LOW_LATENCY, LOW_LATENCY_MESH };
    struct FabricNoCEvent {
        int8_t dst_x;
        int8_t dst_y;
        int8_t mcast_end_dst_x;
        int8_t mcast_end_dst_y;
        FabricPacketType routing_fields_type;
    };

    // represents a fabric routing fields event; follows a FabricNoCEvent
    struct FabricRoutingFields {
        uint32_t routing_fields_value;
    } __attribute__((packed));

    // Union to hold either local or fabric event data
    union EventData {
        LocalNocEvent local_event;
        FabricNoCEvent fabric_event;
        FabricRoutingFields fabric_routing_fields;
    } data;

    // --- Type enum (tag) --- Must be defined before use in constructor
    enum class NocEventType : unsigned char {
        UNDEF = 0,
        READ = 1,
        READ_SET_STATE = 2,
        READ_SET_TRID = 3,
        READ_WITH_STATE = 4,
        READ_WITH_STATE_AND_TRID = 5,
        READ_BARRIER_START = 6,
        READ_BARRIER_END = 7,
        READ_BARRIER_WITH_TRID = 8,
        READ_DRAM_SHARDED_SET_STATE = 9,
        READ_DRAM_SHARDED_WITH_STATE = 10,

        WRITE_ = 11,
        WRITE_WITH_TRID = 12,
        WRITE_INLINE = 13,
        WRITE_MULTICAST = 14,
        WRITE_SET_STATE = 15,
        WRITE_WITH_STATE = 16,
        WRITE_WITH_TRID_SET_STATE = 17,
        WRITE_WITH_TRID_WITH_STATE = 18,
        WRITE_BARRIER_START = 19,
        WRITE_BARRIER_END = 20,
        WRITE_BARRIER_WITH_TRID = 21,
        WRITE_FLUSH = 22,

        FULL_BARRIER = 23,

        ATOMIC_BARRIER = 24,
        SEMAPHORE_INC = 25,
        SEMAPHORE_WAIT = 26,
        SEMAPHORE_SET = 27,

        // NOTE: fabric events should be contiguous to allow quick range check!
        FABRIC_UNICAST_WRITE = 28,
        FABRIC_UNICAST_INLINE_WRITE = 29,
        FABRIC_UNICAST_ATOMIC_INC = 30,
        FABRIC_FUSED_UNICAST_ATOMIC_INC = 31,
        FABRIC_MULTICAST_WRITE = 32,
        FABRIC_MULTICAST_ATOMIC_INC = 33,
        FABRIC_UNICAST_SCATTER_WRITE = 34,
        FABRIC_ROUTING_FIELDS = 35,

        UNSUPPORTED = 36
    };
    NocEventType noc_xfer_type;

    KernelProfilerNocEventMetadata() : data{.local_event = {}}, noc_xfer_type(NocEventType::UNDEF) {}

    // for deserialization on host side
    explicit KernelProfilerNocEventMetadata(const uint64_t raw_data) {
        std::memcpy(this, &raw_data, sizeof(KernelProfilerNocEventMetadata));
    }

    static bool isFabricEventType(NocEventType event_type) {
        return event_type >= NocEventType::FABRIC_UNICAST_WRITE &&
               event_type <= NocEventType::FABRIC_MULTICAST_ATOMIC_INC;
    }
    bool isFabricRoutingFields() const { return noc_xfer_type == NocEventType::FABRIC_ROUTING_FIELDS; }

    static bool isFabricUnicastEventType(NocEventType event_type) {
        return event_type >= NocEventType::FABRIC_UNICAST_WRITE &&
               event_type <= NocEventType::FABRIC_FUSED_UNICAST_ATOMIC_INC;
    }

    // Getter to return the correct variant based on the tag
    std::variant<LocalNocEvent, FabricNoCEvent, FabricRoutingFields> getContents() const {
        if (isFabricEventType(noc_xfer_type)) {
            return data.fabric_event;
        } else if (isFabricRoutingFields()) {
            return data.fabric_routing_fields;
        } else {
            return data.local_event;
        }
    }

    uint64_t asU64() const {
        uint64_t ret;
        std::memcpy(&ret, this, sizeof(uint64_t));
        return ret;
    }
};
static_assert(sizeof(KernelProfilerNocEventMetadata) == sizeof(uint64_t));
