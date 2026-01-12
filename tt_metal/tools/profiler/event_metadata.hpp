// SPDX-FileCopyrightText: 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstring>  // for std::memcpy
#include <variant>  // Added include
#include <limits>
#include <algorithm>

struct alignas(uint64_t) KernelProfilerNocEventMetadata {
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
        WRITE_SET_TRID = 12,
        WRITE_WITH_TRID = 13,
        WRITE_INLINE = 14,
        WRITE_MULTICAST = 15,
        WRITE_SET_STATE = 16,
        WRITE_WITH_STATE = 17,
        WRITE_WITH_TRID_SET_STATE = 18,
        WRITE_WITH_TRID_WITH_STATE = 19,
        WRITE_BARRIER_START = 20,
        WRITE_BARRIER_END = 21,
        WRITE_BARRIER_WITH_TRID = 22,
        WRITE_FLUSH = 23,
        WRITE_FLUSH_WITH_TRID = 24,

        FULL_BARRIER = 25,

        ATOMIC_BARRIER = 26,
        SEMAPHORE_INC = 27,
        SEMAPHORE_WAIT = 28,
        SEMAPHORE_SET = 29,

        // NOTE: fabric events should be contiguous to allow quick range check!
        FABRIC_UNICAST_WRITE = 30,
        FABRIC_UNICAST_INLINE_WRITE = 31,
        FABRIC_UNICAST_ATOMIC_INC = 32,
        FABRIC_FUSED_UNICAST_ATOMIC_INC = 33,
        FABRIC_MULTICAST_WRITE = 34,
        FABRIC_MULTICAST_ATOMIC_INC = 35,
        FABRIC_UNICAST_SCATTER_WRITE = 36,
        FABRIC_ROUTING_FIELDS_1D = 37,
        FABRIC_ROUTING_FIELDS_2D = 38,

        UNSUPPORTED = 39,
    };

    enum class NocType : unsigned char { UNDEF = 0, NOC_0 = 1, NOC_1 = 2 };
    using NocVirtualChannel = int8_t;
    static constexpr uint32_t PAYLOAD_CHUNK_SIZE = 32;

    // New struct for local NOC events
    struct LocalNocEvent {
        NocEventType noc_xfer_type;
        int8_t dst_x;
        int8_t dst_y;
        int8_t mcast_end_dst_x;
        int8_t mcast_end_dst_y;
        NocType noc_type : 4;
        NocVirtualChannel noc_vc : 4;
        uint8_t payload_chunks;
        uint8_t posted : 1;
        uint8_t reserved : 7;

        void setAttributes(uint32_t num_bytes, bool p) {
            uint32_t bytes_rounded_up = (num_bytes + PAYLOAD_CHUNK_SIZE - 1) / PAYLOAD_CHUNK_SIZE;
            payload_chunks = std::min(uint32_t(std::numeric_limits<uint8_t>::max()), bytes_rounded_up);
            posted = p;
        }
        uint32_t getNumBytes() const { return payload_chunks * PAYLOAD_CHUNK_SIZE; }
    };

    // Expected to come after a LocalNocEvent when NoC Debug Mode is enabled.
    struct LocalNocEventDstTrailer {
        uint64_t dst_addr_4b : 22;     // Destination address / 4 (4-byte aligned base)
        uint64_t dst_addr_offset : 4;  // Byte offset within 4-byte chunk (0-15)
        uint64_t src_addr_4b : 22;     // Source address / 4 (4-byte aligned base)
        uint64_t src_addr_offset : 4;  // Byte offset within 4-byte chunk (0-15)
        uint64_t counter_value : 12;   // Counter value

        void setDstAddr(uint32_t addr) {
            dst_addr_4b = addr >> 2;
            dst_addr_offset = addr & 0x3;
        }
        uint32_t getDstAddr() const { return (dst_addr_4b << 2) | (dst_addr_offset & 0x3); }

        void setSrcAddr(uint32_t addr) {
            src_addr_4b = addr >> 2;
            src_addr_offset = addr & 0x3;
        }
        uint32_t getSrcAddr() const { return (src_addr_4b << 2) | (src_addr_offset & 0x3); }
    };

    // represents a fabric NOC event
    enum class FabricPacketType : unsigned char { REGULAR, LOW_LATENCY, LOW_LATENCY_MESH, DYNAMIC_MESH };
    struct FabricNoCEvent {
        NocEventType noc_xfer_type;
        int8_t dst_x;
        int8_t dst_y;
        int8_t mcast_end_dst_x;
        int8_t mcast_end_dst_y;
        NocType dst_noc_type : 4;
        FabricPacketType routing_fields_type : 4;
    };

    struct FabricNoCScatterEvent {
        NocEventType noc_xfer_type;
        int8_t dst_x;
        int8_t dst_y;
        int16_t chunk_size;
        int8_t num_chunks;
        NocType dst_noc_type : 4;
        FabricPacketType routing_fields_type : 4;
    };

    // represents a fabric routing fields event; follows a FabricNoCEvent
    struct FabricRoutingFields1D {
        NocEventType noc_xfer_type;
        uint32_t routing_fields_value;
    } __attribute__((packed));

    struct FabricRoutingFields2D {
        NocEventType noc_xfer_type;
        uint8_t ns_hops;
        uint8_t e_hops;
        uint8_t w_hops;
        bool is_mcast;
    } __attribute__((packed));

    struct RawEvent {
        NocEventType noc_xfer_type;
        uint64_t remaining_data : 56;
    } __attribute__((packed));

    // Union to hold either local or fabric event data
    union EventData {
        RawEvent raw_event;
        LocalNocEvent local_event;
        LocalNocEventDstTrailer local_event_dst_trailer;
        FabricNoCEvent fabric_event;
        FabricNoCScatterEvent fabric_scatter_event;
        FabricRoutingFields1D fabric_routing_fields_1d;
        FabricRoutingFields2D fabric_routing_fields_2d;
    } data{};

    KernelProfilerNocEventMetadata() : data{.raw_event = {NocEventType::UNDEF}} {}

    // for deserialization on host side
    explicit KernelProfilerNocEventMetadata(const uint64_t raw_data) {
        std::memcpy(this, &raw_data, sizeof(KernelProfilerNocEventMetadata));
    }

    static bool isValidEventType(NocEventType event_type) {
        return event_type >= NocEventType::READ && event_type < NocEventType::UNSUPPORTED;
    }

    static bool isFabricEventType(NocEventType event_type) {
        return event_type >= NocEventType::FABRIC_UNICAST_WRITE &&
               event_type <= NocEventType::FABRIC_UNICAST_SCATTER_WRITE;
    }

    static bool isFabricRoutingFields(NocEventType event_type) {
        return event_type == NocEventType::FABRIC_ROUTING_FIELDS_1D ||
               event_type == NocEventType::FABRIC_ROUTING_FIELDS_2D;
    }

    static bool isFabricRoutingFields1D(NocEventType event_type) {
        return event_type == NocEventType::FABRIC_ROUTING_FIELDS_1D;
    }

    static bool isFabricRoutingFields2D(NocEventType event_type) {
        return event_type == NocEventType::FABRIC_ROUTING_FIELDS_2D;
    }

    static bool isFabricUnicastEventType(NocEventType event_type) {
        return event_type >= NocEventType::FABRIC_UNICAST_WRITE &&
               event_type <= NocEventType::FABRIC_FUSED_UNICAST_ATOMIC_INC;
    }

    static bool isFabricScatterEventType(NocEventType event_type) {
        return event_type == NocEventType::FABRIC_UNICAST_SCATTER_WRITE;
    }

    // Getter to return the correct variant based on the tag (noc_xfer_type)
    std::variant<LocalNocEvent, FabricNoCEvent, FabricNoCScatterEvent, FabricRoutingFields1D, FabricRoutingFields2D>
    getContents() const {
        if (isFabricEventType(data.raw_event.noc_xfer_type)) {
            if (isFabricScatterEventType(data.raw_event.noc_xfer_type)) {
                return data.fabric_scatter_event;
            }
            return data.fabric_event;
        }
        if (isFabricRoutingFields1D(data.raw_event.noc_xfer_type)) {
            return data.fabric_routing_fields_1d;
        }
        if (isFabricRoutingFields2D(data.raw_event.noc_xfer_type)) {
            return data.fabric_routing_fields_2d;
        }
        return data.local_event;
    }

    // Getter to return a LocalNocEventDstTrailer from the metadata. Called knows from TS_DATA_16B context that this is
    // a dst trailer.
    LocalNocEventDstTrailer getLocalNocEventDstTrailer() const { return data.local_event_dst_trailer; }

    uint64_t asU64() const {
        uint64_t ret;
        std::memcpy(&ret, this, sizeof(uint64_t));
        return ret;
    }
};
static_assert(sizeof(KernelProfilerNocEventMetadata) == sizeof(uint64_t));
