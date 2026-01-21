// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <climits>
#include <initializer_list>
#include <limits>

#if defined(KERNEL_BUILD) || defined(FW_BUILD)
#include "api/debug/assert.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_utils.hpp"
#include "tt_metal/fabric/hw/inc/fabric_routing_mode.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#else
#include <tt_stl/assert.hpp>
#endif

// Include fabric_common.h for RoutingFieldsConstants and routing_encoding namespace
#include "hostdevcommon/fabric_common.h"

// These functions have different behavior on host or device.
// This causes problems trying to detect unused parameters.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

// NOLINTBEGIN(misc-unused-parameters)
namespace tt::tt_fabric {

// Helper for dependent static_assert that always evaluates to false
template <class>
inline constexpr bool always_false_v = false;

enum TerminationSignal : uint32_t {
    KEEP_RUNNING = 0,

    // Wait for messages to drain
    // Non functional. Use IMMEDIATELY_TERMINATE instead.
    GRACEFULLY_TERMINATE = 1,

    // Immediately terminate - don't wait for any outstanding messages to arrive or drain out
    IMMEDIATELY_TERMINATE = 2
};

enum EDMStatus : uint32_t {
    // EDM kernel has started running
    STARTED = 0xA0B0C0D0,

    // Handshake complete with remote
    REMOTE_HANDSHAKE_COMPLETE = 0xA1B1C1D1,

    // Ready to start listening for packets
    LOCAL_HANDSHAKE_COMPLETE = 0xA2B2C2D2,

    // Ready for traffic
    READY_FOR_TRAFFIC = 0xA3B3C3D3,

    // EDM exiting
    TERMINATED = 0xA4B4C4D4,

    // Fabric Initialization Postcodes
    // Initialization started
    INITIALIZATION_STARTED = 0xB0C0D0E0,

    // TXQ initialized
    TXQ_INITIALIZED = 0xB1C1D1E1,

    // Stream registers initialized
    STREAM_REG_INITIALIZED = 0xB2C2D2E2,

    // Started setting up downstream EDMs
    DOWNSTREAM_EDM_SETUP_STARTED = 0xB3C3D3E3,

    // EDM VC0 setup complete
    EDM_VCS_SETUP_COMPLETE = 0xB4C4D4E4,

    // Worker interfaces initialized
    WORKER_INTERFACES_INITIALIZED = 0xB6C6D6E6,

    // Ethernet handshake complete
    ETHERNET_HANDSHAKE_COMPLETE = 0xB7C7D7E7,

    // VCs opened
    VCS_OPENED = 0xB8C8D8E8,

    // Routing table initialized
    ROUTING_TABLE_INITIALIZED = 0xB9C9D9E9,

    // Initialization complete
    INITIALIZATION_COMPLETE = 0xBACADAEA
};

// 3 bits
enum NocSendType : uint8_t {
    NOC_UNICAST_WRITE = 0,
    NOC_UNICAST_INLINE_WRITE = 1,
    NOC_UNICAST_ATOMIC_INC = 2,
    NOC_FUSED_UNICAST_ATOMIC_INC = 3,
    NOC_UNICAST_SCATTER_WRITE = 4,
    NOC_MULTICAST_WRITE = 5,       // mcast has bug
    NOC_MULTICAST_ATOMIC_INC = 6,  // mcast has bug
    NOC_UNICAST_READ = 7,
    NOC_SEND_TYPE_LAST = NOC_UNICAST_SCATTER_WRITE
};
// How to send the payload across the cluster
// 1 bit
enum ChipSendType : uint8_t { CHIP_UNICAST = 0, CHIP_MULTICAST = 1, CHIP_SEND_TYPE_LAST = CHIP_MULTICAST };

struct RoutingFields {
    static constexpr uint8_t START_DISTANCE_FIELD_BIT_WIDTH = 4;
    static constexpr uint8_t RANGE_HOPS_FIELD_BIT_WIDTH = 4;
    static constexpr uint8_t LAST_HOP_DISTANCE_VAL = 1;
    static constexpr uint8_t LAST_CHIP_IN_MCAST_VAL = 1 << tt::tt_fabric::RoutingFields::START_DISTANCE_FIELD_BIT_WIDTH;
    static constexpr uint8_t HOP_DISTANCE_MASK = (1 << tt::tt_fabric::RoutingFields::RANGE_HOPS_FIELD_BIT_WIDTH) - 1;
    static constexpr uint8_t RANGE_MASK = ((1 << tt::tt_fabric::RoutingFields::RANGE_HOPS_FIELD_BIT_WIDTH) - 1)
                                          << tt::tt_fabric::RoutingFields::START_DISTANCE_FIELD_BIT_WIDTH;
    static constexpr uint8_t LAST_MCAST_VAL = LAST_CHIP_IN_MCAST_VAL | LAST_HOP_DISTANCE_VAL;

    uint8_t value;
};
static_assert(sizeof(RoutingFields) == sizeof(uint8_t), "RoutingFields size is not 1 bytes");
static_assert(
    (RoutingFields::START_DISTANCE_FIELD_BIT_WIDTH + RoutingFields::RANGE_HOPS_FIELD_BIT_WIDTH) <=
        sizeof(RoutingFields) * 8,
    "START_DISTANCE_FIELD_BIT_WIDTH + RANGE_HOPS_FIELD_BIT_WIDTH must equal 8");

struct MulticastRoutingCommandHeader {
    uint8_t start_distance_in_hops;
    uint8_t range_hops;  // 0 implies unicast
};
static_assert(sizeof(MulticastRoutingCommandHeader) == 2, "MulticastRoutingCommandHeader size is not 2 bytes");

struct NocUnicastCommandHeader {
    uint64_t noc_address;
};
#define NOC_SCATTER_WRITE_MAX_CHUNKS 4
static constexpr uint8_t NOC_SCATTER_WRITE_MIN_CHUNKS = 2;
struct NocUnicastScatterCommandHeader {
    uint64_t noc_address[NOC_SCATTER_WRITE_MAX_CHUNKS];
    uint16_t chunk_size[NOC_SCATTER_WRITE_MAX_CHUNKS - 1];  // last chunk size is implicit
    uint8_t chunk_count;
    uint8_t reserved = 0;

    NocUnicastScatterCommandHeader() = delete;
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,hicpp-member-init)
    NocUnicastScatterCommandHeader(
        std::initializer_list<uint64_t> addresses, std::initializer_list<uint16_t> chunk_sizes = {}) {
        const size_t num_addresses = addresses.size();
        this->chunk_count = static_cast<uint8_t>(num_addresses);

        size_t idx = 0;
        for (auto addr : addresses) {
            this->noc_address[idx++] = addr;
        }
        while (idx < NOC_SCATTER_WRITE_MAX_CHUNKS) {
            this->noc_address[idx++] = 0;
        }

        idx = 0;
        for (auto size : chunk_sizes) {
            this->chunk_size[idx++] = size;
        }
        while (idx < NOC_SCATTER_WRITE_MAX_CHUNKS - 1) {
            this->chunk_size[idx++] = 0;
        }
    }

    // Requires: addresses[num_addresses], chunk_sizes[num_addresses - 1].
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,hicpp-member-init)
    NocUnicastScatterCommandHeader(
        const uint64_t* addresses, const uint16_t* chunk_sizes, const uint8_t num_addresses) :
        chunk_count(num_addresses) {
#if defined(KERNEL_BUILD) || defined(FW_BUILD)
        ASSERT(num_addresses > 0 && num_addresses <= NOC_SCATTER_WRITE_MAX_CHUNKS);
#endif

        uint8_t idx = 0;
        for (; idx < num_addresses; idx++) {
            this->noc_address[idx] = addresses[idx];
        }
        while (idx < NOC_SCATTER_WRITE_MAX_CHUNKS) {
            this->noc_address[idx++] = 0;
        }

        idx = 0;
        for (; idx < num_addresses - 1; idx++) {
            this->chunk_size[idx] = chunk_sizes[idx];
        }
        while (idx < NOC_SCATTER_WRITE_MAX_CHUNKS - 1) {
            this->chunk_size[idx++] = 0;
        }
    }
};
struct NocUnicastInlineWriteCommandHeader {
    uint64_t noc_address;
    uint32_t value;
};
struct NocUnicastAtomicIncCommandHeader {
    NocUnicastAtomicIncCommandHeader(uint64_t noc_address, uint32_t val, bool flush = true) :
        noc_address(noc_address), val(val), flush(flush) {}

    uint64_t noc_address;
    uint32_t val;
    bool flush;
};
struct NocUnicastAtomicIncFusedCommandHeader {
    NocUnicastAtomicIncFusedCommandHeader(
        uint64_t noc_address, uint64_t semaphore_noc_address, uint32_t val, bool flush = true) :
        noc_address(noc_address), semaphore_noc_address(semaphore_noc_address), val(val), flush(flush) {}

    uint64_t noc_address;
    uint64_t semaphore_noc_address;
    uint32_t val;
    bool flush;
};
struct NocMulticastCommandHeader {
    uint32_t address;
    uint8_t noc_x_start;
    uint8_t noc_y_start;
    uint8_t mcast_rect_size_x;
    uint8_t mcast_rect_size_y;
};
struct NocMulticastAtomicIncCommandHeader {
    uint32_t address;
    uint32_t val;
    uint8_t noc_x_start;
    uint8_t noc_y_start;
    uint8_t size_x;
    uint8_t size_y;
};
static_assert(sizeof(NocUnicastCommandHeader) == 8, "NocUnicastCommandHeader size is not 8 bytes");
static_assert(sizeof(NocMulticastCommandHeader) == 8, "NocMulticastCommandHeader size is not 8 bytes");
static_assert(
    sizeof(NocUnicastInlineWriteCommandHeader) == 16, "NocUnicastInlineWriteCommandHeader size is not 16 bytes");
static_assert(sizeof(NocUnicastAtomicIncCommandHeader) == 16, "NocUnicastAtomicIncCommandHeader size is not 16 bytes");
static_assert(
    sizeof(NocUnicastAtomicIncFusedCommandHeader) == 24, "NocUnicastAtomicIncFusedCommandHeader size is not 24 bytes");
static_assert(
    sizeof(NocMulticastAtomicIncCommandHeader) == 12, "NocMulticastAtomicIncCommandHeader size is not 12 bytes");

// NOLINTBEGIN(cppcoreguidelines-pro-type-member-init,hicpp-member-init)
union NocCommandFields {
    NocUnicastCommandHeader unicast_write;
    NocUnicastCommandHeader unicast_read;
    NocUnicastInlineWriteCommandHeader unicast_inline_write;
    NocMulticastCommandHeader mcast_write;
    NocUnicastAtomicIncCommandHeader unicast_seminc;
    NocUnicastAtomicIncFusedCommandHeader unicast_seminc_fused;
    NocMulticastAtomicIncCommandHeader mcast_seminc;
    NocUnicastScatterCommandHeader unicast_scatter_write;
};
// NOLINTEND(cppcoreguidelines-pro-type-member-init,hicpp-member-init)
static_assert(sizeof(NocCommandFields) == 40, "CommandFields size is not 40 bytes");

struct UDMWriteControlHeader {
    uint8_t src_chip_id;
    uint16_t src_mesh_id;
    uint8_t src_noc_x;
    uint8_t src_noc_y;
    uint8_t risc_id;
    uint8_t transaction_id;
    uint8_t posted;
    uint8_t initial_direction;
} __attribute__((packed));

struct UDMReadControlHeader {
    uint8_t src_chip_id;
    uint16_t src_mesh_id;
    uint8_t src_noc_x;
    uint8_t src_noc_y;
    uint32_t src_l1_address;
    uint32_t size_bytes;
    uint8_t risc_id;
    uint8_t transaction_id;
    uint8_t initial_direction;
} __attribute__((packed));

static_assert(sizeof(UDMWriteControlHeader) == 9, "UDMWriteControlHeader size is not 9 bytes");
static_assert(sizeof(UDMReadControlHeader) == 16, "UDMReadControlHeader size is not 16 bytes");

union UDMControlFields {
    UDMWriteControlHeader write;
    UDMReadControlHeader read;
} __attribute__((packed));

static_assert(sizeof(UDMControlFields) == 16, "UDMControlFields size is not 16 bytes");

// TODO: wrap this in a debug version that holds type info so we can assert for field/command/
// NOLINTBEGIN(cppcoreguidelines-pro-type-member-init,hicpp-member-init)
template <typename Derived>
struct PacketHeaderBase {
private:
    PacketHeaderBase() = default;
    friend Derived;

public:
    NocCommandFields command_fields;  // size = 40B due to scatter metadata
    uint16_t payload_size_bytes;
    // TODO: trim this down noc_send_type 2 bits (4 values):
    //   -> unicast_write, mcast_write, unicast_seminc, mcast_seminc
    // For now, kept it separate so I could do reads which would be handled differently
    // but for our purposes we shouldn't need read so we should be able to omit the support
    NocSendType noc_send_type;
    // Used only by the EDM sender and receiver channels. Populated by EDM sender channel to
    // indicate to the receiver channel what channel was the source of this packet. Reserved
    // otherwise.
    uint8_t src_ch_id;

    // Returns size of payload in bytes - TODO: convert to words (4B)
    size_t get_payload_size_excluding_header() volatile const { return this->payload_size_bytes; }

    size_t get_payload_size_including_header() volatile const {
        return get_payload_size_excluding_header() + sizeof(Derived);
    }

    const volatile NocCommandFields& get_command_fields() volatile const { return this->command_fields; }
    NocSendType get_noc_send_type() volatile const { return this->noc_send_type; }

    // Setters for noc_send_type, routing_fields, and command_fields
    void set_noc_send_type(NocSendType& type) { this->noc_send_type = type; }
    void set_command_fields(NocCommandFields& fields) { this->command_fields = fields; }

    Derived& to_chip_unicast(uint8_t distance_in_hops) {
        static_cast<Derived*>(this)->to_chip_unicast_impl(distance_in_hops);
        return *static_cast<Derived*>(this);
    }

    Derived& to_chip_multicast(const MulticastRoutingCommandHeader& mcast_routing_command_header) {
        static_cast<Derived*>(this)->to_chip_multicast_impl(mcast_routing_command_header);
        return *static_cast<Derived*>(this);
    }

    Derived& to_noc_unicast_write(
        const NocUnicastCommandHeader& noc_unicast_command_header, size_t payload_size_bytes) {
#if defined(KERNEL_BUILD) || defined(FW_BUILD)
        this->noc_send_type = NOC_UNICAST_WRITE;
        auto noc_address_components = get_noc_address_components(noc_unicast_command_header.noc_address);
        auto noc_addr = safe_get_noc_addr(
            noc_address_components.first.x,
            noc_address_components.first.y,
            noc_address_components.second,
            edm_to_local_chip_noc);
        NocUnicastCommandHeader modified_command_header = noc_unicast_command_header;
        modified_command_header.noc_address = noc_addr;

        this->command_fields.unicast_write = modified_command_header;
        this->payload_size_bytes = payload_size_bytes;
#else
        TT_THROW("Calling to_noc_unicast_write from host is unsupported");
#endif
        return *static_cast<Derived*>(this);
    }

    Derived& to_noc_unicast_read(const NocUnicastCommandHeader& noc_unicast_command_header, size_t payload_size_bytes) {
#if defined(KERNEL_BUILD) || defined(FW_BUILD)
#ifndef UDM_MODE
        static_assert(always_false_v<Derived>, "to_noc_unicast_read requires UDM mode / relay extension to be enabled");
#endif
        this->noc_send_type = NOC_UNICAST_READ;
        auto noc_address_components = get_noc_address_components(noc_unicast_command_header.noc_address);
        auto noc_addr = safe_get_noc_addr(
            noc_address_components.first.x,
            noc_address_components.first.y,
            noc_address_components.second,
            edm_to_local_chip_noc);
        NocUnicastCommandHeader modified_command_header = noc_unicast_command_header;
        modified_command_header.noc_address = noc_addr;

        this->command_fields.unicast_read = modified_command_header;
        this->payload_size_bytes = payload_size_bytes;
#else
        TT_THROW("Calling to_noc_unicast_read from host is unsupported");
#endif
        return *static_cast<Derived*>(this);
    }

    Derived& to_noc_unicast_inline_write(const NocUnicastInlineWriteCommandHeader& noc_unicast_command_header) {
#if defined(KERNEL_BUILD) || defined(FW_BUILD)
        this->noc_send_type = NOC_UNICAST_INLINE_WRITE;
        auto noc_address_components = get_noc_address_components(noc_unicast_command_header.noc_address);
        auto noc_addr = safe_get_noc_addr(
            noc_address_components.first.x,
            noc_address_components.first.y,
            noc_address_components.second,
            edm_to_local_chip_noc);
        NocUnicastInlineWriteCommandHeader modified_command_header = noc_unicast_command_header;
        modified_command_header.noc_address = noc_addr;

        this->command_fields.unicast_inline_write = modified_command_header;
        this->payload_size_bytes = 0;
#else
        TT_THROW("Calling to_noc_unicast_inline_write from host is unsupported");
#endif
        return *static_cast<Derived*>(this);
    }

    Derived& to_noc_multicast(
        const NocMulticastCommandHeader& noc_multicast_command_header, size_t payload_size_bytes) {
        this->noc_send_type = NOC_MULTICAST_WRITE;
        this->command_fields.mcast_write = noc_multicast_command_header;
        this->payload_size_bytes = payload_size_bytes;
        return *static_cast<Derived*>(this);
    }

    Derived& to_noc_unicast_atomic_inc(const NocUnicastAtomicIncCommandHeader& noc_unicast_atomic_inc_command_header) {
#if defined(KERNEL_BUILD) || defined(FW_BUILD)
        this->noc_send_type = NOC_UNICAST_ATOMIC_INC;
        auto noc_address_components = get_noc_address_components(noc_unicast_atomic_inc_command_header.noc_address);
        auto noc_addr = safe_get_noc_addr(
            noc_address_components.first.x,
            noc_address_components.first.y,
            noc_address_components.second,
            edm_to_local_chip_noc);
        NocUnicastAtomicIncCommandHeader modified_command_header = noc_unicast_atomic_inc_command_header;
        modified_command_header.noc_address = noc_addr;

        this->command_fields.unicast_seminc = modified_command_header;
        this->payload_size_bytes = 0;
#else
        TT_THROW("Calling to_noc_unicast_atomic_inc from host is unsupported");
#endif
        return *static_cast<Derived*>(this);
    }

    Derived& to_noc_multicast_atomic_inc(
        const NocMulticastAtomicIncCommandHeader& noc_multicast_atomic_inc_command_header, size_t payload_size_bytes) {
        this->noc_send_type = NOC_MULTICAST_ATOMIC_INC;
        this->command_fields.mcast_seminc = noc_multicast_atomic_inc_command_header;
        this->payload_size_bytes = payload_size_bytes;
        return *static_cast<Derived*>(this);
    }

    volatile Derived* to_chip_unicast(uint8_t distance_in_hops) volatile {
        static_cast<volatile Derived*>(this)->to_chip_unicast_impl(distance_in_hops);
        return static_cast<volatile Derived*>(this);
    }

    volatile Derived* to_chip_multicast(const MulticastRoutingCommandHeader& mcast_routing_command_header) volatile {
        static_cast<volatile Derived*>(this)->to_chip_multicast_impl(mcast_routing_command_header);
        return static_cast<volatile Derived*>(this);
    }

    volatile Derived* to_noc_unicast_write(
        const NocUnicastCommandHeader& noc_unicast_command_header, size_t payload_size_bytes) volatile {
#if defined(KERNEL_BUILD) || defined(FW_BUILD)
        this->noc_send_type = NOC_UNICAST_WRITE;
        auto noc_address_components = get_noc_address_components(noc_unicast_command_header.noc_address);
        auto noc_addr = safe_get_noc_addr(
            noc_address_components.first.x,
            noc_address_components.first.y,
            noc_address_components.second,
            edm_to_local_chip_noc);

        this->command_fields.unicast_write.noc_address = noc_addr;
        this->payload_size_bytes = payload_size_bytes;
#else
        TT_THROW("Calling to_noc_unicast_write from host is unsupported");
#endif
        return static_cast<volatile Derived*>(this);
    }

    volatile Derived* to_noc_unicast_read(
        const NocUnicastCommandHeader& noc_unicast_command_header, size_t payload_size_bytes) volatile {
#if defined(KERNEL_BUILD) || defined(FW_BUILD)
#ifndef UDM_MODE
        static_assert(always_false_v<Derived>, "to_noc_unicast_read requires UDM mode / relay extension to be enabled");
#endif
        this->noc_send_type = NOC_UNICAST_READ;
        auto noc_address_components = get_noc_address_components(noc_unicast_command_header.noc_address);
        auto noc_addr = safe_get_noc_addr(
            noc_address_components.first.x,
            noc_address_components.first.y,
            noc_address_components.second,
            edm_to_local_chip_noc);

        this->command_fields.unicast_read.noc_address = noc_addr;
        this->payload_size_bytes = payload_size_bytes;
#else
        TT_THROW("Calling to_noc_unicast_read from host is unsupported");
#endif
        return static_cast<volatile Derived*>(this);
    }

    volatile Derived* to_noc_unicast_scatter_write(
        const NocUnicastScatterCommandHeader& noc_unicast_scatter_command_header, size_t payload_size_bytes) volatile {
#if defined(KERNEL_BUILD) || defined(FW_BUILD)
        this->noc_send_type = NOC_UNICAST_SCATTER_WRITE;
        const uint8_t chunk_count = noc_unicast_scatter_command_header.chunk_count;
        ASSERT(chunk_count >= NOC_SCATTER_WRITE_MIN_CHUNKS && chunk_count <= NOC_SCATTER_WRITE_MAX_CHUNKS);

        this->command_fields.unicast_scatter_write.chunk_count = chunk_count;

        for (uint8_t i = 0; i < chunk_count; i++) {
            auto noc_address_components = get_noc_address_components(noc_unicast_scatter_command_header.noc_address[i]);
            auto noc_addr = safe_get_noc_addr(
                noc_address_components.first.x,
                noc_address_components.first.y,
                noc_address_components.second,
                edm_to_local_chip_noc);
            this->command_fields.unicast_scatter_write.noc_address[i] = noc_addr;
        }
        for (uint8_t i = chunk_count; i < NOC_SCATTER_WRITE_MAX_CHUNKS; i++) {
            this->command_fields.unicast_scatter_write.noc_address[i] = 0;
        }

        const uint8_t chunk_size_count = chunk_count - 1;
        size_t accumulated_chunk_bytes = 0;
        for (uint8_t i = 0; i < chunk_size_count; i++) {
            uint16_t chunk_bytes = noc_unicast_scatter_command_header.chunk_size[i];
            ASSERT(chunk_bytes > 0);
            accumulated_chunk_bytes += chunk_bytes;
            this->command_fields.unicast_scatter_write.chunk_size[i] = chunk_bytes;
        }
        for (uint8_t i = chunk_size_count; i < NOC_SCATTER_WRITE_MAX_CHUNKS - 1; i++) {
            this->command_fields.unicast_scatter_write.chunk_size[i] = 0;
        }

        ASSERT(accumulated_chunk_bytes < payload_size_bytes);
        this->payload_size_bytes = static_cast<uint16_t>(payload_size_bytes);
#else
        TT_THROW("Calling to_noc_unicast_write from host is unsupported");
#endif
        return static_cast<volatile Derived*>(this);
    }

    volatile Derived* to_noc_unicast_inline_write(
        const NocUnicastInlineWriteCommandHeader& noc_unicast_command_header) volatile {
#if defined(KERNEL_BUILD) || defined(FW_BUILD)
        this->noc_send_type = NOC_UNICAST_INLINE_WRITE;
        auto noc_address_components = get_noc_address_components(noc_unicast_command_header.noc_address);
        auto noc_addr = safe_get_noc_addr(
            noc_address_components.first.x,
            noc_address_components.first.y,
            noc_address_components.second,
            edm_to_local_chip_noc);

        this->command_fields.unicast_inline_write.noc_address = noc_addr;
        this->command_fields.unicast_inline_write.value = noc_unicast_command_header.value;
        this->payload_size_bytes = 0;
#else
        TT_THROW("Calling to_noc_unicast_inline_write from host is unsupported");
#endif
        return static_cast<volatile Derived*>(this);
    }

    volatile Derived* to_noc_multicast(
        const NocMulticastCommandHeader& noc_multicast_command_header, size_t payload_size_bytes) volatile {
        this->noc_send_type = NOC_MULTICAST_WRITE;
        this->command_fields.mcast_write.mcast_rect_size_x = noc_multicast_command_header.mcast_rect_size_x;
        this->command_fields.mcast_write.mcast_rect_size_y = noc_multicast_command_header.mcast_rect_size_y;
        this->command_fields.mcast_write.noc_x_start = noc_multicast_command_header.noc_x_start;
        this->command_fields.mcast_write.noc_y_start = noc_multicast_command_header.noc_y_start;
        this->payload_size_bytes = payload_size_bytes;
        this->command_fields.mcast_write.address = noc_multicast_command_header.address;
        return static_cast<volatile Derived*>(this);
    }

    volatile Derived* to_noc_fused_unicast_write_atomic_inc(
        const NocUnicastAtomicIncFusedCommandHeader& noc_fused_unicast_write_atomic_inc_command_header,
        size_t payload_size_bytes) volatile {
#if defined(KERNEL_BUILD) || defined(FW_BUILD)
        this->noc_send_type = NOC_FUSED_UNICAST_ATOMIC_INC;
        auto noc_address_components =
            get_noc_address_components(noc_fused_unicast_write_atomic_inc_command_header.noc_address);
        auto noc_addr = safe_get_noc_addr(
            noc_address_components.first.x,
            noc_address_components.first.y,
            noc_address_components.second,
            edm_to_local_chip_noc);

        auto semaphore_noc_address_components =
            get_noc_address_components(noc_fused_unicast_write_atomic_inc_command_header.semaphore_noc_address);
        auto semaphore_noc_addr = safe_get_noc_addr(
            semaphore_noc_address_components.first.x,
            semaphore_noc_address_components.first.y,
            semaphore_noc_address_components.second,
            edm_to_local_chip_noc);

        this->command_fields.unicast_seminc_fused.noc_address = noc_addr;
        this->command_fields.unicast_seminc_fused.semaphore_noc_address = semaphore_noc_addr;
        this->command_fields.unicast_seminc_fused.val = noc_fused_unicast_write_atomic_inc_command_header.val;
        this->command_fields.unicast_seminc_fused.flush = noc_fused_unicast_write_atomic_inc_command_header.flush;

        this->payload_size_bytes = payload_size_bytes;
#else
        TT_THROW("Calling to_noc_unicast_atomic_inc from host is unsupported");
#endif
        return static_cast<volatile Derived*>(this);
    }

    volatile Derived* to_noc_unicast_atomic_inc(
        const NocUnicastAtomicIncCommandHeader& noc_unicast_atomic_inc_command_header) volatile {
#if defined(KERNEL_BUILD) || defined(FW_BUILD)
        this->noc_send_type = NOC_UNICAST_ATOMIC_INC;
        auto noc_address_components = get_noc_address_components(noc_unicast_atomic_inc_command_header.noc_address);
        auto noc_addr = safe_get_noc_addr(
            noc_address_components.first.x,
            noc_address_components.first.y,
            noc_address_components.second,
            edm_to_local_chip_noc);

        this->command_fields.unicast_seminc.noc_address = noc_addr;
        this->command_fields.unicast_seminc.val = noc_unicast_atomic_inc_command_header.val;
        this->command_fields.unicast_seminc.flush = noc_unicast_atomic_inc_command_header.flush;
        this->payload_size_bytes = 0;
#else
        TT_THROW("Calling to_noc_unicast_atomic_inc from host is unsupported");
#endif
        return static_cast<volatile Derived*>(this);
    }

    volatile Derived* to_noc_multicast_atomic_inc(
        const NocMulticastAtomicIncCommandHeader& noc_multicast_atomic_inc_command_header,
        size_t payload_size_bytes) volatile {
        this->noc_send_type = NOC_MULTICAST_ATOMIC_INC;
        this->command_fields.mcast_seminc.address = noc_multicast_atomic_inc_command_header.address;
        this->command_fields.mcast_seminc.noc_x_start = noc_multicast_atomic_inc_command_header.noc_x_start;
        this->command_fields.mcast_seminc.noc_y_start = noc_multicast_atomic_inc_command_header.noc_y_start;
        this->command_fields.mcast_seminc.size_x = noc_multicast_atomic_inc_command_header.size_x;
        this->command_fields.mcast_seminc.size_y = noc_multicast_atomic_inc_command_header.size_y;
        this->command_fields.mcast_seminc.val = noc_multicast_atomic_inc_command_header.val;
        this->payload_size_bytes = payload_size_bytes;
        return static_cast<volatile Derived*>(this);
    }

    void set_src_ch_id(uint8_t ch_id) volatile { this->src_ch_id = ch_id; }
};

struct PacketHeader : public PacketHeaderBase<PacketHeader> {
    ChipSendType chip_send_type;
    RoutingFields routing_fields;
    // Sort of hack to work-around DRAM read alignment issues that must be 32B aligned
    // To simplify worker kernel code, we for now decide to pad up the packet header
    // to 32B so the user can simplify shift into their CB chunk by sizeof(tt::tt_fabric::PacketHeader)
    // and automatically work around the DRAM read alignment bug.
    //
    // Future changes will remove this padding and require the worker kernel to be aware of this bug
    // and pad their own CBs conditionally when reading from DRAM. It'll be up to the users to
    // manage this complexity.
    uint8_t padding0[18];

    static uint32_t calculate_chip_unicast_routing_fields_value(uint8_t distance_in_hops) {
        return RoutingFields::LAST_CHIP_IN_MCAST_VAL | distance_in_hops;
    }
    static uint32_t calculate_chip_multicast_routing_fields_value(
        const MulticastRoutingCommandHeader& chip_multicast_command_header) {
        return ((static_cast<uint8_t>(chip_multicast_command_header.range_hops)
                 << RoutingFields::START_DISTANCE_FIELD_BIT_WIDTH)) |
               static_cast<uint8_t>(chip_multicast_command_header.start_distance_in_hops);
    }

public:
    // Setters for PacketHeader-specific fields
    void set_chip_send_type(ChipSendType& type) { this->chip_send_type = type; }

    void set_routing_fields(RoutingFields& fields) { this->routing_fields = fields; }

    void to_chip_unicast_impl(uint8_t distance_in_hops) {
        this->chip_send_type = CHIP_UNICAST;
        this->routing_fields.value = PacketHeader::calculate_chip_unicast_routing_fields_value(distance_in_hops);
    }
    void to_chip_multicast_impl(const MulticastRoutingCommandHeader& chip_multicast_command_header) {
        this->chip_send_type = CHIP_MULTICAST;
        this->routing_fields.value =
            PacketHeader::calculate_chip_multicast_routing_fields_value(chip_multicast_command_header);
    }

    void to_chip_unicast_impl(uint8_t distance_in_hops) volatile {
        this->chip_send_type = CHIP_UNICAST;
        this->routing_fields.value = PacketHeader::calculate_chip_unicast_routing_fields_value(distance_in_hops);
    }
    void to_chip_multicast_impl(const MulticastRoutingCommandHeader& chip_multicast_command_header) volatile {
        this->chip_send_type = CHIP_MULTICAST;
        this->routing_fields.value =
            PacketHeader::calculate_chip_multicast_routing_fields_value(chip_multicast_command_header);
    }
};

// Primary template for 1D routing fields with route buffer (ExtensionWords >= 1)
template <uint32_t ExtensionWords = 1>
struct LowLatencyRoutingFieldsT {
    // Type alias to reference centralized constants
    using LowLatencyFields = RoutingFieldsConstants::LowLatency;

    // Template-specific constants
    static constexpr uint32_t MAX_NUM_ENCODINGS = LowLatencyFields::BASE_HOPS * (1 + ExtensionWords);

    // Block >64 hops until memory map is updated
    static_assert(
        ExtensionWords <= 3,
        "ERROR: 1D routing with >64 hops (ExtensionWords > 3) requires memory map updates.\n"
        "Current L1 allocation (ROUTING_PATH_SIZE_1D = 1024 bytes) supports max 64 hops.");

    uint32_t value;                         // Active routing field (always read by router)
    uint32_t route_buffer[ExtensionWords];  // Extension storage

    /**
     * Copy routing fields to volatile destination (packet header).
     * Needed because router updates a cached copy then writes back to packet.
     */
    void copy_to(volatile LowLatencyRoutingFieldsT<ExtensionWords>* dest) const {
        dest->value = this->value;
        if constexpr (ExtensionWords > 0) {
            for (uint32_t i = 0; i < ExtensionWords; i++) {
                dest->route_buffer[i] = this->route_buffer[i];
            }
        }
    }

    /**
     * Initialize from buffer (used when unpacking encoder output).
     * Buffer layout: [value, route_buffer[0], route_buffer[1], ...]
     */
    static LowLatencyRoutingFieldsT<ExtensionWords> from_buffer(const uint32_t* buffer) {
        LowLatencyRoutingFieldsT<ExtensionWords> result;
        result.value = buffer[0];
        if constexpr (ExtensionWords > 0) {
            for (uint32_t i = 0; i < ExtensionWords; i++) {
                result.route_buffer[i] = buffer[i + 1];
            }
        }
        return result;
    }

} __attribute__((packed));

// Partial specialization for 16-hop mode
template <>
struct LowLatencyRoutingFieldsT<0> {
    // Type alias to reference centralized constants
    using LowLatencyFields = RoutingFieldsConstants::LowLatency;

    static constexpr uint32_t MAX_NUM_ENCODINGS = LowLatencyFields::BASE_HOPS;  // 16 hops max

    uint32_t value;  // Only field - no route_buffer member

    /**
     * Copy to packet header (value only in 16-hop mode)
     */
    void copy_to(volatile LowLatencyRoutingFieldsT<0>* dest) const { dest->value = this->value; }

    /**
     * Initialize from buffer (ExtensionWords=0 specialization)
     */
    static LowLatencyRoutingFieldsT<0> from_buffer(const uint32_t* buffer) {
        LowLatencyRoutingFieldsT<0> result;
        result.value = buffer[0];
        return result;
    }
} __attribute__((packed));

// Template for 1D packet headers with variable routing field sizes
template <uint32_t ExtensionWords = 0>
struct LowLatencyPacketHeaderT : public PacketHeaderBase<LowLatencyPacketHeaderT<ExtensionWords>> {
    LowLatencyRoutingFieldsT<ExtensionWords> routing_fields;

    // Constexpr helpers for size calculation
private:
    static constexpr size_t base_size() {
        // PacketHeaderBase: NocCommandFields(40) + payload_size_bytes(2) + noc_send_type(1) + src_ch_id(1)
        return sizeof(NocCommandFields) + sizeof(uint16_t) + sizeof(NocSendType) + sizeof(uint8_t);
    }

    static constexpr size_t routing_fields_size() { return sizeof(LowLatencyRoutingFieldsT<ExtensionWords>); }

    static constexpr size_t unpadded_size() { return base_size() + routing_fields_size(); }

    static constexpr size_t target_size() {
        // 48B for 16-hop (ExtensionWords=0), 64B for 32-hop and above
        return (ExtensionWords == 0) ? 48 : 64;
    }

    static constexpr size_t padding_size() { return target_size() - unpadded_size(); }

public:
    // Explicit padding to reach target size
    // ExtensionWords=0: 48B total → 0B padding (44 base + 4 routing = 48)
    // ExtensionWords=1: 64B total → 12B padding (44 base + 8 routing + 12 padding = 64)
    // ExtensionWords=2: 64B total → 8B padding (44 base + 12 routing + 8 padding = 64)
    // ExtensionWords=3: 64B total → 4B padding (44 base + 16 routing + 4 padding = 64)
    uint8_t padding0[padding_size()];

    // Helper to calculate routing fields for unicast
    // Delegates to routing_encoding::encode_1d_unicast() - see fabric_common.h for encoding details
    static LowLatencyRoutingFieldsT<ExtensionWords> calculate_chip_unicast_routing_fields(uint8_t distance_in_hops) {
        // Use canonical encoder
        constexpr uint32_t num_words = 1 + ExtensionWords;
        uint32_t buffer[num_words];
        routing_encoding::encode_1d_unicast(distance_in_hops, buffer, num_words);

        // Unpack using helper
        return LowLatencyRoutingFieldsT<ExtensionWords>::from_buffer(buffer);
    }

    // Helper to calculate routing fields for multicast
    // Delegates to routing_encoding::encode_1d_multicast() - see fabric_common.h for encoding details
    static LowLatencyRoutingFieldsT<ExtensionWords> calculate_chip_multicast_routing_fields(
        const MulticastRoutingCommandHeader& chip_multicast_command_header) {
        const uint32_t start_hop = chip_multicast_command_header.start_distance_in_hops;
        const uint32_t range_hops = chip_multicast_command_header.range_hops;

        // Delegate to canonical encoder
        uint32_t buffer[1 + ExtensionWords];
        routing_encoding::encode_1d_multicast(start_hop, range_hops, buffer, 1 + ExtensionWords);

        // Unpack using helper
        return LowLatencyRoutingFieldsT<ExtensionWords>::from_buffer(buffer);
    }

    // Specialized implementations for LowLatencyPacketHeader
    void set_routing_fields(LowLatencyRoutingFieldsT<ExtensionWords>& fields) { this->routing_fields = fields; }

    void to_chip_unicast_impl(uint8_t distance_in_hops) {
        this->routing_fields = calculate_chip_unicast_routing_fields(distance_in_hops);
    }
    void to_chip_multicast_impl(const MulticastRoutingCommandHeader& chip_multicast_command_header) {
        this->routing_fields = calculate_chip_multicast_routing_fields(chip_multicast_command_header);
    }

    void to_chip_unicast_impl(uint8_t distance_in_hops) volatile {
        auto routing = calculate_chip_unicast_routing_fields(distance_in_hops);
        routing.copy_to(&this->routing_fields);
    }
    void to_chip_multicast_impl(const MulticastRoutingCommandHeader& chip_multicast_command_header) volatile {
        auto routing = calculate_chip_multicast_routing_fields(chip_multicast_command_header);
        routing.copy_to(&this->routing_fields);
    }
};

// Validate expected sizes with detailed checks
static_assert(sizeof(LowLatencyPacketHeaderT<0>) == 48, "16-hop total must be 48B");
static_assert(sizeof(LowLatencyPacketHeaderT<1>) == 64, "32-hop total must be 64B");
static_assert(sizeof(LowLatencyPacketHeaderT<2>) == 64, "48-hop total must be 64B");  // NEW for 4×64
static_assert(sizeof(LowLatencyPacketHeaderT<3>) == 64, "64-hop total must be 64B");  // NEW for 4×64

// Conditional type selection based on injected define
#ifndef FABRIC_1D_PKT_HDR_EXTENSION_WORDS
// Default: backward compatibility (host-side compilation or no define set)
using LowLatencyPacketHeader = LowLatencyPacketHeaderT<1>;  // 64B, 32-hop
using LowLatencyRoutingFields = LowLatencyRoutingFieldsT<1>;
#else
// Device-side: use injected define to select appropriate template instantiation
using LowLatencyPacketHeader = LowLatencyPacketHeaderT<FABRIC_1D_PKT_HDR_EXTENSION_WORDS>;
using LowLatencyRoutingFields = LowLatencyRoutingFieldsT<FABRIC_1D_PKT_HDR_EXTENSION_WORDS>;
#endif

// 2D Mesh routing fields struct
// This struct contains routing STATE (hop_index union) for 2D packet headers.
// All constants are centralized in RoutingFieldsConstants::Mesh (fabric_common.h).
// Access constants via: MeshRoutingFields (aliased from RoutingFieldsConstants::Mesh)
struct LowLatencyMeshRoutingFields {
    // Type alias to reference centralized constants
    using MeshRoutingFields = RoutingFieldsConstants::Mesh;

    // Routing state (the actual data members)
    union {
        uint32_t value;  // Referenced for fast increment when updating hop count in packet header.
                         // Also used when doing noc inline dword write to update packet header in next hop
                         // router.
        struct {
            uint16_t hop_index;
            uint8_t branch_east_offset;  // Referenced when updating hop index for mcast east branch
            uint8_t branch_west_offset;  // Referenced when updating hop index for mcast west branch
        };
    };
};

// TODO: https://github.com/tenstorrent/tt-metal/issues/32237
// Primary template for 2D routing headers with variable route buffer size
template <int RouteBufferSize = 35>
struct HybridMeshPacketHeaderT : PacketHeaderBase<HybridMeshPacketHeaderT<RouteBufferSize>> {
    // Block route buffers >67 bytes until memory map is updated
    static_assert(
        RouteBufferSize <= 67,
        "ERROR: 2D routing with >67-byte route buffer requires memory map updates.\n"
        "Current L1 allocation (ROUTING_PATH_SIZE_2D = 1024 bytes) supports max 67 hops.");

    LowLatencyMeshRoutingFields routing_fields;
    uint8_t route_buffer[RouteBufferSize];
    union {
        struct {
            uint16_t dst_start_chip_id;
            uint16_t dst_start_mesh_id;
        };
        uint32_t dst_start_node_id;  // Used for efficiently writing the dst info
    };
    union {
        uint16_t mcast_params[4];  // Array representing the hops in each direction
        uint64_t mcast_params_64;  // Used for efficiently writing to the mcast_params array
    };
    uint8_t is_mcast_active;

    void to_chip_unicast_impl(uint8_t distance_in_hops) {}
    void to_chip_multicast_impl(const MulticastRoutingCommandHeader& chip_multicast_command_header) {}

    void to_chip_unicast_impl(uint8_t distance_in_hops) volatile {}
    void to_chip_multicast_impl(const MulticastRoutingCommandHeader& chip_multicast_command_header) volatile {}
} __attribute__((packed, aligned(16)));

// Validate expected sizes for max-capacity tiers only (one per header size)
// Base size = 61B (command_fields:40 + payload_size:2 + noc_send_type:1 + src_ch_id:1 +
//              routing_fields:4 + dst_start:4 + mcast_params:8 + is_mcast_active:1)
static_assert(sizeof(HybridMeshPacketHeaderT<19>) == 80, "19B buffer must result in 80B header (max capacity)");
static_assert(sizeof(HybridMeshPacketHeaderT<35>) == 96, "35B buffer must result in 96B header (max capacity)");
static_assert(sizeof(HybridMeshPacketHeaderT<51>) == 112, "51B buffer must result in 112B header (max capacity)");
static_assert(sizeof(HybridMeshPacketHeaderT<67>) == 128, "67B buffer must result in 128B header (max capacity)");

// Conditional type selection based on injected define
#ifdef FABRIC_2D_PKT_HDR_ROUTE_BUFFER_SIZE
using HybridMeshPacketHeader = HybridMeshPacketHeaderT<FABRIC_2D_PKT_HDR_ROUTE_BUFFER_SIZE>;
#else
// Default: backward compatibility (96B header with 35B route buffer)
using HybridMeshPacketHeader = HybridMeshPacketHeaderT<35>;
#endif

struct UDMHybridMeshPacketHeader : public HybridMeshPacketHeader {
    UDMControlFields udm_control;

    // Override to return correct size for UDMHybridMeshPacketHeader
    size_t get_payload_size_including_header() volatile const {
        return get_payload_size_excluding_header() + sizeof(UDMHybridMeshPacketHeader);
    }
} __attribute__((packed, aligned(16)));
static_assert(
    sizeof(UDMHybridMeshPacketHeader) == sizeof(HybridMeshPacketHeader) + sizeof(UDMControlFields),
    "UDMHybridMeshPacketHeader size must equal base + UDMControlFields");
// NOLINTEND(cppcoreguidelines-pro-type-member-init,hicpp-member-init)

// TODO: When we remove the 32B padding requirement, reduce to 16B size check
static_assert(sizeof(PacketHeader) == 64, "sizeof(PacketHeader) is not equal to 64B");

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#ifndef ROUTING_MODE
#define PACKET_HEADER_TYPE tt::tt_fabric::LowLatencyPacketHeader
#define ROUTING_FIELDS_TYPE tt::tt_fabric::LowLatencyRoutingFields
#else

// Check if UDM_MODE is defined
#ifdef UDM_MODE

#if (                                                                \
    ((ROUTING_MODE & (ROUTING_MODE_1D | ROUTING_MODE_LINE)) != 0) || \
    ((ROUTING_MODE & (ROUTING_MODE_1D | ROUTING_MODE_RING)) != 0) || \
    ((ROUTING_MODE & (ROUTING_MODE_1D | ROUTING_MODE_NEIGHBOR_EXCHANGE)) != 0))
// 1D routing with UDM is not supported
static_assert(false, "UDM mode does not support 1D routing - use 2D routing instead");

#elif (                                                              \
    ((ROUTING_MODE & (ROUTING_MODE_2D | ROUTING_MODE_MESH)) != 0) || \
    ((ROUTING_MODE & (ROUTING_MODE_2D | ROUTING_MODE_TORUS)) != 0))
// 2D routing with UDM
#if (ROUTING_MODE & ROUTING_MODE_LOW_LATENCY) != 0
#define PACKET_HEADER_TYPE tt::tt_fabric::UDMHybridMeshPacketHeader
#define ROUTING_FIELDS_TYPE tt::tt_fabric::LowLatencyMeshRoutingFields
#else
static_assert(false, "UDM mode requires LOW_LATENCY routing for 2D fabric");
#endif

#else
static_assert(false, "non supported ROUTING_MODE with UDM: " TOSTRING(ROUTING_MODE));
#endif

#else  // UDM_MODE not defined - use default non-UDM headers

#if (                                                                \
    ((ROUTING_MODE & (ROUTING_MODE_1D | ROUTING_MODE_LINE)) != 0) || \
    ((ROUTING_MODE & (ROUTING_MODE_1D | ROUTING_MODE_RING)) != 0))
#if ((ROUTING_MODE & ROUTING_MODE_LOW_LATENCY)) != 0
#define PACKET_HEADER_TYPE tt::tt_fabric::LowLatencyPacketHeader
#define ROUTING_FIELDS_TYPE tt::tt_fabric::LowLatencyRoutingFields

#else
#define PACKET_HEADER_TYPE tt::tt_fabric::PacketHeader
#define ROUTING_FIELDS_TYPE tt::tt_fabric::RoutingFields
#endif

#elif (                                                              \
    ((ROUTING_MODE & (ROUTING_MODE_2D | ROUTING_MODE_MESH)) != 0) || \
    ((ROUTING_MODE & (ROUTING_MODE_2D | ROUTING_MODE_TORUS)) != 0))
#if (ROUTING_MODE & ROUTING_MODE_LOW_LATENCY) != 0
#define PACKET_HEADER_TYPE tt::tt_fabric::HybridMeshPacketHeader
#define ROUTING_FIELDS_TYPE tt::tt_fabric::LowLatencyMeshRoutingFields
#else
#define PACKET_HEADER_TYPE packet_header_t
#endif
#else
static_assert(false, "non supported ROUTING_MODE: " TOSTRING(ROUTING_MODE));
#endif

#endif  // UDM_MODE

#endif  // ROUTING_MODE

}  // namespace tt::tt_fabric

#pragma GCC diagnostic pop
// NOLINTEND(misc-unused-parameters)
