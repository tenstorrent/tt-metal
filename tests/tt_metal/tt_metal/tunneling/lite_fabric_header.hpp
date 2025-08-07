// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <climits>
#include <cstddef>
#include <cstdint>

namespace lite_fabric {

struct NocUnicastCommandHeader {
    uint64_t noc_address;
};

struct NocReadCommandHeader {
    uint64_t noc_address;
    uint64_t event;
};

union LiteFabricCommandFields {
    NocUnicastCommandHeader noc_unicast;
    NocReadCommandHeader noc_read;
};
static_assert(sizeof(LiteFabricCommandFields) == 16, "CommandFields size is not 24 bytes");

struct LiteFabricRoutingFields {
    static constexpr uint32_t FIELD_WIDTH = 2;
    static constexpr uint32_t FIELD_MASK = 0b11;
    static constexpr uint32_t NOOP = 0b00;
    static constexpr uint32_t WRITE_ONLY = 0b01;
    static constexpr uint32_t FORWARD_ONLY = 0b10;
    static constexpr uint32_t WRITE_AND_FORWARD = 0b11;
    static constexpr uint32_t MAX_NUM_ENCODINGS = sizeof(uint32_t) * CHAR_BIT / FIELD_WIDTH;
    static constexpr uint32_t FWD_ONLY_FIELD = 0xAAAAAAAA;
    static constexpr uint32_t WR_ONLY_FIELD = 0x55555555;
    uint32_t value;
};

enum class NocSendType : uint8_t {
    NOC_UNICAST_WRITE = 0,
    NOC_READ = 1,
    NOC_SEND_TYPE_LAST = NOC_READ,
};

struct LiteFabricHeader {
    LiteFabricCommandFields command_fields;
    uint8_t unaligned_offset;
    uint16_t payload_size_bytes;
    lite_fabric::NocSendType noc_send_type;
    // Used only by the EDM sender and receiver channels. Populated by EDM sender channel to
    // indicate to the receiver channel what channel was the source of this packet. Reserved
    // otherwise.
    uint8_t src_ch_id;
    LiteFabricRoutingFields routing_fields;

    lite_fabric::NocSendType get_noc_send_type() volatile const { return this->noc_send_type; }
    uint16_t get_payload_size_bytes() volatile const { return this->payload_size_bytes; }

    // Set the packet to be a NoC write to the target chip
    inline LiteFabricHeader& to_noc_unicast_write(
        const NocUnicastCommandHeader& noc_unicast_command_header, uint16_t payload_size_bytes) {
        this->noc_send_type = lite_fabric::NocSendType::NOC_UNICAST_WRITE;
        this->payload_size_bytes = payload_size_bytes;
        this->command_fields.noc_unicast = noc_unicast_command_header;
        return *static_cast<LiteFabricHeader*>(this);
    }

    // Set the packet to be a NoC read at the target chip
    inline LiteFabricHeader& to_noc_read(
        const NocReadCommandHeader& noc_read_command_header, uint16_t payload_size_bytes) {
        this->noc_send_type = lite_fabric::NocSendType::NOC_READ;
        this->payload_size_bytes = payload_size_bytes;
        this->command_fields.noc_read = noc_read_command_header;
        return *static_cast<LiteFabricHeader*>(this);
    }

    // Set the number of hops along the line for this packet to the target chip
    inline LiteFabricHeader& to_chip_unicast(uint8_t distance_in_hops) {
        // LowLatencyPacketHeader::calculate_chip_unicast_routing_fields_value
        uint32_t value =
            (LiteFabricRoutingFields::FWD_ONLY_FIELD &
             ((1 << (distance_in_hops - 1) * LiteFabricRoutingFields::FIELD_WIDTH) - 1)) |
            (LiteFabricRoutingFields::WRITE_ONLY << (distance_in_hops - 1) * LiteFabricRoutingFields::FIELD_WIDTH);
        this->routing_fields.value = value;
        return *static_cast<LiteFabricHeader*>(this);
    }

    size_t get_payload_size_excluding_header() volatile const { return this->payload_size_bytes; }

    inline size_t get_payload_size_including_header() volatile const {
        return get_payload_size_excluding_header() + sizeof(LiteFabricHeader);
    }
};

static_assert(sizeof(LiteFabricHeader) == 32);

}  // namespace lite_fabric
