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

enum class NocSendTypeEnum : uint8_t {
    NOC_UNICAST_WRITE = 0,
    NOC_READ = 1,
    NOC_SEND_TYPE_LAST = NOC_READ,
};

union NocSendType {
    uint8_t raw;

    struct {
        NocSendTypeEnum send_type : 7;  // bits 0-6
        uint8_t noc_index : 1;          // bit 7
    } fields;

    // Constructors
    explicit NocSendType() = default;
    NocSendType(uint8_t value) : raw(value) {}
    NocSendType(NocSendTypeEnum type, uint8_t noc_idx = 0) {
        fields.send_type = type;
        fields.noc_index = noc_idx & 0x1;
    }

    // Conversion operators
    operator uint8_t() const { return raw; }

    // Helper methods
    inline NocSendTypeEnum get_send_type() const volatile { return fields.send_type; }
    inline uint8_t get_noc_index() const volatile { return fields.noc_index; }

    inline void set_send_type(NocSendTypeEnum type) { fields.send_type = type; }
    inline void set_noc_index(uint8_t noc_idx) { fields.noc_index = noc_idx & 0x1; }

    // Comparison operators
    bool operator==(const NocSendType& other) const { return raw == other.raw; }
    bool operator!=(const NocSendType& other) const { return raw != other.raw; }
    bool operator==(NocSendTypeEnum type) const { return fields.send_type == type && fields.noc_index == 0; }
};

static_assert(sizeof(NocSendType) == 1, "NocSendType must be 1 byte");

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

    explicit LiteFabricHeader() = default;
    lite_fabric::NocSendType get_noc_send_type() volatile const { return lite_fabric::NocSendType(this->noc_send_type.raw); }
    uint16_t get_payload_size_bytes() volatile const { return this->payload_size_bytes; }
    uint8_t get_noc_index() volatile const { return this->noc_send_type.get_noc_index(); }
    lite_fabric::NocSendTypeEnum get_base_send_type() volatile const { return this->noc_send_type.get_send_type(); }

    // Set the packet to be a NoC write to the target chip
    inline LiteFabricHeader& to_noc_unicast_write(
        const NocUnicastCommandHeader& noc_unicast_command_header, uint16_t payload_size_bytes, uint8_t noc_index = 0) {
        this->noc_send_type = lite_fabric::NocSendType(lite_fabric::NocSendTypeEnum::NOC_UNICAST_WRITE, noc_index);
        this->payload_size_bytes = payload_size_bytes;
        this->command_fields.noc_unicast = noc_unicast_command_header;
        return *static_cast<LiteFabricHeader*>(this);
    }

    // Set the packet to be a NoC read at the target chip
    inline LiteFabricHeader& to_noc_read(
        const NocReadCommandHeader& noc_read_command_header, uint16_t payload_size_bytes, uint8_t noc_index = 0) {
        this->noc_send_type = lite_fabric::NocSendType(lite_fabric::NocSendTypeEnum::NOC_READ, noc_index);
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
