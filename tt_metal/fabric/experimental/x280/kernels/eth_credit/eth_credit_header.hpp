// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * kernels/eth_credit/eth_credit_header.hpp
 *
 * The eth_credit use of the shadow fabric header: a per-p sequence number in
 * bytes 8..11, producer p in src_ch_id, and the payload length.
 */
#ifndef X280_ETH_CREDIT_HEADER_HPP
#define X280_ETH_CREDIT_HEADER_HPP

#include <stddef.h>
#include <stdint.h>

#include "../shared/fabric_shadow_header.hpp"
#include "eth_credit_abi.h"

/* NocCommandFields is 40 bytes and a unicast write's NocUnicastCommandHeader
 * is its first 8 (a uint64 noc_address), so 8..11 is the unused tail. */
#define EC_HEADER_SEQ_OFFSET 8u
#define EC_HEADER_UNICAST_COMMAND_BYTES 8u
#define EC_HEADER_COMMAND_FIELDS_BYTES 40u

static_assert(
    EC_HEADER_SEQ_OFFSET >= EC_HEADER_UNICAST_COMMAND_BYTES &&
        EC_HEADER_SEQ_OFFSET + sizeof(uint32_t) <= EC_HEADER_COMMAND_FIELDS_BYTES,
    "seq must sit in NocCommandFields past the unicast command header");
static_assert(
    EC_HEADER_COMMAND_FIELDS_BYTES == FS_PAYLOAD_SIZE_BYTES_OFFSET,
    "NocCommandFields ends where payload_size_bytes begins");
static_assert(
    sizeof(fs_hybrid_mesh_packet_header_t) == EC_HEADER_BYTES, "the slot header is the 96-byte shadow header");

/**
 * Writes the sequence number into bytes 8..11 of the command fields.
 *
 * These bytes are overloaded for this benchmark: production does not read them
 * for a unicast write, whose command is the 8-byte NocUnicastCommandHeader, so
 * a production router forwarding this header ignores them. A different command
 * type would give them meaning.
 */
inline void ec_header_set_seq(volatile fs_hybrid_mesh_packet_header_t* header, uint32_t seq) {
    volatile uint8_t* bytes = reinterpret_cast<volatile uint8_t*>(header);
    bytes[EC_HEADER_SEQ_OFFSET + 0u] = static_cast<uint8_t>(seq);
    bytes[EC_HEADER_SEQ_OFFSET + 1u] = static_cast<uint8_t>(seq >> 8);
    bytes[EC_HEADER_SEQ_OFFSET + 2u] = static_cast<uint8_t>(seq >> 16);
    bytes[EC_HEADER_SEQ_OFFSET + 3u] = static_cast<uint8_t>(seq >> 24);
}

/** Reads the sequence number ec_header_set_seq() wrote. */
inline uint32_t ec_header_seq(const volatile fs_hybrid_mesh_packet_header_t* header) {
    const volatile uint8_t* bytes = reinterpret_cast<const volatile uint8_t*>(header);
    return static_cast<uint32_t>(bytes[EC_HEADER_SEQ_OFFSET + 0u]) |
           (static_cast<uint32_t>(bytes[EC_HEADER_SEQ_OFFSET + 1u]) << 8) |
           (static_cast<uint32_t>(bytes[EC_HEADER_SEQ_OFFSET + 2u]) << 16) |
           (static_cast<uint32_t>(bytes[EC_HEADER_SEQ_OFFSET + 3u]) << 24);
}

/** Writes what a receiver checks of packet `seq` from forwarder `p`: its
 * sequence number and channel. */
inline void ec_header_mark(volatile fs_hybrid_mesh_packet_header_t* header, uint32_t p, uint32_t seq) {
    ec_header_set_seq(header, seq);
    fs_set_src_ch_id(header, static_cast<uint8_t>(p));
}

/**
 * Seeds a complete header for forwarder `p`: fs_seed_unicast_header() rewrites
 * all 96 bytes, so src_ch_id, payload_size_bytes and seq are written after it.
 * `payload_bytes` must be at most EC_MAX_PAYLOAD_BYTES.
 */
inline void ec_header_seed(
    volatile fs_hybrid_mesh_packet_header_t* header,
    uint16_t dst_chip,
    uint16_t dst_mesh,
    uint32_t p,
    uint32_t payload_bytes,
    uint32_t seq) {
    fs_seed_unicast_header(header, dst_chip, dst_mesh);
    fs_set_src_ch_id(header, static_cast<uint8_t>(p));
    fs_set_payload_size_bytes(header, static_cast<uint16_t>(payload_bytes));
    ec_header_set_seq(header, seq);
}

#endif /* X280_ETH_CREDIT_HEADER_HPP */
