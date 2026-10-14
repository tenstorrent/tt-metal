// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * kernels/packet_injection/packet_injection_slot.hpp
 *
 * 128-byte aligned benchmark packet slot containing a 96-byte Fabric header,
 * 8-byte timestamp, and 24-byte padding.
 */
#ifndef X280_PACKET_INJECTION_SLOT_HPP
#define X280_PACKET_INJECTION_SLOT_HPP

#include <stddef.h>
#include <stdint.h>

#include "../shared/fabric_shadow_header.hpp"
#include "packet_injection_abi.h"

struct alignas(64) pi_slot_t {
    fs_hybrid_mesh_packet_header_t header;
    uint64_t start;
    uint8_t padding[24];
};

static_assert(sizeof(pi_slot_t) == PI_SLOT_BYTES, "pi_slot_t must be one ring slot");
static_assert(alignof(pi_slot_t) == 64u, "pi_slot_t must be 64-aligned");
static_assert(offsetof(pi_slot_t, header) == 0u, "header at offset 0");
static_assert(sizeof(fs_hybrid_mesh_packet_header_t) == 96u, "Fabric header must be 96 bytes");
static_assert(offsetof(pi_slot_t, start) == 96u, "start timestamp at offset 96");
static_assert(offsetof(pi_slot_t, padding) == 104u, "padding at offset 104");

#endif /* X280_PACKET_INJECTION_SLOT_HPP */
