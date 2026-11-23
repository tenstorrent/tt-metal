// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace erisc::datamover::handshake {

// The value each end of a link writes into its own scratch[0] and sends into the other end's local_value. An end's
// handshake with its peer has finished once its local_value holds it.
static constexpr uint32_t MAGIC_HANDSHAKE_VALUE = 0xAA;

// Data-Structure used for EDM to EDM Handshaking.
// The scratch buffer is sent to the peer and overwrites the first 16 bytes of their struct,
// populating neighbor_mesh_id and neighbor_device_id with the sender's identity.
struct handshake_info_t {
    uint32_t local_value;        // Bytes 0-3: Updated by remote with MAGIC_HANDSHAKE_VALUE
    uint16_t neighbor_mesh_id;   // Bytes 4-5: Peer's mesh_id (populated via scratch[1])
    uint8_t neighbor_device_id;  // Byte 6: Peer's device_id (populated via scratch[1])
    uint8_t padding0;            // Byte 7: Explicit padding for alignment
    uint32_t padding[2];         // Bytes 8-15: Ensures 16B alignment for scratch register
    uint32_t scratch[4];         // Bytes 16-31: Information that is sent to the peer
};
static_assert(sizeof(handshake_info_t) == 32, "handshake_info_t size is not 32 bytes");

}  // namespace erisc::datamover::handshake
