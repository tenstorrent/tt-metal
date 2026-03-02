// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>

namespace tt::tt_metal::distributed {

/**
 * @brief Serializable descriptor for cross-process H2D/D2H socket attachment.
 *
 * The owner process creates a socket and exports a descriptor (as JSON) containing
 * everything a remote process needs to attach: named shared memory identifiers,
 * buffer layout, device-side L1 addresses, and core coordinates.
 *
 * The connecting process reads this descriptor, opens the named shared memory,
 * and reconstructs the host-side socket state without re-allocating device resources.
 */
struct SocketDescriptor {
    // --- Socket identity ---
    std::string socket_type;  // "h2d" or "d2h"

    // --- Named shared memory ---
    std::string shm_name;   // POSIX shm name (e.g. "/tt_h2d_<id>")
    uint64_t shm_size = 0;  // Total mmap size in bytes

    // --- Buffer layout (offsets within the shm region) ---
    uint32_t data_offset = 0;         // Offset to FIFO data region (always 0 for current layouts)
    uint32_t bytes_acked_offset = 0;  // H2D: offset of bytes_acked (HOST_PUSH: 0, DEVICE_PULL: fifo_size)
    uint32_t bytes_sent_offset = 0;   // D2H: offset of bytes_sent (fifo_size)

    // --- Socket parameters ---
    uint32_t fifo_size = 0;
    uint32_t h2d_mode = 0;  // H2DMode as uint8_t (H2D only; 0=HOST_PUSH, 1=DEVICE_PULL)

    // --- Device-side L1 addresses (already allocated by owner) ---
    uint32_t config_buffer_address = 0;   // L1 address of the socket config buffer
    uint32_t aligned_data_buf_start = 0;  // H2D only: L1 address of the aligned data buffer

    // --- Core identification (for TLB setup by the connector) ---
    uint32_t device_id = 0;  // Physical chip ID
    uint32_t core_x = 0;     // Logical core coordinate X
    uint32_t core_y = 0;     // Logical core coordinate Y

    /**
     * @brief Serialize this descriptor to a JSON file.
     * @param path File path to write (e.g. "/dev/shm/tt_socket_<id>.json").
     */
    void write_to_file(const std::string& path) const;

    /**
     * @brief Deserialize a descriptor from a JSON file.
     * @param path File path to read.
     * @return Populated SocketDescriptor.
     */
    static SocketDescriptor read_from_file(const std::string& path);
};

}  // namespace tt::tt_metal::distributed
