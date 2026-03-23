// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <fmt/format.h>
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>

namespace tt::tt_metal::distributed {

class MeshDevice;
class NamedShm;

/**
 * @brief Serializable descriptor for cross-process H2D/D2H socket attachment.
 *
 * The owner process creates an H2D or D2H socket and exports a flatbuffer
 * descriptor containing everything a remote process needs to attach: named
 * shared memory identifiers, buffer layout, device-side L1 addresses, and
 * pre-resolved core coordinates for PCIe writes.
 *
 * The connecting process reads this descriptor, opens the named shared memory,
 * and reconstructs the host-side socket state without re-allocating device
 * resources or initializing MetalContext.
 */
struct HDSocketDescriptor {
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

    // --- Pre-resolved transport info (connector uses these to bypass MetalContext) ---
    uint32_t virtual_core_x = 0;             // Virtual (translated) core coordinate X
    uint32_t virtual_core_y = 0;             // Virtual (translated) core coordinate Y
    uint32_t pcie_alignment = 0;             // PCIe page alignment in bytes (e.g. 64 for Blackhole)
    uint32_t bytes_acked_device_offset = 0;  // D2H: L1 offset of bytes_acked within config buffer

    /**
     * @brief Populate common fields from the owner socket's state.
     *
     * Sets shm_name, shm_size, fifo_size, config_buffer_address, and resolves
     * device_id, core coordinates (logical + virtual), and pcie_alignment from
     * the mesh device. Called by the owner during export_descriptor().
     */
    void populate_from_owner(
        const std::string& type,
        const NamedShm& shm,
        uint32_t fifo_size,
        uint32_t config_buffer_address,
        MeshDevice* mesh_device,
        const MeshCoreCoord& core);

    /**
     * @brief Serialize this descriptor to a flatbuffer file.
     * @param path File path to write (e.g. "/dev/shm/tt_socket_<id>.bin").
     */
    void write_to_file(const std::string& path) const;

    /**
     * @brief Deserialize a descriptor from a flatbuffer file.
     * @param path File path to read.
     * @return Populated HDSocketDescriptor.
     */
    static HDSocketDescriptor read_from_file(const std::string& path);

    /**
     * @brief Wait for a descriptor file to appear and read it.
     * @param descriptor_path Full path to the descriptor file.
     * @param expected_type Expected socket_type ("h2d" or "d2h").
     * @param timeout_ms Max wait time in milliseconds (default 10000).
     * @return Populated HDSocketDescriptor.
     */
    static HDSocketDescriptor wait_and_read(
        const std::string& descriptor_path, const std::string& expected_type, uint32_t timeout_ms = 10000);
};

inline std::string descriptor_path_for_socket(const std::string& type, const std::string& socket_id) {
    return fmt::format("/dev/shm/tt_{}_{}.bin", type, socket_id);
}

}  // namespace tt::tt_metal::distributed
