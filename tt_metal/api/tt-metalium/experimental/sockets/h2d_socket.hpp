// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include <tt-metalium/experimental/pinned_memory.hpp>
#include <memory>
#include <utility>

namespace tt::umd {
class TlbWindow;
}

namespace tt::tt_metal::distributed {

class NamedShm;
class UmdDeviceAccess;

/**
 * @brief Specifies the data transfer mode for Host-to-Device communication.
 */
enum class H2DMode : uint8_t {
    HOST_PUSH,    ///< Host pushes data to device via UMD TLB writes.
    DEVICE_PULL,  ///< Device pulls data from pinned host memory via PCIe NOC reads.
};

/**
 * @brief A socket for streaming data from host to a device core.
 *
 * H2DSocket provides a FIFO-based interface for transferring data from the host CPU
 * to a specific core on a Tenstorrent device. It supports two transfer modes:
 *
 * - **HOST_PUSH**: The host writes data directly to device L1 memory via TLB-mapped
 *   PCIe writes.
 *
 * - **DEVICE_PULL**: The host writes data to pinned host memory, and the device core
 *   pulls data via PCIe NOC reads.
 * Both Modes require vIOMMU to be enabled on the system, based on the current implementation.
 *
 * The socket uses a circular FIFO buffer with flow control. The host tracks `bytes_sent`
 * and the device kernel updates `bytes_acked` to indicate consumed data. The host blocks
 * on write() if the FIFO is full until the device acknowledges data.
 *
 * Usage:
 * @code
 *   // Owner process creates the socket and exports a descriptor
 *   auto socket = H2DSocket(mesh_device, recv_core, BufferType::L1, fifo_size, H2DMode::HOST_PUSH);
 *   auto desc_path = socket.export_descriptor("my_socket");
 *
 *   // Remote process connects via the descriptor
 *   auto socket = H2DSocket::connect(mesh_device, desc_path);
 *   socket->set_page_size(page_size);
 *   socket->write(data_ptr, num_pages);
 *   socket->barrier();
 * @endcode
 */
class H2DSocket {
public:
    /**
     * @brief Constructs an H2DSocket for streaming data to a device core (owner).
     *
     * Allocates named shared memory for host-side buffers, pins it for device DMA,
     * and sets up device-side config and data buffers. The socket can be exported
     * via export_descriptor() for cross-process attachment.
     *
     * @param mesh_device The mesh device containing the target core.
     * @param recv_core The target core coordinate (device + core) to receive data.
     * @param buffer_type Memory type for the device-side FIFO buffer (L1 or DRAM).
     * @param fifo_size Size of the circular FIFO buffer in bytes. Must be PCIe-aligned.
     * @param h2d_mode Transfer mode: HOST_PUSH or DEVICE_PULL.
     */
    H2DSocket(
        const std::shared_ptr<MeshDevice>& mesh_device,
        const MeshCoreCoord& recv_core,
        BufferType buffer_type,
        uint32_t fifo_size,
        H2DMode h2d_mode);

    /**
     * @brief Connects to an existing H2DSocket from another process via a descriptor file.
     *
     * Opens the named shared memory created by the owner process and sets up TLB access
     * to the same device core. Does not allocate device-side buffers or pin memory.
     * The returned socket is fully functional for write() and barrier() operations.
     *
     * @param mesh_device The mesh device (must contain the same physical device as the owner).
     * @param descriptor_path Path to the JSON descriptor file exported by the owner.
     * @return A connected H2DSocket ready for data transfer.
     */
    static std::unique_ptr<H2DSocket> connect(
        const std::string& socket_id, std::optional<uint32_t> timeout_ms = std::nullopt);

    /**
     * @brief Exports a descriptor file for cross-process socket attachment.
     *
     * Writes a JSON file to /dev/shm/ containing all metadata needed for a remote
     * process to connect: shared memory name, buffer layout, device addresses, and
     * core coordinates.
     *
     * @param socket_id A user-provided identifier used in the descriptor filename.
     * @return The full path to the written descriptor file.
     */
    std::string export_descriptor(const std::string& socket_id);

    /**
     * @brief Destroys the H2DSocket.
     *
     * Owner: waits for device acknowledgement, unpins memory, unlinks shared memory.
     * Connector: waits for device acknowledgement, unmaps shared memory.
     */
    ~H2DSocket() noexcept;

    uint32_t get_page_size() const { return page_size_; }

    uint32_t get_config_buffer_address() const { return config_buffer_address_; }

    void set_page_size(uint32_t page_size);

    void write(void* data, uint32_t num_pages);

    void barrier(std::optional<uint32_t> timeout_ms = std::nullopt);

    std::vector<MeshCoreCoord> get_active_cores() const;

    MeshDevice* get_mesh_device() const;

    H2DMode get_h2d_mode() const;

private:
    H2DSocket() = default;
    H2DSocket(const H2DSocket&) = delete;
    H2DSocket& operator=(const H2DSocket&) = delete;

    struct PinnedBufferInfo {
        uint32_t pcie_xy_enc = 0;
        uint32_t addr_lo = 0;
        uint32_t addr_hi = 0;
    };

    PinnedBufferInfo init_bytes_acked_buffer(
        const std::shared_ptr<MeshDevice>& mesh_device,
        const MeshCoordinateRangeSet& device_range,
        uint32_t pcie_alignment,
        const std::string& shm_name);

    PinnedBufferInfo init_host_data_buffer(
        const std::shared_ptr<MeshDevice>& mesh_device,
        const MeshCoordinateRangeSet& device_range,
        uint32_t pcie_alignment,
        const std::string& shm_name);

    void init_config_buffer(const std::shared_ptr<MeshDevice>& mesh_device);
    void init_data_buffer(const std::shared_ptr<MeshDevice>& mesh_device, uint32_t pcie_alignment);
    void write_socket_metadata(
        const std::shared_ptr<MeshDevice>& mesh_device,
        const PinnedBufferInfo& bytes_acked_info,
        const PinnedBufferInfo& data_info);
    void init_receiver_tlb(
        const std::shared_ptr<MeshDevice>& mesh_device, std::optional<uint32_t> device_id = std::nullopt);

    void reserve_bytes(uint32_t num_bytes);
    void push_bytes(uint32_t num_bytes);
    void notify_receiver();

    std::shared_ptr<MeshBuffer> config_buffer_ = nullptr;
    std::shared_ptr<MeshBuffer> data_buffer_ = nullptr;
    MeshCoreCoord recv_core_;
    BufferType buffer_type_ = BufferType::L1;
    uint32_t fifo_size_ = 0;
    uint32_t page_size_ = 0;
    uint32_t bytes_sent_ = 0;
    uint32_t bytes_acked_ = 0;
    uint32_t write_ptr_ = 0;
    uint32_t fifo_curr_size_ = 0;
    uint32_t aligned_data_buf_start_ = 0;
    uint32_t config_buffer_address_ = 0;
    uint32_t pcie_alignment_ = 0;
    tt::umd::TlbWindow* receiver_core_tlb_ = nullptr;
    std::shared_ptr<tt::tt_metal::experimental::PinnedMemory> pinned_memory_ = nullptr;
    std::shared_ptr<uint32_t[]> host_buffer_ = nullptr;
    uint32_t* bytes_acked_ptr_ = nullptr;
    std::function<void(void*, uint32_t, uint64_t)> pcie_writer = nullptr;
    H2DMode h2d_mode_ = H2DMode::HOST_PUSH;
    std::unique_ptr<NamedShm> shm_;
    std::unique_ptr<UmdDeviceAccess> umd_access_;
    MeshDevice* mesh_device_ = nullptr;
    bool is_owner_ = true;
    std::string descriptor_path_;
    bool exported_ = false;
};

}  // namespace tt::tt_metal::distributed
