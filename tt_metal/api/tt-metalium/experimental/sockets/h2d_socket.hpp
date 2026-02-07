// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include <tt-metalium/experimental/pinned_memory.hpp>
#include <utility>

namespace tt::umd {
class TlbWindow;
}

namespace tt::tt_metal::distributed {

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
 *   auto socket = H2DSocket(mesh_device, recv_core, BufferType::L1, fifo_size, H2DMode::HOST_PUSH);
 *   socket.set_page_size(page_size);
 *   socket.write(data_ptr, num_pages);
 *   socket.barrier();  // Wait for device to consume all data
 * @endcode
 */
class H2DSocket {
public:
    /**
     * @brief Constructs an H2DSocket for streaming data to a device core.
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
     * @brief Destroys the H2DSocket.
     *
     * Frees the pinned memory allocated for the socket.
     * Also issues a barrier to wait for the device to acknowledge all data over the socket.
     */
    ~H2DSocket() noexcept;

    /**
     * @brief Returns the currently configured page size.
     */
    uint32_t get_page_size() const { return page_size_; }

    /**
     * @brief Returns the L1 address of the socket configuration buffer on the device.
     *
     * This address is passed to the device kernel to access socket metadata.
     */
    uint32_t get_config_buffer_address() const { return config_buffer_->address(); }

    /**
     * @brief Sets the page size for subsequent write operations.
     *
     * The page size determines the granularity of data transfers. Must be PCIe-aligned
     * and less than or equal to the FIFO size. The write pointer is aligned to the new
     * page size boundary.
     *
     * @param page_size Page size in bytes. Must be PCIe-aligned.
     */
    void set_page_size(uint32_t page_size);

    /**
     * @brief Writes data pages to the socket FIFO.
     *
     * Blocks if the FIFO does not have enough space, waiting for the device to
     * acknowledge previously written data. After writing, notifies the device
     * that new data is available.
     *
     * @param data Pointer to the source data buffer.
     * @param num_pages Number of pages to write.
     */
    void write(void* data, uint32_t num_pages);

    /**
     * @brief Blocks until the device has acknowledged all written data.
     *
     * Waits until `bytes_acked` equals `bytes_sent`, indicating the device has
     * consumed all data in the FIFO.
     *
     * @param timeout_ms Optional timeout in milliseconds. If specified, the function will throw an exception if the
     * barrier is not met within the timeout.
     */
    void barrier(std::optional<uint32_t> timeout_ms = std::nullopt);

private:
    // Helper struct for pinned buffer NOC address info
    struct PinnedBufferInfo {
        uint32_t pcie_xy_enc = 0;
        uint32_t addr_lo = 0;
        uint32_t addr_hi = 0;
    };

    // Initialization helpers
    PinnedBufferInfo init_bytes_acked_buffer(
        const std::shared_ptr<MeshDevice>& mesh_device,
        const MeshCoordinateRangeSet& device_range,
        uint32_t pcie_alignment);

    PinnedBufferInfo init_host_data_buffer(
        const std::shared_ptr<MeshDevice>& mesh_device,
        const MeshCoordinateRangeSet& device_range,
        uint32_t pcie_alignment);

    void init_config_buffer(const std::shared_ptr<MeshDevice>& mesh_device);
    void init_data_buffer(const std::shared_ptr<MeshDevice>& mesh_device, uint32_t pcie_alignment);
    void write_socket_metadata(
        const std::shared_ptr<MeshDevice>& mesh_device,
        const PinnedBufferInfo& bytes_acked_info,
        const PinnedBufferInfo& data_info);
    void init_receiver_tlb(const std::shared_ptr<MeshDevice>& mesh_device);

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
    tt::umd::TlbWindow* receiver_core_tlb_ = nullptr;
    std::shared_ptr<tt::tt_metal::experimental::PinnedMemory> pinned_memory_ = nullptr;
    std::shared_ptr<uint32_t[]> host_buffer_ = nullptr;
    uint32_t* bytes_acked_ptr_ = nullptr;
    std::function<void(void*, uint32_t, uint64_t)> pcie_writer = nullptr;
    H2DMode h2d_mode_ = H2DMode::HOST_PUSH;
};

}  // namespace tt::tt_metal::distributed
