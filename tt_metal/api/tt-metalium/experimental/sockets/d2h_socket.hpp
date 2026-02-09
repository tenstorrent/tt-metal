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
 * @brief A socket for streaming data from a device core to the host.
 *
 * D2HSocket provides a FIFO-based interface for transferring data from a specific
 * core on a Tenstorrent device to the host CPU. The device kernel writes data to
 * the socket, and the host reads it via this interface.
 *
 * The socket uses pinned host memory that is accessible by the device via PCIe NOC
 * writes. The device kernel writes data to the pinned buffer and updates `bytes_sent`
 * to indicate available data. The host reads data and updates `bytes_acked` to signal
 * that buffer space has been freed.
 *
 * Requirements:
 * - vIOMMU must be enabled on the system for pinned memory to be accessible by the device
 * - Page size must be PCIe-aligned
 *
 * Flow Control:
 * - Device kernel calls `socket_reserve_pages()` to wait for buffer space
 * - Device kernel writes data and calls `socket_push_pages()` + `socket_notify_receiver()`
 * - Host calls `read()` which waits for data, copies it, and acknowledges consumption
 *

 *
 * Usage:
 * @code
 *   // Host side
 *   auto socket = D2HSocket(mesh_device, sender_core, fifo_size);
 *   socket.set_page_size(page_size);
 *   socket.read(data_ptr, num_pages);
 *   socket.barrier();  // Wait for all data to be consumed
 *
 *   // Device kernel side
 *   SocketSenderInterface sender_socket = create_sender_socket_interface(config_addr);
 *   set_sender_socket_page_size(sender_socket, page_size);
 *   socket_reserve_pages(sender_socket, 1);
 *   // Write data to downstream buffer...
 *   socket_push_pages(sender_socket, 1);
 *   socket_notify_receiver(sender_socket);
 * @endcode
 */
class D2HSocket {
public:
    /**
     * @brief Constructs a D2HSocket for streaming data from a device core to host.
     *
     * Allocates pinned host memory for the data FIFO and bytes_sent signaling.
     * Creates a configuration buffer on the device that the kernel uses to access
     * socket metadata and downstream (host) buffer addresses.
     *
     * @param mesh_device The mesh device containing the sender core.
     * @param sender_core The source core coordinate (device + core) that sends data.
     * @param fifo_size Size of the circular FIFO buffer in bytes. Must be PCIe-aligned.
     *
     * @throws TT_FATAL if pinned memory allocation fails or addresses are invalid.
     */
    D2HSocket(const std::shared_ptr<MeshDevice>& mesh_device, const MeshCoreCoord& sender_core, uint32_t fifo_size);

    /**
     * @brief Destroys the D2HSocket.
     *
     * Releases pinned memory mappings before freeing the underlying host buffers.
     * This ensures the DMA mappings are properly cleaned up to avoid "File exists"
     * errors when re-pinning memory at the same virtual address.
     */
    ~D2HSocket() noexcept;

    /**
     * @brief Returns the currently configured page size.
     *
     * @return The page size in bytes, or 0 if not yet set.
     */
    uint32_t get_page_size() const { return page_size_; }

    /**
     * @brief Returns the L1 address of the socket configuration buffer on the device.
     *
     * This address should be passed to the device kernel (typically as a compile-time
     * argument) so it can call `create_sender_socket_interface()` to access socket metadata.
     *
     * @return The L1 address of the configuration buffer.
     */
    uint32_t get_config_buffer_address() const { return config_buffer_->address(); }

    /**
     * @brief Sets the page size for subsequent read operations.
     *
     * The page size determines the granularity of data transfers. Must be PCIe-aligned
     * and less than or equal to the FIFO size. The read pointer is aligned to the new
     * page size boundary.
     *
     * If alignment causes the read pointer to wrap, this function waits for the device
     * to send enough data to cover the alignment adjustment before returning.
     *
     * @param page_size Page size in bytes. Must be PCIe-aligned.
     *
     * @throws TT_FATAL if page_size is not PCIe-aligned or exceeds FIFO size.
     */
    void set_page_size(uint32_t page_size);

    /**
     * @brief Reads data pages from the socket FIFO.
     *
     * Blocks until the requested number of pages are available in the FIFO.
     * Copies data from the pinned host buffer to the provided destination buffer.
     * Optionally notifies the device that buffer space has been freed.
     *
     * @param data Pointer to the destination buffer. Must have space for
     *             `num_pages * page_size` bytes.
     * @param num_pages Number of pages to read.
     * @param notify_sender If true (default), updates `bytes_acked` on the device
     *                      to signal that buffer space is available. Set to false
     *                      if batching multiple reads before acknowledging.
     *
     * @throws TT_FATAL if page_size has not been set or num_pages exceeds FIFO capacity.
     */
    void read(void* data, uint32_t num_pages, bool notify_sender = true);

    /**
     * @brief Blocks until all sent data has been acknowledged.
     *
     * Waits until `bytes_acked` equals `bytes_sent`, indicating the host has
     * consumed all data sent by the device.
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
    PinnedBufferInfo init_host_buffer(
        const std::shared_ptr<MeshDevice>& mesh_device,
        const MeshCoordinateRangeSet& device_range,
        uint32_t pcie_alignment);
    void init_config_buffer(const std::shared_ptr<MeshDevice>& mesh_device);
    void write_socket_metadata(
        const std::shared_ptr<MeshDevice>& mesh_device,
        const PinnedBufferInfo& data_info,
        const PinnedBufferInfo& bytes_sent_info);
    void init_sender_tlb(const std::shared_ptr<MeshDevice>& mesh_device);

    void wait_for_bytes(uint32_t num_bytes);
    void pop_bytes(uint32_t num_bytes);
    void notify_sender();

    std::shared_ptr<MeshBuffer> config_buffer_ = nullptr;
    MeshCoreCoord sender_core_;
    uint32_t fifo_size_ = 0;
    uint32_t page_size_ = 0;
    uint32_t bytes_acked_ = 0;
    uint32_t bytes_sent_ = 0;
    uint32_t read_ptr_ = 0;
    uint32_t fifo_curr_size_ = 0;
    tt::umd::TlbWindow* sender_core_tlb_ = nullptr;
    std::shared_ptr<tt::tt_metal::experimental::PinnedMemory> pinned_memory_ = nullptr;
    std::shared_ptr<uint32_t[]> host_buffer_ = nullptr;
    uint32_t* bytes_sent_ptr_ = nullptr;
    std::function<void(void*, uint32_t, uint64_t)> pcie_writer_ = nullptr;
};

}  // namespace tt::tt_metal::distributed
