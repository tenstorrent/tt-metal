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
     * @param l1_data_buffer_size If non-zero, allocates an L1 staging buffer on the sender core
     *        and writes its address into the socket config so the device kernel can use it.
     *        The address is retrievable via get_l1_data_buffer_address(). Default: 0 (disabled).
     */
    D2HSocket(
        const std::shared_ptr<MeshDevice>& mesh_device,
        const MeshCoreCoord& sender_core,
        uint32_t fifo_size,
        uint32_t l1_data_buffer_size = 0);

    ~D2HSocket() noexcept;

    uint32_t get_page_size() const { return page_size_; }

    uint32_t get_config_buffer_address() const { return config_buffer_->address(); }

    uint32_t get_l1_data_buffer_address() const { return l1_data_buffer_address_; }
    uint32_t get_l1_data_buffer_size() const { return l1_data_buffer_size_; }

    void set_page_size(uint32_t page_size);

    void read(void* data, uint32_t num_pages, bool notify_sender = true);

    void barrier(std::optional<uint32_t> timeout_ms = std::nullopt);

    uint32_t pages_available();

    std::vector<MeshCoreCoord> get_active_cores() const;

    MeshDevice* get_mesh_device() const;

private:
    struct PinnedBufferInfo {
        uint32_t pcie_xy_enc = 0;
        uint32_t addr_lo = 0;
        uint32_t addr_hi = 0;
    };

    PinnedBufferInfo init_host_buffer(
        const std::shared_ptr<MeshDevice>& mesh_device,
        const MeshCoordinateRangeSet& device_range,
        uint32_t pcie_alignment);
    void init_config_buffer(const std::shared_ptr<MeshDevice>& mesh_device);
    void init_l1_data_buffer(const std::shared_ptr<MeshDevice>& mesh_device, uint32_t requested_size);
    void write_socket_metadata(
        const std::shared_ptr<MeshDevice>& mesh_device,
        const PinnedBufferInfo& data_info,
        const PinnedBufferInfo& bytes_sent_info);
    void init_sender_tlb(const std::shared_ptr<MeshDevice>& mesh_device);

    void wait_for_bytes(uint32_t num_bytes);
    void pop_bytes(uint32_t num_bytes);
    void notify_sender();

    std::shared_ptr<MeshBuffer> config_buffer_ = nullptr;
    std::shared_ptr<MeshBuffer> l1_data_buffer_ = nullptr;
    MeshCoreCoord sender_core_;
    uint32_t fifo_size_ = 0;
    uint32_t page_size_ = 0;
    uint32_t bytes_acked_ = 0;
    uint32_t bytes_sent_ = 0;
    uint32_t read_ptr_ = 0;
    uint32_t fifo_curr_size_ = 0;
    uint32_t l1_data_buffer_address_ = 0;
    uint32_t l1_data_buffer_size_ = 0;
    tt::umd::TlbWindow* sender_core_tlb_ = nullptr;
    std::shared_ptr<tt::tt_metal::experimental::PinnedMemory> pinned_memory_ = nullptr;
    std::shared_ptr<uint32_t[]> host_buffer_ = nullptr;
    uint32_t* bytes_sent_ptr_ = nullptr;
    std::function<void(void*, uint32_t, uint64_t)> pcie_writer_ = nullptr;
};

}  // namespace tt::tt_metal::distributed
