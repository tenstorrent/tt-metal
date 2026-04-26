// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
class PCIeCoreWriter;

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
     * @brief Identifies an L1 region on the sender core that the caller has already
     *        reserved for the socket's configuration buffer.
     *
     * Used by callers that own their sender core's L1 layout (e.g. the real-time
     * profiler, which carves its config out of dispatch L1 on the reserved
     * profiler tensix). The region must be at least
     * D2HSocket::required_config_buffer_size() bytes, L1-aligned, and live for
     * the lifetime of the socket.
     */
    struct ExternalConfigBuffer {
        uint32_t address;  // L1 address on the sender core
    };

    /**
     * @brief Minimum size in bytes that an ExternalConfigBuffer region must have.
     *
     * Equals sender_socket_md + bytes_acked + sender_downstream_encoding, each
     * rounded up to L1_ALIGNMENT. Callers carving their own L1 region for the
     * socket's configuration buffer should use this to size that region (or
     * static_assert their own constant against it).
     */
    static uint32_t required_config_buffer_size();

    /**
     * @brief Constructs a D2HSocket using a caller-provided config buffer address.
     *
     * Skips the user-space MeshBuffer allocation that the standard constructor
     * performs and writes the socket metadata directly to `external_config.address`
     * on the sender core. Intended for callers that need their sender-core L1 to
     * be off-allocator (e.g. cores not present in the L1 bank table).
     */
    D2HSocket(
        const std::shared_ptr<MeshDevice>& mesh_device,
        const MeshCoreCoord& sender_core,
        uint32_t fifo_size,
        ExternalConfigBuffer external_config);

    /**
     * @brief Connects to an existing D2HSocket from another process.
     *
     * Waits for the flatbuffer descriptor exported by the owner, opens the named
     * shared memory, and sets up PCIe write access to the device core via
     * PCIeCoreWriter (bypasses MetalContext). The returned socket is fully
     * functional for read() and barrier() operations.
     *
     * @param socket_id The identifier used when the owner called export_descriptor().
     * @param timeout_ms Max time to wait for the descriptor file (default 10s).
     * @return A connected D2HSocket ready for data transfer.
     */
    static std::unique_ptr<D2HSocket> connect(
        const std::string& socket_id, std::optional<uint32_t> timeout_ms = std::nullopt);

    /**
     * @brief Exports a descriptor file for cross-process socket attachment.
     *
     * Writes a flatbuffer binary to /dev/shm/ containing all metadata needed for
     * a remote process to connect: shared memory name, buffer layout, device
     * addresses, pre-resolved core coordinates, and PCIe alignment.
     *
     * @param socket_id A user-provided identifier used in the descriptor filename.
     * @return The full path to the written descriptor file.
     */
    std::string export_descriptor(const std::string& socket_id);

    /**
     * @brief Destroys the D2HSocket.
     *
     * Releases pinned memory mappings before freeing the underlying host buffers.
     * This ensures the DMA mappings are properly cleaned up to avoid "File exists"
     * errors when re-pinning memory at the same virtual address.
     * Owner: waits for device acknowledgement, unpins memory, unlinks shared memory,
     * removes descriptor file.
     * Connector: unmaps shared memory via NamedShm destructor.
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
    uint32_t get_config_buffer_address() const { return config_buffer_address_; }

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
     * @brief Non-blocking check for available data.
     *
     * Returns true if at least one page of data is available in the FIFO
     * without blocking. Useful for poll-based readers that need to check
     * a shutdown flag between iterations.
     *
     * @return true if at least one page can be read immediately.
     *
     * @throws TT_FATAL if page_size has not been set.
     */
    bool has_data();

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

    /**
     * @brief Non-blocking query for the number of pages currently available in the FIFO.
     *
     * @return The number of complete pages that can be read immediately.
     *
     * @throws TT_FATAL if page_size has not been set.
     */
    uint32_t pages_available();

    /**
     * @brief Discards any currently-available pages WITHOUT reading the data
     *        region.  Rebases the host's bytes_acked counter to the current
     *        bytes_sent value (and notifies the device), which is the correct
     *        operation when the host wants to ignore any pending bytes — e.g.
     *        before initiating a sync handshake.
     *
     * Unlike a sequence of `read()` calls, this does NOT touch the data
     * region (which on Wormhole/Blackhole is mapped through PCIe and may
     * contain undefined values from a prior device run or stale shmem
     * counters).
     *
     * @return The number of pages that were discarded (0 if there was nothing
     *         pending).
     *
     * @throws TT_FATAL if page_size has not been set.
     */
    uint32_t discard_pending_pages();

    std::vector<MeshCoreCoord> get_active_cores() const;

    MeshDevice* get_mesh_device() const;

    D2HSocket(const D2HSocket&) = delete;
    D2HSocket& operator=(const D2HSocket&) = delete;

private:
    D2HSocket() = default;

    struct PinnedBufferInfo {
        uint32_t pcie_xy_enc = 0;
        uint32_t addr_lo = 0;
        uint32_t addr_hi = 0;
    };

    PinnedBufferInfo init_host_buffer(
        const std::shared_ptr<MeshDevice>& mesh_device,
        const MeshCoordinateRangeSet& device_range,
        uint32_t pcie_alignment,
        const std::string& shm_name);
    PinnedBufferInfo init_host_buffer_hugepage(const std::shared_ptr<MeshDevice>& mesh_device);
    void init_config_buffer(const std::shared_ptr<MeshDevice>& mesh_device);
    void write_socket_metadata(
        const std::shared_ptr<MeshDevice>& mesh_device,
        const PinnedBufferInfo& data_info,
        const PinnedBufferInfo& bytes_sent_info);
    void init_sender_tlb(
        const std::shared_ptr<MeshDevice>& mesh_device, std::optional<uint32_t> device_id = std::nullopt);

    void wait_for_bytes(uint32_t num_bytes);
    void pop_bytes(uint32_t num_bytes);
    void notify_sender();

    // Common host-side init shared by both constructors. Brings up pinned host
    // memory (or hugepage fallback), writes socket metadata to `config_buffer_address_`,
    // and configures the sender-side TLB. The caller must populate
    // `config_buffer_address_` (and own the L1 reservation behind it) before calling.
    void init_common(const std::shared_ptr<MeshDevice>& mesh_device);

    // Owned only by the standard ctor. The external-config ctor leaves this null
    // and points `config_buffer_address_` at caller-owned L1.
    std::shared_ptr<MeshBuffer> config_buffer_ = nullptr;
    MeshCoreCoord sender_core_;
    uint32_t fifo_size_ = 0;
    uint32_t page_size_ = 0;
    uint32_t bytes_acked_ = 0;
    uint32_t bytes_sent_ = 0;
    uint32_t read_ptr_ = 0;
    uint32_t fifo_curr_size_ = 0;
    uint32_t config_buffer_address_ = 0;
    uint32_t pcie_alignment_ = 0;
    uint32_t bytes_acked_device_offset_ = 0;
    tt::umd::TlbWindow* sender_core_tlb_ = nullptr;
    std::shared_ptr<tt::tt_metal::experimental::PinnedMemory> pinned_memory_ = nullptr;
    std::shared_ptr<uint32_t[]> host_buffer_ = nullptr;
    uint32_t* bytes_sent_ptr_ = nullptr;
    std::function<void(void*, uint32_t, uint64_t)> pcie_writer_ = nullptr;
    std::unique_ptr<NamedShm> shm_;
    std::unique_ptr<PCIeCoreWriter> pcie_writer_instance_;
    MeshDevice* mesh_device_ = nullptr;
    bool is_owner_ = true;
    std::string descriptor_path_;
    bool exported_ = false;

    bool using_hugepage_ = false;
    uint32_t* hugepage_data_host_ptr_ = nullptr;
    volatile uint32_t* hugepage_bytes_sent_host_ptr_ = nullptr;
};

}  // namespace tt::tt_metal::distributed
