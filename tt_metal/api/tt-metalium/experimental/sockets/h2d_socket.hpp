// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>

namespace tt::tt_metal::experimental::detail {
struct H2DSocketTryWriteAccess;
struct H2DSocketDramRecvAccess;
}  // namespace tt::tt_metal::experimental::detail

namespace tt::tt_metal::distributed {

class H2DSocketImpl;
class MeshDevice;
struct HDSocketDescriptor;

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
 * - **HOST_PUSH**: The host writes data directly to device L1 memory via PCIe writes.
 * - **DEVICE_PULL**: The host writes data to pinned host memory, and the device core
 *   pulls data via PCIe NOC reads.
 *
 * Both modes require vIOMMU to be enabled on the system.
 *
 * The socket uses a circular FIFO buffer with flow control. The host tracks `bytes_sent`
 * and the device kernel updates `bytes_acked` to indicate consumed data. The host blocks
 * on write() if the FIFO is full until the device acknowledges data.
 *
 * Thread safety:
 * - The FIFO is single-producer/single-consumer: one host producer writes data and one
 *   device consumer acknowledges it.
 * - An H2DSocket instance is not internally synchronized. Calls on the same instance
 *   must not overlap across host threads unless the caller provides external synchronization.
 * - A descriptor attaches another handle to the same FIFO; it does not create an independent
 *   channel. At most one host process may actively write through the owner/connector handles.
 *
 * Supports cross-process usage: the owner process creates the socket and exports a
 * flatbuffer descriptor. A remote process connects via the descriptor using only UMD
 * (no MetalContext required).
 *
 * Usage:
 * @code
 *   // Owner process creates the socket and exports a descriptor
 *   auto socket = H2DSocket(mesh_device, recv_core, BufferType::L1, fifo_size, H2DMode::HOST_PUSH);
 *   socket.export_descriptor("my_socket");
 *
 *   // Remote process connects via socket ID (no MetalContext needed)
 *   auto remote = H2DSocket::connect("my_socket");
 *   remote->set_page_size(page_size);
 *   remote->write(data_ptr, num_pages);
 *   remote->barrier();
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
     * All ranks sharing the mesh must construct the socket to reserve device buffers together.
     * Shared meshes support worker-core endpoints only; claimed service cores are rejected on every rank.
     * Only the rank owning recv_core maps host memory and may write or export the socket.
     * Non-owning ranks may set the page size and query configuration; barrier() is a no-op.
     * Host I/O on a non-owning rank throws. Descriptor connectors retain host I/O access.
     *
     * If `recv_core` is a claimed service core, the device-side buffers are allocated
     * from that core's service-core L1 region instead of the worker-grid BankManager.
     *
     * @param mesh_device The mesh device containing the target core.
     * @param recv_core The target core coordinate (device + core) to receive data.
     * @param buffer_type Memory type for the device-side FIFO buffer (currently only L1).
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
     * @brief Constructs an H2DSocket targeting an L2CPU receiver.
     *
     * Behaves as the standard constructor, with an L2CPU tile as the receiver.
     * L2CPU LIM has no allocator in tt-metal, so the config buffer and data FIFO
     * addresses are caller-supplied rather than allocated here.
     *
     * @param mesh_device The mesh device containing the receiver L2CPU.
     * @param recv_l2cpu The receiving L2CPU tile. @c core_coord must be the TRANSLATED NOC
     *                   coord of an L2CPU tile on the target device.
     * @param fifo_size Size of the circular FIFO buffer in bytes. Must be PCIe-aligned.
     * @param config_buffer_address LIM address on the receiver L2CPU for the socket metadata.
     *                              Must be PCIe-aligned and within the L2CPU's IoWindow.
     * @param data_fifo_address LIM address for the data FIFO. In HOST_PUSH this is the ring
     *                          itself and must be PCIe-aligned, disjoint from the config buffer,
     *                          and fit with fifo_size inside the L2CPU's IoWindow. In
     *                          DEVICE_PULL the ring lives in pinned host memory and this is the
     *                          base the device computes ring offsets against.
     * @param h2d_mode Transfer mode: HOST_PUSH or DEVICE_PULL.
     */
    H2DSocket(
        MeshDevice& mesh_device,
        const MeshCoreCoord& recv_l2cpu,
        uint32_t fifo_size,
        uint32_t config_buffer_address,
        uint32_t data_fifo_address,
        H2DMode h2d_mode = H2DMode::HOST_PUSH);

    /**
     * @brief Connects to an existing H2DSocket from another process.
     *
     * Waits for the flatbuffer descriptor exported by the owner, opens the named
     * shared memory, and sets up PCIe write access to the device core via
     * PCIeCoreWriter (bypasses MetalContext). Does not allocate device-side buffers
     * or pin memory. The returned socket is fully functional for write() and
     * barrier() operations.
     *
     * @param socket_id The identifier used when the owner called export_descriptor().
     * @param timeout_ms Max time to wait for the descriptor file (default 10s).
     * @return A connected H2DSocket ready for data transfer.
     */
    static std::unique_ptr<H2DSocket> connect(
        const std::string& socket_id, std::optional<uint32_t> timeout_ms = std::nullopt);

    /**
     * @brief Connects to an H2DSocket from a pre-loaded descriptor.
     *
     * Like `connect(socket_id, ...)` but takes the descriptor directly rather than reading
     * it from `/dev/shm/`, for callers that aggregate descriptors in a higher-level container.
     *
     * @param desc A populated socket descriptor.
     * @return A connected H2DSocket ready for data transfer.
     */
    static std::unique_ptr<H2DSocket> connect_from_descriptor(const HDSocketDescriptor& desc);

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
     * @brief Populate a descriptor for this socket without writing to disk.
     *
     * Returns the same HDSocketDescriptor `export_descriptor` would write, but does no
     * file I/O. Only callable on the owner-side socket.
     */
    HDSocketDescriptor populate_descriptor() const;

    /**
     * @brief Destroys the H2DSocket.
     *
     * Owner: waits for device acknowledgement, unpins memory, unlinks shared memory.
     * Connector: waits for device acknowledgement, unmaps shared memory.
     */
    ~H2DSocket() noexcept;

    uint32_t get_page_size() const;

    uint32_t get_config_buffer_address() const;

    uint32_t get_fifo_size() const;

    bool has_space(std::optional<uint32_t> num_bytes_to_check);

    // Cumulative bytes pushed into the FIFO (wraps modulo 2^32). Snapshot this
    // after a write() call to get a watermark, then pass it to acked_past()
    // later to test whether the device has consumed up to that point.
    uint32_t get_bytes_sent() const;

    // Returns true iff the device has acked past `watermark`. Uses the
    // cached acknowledgement count on the fast path; mfences + refreshes from
    // the device-written counter only when the cache is insufficient.
    bool acked_past(uint32_t watermark);

    void set_page_size(uint32_t page_size);

    void write(void* data, uint32_t num_pages);

    void barrier(std::optional<uint32_t> timeout_ms = std::nullopt);

    std::vector<MeshCoreCoord> get_active_cores() const;

    MeshDevice* get_mesh_device() const;

    H2DMode get_h2d_mode() const;

    /**
     * @brief Returns whether the prior connector process shut down cleanly.
     *
     * On the owner side this is always true (no prior connector existed). On a
     * connector created via connect(), this reflects the clean_shutdown flag
     * left in SHM by the previous process: true if it ran its destructor, false
     * if it exited via crash, _exit, or kill. Useful for warning the operator
     * or running cleanup (e.g. discard_pending_pages on the paired D2H socket).
     */
    bool had_clean_prior_shutdown() const;

    // Non-copyable and non-movable: the destructor drains the device acknowledgement and
    // releases pinned/shared memory, so the instance must stay put. Hold it by unique_ptr.
    H2DSocket(const H2DSocket&) = delete;
    H2DSocket& operator=(const H2DSocket&) = delete;
    H2DSocket(H2DSocket&&) = delete;
    H2DSocket& operator=(H2DSocket&&) = delete;

private:
    // Wrap an already-built implementation object (connect paths).
    explicit H2DSocket(std::unique_ptr<H2DSocketImpl> impl);

    // DRAM-receiver ctor (invoked only by H2DSocketDramRecvAccess). Bypasses the
    // MeshBuffer paths (which have no DRAM-core L1 allocator) and consumes
    // pre-allocated DRISC-L1 offsets for the config and data buffers, plus the
    // DRAM-L1 NOC offset that host writes need to add on top.
    H2DSocket(
        const std::shared_ptr<MeshDevice>& mesh_device,
        const MeshCoreCoord& recv_core,
        uint32_t fifo_size,
        uint32_t config_l1_local_addr,
        uint32_t data_l1_local_addr,
        uint64_t dram_l1_noc_offset);

    // Non-blocking write. Returns false immediately if the FIFO can't fit `num_pages`
    // without spinning. Accessible only via tt::tt_metal::experimental::detail::try_write.
    bool try_write_impl(void* data, uint32_t num_pages);
    // Writes as many of the `num_pages` pages at `pages` (one pointer per page, each get_page_size() bytes)
    // as the FIFO has room for, in order, without store fences or a receiver notification, and returns how
    // many it wrote. Only the pages it writes are read. They are counted as sent and reach the receiver only
    // through publish_staged. Accessible only via tt::tt_metal::experimental::detail::try_stage /
    // publish_staged.
    uint32_t try_stage_impl(const uint8_t* const* pages, uint32_t num_pages);
    // Publishes bytes_sent to the receiver without a store fence after it. Accessible only via
    // tt::tt_metal::experimental::detail::publish_staged.
    void notify_receiver_unfenced();

    friend struct tt::tt_metal::experimental::detail::H2DSocketTryWriteAccess;
    friend struct tt::tt_metal::experimental::detail::H2DSocketDramRecvAccess;

    // All state lives in the implementation object (tt_metal/distributed/h2d_socket_impl.hpp) so
    // this header stays free of MeshBuffer, PinnedMemory, UMD and SHM types.
    std::unique_ptr<H2DSocketImpl> pimpl_;
};

}  // namespace tt::tt_metal::distributed
