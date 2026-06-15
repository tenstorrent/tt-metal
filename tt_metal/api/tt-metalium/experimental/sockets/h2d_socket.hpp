// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include <tt-metalium/experimental/pinned_memory.hpp>
#include <tt-metalium/hal_types.hpp>
#include <memory>
#include <optional>
#include <utility>

namespace tt::umd {
class TlbWindow;
}

namespace tt::tt_metal::experimental::detail {
struct H2DSocketTryWriteAccess;
struct H2DSocketDramRecvAccess;
}  // namespace tt::tt_metal::experimental::detail

namespace tt::tt_metal::distributed {

class NamedShm;
class PCIeCoreWriter;
struct HDSocketConnectorState;
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

    uint32_t get_page_size() const { return page_size_; }

    uint32_t get_config_buffer_address() const { return config_buffer_address_; }

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
    bool had_clean_prior_shutdown() const { return prior_clean_shutdown_; }

    H2DSocket(const H2DSocket&) = delete;
    H2DSocket& operator=(const H2DSocket&) = delete;

private:
    H2DSocket() = default;

    // Programmable-core type of the receiver core. Recorded explicitly at
    // construction rather than inferred at use sites: logical coordinates
    // overlap across core types (DRAM logical (x,y) is a different physical
    // core than Tensix logical (x,y)), and keying off the DRAM-L1 NOC offset
    // conflates "needs an L1 offset" with "is a DRAM core". The DRAM-recv ctor
    // sets Dram; every other path (owner worker ctor, connect()) is Tensix.
    enum class RecvCoreType { Tensix, Dram };

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

    // Non-blocking write. Returns false immediately if the FIFO can't fit `num_pages`
    // without spinning. Accessible only via tt::tt_metal::experimental::detail::try_write.
    bool try_write_impl(void* data, uint32_t num_pages);

    friend struct tt::tt_metal::experimental::detail::H2DSocketTryWriteAccess;
    friend struct tt::tt_metal::experimental::detail::H2DSocketDramRecvAccess;

    std::shared_ptr<MeshBuffer> config_buffer_ = nullptr;
    std::shared_ptr<MeshBuffer> data_buffer_ = nullptr;
    // Set only when recv_core_ is a claimed service core; the dtor releases them to the per-core allocator.
    std::optional<DeviceAddr> svc_config_l1_addr_;
    std::optional<DeviceAddr> svc_data_l1_addr_;
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
    std::unique_ptr<PCIeCoreWriter> pcie_writer_instance_;
    MeshDevice* mesh_device_ = nullptr;
    bool is_owner_ = true;
    std::string descriptor_path_;
    HDSocketConnectorState* connector_state_ = nullptr;
    uint32_t connector_state_offset_ = 0;
    bool prior_clean_shutdown_ = true;
    // Non-zero when the recv_core is a DRAM programmable core: every NOC write
    // from host to its L1 must add this offset on top of the local L1 address.
    // Zero for worker recv cores (worker L1 has local==NOC space). Captured
    // into the pcie_writer lambda in init_receiver_tlb so write() can keep
    // passing local addresses.
    uint64_t dram_l1_noc_offset_ = 0;
    // Receiver core type, set at construction. The authoritative signal for
    // CoreType resolution and the DRAM-recv write path in init_receiver_tlb.
    RecvCoreType recv_core_type_ = RecvCoreType::Tensix;
};

}  // namespace tt::tt_metal::distributed
