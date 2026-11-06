// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Implementation object behind the public H2DSocket handle (see
// tt-metalium/experimental/sockets/h2d_socket.hpp). Private to libtt_metal: it names
// MeshBuffer, PinnedMemory, UMD windows and the SHM connector state that the public
// header deliberately does not include.

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/experimental/pinned_memory.hpp>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/mesh_buffer.hpp>

namespace tt::umd {
class IoWindow;
}

namespace tt::tt_metal::distributed {

class NamedShm;
class PCIeCoreWriter;
struct HDSocketConnectorState;
struct HDSocketDescriptor;

class H2DSocketImpl {
public:
    // Owner, worker-core receiver. Mirrors H2DSocket's public ctor.
    H2DSocketImpl(
        const std::shared_ptr<MeshDevice>& mesh_device,
        const MeshCoreCoord& recv_core,
        BufferType buffer_type,
        uint32_t fifo_size,
        H2DMode h2d_mode);

    // Owner, L2CPU receiver. Mirrors H2DSocket's L2CPU ctor.
    H2DSocketImpl(
        MeshDevice& mesh_device,
        const MeshCoreCoord& recv_l2cpu,
        uint32_t fifo_size,
        uint32_t config_buffer_address,
        uint32_t data_fifo_address,
        H2DMode h2d_mode);

    // Owner, DRAM-core receiver (reached via experimental::detail::H2DSocketDramRecvAccess).
    // Bypasses the MeshBuffer paths (which have no DRAM-core L1 allocator) and consumes
    // pre-allocated DRISC-L1 offsets for the config and data buffers, plus the DRAM-L1 NOC
    // offset that host writes need to add on top.
    H2DSocketImpl(
        const std::shared_ptr<MeshDevice>& mesh_device,
        const MeshCoreCoord& recv_core,
        uint32_t fifo_size,
        uint32_t config_l1_local_addr,
        uint32_t data_l1_local_addr,
        uint64_t dram_l1_noc_offset);

    // Connector: attach to an owner's socket from its descriptor. No MetalContext required.
    static std::unique_ptr<H2DSocketImpl> connect_from_descriptor(const HDSocketDescriptor& desc);

    ~H2DSocketImpl() noexcept;

    // Not movable: enable_mock_flow_control aliases bytes_acked_ptr_ to &bytes_sent_, and the
    // pcie_writer lambdas capture `this`.
    H2DSocketImpl(const H2DSocketImpl&) = delete;
    H2DSocketImpl& operator=(const H2DSocketImpl&) = delete;

    std::string export_descriptor(const std::string& socket_id);
    HDSocketDescriptor populate_descriptor() const;

    uint32_t get_page_size() const { return page_size_; }
    uint32_t get_config_buffer_address() const { return config_buffer_address_; }
    uint32_t get_fifo_size() const { return fifo_size_; }
    uint32_t get_bytes_sent() const { return bytes_sent_; }
    bool had_clean_prior_shutdown() const { return prior_clean_shutdown_; }
    MeshDevice* get_mesh_device() const { return mesh_device_; }
    H2DMode get_h2d_mode() const { return h2d_mode_; }
    const MeshCoreCoord& get_recv_core() const { return recv_core_; }

    bool has_space(std::optional<uint32_t> num_bytes_to_check);
    bool acked_past(uint32_t watermark);
    void set_page_size(uint32_t page_size);
    void write(void* data, uint32_t num_pages);
    // Non-blocking write; false when the FIFO cannot take `num_pages` right now.
    bool try_write(void* data, uint32_t num_pages);
    // Stages up to `num_pages` pages without fences or a notification; see H2DSocket::try_stage_impl.
    uint32_t try_stage(const uint8_t* const* pages, uint32_t num_pages);
    // notify_receiver without its store fence; publish_staged fences a batch of sockets once.
    void notify_receiver_unfenced();
    void barrier(std::optional<uint32_t> timeout_ms);

private:
    H2DSocketImpl() = default;

    // Programmable-core type of the receiver core. Recorded explicitly at
    // construction rather than inferred at use sites: logical coordinates
    // overlap across core types (DRAM logical (x,y) is a different physical
    // core than Tensix logical (x,y)), and keying off the DRAM-L1 NOC offset
    // conflates "needs an L1 offset" with "is a DRAM core". The DRAM-recv ctor
    // sets Dram; every other path (owner worker ctor, connect()) is Tensix.
    enum class RecvCoreType { Tensix, Dram };

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
    void init_receiver_tlb(const std::shared_ptr<MeshDevice>& mesh_device);

    // Mock owner only: alias bytes_acked_ptr_ to bytes_sent_ so the FIFO reads as drained.
    // Connectors remain context-free and do not use this path.
    void enable_mock_flow_control(const MeshDevice& mesh_device);

    void reserve_bytes(uint32_t num_bytes);
    void push_bytes(uint32_t num_bytes);
    void notify_receiver();

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
    std::unique_ptr<tt::umd::IoWindow> receiver_core_window_;
    std::shared_ptr<tt::tt_metal::experimental::PinnedMemory> pinned_memory_ = nullptr;
    std::shared_ptr<uint32_t[]> host_buffer_ = nullptr;
    // HOST_PUSH only: try_stage gathers a run of pages here so that the run goes out in one write.
    std::vector<uint8_t> stage_buffer_;
    uint32_t* bytes_acked_ptr_ = nullptr;
    std::function<void(void*, uint32_t, uint64_t)> pcie_writer = nullptr;
    H2DMode h2d_mode_ = H2DMode::HOST_PUSH;
    std::unique_ptr<NamedShm> shm_;
    std::unique_ptr<PCIeCoreWriter> pcie_writer_instance_;
    MeshDevice* mesh_device_ = nullptr;
    // Resolved once at construction so I/O never reads the mesh view; see validate_host_socket_access.
    bool rank_owns_endpoint_ = true;
    bool is_owner_ = true;
    std::string descriptor_path_;
    HDSocketConnectorState* connector_state_ = nullptr;
    uint32_t connector_state_offset_ = 0;
    bool prior_clean_shutdown_ = true;
    // Non-zero when the recv_core is a DRAM programmable core: every NOC write
    // from host to its L1 must add this offset on top of the local L1 address.
    // Zero for worker recv cores (worker L1 has local==NOC space). init_receiver_tlb
    // anchors the receiver's window here, or adds it to each write when there is no
    // window, so write() can keep passing local addresses.
    uint64_t dram_l1_noc_offset_ = 0;
    // Receiver core type, set at construction. The authoritative signal for
    // CoreType resolution and the DRAM-recv write path in init_receiver_tlb.
    RecvCoreType recv_core_type_ = RecvCoreType::Tensix;

    // True when the receiver is an L2CPU tile. Selects the L2CPU code paths in
    // init_receiver_tlb() / write_socket_metadata() and blocks descriptor export.
    bool is_l2cpu_ = false;
};

}  // namespace tt::tt_metal::distributed
