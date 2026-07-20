// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <internal/service/service_core_manager.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/distributed/mesh_socket_utils.hpp"
#include "tt_metal/distributed/named_shm.hpp"
#include "tt_metal/distributed/hd_socket_connector_state.hpp"
#include "tt_metal/distributed/hd_socket_descriptor.hpp"
#include "tt_metal/distributed/pcie_core_writer.hpp"
#include "tt_metal/distributed/shm_resource_tracker.hpp"
#include "tt_metal/impl/buffers/h2d_socket_internal.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_metal/hw/inc/hostdev/socket.h"
#include "tt_metal/llrt/tt_cluster.hpp"
#include <tt-metalium/tt_align.hpp>
#include <umd/device/chip_helpers/tlb_manager.hpp>
#include <tt-logger/tt-logger.hpp>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>

namespace tt::tt_metal::distributed {

namespace {

void advance_h2d_simulator_socket_device(MeshDevice* mesh_device, const MeshCoordinate& device_coord) {
    if (mesh_device == nullptr) {
        return;
    }

    const auto& cluster = MetalContext::instance().get_cluster();
    if (cluster.get_target_device_type() != tt::TargetDevice::Simulator) {
        return;
    }

    cluster.advance_device_execution(mesh_device->get_device(device_coord)->id());
}

}  // namespace

H2DSocket::PinnedBufferInfo H2DSocket::init_bytes_acked_buffer(
    const std::shared_ptr<MeshDevice>& mesh_device,
    const MeshCoordinateRangeSet& device_range,
    uint32_t pcie_alignment,
    const std::string& shm_name) {
    size_t page_size = sysconf(_SC_PAGESIZE);
    // The pinned region is just a 4-byte bytes_acked counter; the rest of the
    // page hosts the connector-state struct so consecutive driver processes
    // can resume from the previous driver's bytes_sent / write_ptr / page_size.
    // Place the struct immediately after the pinned region, aligned up to its
    // own alignment requirement so the reinterpret_cast<> in connect() is
    // well-defined.
    connector_state_offset_ = static_cast<uint32_t>(align(sizeof(uint32_t), alignof(HDSocketConnectorState)));
    TT_FATAL(
        page_size >= connector_state_offset_ + sizeof(HDSocketConnectorState),
        "System page size too small to host HDSocketConnectorState.");
    shm_ = std::make_unique<NamedShm>(NamedShm::create(shm_name, page_size));
    void* aligned_ptr = shm_->ptr();
    TT_FATAL(
        reinterpret_cast<uintptr_t>(aligned_ptr) % pcie_alignment == 0,
        "System Memory Allocation Error: Bytes_acked buffer must be aligned to the PCIe alignment.");
    // NamedShm::create zero-initializes the region; no explicit memset needed.
    host_buffer_ = std::shared_ptr<uint32_t[]>(static_cast<uint32_t*>(aligned_ptr), [](uint32_t*) {});
    tt::tt_metal::HostBuffer bytes_acked_buffer_view(
        ttsl::Span<uint32_t>(host_buffer_.get(), 1), tt::tt_metal::MemoryPin(host_buffer_));
    pinned_memory_ =
        tt::tt_metal::experimental::PinnedMemory::Create(*mesh_device, device_range, bytes_acked_buffer_view, true);

    const auto& noc_addr = pinned_memory_->get_noc_addr(mesh_device->get_device(recv_core_.device_coord)->id());
    TT_FATAL(noc_addr.has_value(), "Failed to get NOC address for bytes_acked pinned memory.");
    TT_FATAL(
        noc_addr.value().device_id == mesh_device->get_device(recv_core_.device_coord)->id(),
        "Pinned Memory used for H2D sockets must be mapped to the same device as the receiver core. H2D Sockets cannot "
        "communicate with remote devices");
    return PinnedBufferInfo{
        .pcie_xy_enc = noc_addr.value().pcie_xy_enc,
        .addr_lo = static_cast<uint32_t>(noc_addr.value().addr & 0xFFFFFFFFull),
        .addr_hi = static_cast<uint32_t>(noc_addr.value().addr >> 32)};
}

H2DSocket::PinnedBufferInfo H2DSocket::init_host_data_buffer(
    const std::shared_ptr<MeshDevice>& mesh_device,
    const MeshCoordinateRangeSet& device_range,
    uint32_t pcie_alignment,
    const std::string& shm_name) {
    uint32_t host_buffer_size_bytes = fifo_size_ + sizeof(uint32_t);
    uint32_t host_buffer_size_words = host_buffer_size_bytes / sizeof(uint32_t);
    size_t page_size = sysconf(_SC_PAGESIZE);
    // Reserve room for HDSocketConnectorState immediately after the pinned region,
    // aligned up to its own alignment requirement so the reinterpret_cast<> in
    // connect() is well-defined. The pinned HostBuffer view below still spans
    // only [data | bytes_acked], so the device never touches the state struct.
    connector_state_offset_ = align(host_buffer_size_bytes, alignof(HDSocketConnectorState));
    size_t alloc_size = align(connector_state_offset_ + sizeof(HDSocketConnectorState), page_size);
    shm_ = std::make_unique<NamedShm>(NamedShm::create(shm_name, alloc_size));
    void* aligned_ptr = shm_->ptr();
    TT_FATAL(
        reinterpret_cast<uintptr_t>(aligned_ptr) % pcie_alignment == 0,
        "System Memory Allocation Error: Host data buffer must be aligned to the PCIe alignment.");
    host_buffer_ = std::shared_ptr<uint32_t[]>(static_cast<uint32_t*>(aligned_ptr), [](uint32_t*) {});

    tt::tt_metal::HostBuffer host_buffer_view(
        ttsl::Span<uint32_t>(host_buffer_.get(), host_buffer_size_words), tt::tt_metal::MemoryPin(host_buffer_));
    pinned_memory_ =
        tt::tt_metal::experimental::PinnedMemory::Create(*mesh_device, device_range, host_buffer_view, true);

    const auto& noc_addr = pinned_memory_->get_noc_addr(mesh_device->get_device(recv_core_.device_coord)->id());
    TT_FATAL(noc_addr.has_value(), "Failed to get NOC address for data pinned memory.");

    return PinnedBufferInfo{
        .pcie_xy_enc = noc_addr.value().pcie_xy_enc,
        .addr_lo = static_cast<uint32_t>(noc_addr.value().addr & 0xFFFFFFFFull),
        .addr_hi = static_cast<uint32_t>(noc_addr.value().addr >> 32)};
}

void H2DSocket::init_config_buffer(const std::shared_ptr<MeshDevice>& mesh_device) {
    uint32_t config_buffer_size = sizeof(receiver_socket_md);
    auto num_cores = 1;
    auto shard_params = ShardSpecBuffer(
        CoreRangeSet(recv_core_.core_coord),
        {1, 1},
        ShardOrientation::ROW_MAJOR,
        {1, 1},
        {static_cast<uint32_t>(num_cores), 1});

    DeviceLocalBufferConfig config_buffer_specs = {
        .page_size = config_buffer_size,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(shard_params, TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = std::nullopt,
        .sub_device_id = std::nullopt,
    };

    MeshBufferConfig config_mesh_buffer_specs = ReplicatedBufferConfig{
        .size = config_buffer_size,
    };

    // On a claimed service core the worker-grid BankManager can't reach L1; allocate from the service-core allocator.
    std::optional<DeviceAddr> preallocated_addr;
    auto& svc = tt::tt_metal::MetalContext::instance().get_service_core_manager();
    auto* recv_device = mesh_device->get_device(recv_core_.device_coord);
    if (svc.claimed_cores(recv_device->id()).contains(recv_core_.core_coord)) {
        svc_config_l1_addr_ = svc.allocate_l1(recv_device, recv_core_.core_coord, config_buffer_size);
        preallocated_addr = svc_config_l1_addr_;
    }

    config_buffer_ =
        MeshBuffer::create(config_mesh_buffer_specs, config_buffer_specs, mesh_device.get(), preallocated_addr);
}

void H2DSocket::init_data_buffer(const std::shared_ptr<MeshDevice>& mesh_device, uint32_t pcie_alignment) {
    if (h2d_mode_ != H2DMode::HOST_PUSH) {
        // DEVICE_PULL: data FIFO lives in pinned host memory; no device-side L1
        // allocation needed.
        write_ptr_ = 0;
        return;
    }

    auto& svc = tt::tt_metal::MetalContext::instance().get_service_core_manager();
    auto* recv_device = mesh_device->get_device(recv_core_.device_coord);
    if (svc.claimed_cores(recv_device->id()).contains(recv_core_.core_coord)) {
        const uint64_t alloc_size = fifo_size_ + pcie_alignment;
        DeviceAddr raw_addr = svc.allocate_l1(recv_device, recv_core_.core_coord, alloc_size);
        svc_data_l1_addr_ = raw_addr;

        auto shard_params =
            ShardSpecBuffer(CoreRangeSet(recv_core_.core_coord), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});
        DeviceLocalBufferConfig data_buffer_specs = {
            .page_size = static_cast<uint32_t>(alloc_size),
            .buffer_type = buffer_type_,
            .sharding_args = BufferShardingArgs(shard_params, TensorMemoryLayout::HEIGHT_SHARDED),
            .bottom_up = std::nullopt,
            .sub_device_id = std::nullopt,
        };
        MeshBufferConfig data_mesh_buffer_specs = ReplicatedBufferConfig{.size = alloc_size};
        data_buffer_ = MeshBuffer::create(
            data_mesh_buffer_specs, data_buffer_specs, mesh_device.get(), std::make_optional<DeviceAddr>(raw_addr));
        aligned_data_buf_start_ = tt::align(raw_addr, pcie_alignment);
        write_ptr_ = 0;
        return;
    }

    auto num_data_cores = mesh_device->num_worker_cores(HalProgrammableCoreType::TENSIX, SubDeviceId{0});
    auto shard_grid = mesh_device->worker_cores(HalProgrammableCoreType::TENSIX, SubDeviceId{0});

    // Allocate buffer at a PCIe aligned address. This requires extra memory to be allocated.
    auto total_data_buffer_size = num_data_cores * (fifo_size_ + pcie_alignment);

    DeviceLocalBufferConfig data_buffer_specs = {
        .page_size = fifo_size_ + pcie_alignment,
        .buffer_type = buffer_type_,
        .sharding_args = BufferShardingArgs(
            ShardSpecBuffer(shard_grid, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {num_data_cores, 1}),
            TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = std::nullopt,
        .sub_device_id = std::nullopt,
    };
    MeshBufferConfig data_mesh_buffer_specs = ReplicatedBufferConfig{
        .size = total_data_buffer_size,
    };
    data_buffer_ = MeshBuffer::create(data_mesh_buffer_specs, data_buffer_specs, mesh_device.get());
    aligned_data_buf_start_ = tt::align(data_buffer_->address(), pcie_alignment);
    write_ptr_ = 0;
}

void H2DSocket::write_socket_metadata(
    const std::shared_ptr<MeshDevice>& mesh_device,
    const PinnedBufferInfo& bytes_acked_info,
    const PinnedBufferInfo& data_info) {
    // init_config_buffer hardcodes num_cores = 1, so the config buffer always has exactly one slot at index 0.
    std::vector<receiver_socket_md> config_data(
        config_buffer_->size() / sizeof(receiver_socket_md), receiver_socket_md());

    auto& md = config_data[0];
    md.bytes_sent = 0;
    md.bytes_acked = 0;
    md.read_ptr = aligned_data_buf_start_;
    md.fifo_addr = aligned_data_buf_start_;
    md.fifo_total_size = fifo_size_;
    md.is_h2d = 1;
    md.h2d.bytes_acked_addr_lo = bytes_acked_info.addr_lo;
    md.h2d.bytes_acked_addr_hi = bytes_acked_info.addr_hi;
    md.h2d.data_addr_lo = data_info.addr_lo;
    md.h2d.data_addr_hi = data_info.addr_hi;
    md.h2d.pcie_xy_enc = bytes_acked_info.pcie_xy_enc;

    if (svc_config_l1_addr_.has_value()) {
        // WriteShard can't reach service cores, so write L1 directly. config_buffer_address_ isn't assigned yet.
        auto* device = mesh_device->get_device(recv_core_.device_coord);
        std::span<const uint8_t> bytes(reinterpret_cast<const uint8_t*>(&md), sizeof(md));
        tt::tt_metal::detail::WriteToDeviceL1(
            device, recv_core_.core_coord, static_cast<uint32_t>(config_buffer_->address()), bytes);
    } else {
        distributed::WriteShard(
            mesh_device->mesh_command_queue(0), config_buffer_, config_data, recv_core_.device_coord, true);
    }
}

void H2DSocket::init_receiver_tlb(const std::shared_ptr<MeshDevice>& mesh_device, std::optional<uint32_t> device_id) {
    TT_FATAL(mesh_device || device_id.has_value(), "Either mesh_device or device_id must be provided.");

    uint32_t recv_device_id;
    CoreCoord recv_virtual_core;

    const auto& cluster = MetalContext::instance().get_cluster();

    // MockChip has no TLB manager (get_tlb_manager() == nullptr), so skip TLB window
    // setup entirely: pcie_writer stays unset. Safe under Mock because the runtime I/O
    // paths that use pcie_writer -- write() (HOST_PUSH) and notify_receiver() -- never
    // execute for mock devices (mock only exercises socket construction / JIT).
    //
    // TODO(emule): this over-skips for Emule. SWEmuleChip also lacks a TLB manager but has
    // real memory-backed I/O, so it should skip only the TLB-window path and still install
    // the cluster.write_core() fallback for pcie_writer. As written, pcie_writer is left
    // null, so enabling H2D socket runtime I/O under emule would null-deref in
    // notify_receiver() / write().
    if (cluster.is_mock_or_emulated()) {
        return;
    }

    // Receiver core type is recorded explicitly at construction (the DRAM-recv
    // ctor sets Dram, every other path is Tensix). Used only to resolve the
    // virtual coordinate — logical coords overlap across core types, so this
    // can't be inferred from coordinates here.
    const CoreType recv_umd_core_type = (recv_core_type_ == RecvCoreType::Dram) ? CoreType::DRAM : CoreType::TENSIX;

    if (mesh_device) {
        IDevice* recv_device = mesh_device->get_device(recv_core_.device_coord);
        recv_device_id = recv_device->id();
        recv_virtual_core = recv_device->virtual_core_from_logical_core(recv_core_.core_coord, recv_umd_core_type);
    } else {
        recv_device_id = device_id.value();
        recv_virtual_core = cluster.get_virtual_coordinate_from_logical_coordinates(
            recv_device_id, recv_core_.core_coord, recv_umd_core_type);
    }

    // For DRAM-core recv, every host NOC write to the DRISC L1 needs the DRAM-L1
    // NOC offset added on top of the local L1 address (DRAM cores have two NOC
    // spaces: low addresses route to DRAM bank, high addresses route to L1).
    // Captured into the lambdas below so write() can keep passing local addresses.
    const uint64_t l1_offset = dram_l1_noc_offset_;

    // Take the static-TLB path only when UMD reports that our actual write target
    // lives inside a static window for this core — ask the TLB manager rather than
    // assuming based on core type. On Blackhole, Tensix/Eth cores get a static
    // window mapping L1, so their writes land inside it; DRAM cores also get a
    // static window, but it maps the DRAM-bank space at [0, 4 GB) while our writes
    // target device_addr + l1_offset (a high DRAM-L1 NOC address, e.g.
    // 0x2000000000+…) outside that window, so is_tlb_mapped reports false and we
    // fall through to cluster.write_core. Also gated on owning a mesh_device
    // (statically initialized TLBs) and Blackhole — on Wormhole B0 the device
    // address space isn't fully statically mapped and a mapped window may still
    // need a per-write driver reconfig.
    const tt_xy_pair tlb_core(recv_virtual_core.x, recv_virtual_core.y);
    const bool target_in_static_tlb =
        mesh_device && MetalContext::instance().hal().get_arch() == tt::ARCH::BLACKHOLE &&
        cluster.get_driver()
            ->get_chip(recv_device_id)
            ->get_tlb_manager()
            ->is_tlb_mapped(tlb_core, static_cast<uint64_t>(aligned_data_buf_start_) + l1_offset, fifo_size_);

    if (target_in_static_tlb) {
        receiver_core_tlb_ =
            cluster.get_driver()->get_chip(recv_device_id)->get_tlb_manager()->get_tlb_window(tlb_core);
        pcie_writer = [this, l1_offset](void* data, uint32_t num_bytes, uint64_t device_addr) {
            receiver_core_tlb_->write_block(device_addr + l1_offset, data, num_bytes);
        };
    } else {
        // Mesh device not owned, non-Blackhole, or no static window covers the
        // target: use dynamic TLBs through UMD (the driver may reconfigure the TLB
        // per write). Covers Wormhole B0 and the DRAM-recv L1 path described above.
        pcie_writer = [recv_device_id, recv_virtual_core, l1_offset](
                          void* data, uint32_t num_bytes, uint64_t device_addr) {
            const auto& cluster = MetalContext::instance().get_cluster();
            cluster.write_core(
                data, num_bytes, tt_cxy_pair(recv_device_id, recv_virtual_core), device_addr + l1_offset);
        };
    }
}

H2DSocket::H2DSocket(
    const std::shared_ptr<MeshDevice>& mesh_device,
    const MeshCoreCoord& recv_core,
    BufferType buffer_type,
    uint32_t fifo_size,
    H2DMode h2d_mode) :
    recv_core_(recv_core),
    buffer_type_(buffer_type),
    fifo_size_(fifo_size),
    pcie_alignment_(MetalContext::instance().hal().get_alignment(HalMemType::HOST)),
    pinned_memory_(nullptr),
    h2d_mode_(h2d_mode),
    mesh_device_(mesh_device.get()) {
    MeshCoordinateRangeSet recv_device_range_set;
    recv_device_range_set.merge(MeshCoordinateRange(recv_core_.device_coord));

    const uint32_t pcie_alignment = pcie_alignment_;
    TT_FATAL(fifo_size_ % pcie_alignment == 0, "FIFO size must be PCIE-aligned.");
    TT_FATAL(buffer_type_ == BufferType::L1, "H2D sockets currently only support data buffers in SRAM.");

    std::string shm_name = generate_shm_name("h2d");

    PinnedBufferInfo bytes_acked_info = {};
    PinnedBufferInfo data_info = {};
    if (h2d_mode_ == H2DMode::DEVICE_PULL) {
        data_info = init_host_data_buffer(mesh_device, recv_device_range_set, pcie_alignment, shm_name);
        bytes_acked_info = data_info;
        auto bytes_acked_addr = (static_cast<uint64_t>(data_info.addr_hi) << 32 | data_info.addr_lo) + fifo_size_;
        bytes_acked_info.addr_hi = static_cast<uint32_t>(bytes_acked_addr >> 32);
        bytes_acked_info.addr_lo = static_cast<uint32_t>(bytes_acked_addr & 0xFFFFFFFFull);
        bytes_acked_ptr_ = host_buffer_.get() + (fifo_size_ / sizeof(uint32_t));
        TT_FATAL(
            bytes_acked_info.pcie_xy_enc == data_info.pcie_xy_enc,
            "Bytes_acked and data pinned memory must be mapped to the same PCIe core.");
    } else {
        bytes_acked_info = init_bytes_acked_buffer(mesh_device, recv_device_range_set, pcie_alignment, shm_name);
        bytes_acked_ptr_ = host_buffer_.get();
    }

    init_config_buffer(mesh_device);
    init_data_buffer(mesh_device, pcie_alignment);
    write_socket_metadata(mesh_device, bytes_acked_info, data_info);
    init_receiver_tlb(mesh_device);

    config_buffer_address_ = config_buffer_->address();

    // Initialize the persistent connector-state struct living in SHM.
    // NamedShm::create zero-initialized the region; we stamp the version and
    // mark clean_shutdown=1 so the first connect() sees "no prior crash" (the
    // owner side has nothing to recover from).
    connector_state_ =
        reinterpret_cast<HDSocketConnectorState*>(static_cast<uint8_t*>(shm_->ptr()) + connector_state_offset_);
    connector_state_->version = kHDSocketConnectorStateVersion;
    connector_state_->clean_shutdown = 1;
}

H2DSocket::H2DSocket(
    const std::shared_ptr<MeshDevice>& mesh_device,
    const MeshCoreCoord& recv_core,
    uint32_t fifo_size,
    uint32_t config_l1_local_addr,
    uint32_t data_l1_local_addr,
    uint64_t dram_l1_noc_offset) :
    recv_core_(recv_core),
    fifo_size_(fifo_size),
    pcie_alignment_(
        MetalContext::instance(extract_context_id(mesh_device.get())).hal().get_alignment(HalMemType::HOST)),
    pinned_memory_(nullptr),
    mesh_device_(mesh_device.get()),
    dram_l1_noc_offset_(dram_l1_noc_offset),
    recv_core_type_(RecvCoreType::Dram) {
    MeshCoordinateRangeSet recv_device_range_set;
    recv_device_range_set.merge(MeshCoordinateRange(recv_core_.device_coord));

    TT_FATAL(fifo_size_ % pcie_alignment_ == 0, "FIFO size must be PCIE-aligned.");

    // Allocate the bytes_acked host-pinned buffer (same path as the worker ctor).
    std::string shm_name = generate_shm_name("h2d");
    PinnedBufferInfo bytes_acked_info =
        init_bytes_acked_buffer(mesh_device, recv_device_range_set, pcie_alignment_, shm_name);
    bytes_acked_ptr_ = host_buffer_.get();

    // Take the caller-supplied DRISC L1 offsets verbatim. No MeshBuffer allocation:
    // the framework's L1 allocator is worker-only, and host writes to DRAM-L1 go
    // through the DRAM_L1_NOC_OFFSET path below.
    config_buffer_address_ = config_l1_local_addr;
    aligned_data_buf_start_ = data_l1_local_addr;
    write_ptr_ = 0;

    // Initialize the TLB / pcie_writer before writing socket metadata — the
    // metadata write goes through the same NOC-offset path as subsequent data
    // writes, so we use cluster.write_core directly here rather than via
    // pcie_writer to keep the dependency order simple.
    init_receiver_tlb(mesh_device);

    receiver_socket_md md{};
    md.bytes_sent = 0;
    md.bytes_acked = 0;
    md.read_ptr = aligned_data_buf_start_;
    md.fifo_addr = aligned_data_buf_start_;
    md.fifo_total_size = fifo_size_;
    md.is_h2d = 1;
    md.h2d.bytes_acked_addr_lo = bytes_acked_info.addr_lo;
    md.h2d.bytes_acked_addr_hi = bytes_acked_info.addr_hi;
    // For DEVICE_PULL we'd populate data_addr_*; HOST_PUSH-only here.
    md.h2d.data_addr_lo = 0;
    md.h2d.data_addr_hi = 0;
    md.h2d.pcie_xy_enc = bytes_acked_info.pcie_xy_enc;

    const CoreCoord virtual_core = mesh_device->get_device(recv_core_.device_coord)
                                       ->virtual_core_from_logical_core(recv_core_.core_coord, CoreType::DRAM);
    MetalContext::instance(extract_context_id(mesh_device.get()))
        .get_cluster()
        .write_core(
            mesh_device->get_device(recv_core_.device_coord)->id(),
            tt_cxy_pair(mesh_device->get_device(recv_core_.device_coord)->id(), virtual_core),
            std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(&md), sizeof(md)),
            static_cast<uint64_t>(config_buffer_address_) + dram_l1_noc_offset_);
}

H2DSocket::~H2DSocket() noexcept {
    try {
        barrier(1000);
    } catch (const std::exception& e) {
        log_warning(LogMetal, "H2DSocket destructor: barrier failed with exception: {}", e.what());
    } catch (...) {
        log_warning(LogMetal, "H2DSocket destructor: barrier failed with unknown exception");
    }
    // Drop the MeshBuffer views before deallocating their L1 so no Buffer outlives its memory.
    if (svc_config_l1_addr_.has_value() || svc_data_l1_addr_.has_value()) {
        try {
            config_buffer_.reset();
            data_buffer_.reset();
            auto& svc = tt::tt_metal::MetalContext::instance().get_service_core_manager();
            auto* recv_device = mesh_device_->get_device(recv_core_.device_coord);
            if (svc_config_l1_addr_.has_value()) {
                svc.deallocate_l1(recv_device, recv_core_.core_coord, svc_config_l1_addr_.value());
            }
            if (svc_data_l1_addr_.has_value()) {
                svc.deallocate_l1(recv_device, recv_core_.core_coord, svc_data_l1_addr_.value());
            }
        } catch (const std::exception& e) {
            log_warning(LogMetal, "H2DSocket destructor: service-core L1 release failed: {}", e.what());
        } catch (...) {
            log_warning(LogMetal, "H2DSocket destructor: service-core L1 release failed with unknown exception");
        }
    }
    // Mark a clean shutdown so the next connector sees clean_shutdown=1. A process
    // that exits without running this destructor (crash, _exit, kill) leaves the
    // 0 written by connect()/owner-construct in place, signalling unclean exit.
    if (connector_state_) {
        connector_state_->clean_shutdown = 1;
    }
    if (is_owner_) {
        pinned_memory_.reset();
        if (shm_) {
            shm_->unlink();
        }
        if (!descriptor_path_.empty()) {
            if (std::remove(descriptor_path_.c_str()) == 0 || errno == ENOENT) {
                ShmResourceTracker::instance().untrack_file(descriptor_path_);
            }
        }
    }
}

void H2DSocket::reserve_bytes(uint32_t num_bytes) {
    uint32_t bytes_free = fifo_size_ - (bytes_sent_ - bytes_acked_);
    while (bytes_free < num_bytes) {
        advance_h2d_simulator_socket_device(mesh_device_, recv_core_.device_coord);
        tt_driver_atomics::mfence();
        volatile uint32_t bytes_acked_value = bytes_acked_ptr_[0];
        bytes_free = fifo_size_ - (bytes_sent_ - bytes_acked_value);
        bytes_acked_ = bytes_acked_value;
    }
}

bool H2DSocket::has_space(std::optional<uint32_t> num_bytes_to_check) {
    TT_FATAL(page_size_ > 0, "Page size must be set before checking for data.");
    uint32_t num_bytes = num_bytes_to_check.value_or(page_size_);
    uint32_t bytes_free = fifo_size_ - (bytes_sent_ - bytes_acked_);

    // bytes_acked_ is monotonically increasing -> more bytes acked => more bytes_free
    // If we bytes_acked_old < bytes_acked_new => bytes_free_old < bytes_free_new
    // If bytes_free_old > num_bytes then this is safe as bytes_free_new > bytes_free_old > num_bytes necessarily
    if (bytes_free >= num_bytes) {
        return true;
    }

    tt_driver_atomics::mfence();
    volatile uint32_t bytes_acked_value = bytes_acked_ptr_[0];
    bytes_free = fifo_size_ - (bytes_sent_ - bytes_acked_value);
    bytes_acked_ = bytes_acked_value;
    return bytes_free >= num_bytes;
}

bool H2DSocket::acked_past(uint32_t watermark) {
    // in_flight = bytes_sent_ - bytes_acked_ (unsigned, always <= fifo_size_)
    // bytes_since_watermark = bytes_sent_ - watermark (unsigned, in [0, fifo_size_])
    // Write at watermark is done iff bytes_acked_ >= watermark, equivalently
    // in_flight <= bytes_since_watermark.
    uint32_t bytes_since_watermark = bytes_sent_ - watermark;
    if (bytes_sent_ - bytes_acked_ <= bytes_since_watermark) {
        return true;
    }
    tt_driver_atomics::mfence();
    volatile uint32_t bytes_acked_value = bytes_acked_ptr_[0];
    bytes_acked_ = bytes_acked_value;
    return bytes_sent_ - bytes_acked_ <= bytes_since_watermark;
}

void H2DSocket::push_bytes(uint32_t num_bytes) {
    if (write_ptr_ + num_bytes >= fifo_curr_size_) {
        write_ptr_ = write_ptr_ + num_bytes - fifo_curr_size_;
        bytes_sent_ += num_bytes + fifo_size_ - fifo_curr_size_;
    } else {
        write_ptr_ += num_bytes;
        bytes_sent_ += num_bytes;
    }
    // Crash-safe persistence for the next driver process. The DramRecv socket
    // (in-process prefetcher owner) skips connector-state setup — there's no
    // separate connector that ever attaches — so the writes are guarded.
    if (connector_state_) {
        connector_state_->bytes_sent = bytes_sent_;
        connector_state_->write_ptr = write_ptr_;
    }
}

void H2DSocket::notify_receiver() {
    uint32_t bytes_sent_addr = config_buffer_address_ + offsetof(receiver_socket_md, bytes_sent);
    pcie_writer(&bytes_sent_, sizeof(bytes_sent_), bytes_sent_addr);
    tt_driver_atomics::sfence();
}

void H2DSocket::set_page_size(uint32_t page_size) {
    TT_FATAL(pcie_alignment_ > 0, "PCIe alignment not initialized.");
    TT_FATAL(page_size % pcie_alignment_ == 0, "Page size must be PCIE-aligned.");
    TT_FATAL(page_size <= fifo_size_, "Page size must be less than or equal to the FIFO size.");

    // tt::align() uses a bitwise-OR formula that only produces correct
    // results when alignment is a power of two. Socket page sizes can be
    // non-power-of-two (e.g. 2560 = 5×512 for some shard sizes), where
    // tt::align(5120, 2560) returns 7168 instead of 5120. Use modular
    // arithmetic so this works for any positive alignment.
    uint32_t next_fifo_wr_ptr = ((write_ptr_ + page_size - 1) / page_size) * page_size;
    uint32_t fifo_page_aligned_size = fifo_size_ - (fifo_size_ % page_size);

    if (next_fifo_wr_ptr >= fifo_page_aligned_size) {
        bytes_sent_ += fifo_size_ - next_fifo_wr_ptr;
        next_fifo_wr_ptr = 0;
    }
    write_ptr_ = next_fifo_wr_ptr;
    page_size_ = page_size;
    fifo_curr_size_ = fifo_page_aligned_size;
    // DramRecv sockets skip the SHM connector-state region (no cross-process
    // attach is supported on the DRISC path), so guard the writes.
    if (connector_state_) {
        connector_state_->page_size = page_size_;
        connector_state_->fifo_curr_size = fifo_curr_size_;
        connector_state_->bytes_sent = bytes_sent_;
        connector_state_->write_ptr = write_ptr_;
    }
}

void H2DSocket::barrier(std::optional<uint32_t> timeout_ms) {
    // Re-sync bytes_sent_ from connector SHM each iteration (mirrors D2HSocket::barrier).
    auto refresh_connector_write_state = [this]() {
        if (connector_state_) {
            tt_driver_atomics::mfence();
            bytes_sent_ = connector_state_->bytes_sent;
        }
    };
    refresh_connector_write_state();
    volatile uint32_t bytes_acked_value = bytes_acked_ptr_[0];
    auto start_time = std::chrono::high_resolution_clock::now();
    while (bytes_sent_ - bytes_acked_value != 0) {
        refresh_connector_write_state();
        advance_h2d_simulator_socket_device(mesh_device_, recv_core_.device_coord);
        tt_driver_atomics::mfence();
        bytes_acked_value = bytes_acked_ptr_[0];
        if (timeout_ms.has_value()) {
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::high_resolution_clock::now() - start_time)
                                  .count();
            if (elapsed_ms > timeout_ms.value()) {
                TT_THROW(
                    "Timeout waiting for device to send acknowledgement over H2D socket. Bytes sent: {}, Bytes "
                    "acknowledged: {}",
                    bytes_sent_,
                    bytes_acked_value);
            }
        }
    }
}

void H2DSocket::write(void* data, uint32_t num_pages) {
    TT_FATAL(page_size_ > 0, "Page size must be set before writing.");
    uint32_t num_bytes = num_pages * page_size_;
    TT_FATAL(num_bytes <= fifo_curr_size_, "Cannot write more pages than the socket FIFO size.");
    auto data_addr = aligned_data_buf_start_ + write_ptr_;
    this->reserve_bytes(num_bytes);

    if (h2d_mode_ == H2DMode::HOST_PUSH) {
        pcie_writer(data, num_bytes, data_addr);
        tt_driver_atomics::sfence();
    } else {
        uint32_t* data_ptr = host_buffer_.get() + (write_ptr_ / sizeof(uint32_t));
        std::memcpy(data_ptr, data, num_bytes);
    }
    this->push_bytes(num_bytes);
    this->notify_receiver();
}

bool H2DSocket::try_write_impl(void* data, uint32_t num_pages) {
    TT_FATAL(page_size_ > 0, "Page size must be set before writing.");
    uint32_t num_bytes = num_pages * page_size_;
    TT_FATAL(num_bytes <= fifo_curr_size_, "Cannot write more pages than the socket FIFO size.");

    // Non-blocking variant of reserve_bytes(): if there's not enough room, bail.
    uint32_t bytes_free = fifo_size_ - (bytes_sent_ - bytes_acked_);
    if (bytes_free < num_bytes) {
        tt_driver_atomics::mfence();
        volatile uint32_t bytes_acked_value = bytes_acked_ptr_[0];
        bytes_acked_ = bytes_acked_value;
        bytes_free = fifo_size_ - (bytes_sent_ - bytes_acked_);
        if (bytes_free < num_bytes) {
            return false;
        }
    }

    auto data_addr = aligned_data_buf_start_ + write_ptr_;
    if (h2d_mode_ == H2DMode::HOST_PUSH) {
        pcie_writer(data, num_bytes, data_addr);
        tt_driver_atomics::sfence();
    } else {
        uint32_t* data_ptr = host_buffer_.get() + (write_ptr_ / sizeof(uint32_t));
        std::memcpy(data_ptr, data, num_bytes);
    }
    this->push_bytes(num_bytes);
    this->notify_receiver();
    return true;
}

std::vector<MeshCoreCoord> H2DSocket::get_active_cores() const { return {recv_core_}; }

MeshDevice* H2DSocket::get_mesh_device() const { return mesh_device_; }

H2DMode H2DSocket::get_h2d_mode() const { return h2d_mode_; }

HDSocketDescriptor H2DSocket::populate_descriptor() const {
    TT_FATAL(is_owner_, "Only the owner process can populate a socket descriptor.");
    TT_FATAL(shm_ && shm_->is_open(), "Cannot populate descriptor: shared memory is not initialized.");

    HDSocketDescriptor desc;
    desc.populate_from_owner("h2d", *shm_, fifo_size_, config_buffer_address_, mesh_device_, recv_core_);
    desc.bytes_acked_offset = (h2d_mode_ == H2DMode::DEVICE_PULL) ? fifo_size_ : 0;
    desc.h2d_mode = static_cast<uint32_t>(h2d_mode_);
    desc.aligned_data_buf_start = aligned_data_buf_start_;
    desc.connector_state_offset = connector_state_offset_;
    return desc;
}

std::string H2DSocket::export_descriptor(const std::string& socket_id) {
    auto desc = populate_descriptor();
    descriptor_path_ = descriptor_path_for_socket("h2d", socket_id);
    desc.write_to_file(descriptor_path_);
    ShmResourceTracker::instance().track_file(descriptor_path_);
    return descriptor_path_;
}

std::unique_ptr<H2DSocket> H2DSocket::connect(const std::string& socket_id, std::optional<uint32_t> timeout_ms) {
    auto desc = HDSocketDescriptor::wait_and_read(
        descriptor_path_for_socket("h2d", socket_id), "h2d", timeout_ms.value_or(10000));
    return connect_from_descriptor(desc);
}

std::unique_ptr<H2DSocket> H2DSocket::connect_from_descriptor(const HDSocketDescriptor& desc) {
    auto socket = std::unique_ptr<H2DSocket>(new H2DSocket());
    socket->is_owner_ = false;
    socket->fifo_size_ = desc.fifo_size;
    socket->config_buffer_address_ = desc.config_buffer_address;
    socket->pcie_alignment_ = desc.pcie_alignment;
    // Must match the owner-side coord; empty mesh_coord (pre-mesh-coord descriptors) defaults to (0, 0).
    MeshCoordinate device_coord =
        desc.mesh_coord.empty()
            ? MeshCoordinate(0, 0)
            : MeshCoordinate(ttsl::SmallVector<uint32_t>(desc.mesh_coord.begin(), desc.mesh_coord.end()));
    socket->recv_core_ = MeshCoreCoord(device_coord, CoreCoord(desc.core_x, desc.core_y));
    socket->h2d_mode_ = static_cast<H2DMode>(desc.h2d_mode);
    socket->aligned_data_buf_start_ = desc.aligned_data_buf_start;
    socket->shm_ = std::make_unique<NamedShm>(NamedShm::open(desc.shm_name, desc.shm_size));

    if (socket->h2d_mode_ == H2DMode::DEVICE_PULL) {
        socket->host_buffer_ =
            std::shared_ptr<uint32_t[]>(static_cast<uint32_t*>(socket->shm_->ptr()), [](uint32_t*) {});
        socket->bytes_acked_ptr_ =
            static_cast<uint32_t*>(socket->shm_->ptr()) + (desc.bytes_acked_offset / sizeof(uint32_t));
    } else {
        socket->bytes_acked_ptr_ = static_cast<uint32_t*>(socket->shm_->ptr());
    }

    socket->pcie_writer_instance_ =
        std::make_unique<PCIeCoreWriter>(desc.device_id, desc.virtual_core_x, desc.virtual_core_y);
    socket->pcie_writer = socket->pcie_writer_instance_->get_pcie_writer();

    // Restore connector-mutable state left behind by any prior driver process.
    // First connector after owner-init sees an all-zero struct (version stamped
    // by the owner), which matches a fresh socket.
    TT_FATAL(
        desc.connector_state_offset + sizeof(HDSocketConnectorState) <= desc.shm_size,
        "Descriptor connector_state_offset out of range for SHM size {}.",
        desc.shm_size);
    socket->connector_state_offset_ = desc.connector_state_offset;
    socket->connector_state_ = reinterpret_cast<HDSocketConnectorState*>(
        static_cast<uint8_t*>(socket->shm_->ptr()) + desc.connector_state_offset);
    TT_FATAL(
        socket->connector_state_->version == kHDSocketConnectorStateVersion,
        "HDSocketConnectorState version mismatch: got {}, expected {}.",
        socket->connector_state_->version,
        kHDSocketConnectorStateVersion);
    // Capture the prior process's clean_shutdown before overwriting it. A 0 here
    // means the previous connector exited without running its destructor (crash,
    // _exit, kill); callers can query had_clean_prior_shutdown() to react.
    socket->prior_clean_shutdown_ = (socket->connector_state_->clean_shutdown != 0);
    if (!socket->prior_clean_shutdown_) {
        log_warning(
            LogMetal,
            "H2DSocket::connect: prior connector process exited without running its destructor. State has been "
            "recovered from SHM, but downstream effects (in-flight writes, device-side counters) may need manual "
            "inspection.");
    }
    socket->connector_state_->clean_shutdown = 0;
    socket->page_size_ = socket->connector_state_->page_size;
    socket->fifo_curr_size_ = socket->connector_state_->fifo_curr_size;
    socket->bytes_sent_ = socket->connector_state_->bytes_sent;
    socket->write_ptr_ = socket->connector_state_->write_ptr;
    socket->bytes_acked_ = socket->bytes_acked_ptr_[0];

    socket->notify_receiver();

    return socket;
}

}  // namespace tt::tt_metal::distributed

namespace tt::tt_metal::experimental::detail {

bool H2DSocketTryWriteAccess::try_write(distributed::H2DSocket& socket, void* data, uint32_t num_pages) {
    return socket.try_write_impl(data, num_pages);
}

std::unique_ptr<distributed::H2DSocket> H2DSocketDramRecvAccess::create(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const distributed::MeshCoreCoord& recv_core,
    uint32_t fifo_size,
    uint32_t config_l1_local_addr,
    uint32_t data_l1_local_addr,
    uint64_t dram_l1_noc_offset) {
    return std::unique_ptr<distributed::H2DSocket>(new distributed::H2DSocket(
        mesh_device, recv_core, fifo_size, config_l1_local_addr, data_l1_local_addr, dram_l1_noc_offset));
}

}  // namespace tt::tt_metal::experimental::detail
