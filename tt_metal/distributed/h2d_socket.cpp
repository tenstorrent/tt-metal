// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include "tt_metal/distributed/h2d_socket_impl.hpp"
#include <internal/service/service_core_manager.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/distributed/mesh_socket_utils.hpp"
#include <tt-metalium/experimental/sockets/named_shm.hpp>
#include "tt_metal/distributed/hd_socket_connector_state.hpp"
#include <tt-metalium/experimental/sockets/hd_socket_descriptor.hpp>
#include "tt_metal/distributed/pcie_core_writer.hpp"
#include <tt-metalium/experimental/sockets/shm_resource_tracker.hpp>
#include "tt_metal/impl/buffers/h2d_socket_internal.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/context/metal_env_impl.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "tt_metal/hw/inc/hostdev/socket.h"
#include "tt_metal/llrt/tt_cluster.hpp"
#include "tt_metal/llrt/l2cpu_lim.hpp"  // kL2cpuLimBase / kL2cpuLimTlbEnd
#ifdef TT_METAL_USE_EMULE
#include "emulated_program_runner.hpp"  // emule::pump_device
#endif
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>
#include <tt-metalium/experimental/per_core_allocation/mesh_buffer.hpp>
#include <tt-umd/io_window/io_window.hpp>
#include <tt-logger/tt-logger.hpp>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <sys/mman.h>
#include <unistd.h>

namespace tt::tt_metal::distributed {

namespace {

// Drive the device forward one step from a host socket credit-wait loop. Simulator co-steps the umd
// clock; emule pumps the parked fiber scheduler so a kernel blocked on this host-fed socket resumes.
void advance_h2d_simulator_socket_device(MeshDevice* mesh_device, const MeshCoordinate& device_coord) {
    if (mesh_device == nullptr) {
        return;
    }

    const auto& cluster = mesh_device->impl().metal_env().get_cluster();
#ifdef TT_METAL_USE_EMULE
    if (cluster.get_target_device_type() == tt::TargetDevice::Emule) {
        tt::tt_metal::emule::pump_device();
        return;
    }
#endif
    if (cluster.get_target_device_type() != tt::TargetDevice::Simulator) {
        return;
    }

    cluster.advance_device_execution(mesh_device->get_device(device_coord)->id());
}

}  // namespace

void H2DSocketImpl::enable_mock_flow_control(const MeshDevice& mesh_device) {
    // Emule executes the receiver, so only Mock needs synthetic acknowledgements.
    if (mesh_device.impl().metal_env().get_cluster().get_target_device_type() != tt::TargetDevice::Mock) {
        return;
    }

    // Alias the acknowledgement counter to bytes_sent_ so the FIFO always reads as drained. All
    // host-side uses of bytes_acked_ptr_ are reads, and the non-movable socket keeps the pointer valid.
    bytes_acked_ptr_ = &bytes_sent_;
}

H2DSocketImpl::PinnedBufferInfo H2DSocketImpl::init_bytes_acked_buffer(
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

H2DSocketImpl::PinnedBufferInfo H2DSocketImpl::init_host_data_buffer(
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

void H2DSocketImpl::init_config_buffer(const std::shared_ptr<MeshDevice>& mesh_device) {
    validate_host_socket_allocation(*mesh_device, recv_core_);
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
    auto& svc = mesh_device->impl().metal_context().get_service_core_manager();
    auto* recv_device =
        mesh_device->is_local(recv_core_.device_coord) ? mesh_device->get_device(recv_core_.device_coord) : nullptr;
    if (recv_device && svc.claimed_cores(recv_device->id()).contains(recv_core_.core_coord)) {
        svc_config_l1_addr_ = svc.allocate_l1(recv_device, recv_core_.core_coord, config_buffer_size);
        preallocated_addr = svc_config_l1_addr_;
    }

    config_buffer_ =
        MeshBuffer::create(config_mesh_buffer_specs, config_buffer_specs, mesh_device.get(), preallocated_addr);
}

void H2DSocketImpl::init_data_buffer(const std::shared_ptr<MeshDevice>& mesh_device, uint32_t pcie_alignment) {
    if (h2d_mode_ != H2DMode::HOST_PUSH) {
        // DEVICE_PULL: data FIFO lives in pinned host memory; no device-side L1
        // allocation needed.
        write_ptr_ = 0;
        return;
    }

    auto& svc = mesh_device->impl().metal_context().get_service_core_manager();
    auto* recv_device =
        mesh_device->is_local(recv_core_.device_coord) ? mesh_device->get_device(recv_core_.device_coord) : nullptr;
    if (recv_device && svc.claimed_cores(recv_device->id()).contains(recv_core_.core_coord)) {
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

    // HYBRID mode: the FIFO is only used on the receiver core, so allocate it
    // there per-core instead of reserving fifo_size on every worker core.
    // Read the mode through the mesh's own context rather than the default one: MeshBuffer::create
    // below resolves it the same way, and the two must agree or we hand per-core sharding args to a
    // lockstep allocator.
    const bool per_core =
        buffer_type_ == BufferType::L1 && mesh_device->impl().metal_env().get_rtoptions().get_allocator_mode_hybrid();
    if (per_core) {
        shard_grid = CoreRangeSet(CoreRange(recv_core_.core_coord));
        num_data_cores = 1;
    }

    // Allocate buffer at a PCIe aligned address. This requires extra memory to be allocated.
    auto total_data_buffer_size = num_data_cores * (fifo_size_ + pcie_alignment);

    auto sharding_args = BufferShardingArgs(
        ShardSpecBuffer(shard_grid, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {num_data_cores, 1}),
        TensorMemoryLayout::HEIGHT_SHARDED);
    if (per_core) {
        experimental::per_core_allocation::set_per_core_allocation(sharding_args, true);
    }
    DeviceLocalBufferConfig data_buffer_specs = {
        .page_size = fifo_size_ + pcie_alignment,
        .buffer_type = buffer_type_,
        .sharding_args = sharding_args,
        .bottom_up = std::nullopt,
        .sub_device_id = std::nullopt,
    };
    MeshBufferConfig data_mesh_buffer_specs = ReplicatedBufferConfig{
        .size = total_data_buffer_size,
    };
    data_buffer_ = MeshBuffer::create(data_mesh_buffer_specs, data_buffer_specs, mesh_device.get());
    if (!mesh_device->is_local(recv_core_.device_coord)) {
        return;
    }
    // Per-core buffers have a real address only via the per-core API;
    // address() is not valid for them (host would push to a bogus L1 spot).
    const DeviceAddr data_buf_base = per_core
                                         ? experimental::per_core_allocation::get_per_core_address(
                                               *data_buffer_, recv_core_.device_coord, recv_core_.core_coord)
                                         : data_buffer_->address();
    aligned_data_buf_start_ = tt::align(data_buf_base, pcie_alignment);
    write_ptr_ = 0;
}

void H2DSocketImpl::write_socket_metadata(
    const std::shared_ptr<MeshDevice>& mesh_device,
    const PinnedBufferInfo& bytes_acked_info,
    const PinnedBufferInfo& data_info) {
    // The L2CPU path has no MeshBuffer-backed config_buffer_ (the caller pre-reserves a fixed LIM address and
    // init_config_buffer is never called), so config_buffer_ is null there; it still has exactly one md slot.
    const size_t num_md_slots = is_l2cpu_ ? 1u : (config_buffer_->size() / sizeof(receiver_socket_md));
    std::vector<receiver_socket_md> config_data(num_md_slots, receiver_socket_md());

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

    if (is_l2cpu_) {
        // L2CPU has no MeshBuffer-backed config and no fast-dispatch path; write the
        // struct directly to the caller-provided LIM address.
        const auto& cluster = mesh_device->impl().metal_env().get_cluster();
        const uint32_t device_id = mesh_device->get_device(recv_core_.device_coord)->id();
        cluster.write_core(&md, sizeof(md), tt_cxy_pair(device_id, recv_core_.core_coord), config_buffer_address_);
        return;
    }

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

void H2DSocketImpl::init_receiver_tlb(const std::shared_ptr<MeshDevice>& mesh_device) {
    TT_FATAL(mesh_device, "init_receiver_tlb requires a MeshDevice (owner path only; connectors use PCIeCoreWriter).");

    auto& env = mesh_device->impl().metal_env();
    auto& cluster = env.get_cluster();

    // Mock/emulated chips have no device to map, so they skip the window creation below (guarded by
    // !is_mock_or_emulated()) and fall through to the cluster.write_core() dynamic writer.
    // SWEmuleChip backs that with real memory-backed I/O; MockChip never invokes pcie_writer at
    // runtime (only socket construction / JIT), so the installed writer is harmless there.

    // Receiver core type is recorded explicitly at construction (the DRAM-recv
    // ctor sets Dram, every other path is Tensix). Used to resolve the virtual
    // coordinate — logical coords overlap across core types, so this can't be
    // inferred from coordinates here — and to pick the window's ordering below.
    const CoreType recv_umd_core_type = (recv_core_type_ == RecvCoreType::Dram) ? CoreType::DRAM : CoreType::TENSIX;

    uint32_t recv_device_id;
    CoreCoord recv_virtual_core;
    if (is_l2cpu_) {
        // recv_core_.core_coord is already a TRANSLATED L2CPU NOC coord, so no
        // logical->virtual translation is applied. The window is anchored at the LIM
        // base rather than at 0, because LIM does not start at 0.
        recv_device_id = mesh_device->get_device(recv_core_.device_coord)->id();
        recv_virtual_core = recv_core_.core_coord;
        if (!cluster.is_mock_or_emulated()) {
            receiver_core_window_ = cluster.get_driver()->create_io_window(
                recv_device_id,
                cluster.get_soc_desc(recv_device_id).get_coord_at(recv_virtual_core, tt::CoordSystem::TRANSLATED),
                ll_api::kL2cpuLimBase,
                {.size = ll_api::kL2cpuLimTlbSize});
        }
    } else {
        // Per-device translation (see metal_SocDescriptor::dram_bank_endpoint_coords): the
        // mesh-level translation validates that every device agrees and throws when they do not,
        // which a logical DRAM coord on a harvested mesh does not.
        IDevice* recv_device = mesh_device->get_device(recv_core_.device_coord);
        recv_device_id = recv_device->id();
        recv_virtual_core = recv_device->virtual_core_from_logical_core(recv_core_.core_coord, recv_umd_core_type);
    }

    // For DRAM-core recv, every host NOC write to the DRISC L1 needs the DRAM-L1
    // NOC offset added on top of the local L1 address (DRAM cores have two NOC
    // spaces: low addresses route to DRAM bank, high addresses route to L1).
    // Captured into the lambdas below so write() can keep passing local addresses.
    const uint64_t l1_offset = dram_l1_noc_offset_;

    if (is_l2cpu_ && !cluster.is_mock_or_emulated()) {
        // The L2CPU window is anchored at the LIM base, so absolute addresses are
        // converted to window-relative offsets before write_block(). Mock/emule
        // create no window and fall through to the write_core() writer below.
        const uint64_t l2cpu_window_base = receiver_core_window_->get_target_config().addr;
        pcie_writer = [this, l2cpu_window_base](void* data, uint32_t num_bytes, uint64_t device_addr) {
            receiver_core_window_->write_block(device_addr - l2cpu_window_base, data, num_bytes);
        };
        return;
    }

    // Write through a window of our own only when our actual write target lives inside one — ask for
    // the window and test it rather than assuming based on core type. The window is anchored where
    // the receiver's L1 starts in its NOC address space: 0 for a Tensix/Eth core, l1_offset (a high
    // DRAM-L1 NOC address, e.g. 0x2000000000) for a DRAM core. Writes through it are plain stores;
    // cluster.write_core may reconfigure a driver window on every write, which costs the Tensor
    // prefetcher's request sockets microseconds per page. Gated on a real device and Blackhole — on
    // Wormhole B0 the device address space isn't fully mapped this way and a write may still need a
    // per-write driver reconfig. The window holds one of the device's TLBs until the socket is destroyed;
    // when none is free, the socket writes through cluster.write_core, which holds none.
    //
    // A DRAM core acknowledges a NOC write only once its L1 has taken the write over AXI, and a Strict
    // window holds each write until the one before it is acknowledged, so it moves one write per round
    // trip (about 0.4 us per 128 B page). A DRAM receiver's window posts its writes instead, on a static
    // write VC: a NOC packet on a dynamic VC may switch VCs at each router, and a DRAM core gives each VC
    // its own AXI ID, so a bytes_sent update could overtake the pages it counts; on one VC, writes to the
    // receiver arrive in the order they were made.
    const bool dram_receiver = recv_core_type_ == RecvCoreType::Dram;
    if (!cluster.is_mock_or_emulated() && env.get_hal().get_arch() == tt::ARCH::BLACKHOLE) {
        std::unique_ptr<tt::umd::IoWindow> window;
        try {
            window = cluster.get_driver()->create_io_window(
                recv_device_id,
                cluster.get_soc_desc(recv_device_id).get_coord_at(recv_virtual_core, tt::CoordSystem::TRANSLATED),
                /*addr=*/l1_offset,
                /*host=*/{},
                dram_receiver ? tt::umd::IoOrdering::Posted : tt::umd::IoOrdering::Strict,
                /*core_end=*/std::nullopt,
                dram_receiver ? tt::umd::WindowFlags::UnicastWrite : tt::umd::WindowFlags::None);
        } catch (const std::runtime_error& e) {
            log_warning(
                LogMetal,
                "H2DSocket: no TLB window for receiver core {} on device {}; its writes go through the driver, "
                "which is slower: {}",
                recv_virtual_core.str(),
                recv_device_id,
                e.what());
        }
        // The window is addressed from its anchor, so a local L1 address is its offset into the window.
        const uint64_t fifo_end = static_cast<uint64_t>(aligned_data_buf_start_) + fifo_size_;
        const uint64_t config_end = static_cast<uint64_t>(config_buffer_address_) + sizeof(receiver_socket_md);
        if (window != nullptr && std::max(fifo_end, config_end) <= window->get_size()) {
            receiver_core_window_ = std::move(window);
        }
    }

    if (receiver_core_window_ != nullptr) {
        pcie_writer = [this](void* data, uint32_t num_bytes, uint64_t device_addr) {
            receiver_core_window_->write_block(device_addr, data, num_bytes);
        };
    } else {
        // Non-Blackhole, no TLB free, or no window reaches the target: write through UMD instead
        // (the driver may reconfigure a window per write). Covers Wormhole B0, mock and
        // emulated devices.
        pcie_writer = [&cluster, recv_device_id, recv_virtual_core, l1_offset](
                          void* data, uint32_t num_bytes, uint64_t device_addr) {
            cluster.write_core(
                data, num_bytes, tt_cxy_pair(recv_device_id, recv_virtual_core), device_addr + l1_offset);
        };
    }
}

H2DSocketImpl::H2DSocketImpl(
    const std::shared_ptr<MeshDevice>& mesh_device,
    const MeshCoreCoord& recv_core,
    BufferType buffer_type,
    uint32_t fifo_size,
    H2DMode h2d_mode) :
    recv_core_(recv_core),
    buffer_type_(buffer_type),
    fifo_size_(fifo_size),
    pcie_alignment_(mesh_device->impl().metal_env().get_hal().get_alignment(HalMemType::HOST)),
    pinned_memory_(nullptr),
    h2d_mode_(h2d_mode),
    mesh_device_(mesh_device.get()),
    rank_owns_endpoint_(rank_owns_host_socket_endpoint(mesh_device.get(), recv_core)) {
    MeshCoordinateRangeSet recv_device_range_set;
    recv_device_range_set.merge(MeshCoordinateRange(recv_core_.device_coord));

    const uint32_t pcie_alignment = pcie_alignment_;
    TT_FATAL(fifo_size_ % pcie_alignment == 0, "FIFO size must be PCIE-aligned.");
    TT_FATAL(buffer_type_ == BufferType::L1, "H2D sockets currently only support data buffers in SRAM.");

    // Preserve collective L1 allocation order on all co-owners before owner-local PCIe setup.
    init_config_buffer(mesh_device);
    init_data_buffer(mesh_device, pcie_alignment);
    config_buffer_address_ = config_buffer_->address();
    if (!rank_owns_endpoint_) {
        return;
    }

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
    enable_mock_flow_control(*mesh_device);

    write_socket_metadata(mesh_device, bytes_acked_info, data_info);
    init_receiver_tlb(mesh_device);

    // Initialize the persistent connector-state struct living in SHM.
    // NamedShm::create zero-initialized the region; we stamp the version and
    // mark clean_shutdown=1 so the first connect() sees "no prior crash" (the
    // owner side has nothing to recover from).
    connector_state_ =
        reinterpret_cast<HDSocketConnectorState*>(static_cast<uint8_t*>(shm_->ptr()) + connector_state_offset_);
    connector_state_->version = kHDSocketConnectorStateVersion;
    connector_state_->clean_shutdown = 1;
}

H2DSocketImpl::H2DSocketImpl(
    const std::shared_ptr<MeshDevice>& mesh_device,
    const MeshCoreCoord& recv_core,
    uint32_t fifo_size,
    uint32_t config_l1_local_addr,
    uint32_t data_l1_local_addr,
    uint64_t dram_l1_noc_offset) :
    recv_core_(recv_core),
    fifo_size_(fifo_size),
    pcie_alignment_(mesh_device->impl().metal_env().get_hal().get_alignment(HalMemType::HOST)),
    pinned_memory_(nullptr),
    mesh_device_(mesh_device.get()),
    rank_owns_endpoint_(rank_owns_host_socket_endpoint(mesh_device.get(), recv_core)),
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
    enable_mock_flow_control(*mesh_device);

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
    mesh_device->impl().metal_env().get_cluster().write_core(
        mesh_device->get_device(recv_core_.device_coord)->id(),
        tt_cxy_pair(mesh_device->get_device(recv_core_.device_coord)->id(), virtual_core),
        std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(&md), sizeof(md)),
        static_cast<uint64_t>(config_buffer_address_) + dram_l1_noc_offset_);
}

H2DSocketImpl::H2DSocketImpl(
    MeshDevice& mesh_device,
    const MeshCoreCoord& recv_l2cpu,
    uint32_t fifo_size,
    uint32_t config_buffer_address,
    uint32_t data_fifo_address,
    H2DMode h2d_mode) :
    recv_core_(recv_l2cpu),
    fifo_size_(fifo_size),
    pcie_alignment_(mesh_device.impl().metal_env().get_hal().get_alignment(HalMemType::HOST)),
    pinned_memory_(nullptr),
    h2d_mode_(h2d_mode),
    mesh_device_(&mesh_device),
    rank_owns_endpoint_(rank_owns_host_socket_endpoint(&mesh_device, recv_l2cpu)),
    is_l2cpu_(true) {
    // Helpers below still take a shared_ptr; the socket itself stores only a raw
    // MeshDevice* and does not extend the device's lifetime.
    const auto mesh_device_ptr = mesh_device.shared_from_this();
    MeshCoordinateRangeSet recv_device_range_set;
    recv_device_range_set.merge(MeshCoordinateRange(recv_core_.device_coord));

    const uint32_t pcie_alignment = pcie_alignment_;
    TT_FATAL(
        mesh_device.impl().metal_env().get_hal().get_arch() == tt::ARCH::BLACKHOLE,
        "L2CPU H2D sockets are only supported on Blackhole architectures.");
    TT_FATAL(fifo_size_ > 0 && fifo_size_ % pcie_alignment == 0, "FIFO size must be non-zero and PCIe-aligned.");
    TT_FATAL(config_buffer_address != 0, "L2CPU config buffer LIM address must be non-zero.");
    TT_FATAL(data_fifo_address != 0, "L2CPU data FIFO LIM address must be non-zero.");
    TT_FATAL(
        data_fifo_address % pcie_alignment == 0,
        "L2CPU data FIFO LIM address 0x{:x} must be PCIe-aligned ({} B).",
        data_fifo_address,
        pcie_alignment);
    TT_FATAL(
        config_buffer_address % pcie_alignment == 0,
        "L2CPU config buffer LIM address 0x{:x} must be PCIe-aligned ({} B).",
        config_buffer_address,
        pcie_alignment);

    // The caller-supplied coordinate is taken as an already-TRANSLATED L2CPU NOC
    // coord, so a wrong value would silently write the socket blob to another
    // core and pick up that core's TLB base. Check membership rather than trust it.
    {
        const auto& cluster = mesh_device.impl().metal_env().get_cluster();
        const uint32_t device_id = mesh_device_ptr->get_device(recv_core_.device_coord)->id();
        const auto l2cpu_cores =
            cluster.get_soc_desc(device_id).get_cores(tt::CoreType::L2CPU, tt::CoordSystem::TRANSLATED);
        const bool is_l2cpu_core = std::any_of(l2cpu_cores.begin(), l2cpu_cores.end(), [this](const auto& c) {
            return c.x == recv_core_.core_coord.x && c.y == recv_core_.core_coord.y;
        });
        TT_FATAL(
            is_l2cpu_core,
            "({}, {}) is not an L2CPU tile on device {}. recv_l2cpu.core_coord must be the TRANSLATED NOC coord of "
            "an L2CPU tile.",
            recv_core_.core_coord.x,
            recv_core_.core_coord.y,
            device_id);
    }

    // The receiver_socket_md is written through the L2CPU IoWindow, and
    // notify_receiver() later routes config_buffer_address_ + bytes_sent through
    // the same window (subtracting its base). An address outside the window is
    // accepted by the alignment checks above but underflows or trips the window's
    // bounds check on the first notification, so reject it up front.
    const uint64_t config_end = static_cast<uint64_t>(config_buffer_address) + sizeof(receiver_socket_md);
    TT_FATAL(
        config_buffer_address >= ll_api::kL2cpuLimBase && config_end <= ll_api::kL2cpuLimTlbEnd,
        "L2CPU H2D config buffer [0x{:x}, 0x{:x}) must lie inside the IoWindow [0x{:x}, 0x{:x}).",
        config_buffer_address,
        config_end,
        ll_api::kL2cpuLimBase,
        ll_api::kL2cpuLimTlbEnd);

    // HOST_PUSH only: the H2D FIFO writes are issued through the same window via
    // pcie_writer, so the FIFO must end inside it. Going past that boundary would
    // surface later as the window throwing "Out of bounds access" on the first
    // wrapping write -- catch it here instead.
    //
    // DEVICE_PULL keeps the ring in pinned host memory, so it is not bounded by
    // the window and data_fifo_address is only a base for ring offsets.
    if (h2d_mode_ == H2DMode::HOST_PUSH) {
        const uint64_t fifo_end = static_cast<uint64_t>(data_fifo_address) + static_cast<uint64_t>(fifo_size_);
        TT_FATAL(
            fifo_end <= ll_api::kL2cpuLimTlbEnd,
            "L2CPU H2D data FIFO [0x{:x}, 0x{:x}) does not fit in the {} MiB LIM "
            "window [0x{:x}, 0x{:x}). Reduce fifo_size or move data_fifo_address "
            "earlier in LIM. (HOST_PUSH only; DEVICE_PULL has no such limit.)",
            data_fifo_address,
            fifo_end,
            ll_api::kL2cpuLimTlbSize / (1024 * 1024),
            ll_api::kL2cpuLimBase,
            ll_api::kL2cpuLimTlbEnd);
        TT_FATAL(
            data_fifo_address >= ll_api::kL2cpuLimBase,
            "L2CPU H2D data FIFO address 0x{:x} must lie inside the LIM region "
            "[0x{:x}, 0x{:x}) the window covers.",
            data_fifo_address,
            ll_api::kL2cpuLimBase,
            ll_api::kL2cpuLimTlbEnd);
        // In HOST_PUSH the payload ring lives in LIM alongside the metadata, so an
        // overlapping reservation would let advancing writes scribble over
        // receiver_socket_md and corrupt the socket's own counters.
        TT_FATAL(
            fifo_end <= static_cast<uint64_t>(config_buffer_address) || data_fifo_address >= config_end,
            "L2CPU H2D data FIFO [0x{:x}, 0x{:x}) overlaps the config buffer [0x{:x}, 0x{:x}); the caller-reserved "
            "LIM regions must be disjoint.",
            data_fifo_address,
            fifo_end,
            config_buffer_address,
            config_end);
    }

    // Cache the externally-supplied LIM addresses; the existing
    // write_socket_metadata path uses these via is_l2cpu_ instead of
    // pulling them out of MeshBuffer.
    config_buffer_address_ = config_buffer_address;
    aligned_data_buf_start_ = data_fifo_address;
    write_ptr_ = 0;

    // Pinned host RAM layout mirrors the Tensix paths: HOST_PUSH only
    // needs the 4 B bytes_acked counter (data lives in LIM), DEVICE_PULL
    // additionally needs the data FIFO co-located with bytes_acked so the device
    // can pull from a single contiguous PCIe range.
    std::string shm_name = generate_shm_name(h2d_mode_ == H2DMode::DEVICE_PULL ? "h2d-l2cpu-pull" : "h2d-l2cpu-push");
    PinnedBufferInfo bytes_acked_info{};
    PinnedBufferInfo data_info{};
    if (h2d_mode_ == H2DMode::DEVICE_PULL) {
        // Same layout as the Tensix DEVICE_PULL path: [data ring | bytes_acked]
        // contiguous in one PinnedMemory, bytes_acked at offset fifo_size_.
        data_info = init_host_data_buffer(mesh_device_ptr, recv_device_range_set, pcie_alignment, shm_name);
        bytes_acked_info = data_info;
        auto bytes_acked_addr = (static_cast<uint64_t>(data_info.addr_hi) << 32 | data_info.addr_lo) + fifo_size_;
        bytes_acked_info.addr_hi = static_cast<uint32_t>(bytes_acked_addr >> 32);
        bytes_acked_info.addr_lo = static_cast<uint32_t>(bytes_acked_addr & 0xFFFFFFFFull);
        bytes_acked_ptr_ = host_buffer_.get() + (fifo_size_ / sizeof(uint32_t));
        TT_FATAL(
            bytes_acked_info.pcie_xy_enc == data_info.pcie_xy_enc,
            "L2CPU DEVICE_PULL: bytes_acked and data pinned memory must be mapped to the same PCIe core.");
    } else {
        // HOST_PUSH allocates only bytes_acked; data_info stays zeroed because
        // the ring lives in LIM at aligned_data_buf_start_.
        bytes_acked_info = init_bytes_acked_buffer(mesh_device_ptr, recv_device_range_set, pcie_alignment, shm_name);
        bytes_acked_ptr_ = host_buffer_.get();
    }
    write_socket_metadata(mesh_device_ptr, bytes_acked_info, data_info);
    init_receiver_tlb(mesh_device_ptr);

    // Stamp the connector-state header so it is well-formed rather than zeroed.
    connector_state_ =
        reinterpret_cast<HDSocketConnectorState*>(static_cast<uint8_t*>(shm_->ptr()) + connector_state_offset_);
    connector_state_->version = kHDSocketConnectorStateVersion;
    connector_state_->clean_shutdown = 1;
}

H2DSocketImpl::~H2DSocketImpl() noexcept {
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
            auto& svc = mesh_device_->impl().metal_context().get_service_core_manager();
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

void H2DSocketImpl::reserve_bytes(uint32_t num_bytes) {
    uint32_t bytes_free = fifo_size_ - (bytes_sent_ - bytes_acked_);
    while (bytes_free < num_bytes) {
        advance_h2d_simulator_socket_device(mesh_device_, recv_core_.device_coord);
        tt_driver_atomics::mfence();
        volatile uint32_t bytes_acked_value = bytes_acked_ptr_[0];
        bytes_free = fifo_size_ - (bytes_sent_ - bytes_acked_value);
        bytes_acked_ = bytes_acked_value;
    }
}

bool H2DSocketImpl::has_space(std::optional<uint32_t> num_bytes_to_check) {
    validate_host_socket_access(rank_owns_endpoint_, recv_core_);
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

bool H2DSocketImpl::acked_past(uint32_t watermark) {
    validate_host_socket_access(rank_owns_endpoint_, recv_core_);
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

void H2DSocketImpl::push_bytes(uint32_t num_bytes) {
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

void H2DSocketImpl::notify_receiver_unfenced() {
    uint32_t bytes_sent_addr = config_buffer_address_ + offsetof(receiver_socket_md, bytes_sent);
    pcie_writer(&bytes_sent_, sizeof(bytes_sent_), bytes_sent_addr);
}

void H2DSocketImpl::notify_receiver() {
    notify_receiver_unfenced();
    tt_driver_atomics::sfence();
}

void H2DSocketImpl::set_page_size(uint32_t page_size) {
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

void H2DSocketImpl::barrier(std::optional<uint32_t> timeout_ms) {
    if (mesh_device_ && !mesh_device_->is_local(recv_core_.device_coord)) {
        return;
    }
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

void H2DSocketImpl::write(void* data, uint32_t num_pages) {
    validate_host_socket_access(rank_owns_endpoint_, recv_core_);
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

bool H2DSocketImpl::try_write(void* data, uint32_t num_pages) {
    validate_host_socket_access(rank_owns_endpoint_, recv_core_);
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
            advance_h2d_simulator_socket_device(mesh_device_, recv_core_.device_coord);
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

uint32_t H2DSocketImpl::try_stage(const uint8_t* const* pages, uint32_t num_pages) {
    validate_host_socket_access(rank_owns_endpoint_, recv_core_);
    TT_FATAL(page_size_ > 0, "Page size must be set before writing.");
    if ((fifo_size_ - (bytes_sent_ - bytes_acked_)) / page_size_ < num_pages) {
        tt_driver_atomics::mfence();
        volatile uint32_t bytes_acked_value = bytes_acked_ptr_[0];
        bytes_acked_ = bytes_acked_value;
    }
    uint32_t staged = 0;
    while (staged < num_pages) {
        // One write per run of pages that fits in the FIFO before it wraps.
        const uint32_t bytes_free = fifo_size_ - (bytes_sent_ - bytes_acked_);
        const uint32_t pages_before_wrap = (fifo_curr_size_ - write_ptr_) / page_size_;
        const uint32_t run = std::min({num_pages - staged, bytes_free / page_size_, pages_before_wrap});
        if (run == 0) {
            break;
        }
        const uint32_t num_bytes = run * page_size_;
        const auto gather = [&](uint8_t* dst) {
            for (uint32_t i = 0; i < run; ++i) {
                std::memcpy(dst + static_cast<size_t>(i) * page_size_, pages[staged + i], page_size_);
            }
        };
        if (h2d_mode_ == H2DMode::HOST_PUSH) {
            stage_buffer_.resize(num_bytes);
            gather(stage_buffer_.data());
            pcie_writer(stage_buffer_.data(), num_bytes, aligned_data_buf_start_ + write_ptr_);
        } else {
            gather(reinterpret_cast<uint8_t*>(host_buffer_.get()) + write_ptr_);
        }
        this->push_bytes(num_bytes);
        staged += run;
    }
    if (staged == 0) {
        advance_h2d_simulator_socket_device(mesh_device_, recv_core_.device_coord);
    }
    return staged;
}

HDSocketDescriptor H2DSocketImpl::populate_descriptor() const {
    validate_host_socket_access(rank_owns_endpoint_, recv_core_);
    TT_FATAL(is_owner_, "Only the owner process can populate a socket descriptor.");
    // The descriptor schema has no fields for the L2CPU LIM addresses, so a connector
    // could not rebuild the device side. Checked here rather than in export_descriptor()
    // so the H2DSocketService aggregation path is covered too.
    TT_FATAL(!is_l2cpu_, "Descriptor export is not supported for L2CPU H2D sockets.");
    TT_FATAL(shm_ && shm_->is_open(), "Cannot populate descriptor: shared memory is not initialized.");

    HDSocketDescriptor desc;
    desc.populate_from_owner("h2d", *shm_, fifo_size_, config_buffer_address_, mesh_device_, recv_core_);
    desc.bytes_acked_offset = (h2d_mode_ == H2DMode::DEVICE_PULL) ? fifo_size_ : 0;
    desc.h2d_mode = static_cast<uint32_t>(h2d_mode_);
    desc.aligned_data_buf_start = aligned_data_buf_start_;
    desc.connector_state_offset = connector_state_offset_;
    return desc;
}

std::string H2DSocketImpl::export_descriptor(const std::string& socket_id) {
    auto desc = populate_descriptor();
    descriptor_path_ = descriptor_path_for_socket("h2d", socket_id);
    desc.write_to_file(descriptor_path_);
    ShmResourceTracker::instance().track_file(descriptor_path_);
    return descriptor_path_;
}

std::unique_ptr<H2DSocketImpl> H2DSocketImpl::connect_from_descriptor(const HDSocketDescriptor& desc) {
    auto socket = std::unique_ptr<H2DSocketImpl>(new H2DSocketImpl());
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

// ---------------------------------------------------------------------------
// H2DSocket: thin handle forwarding to H2DSocketImpl.
// ---------------------------------------------------------------------------

H2DSocket::H2DSocket(std::unique_ptr<H2DSocketImpl> impl) : pimpl_(std::move(impl)) {}

H2DSocket::H2DSocket(
    const std::shared_ptr<MeshDevice>& mesh_device,
    const MeshCoreCoord& recv_core,
    BufferType buffer_type,
    uint32_t fifo_size,
    H2DMode h2d_mode) :
    pimpl_(std::make_unique<H2DSocketImpl>(mesh_device, recv_core, buffer_type, fifo_size, h2d_mode)) {}

H2DSocket::H2DSocket(
    MeshDevice& mesh_device,
    const MeshCoreCoord& recv_l2cpu,
    uint32_t fifo_size,
    uint32_t config_buffer_address,
    uint32_t data_fifo_address,
    H2DMode h2d_mode) :
    pimpl_(std::make_unique<H2DSocketImpl>(
        mesh_device, recv_l2cpu, fifo_size, config_buffer_address, data_fifo_address, h2d_mode)) {}

H2DSocket::H2DSocket(
    const std::shared_ptr<MeshDevice>& mesh_device,
    const MeshCoreCoord& recv_core,
    uint32_t fifo_size,
    uint32_t config_l1_local_addr,
    uint32_t data_l1_local_addr,
    uint64_t dram_l1_noc_offset) :
    pimpl_(std::make_unique<H2DSocketImpl>(
        mesh_device, recv_core, fifo_size, config_l1_local_addr, data_l1_local_addr, dram_l1_noc_offset)) {}

// Out of line so the unique_ptr<H2DSocketImpl> destructor sees the complete type.
H2DSocket::~H2DSocket() noexcept = default;

std::unique_ptr<H2DSocket> H2DSocket::connect(const std::string& socket_id, std::optional<uint32_t> timeout_ms) {
    auto desc = HDSocketDescriptor::wait_and_read(
        descriptor_path_for_socket("h2d", socket_id), "h2d", timeout_ms.value_or(10000));
    return connect_from_descriptor(desc);
}

std::unique_ptr<H2DSocket> H2DSocket::connect_from_descriptor(const HDSocketDescriptor& desc) {
    return std::unique_ptr<H2DSocket>(new H2DSocket(H2DSocketImpl::connect_from_descriptor(desc)));
}

std::string H2DSocket::export_descriptor(const std::string& socket_id) { return pimpl_->export_descriptor(socket_id); }

HDSocketDescriptor H2DSocket::populate_descriptor() const { return pimpl_->populate_descriptor(); }

uint32_t H2DSocket::get_page_size() const { return pimpl_->get_page_size(); }

uint32_t H2DSocket::get_config_buffer_address() const { return pimpl_->get_config_buffer_address(); }

uint32_t H2DSocket::get_fifo_size() const { return pimpl_->get_fifo_size(); }

uint32_t H2DSocket::get_bytes_sent() const { return pimpl_->get_bytes_sent(); }

bool H2DSocket::had_clean_prior_shutdown() const { return pimpl_->had_clean_prior_shutdown(); }

bool H2DSocket::has_space(std::optional<uint32_t> num_bytes_to_check) { return pimpl_->has_space(num_bytes_to_check); }

bool H2DSocket::acked_past(uint32_t watermark) { return pimpl_->acked_past(watermark); }

void H2DSocket::set_page_size(uint32_t page_size) { pimpl_->set_page_size(page_size); }

void H2DSocket::write(void* data, uint32_t num_pages) { pimpl_->write(data, num_pages); }

bool H2DSocket::try_write_impl(void* data, uint32_t num_pages) { return pimpl_->try_write(data, num_pages); }

uint32_t H2DSocket::try_stage_impl(const uint8_t* const* pages, uint32_t num_pages) {
    return pimpl_->try_stage(pages, num_pages);
}

void H2DSocket::notify_receiver_unfenced() { pimpl_->notify_receiver_unfenced(); }

void H2DSocket::barrier(std::optional<uint32_t> timeout_ms) { pimpl_->barrier(timeout_ms); }

std::vector<MeshCoreCoord> H2DSocket::get_active_cores() const { return {pimpl_->get_recv_core()}; }

MeshDevice* H2DSocket::get_mesh_device() const { return pimpl_->get_mesh_device(); }

H2DMode H2DSocket::get_h2d_mode() const { return pimpl_->get_h2d_mode(); }

}  // namespace tt::tt_metal::distributed

namespace tt::tt_metal::experimental::detail {

bool H2DSocketTryWriteAccess::try_write(distributed::H2DSocket& socket, void* data, uint32_t num_pages) {
    return socket.try_write_impl(data, num_pages);
}

uint32_t H2DSocketTryWriteAccess::try_stage(distributed::H2DSocket& socket, ttsl::Span<const uint8_t* const> pages) {
    return socket.try_stage_impl(pages.data(), static_cast<uint32_t>(pages.size()));
}

void H2DSocketTryWriteAccess::publish_staged(ttsl::Span<distributed::H2DSocket* const> sockets) {
    // The pages land before any receiver can see them counted.
    tt_driver_atomics::sfence();
    for (distributed::H2DSocket* socket : sockets) {
        socket->notify_receiver_unfenced();
    }
    tt_driver_atomics::sfence();
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
