// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include "tt_metal/distributed/mesh_socket_utils.hpp"
#include "tt_metal/distributed/named_shm.hpp"
#include "tt_metal/distributed/hd_socket_connector_state.hpp"
#include "tt_metal/distributed/hd_socket_descriptor.hpp"
#include "tt_metal/distributed/pcie_core_writer.hpp"
#include "tt_metal/distributed/shm_resource_tracker.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_metal/hw/inc/hostdev/socket.h"
#include "tt_metal/llrt/tt_cluster.hpp"
#ifdef TT_METAL_USE_EMULE
#include "tt_metal/impl/emulation/emulated_program_runner.hpp"  // emule::pump_device (host-interleaved socket)
#endif
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-logger/tt-logger.hpp>
#include "impl/dispatch/system_memory_manager.hpp"
#include <umd/device/chip_helpers/tlb_manager.hpp>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#else
// The hugepage D2H path requires explicit cache-line eviction (_mm_clflush + _mm_lfence)
// because device PCIe writes may be non-snooped on WH.  This is x86-specific.
// init_host_buffer_hugepage() will TT_FATAL before any of these are reached on non-x86;
// stubs exist solely to allow the translation unit to compile.
static inline void _mm_clflush(const void*) noexcept {
    TT_THROW("D2H hugepage cache flush is x86-only and should never be reached on this architecture");
}
static inline void _mm_lfence() noexcept {
    TT_THROW("D2H hugepage cache flush is x86-only and should never be reached on this architecture");
}
#endif

namespace tt::tt_metal::distributed {

namespace {

// `_mm_clflush` invalidates one host cache line; 64 B is the line size on typical x86-64.
constexpr uint32_t k_x86_clflush_line_bytes = 64;

// Drive the device forward one step from a host socket credit-wait loop. Simulator co-steps the umd
// clock; emule pumps the parked fiber scheduler so a kernel blocked on this host-fed socket resumes.
void advance_d2h_simulator_socket_device(MeshDevice* mesh_device, const MeshCoordinate& device_coord) {
    if (mesh_device == nullptr) {
        return;
    }

    const auto& cluster = MetalContext::instance().get_cluster();
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

D2HSocket::PinnedBufferInfo D2HSocket::init_host_buffer(
    const std::shared_ptr<MeshDevice>& mesh_device,
    const MeshCoordinateRangeSet& device_range,
    uint32_t pcie_alignment,
    const std::string& shm_name) {
    // Buffer layout: [data_region (fifo_size bytes)][bytes_sent (4 bytes)][HDSocketConnectorState]
    uint32_t total_buffer_size_bytes = fifo_size_ + sizeof(uint32_t);
    uint32_t total_buffer_size_words = total_buffer_size_bytes / sizeof(uint32_t);
    size_t page_size = sysconf(_SC_PAGESIZE);
    // Reserve room for HDSocketConnectorState past the pinned region, aligned up
    // to its own alignment requirement so the reinterpret_cast<> in connect() is
    // well-defined. The pinned HostBuffer view below still spans only
    // [data | bytes_sent], so the device never touches the state struct.
    connector_state_offset_ = align(total_buffer_size_bytes, alignof(HDSocketConnectorState));
    size_t alloc_size = align(connector_state_offset_ + sizeof(HDSocketConnectorState), page_size);

    shm_ = std::make_unique<NamedShm>(NamedShm::create(shm_name, alloc_size));
    void* aligned_ptr = shm_->ptr();
    TT_FATAL(
        reinterpret_cast<uintptr_t>(aligned_ptr) % pcie_alignment == 0,
        "System Memory Allocation Error: D2H socket buffer must be aligned to the PCIe alignment.");
    // NamedShm::create zero-initializes the region; no explicit memset needed.
    host_buffer_ = std::shared_ptr<uint32_t[]>(static_cast<uint32_t*>(aligned_ptr), [](uint32_t*) {});
    bytes_sent_ptr_ = host_buffer_.get() + (fifo_size_ / sizeof(uint32_t));

    tt::tt_metal::HostBuffer host_buffer_view(
        tt::stl::Span<uint32_t>(host_buffer_.get(), total_buffer_size_words), tt::tt_metal::MemoryPin(host_buffer_));
    pinned_memory_ =
        tt::tt_metal::experimental::PinnedMemory::Create(*mesh_device, device_range, host_buffer_view, true);

    const auto& noc_addr = pinned_memory_->get_noc_addr(mesh_device->get_device(sender_core_.device_coord)->id());
    TT_FATAL(noc_addr.has_value(), "Failed to get NOC address for D2H socket pinned memory.");
    TT_FATAL(
        noc_addr.value().device_id == mesh_device->get_device(sender_core_.device_coord)->id(),
        "Pinned Memory used for D2H sockets must be mapped to the same device as the sender core. D2H Sockets cannot "
        "communicate with remote devices.");

    return PinnedBufferInfo{
        .pcie_xy_enc = noc_addr.value().pcie_xy_enc,
        .addr_lo = static_cast<uint32_t>(noc_addr.value().addr & 0xFFFFFFFFull),
        .addr_hi = static_cast<uint32_t>(noc_addr.value().addr >> 32)};
}

D2HSocket::PinnedBufferInfo D2HSocket::init_host_buffer_hugepage(const std::shared_ptr<MeshDevice>& mesh_device) {
#if !defined(__x86_64__) && !defined(__i386__)
    // Cache management for WB + non-snooped PCIe DMA is x86-specific (clflush + lfence).
    // WH — the only architecture that takes this hugepage path — is x86-only, so this
    // should never be reachable on other architectures.
    TT_FATAL(false, "D2H hugepage path is not supported on non-x86 architectures");
    return {};
#endif
    using_hugepage_ = true;

    auto* device = mesh_device->get_device(sender_core_.device_coord);
    auto device_id = device->id();
    auto& sysmem_mgr = device->sysmem_manager();

    auto [data_host_ptr, data_dev_addr] = sysmem_mgr.allocate_region(fifo_size_);
    hugepage_data_host_ptr_ = static_cast<uint32_t*>(data_host_ptr);
    std::memset(hugepage_data_host_ptr_, 0, fifo_size_);

    const auto& cluster = MetalContext::instance().get_cluster();
    const auto& hal = MetalContext::instance().hal();
    ChipId mmio_device_id = cluster.get_associated_mmio_device(device_id);
    const auto& soc = cluster.get_soc_desc(mmio_device_id);
    const auto& pcie_cores = soc.get_cores(CoreType::PCIE, CoordSystem::NOC0);
    TT_ASSERT(!pcie_cores.empty());
    auto pcie_xy = pcie_cores.front();
    uint32_t pcie_xy_enc = hal.noc_xy_pcie64_encoding(pcie_xy.x, pcie_xy.y);

    log_info(
        tt::LogMetal,
        "D2HSocket: Using hugepage fallback for device {} "
        "(data_dev_addr=0x{:x}, pcie_xy_enc=0x{:x})",
        device_id,
        data_dev_addr,
        pcie_xy_enc);

    return PinnedBufferInfo{.pcie_xy_enc = pcie_xy_enc, .addr_lo = data_dev_addr, .addr_hi = 0};
}

void D2HSocket::init_config_buffer(const std::shared_ptr<MeshDevice>& mesh_device) {
    const SocketSenderSize sender_size;
    uint32_t config_buffer_size = sender_size.md_size_bytes + sender_size.ack_size_bytes + sender_size.enc_size_bytes;

    auto shard_params = ShardSpecBuffer(
        CoreRangeSet(CoreRange(sender_core_.core_coord)), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});

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

    config_buffer_ = MeshBuffer::create(config_mesh_buffer_specs, config_buffer_specs, mesh_device.get());
    config_buffer_address_ = config_buffer_->address();
}

void D2HSocket::write_socket_metadata(
    const std::shared_ptr<MeshDevice>& mesh_device,
    const PinnedBufferInfo& data_info,
    const PinnedBufferInfo& bytes_sent_info) const {
    const SocketSenderSize sender_size;
    const uint32_t total_config_bytes =
        sender_size.md_size_bytes + sender_size.ack_size_bytes + sender_size.enc_size_bytes;

    std::vector<uint32_t> config_data(total_config_bytes / sizeof(uint32_t), 0);
    config_data[0] = 0;                        // bytes_sent
    config_data[1] = 1;                        // num_downstreams
    config_data[2] = 0;                        // write_ptr (offset from downstream_fifo_addr)
    config_data[3] = bytes_sent_info.addr_lo;  // downstream_bytes_sent_addr
    config_data[4] = data_info.addr_lo;        // downstream_fifo_addr
    config_data[5] = fifo_size_;               // fifo_total_size
    config_data[6] = 1;                        // is_d2h

    uint32_t host_addr_offset = (sender_size.md_size_bytes + sender_size.ack_size_bytes) / sizeof(uint32_t);
    config_data[host_addr_offset] = bytes_sent_info.addr_hi;
    config_data[host_addr_offset + 1] = data_info.addr_hi;
    config_data[host_addr_offset + 2] = data_info.pcie_xy_enc;

    // External-config ctor skips MeshBuffer allocation; use direct L1 write. Standard
    // ctor owns config_buffer_; use fast-dispatch WriteShard like pre-RT-profiler path.
    if (config_buffer_) {
        distributed::WriteShard(
            mesh_device->mesh_command_queue(0), config_buffer_, config_data, sender_core_.device_coord, true);
    } else {
        IDevice* device = mesh_device->get_device(sender_core_.device_coord);
        tt::tt_metal::detail::WriteToDeviceL1(
            device, sender_core_.core_coord, config_buffer_address_, config_data, CoreType::WORKER);
    }
}

void D2HSocket::init_sender_tlb(const std::shared_ptr<MeshDevice>& mesh_device, std::optional<uint32_t> device_id) {
    TT_FATAL(mesh_device || device_id.has_value(), "Either mesh_device or device_id must be provided.");

    uint32_t sender_device_id;
    CoreCoord sender_virtual_core;

    const auto& cluster = MetalContext::instance().get_cluster();

    if (mesh_device) {
        sender_device_id = mesh_device->get_device(sender_core_.device_coord)->id();
        sender_virtual_core = mesh_device->worker_core_from_logical_core(sender_core_.core_coord);
        // Mock/emulated chips have no real TLB manager (SWEmuleChip::get_tlb_manager() always
        // returns nullptr) — skip the static-TLB fetch below; the dynamic-TLB pcie_writer_ branch
        // (cluster.write_core, which SWEmuleChip does support) is used for them instead.
        if (!cluster.is_mock_or_emulated()) {
            sender_core_tlb_ = cluster.get_driver()
                                   ->get_chip(sender_device_id)
                                   ->get_tlb_manager()
                                   ->get_tlb_window(tt_xy_pair(sender_virtual_core.x, sender_virtual_core.y));
        }
    } else {
        sender_device_id = device_id.value();
        sender_virtual_core = cluster.get_virtual_coordinate_from_logical_coordinates(
            sender_device_id, sender_core_.core_coord, CoreType::TENSIX);
    }

    auto arch = MetalContext::instance().hal().get_arch();
    if (arch == tt::ARCH::BLACKHOLE && mesh_device && !cluster.is_mock_or_emulated()) {
        // This process owns a mesh_device and hence has statically initialized TLBs.
        // Entire device address space for Blackhole is statically mapped.
        // Safe to use static TLBs without requiring the driver to do a reconfig.
        pcie_writer_ = [this](void* data, uint32_t num_bytes, uint64_t device_addr) {
            sender_core_tlb_->write_block(device_addr, data, num_bytes);
        };
    } else {
        // Mesh Device not owned - use dynamic TLBs through UMD.
        // Wormhole B0 may require the driver to do a reconfig of the TLB for each write,
        // since the device address space is not statically mapped.
        pcie_writer_ = [sender_device_id, sender_virtual_core](void* data, uint32_t num_bytes, uint64_t device_addr) {
            const auto& cluster = MetalContext::instance().get_cluster();
            cluster.write_core(data, num_bytes, tt_cxy_pair(sender_device_id, sender_virtual_core), device_addr);
        };
    }
}

void D2HSocket::init_common(const std::shared_ptr<MeshDevice>& mesh_device) {
    MeshCoordinateRangeSet sender_device_range_set;
    sender_device_range_set.merge(MeshCoordinateRange(sender_core_.device_coord));

    const auto& cluster = MetalContext::instance().get_cluster();
    const auto& hal = MetalContext::instance().hal();
    const uint32_t pcie_alignment = pcie_alignment_;
    TT_FATAL(fifo_size_ % pcie_alignment == 0, "FIFO size must be PCIe-aligned.");

    bool can_use_pinned_memory = cluster.is_iommu_enabled() || hal.get_supports_64_bit_pcie_addressing();

    PinnedBufferInfo data_info;
    PinnedBufferInfo bytes_sent_info;

    if (can_use_pinned_memory) {
        std::string shm_name = generate_shm_name("d2h");
        data_info = init_host_buffer(mesh_device, sender_device_range_set, pcie_alignment, shm_name);
        uint64_t bytes_sent_addr = (static_cast<uint64_t>(data_info.addr_hi) << 32 | data_info.addr_lo) + fifo_size_;
        bytes_sent_info = data_info;
        bytes_sent_info.addr_lo = static_cast<uint32_t>(bytes_sent_addr & 0xFFFFFFFFull);
        bytes_sent_info.addr_hi = static_cast<uint32_t>(bytes_sent_addr >> 32);

        // Map the persistent connector-state struct living past the pinned region.
        // NamedShm::create zero-initialized the page; we stamp the version and set
        // clean_shutdown=1 so the first connect() sees "no prior crash" (the owner
        // side has nothing to recover from). Hugepage fallback does not create an
        // SHM region and cannot be exported, so connector_state_ stays null in
        // that path.
        connector_state_ =
            reinterpret_cast<HDSocketConnectorState*>(static_cast<uint8_t*>(shm_->ptr()) + connector_state_offset_);
        connector_state_->version = kHDSocketConnectorStateVersion;
        connector_state_->clean_shutdown = 1;
    } else {
        data_info = init_host_buffer_hugepage(mesh_device);

        auto* device = mesh_device->get_device(sender_core_.device_coord);
        auto& sysmem_mgr = device->sysmem_manager();
        auto [bs_host_ptr, bs_dev_addr] = sysmem_mgr.allocate_region(sizeof(uint32_t));
        hugepage_bytes_sent_host_ptr_ = static_cast<volatile uint32_t*>(bs_host_ptr);
        *const_cast<uint32_t*>(hugepage_bytes_sent_host_ptr_) = 0;

        bytes_sent_info = data_info;
        bytes_sent_info.addr_lo = bs_dev_addr;
        bytes_sent_info.addr_hi = 0;
    }

    write_socket_metadata(mesh_device, data_info, bytes_sent_info);
    init_sender_tlb(mesh_device);

    const SocketSenderSize sender_size;
    bytes_acked_device_offset_ = sender_size.md_size_bytes;
}

D2HSocket::D2HSocket(
    const std::shared_ptr<MeshDevice>& mesh_device, const MeshCoreCoord& sender_core, uint32_t fifo_size) :
    sender_core_(sender_core),
    fifo_size_(fifo_size),
    pcie_alignment_(MetalContext::instance().hal().get_alignment(HalMemType::HOST)),
    mesh_device_(mesh_device.get()) {
    init_config_buffer(mesh_device);
    init_common(mesh_device);
}

D2HSocket::D2HSocket(
    const std::shared_ptr<MeshDevice>& mesh_device,
    const MeshCoreCoord& sender_core,
    uint32_t fifo_size,
    ExternalConfigBuffer external_config) :
    sender_core_(sender_core),
    fifo_size_(fifo_size),
    pcie_alignment_(MetalContext::instance().hal().get_alignment(HalMemType::HOST)),
    mesh_device_(mesh_device.get()) {
    TT_FATAL(external_config.address != 0, "External config buffer address must be non-zero.");
    const uint32_t l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);
    TT_FATAL(
        external_config.address % l1_alignment == 0,
        "External config buffer address 0x{:x} must be L1-aligned ({} B).",
        external_config.address,
        l1_alignment);
    config_buffer_address_ = external_config.address;
    init_common(mesh_device);
}

uint32_t D2HSocket::required_config_buffer_size() {
    const SocketSenderSize sender_size;
    return sender_size.md_size_bytes + sender_size.ack_size_bytes + sender_size.enc_size_bytes;
}

D2HSocket::~D2HSocket() noexcept {
    try {
        if (!exported_) {
            barrier(1000);
        }
    } catch (const std::exception& e) {
        log_warning(LogMetal, "D2HSocket destructor: barrier failed with exception: {}", e.what());
    } catch (...) {
        log_warning(LogMetal, "D2HSocket destructor: barrier failed with unknown exception");
    }
    // Mark a clean shutdown so the next connector sees clean_shutdown=1. A process
    // that exits without running this destructor (crash, _exit, kill) leaves the
    // 0 written by connect()/owner-construct in place, signalling unclean exit.
    // connector_state_ is null in hugepage fallback mode (no SHM region).
    if (connector_state_) {
        connector_state_->clean_shutdown = 1;
    }
    if (is_owner_ && !using_hugepage_) {
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

void D2HSocket::set_page_size(uint32_t page_size) {
    TT_FATAL(pcie_alignment_ > 0, "PCIe alignment not initialized.");
    TT_FATAL(page_size % pcie_alignment_ == 0, "Page size must be PCIE-aligned.");
    TT_FATAL(page_size <= fifo_size_, "Page size must be less than or equal to the FIFO size.");

    uint32_t next_fifo_rd_ptr = align(read_ptr_, page_size);
    uint32_t fifo_page_aligned_size = fifo_size_ - (fifo_size_ % page_size);

    if (next_fifo_rd_ptr >= fifo_page_aligned_size) {
        uint32_t bytes_adjustment = fifo_size_ - next_fifo_rd_ptr;
        uint32_t bytes_recv = bytes_sent_ - bytes_acked_;

        while (bytes_recv < bytes_adjustment) {
            volatile uint32_t bytes_sent_value = using_hugepage_ ? *hugepage_bytes_sent_host_ptr_ : bytes_sent_ptr_[0];
            bytes_recv = bytes_sent_value - bytes_acked_;
            bytes_sent_ = bytes_sent_value;
        }
        bytes_acked_ += bytes_adjustment;
        next_fifo_rd_ptr = 0;
    }
    read_ptr_ = next_fifo_rd_ptr;
    page_size_ = page_size;
    fifo_curr_size_ = fifo_page_aligned_size;
    if (connector_state_) {
        connector_state_->page_size = page_size_;
        connector_state_->fifo_curr_size = fifo_curr_size_;
        connector_state_->bytes_acked = bytes_acked_;
        connector_state_->read_ptr = read_ptr_;
    }
}

bool D2HSocket::has_data(std::optional<uint32_t> num_bytes_to_check) {
    TT_FATAL(page_size_ > 0, "Page size must be set before checking for data.");
    uint32_t num_bytes = num_bytes_to_check.value_or(page_size_);
    if (read_ptr_ + num_bytes >= fifo_curr_size_) {
        num_bytes += fifo_size_ - fifo_curr_size_;
    }
    tt_driver_atomics::mfence();
    volatile uint32_t bytes_sent_value = bytes_sent_ptr_[0];
    bytes_sent_ = bytes_sent_value;
    uint32_t bytes_recv = bytes_sent_value - bytes_acked_;
    return bytes_recv >= num_bytes;
}

void D2HSocket::wait_for_bytes(uint32_t num_bytes) {
    if (read_ptr_ + num_bytes >= fifo_curr_size_) {
        num_bytes += fifo_size_ - fifo_curr_size_;
    }
    uint32_t bytes_recv = bytes_sent_ - bytes_acked_;
    while (bytes_recv < num_bytes) {
        advance_d2h_simulator_socket_device(mesh_device_, sender_core_.device_coord);
        if (using_hugepage_) {
            _mm_clflush(const_cast<void*>(reinterpret_cast<const volatile void*>(hugepage_bytes_sent_host_ptr_)));
            _mm_lfence();
            uint32_t bytes_sent_value = *hugepage_bytes_sent_host_ptr_;
            bytes_recv = bytes_sent_value - bytes_acked_;
            bytes_sent_ = bytes_sent_value;
        } else {
            tt_driver_atomics::mfence();
            volatile uint32_t bytes_sent_value = bytes_sent_ptr_[0];
            bytes_recv = bytes_sent_value - bytes_acked_;
            bytes_sent_ = bytes_sent_value;
        }
    }
}

void D2HSocket::pop_bytes(uint32_t num_bytes) {
    if (read_ptr_ + num_bytes >= fifo_curr_size_) {
        read_ptr_ = read_ptr_ + num_bytes - fifo_curr_size_;
        bytes_acked_ += num_bytes + fifo_size_ - fifo_curr_size_;
    } else {
        read_ptr_ += num_bytes;
        bytes_acked_ += num_bytes;
    }
    // Crash-safe persistence for the next driver process. Null in hugepage
    // fallback mode, which doesn't support cross-process attach.
    if (connector_state_) {
        connector_state_->bytes_acked = bytes_acked_;
        connector_state_->read_ptr = read_ptr_;
    }
}

uint32_t D2HSocket::discard_pending_pages() {
    TT_FATAL(page_size_ > 0, "Page size must be set before discarding pages.");
    uint32_t bytes_sent_value;
    if (using_hugepage_) {
        _mm_clflush(const_cast<void*>(reinterpret_cast<const volatile void*>(hugepage_bytes_sent_host_ptr_)));
        _mm_lfence();
        bytes_sent_value = *hugepage_bytes_sent_host_ptr_;
    } else {
        tt_driver_atomics::mfence();
        bytes_sent_value = bytes_sent_ptr_[0];
    }
    bytes_sent_ = bytes_sent_value;
    uint32_t bytes_recv = bytes_sent_value - bytes_acked_;
    uint32_t pages = bytes_recv / page_size_;
    if (pages == 0) {
        return 0;
    }
    // Rebase: ack everything currently visible without touching the data region; advance
    // read_ptr_ as a real read() would so subsequent reads stay consistent.
    uint32_t bytes_to_discard = pages * page_size_;
    uint32_t cursor = read_ptr_ + bytes_to_discard;
    if (fifo_curr_size_ > 0) {
        cursor %= fifo_curr_size_;
    }
    read_ptr_ = cursor;
    bytes_acked_ += bytes_to_discard;
    if (connector_state_) {
        connector_state_->bytes_acked = bytes_acked_;
        connector_state_->read_ptr = read_ptr_;
    }
    notify_sender();
    return pages;
}

void D2HSocket::notify_sender() {
    uint32_t bytes_acked_addr = config_buffer_address_ + bytes_acked_device_offset_;
    pcie_writer_(&bytes_acked_, sizeof(bytes_acked_), bytes_acked_addr);
    tt_driver_atomics::sfence();
}

void D2HSocket::barrier(std::optional<uint32_t> timeout_ms) {
    auto read_bytes_sent = [this]() -> uint32_t {
        if (using_hugepage_) {
            _mm_clflush(const_cast<void*>(reinterpret_cast<const volatile void*>(hugepage_bytes_sent_host_ptr_)));
            _mm_lfence();
            return *hugepage_bytes_sent_host_ptr_;
        }
        tt_driver_atomics::mfence();
        return bytes_sent_ptr_[0];
    };

    volatile uint32_t bytes_sent_value = read_bytes_sent();
    auto start_time = std::chrono::high_resolution_clock::now();
    while (bytes_acked_ - bytes_sent_value != 0) {
        bytes_sent_value = read_bytes_sent();
        if (timeout_ms.has_value()) {
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::high_resolution_clock::now() - start_time)
                                  .count();
            if (elapsed_ms > timeout_ms.value()) {
                TT_THROW(
                    "Timeout waiting for host to acknowledge data over D2H socket. Bytes sent: {}, Bytes "
                    "acknowledged: {}. Barrier was potentially issued on host before all required reads were "
                    "completed.",
                    bytes_sent_,
                    bytes_sent_value);
            }
        }
    }
}

void D2HSocket::read(void* data, uint32_t num_pages, bool notify_sender) {
    TT_FATAL(page_size_ > 0, "Page size must be set before reading.");
    uint32_t num_bytes = num_pages * page_size_;
    TT_FATAL(num_bytes <= fifo_curr_size_, "Cannot read more pages than the socket FIFO size.");
    this->wait_for_bytes(num_bytes);
    uint32_t* src = using_hugepage_ ? hugepage_data_host_ptr_ + (read_ptr_ / sizeof(uint32_t))
                                    : host_buffer_.get() + (read_ptr_ / sizeof(uint32_t));
    if (using_hugepage_) {
        for (uint32_t i = 0; i < num_bytes; i += k_x86_clflush_line_bytes) {
            _mm_clflush(reinterpret_cast<char*>(src) + i);
        }
        _mm_lfence();
    }
    std::memcpy(data, src, num_bytes);
    this->pop_bytes(num_bytes);

    if (notify_sender) {
        this->notify_sender();
    }
}

uint32_t D2HSocket::pages_available() {
    TT_FATAL(page_size_ > 0, "Page size must be set before checking available pages.");
    uint32_t bytes_sent_value;
    if (using_hugepage_) {
        _mm_clflush(const_cast<void*>(reinterpret_cast<const volatile void*>(hugepage_bytes_sent_host_ptr_)));
        _mm_lfence();
        bytes_sent_value = *hugepage_bytes_sent_host_ptr_;
    } else {
        tt_driver_atomics::mfence();
        bytes_sent_value = bytes_sent_ptr_[0];
    }
    bytes_sent_ = bytes_sent_value;
    uint32_t bytes_recv = bytes_sent_value - bytes_acked_;
    return bytes_recv / page_size_;
}

std::vector<MeshCoreCoord> D2HSocket::get_active_cores() const { return {sender_core_}; }

MeshDevice* D2HSocket::get_mesh_device() const { return mesh_device_; }

std::string D2HSocket::export_descriptor(const std::string& socket_id) {
    TT_FATAL(is_owner_, "Only the owner process can export a socket descriptor.");
    TT_FATAL(shm_ && shm_->is_open(), "Cannot export descriptor: shared memory is not initialized.");

    HDSocketDescriptor desc;
    desc.populate_from_owner("d2h", *shm_, fifo_size_, config_buffer_address_, mesh_device_, sender_core_);
    desc.bytes_sent_offset = fifo_size_;
    desc.bytes_acked_device_offset = bytes_acked_device_offset_;
    desc.connector_state_offset = connector_state_offset_;

    descriptor_path_ = descriptor_path_for_socket("d2h", socket_id);
    desc.write_to_file(descriptor_path_);
    ShmResourceTracker::instance().track_file(descriptor_path_);
    exported_ = true;
    return descriptor_path_;
}

std::unique_ptr<D2HSocket> D2HSocket::connect(const std::string& socket_id, std::optional<uint32_t> timeout_ms) {
    auto desc = HDSocketDescriptor::wait_and_read(
        descriptor_path_for_socket("d2h", socket_id), "d2h", timeout_ms.value_or(10000));

    auto socket = std::unique_ptr<D2HSocket>(new D2HSocket());
    socket->is_owner_ = false;
    socket->fifo_size_ = desc.fifo_size;
    socket->config_buffer_address_ = desc.config_buffer_address;
    socket->pcie_alignment_ = desc.pcie_alignment;
    socket->sender_core_ = MeshCoreCoord(MeshCoordinate(0, 0), CoreCoord(desc.core_x, desc.core_y));
    socket->bytes_acked_device_offset_ = desc.bytes_acked_device_offset;

    socket->shm_ = std::make_unique<NamedShm>(NamedShm::open(desc.shm_name, desc.shm_size));
    socket->host_buffer_ = std::shared_ptr<uint32_t[]>(static_cast<uint32_t*>(socket->shm_->ptr()), [](uint32_t*) {});
    socket->bytes_sent_ptr_ = static_cast<uint32_t*>(socket->shm_->ptr()) + (desc.bytes_sent_offset / sizeof(uint32_t));

    socket->pcie_writer_instance_ =
        std::make_unique<PCIeCoreWriter>(desc.device_id, desc.virtual_core_x, desc.virtual_core_y);
    socket->pcie_writer_ = socket->pcie_writer_instance_->get_pcie_writer();

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
    // _exit, kill); callers can query had_clean_prior_shutdown() to react (e.g.
    // call discard_pending_pages() to drop stale data).
    socket->prior_clean_shutdown_ = (socket->connector_state_->clean_shutdown != 0);
    if (!socket->prior_clean_shutdown_) {
        log_warning(
            LogMetal,
            "D2HSocket::connect: prior connector process exited without running its destructor. State has been "
            "recovered from SHM, but stale pages may be pending in the FIFO; consider discard_pending_pages().");
    }
    socket->connector_state_->clean_shutdown = 0;
    socket->page_size_ = socket->connector_state_->page_size;
    socket->fifo_curr_size_ = socket->connector_state_->fifo_curr_size;
    socket->bytes_acked_ = socket->connector_state_->bytes_acked;
    socket->read_ptr_ = socket->connector_state_->read_ptr;
    // bytes_sent_ is the cached copy of the device-written counter that already
    // lives in SHM. Read it live so wait_for_bytes() sees fresh data immediately.
    socket->bytes_sent_ = socket->bytes_sent_ptr_[0];

    // Reconcile the device-side bytes_acked with the restored SHM value. The
    // previous driver process may have died between pop_bytes (SHM flushed)
    // and notify_sender (PCIe write to the device's config buffer), leaving
    // the device's bytes_acked behind. Without this, the device kernel may
    // stall thinking the FIFO is full while the new connector waits for fresh
    // data. For a fresh socket this writes 0 over 0 — a no-op.
    socket->notify_sender();

    return socket;
}

}  // namespace tt::tt_metal::distributed
