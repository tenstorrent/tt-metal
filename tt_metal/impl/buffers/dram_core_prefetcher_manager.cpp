// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/buffers/dram_core_prefetcher_manager.hpp"

#include "distributed/mesh_device_impl.hpp"
#include "impl/buffers/drisc_l1_arena.hpp"
#include "impl/buffers/h2d_socket_internal.hpp"

#include <cstdint>
#include <cstring>
#include <utility>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/experimental/global_circular_buffer.hpp>
#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>
#include <tt_stl/assert.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"  // DramConfig + CreateKernel(DramConfig)
#include "llrt/metal_soc_descriptor.hpp"
#include "tt_metal/hw/inc/hostdev/socket.h"  // receiver_socket_md (for L1 layout sizing)

namespace tt::tt_metal::distributed {

namespace {

constexpr uint32_t kRemoteCBId = 31;

constexpr const char* kKernelPath = "tt_metal/impl/buffers/kernels/dram_core_prefetcher.cpp";

inline uint32_t align_up(uint32_t a, uint32_t align) { return (a + align - 1) & ~(align - 1); }

// Largest `page` (multiple of tile_size, <= max_page_size) such that num_tiles*tile_size
// is divisible by page. Returns (page_size, num_pages). Identical to the existing
// implementation; carried over verbatim from the pre-queueable manager.
std::pair<uint32_t, uint32_t> pick_page_size(uint32_t max_page_size, uint32_t num_tiles, uint32_t tile_size) {
    const uint64_t total = static_cast<uint64_t>(num_tiles) * tile_size;
    uint32_t page = (max_page_size / tile_size) * tile_size;
    while (page >= tile_size && total % page != 0) {
        page -= tile_size;
    }
    TT_FATAL(
        page >= tile_size,
        "pick_page_size could not find a page that divides num_tiles*tile_size={} (tile={})",
        static_cast<uint64_t>(total),
        tile_size);
    return {page, static_cast<uint32_t>(total / page)};
}

// Address-independent per-tensor geometry for the DRAM-core prefetcher kernel — see
// DramCorePrefetcherTensorLayout in impl/buffers/dram_core_prefetcher_request.hpp for
// the field-by-field documentation, and tt_metal/impl/buffers/prefetcher_matmul_design.md
// §6 for the fit ladder. The tensor's bank-local address is carried separately in the
// per-tensor DramCorePrefetcherEntry, so identical-geometry tensors share one layout.
DramCorePrefetcherTensorLayout compute_tensor_layout(
    const MeshTensor& t, uint32_t block_count, uint32_t num_receivers, uint32_t ring_half, ContextId context_id) {
    const auto* ref_buffer = t.mesh_buffer().get_reference_buffer();
    const auto shard_shape = ref_buffer->shard_spec().shape();
    const uint32_t tile_bytes = tt::tile_size(datatype_to_dataformat_converter(t.dtype()));
    TT_FATAL(
        shard_shape[0] % tt::constants::TILE_HEIGHT == 0 && shard_shape[1] % tt::constants::TILE_WIDTH == 0,
        "DRAM-core prefetcher requires tile-aligned shards; got shard_shape=({}, {}), tile=({}, {})",
        shard_shape[0],
        shard_shape[1],
        tt::constants::TILE_HEIGHT,
        tt::constants::TILE_WIDTH);
    const uint32_t k_tiles_raw = shard_shape[0] / tt::constants::TILE_HEIGHT;
    const uint32_t n_per_bank = shard_shape[1] / tt::constants::TILE_WIDTH;
    TT_FATAL(block_count > 0, "DRAM-core prefetcher block_count must be > 0");
    TT_FATAL(
        n_per_bank % num_receivers == 0,
        "n_per_bank ({}) must divide num_receivers ({}); reduce N_per_bank or grow num_receivers",
        n_per_bank,
        num_receivers);
    // block_count K-blocks per tensor (replaces the GCB ring size here); the
    // consuming matmul does wait_front(block_count) per layer.
    const uint32_t k_block_w_tiles = (k_tiles_raw + block_count - 1) / block_count;
    const uint32_t n_per_recv = n_per_bank / num_receivers;
    const uint32_t row_bytes = n_per_bank * tile_bytes;
    const uint32_t block_bytes = k_block_w_tiles * row_bytes;
    const uint32_t noc_max_burst = MetalContext::instance(context_id).hal().get_noc_max_burst_size_bytes();
    const auto [coalesced_page_size, coalesced_num_pages] = pick_page_size(noc_max_burst, n_per_recv, tile_bytes);
    TT_FATAL(
        coalesced_page_size <= noc_max_burst,
        "DRAM-core prefetcher coalesced page size ({} B) exceeds the one-packet NoC write limit ({} B).",
        coalesced_page_size,
        noc_max_burst);

    uint32_t rows_per_sub = 0;
    uint32_t M = 1;
    if (block_bytes <= ring_half) {
        rows_per_sub = k_block_w_tiles;
        M = 1;
    } else if (row_bytes <= ring_half) {
        rows_per_sub = 1;
        for (uint32_t d = k_block_w_tiles; d >= 1; --d) {
            if (k_block_w_tiles % d == 0 && static_cast<uint64_t>(d) * row_bytes <= ring_half) {
                rows_per_sub = d;
                break;
            }
        }
        M = 1;
    } else {
        rows_per_sub = 1;
        bool picked = false;
        for (uint32_t m = 1; m <= num_receivers; ++m) {
            if (num_receivers % m == 0 && (row_bytes / m) <= ring_half) {
                M = m;
                picked = true;
                break;
            }
        }
        TT_FATAL(
            picked,
            "DRAM-core prefetcher cannot fit one K-row of tensor (k_tiles={}, n_per_bank={}, tile_bytes={}, "
            "row_bytes={} B) into DRISC L1 stage half ({} B) even with M=num_receivers ({}).",
            k_tiles_raw,
            n_per_bank,
            tile_bytes,
            row_bytes,
            ring_half,
            num_receivers);
    }
    const uint32_t num_sub = k_block_w_tiles / rows_per_sub;
    const uint32_t sub_chunk_bytes = rows_per_sub * (n_per_bank / M) * tile_bytes;
    TT_FATAL(
        sub_chunk_bytes <= ring_half, "Internal: chunk size {} B exceeds ring_half {} B", sub_chunk_bytes, ring_half);

    DramCorePrefetcherTensorLayout g;
    g.num_sub = num_sub;
    g.M = M;
    g.rows_per_sub = rows_per_sub;
    g.coalesced_page_size = coalesced_page_size;
    g.coalesced_num_pages = coalesced_num_pages;
    g.sub_chunk_bytes = sub_chunk_bytes;
    g.sub_stride_bytes = rows_per_sub * row_bytes;
    g.block_stride_bytes = k_block_w_tiles * row_bytes;
    g.page_bytes_per_recv = k_block_w_tiles * coalesced_num_pages * coalesced_page_size;
    g.block_count = block_count;
    return g;
}

// Two layouts are interchangeable iff every geometry field matches. The struct is a
// packed POD of uint32_t fields, so a byte compare is exact (no padding).
bool layout_equal(const DramCorePrefetcherTensorLayout& a, const DramCorePrefetcherTensorLayout& b) {
    return std::memcmp(&a, &b, sizeof(DramCorePrefetcherTensorLayout)) == 0;
}

}  // namespace

DramCorePrefetcherManager::DramCorePrefetcherManager(
    MeshDevice* mesh_device, std::function<std::lock_guard<std::mutex>()> lock_api_function) :
    mesh_device_(mesh_device), lock_api_function_(std::move(lock_api_function)) {
    TT_FATAL(mesh_device_ != nullptr, "DramCorePrefetcherManager requires a non-null MeshDevice");
    TT_FATAL(static_cast<bool>(lock_api_function_), "DramCorePrefetcherManager requires a valid lock_api_function");
}

DramCorePrefetcherManager::~DramCorePrefetcherManager() { stop(); }

void DramCorePrefetcherManager::enumerate_dram_senders() {
    const auto context_id = mesh_device_->impl().get_context_id();
    const auto& soc_desc = MetalContext::instance(context_id)
                               .get_cluster()
                               .get_soc_desc(mesh_device_->get_view().get_devices().front()->id());
    const uint32_t num_banks = soc_desc.get_num_dram_views();
    sender_logical_cores_.clear();
    sender_logical_cores_.reserve(num_banks);
    for (uint32_t b = 0; b < num_banks; ++b) {
        sender_logical_cores_.push_back(mesh_device_->impl().pick_unused_dram_logical_core(b));
    }
    num_senders_ = num_banks;
}

void DramCorePrefetcherManager::allocate_sockets() {
    const auto& hal = MetalContext::instance(mesh_device_->impl().get_context_id()).hal();
    const uint32_t pcie_alignment = hal.get_alignment(HalMemType::HOST);
    const uint32_t page_size = align_up(kRequestPageBytes, pcie_alignment);
    const uint32_t fifo_size = align_up(page_size * kSocketFifoPages, pcie_alignment);

    sockets_.clear();
    sockets_.reserve(devices_.size() * num_senders_);

    auto mesh_device_sp = mesh_device_->shared_from_this();
    const uint64_t dram_l1_noc_offset = hal.get_l1_noc_offset(HalProgrammableCoreType::DRAM);

    for (auto* device : devices_) {
        const MeshCoordinate device_coord = mesh_device_->get_view().find_device(device->id());
        for (uint32_t s = 0; s < num_senders_; ++s) {
            const CoreCoord sender_logical = sender_logical_cores_[s];
            // Uniform L1 layout per DRAM core: only one socket lives on each core
            // (the one for that core's own sender). Use the same offsets carved by
            // start() — socket_config_l1_addr_ then socket_data_l1_addr_.
            auto socket = experimental::detail::create_h2d_socket_for_dram_recv(
                mesh_device_sp,
                MeshCoreCoord(device_coord, sender_logical),
                fifo_size,
                socket_config_l1_addr_,
                socket_data_l1_addr_,
                dram_l1_noc_offset);
            socket->set_page_size(page_size);
            sockets_.push_back(std::move(socket));
        }
    }
}

void DramCorePrefetcherManager::build_and_launch_programs(uint32_t stage_ring_base, uint32_t stage_ring_size) {
    // Sockets must already be allocated so each kernel can be given its
    // socket_config_addr as a runtime arg.
    TT_FATAL(sockets_.size() == devices_.size() * num_senders_, "sockets must be allocated before programs");

    const uint32_t pcie_alignment =
        MetalContext::instance(mesh_device_->impl().get_context_id()).hal().get_alignment(HalMemType::HOST);
    const uint32_t socket_page_size = align_up(kRequestPageBytes, pcie_alignment);

    programs_.clear();
    for (uint32_t d = 0; d < devices_.size(); ++d) {
        auto program = std::make_unique<Program>();

        for (uint32_t s = 0; s < num_senders_; ++s) {
            const CoreCoord sender_logical = sender_logical_cores_[s];

            std::vector<uint32_t> compile_args = {
                stage_ring_base,
                stage_ring_size,
                kRemoteCBId,
                socket_page_size,
            };

            KernelHandle kernel_id = CreateKernel(
                *program, kKernelPath, sender_logical, DramConfig{.noc = NOC::NOC_0, .compile_args = compile_args});

            const uint32_t socket_addr = sockets_[d * num_senders_ + s]->get_config_buffer_address();
            std::vector<uint32_t> rt_args = {/*bank_id=*/static_cast<uint32_t>(sender_logical.x), socket_addr};
            SetRuntimeArgs(*program, kernel_id, sender_logical, rt_args);
        }

        programs_.push_back(std::move(program));
    }
}

void DramCorePrefetcherManager::start(const experimental::DramCorePrefetcherConfig& config) {
    auto lock = lock_api_function_();
    (void)config;
    TT_FATAL(
        !active_, "A DRAM-core prefetcher is already active on this mesh device. Call StopDramCorePrefetcher first.");

    const auto& hal = MetalContext::instance(mesh_device_->impl().get_context_id()).hal();
    TT_FATAL(
        hal.has_programmable_core_type(HalProgrammableCoreType::DRAM),
        "DRAM-core prefetcher requires programmable DRAM cores; set "
        "TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES=1");

    enumerate_dram_senders();

    // DRISC L1 layout: the kernel working region (above the GCB zone) is now
    // entirely the ping-pong stage ring. noc_xy table, config block, and the
    // RemoteSenderCBInterface live inside each GCB's sender state block, owned
    // by the GCB rather than the prefetcher kernel.
    auto& arena = mesh_device_->impl().drisc_l1_arena();
    const uint32_t kernel_region_size = arena.kernel_working_region_size();
    const uint32_t l1_alignment = hal::get_l1_alignment();
    TT_FATAL(
        kernel_region_size >= 2 * l1_alignment,
        "DRISC L1 kernel region ({} B) too small for the prefetcher ping-pong stage",
        kernel_region_size);
    // Carve the per-core DRISC L1 kernel working region into:
    //   [socket_config | socket_data FIFO | stage ring].
    // Each DRAM core hosts exactly one H2DSocket recv (for its own sender), so
    // the layout is uniform across all DRAM cores. The MeshBuffer L1 allocator
    // can't reach DRAM-core L1, so we hand the addresses to the H2DSocket
    // bypass ctor directly (see h2d_socket_internal.hpp).
    const uint32_t pcie_alignment_for_layout = hal.get_alignment(HalMemType::HOST);
    const uint32_t page_size_for_layout = align_up(kRequestPageBytes, pcie_alignment_for_layout);
    const uint32_t socket_fifo_size_for_layout =
        align_up(page_size_for_layout * kSocketFifoPages, pcie_alignment_for_layout);
    const uint32_t socket_config_bytes = align_up(sizeof(receiver_socket_md), pcie_alignment_for_layout);
    const uint32_t socket_data_bytes = socket_fifo_size_for_layout + pcie_alignment_for_layout;
    const uint32_t kernel_region_base = static_cast<uint32_t>(arena.kernel_working_region_base());
    socket_config_l1_addr_ = align_up(kernel_region_base, pcie_alignment_for_layout);
    socket_data_l1_addr_ = align_up(socket_config_l1_addr_ + socket_config_bytes, pcie_alignment_for_layout);
    stage_ring_base_ = align_up(socket_data_l1_addr_ + socket_data_bytes, l1_alignment);
    const uint32_t kernel_region_end = kernel_region_base + kernel_region_size;
    TT_FATAL(
        stage_ring_base_ < kernel_region_end,
        "DRISC L1 kernel region ({} B) too small for socket buffers + stage ring",
        kernel_region_size);
    stage_ring_size_ = kernel_region_end - stage_ring_base_;
    stage_ring_size_ &= ~(2 * l1_alignment - 1);
    // After masking down, make sure the ring is still big enough for at least
    // one minimal sub-chunk per half. Catches accidental shrink-to-zero if the
    // socket carve-out ever grows past the L1 region.
    TT_FATAL(
        stage_ring_size_ >= 4 * l1_alignment,
        "DRISC L1 stage ring shrank to {} B after alignment masking — socket buffers consumed too much of the {} B "
        "kernel region",
        stage_ring_size_,
        kernel_region_size);
    ring_half_ = stage_ring_size_ / 2;

    // Populate devices_ list once; both allocate_sockets and build_and_launch_programs use it.
    // Build the coord->index map at the same time so worker_loop fan-out is O(targets).
    devices_.clear();
    device_index_by_coord_.clear();
    for (auto* device : mesh_device_->get_view().get_devices()) {
        const uint32_t d = static_cast<uint32_t>(devices_.size());
        devices_.push_back(device);
        device_index_by_coord_.emplace(mesh_device_->get_view().find_device(device->id()), d);
    }

    allocate_sockets();
    build_and_launch_programs(stage_ring_base_, stage_ring_size_);

    // Launch programs (non-blocking — kernels park on the socket immediately).
    for (uint32_t d = 0; d < devices_.size(); ++d) {
        ::tt::tt_metal::detail::CompileProgram(devices_[d], *programs_[d], /*force_slow_dispatch=*/true);
        ::tt::tt_metal::detail::WriteRuntimeArgsToDevice(devices_[d], *programs_[d], /*force_slow_dispatch=*/true);
        ::tt::tt_metal::detail::LaunchProgram(
            devices_[d], *programs_[d], /*wait_until_cores_done=*/false, /*force_slow_dispatch=*/true);
    }

    stop_requested_.store(false);
    host_worker_ = std::thread(&DramCorePrefetcherManager::worker_loop, this);
    active_ = true;
}

MeshCoordinateRangeSet DramCorePrefetcherManager::full_mesh_subset() const {
    MeshCoordinateRangeSet out;
    out.merge(MeshCoordinateRange(mesh_device_->shape()));
    return out;
}

std::vector<std::vector<uint8_t>> DramCorePrefetcherManager::serialize_request_pages(
    const experimental::GlobalCircularBuffer& gcb,
    const std::vector<experimental::DramCorePrefetcherInput>& data_tensors) const {
    TT_FATAL(!data_tensors.empty(), "QueueDramCorePrefetcherRequest requires at least one tensor");

    const ContextId context_id = mesh_device_->impl().get_context_id();

    // Derive num_receivers from the GCB itself so each Queue call can target a
    // GCB with a different receiver count. The DRAM-sender GCB ctor enforces a
    // uniform receiver count across senders, so front() speaks for every sender.
    const uint32_t gcb_num_receivers = gcb.sender_receiver_core_mapping().front().second.num_cores();
    const uint32_t gcb_state_addr = static_cast<uint32_t>(experimental::sender_state_drisc_l1_base(gcb));

    const uint32_t pcie_alignment = MetalContext::instance(context_id).hal().get_alignment(HalMemType::HOST);
    const uint32_t aligned_page_bytes = align_up(kRequestPageBytes, pcie_alignment);

    constexpr uint32_t kHeaderBytes = sizeof(DramCorePrefetcherRequestHeader);
    constexpr uint32_t kEntryBytes = sizeof(DramCorePrefetcherEntry);
    constexpr uint32_t kLayoutBytes = sizeof(DramCorePrefetcherTensorLayout);

    std::vector<std::vector<uint8_t>> pages;

    // Per-page packing state. Entries grow forward from kHeaderBytes; the layout table
    // grows backward from kRequestPageBytes (layout i at kRequestPageBytes -
    // (i+1)*kLayoutBytes). `seen` holds this page's deduplicated layouts in index order.
    std::vector<uint8_t> page;
    uint32_t num_entries = 0;
    uint32_t num_layouts = 0;
    std::vector<DramCorePrefetcherTensorLayout> seen;

    auto begin_page = [&]() {
        page.assign(aligned_page_bytes, 0);
        num_entries = 0;
        num_layouts = 0;
        seen.clear();
    };
    auto finalize_page = [&]() {
        auto* header = reinterpret_cast<DramCorePrefetcherRequestHeader*>(page.data());
        header->num_entries = num_entries;
        header->num_layouts = num_layouts;
        header->gcb_state_addr = gcb_state_addr;
        pages.push_back(std::move(page));
    };

    begin_page();
    for (uint32_t t = 0; t < data_tensors.size(); ++t) {
        const experimental::DramCorePrefetcherInput& input = data_tensors[t];
        TT_FATAL(input.tensor != nullptr, "QueueDramCorePrefetcherRequest: input tensor {} is null", t);
        // block_count is per-tensor: it sets how many K-blocks the kernel pushes
        // (and how K is divided in compute_tensor_layout), replacing the GCB ring size.
        const DramCorePrefetcherTensorLayout layout =
            compute_tensor_layout(*input.tensor, input.block_count, gcb_num_receivers, ring_half_, context_id);
        const uint32_t bank_local_base = static_cast<uint32_t>(input.tensor->mesh_buffer().address());

        // Find this layout in the current page (dedup), or decide it needs adding.
        auto find_layout = [&]() -> int32_t {
            for (uint32_t i = 0; i < num_layouts; ++i) {
                if (layout_equal(seen[i], layout)) {
                    return static_cast<int32_t>(i);
                }
            }
            return -1;
        };
        int32_t layout_idx = find_layout();
        const uint32_t need = kEntryBytes + (layout_idx < 0 ? kLayoutBytes : 0);
        const uint32_t entry_high = kHeaderBytes + num_entries * kEntryBytes;
        const uint32_t layout_low = kRequestPageBytes - num_layouts * kLayoutBytes;
        if (need > layout_low - entry_high) {
            // No room in the current page — emit it and start a fresh one. The tensor's
            // layout is page-local, so it becomes a new layout in the next page.
            finalize_page();
            begin_page();
            layout_idx = -1;
        }

        if (layout_idx < 0) {
            layout_idx = static_cast<int32_t>(num_layouts);
            std::memcpy(page.data() + (kRequestPageBytes - (num_layouts + 1) * kLayoutBytes), &layout, kLayoutBytes);
            seen.push_back(layout);
            ++num_layouts;
        }

        DramCorePrefetcherEntry entry;
        entry.bank_local_base = bank_local_base;
        entry.layout_index = static_cast<uint32_t>(layout_idx);
        std::memcpy(page.data() + (kHeaderBytes + num_entries * kEntryBytes), &entry, kEntryBytes);
        ++num_entries;
    }
    finalize_page();

    return pages;
}

void DramCorePrefetcherManager::queue(
    const experimental::GlobalCircularBuffer& gcb,
    const std::optional<MeshCoordinateRangeSet>& device_subset,
    const std::vector<experimental::DramCorePrefetcherInput>& tensors) {
    auto lock = lock_api_function_();
    TT_FATAL(active_, "QueueDramCorePrefetcherRequest called before StartDramCorePrefetcher");
    TT_FATAL(
        experimental::sender_core_type(gcb) == experimental::SenderCoreType::Dram,
        "QueueDramCorePrefetcherRequest requires a DRAM-sender GlobalCircularBuffer");
    TT_FATAL(
        gcb.sender_receiver_core_mapping().size() == num_senders_,
        "GCB num_senders ({}) does not match prefetcher num_senders ({})",
        gcb.sender_receiver_core_mapping().size(),
        num_senders_);
    TT_FATAL(!tensors.empty(), "QueueDramCorePrefetcherRequest requires at least one tensor");

    // A Queue call may span more tensors than fit in one socket page; serialize into one
    // or more pages, each an independent request. The per-GCB fifo_wr_ptr persists across
    // requests, so the split is invisible to the receiver.
    std::vector<std::vector<uint8_t>> pages = serialize_request_pages(gcb, tensors);

    // Target devices: subset if given, else full mesh. Caller is responsible
    // for keeping tensors and the GCB alive until stop() — see the public API doc.
    std::vector<MeshCoordinate> target_devices;
    MeshCoordinateRangeSet effective_subset = device_subset.has_value() ? *device_subset : full_mesh_subset();
    for (const auto& range : effective_subset.ranges()) {
        for (const auto& coord : range) {
            TT_FATAL(
                device_index_by_coord_.contains(coord),
                "QueueDramCorePrefetcherRequest target MeshCoordinate {} is not in the mesh this prefetcher was "
                "started "
                "on",
                coord);
            target_devices.push_back(coord);
        }
    }

    {
        // Push all pages of this call under one lock so worker_loop sends them in order
        // (required for fifo_wr_ptr continuity across the split).
        std::lock_guard<std::mutex> lk(queue_mu_);
        for (auto& page : pages) {
            Request req;
            req.page = std::move(page);
            req.target_devices = target_devices;
            pending_.push_back(std::move(req));
        }
    }
    queue_cv_.notify_one();
}

void DramCorePrefetcherManager::worker_loop() {
    while (true) {
        Request req;
        {
            std::unique_lock<std::mutex> lk(queue_mu_);
            queue_cv_.wait(lk, [&] { return !pending_.empty() || stop_requested_.load(); });
            if (pending_.empty()) {
                // Stop with empty queue → break out, the main thread handles
                // sentinel broadcast outside the lock.
                return;
            }
            req = std::move(pending_.front());
            pending_.pop_front();
        }

        // Fan out: try_write to every target socket; round-robin until each succeeds.
        std::vector<uint32_t> remaining_target_sockets;
        remaining_target_sockets.reserve(req.target_devices.size() * num_senders_);
        for (const auto& dev_coord : req.target_devices) {
            auto it = device_index_by_coord_.find(dev_coord);
            TT_FATAL(
                it != device_index_by_coord_.end(),
                "QueueDramCorePrefetcherRequest target MeshCoordinate {} is not in the mesh this prefetcher was "
                "started "
                "on; would silently drop the request for that device",
                dev_coord);
            const uint32_t d = it->second;
            for (uint32_t s = 0; s < num_senders_; ++s) {
                remaining_target_sockets.push_back(d * num_senders_ + s);
            }
        }

        // Round-robin try_write with non-blocking attempts.
        std::vector<uint32_t> still_pending = std::move(remaining_target_sockets);
        while (!still_pending.empty()) {
            std::vector<uint32_t> next_pending;
            for (uint32_t sock_idx : still_pending) {
                if (experimental::detail::try_write(*sockets_[sock_idx], req.page.data(), 1)) {
                    // Wrote successfully.
                } else {
                    next_pending.push_back(sock_idx);
                }
            }
            // If no socket drained this pass, every remaining target is back-pressured;
            // yield rather than busy-spinning at 100% CPU until a receiver frees a page.
            if (next_pending.size() == still_pending.size()) {
                std::this_thread::yield();
            }
            still_pending = std::move(next_pending);
        }
    }
}

void DramCorePrefetcherManager::stop() {
    // Note: the MeshDevice close path (close_impl) calls stop() without holding
    // api_mutex_, and the destructor only reaches here after active_ is already
    // false, so taking the lock here never self-deadlocks.
    auto lock = lock_api_function_();
    if (!active_) {
        return;
    }
    // Stop = a zero-filled page (num_entries == 0) broadcast to every device in
    // the mesh. The kernel exits its request loop on `num_entries == 0`. The
    // worker_loop returns once pending is drained and stop_requested_ is set.
    Request sentinel;
    const uint32_t pcie_alignment =
        MetalContext::instance(mesh_device_->impl().get_context_id()).hal().get_alignment(HalMemType::HOST);
    const uint32_t page_bytes = align_up(kRequestPageBytes, pcie_alignment);
    sentinel.page.assign(page_bytes, 0);
    const MeshCoordinateRangeSet full_subset = full_mesh_subset();
    for (const auto& range : full_subset.ranges()) {
        for (const auto& coord : range) {
            sentinel.target_devices.push_back(coord);
        }
    }

    {
        std::lock_guard<std::mutex> lk(queue_mu_);
        pending_.push_back(std::move(sentinel));
        stop_requested_.store(true);
    }
    queue_cv_.notify_all();

    if (host_worker_.joinable()) {
        host_worker_.join();
    }

    // Wait for kernels to drain their request loop (they exit on the sentinel).
    for (uint32_t d = 0; d < devices_.size(); ++d) {
        ::tt::tt_metal::detail::WaitProgramDone(devices_[d], *programs_[d]);
    }

    sockets_.clear();
    programs_.clear();
    devices_.clear();
    device_index_by_coord_.clear();
    sender_logical_cores_.clear();
    num_senders_ = 0;
    active_ = false;
}

}  // namespace tt::tt_metal::distributed

// -----------------------------------------------------------------------------
// experimental::StartDramCorePrefetcher / Queue / Stop
// -----------------------------------------------------------------------------
namespace tt::tt_metal::experimental {

bool IsDramCorePrefetcherSupported(const distributed::MeshDevice& mesh_device) {
    const auto& hal = MetalContext::instance(mesh_device.impl().get_context_id()).hal();
    return hal.has_programmable_core_type(HalProgrammableCoreType::DRAM);
}

void StartDramCorePrefetcher(distributed::MeshDevice& mesh_device, const DramCorePrefetcherConfig& config) {
    auto& manager = mesh_device.impl().dram_core_prefetcher(&mesh_device);
    manager.start(config);
}

void QueueDramCorePrefetcherRequest(
    distributed::MeshDevice& mesh_device,
    const GlobalCircularBuffer& gcb,
    const std::optional<distributed::MeshCoordinateRangeSet>& device_subset,
    const std::vector<DramCorePrefetcherInput>& input_tensors) {
    auto& manager = mesh_device.impl().dram_core_prefetcher(&mesh_device);
    manager.queue(gcb, device_subset, input_tensors);
}

void StopDramCorePrefetcher(distributed::MeshDevice& mesh_device) {
    auto& manager = mesh_device.impl().dram_core_prefetcher(&mesh_device);
    manager.stop();
}

}  // namespace tt::tt_metal::experimental
