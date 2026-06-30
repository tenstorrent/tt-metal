// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/buffers/tensor_prefetcher_manager.hpp"

#include "distributed/mesh_device_impl.hpp"
#include "distributed/mesh_command_queue_base.hpp"
#include "impl/buffers/drisc_l1_arena.hpp"
#include "impl/buffers/h2d_socket_internal.hpp"

#include <cstdint>
#include <cstring>
#include <utility>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
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

constexpr const char* kKernelPath = "tt_metal/impl/buffers/kernels/tensor_prefetcher.cpp";

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

// Distinguishes the two supported buffer layouts the prefetcher knows how to
// consume. KRowMajor is the legacy layout: one shard per DRAM bank,
// K-row-major within the bank. ReceiverContiguous stacks num_receivers slabs
// per bank, each slab being (K, n_per_recv) contiguous bytes. Detection is
// implicit per the design doc — see detect_layout_mode().
enum class LayoutMode : uint32_t {
    KRowMajor = 0,
    ReceiverContiguous = 1,
};

// Detection keys on how the weight was allocated, NOT the shard count: receiver-contiguous
// weights are created with an NdShardSpec (num_shards == ring_size), K-row-major weights use a
// legacy (WIDTH_SHARDED) shard spec. Counting shards is ambiguous when total_receivers ==
// num_banks (one receiver per bank, num_shards == num_banks == total_receivers): the count tie
// would route a recv-contig tensor down the K-row path, where compute_tensor_layout_krow_major
// calls shard_spec() and TT_FATALs because the buffer only has an NdShardSpec.
LayoutMode detect_layout_mode(const MeshTensor& t, const Buffer& buf, uint32_t total_receivers) {
    // Legacy ShardSpec path: one wide shard per bank by construction. Some WIDTH_SHARDED
    // tensors also carry an NdShardSpec-like descriptor through BDS, so prefer the explicit
    // legacy shard spec before classifying a tensor as receiver-contiguous.
    if (buf.has_shard_spec()) {
        return LayoutMode::KRowMajor;
    }

    if (t.nd_shard_spec().has_value()) {
        const auto& bds_opt = buf.buffer_distribution_spec();
        TT_FATAL(
            bds_opt.has_value() && static_cast<uint32_t>(bds_opt->num_shards()) == total_receivers,
            "Receiver-contiguous Tensor prefetcher weight must have num_shards == total_receivers "
            "(ring_size = {}); got {} shards.",
            total_receivers,
            bds_opt.has_value() ? static_cast<uint32_t>(bds_opt->num_shards()) : 0u);
        return LayoutMode::ReceiverContiguous;
    }
    return LayoutMode::KRowMajor;
}

// Address-independent per-tensor geometry for the K-row-major DRAM layout — see
// TensorPrefetcherTensorLayout in impl/buffers/tensor_prefetcher_request.hpp for
// the field-by-field documentation, and tt_metal/impl/buffers/prefetcher_matmul_design.md
// §6 for the fit ladder. The tensor's bank-local address is carried separately in the
// per-tensor TensorPrefetcherEntry, so identical-geometry tensors share one layout.
TensorPrefetcherTensorLayout compute_tensor_layout_krow_major(
    const MeshTensor& t, uint32_t block_count, uint32_t num_receivers, uint32_t ring_half, ContextId context_id) {
    const auto* ref_buffer = t.mesh_buffer().get_reference_buffer();
    const auto shard_shape = ref_buffer->shard_spec().shape();
    const uint32_t tile_bytes = tt::tile_size(datatype_to_dataformat_converter(t.dtype()));
    TT_FATAL(
        shard_shape[0] % tt::constants::TILE_HEIGHT == 0 && shard_shape[1] % tt::constants::TILE_WIDTH == 0,
        "Tensor prefetcher requires tile-aligned shards; got shard_shape=({}, {}), tile=({}, {})",
        shard_shape[0],
        shard_shape[1],
        tt::constants::TILE_HEIGHT,
        tt::constants::TILE_WIDTH);
    const uint32_t k_tiles_raw = shard_shape[0] / tt::constants::TILE_HEIGHT;
    const uint32_t n_per_bank = shard_shape[1] / tt::constants::TILE_WIDTH;
    TT_FATAL(block_count > 0, "Tensor prefetcher block_count must be > 0");
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
        "Tensor prefetcher coalesced page size ({} B) exceeds the one-packet NoC write limit ({} B).",
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
            "Tensor prefetcher cannot fit one K-row of tensor (k_tiles={}, n_per_bank={}, tile_bytes={}, "
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

    TensorPrefetcherTensorLayout g;
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
bool layout_equal(const TensorPrefetcherTensorLayout& a, const TensorPrefetcherTensorLayout& b) {
    return std::memcmp(&a, &b, sizeof(TensorPrefetcherTensorLayout)) == 0;
}

// Receiver-contiguous layout: BDS holds `ring_size` shards round-robin across
// `num_senders` banks, so each bank stacks `num_receivers` shards each of shape
// (K_tiles, n_per_recv_tiles). Per (receiver, block) the source bytes are a
// single contiguous DRAM region. The kernel dynamically batches B blocks per
// per-receiver visit (B clamped by free space and fifo wrap at runtime); the
// manager just computes the static ceiling target_per_visit_pages. The tensor's
// bank-local address is carried separately in the per-tensor
// TensorPrefetcherEntry, so identical-geometry tensors share one layout.
TensorPrefetcherTensorLayout compute_tensor_layout_recv_contig(
    const MeshTensor& t, uint32_t block_count, uint32_t stage_third, ContextId context_id) {
    // Read the original (non-squeezed) NdShardSpec from the MemoryConfig — the
    // BDS internally collapses adjacent matching dims, so its
    // shard_shape_in_pages() can come back rank-1 even though the caller
    // passed a 2D shape.
    const auto& nd_opt = t.nd_shard_spec();
    TT_FATAL(
        nd_opt.has_value(),
        "Receiver-contiguous Tensor prefetcher tensor must be allocated with an "
        "NdShardSpec (e.g. ttnn.MemoryConfig(BufferType.DRAM, NdShardSpec(...))).");
    const auto& shard_shape = nd_opt->shard_shape;
    TT_FATAL(
        shard_shape.rank() == 2,
        "Receiver-contiguous NdShardSpec shard shape must be 2D (K_elems, n_per_recv_elems); got rank {}",
        shard_shape.rank());
    const uint32_t k_elems = shard_shape[0];
    const uint32_t n_per_recv_elems = shard_shape[1];
    TT_FATAL(
        k_elems % tt::constants::TILE_HEIGHT == 0 && n_per_recv_elems % tt::constants::TILE_WIDTH == 0,
        "Receiver-contiguous shard shape ({}, {}) must be tile-aligned (TILE={}, {})",
        k_elems,
        n_per_recv_elems,
        tt::constants::TILE_HEIGHT,
        tt::constants::TILE_WIDTH);
    const uint32_t k_tiles_raw = k_elems / tt::constants::TILE_HEIGHT;
    const uint32_t n_per_recv = n_per_recv_elems / tt::constants::TILE_WIDTH;
    TT_FATAL(
        k_tiles_raw > 0 && n_per_recv > 0,
        "Receiver-contiguous shard shape has zero dim: ({}, {})",
        k_tiles_raw,
        n_per_recv);

    const uint32_t tile_bytes = tt::tile_size(datatype_to_dataformat_converter(t.dtype()));
    TT_FATAL(block_count > 0, "Tensor prefetcher block_count must be > 0");
    // K must divide evenly into block_count K-blocks. A ceil here would make the kernel push
    // block_count * ceil(K_tiles/block_count) K-rows per receiver — more than the slab
    // (recv_stride = K_tiles * n_per_recv * tile_bytes) holds — so the last block over-reads
    // into the next receiver's slab (or past the buffer for the last slab). For a matmul-fed
    // weight, compute block_count via tensor_prefetcher_block_count_for_matmul_1d(), which
    // also pins block_count == ring_size.
    TT_FATAL(
        k_tiles_raw % block_count == 0,
        "Receiver-contiguous: weight K ({} tiles) must be divisible by block_count ({}); remainder {}. "
        "block_count must equal the matmul ring_size.",
        k_tiles_raw,
        block_count,
        k_tiles_raw % block_count);
    const uint32_t k_block_w_tiles = k_tiles_raw / block_count;

    const uint32_t noc_max_burst = MetalContext::instance(context_id).hal().get_noc_max_burst_size_bytes();
    const auto [coalesced_page_size, coalesced_num_pages] = pick_page_size(noc_max_burst, n_per_recv, tile_bytes);
    TT_FATAL(
        coalesced_page_size <= noc_max_burst,
        "Tensor prefetcher coalesced page size ({} B) exceeds the one-packet NoC write limit ({} B).",
        coalesced_page_size,
        noc_max_burst);

    const uint32_t bytes_per_recv_per_block = k_block_w_tiles * n_per_recv * tile_bytes;
    const uint32_t recv_stride_bytes = k_tiles_raw * n_per_recv * tile_bytes;

    // Fit ladder. Rung 1: full block fits in one stage-third (the kernel uses
    // 3 rotating slots, not 2 halves, so the constraint is tighter than the
    // legacy K-row path). Rung 2: K-sub split for shapes where one block
    // exceeds the stage_third; the kernel walks sub-bands across slots and
    // multi-block batching still works because B blocks of a receiver are
    // contiguous in DRAM.
    uint32_t rows_per_sub = 0;
    uint32_t num_sub = 1;
    if (bytes_per_recv_per_block <= stage_third) {
        rows_per_sub = k_block_w_tiles;
        num_sub = 1;
    } else {
        rows_per_sub = 0;
        for (uint32_t d = k_block_w_tiles; d >= 1; --d) {
            if (k_block_w_tiles % d == 0 && static_cast<uint64_t>(d) * n_per_recv * tile_bytes <= stage_third) {
                rows_per_sub = d;
                break;
            }
        }
        TT_FATAL(
            rows_per_sub >= 1,
            "Receiver-contiguous mode cannot fit a single K-row slice "
            "(n_per_recv={}, tile_bytes={}, stage_third={} B). Reduce n_per_recv or grow num_global_cb_receivers.",
            n_per_recv,
            tile_bytes,
            stage_third);
        num_sub = k_block_w_tiles / rows_per_sub;
    }

    const uint32_t sub_chunk_bytes = rows_per_sub * n_per_recv * tile_bytes;
    TT_FATAL(
        sub_chunk_bytes <= stage_third,
        "Internal: receiver-contiguous chunk size {} B exceeds stage_third {} B",
        sub_chunk_bytes,
        stage_third);

    // target_per_visit_pages: ~6 stage thirds' worth of blocks per visit. The
    // kernel amortizes one noc_async_write_one_packet_set_state per receiver
    // visit, so making each visit cover many blocks reduces set_state cost.
    // The kernel further clamps by free downstream space, remaining blocks,
    // and fifo-wrap distance, so the static ceiling is a hint, not a contract.
    const uint32_t page_bytes_per_recv = k_block_w_tiles * coalesced_num_pages * coalesced_page_size;
    const uint32_t stage_slot_pages = page_bytes_per_recv > 0 ? stage_third / page_bytes_per_recv : 0;
    constexpr uint32_t kVisitStageSlotMultiplier = 6;
    uint32_t target_per_visit_pages = stage_slot_pages * kVisitStageSlotMultiplier;
    if (target_per_visit_pages == 0) {
        target_per_visit_pages = 1;
    }

    TensorPrefetcherTensorLayout g;
    g.num_sub = num_sub;
    g.M = 1;  // unused in recv-contig (no N-chunking)
    g.rows_per_sub = rows_per_sub;
    g.coalesced_page_size = coalesced_page_size;
    g.coalesced_num_pages = coalesced_num_pages;
    g.sub_chunk_bytes = sub_chunk_bytes;
    g.sub_stride_bytes = rows_per_sub * n_per_recv * tile_bytes;
    g.block_stride_bytes = k_block_w_tiles * n_per_recv * tile_bytes;  // within-slab K-block stride
    g.page_bytes_per_recv = page_bytes_per_recv;
    g.layout_mode = static_cast<uint32_t>(LayoutMode::ReceiverContiguous);
    g.target_per_visit_pages = target_per_visit_pages;
    g.recv_stride_bytes = recv_stride_bytes;
    g.block_count = block_count;
    return g;
}

TensorPrefetcherTensorLayout compute_tensor_layout(
    const MeshTensor& t,
    uint32_t block_count,
    uint32_t num_banks,
    uint32_t receivers_per_bank,
    uint32_t total_receivers,
    uint32_t ring_half,
    uint32_t stage_third,
    ContextId context_id) {
    const auto* ref_buffer = t.mesh_buffer().get_reference_buffer();
    const LayoutMode mode = detect_layout_mode(t, *ref_buffer, total_receivers);
    if (mode == LayoutMode::KRowMajor) {
        // KRowMajor is single-sender-per-bank only, so receivers_per_bank is the bank's
        // full receiver count and the bank receiver counts must be uniform. (Receiver-
        // contiguous derives its geometry from the shard shape and ignores the receiver
        // count entirely, and tolerates non-uniform per-bank receivers.)
        TT_FATAL(
            num_banks > 0 && total_receivers % num_banks == 0,
            "Tensor prefetcher (K-row-major): total receivers ({}) must divide evenly across {} banks. "
            "Non-uniform per-bank receiver counts are only supported by the receiver-contiguous layout.",
            total_receivers,
            num_banks);
        return compute_tensor_layout_krow_major(t, block_count, receivers_per_bank, ring_half, context_id);
    }
    return compute_tensor_layout_recv_contig(t, block_count, stage_third, context_id);
}

}  // namespace

TensorPrefetcherManager::TensorPrefetcherManager(
    MeshDevice* mesh_device, std::function<std::lock_guard<std::mutex>()> lock_api_function) :
    mesh_device_(mesh_device), lock_api_function_(std::move(lock_api_function)) {
    TT_FATAL(mesh_device_ != nullptr, "TensorPrefetcherManager requires a non-null MeshDevice");
    TT_FATAL(static_cast<bool>(lock_api_function_), "TensorPrefetcherManager requires a valid lock_api_function");
}

TensorPrefetcherManager::~TensorPrefetcherManager() { stop(); }

void TensorPrefetcherManager::enumerate_dram_senders() {
    const auto context_id = mesh_device_->impl().get_context_id();
    const auto& soc_desc = MetalContext::instance(context_id)
                               .get_cluster()
                               .get_soc_desc(mesh_device_->get_view().get_devices().front()->id());
    const uint32_t num_banks = soc_desc.get_num_dram_views();
    num_banks_ = num_banks;
    sender_logical_cores_.clear();
    sender_logical_cores_.reserve((dual_senders_per_bank_ ? 2 : 1) * num_banks);
    for (uint32_t b = 0; b < num_banks; ++b) {
        if (dual_senders_per_bank_) {
            // Two senders per bank: the free subchannel then the NOC1-endpoint subchannel.
            // Must match the GCB factory's build_dram_sender_mapping ordering.
            for (const CoreCoord& core : mesh_device_->impl().dram_sender_logical_cores(b)) {
                sender_logical_cores_.push_back(core);
            }
        } else {
            sender_logical_cores_.push_back(mesh_device_->impl().pick_unused_dram_logical_core(b));
        }
    }
    num_senders_ = static_cast<uint32_t>(sender_logical_cores_.size());
}

void TensorPrefetcherManager::allocate_sockets() {
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

void TensorPrefetcherManager::build_and_launch_programs(uint32_t stage_ring_base, uint32_t stage_ring_size) {
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
                cq_signal_l1_addr_,
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

void TensorPrefetcherManager::start(const experimental::TensorPrefetcherConfig& config) {
    auto lock = lock_api_function_();
    dual_senders_per_bank_ = config.dual_senders_per_bank;
    TT_FATAL(!active_, "A Tensor prefetcher is already active on this mesh device. Call StopTensorPrefetcher first.");

    const auto& hal = MetalContext::instance(mesh_device_->impl().get_context_id()).hal();
    TT_FATAL(
        hal.has_programmable_core_type(HalProgrammableCoreType::DRAM),
        "Tensor prefetcher requires programmable DRAM cores; set "
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
    // Per-CQ signal slots at the front of the region: a small uint32 counter per
    // command queue, written by the dispatcher for WaitForCqOnTensorPrefetcher
    // and polled by the kernel's WAIT_CQ handler.
    const uint32_t cq_signal_bytes = align_up(kNumCqSignalSlots * sizeof(uint32_t), l1_alignment);
    cq_signal_l1_addr_ = align_up(kernel_region_base, l1_alignment);
    socket_config_l1_addr_ = align_up(cq_signal_l1_addr_ + cq_signal_bytes, pcie_alignment_for_layout);
    socket_data_l1_addr_ = align_up(socket_config_l1_addr_ + socket_config_bytes, pcie_alignment_for_layout);
    stage_ring_base_ = align_up(socket_data_l1_addr_ + socket_data_bytes, l1_alignment);
    const uint32_t kernel_region_end = kernel_region_base + kernel_region_size;
    TT_FATAL(
        stage_ring_base_ < kernel_region_end,
        "DRISC L1 kernel region ({} B) too small for socket buffers + stage ring",
        kernel_region_size);
    stage_ring_size_ = kernel_region_end - stage_ring_base_;
    // Align stage_ring_size to LCM(2, 3, l1_alignment) so both halves (K-row
    // path) and thirds (recv-contig path) are individually l1-aligned. With
    // l1_alignment=16, LCM = 48. Bitmask shortcut doesn't work because 48 is
    // not a power of 2 — use integer division.
    const uint32_t kRingSizeAlign = 6u * l1_alignment;  // = LCM(2*3, l1_alignment) when l1_alignment is even
    stage_ring_size_ = (stage_ring_size_ / kRingSizeAlign) * kRingSizeAlign;
    // After aligning down, make sure the ring is still big enough for at least
    // one minimal sub-chunk per slot. Catches accidental shrink-to-zero if the
    // socket carve-out ever grows past the L1 region.
    TT_FATAL(
        stage_ring_size_ >= 6 * l1_alignment,
        "DRISC L1 stage ring shrank to {} B after alignment — socket buffers consumed too much of the {} B "
        "kernel region",
        stage_ring_size_,
        kernel_region_size);
    ring_half_ = stage_ring_size_ / 2;
    stage_third_ = stage_ring_size_ / 3;

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
    host_worker_ = std::thread(&TensorPrefetcherManager::worker_loop, this);
    active_ = true;
}

MeshCoordinateRangeSet TensorPrefetcherManager::full_mesh_subset() const {
    MeshCoordinateRangeSet out;
    out.merge(MeshCoordinateRange(mesh_device_->shape()));
    return out;
}

std::vector<std::vector<uint8_t>> TensorPrefetcherManager::serialize_request_pages(
    const experimental::GlobalCircularBuffer& gcb,
    const std::vector<experimental::TensorPrefetcherInput>& data_tensors) const {
    TT_FATAL(!data_tensors.empty(), "QueueTensorPrefetcherRequest requires at least one tensor");

    const ContextId context_id = mesh_device_->impl().get_context_id();

    // Derive the receiver counts from the GCB itself so each Queue call can target a
    // GCB with a different receiver count. total_receivers (== ring_size) and
    // receivers_per_bank are independent of how many DRISC senders drive a bank; the
    // per-sender split (when dual_senders_per_bank_) just partitions a bank's receivers.
    const auto& mapping = gcb.sender_receiver_core_mapping();
    uint32_t total_receivers = 0;
    for (const auto& [_sender, receivers] : mapping) {
        total_receivers += receivers.num_cores();
    }
    TT_FATAL(num_banks_ > 0, "Tensor prefetcher: num_banks must be > 0");
    // receivers_per_bank is only consumed by the K-row-major layout (single sender per
    // bank). Recv-contig derives its geometry from the shard shape and ignores it, and
    // its receivers need not be uniform per bank — so the even-divisibility requirement
    // is enforced per-tensor in compute_tensor_layout(), only for K-row-major tensors.
    const uint32_t receivers_per_bank = total_receivers / num_banks_;
    const uint32_t gcb_state_addr = static_cast<uint32_t>(experimental::sender_state_drisc_l1_base(gcb));

    const uint32_t pcie_alignment = MetalContext::instance(context_id).hal().get_alignment(HalMemType::HOST);
    const uint32_t aligned_page_bytes = align_up(kRequestPageBytes, pcie_alignment);

    constexpr uint32_t kHeaderBytes = sizeof(TensorPrefetcherRequestHeader);
    constexpr uint32_t kEntryBytes = sizeof(TensorPrefetcherEntry);
    constexpr uint32_t kLayoutBytes = sizeof(TensorPrefetcherTensorLayout);

    std::vector<std::vector<uint8_t>> pages;

    // Per-page packing state. Entries grow forward from kHeaderBytes; the layout table
    // grows backward from kRequestPageBytes (layout i at kRequestPageBytes -
    // (i+1)*kLayoutBytes). `seen` holds this page's deduplicated layouts in index order.
    std::vector<uint8_t> page;
    uint32_t num_entries = 0;
    uint32_t num_layouts = 0;
    std::vector<TensorPrefetcherTensorLayout> seen;

    auto begin_page = [&]() {
        page.assign(aligned_page_bytes, 0);
        num_entries = 0;
        num_layouts = 0;
        seen.clear();
    };
    auto finalize_page = [&]() {
        auto* header = reinterpret_cast<TensorPrefetcherRequestHeader*>(page.data());
        header->base.cmd_id = DRAM_PREFETCHER_CMD_PREFETCH;
        header->prefetch.num_entries = static_cast<uint16_t>(num_entries);
        header->prefetch.num_layouts = num_layouts;
        header->prefetch.gcb_state_addr = gcb_state_addr;
        pages.push_back(std::move(page));
    };

    begin_page();
    for (size_t tensor_idx = 0; tensor_idx < data_tensors.size(); ++tensor_idx) {
        const auto& input = data_tensors[tensor_idx];
        // block_count is per-tensor: it sets how many K-blocks the kernel pushes
        // (and how K is divided in compute_tensor_layout), replacing the GCB ring size.
        const TensorPrefetcherTensorLayout layout = compute_tensor_layout(
            input.tensor.get(),
            input.block_count,
            num_banks_,
            receivers_per_bank,
            total_receivers,
            ring_half_,
            stage_third_,
            context_id);
        // dual_senders_per_bank only makes sense for the receiver-contiguous layout (a K-row-major
        // bank holds one shard, nothing to split). Reject the mismatch here rather than silently
        // building wrong per-sender geometry.
        TT_FATAL(
            !dual_senders_per_bank_ || layout.layout_mode == static_cast<uint32_t>(LayoutMode::ReceiverContiguous),
            "Tensor prefetcher: dual_senders_per_bank is only supported for the receiver-contiguous "
            "DRAM layout, but input tensor {} is K-row-major.",
            tensor_idx);

        // The sender's free-space poll counts whole per-receiver pages; if the GCB's per-receiver
        // fifo can't hold even one full page the poll never reaches a usable block and the DRISC
        // kernel hangs. Guard it here (applies to both layouts).
        TT_FATAL(
            gcb.size() >= layout.page_bytes_per_recv,
            "Tensor prefetcher: GCB per-receiver fifo size ({} B) must be at least one full per-receiver "
            "page ({} B) for input tensor {}; a smaller fifo makes the sender's free-space poll spin forever.",
            gcb.size(),
            layout.page_bytes_per_recv,
            tensor_idx);

        const uint32_t bank_local_base = static_cast<uint32_t>(input.tensor.get().mesh_buffer().address());

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

        TensorPrefetcherEntry entry;
        entry.bank_local_base = bank_local_base;
        entry.layout_index = static_cast<uint32_t>(layout_idx);
        std::memcpy(page.data() + (kHeaderBytes + num_entries * kEntryBytes), &entry, kEntryBytes);
        ++num_entries;
    }
    finalize_page();

    return pages;
}

void TensorPrefetcherManager::queue(
    const experimental::GlobalCircularBuffer& gcb,
    const std::optional<MeshCoordinateRangeSet>& device_subset,
    const std::vector<experimental::TensorPrefetcherInput>& tensors,
    std::optional<uint8_t> cq_id) {
    auto lock = lock_api_function_();
    TT_FATAL(active_, "QueueTensorPrefetcherRequest called before StartTensorPrefetcher");
    TT_FATAL(
        experimental::sender_core_type(gcb) == experimental::SenderCoreType::Dram,
        "QueueTensorPrefetcherRequest requires a DRAM-sender GlobalCircularBuffer");
    TT_FATAL(
        gcb.sender_receiver_core_mapping().size() == num_senders_,
        "GCB num_senders ({}) does not match prefetcher num_senders ({})",
        gcb.sender_receiver_core_mapping().size(),
        num_senders_);
    TT_FATAL(!tensors.empty(), "QueueTensorPrefetcherRequest requires at least one tensor");

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
                "QueueTensorPrefetcherRequest target MeshCoordinate {} is not in the mesh this prefetcher was "
                "started "
                "on",
                coord);
            target_devices.push_back(coord);
        }
    }

    // If the target command queue is mid trace-capture, capture this request into the trace
    // instead of sending it now; it is (re)sent on every replay of that trace. Otherwise send
    // immediately via the host worker.
    const std::optional<MeshTraceId> recording_trace_id = mesh_device_->mesh_command_queue(cq_id).trace_id();

    {
        // Push all pages of this call under one lock so they stay contiguous and ordered
        // (required for fifo_wr_ptr continuity across the split), whether captured or sent.
        std::lock_guard<std::mutex> lk(queue_mu_);
        for (auto& page : pages) {
            Request req;
            req.page = std::move(page);
            req.target_devices = target_devices;
            if (recording_trace_id.has_value()) {
                trace_requests_[*recording_trace_id].push_back(std::move(req));
            } else {
                pending_.push_back(std::move(req));
            }
        }
    }
    // Only the immediate path needs to wake the worker; captured requests wait for replay.
    if (!recording_trace_id.has_value()) {
        queue_cv_.notify_one();
    }
}

void TensorPrefetcherManager::replay_trace(const MeshTraceId& trace_id) {
    {
        std::lock_guard<std::mutex> lk(queue_mu_);
        auto it = trace_requests_.find(trace_id);
        if (it == trace_requests_.end()) {
            // No prefetcher requests were captured during this trace's capture.
            return;
        }
        // Copy (not move) so the captured requests survive for the next replay. Pushed in
        // capture order so fifo_wr_ptr continuity matches the original Queue calls.
        pending_.insert(pending_.end(), it->second.begin(), it->second.end());
    }
    queue_cv_.notify_one();
}

void TensorPrefetcherManager::enqueue_cq_signal_and_wait(
    uint8_t cq_id, const std::optional<MeshCoordinateRangeSet>& device_subset) {
    // Hold the API lock across this whole call. Three things must be atomic together:
    //   1. the counter bump (++cq_signal_counter_[cq_id]),
    //   2. the dispatcher write that pushes that value to the device, and
    //   3. the WAIT_CQ enqueue into pending_.
    // If the lock were dropped between them, two concurrent callers could interleave their
    // dispatcher writes out of counter order, and stop() could slip its STOP sentinel into
    // pending_ ahead of this WAIT_CQ request — worker_loop would then try_write() the
    // WAIT_CQ to a kernel that has already exited and spin forever (try_write has no
    // stop_requested_ check). queue() and stop() take the lock the same way, so all three
    // serialize. enqueue_write_dram_core_counter is documented to run under the caller's
    // api lock and does NOT re-lock, so holding it here does not self-deadlock.
    auto lock = lock_api_function_();
    TT_FATAL(active_, "WaitForCqOnTensorPrefetcher called before StartTensorPrefetcher");
    TT_FATAL(
        cq_id < cq_signal_counter_.size(),
        "WaitForCqOnTensorPrefetcher cq_id ({}) out of range [0, {})",
        cq_id,
        cq_signal_counter_.size());

    // Monotonic value for this signal; wrap is handled by the kernel's signed compare.
    const uint32_t signal_value = ++cq_signal_counter_[cq_id];

    // Resolve target device coords (subset if given, else full mesh).
    std::vector<MeshCoordinate> target_devices;
    const MeshCoordinateRangeSet effective_subset = device_subset.has_value() ? *device_subset : full_mesh_subset();
    for (const auto& range : effective_subset.ranges()) {
        for (const auto& coord : range) {
            TT_FATAL(
                device_index_by_coord_.contains(coord),
                "WaitForCqOnTensorPrefetcher target MeshCoordinate {} is not in the mesh this prefetcher was "
                "started on",
                coord);
            target_devices.push_back(coord);
        }
    }

    // Destination of the dispatcher write: the full device address (the kernel's
    // local slot base plus the programmable-DRAM-core L1 NOC offset), so the
    // dispatcher must NOT apply a bank offset.
    const uint64_t dram_l1_noc_offset = MetalContext::instance(mesh_device_->impl().get_context_id())
                                            .hal()
                                            .get_l1_noc_offset(HalProgrammableCoreType::DRAM);
    const uint64_t slot_addr = static_cast<uint64_t>(cq_signal_l1_addr_) +
                               static_cast<uint64_t>(cq_id) * sizeof(uint32_t) + dram_l1_noc_offset;

    std::vector<DeviceMemoryAddress> targets;
    targets.reserve(target_devices.size() * num_senders_);
    for (const auto& coord : target_devices) {
        IDevice* device = devices_[device_index_by_coord_.at(coord)];
        for (uint32_t s = 0; s < num_senders_; ++s) {
            const CoreCoord virtual_core =
                device->virtual_core_from_logical_core(sender_logical_cores_[s], CoreType::DRAM);
            targets.push_back(DeviceMemoryAddress{coord, virtual_core, slot_addr});
        }
    }

    // (a) Dispatcher write: bump every target DRAM core's signal slot for this CQ. Runs
    // under the api lock we already hold (the method does not re-lock).
    mesh_device_->impl().mesh_command_queue_base(cq_id).enqueue_write_dram_core_counter(
        tt::stl::Span<const DeviceMemoryAddress>(targets), signal_value, /*blocking=*/false);

    // (b) Queue a WAIT_CQ request. It rides the same async worker path as prefetch
    // requests, so it lands in each socket's FIFO ahead of the next prefetch request;
    // the kernel blocks on it until it observes signal_value.
    const uint32_t pcie_alignment =
        MetalContext::instance(mesh_device_->impl().get_context_id()).hal().get_alignment(HalMemType::HOST);
    const uint32_t page_bytes = align_up(kRequestPageBytes, pcie_alignment);
    Request req;
    req.page.assign(page_bytes, 0);
    auto* header = reinterpret_cast<TensorPrefetcherRequestHeader*>(req.page.data());
    header->base.cmd_id = DRAM_PREFETCHER_CMD_WAIT_CQ;
    header->wait_cq.cq_index = cq_id;
    header->wait_cq.cq_wait_value = signal_value;
    req.target_devices = std::move(target_devices);

    {
        std::lock_guard<std::mutex> lk(queue_mu_);
        pending_.push_back(std::move(req));
    }
    queue_cv_.notify_one();
}

void TensorPrefetcherManager::release_trace(const MeshTraceId& trace_id) {
    std::lock_guard<std::mutex> lk(queue_mu_);
    trace_requests_.erase(trace_id);
}

void TensorPrefetcherManager::worker_loop() {
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
                "QueueTensorPrefetcherRequest target MeshCoordinate {} is not in the mesh this prefetcher was "
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

void TensorPrefetcherManager::stop() {
    // Note: the MeshDevice close path (close_impl) calls stop() without holding
    // api_mutex_, and the destructor only reaches here after active_ is already
    // false, so taking the lock here never self-deadlocks.
    auto lock = lock_api_function_();
    if (!active_) {
        return;
    }
    // Stop = a zero-filled page broadcast to every device in the mesh. The leading
    // command id byte is 0 == DRAM_PREFETCHER_CMD_STOP, so the kernel exits its
    // request loop on it. The worker_loop returns once pending is drained and
    // stop_requested_ is set.
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
    trace_requests_.clear();
    num_senders_ = 0;
    active_ = false;
}

}  // namespace tt::tt_metal::distributed

// -----------------------------------------------------------------------------
// experimental::StartTensorPrefetcher / Queue / Stop
// -----------------------------------------------------------------------------
namespace tt::tt_metal::experimental {

bool IsTensorPrefetcherSupported(const distributed::MeshDevice& mesh_device) {
    const auto& hal = MetalContext::instance(mesh_device.impl().get_context_id()).hal();
    return hal.has_programmable_core_type(HalProgrammableCoreType::DRAM);
}

void StartTensorPrefetcher(distributed::MeshDevice& mesh_device, const TensorPrefetcherConfig& config) {
    auto& manager = mesh_device.impl().tensor_prefetcher(&mesh_device);
    manager.start(config);
}

void QueueTensorPrefetcherRequest(
    distributed::MeshDevice& mesh_device,
    const GlobalCircularBuffer& gcb,
    const std::optional<distributed::MeshCoordinateRangeSet>& device_subset,
    const std::vector<TensorPrefetcherInput>& input_tensors,
    std::optional<uint8_t> cq_id) {
    auto& manager = mesh_device.impl().tensor_prefetcher(&mesh_device);
    manager.queue(gcb, device_subset, input_tensors, cq_id);
}

void WaitForCqOnTensorPrefetcher(
    distributed::MeshDevice& mesh_device,
    uint8_t cq_id,
    const std::optional<distributed::MeshCoordinateRangeSet>& device_subset) {
    auto& manager = mesh_device.impl().tensor_prefetcher(&mesh_device);
    manager.enqueue_cq_signal_and_wait(cq_id, device_subset);
}

void StopTensorPrefetcher(distributed::MeshDevice& mesh_device) {
    auto& manager = mesh_device.impl().tensor_prefetcher(&mesh_device);
    manager.stop();
}

}  // namespace tt::tt_metal::experimental
