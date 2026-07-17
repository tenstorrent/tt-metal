// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/buffers/tensor_prefetcher_manager.hpp"

#include "distributed/mesh_device_impl.hpp"
#include "distributed/mesh_command_queue_base.hpp"
#include "impl/buffers/drisc_l1_arena.hpp"
#include "impl/buffers/h2d_socket_internal.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

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

// Validate a streaming (receiver-contiguous) weight and return the shard distribution strategy that
// governs how the host maps a receiver's (bank, bank-local slab index) to a global receiver position
// when slicing the rotation table. This is the consumer's concept (a ring matmul calls it a "ring
// position") — the GCB stays order-agnostic (see receiver_slab_indices). TT_FATALs on a
// non-recv-contig (no BDS) tensor or an unsupported distribution strategy; only the two strategies
// below reach the packing loop:
//   ROUND_ROBIN_1D (strided):    global = bank + slab_idx * num_banks
//   CONTIGUOUS_1D  (contiguous): global = bank * receivers_per_bank + slab_idx
ShardDistributionStrategy shard_strategy_for_streaming_tensor(const MeshTensor& t, uint32_t tensor_idx) {
    const auto* ref_buffer = t.mesh_buffer().get_reference_buffer();
    const auto& bds_opt = ref_buffer->buffer_distribution_spec();
    TT_FATAL(
        bds_opt.has_value(),
        "Streaming Tensor prefetcher tensor {} must be a receiver-contiguous (nd-sharded) weight, but it has no "
        "buffer distribution spec (it looks K-row-major / legacy-sharded).",
        tensor_idx);
    const auto strategy = bds_opt->shard_distribution_strategy();
    TT_FATAL(
        strategy == ShardDistributionStrategy::ROUND_ROBIN_1D || strategy == ShardDistributionStrategy::CONTIGUOUS_1D,
        "Streaming Tensor prefetcher tensor {} uses an unsupported shard distribution strategy ({}); only "
        "ROUND_ROBIN_1D (strided) and CONTIGUOUS_1D (contiguous) receiver-contiguous weights are supported.",
        tensor_idx,
        static_cast<int>(strategy));
    return strategy;
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
    sender_logical_cores_.reserve(2 * num_banks);
    for (uint32_t b = 0; b < num_banks; ++b) {
        // Two senders per bank: the free subchannel then the NOC1-endpoint subchannel.
        // A queued GCB may use the primary only or both; PREFETCH fan-out targets its mapping.
        for (const CoreCoord& core : mesh_device_->impl().dram_sender_logical_cores(b)) {
            sender_logical_cores_.push_back(core);
        }
    }
    num_senders_ = static_cast<uint32_t>(sender_logical_cores_.size());
}

std::vector<uint32_t> TensorPrefetcherManager::sender_indices_for_gcb(
    const experimental::GlobalCircularBuffer& gcb) const {
    const auto& mapping = gcb.sender_receiver_core_mapping();
    TT_FATAL(!mapping.empty(), "Tensor prefetcher: GCB sender mapping must not be empty");

    std::vector<uint32_t> sender_indices;
    sender_indices.reserve(mapping.size());
    for (const auto& [sender, _receivers] : mapping) {
        const auto it = std::find(sender_logical_cores_.begin(), sender_logical_cores_.end(), sender);
        TT_FATAL(
            it != sender_logical_cores_.end(),
            "Tensor prefetcher: GCB sender core ({}, {}) is not one of the {} provisioned DRAM sender cores",
            sender.x,
            sender.y,
            num_senders_);
        sender_indices.push_back(static_cast<uint32_t>(std::distance(sender_logical_cores_.begin(), it)));
    }
    return sender_indices;
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

void TensorPrefetcherManager::start() {
    auto lock = lock_api_function_();
    TT_FATAL(!active_, "A Tensor prefetcher is already active on this mesh device. Call StopTensorPrefetcher first.");

    const auto& hal = MetalContext::instance(mesh_device_->impl().get_context_id()).hal();
    TT_FATAL(
        hal.has_programmable_core_type(HalProgrammableCoreType::DRAM),
        "Tensor prefetcher requires programmable DRAM cores, which auto-enable on Blackhole with firmware "
        ">= 19.12.0.0 and either no harvested DRAM channels or a single device");

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

std::vector<std::vector<std::vector<uint8_t>>> TensorPrefetcherManager::serialize_request_pages(
    const experimental::GlobalCircularBuffer& gcb,
    const std::vector<experimental::TensorPrefetcherInput>& data_tensors) const {
    TT_FATAL(!data_tensors.empty(), "QueueTensorPrefetcherRequest requires at least one tensor");

    const ContextId context_id = mesh_device_->impl().get_context_id();

    // Derive the receiver counts from the GCB itself so each Queue call can target a
    // GCB with a different receiver count. total_receivers (== ring_size) and
    // receivers_per_bank are independent of how many DRISC senders drive a bank.
    const auto& mapping = gcb.sender_receiver_core_mapping();
    uint32_t total_receivers = 0;
    for (const auto& [_sender, receivers] : mapping) {
        total_receivers += receivers.num_cores();
    }
    TT_FATAL(num_banks_ > 0, "Tensor prefetcher: num_banks must be > 0");
    // K-row-major stores one complete per-bank shard, so all receivers for a bank must
    // remain on its primary sender. Receiver-contiguous tensors may use either this
    // topology or a dual-sender split.
    bool krow_compatible_mapping = mapping.size() == num_banks_;
    std::vector<bool> primary_bank_seen(num_banks_, false);
    if (krow_compatible_mapping) {
        for (const auto& [sender, _receivers] : mapping) {
            const uint32_t bank = static_cast<uint32_t>(sender.x);
            if (bank >= num_banks_ || primary_bank_seen[bank] || sender != sender_logical_cores_[2 * bank]) {
                krow_compatible_mapping = false;
                break;
            }
            primary_bank_seen[bank] = true;
        }
    }
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

    // max_receivers sizes the uniform rotation slot so every sender's page packs identically
    // (dedup/fit decisions below are sender-independent); the kernel recovers it from the GCB's
    // max_num_receivers. It is just the largest receiver count over the GCB's senders.
    uint32_t max_receivers = 0;
    for (const auto& [_sender, receivers] : mapping) {
        max_receivers = std::max(max_receivers, receivers.num_cores());
    }
    const uint32_t layout_stride = kLayoutBytes + max_receivers * static_cast<uint32_t>(sizeof(uint32_t));

    // Per-GCB-sender bank-local slab index map, needed only when a tensor streams. The GCB owns the
    // recv_index_base accounting, so the slab indices come from its experimental accessor (single
    // source of truth) rather than being re-derived here. It is order-agnostic: this function maps
    // each receiver's (bank, slab index) to a global receiver position per tensor using that
    // tensor's shard distribution.
    bool any_streaming = false;
    for (const auto& input : data_tensors) {
        if (!input.rotation.empty()) {
            any_streaming = true;
            break;
        }
    }
    std::vector<std::vector<uint32_t>> slab_idx_by_sender;
    if (any_streaming) {
        slab_idx_by_sender = experimental::receiver_slab_indices(gcb);
        TT_FATAL(
            slab_idx_by_sender.size() == mapping.size(),
            "Tensor prefetcher: GCB returned {} sender slab-index lists for {} sender mappings",
            slab_idx_by_sender.size(),
            mapping.size());

        // Both the strided and contiguous (bank, slab index) -> global position formulas are
        // bijections onto [0, total_receivers) only when the DRAM banks are dense 0..num_banks-1 and
        // every bank has exactly receivers_per_bank receivers. Guard that topology invariant once
        // here so the per-sender rotation fill below is a plain gather with no inner-loop range check.
        std::vector<uint32_t> bank_receiver_count(num_banks_, 0);
        for (const auto& [sender_logical, receivers] : mapping) {
            const uint32_t bank = static_cast<uint32_t>(sender_logical.x);
            TT_FATAL(
                bank < num_banks_,
                "Tensor prefetcher: streaming requires dense DRAM bank ids 0..{}, but a sender occupies bank {}.",
                num_banks_ - 1,
                bank);
            bank_receiver_count[bank] += receivers.num_cores();
        }
        for (uint32_t b = 0; b < num_banks_; ++b) {
            TT_FATAL(
                bank_receiver_count[b] == receivers_per_bank,
                "Tensor prefetcher: streaming requires a uniform receiver-contiguous topology — bank {} has {} "
                "receivers but expected receivers_per_bank ({} = total_receivers {} / num_banks {}).",
                b,
                bank_receiver_count[b],
                receivers_per_bank,
                total_receivers,
                num_banks_);
        }
    }
    TT_FATAL(
        kHeaderBytes + layout_stride + kEntryBytes <= kRequestPageBytes,
        "Tensor prefetcher: request page ({} B) too small for one tensor: header({}) + layout slot ({} = "
        "geometry {} + {} rotation entries) + entry({}). Reduce receivers per sender or grow kRequestPageBytes.",
        kRequestPageBytes,
        kHeaderBytes,
        layout_stride,
        kLayoutBytes,
        max_receivers,
        kEntryBytes);

    // ---- Abstract page plan (sender-independent): entries + dedup'd geometry+rotation slots ----
    struct Slot {
        TensorPrefetcherTensorLayout geom;
        std::vector<uint32_t> rotation;  // caller's global rotation (total_receivers entries), or empty == batched
        // Shard distribution used to slice `rotation` per receiver; only meaningful when streaming. Part
        // of slot identity: two tensors with the same geometry+rotation but different strategies pack
        // different per-sender rotation bytes, so they must not dedup together.
        ShardDistributionStrategy strategy = ShardDistributionStrategy::ROUND_ROBIN_1D;
    };
    struct PlanEntry {
        uint32_t bank_local_base = 0;
        uint32_t layout_index = 0;
    };
    struct PagePlan {
        std::vector<PlanEntry> entries;
        std::vector<Slot> slots;
    };
    auto slot_equal = [](const Slot& a, const Slot& b) {
        return layout_equal(a.geom, b.geom) && a.rotation == b.rotation && a.strategy == b.strategy;
    };
    std::vector<PagePlan> plans(1);

    for (size_t tensor_idx = 0; tensor_idx < data_tensors.size(); ++tensor_idx) {
        const auto& input = data_tensors[tensor_idx];
        const bool streaming = !input.rotation.empty();
        // Streaming delivers block (rotation[r] + p) mod block_count, only a valid permutation when
        // block_count == ring_size (== total_receivers). The consuming ring matmul always uses
        // num_blocks = ring_size, so this holds for the intended use.
        if (streaming) {
            TT_FATAL(
                input.block_count == total_receivers,
                "Streaming Tensor prefetcher requires block_count ({}) == ring_size ({}) for tensor {}",
                input.block_count,
                total_receivers,
                tensor_idx);
            TT_FATAL(
                input.rotation.size() == total_receivers,
                "Streaming rotation for tensor {} has {} entries but must have total_receivers ({}); it is indexed "
                "by global receiver position.",
                tensor_idx,
                input.rotation.size(),
                total_receivers);
            for (uint32_t v : input.rotation) {
                TT_FATAL(
                    v < input.block_count,
                    "Streaming rotation entry {} for tensor {} is out of range [0, block_count={}).",
                    v,
                    tensor_idx,
                    input.block_count);
            }
        }
        // block_count is per-tensor: it sets how many K-blocks the kernel pushes
        // (and how K is divided in compute_tensor_layout), replacing the GCB ring size.
        TensorPrefetcherTensorLayout layout = compute_tensor_layout(
            input.tensor.get(),
            input.block_count,
            num_banks_,
            receivers_per_bank,
            total_receivers,
            ring_half_,
            stage_third_,
            context_id);
        // Streaming is a per-tensor delivery attribute carried in the layout flag; the appended
        // rotation participates in dedup (slot_equal), so tensors that differ only in rotation get
        // distinct slots.
        layout.streaming = streaming ? 1u : 0u;
        TT_FATAL(
            layout.layout_mode != static_cast<uint32_t>(LayoutMode::KRowMajor) || krow_compatible_mapping,
            "Tensor prefetcher: K-row-major input tensor {} requires exactly one primary sender per DRAM bank; "
            "the supplied GCB uses a split or incompatible sender topology.",
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

        Slot slot;
        slot.geom = layout;
        if (streaming) {
            TT_FATAL(
                layout.layout_mode == static_cast<uint32_t>(LayoutMode::ReceiverContiguous),
                "Streaming Tensor prefetcher requires a receiver-contiguous weight, but input tensor {} is "
                "K-row-major.",
                tensor_idx);
            slot.rotation = input.rotation;
            slot.strategy = shard_strategy_for_streaming_tensor(input.tensor.get(), static_cast<uint32_t>(tensor_idx));
        }

        // Find this slot in the current page (dedup), or decide it needs adding.
        PagePlan* plan = &plans.back();
        int32_t slot_idx = -1;
        for (uint32_t i = 0; i < plan->slots.size(); ++i) {
            if (slot_equal(plan->slots[i], slot)) {
                slot_idx = static_cast<int32_t>(i);
                break;
            }
        }
        const uint32_t need = kEntryBytes + (slot_idx < 0 ? layout_stride : 0);
        const uint32_t entry_high = kHeaderBytes + static_cast<uint32_t>(plan->entries.size()) * kEntryBytes;
        const uint32_t layout_low = kRequestPageBytes - static_cast<uint32_t>(plan->slots.size()) * layout_stride;
        if (need > layout_low - entry_high) {
            // No room in the current page — start a fresh one. The slot is page-local, so it
            // becomes a new slot in the next page.
            plans.emplace_back();
            plan = &plans.back();
            slot_idx = -1;
        }
        if (slot_idx < 0) {
            slot_idx = static_cast<int32_t>(plan->slots.size());
            plan->slots.push_back(std::move(slot));
        }
        const uint32_t bank_local_base = static_cast<uint32_t>(input.tensor.get().mesh_buffer().address());
        plan->entries.push_back(PlanEntry{bank_local_base, static_cast<uint32_t>(slot_idx)});
    }

    // ---- Materialize each logical page into one byte buffer per sender ----
    // Header/entry/geometry bytes are identical across senders; only each slot's rotation region
    // differs (this sender's slice of the caller's global rotation). A page whose every slot is
    // batched (no rotation) is byte-identical for all mapped GCB senders, so it is emitted once
    // rather than making identical copies.
    std::vector<std::vector<std::vector<uint8_t>>> pages;
    pages.reserve(plans.size());
    for (const auto& plan : plans) {
        bool page_has_rotation = false;
        for (const auto& slot : plan.slots) {
            if (!slot.rotation.empty()) {
                page_has_rotation = true;
                break;
            }
        }
        // Build the sender-independent template once (header + entries + each slot's geometry,
        // rotation regions left zero); each sender's page is a copy with only its rotation slices
        // overwritten. Avoids re-stamping the identical header/entry/geometry bytes per sender.
        std::vector<uint8_t> templ(aligned_page_bytes, 0);
        auto* header = reinterpret_cast<TensorPrefetcherRequestHeader*>(templ.data());
        header->base.cmd_id = DRAM_PREFETCHER_CMD_PREFETCH;
        header->prefetch.num_entries = static_cast<uint16_t>(plan.entries.size());
        header->prefetch.num_layouts = static_cast<uint32_t>(plan.slots.size());
        header->prefetch.gcb_state_addr = gcb_state_addr;
        for (uint32_t k = 0; k < plan.entries.size(); ++k) {
            TensorPrefetcherEntry entry;
            entry.bank_local_base = plan.entries[k].bank_local_base;
            entry.layout_index = plan.entries[k].layout_index;
            std::memcpy(templ.data() + (kHeaderBytes + k * kEntryBytes), &entry, kEntryBytes);
        }
        for (uint32_t i = 0; i < plan.slots.size(); ++i) {
            const uint32_t slot_start = kRequestPageBytes - (i + 1) * layout_stride;
            std::memcpy(templ.data() + slot_start, &plan.slots[i].geom, kLayoutBytes);
        }

        const uint32_t num_variants = page_has_rotation ? static_cast<uint32_t>(mapping.size()) : 1u;
        std::vector<std::vector<uint8_t>> per_sender(num_variants);
        for (uint32_t s = 0; s < num_variants; ++s) {
            std::vector<uint8_t> page = templ;
            if (page_has_rotation) {
                const auto& slab = slab_idx_by_sender[s];
                const uint32_t bank = static_cast<uint32_t>(mapping[s].first.x);
                for (uint32_t i = 0; i < plan.slots.size(); ++i) {
                    if (plan.slots[i].rotation.empty()) {
                        continue;
                    }
                    const uint32_t slot_start = kRequestPageBytes - (i + 1) * layout_stride;
                    // Map each receiver's (bank, bank-local slab index) to its global receiver
                    // position, then gather this sender's slice of the caller's global rotation. The
                    // topology guard above makes both formulas a bijection onto [0, total_receivers),
                    // so no inner-loop range check is needed. Only ROUND_ROBIN_1D (strided) and
                    // CONTIGUOUS_1D reach here (see shard_strategy_for_streaming_tensor).
                    const bool strided = plan.slots[i].strategy == ShardDistributionStrategy::ROUND_ROBIN_1D;
                    auto* rot = reinterpret_cast<uint32_t*>(page.data() + slot_start + kLayoutBytes);
                    for (uint32_t r = 0; r < slab.size(); ++r) {
                        const uint32_t g =
                            strided ? (bank + slab[r] * num_banks_) : (bank * receivers_per_bank + slab[r]);
                        rot[r] = plan.slots[i].rotation[g];
                    }
                }
            }
            per_sender[s] = std::move(page);
        }
        pages.push_back(std::move(per_sender));
    }

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
    const std::vector<uint32_t> target_sender_indices = sender_indices_for_gcb(gcb);
    TT_FATAL(!tensors.empty(), "QueueTensorPrefetcherRequest requires at least one tensor");

    // A Queue call may span more tensors than fit in one socket page; serialize into one
    // or more pages, each an independent request. The per-GCB fifo_wr_ptr persists across
    // requests, so the split is invisible to the receiver. A streaming logical page is materialized
    // per mapped GCB sender because each carries a different rotation slice.
    std::vector<std::vector<std::vector<uint8_t>>> pages = serialize_request_pages(gcb, tensors);

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
        for (auto& sender_pages : pages) {
            Request req;
            req.sender_pages = std::move(sender_pages);
            req.target_devices = target_devices;
            req.target_sender_indices = target_sender_indices;
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
        ttsl::Span<const DeviceMemoryAddress>(targets), signal_value, /*blocking=*/false);

    // (b) Queue a WAIT_CQ request. It rides the same async worker path as prefetch
    // requests, so it lands in each socket's FIFO ahead of the next prefetch request;
    // the kernel blocks on it until it observes signal_value.
    const uint32_t pcie_alignment =
        MetalContext::instance(mesh_device_->impl().get_context_id()).hal().get_alignment(HalMemType::HOST);
    const uint32_t page_bytes = align_up(kRequestPageBytes, pcie_alignment);
    // WAIT_CQ has no rotation, so one page broadcast to every sender (sender_pages size 1).
    Request req;
    req.sender_pages.assign(1, std::vector<uint8_t>(page_bytes, 0));
    auto* header = reinterpret_cast<TensorPrefetcherRequestHeader*>(req.sender_pages[0].data());
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

        // PREFETCH targets only the sender cores mapped by its GCB. STOP / WAIT_CQ leave
        // target_sender_indices empty and broadcast to every provisioned sender.
        const bool fanout_all_senders = req.target_sender_indices.empty();
        TT_FATAL(!req.sender_pages.empty(), "Tensor prefetcher worker received a request with no socket page");
        TT_FATAL(
            fanout_all_senders || req.sender_pages.size() == 1 ||
                req.sender_pages.size() == req.target_sender_indices.size(),
            "Tensor prefetcher request has {} sender pages for {} target senders",
            req.sender_pages.size(),
            req.target_sender_indices.size());
        struct TargetSocket {
            uint32_t socket_index = 0;
            uint32_t page_index = 0;
        };
        std::vector<TargetSocket> remaining_target_sockets;
        const uint32_t senders_per_device =
            fanout_all_senders ? num_senders_ : static_cast<uint32_t>(req.target_sender_indices.size());
        remaining_target_sockets.reserve(req.target_devices.size() * senders_per_device);
        for (const auto& dev_coord : req.target_devices) {
            auto it = device_index_by_coord_.find(dev_coord);
            TT_FATAL(
                it != device_index_by_coord_.end(),
                "QueueTensorPrefetcherRequest target MeshCoordinate {} is not in the mesh this prefetcher was "
                "started "
                "on; would silently drop the request for that device",
                dev_coord);
            const uint32_t d = it->second;
            if (fanout_all_senders) {
                for (uint32_t s = 0; s < num_senders_; ++s) {
                    remaining_target_sockets.push_back(TargetSocket{d * num_senders_ + s, 0});
                }
            } else {
                for (uint32_t page_index = 0; page_index < req.target_sender_indices.size(); ++page_index) {
                    const uint32_t sender_index = req.target_sender_indices[page_index];
                    TT_FATAL(
                        sender_index < num_senders_,
                        "Tensor prefetcher request targets sender index {} but only {} senders are provisioned",
                        sender_index,
                        num_senders_);
                    remaining_target_sockets.push_back(
                        TargetSocket{d * num_senders_ + sender_index, req.sender_pages.size() == 1 ? 0u : page_index});
                }
            }
        }

        // Round-robin try_write with non-blocking attempts.
        std::vector<TargetSocket> still_pending = std::move(remaining_target_sockets);
        while (!still_pending.empty()) {
            std::vector<TargetSocket> next_pending;
            for (const TargetSocket& target : still_pending) {
                std::vector<uint8_t>& page = req.sender_pages[target.page_index];
                if (experimental::detail::try_write(*sockets_[target.socket_index], page.data(), 1)) {
                    // Wrote successfully.
                } else {
                    next_pending.push_back(target);
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
    // STOP is all-zero (cmd_id 0) and rotation-free, so one page broadcast to every sender.
    sentinel.sender_pages.assign(1, std::vector<uint8_t>(page_bytes, 0));
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

void StartTensorPrefetcher(distributed::MeshDevice& mesh_device, const TensorPrefetcherConfig&) {
    auto& manager = mesh_device.impl().tensor_prefetcher(&mesh_device);
    manager.start();
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
