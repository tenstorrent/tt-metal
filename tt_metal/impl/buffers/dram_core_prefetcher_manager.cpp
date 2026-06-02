// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/buffers/dram_core_prefetcher_manager.hpp"

#include "distributed/mesh_device_impl.hpp"
#include "impl/buffers/drisc_l1_arena.hpp"

#include <cstdint>
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
#include <tt-metalium/experimental/global_circular_buffer.hpp>
#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>
#include <tt_stl/assert.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"  // DramConfig + CreateKernel(DramConfig)

namespace tt::tt_metal::distributed {

namespace {

constexpr uint32_t kRemoteCBId = 31;

// Bytes reserved above the arena's kernel_working_region_base for noc_xy + config and
// alignment slack. The remainder of the kernel region is the budget for the two
// ping-pong stage buffers (2 * chunk_size).
constexpr uint32_t kKernelOverheadBytes = 2 * 1024;

constexpr const char* kKernelPath = "tt_metal/impl/buffers/kernels/dram_core_prefetcher.cpp";

inline uint32_t align_up(uint32_t a, uint32_t align) { return (a + align - 1) & ~(align - 1); }

// Largest `page` (multiple of tile_size, <= max_page_size) such that num_tiles*tile_size
// is divisible by page. Returns (page_size, num_pages). Identical algorithm to
// ttnn::prim::get_max_page_size_and_num_pages.
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

// Per-tensor geometry handed to the DRAM-core prefetcher kernel. All values are derived
// from the tensor shape + dtype + GCB ring topology + DRISC L1 stage budget; the caller
// (compute_tensor_geom) picks (rows_per_sub, M) by the fit ladder documented in
// tt_metal/impl/buffers/prefetcher_matmul_design.md §6.
//
// Invariant: rows_per_sub > 1 implies M == 1 (the kernel cannot row-stride DMA).
struct TensorGeom {
    uint32_t bank_local_base = 0;      // GDDR offset where this tensor starts in the bank
    uint32_t num_sub = 0;              // sub-bands per ring-block
    uint32_t M = 0;                    // N-chunks per sub-band (divides num_receivers)
    uint32_t rows_per_sub = 0;         // K-rows per sub-band
    uint32_t coalesced_page_size = 0;  // bytes per K-row per receiver per coalesced page
    uint32_t coalesced_num_pages = 0;  // coalesced pages per K-row per receiver
    uint32_t sub_chunk_bytes = 0;      // bytes per DMA into one ring half
    uint32_t sub_stride_bytes = 0;     // DRAM byte stride between sub-bands within a block
    uint32_t block_stride_bytes = 0;   // DRAM byte stride between ring-blocks
    uint32_t page_bytes_per_recv = 0;  // bytes per receiver per full block (fifo_page_size)
};

TensorGeom compute_tensor_geom(
    const MeshTensor& t, uint32_t bank_local_base, uint32_t num_senders, uint32_t num_receivers, uint32_t ring_half) {
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
    const uint32_t num_blocks = num_senders * num_receivers;
    TT_FATAL(
        n_per_bank % num_receivers == 0,
        "n_per_bank ({}) must divide num_receivers ({}); reduce N_per_bank or grow num_receivers",
        n_per_bank,
        num_receivers);
    // ceil-up matches worker-core ttnn.dram_prefetcher's tt::round_up(height, num_blocks);
    // the tail block reads past the tensor bytes for tensors whose K_tiles doesn't divide
    // ring_size. See tt_metal/impl/buffers/prefetcher_matmul_design.md §5.
    const uint32_t k_block_w_tiles = (k_tiles_raw + num_blocks - 1) / num_blocks;
    const uint32_t n_per_recv = n_per_bank / num_receivers;
    const uint32_t row_bytes = n_per_bank * tile_bytes;
    const uint32_t block_bytes = k_block_w_tiles * row_bytes;
    const uint32_t noc_max_burst = MetalContext::instance().hal().get_noc_max_burst_size_bytes();
    const auto [coalesced_page_size, coalesced_num_pages] = pick_page_size(noc_max_burst, n_per_recv, tile_bytes);
    TT_FATAL(
        coalesced_page_size <= noc_max_burst,
        "DRAM-core prefetcher coalesced page size ({} B) exceeds the one-packet NoC write "
        "limit ({} B). Reduce N_per_bank or increase num_global_cb_receivers.",
        coalesced_page_size,
        noc_max_burst);

    // Fit ladder (tt_metal/impl/buffers/prefetcher_matmul_design.md §6).
    uint32_t rows_per_sub = 0;
    uint32_t M = 1;
    if (block_bytes <= ring_half) {
        // Case 1: full block fits — fast path, single-shot push.
        rows_per_sub = k_block_w_tiles;
        M = 1;
    } else if (row_bytes <= ring_half) {
        // Case 2: K-sub only. Largest divisor of k_block_w_tiles whose row-band fits.
        rows_per_sub = 1;
        for (uint32_t d = k_block_w_tiles; d >= 1; --d) {
            if (k_block_w_tiles % d == 0 && static_cast<uint64_t>(d) * row_bytes <= ring_half) {
                rows_per_sub = d;
                break;
            }
        }
        M = 1;
    } else {
        // Case 3: K-sub + N-chunk; rows_per_sub=1 forces M-chunking onto a single K-row
        // (contiguous N-stripe per chunk).
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
            "DRAM-core prefetcher cannot fit one K-row of tensor (k_tiles={}, n_per_bank={}, "
            "tile_bytes={}, row_bytes={} B) into DRISC L1 stage half ({} B) even with "
            "M=num_receivers ({}). Increase num_global_cb_receivers, reduce N_per_bank, or "
            "use the worker-core prefetcher.",
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
        sub_chunk_bytes <= ring_half,
        "Internal: chunk size {} B exceeds ring_half {} B after fit ladder. This is a bug "
        "in compute_tensor_geom; please file an issue with k_tiles={}, n_per_bank={}, "
        "tile_bytes={}, rows_per_sub={}, M={}.",
        sub_chunk_bytes,
        ring_half,
        k_tiles_raw,
        n_per_bank,
        tile_bytes,
        rows_per_sub,
        M);

    TensorGeom g;
    g.bank_local_base = bank_local_base;
    g.num_sub = num_sub;
    g.M = M;
    g.rows_per_sub = rows_per_sub;
    g.coalesced_page_size = coalesced_page_size;
    g.coalesced_num_pages = coalesced_num_pages;
    g.sub_chunk_bytes = sub_chunk_bytes;
    g.sub_stride_bytes = rows_per_sub * row_bytes;
    g.block_stride_bytes = k_block_w_tiles * row_bytes;
    g.page_bytes_per_recv = k_block_w_tiles * coalesced_num_pages * coalesced_page_size;
    return g;
}

// Build a single-device DRISC prefetcher Program from the GCB + tensor metadata.
// All devices in a LOCKSTEP mesh produce byte-identical Programs because buffer
// addresses, GCB addresses, and DRISC L1 layout are all device-uniform.
std::unique_ptr<Program> build_program(
    MeshDevice* mesh_device,
    const std::vector<const MeshTensor*>& input_tensors,
    const experimental::GlobalCircularBuffer& gcb,
    const experimental::DramCorePrefetcherConfig& config) {
    TT_FATAL(input_tensors.size() >= 2, "Need at least one data tensor + the tensor_addrs tensor");

    // Last tensor is the addrs tensor (kept for op-contract parity). DRISC path doesn't
    // read it — runtime args carry buffer base addresses directly.
    std::vector<const MeshTensor*> data_tensors;
    data_tensors.reserve(input_tensors.size() - 1);
    for (size_t i = 0; i + 1 < input_tensors.size(); ++i) {
        data_tensors.push_back(input_tensors[i]);
    }
    const uint32_t num_tensors = static_cast<uint32_t>(data_tensors.size());

    const auto& sender_receiver_core_mapping = gcb.sender_receiver_core_mapping();
    const uint32_t num_senders = static_cast<uint32_t>(sender_receiver_core_mapping.size());
    TT_FATAL(num_senders > 0, "DRAM-sender GlobalCircularBuffer must have at least one sender");

    uint32_t num_receivers = sender_receiver_core_mapping.front().second.num_cores();
    for (const auto& [_, recv] : sender_receiver_core_mapping) {
        TT_FATAL(
            recv.num_cores() == num_receivers,
            "All senders must have the same receiver count for the DRAM-core prefetcher");
    }

    // num_blocks (= ring_size) is the per-layer push count per receiver per tensor;
    // matches the receiver matmul's wait_front(num_blocks) at
    // ttnn/.../reader_bmm_tile_layout_in1_ring_all_gather.cpp:141. All tensors share it.
    const uint32_t num_blocks = num_senders * num_receivers;

    // DRISC L1 layout (see tt_metal/impl/buffers/drisc_l1_arena.hpp):
    //   kernel_working_region_base + [noc_xy table][config][stage_ring]
    // The stage_ring is split into two halves by the kernel (ping-pong, ring_depth=2).
    auto& arena = mesh_device->impl().drisc_l1_arena();
    const uint32_t kernel_region_size = arena.kernel_working_region_size();
    TT_FATAL(
        kernel_region_size > kKernelOverheadBytes,
        "DRISC L1 kernel region ({} B) too small for the prefetcher kernel overhead ({} B)",
        kernel_region_size,
        kKernelOverheadBytes);
    const uint32_t l1_alignment = hal::get_l1_alignment();
    const uint32_t pages_sent_addr = static_cast<uint32_t>(experimental::pages_sent_drisc_l1_base(gcb));
    uint32_t cursor = static_cast<uint32_t>(arena.kernel_working_region_base());
    cursor = align_up(cursor, l1_alignment);
    const uint32_t noc_xy_addr = cursor;
    cursor += 2 * sizeof(uint32_t) * num_receivers;
    cursor = align_up(cursor, l1_alignment);
    const uint32_t config_addr = cursor;
    cursor += 16;
    cursor = align_up(cursor, l1_alignment);
    const uint32_t stage_ring_base = cursor;
    const uint32_t region_end = static_cast<uint32_t>(arena.kernel_working_region_base()) + kernel_region_size;
    TT_FATAL(
        region_end > stage_ring_base,
        "DRISC L1 kernel region too small after overhead: region_end={} stage_ring_base={}",
        region_end,
        stage_ring_base);
    // Two halves; each half must be l1-aligned, so round the total down.
    uint32_t stage_ring_size = (region_end - stage_ring_base);
    stage_ring_size &= ~(2 * l1_alignment - 1);  // ensure both halves are l1-aligned
    TT_FATAL(
        stage_ring_size >= 2 * l1_alignment,
        "DRISC L1 stage ring too small: {} B (region_end={} stage_ring_base={})",
        stage_ring_size,
        region_end,
        stage_ring_base);
    const uint32_t ring_half = stage_ring_size / 2;

    // Compute per-tensor geometry; the fit ladder lives in compute_tensor_geom.
    std::vector<TensorGeom> geoms;
    geoms.reserve(data_tensors.size());
    for (const auto* t : data_tensors) {
        const uint32_t bank_local_base = static_cast<uint32_t>(t->mesh_buffer().address());
        geoms.push_back(compute_tensor_geom(*t, bank_local_base, num_senders, num_receivers, ring_half));
    }

    auto program = std::make_unique<Program>();

    for (uint32_t s = 0; s < num_senders; ++s) {
        const auto& [sender_logical, _receivers] = sender_receiver_core_mapping[s];
        std::vector<uint32_t> compile_args = {
            config.num_layers,
            num_tensors,
            num_blocks,
            num_receivers,
            stage_ring_base,
            stage_ring_size,
            kRemoteCBId,
            pages_sent_addr,
            noc_xy_addr,
            config_addr,
            gcb.size(),
            static_cast<uint32_t>(gcb.buffer_address()),
            static_cast<uint32_t>(experimental::pages_sent_worker_l1_base(gcb)),
        };

        KernelHandle kernel_id = CreateKernel(
            *program, kKernelPath, sender_logical, DramConfig{.noc = NOC::NOC_0, .compile_args = compile_args});

        // RT args: bank_id, then per-tensor blocks (length num_tensors each), then [recv_xy].
        // Order must match the kernel's read order at the top of kernel_main.
        std::vector<uint32_t> rt_args;
        rt_args.reserve(1 + 10 * num_tensors + 2 * num_receivers);
        rt_args.push_back(/*bank_id=*/sender_logical.x);
        for (const auto& g : geoms) {
            rt_args.push_back(g.bank_local_base);
        }
        for (const auto& g : geoms) {
            rt_args.push_back(g.num_sub);
        }
        for (const auto& g : geoms) {
            rt_args.push_back(g.M);
        }
        for (const auto& g : geoms) {
            rt_args.push_back(g.rows_per_sub);
        }
        for (const auto& g : geoms) {
            rt_args.push_back(g.coalesced_page_size);
        }
        for (const auto& g : geoms) {
            rt_args.push_back(g.coalesced_num_pages);
        }
        for (const auto& g : geoms) {
            rt_args.push_back(g.sub_chunk_bytes);
        }
        for (const auto& g : geoms) {
            rt_args.push_back(g.sub_stride_bytes);
        }
        for (const auto& g : geoms) {
            rt_args.push_back(g.block_stride_bytes);
        }
        for (const auto& g : geoms) {
            rt_args.push_back(g.page_bytes_per_recv);
        }
        const auto& receiver_phys = experimental::receiver_coords_per_sender(gcb).at(s);
        for (const auto& c : receiver_phys) {
            rt_args.push_back(c.x);
            rt_args.push_back(c.y);
        }
        SetRuntimeArgs(*program, kernel_id, sender_logical, rt_args);
    }

    return program;
}

}  // namespace

DramCorePrefetcherManager::DramCorePrefetcherManager(MeshDevice* mesh_device) : mesh_device_(mesh_device) {
    TT_FATAL(mesh_device_ != nullptr, "DramCorePrefetcherManager requires a non-null MeshDevice");
}

DramCorePrefetcherManager::~DramCorePrefetcherManager() { stop(); }

void DramCorePrefetcherManager::start(
    const std::vector<const MeshTensor*>& input_tensors,
    const experimental::GlobalCircularBuffer& gcb,
    const experimental::DramCorePrefetcherConfig& config) {
    TT_FATAL(
        !is_active(),
        "A DRAM-core prefetcher is already active on this mesh device. Call "
        "StopDramCorePrefetcher before starting another.");
    TT_FATAL(
        experimental::sender_core_type(gcb) == experimental::SenderCoreType::Dram,
        "StartDramCorePrefetcher requires a DRAM-sender GlobalCircularBuffer");

    // Hold the GCB by value so it outlives Stop even if the caller drops their copy.
    gcb_ = gcb;

    // Build one Program per IDevice in the mesh and launch it in slow dispatch with
    // wait_until_cores_done=false so the host returns immediately.
    for (auto* device : mesh_device_->get_view().get_devices()) {
        auto program = build_program(mesh_device_, input_tensors, gcb, config);
        ::tt::tt_metal::detail::CompileProgram(device, *program, /*force_slow_dispatch=*/true);
        ::tt::tt_metal::detail::WriteRuntimeArgsToDevice(device, *program, /*force_slow_dispatch=*/true);
        ::tt::tt_metal::detail::LaunchProgram(
            device, *program, /*wait_until_cores_done=*/false, /*force_slow_dispatch=*/true);
        per_device_state_.push_back(PerDeviceState{device, std::move(program)});
    }
}

void DramCorePrefetcherManager::stop() {
    if (!is_active()) {
        return;
    }
    for (auto& state : per_device_state_) {
        if (state.device != nullptr && state.program != nullptr) {
            ::tt::tt_metal::detail::WaitProgramDone(state.device, *state.program);
        }
    }
    per_device_state_.clear();
    gcb_.reset();
}

}  // namespace tt::tt_metal::distributed

// -----------------------------------------------------------------------------
// experimental::StartDramCorePrefetcher / StopDramCorePrefetcher
//
// Thin entry points that delegate to the per-mesh manager owned by MeshDeviceImpl.
// Defined here (next to the manager) to avoid a separate TU just for two free
// functions, and to keep MeshDeviceImpl::dram_core_prefetcher() the single point
// of contact.
// -----------------------------------------------------------------------------
namespace tt::tt_metal::experimental {

void StartDramCorePrefetcher(
    distributed::MeshDevice& mesh_device,
    const std::vector<const MeshTensor*>& input_tensors,
    const GlobalCircularBuffer& gcb,
    const DramCorePrefetcherConfig& config) {
    auto& manager = mesh_device.impl().dram_core_prefetcher(&mesh_device);
    manager.start(input_tensors, gcb, config);
}

void StopDramCorePrefetcher(distributed::MeshDevice& mesh_device) {
    auto& manager = mesh_device.impl().dram_core_prefetcher(&mesh_device);
    manager.stop();
}

}  // namespace tt::tt_metal::experimental
