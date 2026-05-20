// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "distributed/dram_core_prefetcher_manager.hpp"

#include "distributed/mesh_device_impl.hpp"

#include <algorithm>
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

#include "impl/kernels/kernel.hpp"  // DramConfig + CreateKernel(DramConfig)

namespace tt::tt_metal::distributed {

namespace {

constexpr uint32_t kRemoteCBId = 31;

// Conservative DRISC L1 budget for the two ping-pong stage buffers. BH DRISC L1 is 128 KB
// total; ~35 KB is reserved for firmware/mailbox/kernel-config, leaving ~93 KB. Reserve
// some for pages_sent/noc_xy/config (a few KB) -> ~80 KB for stages.
constexpr uint32_t kDriscL1Budget = 80 * 1024;

constexpr const char* kKernelPath = "tt_metal/distributed/kernels/dram_core_prefetcher.cpp";

inline uint32_t align_up(uint32_t a, uint32_t align) { return (a + align - 1) & ~(align - 1); }

struct BlockGeom {
    uint32_t num_blocks = 0;
    uint32_t dma_block_size = 0;  // bytes/K-block/bank
    uint32_t push_page_size = 0;  // bytes/K-block/receiver
};

BlockGeom compute_block_geom(const MeshTensor& t, uint32_t num_receivers, uint32_t k_block_w_tiles) {
    const auto* ref_buffer = t.mesh_buffer().get_reference_buffer();
    const auto shard_shape = ref_buffer->shard_spec().shape();
    const uint32_t bytes_per_tile = tt::tile_size(datatype_to_dataformat_converter(t.dtype()));
    const uint32_t k_tiles = shard_shape[0] / tt::constants::TILE_HEIGHT;
    const uint32_t n_tiles_per_bank = shard_shape[1] / tt::constants::TILE_WIDTH;
    TT_FATAL(
        n_tiles_per_bank % num_receivers == 0,
        "n_tiles_per_bank ({}) must divide num_receivers ({})",
        n_tiles_per_bank,
        num_receivers);
    TT_FATAL(
        k_tiles % k_block_w_tiles == 0,
        "k_tiles ({}) must be divisible by dram_core_k_block_w_tiles ({})",
        k_tiles,
        k_block_w_tiles);
    const uint32_t n_tiles_per_receiver = n_tiles_per_bank / num_receivers;
    BlockGeom g;
    g.num_blocks = k_tiles / k_block_w_tiles;
    g.dma_block_size = k_block_w_tiles * n_tiles_per_bank * bytes_per_tile;
    g.push_page_size = k_block_w_tiles * n_tiles_per_receiver * bytes_per_tile;
    return g;
}

// Smallest M dividing num_receivers such that 2 * (dma_block_size / M) <= kDriscL1Budget
// for every tensor. M must divide num_receivers; M=1 is the original single-push behavior.
uint32_t pick_M(const std::vector<BlockGeom>& geoms, uint32_t num_receivers) {
    for (uint32_t m = 1; m <= num_receivers; ++m) {
        if (num_receivers % m != 0) {
            continue;
        }
        bool ok = true;
        for (const auto& g : geoms) {
            const uint32_t chunk_size = g.dma_block_size / m;
            if (2 * chunk_size > kDriscL1Budget) {
                ok = false;
                break;
            }
        }
        if (ok) {
            return m;
        }
    }
    return 0u;
}

// Build a single-device DRISC prefetcher Program from the GCB + tensor metadata.
// All devices in a LOCKSTEP mesh produce byte-identical Programs because buffer
// addresses, GCB addresses, and DRISC L1 layout are all device-uniform.
std::unique_ptr<Program> build_program(
    const std::vector<const MeshTensor*>& input_tensors,
    const GlobalCircularBuffer& gcb,
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

    const uint32_t k_block_w_tiles = config.dram_core_k_block_w_tiles;
    TT_FATAL(k_block_w_tiles > 0, "dram_core_k_block_w_tiles must be > 0");

    // Per-tensor block geometry; the manager assumes all tensors share num_blocks.
    std::vector<BlockGeom> geoms;
    geoms.reserve(data_tensors.size());
    uint32_t first_num_blocks = 0;
    for (size_t ti = 0; ti < data_tensors.size(); ++ti) {
        auto g = compute_block_geom(*data_tensors[ti], num_receivers, k_block_w_tiles);
        if (ti == 0) {
            first_num_blocks = g.num_blocks;
        }
        TT_FATAL(g.num_blocks == first_num_blocks, "All tensors must share the same num_blocks");
        geoms.push_back(g);
    }
    const uint32_t M = pick_M(geoms, num_receivers);
    TT_FATAL(
        M > 0,
        "No valid num_dma_chunks_per_block found: even with M=num_receivers ({}), the per-chunk "
        "stage buffer ({} B) exceeds half the DRISC L1 budget ({} B). Increase the ring size or "
        "reduce N_per_bank.",
        num_receivers,
        geoms.empty() ? 0 : geoms.front().dma_block_size / num_receivers,
        kDriscL1Budget);

    uint32_t max_chunk_size = 0;
    for (const auto& g : geoms) {
        max_chunk_size = std::max(max_chunk_size, g.dma_block_size / M);
    }
    const uint32_t l1_alignment = hal::get_l1_alignment();
    max_chunk_size = align_up(max_chunk_size, l1_alignment);

    // DRISC L1 layout above pages_sent (the GCB owns pages_sent).
    const uint32_t pages_sent_addr = static_cast<uint32_t>(experimental::pages_sent_drisc_l1_base(gcb));
    uint32_t cursor = pages_sent_addr + 2 * l1_alignment * num_receivers;
    cursor = align_up(cursor, l1_alignment);
    const uint32_t noc_xy_addr = cursor;
    cursor += 2 * sizeof(uint32_t) * num_receivers;
    cursor = align_up(cursor, l1_alignment);
    const uint32_t config_addr = cursor;
    cursor += 16;
    cursor = align_up(cursor, l1_alignment);
    const uint32_t stage_a_addr = cursor;
    cursor += max_chunk_size;
    cursor = align_up(cursor, l1_alignment);
    const uint32_t stage_b_addr = cursor;

    auto program = std::make_unique<Program>();

    for (uint32_t s = 0; s < num_senders; ++s) {
        const auto& [sender_logical, _receivers] = sender_receiver_core_mapping[s];
        std::vector<uint32_t> compile_args = {
            config.num_layers,
            num_tensors,
            first_num_blocks,
            num_receivers,
            max_chunk_size,
            stage_a_addr,
            stage_b_addr,
            kRemoteCBId,
            pages_sent_addr,
            noc_xy_addr,
            config_addr,
            gcb.size(),
            static_cast<uint32_t>(gcb.buffer_address()),
            static_cast<uint32_t>(experimental::pages_sent_worker_l1_base(gcb)),
            M,
        };

        KernelHandle kernel_id = CreateKernel(
            *program, kKernelPath, sender_logical, DramConfig{.noc = NOC::NOC_0, .compile_args = compile_args});

        // RT args: bank_id, [tensor_addrs], [dma_block_sizes], [push_page_sizes], [recv_xy]
        std::vector<uint32_t> rt_args;
        rt_args.reserve(1 + 3 * num_tensors + 2 * num_receivers);
        rt_args.push_back(/*bank_id=*/sender_logical.x);
        for (const auto* t : data_tensors) {
            rt_args.push_back(static_cast<uint32_t>(t->mesh_buffer().address()));
        }
        for (const auto& g : geoms) {
            rt_args.push_back(g.dma_block_size);
        }
        for (const auto& g : geoms) {
            rt_args.push_back(g.push_page_size);
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
    const GlobalCircularBuffer& gcb,
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
        auto program = build_program(input_tensors, gcb, config);
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
            WaitProgramDone(state.device, *state.program);
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
    distributed::MeshDevice* mesh_device,
    const std::vector<const MeshTensor*>& input_tensors,
    const GlobalCircularBuffer& gcb,
    const DramCorePrefetcherConfig& config) {
    TT_FATAL(mesh_device != nullptr, "StartDramCorePrefetcher requires a non-null MeshDevice");
    auto& manager = mesh_device->impl().dram_core_prefetcher(mesh_device);
    manager.start(input_tensors, gcb, config);
}

void StopDramCorePrefetcher(distributed::MeshDevice* mesh_device) {
    TT_FATAL(mesh_device != nullptr, "StopDramCorePrefetcher requires a non-null MeshDevice");
    auto& manager = mesh_device->impl().dram_core_prefetcher(mesh_device);
    manager.stop();
}

}  // namespace tt::tt_metal::experimental
