// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Two-core all-reduce: sender core runs dual fabric writers (NCRISC + BRISC,
// DM_DYNAMIC_NOC, one link per RISC), receiver core runs reader (BRISC) and
// compute (TRISC).  No sync_cb needed: local operand is NOC-copied to a
// receiver-only CB, so TRISC is not racing another RISC on the same local CB.
// The sender RISC with signal_local_ready=1 signals the receiver after
// confirming local data is present (via cb_push_back or cb_wait_front
// depending on skip_local_push).
//
// CB layout (each ID is unique per logical buffer):
//   sender core:   local_data_cb_id      — input tensor shard
//   receiver core: recv_local_data_cb_id — NOC-read copy of local data
//                  remote_data_cb_id     — fabric-received remote data
//                  output_cb_id          — reduction output
//                  residual_cb_id        — optional residual

#include "kernel_op_api.hpp"
#include "kernel_utils.hpp"

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#include <array>
#include <cstdint>
#endif

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#endif

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#endif

namespace deepseek_b1_ops {
namespace AllReduce {

static constexpr uint32_t MAX_NUM_LINKS = 2;

template <
    uint32_t localDataCbId,
    uint32_t inputNumTiles,
    uint32_t pageSizeBytes,
    uint32_t tilesPerChunk,
    uint32_t lastChunkTiles,
    uint32_t numChunks,
    uint32_t numLinks,
    uint32_t linkIndex,
    uint32_t signalLocalReady,
    uint32_t skipLocalPush>
struct WriterLinkCTArgs {
    static constexpr uint32_t local_data_cb_id = localDataCbId;
    static constexpr uint32_t input_num_tiles = inputNumTiles;
    static constexpr uint32_t page_size_bytes = pageSizeBytes;
    static constexpr uint32_t tiles_per_chunk = tilesPerChunk;
    static constexpr uint32_t last_chunk_tiles = lastChunkTiles;
    static constexpr uint32_t num_chunks = numChunks;
    static constexpr uint32_t num_links = numLinks;
    static constexpr uint32_t link_index = linkIndex;
    static constexpr bool signal_local_ready = signalLocalReady != 0;
    static constexpr bool skip_local_push = skipLocalPush != 0;
};

template <
    uint32_t recvLocalDataCbId,
    uint32_t remoteDataCbId,
    uint32_t residualCbId,
    uint32_t hasResidual,
    uint32_t totalNumTiles,
    uint32_t pageSizeBytes,
    uint32_t tilesPerChunk,
    uint32_t lastChunkTiles,
    uint32_t numChunks,
    uint32_t numLinks>
struct ReaderCTArgs {
    static constexpr uint32_t recv_local_data_cb_id = recvLocalDataCbId;
    static constexpr uint32_t remote_data_cb_id = remoteDataCbId;
    static constexpr uint32_t residual_cb_id = residualCbId;
    static constexpr bool has_residual = hasResidual != 0;
    static constexpr uint32_t total_num_tiles = totalNumTiles;
    static constexpr uint32_t page_size_bytes = pageSizeBytes;
    static constexpr uint32_t tiles_per_chunk = tilesPerChunk;
    static constexpr uint32_t last_chunk_tiles = lastChunkTiles;
    static constexpr uint32_t num_chunks = numChunks;
    static constexpr uint32_t num_links = numLinks;
};

template <
    uint32_t cbRemote,
    uint32_t cbLocal,
    uint32_t cbOut,
    uint32_t cbResidual,
    uint32_t hasResidual,
    uint32_t numTiles>
struct ComputeCTArgs {
    static constexpr uint32_t cb_remote = cbRemote;
    static constexpr uint32_t cb_local = cbLocal;
    static constexpr uint32_t cb_out = cbOut;
    static constexpr uint32_t cb_residual = cbResidual;
    static constexpr bool has_residual = hasResidual != 0;
    static constexpr uint32_t num_tiles = numTiles;
};

// Sender core common RT args: neighbor intermediate buffer base + dest NOC
// coordinates (receiver on neighbor chip) for payload writes and fabric sem
// increments.
// Per-core RT layout: [dst_mesh_id, dst_chip_id, link_sem_bank_addr]; if
// signal_local_ready: [local_ready_noc_x, local_ready_noc_y,
// local_ready_sem_bank_addr]; then fabric connection args from
// setup_fabric_connection.
struct SenderFabricArgs {
    uint32_t intermediate_buffer_address;
    uint32_t dest_noc_x;
    uint32_t dest_noc_y;
    uint32_t per_core_rta_start_idx = 0;
};

struct ReceiverArgs {
    uint32_t sem_bank_addr_0;
    uint32_t sem_bank_addr_1;
    uint32_t sender_noc_x;
    uint32_t sender_noc_y;
    uint32_t sender_local_data_l1_addr;
    uint32_t local_ready_sem_bank_addr;
};

struct ComputeArgs {};

using RTArgs = unified_kernels::SelectByRISCV<SenderFabricArgs, SenderFabricArgs, ComputeArgs>;

template <typename CT>
class WriterSingleLink {
public:
    void operator()(const SenderFabricArgs& args) { impl(args); }

private:
    void impl([[maybe_unused]] const SenderFabricArgs& args) {
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
        if constexpr (CT::link_index >= CT::num_links) {
            return;
        }

        size_t arg_idx = size_t(args.per_core_rta_start_idx);
        const uint32_t dst_mesh_id = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t dst_chip_id = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t link_sem_bank_addr = get_arg_val<uint32_t>(arg_idx++);

        uint64_t local_ready_noc_addr = 0;
        if constexpr (CT::signal_local_ready) {
            const uint32_t local_ready_dest_noc_x = get_arg_val<uint32_t>(arg_idx++);
            const uint32_t local_ready_dest_noc_y = get_arg_val<uint32_t>(arg_idx++);
            const uint32_t local_ready_sem_bank = get_arg_val<uint32_t>(arg_idx++);
            local_ready_noc_addr =
                safe_get_noc_addr(local_ready_dest_noc_x, local_ready_dest_noc_y, local_ready_sem_bank, 0);
        }

        auto connection =
            tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);
        connection.open_start();

        PacketHeaderPool::reset();
        volatile tt_l1_ptr PACKET_HEADER_TYPE* header = PacketHeaderPool::allocate_header();
        fabric_set_unicast_route(header, dst_chip_id, dst_mesh_id);

        const uint64_t dst_noc_base =
            safe_get_noc_addr(args.dest_noc_x, args.dest_noc_y, args.intermediate_buffer_address, 0);
        const uint64_t remote_sem_noc = safe_get_noc_addr(args.dest_noc_x, args.dest_noc_y, link_sem_bank_addr, 0);

        // Ensure local data is available in the CB before signaling or sending.
        if constexpr (CT::skip_local_push) {
            // Fused path: preceding op (e.g. gather3) pushes to this CB.
            cb_wait_front(CT::local_data_cb_id, CT::input_num_tiles);
        } else if constexpr (CT::signal_local_ready) {
            // Standalone path: push the sharded tensor into the CB.
            // Only the RISC with signal_local_ready does the push; the other
            // RISC reads get_read_ptr directly (CB base points to the tensor).
            cb_reserve_back(CT::local_data_cb_id, CT::input_num_tiles);
            cb_push_back(CT::local_data_cb_id, CT::input_num_tiles);
        }

        // Signal local_ready AFTER data is confirmed present so the receiver
        // can safely NOC-read from the sender's L1.
        if constexpr (CT::signal_local_ready) {
            noc_semaphore_inc(local_ready_noc_addr, 1);
            noc_async_atomic_barrier();
        }

        const uint32_t src_base_addr = get_read_ptr(CT::local_data_cb_id);

        connection.open_finish();

        constexpr uint32_t stride_bytes = CT::num_links * CT::tiles_per_chunk * CT::page_size_bytes;
        uint32_t offset = CT::link_index * CT::tiles_per_chunk * CT::page_size_bytes;

        constexpr uint32_t regular_payload_bytes = CT::tiles_per_chunk * CT::page_size_bytes;
        constexpr uint32_t last_payload_bytes = CT::last_chunk_tiles * CT::page_size_bytes;

        auto send_payload = [&](uint32_t chunk_offset, uint32_t payload_bytes) __attribute__((always_inline)) {
            header->to_noc_fused_unicast_write_atomic_inc(
                {dst_noc_base + chunk_offset, remote_sem_noc, 1, false}, payload_bytes);
            connection.wait_for_empty_write_slot();
            connection.send_payload_without_header_non_blocking_from_address(
                src_base_addr + chunk_offset, payload_bytes);
            connection.send_payload_flush_non_blocking_from_address(
                reinterpret_cast<uint32_t>(header), sizeof(PACKET_HEADER_TYPE));
        };

        uint32_t chunk_idx = CT::link_index;
        for (; chunk_idx < CT::num_chunks - 1; chunk_idx += CT::num_links) {
            send_payload(offset, regular_payload_bytes);
            offset += stride_bytes;
            noc_async_writes_flushed();
        }

        if (chunk_idx < CT::num_chunks) {
            send_payload(offset, last_payload_bytes);
        }

        connection.close_start();

        if constexpr (CT::signal_local_ready) {
            cb_pop_front(CT::local_data_cb_id, CT::input_num_tiles);
        }

        connection.close_finish();

        noc_async_full_barrier();
#endif
    }
};

template <typename CT>
class Reader {
public:
    void operator()(const ReceiverArgs& args) { impl(args); }

private:
    void impl([[maybe_unused]] const ReceiverArgs& args) {
#if defined(COMPILE_FOR_BRISC)
        if constexpr (CT::has_residual) {
            cb_reserve_back(CT::residual_cb_id, CT::total_num_tiles);
            cb_push_back(CT::residual_cb_id, CT::total_num_tiles);
        }

        cb_reserve_back(CT::recv_local_data_cb_id, CT::total_num_tiles);
        cb_reserve_back(CT::remote_data_cb_id, CT::total_num_tiles);
        constexpr uint32_t local_payload_bytes = CT::total_num_tiles * CT::page_size_bytes;
        const uint64_t local_data_src_noc =
            safe_get_noc_addr(args.sender_noc_x, args.sender_noc_y, args.sender_local_data_l1_addr, 0);
        const uint32_t local_data_dst = get_write_ptr(CT::recv_local_data_cb_id);

        volatile tt_l1_ptr uint32_t* local_ready_sem_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.local_ready_sem_bank_addr);
        noc_semaphore_wait_min(local_ready_sem_ptr, 1);
        unified_kernels::semaphore_dec(local_ready_sem_ptr, 1);

        noc_async_read(local_data_src_noc, local_data_dst, local_payload_bytes);

        const std::array<uint32_t, MAX_NUM_LINKS> sem_addrs = {args.sem_bank_addr_0, args.sem_bank_addr_1};
        std::array<volatile tt_l1_ptr uint32_t*, CT::num_links> sem_ptrs;
        std::array<uint32_t, CT::num_links> link_counters = {};

        for (uint32_t link_idx = 0; link_idx < CT::num_links; link_idx++) {
            sem_ptrs[link_idx] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addrs[link_idx]);
        }

        noc_async_read_barrier();
        cb_push_back(CT::recv_local_data_cb_id, CT::total_num_tiles);

        uint32_t current_link = 0;
        for (uint32_t chunk_idx = 0; chunk_idx < CT::num_chunks; chunk_idx++) {
            const uint32_t tiles = (chunk_idx < CT::num_chunks - 1) ? CT::tiles_per_chunk : CT::last_chunk_tiles;

            link_counters[current_link]++;
            noc_semaphore_wait_min(sem_ptrs[current_link], link_counters[current_link]);
            cb_push_back(CT::remote_data_cb_id, tiles);

            if (++current_link == CT::num_links) {
                current_link = 0;
            }
        }

        for (uint32_t link_idx = 0; link_idx < CT::num_links; link_idx++) {
            if (link_counters[link_idx] > 0) {
                unified_kernels::semaphore_dec(sem_ptrs[link_idx], link_counters[link_idx]);
            }
        }
#endif
    }
};

template <typename CT>
class Compute {
public:
    void operator()(const ComputeArgs&) { impl(); }

private:
#if defined(COMPILE_FOR_TRISC)
    static constexpr uint32_t MAX_DST_TILES = 4;

    template <bool acquire_regs>
    static FORCE_INLINE void batched_add(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out, uint32_t num_tiles) {
        uint32_t num_batches = (num_tiles + MAX_DST_TILES - 1) / MAX_DST_TILES;

        MATH((t6_semaphore_wait_on_max<p_stall::STALL_MATH>(semaphore::FPU_SFPU)));

        cb_reserve_back(cb_out, num_tiles);

        if constexpr (acquire_regs) {
            tile_regs_acquire();
        }

        for (uint32_t batch = 0; batch < num_batches; ++batch) {
            uint32_t start_tile = batch * MAX_DST_TILES;
            uint32_t batch_size = (start_tile + MAX_DST_TILES <= num_tiles) ? MAX_DST_TILES : (num_tiles - start_tile);
            uint32_t tiles_needed = start_tile + batch_size;

            // Streaming wait: block only until this batch's tiles are present.
            // Local CB (cb_a) returns immediately (all pushed at once by reader);
            // remote CB (cb_b) blocks until enough fabric chunks have arrived.
            cb_wait_front(cb_a, tiles_needed);
            cb_wait_front(cb_b, tiles_needed);

            if (batch == num_batches - 1) {
                tile_regs_wait();
            } else {
                PACK(t6_semaphore_wait_on_zero<p_stall::STALL_PACK>(semaphore::FPU_SFPU));
            }

            for (uint32_t i = 0; i < batch_size; ++i) {
                add_tiles(cb_a, cb_b, start_tile + i, start_tile + i, start_tile + i);
                pack_tile(start_tile + i, cb_out);
            }

            if (batch == num_batches - 1) {
                tile_regs_commit();
            } else {
                PACK(t6_semaphore_get<p_stall::PACK>(semaphore::FPU_SFPU));
                MATH((t6_semaphore_post<p_stall::MATH>(semaphore::FPU_SFPU)));
            }
        }

        tile_regs_release();

        cb_pop_front(cb_a, num_tiles);
        cb_pop_front(cb_b, num_tiles);
        cb_push_back(cb_out, num_tiles);
    }
#endif

    void impl() {
#if defined(COMPILE_FOR_TRISC)
        reconfig_data_format<false, true>(CT::cb_local, CT::cb_remote);
        pack_reconfig_data_format<true>(CT::cb_out);

        if constexpr (CT::has_residual) {
            copy_tile_to_dst_init_short(CT::cb_residual);
            cb_wait_front(CT::cb_residual, CT::num_tiles);
            tile_regs_acquire();
            for (uint32_t i = 0; i < CT::num_tiles; i++) {
                copy_tile(CT::cb_residual, i, i);
            }
            cb_pop_front(CT::cb_residual, CT::num_tiles);
            add_tiles_init(CT::cb_local, CT::cb_remote, true);
            batched_add<false>(CT::cb_local, CT::cb_remote, CT::cb_out, CT::num_tiles);
        } else {
            add_tiles_init(CT::cb_local, CT::cb_remote);
            batched_add<true>(CT::cb_local, CT::cb_remote, CT::cb_out, CT::num_tiles);
        }
#endif
    }
};

}  // namespace AllReduce
}  // namespace deepseek_b1_ops
