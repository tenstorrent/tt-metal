// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Standalone optional path (HLD §11): two-core layout with dual fabric senders on the
// non-reduce core (NCRISC + BRISC, DM_DYNAMIC_NOC). No sync_cb on the dual path: local
// operand is NOC-copied to a receiver-only CB, so TRISC is not racing another RISC on
// the same local CB. NCRISC alone signals "local ready" on a global sem before fabric
// sends so the receiver does not NOC-read the sender's local CB before the sender DM
// has completed cb_wait_front.

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
namespace AllReduceDualCore {

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
    uint32_t signalLocalReady>
struct WriterLinkCTArgs {
    static constexpr uint32_t local_data_cb_id = localDataCbId;
    static constexpr uint32_t input_num_tiles = inputNumTiles;
    static constexpr uint32_t page_size_bytes = pageSizeBytes;
    static constexpr uint32_t tiles_per_chunk = tilesPerChunk;
    static constexpr uint32_t last_chunk_tiles = lastChunkTiles;
    static constexpr uint32_t num_chunks = numChunks;
    static constexpr uint32_t num_links = numLinks;
    static constexpr uint32_t link_index = linkIndex;
    static constexpr uint32_t signal_local_ready = signalLocalReady;
};

template <
    uint32_t localDataCbId,
    uint32_t remoteDataCbId,
    uint32_t residualCbId,
    uint32_t hasResidual,
    uint32_t skipLocalPush,
    uint32_t totalNumTiles,
    uint32_t pageSizeBytes,
    uint32_t tilesPerChunk,
    uint32_t lastChunkTiles,
    uint32_t numChunks,
    uint32_t numLinks>
struct ReaderTwoCoreCTArgs {
    static constexpr uint32_t local_data_cb_id = localDataCbId;
    static constexpr uint32_t remote_data_cb_id = remoteDataCbId;
    static constexpr uint32_t residual_cb_id = residualCbId;
    static constexpr bool has_residual = hasResidual != 0;
    static constexpr bool skip_local_push = skipLocalPush != 0;
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
    uint32_t cbTemp,
    uint32_t hasResidual,
    uint32_t numTiles>
struct ComputeDualCoreCTArgs {
    static constexpr uint32_t cb_remote = cbRemote;
    static constexpr uint32_t cb_local = cbLocal;
    static constexpr uint32_t cb_out = cbOut;
    static constexpr uint32_t cb_residual = cbResidual;
    static constexpr uint32_t cb_temp = cbTemp;
    static constexpr bool has_residual = hasResidual != 0;
    static constexpr uint32_t num_tiles = numTiles;
};

// Common RT: neighbor intermediate base + dest NOC (receiver on neighbor chip) for payload + fabric sem inc.
// Per-core RT: [dst_mesh_id, dst_chip_id, link_sem_bank]; if signal_local_ready (named CT on this kernel):
// [local_ready_noc_x, local_ready_noc_y, local_ready_sem_bank]; then fabric connection args (WriterSingleLink).
// per_core_rta_start_idx: first index into get_arg_val for that prefix + fabric (not common RT args).
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
    void impl([[maybe_unused]] const SenderFabricArgs& a) {
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
        if constexpr (CT::link_index >= CT::num_links) {
            return;
        }

        size_t arg_idx = size_t(a.per_core_rta_start_idx);
        const uint32_t dst_mesh_id = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t dst_chip_id = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t link_sem_bank = get_arg_val<uint32_t>(arg_idx++);

        if constexpr (CT::signal_local_ready != 0) {
            const uint32_t lr_dest_x = get_arg_val<uint32_t>(arg_idx++);
            const uint32_t lr_dest_y = get_arg_val<uint32_t>(arg_idx++);
            const uint32_t lr_sem_bank = get_arg_val<uint32_t>(arg_idx++);
            const uint64_t lr_noc = safe_get_noc_addr(lr_dest_x, lr_dest_y, lr_sem_bank, 0);
            noc_semaphore_inc(lr_noc, 1);
            noc_async_atomic_barrier();
        }

        auto connection =
            tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);
        connection.open_start();

        PacketHeaderPool::reset();
        volatile tt_l1_ptr PACKET_HEADER_TYPE* header = PacketHeaderPool::allocate_header();
        fabric_set_unicast_route(header, dst_chip_id, dst_mesh_id);

        const uint64_t dst_noc_base = safe_get_noc_addr(a.dest_noc_x, a.dest_noc_y, a.intermediate_buffer_address, 0);
        const uint64_t remote_sem_noc = safe_get_noc_addr(a.dest_noc_x, a.dest_noc_y, link_sem_bank, 0);

        if constexpr (CT::signal_local_ready != 0) {
            cb_reserve_back(CT::local_data_cb_id, CT::input_num_tiles);
            cb_push_back(CT::local_data_cb_id, CT::input_num_tiles);
        }
        const uint32_t src_base_addr = get_read_ptr(CT::local_data_cb_id);

        connection.open_finish();

        // Byte step between this link's consecutive stripes: chunks [chunk_idx, chunk_idx+num_links) are
        // all strictly before the global tail iff chunk_idx+num_links < num_chunks, so each uses
        // tiles_per_chunk → constant stride (compile-time).
        constexpr uint32_t stride_bytes = CT::num_links * CT::tiles_per_chunk * CT::page_size_bytes;
        uint32_t offset = CT::link_index * CT::tiles_per_chunk * CT::page_size_bytes;

        constexpr uint32_t regular_payload_bytes = CT::tiles_per_chunk * CT::page_size_bytes;
        constexpr uint32_t last_payload_bytes = CT::last_chunk_tiles * CT::page_size_bytes;

        auto send_payload = [&](uint32_t offset, uint32_t payload_bytes) __attribute__((always_inline)) {
            header->to_noc_fused_unicast_write_atomic_inc(
                {dst_noc_base + offset, remote_sem_noc, 1, false}, payload_bytes);
            connection.wait_for_empty_write_slot();
            connection.send_payload_without_header_non_blocking_from_address(src_base_addr + offset, payload_bytes);
            connection.send_payload_flush_non_blocking_from_address(
                reinterpret_cast<uint32_t>(header), sizeof(PACKET_HEADER_TYPE));
        };

        // first loop uintil last chunk - 1
        uint32_t chunk_idx = CT::link_index;
        for (; chunk_idx < CT::num_chunks - 1; chunk_idx += CT::num_links) {
            send_payload(offset, regular_payload_bytes);

            // Stride add on the final iteration is dead (offset unused); avoids duplicating the for-loop exit test.
            offset += stride_bytes;

            noc_async_writes_flushed();
        }

        // handle last chunk
        if (chunk_idx < CT::num_chunks) {
            send_payload(offset, last_payload_bytes);
            // no need to flush, we have a barrier at the end
        }

        connection.close_start();

        // pop local cb? need to confirm if we can do it or not
        if constexpr (CT::signal_local_ready != 0) {
            cb_pop_front(CT::local_data_cb_id, CT::input_num_tiles);
        }

        connection.close_finish();

        noc_async_full_barrier();
#endif
    }
};

template <typename CT>
class ReaderTwoCore {
public:
    void operator()(const ReceiverArgs& a) { impl(a); }

private:
    void impl([[maybe_unused]] const ReceiverArgs& a) {
#if defined(COMPILE_FOR_BRISC)
        // push residual to unblock TRISC
        if constexpr (CT::has_residual) {
            cb_reserve_back(CT::residual_cb_id, CT::total_num_tiles);
            cb_push_back(CT::residual_cb_id, CT::total_num_tiles);
        }

        cb_reserve_back(CT::local_data_cb_id, CT::total_num_tiles);
        cb_reserve_back(CT::remote_data_cb_id, CT::total_num_tiles);
        constexpr uint32_t local_payload_bytes = CT::total_num_tiles * CT::page_size_bytes;
        const uint64_t local_data_src_noc =
            safe_get_noc_addr(a.sender_noc_x, a.sender_noc_y, a.sender_local_data_l1_addr, 0);
        const uint32_t local_data_dst = get_write_ptr(CT::local_data_cb_id);

        volatile tt_l1_ptr uint32_t* local_ready_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a.local_ready_sem_bank_addr);
        noc_semaphore_wait_min(local_ready_ptr, 1);
        unified_kernels::semaphore_dec(local_ready_ptr, 1);

        // Issue the read for local data as early as possible
        noc_async_read(local_data_src_noc, local_data_dst, local_payload_bytes);

        const std::array<uint32_t, MAX_NUM_LINKS> sem_addrs = {a.sem_bank_addr_0, a.sem_bank_addr_1};
        std::array<volatile tt_l1_ptr uint32_t*, CT::num_links> sem_ptrs;
        std::array<uint32_t, CT::num_links> link_counters = {};

        for (uint32_t link_idx = 0; link_idx < CT::num_links; link_idx++) {
            sem_ptrs[link_idx] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addrs[link_idx]);
        }

        // block for read to finish
        noc_async_read_barrier();
        cb_push_back(CT::local_data_cb_id, CT::total_num_tiles);

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
class ComputeDualCore {
public:
    void operator()(const ComputeArgs&) { impl(); }

private:
#if defined(COMPILE_FOR_TRISC)
    template <bool acquire_regs, bool pop_cbs>
    static FORCE_INLINE void batched_add(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out, uint32_t num_tiles) {
        cb_reserve_back(cb_out, num_tiles);

        if constexpr (acquire_regs) {
            tile_regs_acquire();
        }
        for (uint32_t i = 0; i < num_tiles; i++) {
            cb_wait_front(cb_a, i + 1);
            cb_wait_front(cb_b, i + 1);
            add_tiles(cb_a, cb_b, i, i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < num_tiles; i++) {
            pack_tile(i, cb_out, i);
        }
        tile_regs_release();

        if constexpr (pop_cbs) {
            cb_pop_front(cb_a, num_tiles);
            cb_pop_front(cb_b, num_tiles);
        }
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
            batched_add<false, false>(CT::cb_local, CT::cb_remote, CT::cb_out, CT::num_tiles);
        } else {
            add_tiles_init(CT::cb_local, CT::cb_remote);
            batched_add<true, false>(CT::cb_local, CT::cb_remote, CT::cb_out, CT::num_tiles);
        }

        cb_pop_front(CT::cb_local, CT::num_tiles);
        cb_pop_front(CT::cb_remote, CT::num_tiles);
#endif
    }
};

}  // namespace AllReduceDualCore
}  // namespace deepseek_b1_ops
