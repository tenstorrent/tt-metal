// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"
#include "kernel_utils.hpp"

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#include <array>
#include <cstdint>
#endif

#if defined(COMPILE_FOR_NCRISC)
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
    uint32_t syncCbId,
    uint32_t inputNumTiles,
    uint32_t pageSizeBytes,
    uint32_t tilesPerChunk,
    uint32_t lastChunkTiles,
    uint32_t numChunks,
    uint32_t numLinks>
struct WriterCTArgs {
    static constexpr uint32_t local_data_cb_id = localDataCbId;
    static constexpr uint32_t sync_cb_id = syncCbId;
    static constexpr uint32_t input_num_tiles = inputNumTiles;
    static constexpr uint32_t page_size_bytes = pageSizeBytes;
    static constexpr uint32_t tiles_per_chunk = tilesPerChunk;
    static constexpr uint32_t last_chunk_tiles = lastChunkTiles;
    static constexpr uint32_t num_chunks = numChunks;
    static constexpr uint32_t num_links = numLinks;
};

template <
    uint32_t localDataCbId,
    uint32_t remoteDataCbId,
    uint32_t residualCbId,
    uint32_t hasResidual,
    uint32_t skipLocalPush,
    uint32_t totalNumTiles,
    uint32_t tilesPerChunk,
    uint32_t lastChunkTiles,
    uint32_t numChunks,
    uint32_t numLinks>
struct ReaderCTArgs {
    static constexpr uint32_t local_data_cb_id = localDataCbId;
    static constexpr uint32_t remote_data_cb_id = remoteDataCbId;
    static constexpr uint32_t residual_cb_id = residualCbId;
    static constexpr bool has_residual = hasResidual != 0;
    static constexpr bool skip_local_push = skipLocalPush != 0;
    static constexpr uint32_t total_num_tiles = totalNumTiles;
    static constexpr uint32_t tiles_per_chunk = tilesPerChunk;
    static constexpr uint32_t last_chunk_tiles = lastChunkTiles;
    static constexpr uint32_t num_chunks = numChunks;
    static constexpr uint32_t num_links = numLinks;
};

template <
    uint32_t cbRemote,
    uint32_t cbLocal,
    uint32_t cbOut,
    uint32_t syncCbId,
    uint32_t cbResidual,
    uint32_t cbTemp,
    uint32_t hasResidual,
    uint32_t numTiles,
    uint32_t numChunks,
    uint32_t tilesPerChunk,
    uint32_t lastChunkTiles>
struct ComputeCTArgs {
    static constexpr uint32_t cb_remote = cbRemote;
    static constexpr uint32_t cb_local = cbLocal;
    static constexpr uint32_t cb_out = cbOut;
    static constexpr uint32_t sync_cb_id = syncCbId;
    static constexpr uint32_t cb_residual = cbResidual;
    static constexpr uint32_t cb_temp = cbTemp;
    static constexpr bool has_residual = hasResidual != 0;
    static constexpr uint32_t num_tiles = numTiles;
    static constexpr uint32_t num_chunks = numChunks;
    static constexpr uint32_t tiles_per_chunk = tilesPerChunk;
    static constexpr uint32_t last_chunk_tiles = lastChunkTiles;
};

struct WriterArgs {
    uint32_t intermediate_buffer_address;
    uint32_t my_noc_x;
    uint32_t my_noc_y;
    uint32_t sem_bank_addr_0;
    uint32_t sem_bank_addr_1;
    uint32_t fabric_args_start_index = 0;
};

struct ReaderArgs {
    uint32_t sem_bank_addr_0;
    uint32_t sem_bank_addr_1;
};

struct ComputeArgs {};

using RTArgs = unified_kernels::SelectByRISCV<WriterArgs, ReaderArgs, ComputeArgs>;

template <typename CT>
class Writer {
public:
    void operator()(const RTArgs& args) { impl(args); }

private:
    void impl([[maybe_unused]] const RTArgs& args) {
#if defined(COMPILE_FOR_NCRISC)
        PacketHeaderPool::reset();

        std::array<tt::tt_fabric::WorkerToFabricEdmSender, CT::num_links> connections;
        std::array<volatile tt_l1_ptr PACKET_HEADER_TYPE*, CT::num_links> headers;
        std::array<uint64_t, CT::num_links> remote_sem_nocs;

        size_t arg_idx = size_t(args.fabric_args_start_index);
        const uint32_t dst_mesh_id = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t dst_chip_id = get_arg_val<uint32_t>(arg_idx++);

        const std::array<uint32_t, MAX_NUM_LINKS> sem_addrs = {args.sem_bank_addr_0, args.sem_bank_addr_1};
        for (uint32_t link_idx = 0; link_idx < CT::num_links; link_idx++) {
            connections[link_idx] =
                tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);
            connections[link_idx].open_start();
            headers[link_idx] = PacketHeaderPool::allocate_header();
            fabric_set_unicast_route(headers[link_idx], dst_chip_id, dst_mesh_id);
            remote_sem_nocs[link_idx] = safe_get_noc_addr(args.my_noc_x, args.my_noc_y, sem_addrs[link_idx], 0);
            connections[link_idx].open_finish();
        }

        const uint64_t dst_noc_base =
            safe_get_noc_addr(args.my_noc_x, args.my_noc_y, args.intermediate_buffer_address, 0);

        cb_wait_front(CT::local_data_cb_id, CT::input_num_tiles);
        const uint32_t src_addr = get_read_ptr(CT::local_data_cb_id);

        uint32_t current_link = 0;
        uint32_t offset = 0;
        for (uint32_t chunk_idx = 0; chunk_idx < CT::num_chunks; chunk_idx++) {
            const uint32_t tiles = (chunk_idx < CT::num_chunks - 1) ? CT::tiles_per_chunk : CT::last_chunk_tiles;
            const uint32_t payload_size = tiles * CT::page_size_bytes;

            headers[current_link]->to_noc_fused_unicast_write_atomic_inc(
                {dst_noc_base + offset, remote_sem_nocs[current_link], 1, false}, payload_size);
            connections[current_link].wait_for_empty_write_slot();
            connections[current_link].send_payload_without_header_non_blocking_from_address(
                src_addr + offset, payload_size);
            connections[current_link].send_payload_flush_non_blocking_from_address(
                reinterpret_cast<uint32_t>(headers[current_link]), sizeof(PACKET_HEADER_TYPE));

            offset += payload_size;
            if (++current_link == CT::num_links) {
                current_link = 0;
                noc_async_writes_flushed();
            }
        }
        noc_async_writes_flushed();

        cb_reserve_back(CT::sync_cb_id, 1);
        cb_push_back(CT::sync_cb_id, 1);

        for (uint32_t i = 0; i < CT::num_links; i++) {
            connections[i].close();
        }
#endif
    }
};

template <typename CT>
class Reader {
public:
    void operator()(const RTArgs& args) { impl(args); }

private:
    void impl([[maybe_unused]] const RTArgs& args) {
#if defined(COMPILE_FOR_BRISC)
        if constexpr (!CT::skip_local_push) {
            cb_reserve_back(CT::local_data_cb_id, CT::total_num_tiles);
            cb_push_back(CT::local_data_cb_id, CT::total_num_tiles);
        }
        if constexpr (CT::has_residual) {
            cb_reserve_back(CT::residual_cb_id, CT::total_num_tiles);
            cb_push_back(CT::residual_cb_id, CT::total_num_tiles);
        }

        const std::array<uint32_t, MAX_NUM_LINKS> sem_addrs = {args.sem_bank_addr_0, args.sem_bank_addr_1};
        std::array<volatile tt_l1_ptr uint32_t*, CT::num_links> sem_ptrs;
        std::array<uint32_t, CT::num_links> link_counters = {};

        for (uint32_t link_idx = 0; link_idx < CT::num_links; link_idx++) {
            sem_ptrs[link_idx] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addrs[link_idx]);
        }

        uint32_t current_link = 0;
        for (uint32_t chunk_idx = 0; chunk_idx < CT::num_chunks; chunk_idx++) {
            link_counters[current_link]++;
            noc_semaphore_wait_min(sem_ptrs[current_link], link_counters[current_link]);

            const uint32_t tiles = (chunk_idx < CT::num_chunks - 1) ? CT::tiles_per_chunk : CT::last_chunk_tiles;
            cb_reserve_back(CT::remote_data_cb_id, tiles);
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
    void operator()(const RTArgs& args) { impl(args); }

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

    void impl([[maybe_unused]] const RTArgs& args) {
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

        cb_wait_front(CT::sync_cb_id, 1);
        cb_pop_front(CT::cb_local, CT::num_tiles);
        cb_pop_front(CT::cb_remote, CT::num_tiles);
        cb_pop_front(CT::sync_cb_id, 1);
#endif
    }
};

}  // namespace AllReduce
}  // namespace deepseek_b1_ops
