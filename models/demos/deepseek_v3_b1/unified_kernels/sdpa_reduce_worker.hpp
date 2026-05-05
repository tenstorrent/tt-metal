// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"
#include "kernel_utils.hpp"
#include "dataflow_utils.hpp"

// =============================================================================
// Per-RISC includes
// =============================================================================

#if defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#include <cstdint>

using tt::data_movement::common::tt_memmove;

#elif defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include <cstdint>

#elif defined(COMPILE_FOR_TRISC)
// REDUCE_OP and REDUCE_DIM must be defined before including compute headers
#ifndef REDUCE_OP
#define REDUCE_OP (PoolType::MAX)
#endif
#ifndef REDUCE_DIM
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)
#endif
#ifndef EXP_APPROX_MODE
#define EXP_APPROX_MODE false
#endif

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "api/compute/pack.h"
#include <cstdint>

// Include SDPA LLK APIs for srcB reuse pattern and sdpa_tail reduction
#include "models/demos/deepseek_v3_b1/kernel_includes/tt_metal/include/compute_kernel_api/sdpa.h"
#endif

namespace deepseek_b1_ops {

// =============================================================================
// BRISC-only helper types for writer
// =============================================================================
#if defined(COMPILE_FOR_BRISC)

/** Per-round configuration for sending packets. */
struct SdpaRoundConfig {
    uint32_t cb_l;
    uint32_t cb_ms;
    uint32_t dst_mesh_id;
    uint32_t dst_chip_id;
    uint32_t dst_base_addr;
    uint32_t sem_addr;
    uint32_t fwd_slot_addr;
    uint32_t fwd_sem_addr;
    uint32_t base_slot_idx;
    uint32_t header_addr;
};

/**
 * Packet sender with cached core coordinates and round configuration.
 * Template parameters encode size constants for zero-overhead abstraction.
 */
template <
    uint32_t l1_alignment,
    uint32_t slot_size,
    uint32_t ms_tile_size_bytes,
    uint32_t l_chunk_size_bytes,
    uint32_t num_l_chunks,
    uint32_t tiles_per_l_chunk>
struct SdpaChunkSender {
    static constexpr bool use_posted_forwarder_writes = true;

    // Core coordinates (constant across rounds)
    uint32_t current_core_x;
    uint32_t current_core_y;
    uint32_t fwd_core_x;
    uint32_t fwd_core_y;

    // Round configuration (set by setup_round)
    SdpaRoundConfig cfg;

    // Precomputed NOC addresses (computed once per round in setup_round)
    uint64_t dst_base_noc;       // Fabric destination payload base for L chunks
    uint64_t sem_noc;            // Fabric destination semaphore
    uint64_t fwd_sem_noc;        // Forwarder semaphore
    volatile PACKET_HEADER_TYPE* header;

    // Derived constants
    static constexpr uint32_t total_l_bytes = num_l_chunks * l_chunk_size_bytes;
    static constexpr size_t packet_header_size_bytes = sizeof(PACKET_HEADER_TYPE);
    static constexpr uint32_t aligned_ms_payload_bytes = align(ms_tile_size_bytes, l1_alignment);
    static constexpr uint32_t aligned_l_chunk_payload_bytes = align(l_chunk_size_bytes, l1_alignment);

    // Slot indices: MS = slot 0, L_chunk_i = slot (1 + i)
    static constexpr uint32_t MS_SLOT_OFFSET = 0;
    static constexpr uint32_t L_SLOT_OFFSET = 1;

    FORCE_INLINE void setup_forwarder_write_state() const {
        // Only the forwarder core coordinate is fixed across rounds; the destination L1 address is still per-slot.
        ncrisc_noc_write_set_state</*posted=*/use_posted_forwarder_writes, /*one_packet=*/false>(
            noc_index, write_cmd_buf, get_noc_addr(fwd_core_x, fwd_core_y, 0), 0, NOC_UNICAST_WRITE_VC);
    }

    FORCE_INLINE void setup_round(const SdpaRoundConfig& new_cfg) {
        cfg = new_cfg;
        dst_base_noc = get_noc_addr(current_core_x, current_core_y, cfg.dst_base_addr);
        sem_noc = get_noc_addr(current_core_x, current_core_y, cfg.sem_addr);
        fwd_sem_noc = get_noc_addr(fwd_core_x, fwd_core_y, cfg.fwd_sem_addr);

        header = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(cfg.header_addr);

        // SDPA reduce-to-all only exchanges immediate torus neighbors, so the next hop is the final destination.
        const auto next_hop_direction = get_next_hop_router_direction(cfg.dst_mesh_id, cfg.dst_chip_id);
        fabric_set_single_hop_unicast_route_from_direction(
            header, next_hop_direction, cfg.dst_chip_id, cfg.dst_mesh_id);

        constexpr uint32_t ATOMIC_INC_VAL = 1;
        constexpr bool FLUSH_WRITES = false;
        header->to_noc_fused_unicast_write_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst_base_noc, sem_noc, ATOMIC_INC_VAL, FLUSH_WRITES},
            aligned_l_chunk_payload_bytes);
    }

    FORCE_INLINE void send_packet(
        uint64_t dst_noc,
        uint32_t fwd_slot_addr,
        uint32_t fwd_slot_idx,
        uint32_t src_addr,
        uint32_t payload_size) const {
        header->set_fused_unicast_write_atomic_inc_write_noc_address(dst_noc);
        ncrisc_noc_write_with_state<
            noc_mode,
            /*posted=*/use_posted_forwarder_writes,
            /*update_counter=*/true,
            /*one_packet=*/false>(noc_index, write_cmd_buf, cfg.header_addr, fwd_slot_addr, packet_header_size_bytes);
        ncrisc_noc_write_with_state<
            noc_mode,
            /*posted=*/use_posted_forwarder_writes,
            /*update_counter=*/true,
            /*one_packet=*/false>(
            noc_index, write_cmd_buf, src_addr, fwd_slot_addr + packet_header_size_bytes, payload_size);
        // The forwarder reads packet size from the staged header, so the whole slot must be visible before signaling.
        if constexpr (use_posted_forwarder_writes) {
            noc_async_posted_writes_flushed();
        } else {
            noc_async_writes_flushed();
        }
        noc_semaphore_inc(fwd_sem_noc, 1u << fwd_slot_idx);
    }

    FORCE_INLINE void send_ms(uint32_t src_addr) const {
        if constexpr (aligned_ms_payload_bytes != aligned_l_chunk_payload_bytes) {
            header->set_payload_size_bytes(aligned_ms_payload_bytes);
        }
        send_packet(dst_base_noc + total_l_bytes, cfg.fwd_slot_addr, cfg.base_slot_idx, src_addr, ms_tile_size_bytes);
    }

    template <bool streaming>
    FORCE_INLINE void send_l_chunks(uint32_t src_addr) const {
        if constexpr (aligned_ms_payload_bytes != aligned_l_chunk_payload_bytes) {
            header->set_payload_size_bytes(aligned_l_chunk_payload_bytes);
        }

        uint64_t current_dst_noc = dst_base_noc;
        uint32_t current_fwd_slot_addr = cfg.fwd_slot_addr + (L_SLOT_OFFSET * slot_size);
        uint32_t current_fwd_slot_idx = cfg.base_slot_idx + L_SLOT_OFFSET;
        uint32_t current_src_addr = src_addr;
        for (uint32_t i = 0; i < num_l_chunks; i++) {
            if constexpr (streaming) {
                cb_wait_front(cfg.cb_l, (i + 1) * tiles_per_l_chunk);
            }

            send_packet(
                current_dst_noc, current_fwd_slot_addr, current_fwd_slot_idx, current_src_addr, l_chunk_size_bytes);
            current_dst_noc += l_chunk_size_bytes;
            current_fwd_slot_addr += slot_size;
            current_fwd_slot_idx++;
            current_src_addr += l_chunk_size_bytes;
        }
    }

    FORCE_INLINE void send_all() const {
        cb_wait_front(cfg.cb_ms, 1);
        send_ms(get_read_ptr(cfg.cb_ms));
        cb_wait_front(cfg.cb_l, num_l_chunks * tiles_per_l_chunk);
        send_l_chunks</*streaming=*/false>(get_read_ptr(cfg.cb_l));
    }

    FORCE_INLINE void send_streaming() const {
        cb_wait_front(cfg.cb_ms, 1);
        send_ms(get_read_ptr(cfg.cb_ms));
        send_l_chunks</*streaming=*/true>(get_read_ptr(cfg.cb_l));
    }
};

#endif  // COMPILE_FOR_BRISC

// =============================================================================
// TRISC-only helper functions for compute
// =============================================================================
#if defined(COMPILE_FOR_TRISC)

/**
 * Streaming SDPA tail reduction that processes L tiles in chunks.
 */
template <
    bool SDPA_EXP_APPROX_MODE,
    bool normalize,
    bool untilize,
    uint32_t block_size,
    uint32_t scale_fp32,
    uint32_t num_l_chunks,
    int vector_mode = (int)VectorMode::C>
ALWI void sdpa_tail_streaming(
    uint32_t cb_worker_max_sum,
    uint32_t cb_prev_max_sum,
    uint32_t cb_cur_max_sum,
    uint32_t cb_l1,
    uint32_t cb_l2,
    uint32_t cb_l_out) {
    constexpr bool dense = untilize;
    constexpr uint32_t total_size = num_l_chunks * block_size;
    ckernel::sdpa_tail_ms_reduce<
        SDPA_EXP_APPROX_MODE,
        normalize,
        untilize ? total_size : block_size,
        scale_fp32,
        vector_mode,
        false,
        dense>(cb_worker_max_sum, cb_prev_max_sum, cb_cur_max_sum, cb_l1);

    // TODO: Unit test perf seemed better if we operated on all chunks
    // Retest in streaming context since unit test doesn't need to wait for input
    if constexpr (untilize) {
        custom_pack_untilize_dest_init<total_size, total_size, false, TILE_C_DIM, dense>(cb_l_out, 8, dense ? 2 : 4);
        cb_wait_front(cb_l1, total_size);
        cb_wait_front(cb_l2, total_size);
        cb_reserve_back(cb_l_out, total_size);
        ckernel::sdpa_tail_l_block<total_size, 1, untilize, dense, false>(cb_l1, cb_l2, cb_l_out, 0, 0, false);
        cb_push_back(cb_l_out, total_size);
        pack_untilize_uninit(cb_l_out);
    } else {
        bool acquire_regs = !normalize;
        pack_block_contiguous_init(cb_l_out);
        for (uint32_t chunk = 0; chunk < num_l_chunks; chunk++) {
            cb_wait_front(cb_l1, (chunk + 1) * block_size);
            cb_wait_front(cb_l2, (chunk + 1) * block_size);
            cb_reserve_back(cb_l_out, block_size);
            uint32_t tile_index = chunk * block_size;
            ckernel::sdpa_tail_l_block<block_size, 1, untilize, dense, false>(
                cb_l1, cb_l2, cb_l_out, tile_index, 0, acquire_regs);
            acquire_regs = true;
            cb_push_back(cb_l_out, block_size);
        }
    }

    // Postamble only — caller handles MS pops based on round context
    // (R1 inputs are TRISC-owned, R2 prev MS is BRISC-owned)
    ckernel::sdpa_bcast_col_reuse_postamble();
}

ALWI void sdpa_forward_data(
    uint32_t cb_prev_max_sum,
    uint32_t cb_cur_max_sum,
    uint32_t num_l_chunks,
    uint32_t cb_l1,
    uint32_t cb_l_out,
    uint32_t block_size) {
    copy_tile_init(cb_prev_max_sum);
    // Reconfigure from pack_block_contiguous mop configuration back to regular tile packing
    PACK((llk_pack_mop_config<false, false>(cb_cur_max_sum)));
    cb_wait_front(cb_prev_max_sum, 1);
    cb_reserve_back(cb_cur_max_sum, 1);

    tile_regs_acquire();
    copy_tile(cb_prev_max_sum, 0, 0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(0, cb_cur_max_sum);
    tile_regs_release();

    cb_push_back(cb_cur_max_sum, 1);
    pack_block_contiguous_init(cb_l_out);
    for (uint32_t chunk = 0; chunk < num_l_chunks; chunk++) {
        cb_wait_front(cb_l1, (chunk + 1) * block_size);
        cb_reserve_back(cb_l_out, block_size);

        uint32_t tile_index = chunk * block_size;
        tile_regs_acquire();
        for (uint32_t i = 0; i < block_size; i++) {
            copy_tile(cb_l1, tile_index + i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_block_contiguous(0, cb_l_out, block_size);
        tile_regs_release();
        cb_push_back(cb_l_out, block_size);
    }
}

template <
    bool SDPA_EXP_APPROX_MODE,
    bool normalize,
    bool untilize,
    uint32_t block_size,
    uint32_t scale_fp32,
    uint32_t num_l_chunks,
    int vector_mode = (int)VectorMode::C>
ALWI void sdpa_tail_streaming_conditional(
    uint32_t cb_worker_max_sum,
    uint32_t cb_prev_max_sum,
    uint32_t cb_cur_max_sum,
    uint32_t cb_l1,
    uint32_t cb_l2,
    uint32_t cb_l_out,
    bool neighbor_valid,
    bool local_valid,
    bool swap_reduction_order) {
    // Only local valid - copy local data to output
    if (!neighbor_valid && local_valid) {
        // TODO: Can be optimized to just division
        if constexpr (normalize) {
            sdpa_tail_streaming<
                SDPA_EXP_APPROX_MODE,
                normalize,
                untilize,
                block_size,
                scale_fp32,
                num_l_chunks,
                vector_mode>(cb_prev_max_sum, cb_prev_max_sum, cb_cur_max_sum, cb_l2, cb_l2, cb_l_out);
        } else {
            sdpa_forward_data(cb_prev_max_sum, cb_cur_max_sum, num_l_chunks, cb_l2, cb_l_out, block_size);
        }
        // Wait for remote data to arrive still since we need to properly pop the CBs
        cb_wait_front(cb_worker_max_sum, 1);
        cb_wait_front(cb_l1, num_l_chunks * block_size);
        return;
    }

    // Only neighbor valid - copy neighbor data to output
    if (neighbor_valid && !local_valid) {
        // TODO: Can be optimized to just division
        if constexpr (normalize) {
            sdpa_tail_streaming<
                SDPA_EXP_APPROX_MODE,
                normalize,
                untilize,
                block_size,
                scale_fp32,
                num_l_chunks,
                vector_mode>(cb_worker_max_sum, cb_worker_max_sum, cb_cur_max_sum, cb_l1, cb_l1, cb_l_out);
        } else {
            sdpa_forward_data(cb_worker_max_sum, cb_cur_max_sum, num_l_chunks, cb_l1, cb_l_out, block_size);
        }
        // We don't pop local data so no need to wait for it
        return;
    }

    // Just copy local data as fallback
    if (!neighbor_valid && !local_valid) {
        sdpa_forward_data(cb_prev_max_sum, cb_cur_max_sum, num_l_chunks, cb_l2, cb_l_out, block_size);
        // Wait for remote data to arrive still since we need to properly pop the CBs
        cb_wait_front(cb_worker_max_sum, 1);
        cb_wait_front(cb_l1, num_l_chunks * block_size);
        return;
    }

    // Both valid - perform normal SDPA reduction
    if (swap_reduction_order) {
        sdpa_tail_streaming<
            SDPA_EXP_APPROX_MODE,
            normalize,
            untilize,
            block_size,
            scale_fp32,
            num_l_chunks,
            vector_mode>(cb_prev_max_sum, cb_worker_max_sum, cb_cur_max_sum, cb_l2, cb_l1, cb_l_out);
    } else {
        sdpa_tail_streaming<
            SDPA_EXP_APPROX_MODE,
            normalize,
            untilize,
            block_size,
            scale_fp32,
            num_l_chunks,
            vector_mode>(cb_worker_max_sum, cb_prev_max_sum, cb_cur_max_sum, cb_l1, cb_l2, cb_l_out);
    }
}

#endif  // COMPILE_FOR_TRISC

// ============================================================================
// SDPA Reduce-to-All Worker Operations
//
// Worker core functionality:
// - NCRISC (Reader): Prepares local and neighbor MS/L data for compute
// - BRISC (Writer): Sends local/R1 data to neighbors via forwarders, scatters output
// - TRISC (Compute): Performs streaming SDPA tail reduction (R1 + R2)
// ============================================================================
struct SdpaReduceWorker {
    // ========================================================================
    // Compile-time args structs
    // ========================================================================

    // Reader CTArgs (NCRISC)
    template <
        uint32_t cbLocalL,
        uint32_t cbLocalMs,
        uint32_t cbNeighborL,
        uint32_t cbNeighborMs,
        uint32_t msTileSizeBytes,
        uint32_t lChunkSizeBytes,
        uint32_t numLChunks,
        uint32_t tilesPerLChunk,
        uint32_t positionEnabled,
        uint32_t perDeviceChunkSize>
    struct ReaderCTArgs {
        static constexpr uint32_t cb_local_l = cbLocalL;
        static constexpr uint32_t cb_local_ms = cbLocalMs;
        static constexpr uint32_t cb_neighbor_l = cbNeighborL;
        static constexpr uint32_t cb_neighbor_ms = cbNeighborMs;
        static constexpr uint32_t ms_tile_size_bytes = msTileSizeBytes;
        static constexpr uint32_t l_chunk_size_bytes = lChunkSizeBytes;
        static constexpr uint32_t num_l_chunks = numLChunks;
        static constexpr uint32_t tiles_per_l_chunk = tilesPerLChunk;
        static constexpr uint32_t position_enabled = positionEnabled;
        static constexpr uint32_t per_device_chunk_size = perDeviceChunkSize;
        // Derived constants
        static constexpr uint32_t out_tiles = numLChunks * tilesPerLChunk;
        static constexpr uint32_t total_l_bytes = numLChunks * lChunkSizeBytes;
        static constexpr uint32_t MS_SEM_THRESHOLD = 1;
        static constexpr uint32_t L_SEM_BASE_THRESHOLD = 2;
    };

    // Writer CTArgs (BRISC)
    template <
        uint32_t cbLocalL,
        uint32_t cbLocalMs,
        uint32_t cbR1ResultL,
        uint32_t cbR1ResultMs,
        uint32_t l1Alignment,
        uint32_t pageSizeBytes,
        uint32_t slotSize,
        uint32_t msTileSizeBytes,
        uint32_t lChunkSizeBytes,
        uint32_t numLChunks,
        uint32_t tilesPerLChunk,
        uint32_t cbLOut,
        uint32_t scatterNumTiles,
        uint32_t scatterSrcTileSize,
        uint32_t scatterDstTileSize,
        uint32_t scatterFaceSize,
        uint32_t scatterRowFaceSize,
        uint32_t scatterNumRows,
        uint32_t scatterArrivalEnabled = 0>
    struct WriterCTArgs {
        static constexpr uint32_t cb_local_l = cbLocalL;
        static constexpr uint32_t cb_local_ms = cbLocalMs;
        static constexpr uint32_t cb_r1_result_l = cbR1ResultL;
        static constexpr uint32_t cb_r1_result_ms = cbR1ResultMs;
        static constexpr uint32_t l1_alignment = l1Alignment;
        static constexpr uint32_t page_size_bytes = pageSizeBytes;
        static constexpr uint32_t slot_size = slotSize;
        static constexpr uint32_t ms_tile_size_bytes = msTileSizeBytes;
        static constexpr uint32_t l_chunk_size_bytes = lChunkSizeBytes;
        static constexpr uint32_t num_l_chunks = numLChunks;
        static constexpr uint32_t tiles_per_l_chunk = tilesPerLChunk;
        static constexpr uint32_t cb_l_out = cbLOut;
        static constexpr uint32_t scatter_num_tiles = scatterNumTiles;
        static constexpr uint32_t scatter_src_tile_size = scatterSrcTileSize;
        static constexpr uint32_t scatter_dst_tile_size = scatterDstTileSize;
        static constexpr uint32_t scatter_face_size = scatterFaceSize;
        static constexpr uint32_t scatter_row_face_size = scatterRowFaceSize;
        static constexpr uint32_t scatter_num_rows = scatterNumRows;
        // Optional scatter arrival semaphore: when enabled, signals each destination core
        // after scatter write completes (used by fused ops to synchronize downstream stages)
        static constexpr bool scatter_arrival_enabled = scatterArrivalEnabled != 0;
    };

    // Compute CTArgs (TRISC)
    template <
        uint32_t cbLocalL,
        uint32_t cbLocalMs,
        uint32_t cbNeighborL,
        uint32_t cbNeighborMs,
        uint32_t cbR1ResultL,
        uint32_t cbR1ResultMs,
        uint32_t cbLOut,
        uint32_t scaleFp32,
        uint32_t tilesPerLChunk,
        uint32_t numLChunks,
        uint32_t computeBlockSize,
        uint32_t positionEnabled,
        uint32_t perDeviceChunkSize,
        uint32_t finalReduction>
    struct ComputeCTArgs {
        static constexpr uint32_t cb_local_l = cbLocalL;
        static constexpr uint32_t cb_local_ms = cbLocalMs;
        static constexpr uint32_t cb_neighbor_l = cbNeighborL;
        static constexpr uint32_t cb_neighbor_ms = cbNeighborMs;
        static constexpr uint32_t cb_r1_result_l = cbR1ResultL;
        static constexpr uint32_t cb_r1_result_ms = cbR1ResultMs;
        static constexpr uint32_t cb_l_out = cbLOut;
        static constexpr uint32_t scale_fp32 = scaleFp32;
        static constexpr uint32_t transport_tiles_per_l_chunk = tilesPerLChunk;
        static constexpr uint32_t transport_num_l_chunks = numLChunks;
        static constexpr uint32_t position_enabled = positionEnabled;
        static constexpr uint32_t per_device_chunk_size = perDeviceChunkSize;
        static constexpr bool final_reduction = finalReduction;
        static constexpr uint32_t total_l_tiles = transport_num_l_chunks * transport_tiles_per_l_chunk;
        // Blackhole's non-dense SDPA srcB-reuse path tops out at 8 logical 8x32 tiles per block.
        static constexpr uint32_t max_compute_block_size = 8;
        // SDPA uses "block_size" terminology on the compute path.
        static constexpr uint32_t block_size = computeBlockSize;
        static_assert(block_size > 0, "compute block_size must be > 0");
        static_assert(block_size <= max_compute_block_size, "compute block_size exceeds supported maximum");
        static_assert(total_l_tiles % block_size == 0, "total_l_tiles must be divisible by compute block_size");
        static constexpr uint32_t num_l_blocks = total_l_tiles / block_size;
    };

    // ========================================================================
    // Runtime args structs
    // ========================================================================

    // Reader args (NCRISC): semaphore and buffer addresses for neighbor data
    struct ReaderArgs {
        uint32_t r1_neighbor_sem_addr;
        uint32_t r2_neighbor_sem_addr;
        uint32_t r1_recv_buffer_addr;
        uint32_t r2_recv_buffer_addr;
        // Position args (only meaningful when position_enabled CTArg is set)
        uint32_t global_pos;
        uint32_t r1_neighbor_device_idx;
        uint32_t r2_neighbor_device_idx;
        uint32_t r2_neighbor_r1_neighbor_idx;
    };

    // Writer args (BRISC): fabric destinations, core coordinates, forwarder config
    struct WriterArgs {
        uint32_t r1_dst_mesh_id;
        uint32_t r1_dst_chip_id;
        uint32_t r1_neighbor_dst_addr;
        uint32_t r1_neighbor_sem_addr;
        uint32_t r2_dst_mesh_id;
        uint32_t r2_dst_chip_id;
        uint32_t r2_neighbor_dst_addr;
        uint32_t r2_neighbor_sem_addr;
        uint32_t current_core_x;
        uint32_t current_core_y;
        uint32_t fwd_core_x;
        uint32_t fwd_core_y;
        uint32_t r1_fwd_slot_addr;
        uint32_t r1_fwd_sem_addr;
        uint32_t r1_base_slot_idx;
        uint32_t r2_fwd_slot_addr;
        uint32_t r2_fwd_sem_addr;
        uint32_t r2_base_slot_idx;
        uint32_t scatter_dest_l1_addr;
        uint32_t scatter_dest_coords_addr;
        uint32_t scatter_arrival_sem_addr;
    };

    // Compute args (TRISC): position validity for SDPA reduction
    struct ComputeArgs {
        uint32_t global_pos;
        uint32_t device_idx;
        uint32_t r1_neighbor_device_idx;
        uint32_t r2_neighbor_device_idx;
        uint32_t r2_neighbor_r1_neighbor_idx;
        uint32_t swap_r1_reduction_order;
        uint32_t swap_r2_reduction_order;
    };

    using RTArgs = unified_kernels::SelectByRISCV<ReaderArgs, WriterArgs, ComputeArgs>;

    // ========================================================================
    // Op - unified worker operation
    // ========================================================================
    template <typename CTArgs>
    class Op {
    public:
        void operator()([[maybe_unused]] const RTArgs& args) {
#if defined(COMPILE_FOR_NCRISC)
            reader_impl(args);
#elif defined(COMPILE_FOR_BRISC)
            writer_impl(args);
#elif defined(COMPILE_FOR_TRISC)
            compute_impl(args);
#endif
        }

        void set_global_pos([[maybe_unused]] RTArgs& args, [[maybe_unused]] uint32_t global_pos) {
#if defined(COMPILE_FOR_TRISC) || defined(COMPILE_FOR_NCRISC)
            args.global_pos = global_pos;
#endif
        }

    private:
#if defined(COMPILE_FOR_NCRISC)
        // ==================================================================
        // NCRISC (Reader) - prepares MS/L data for compute
        // ==================================================================
        FORCE_INLINE void prepare_ms_for_compute(
            uint32_t cb_ms, volatile tt_l1_ptr uint32_t* sem_ptr, uint32_t recv_buffer_addr) {
            // The pointer doesn't matter for NCRISC, just need to reserve and push
            cb_reserve_back(cb_ms, 1);
            noc_semaphore_wait_min(sem_ptr, CTArgs::MS_SEM_THRESHOLD);
            cb_push_back(cb_ms, 1);
        }

        FORCE_INLINE void prepare_l_chunk_for_compute(
            uint32_t cb_l, volatile tt_l1_ptr uint32_t* sem_ptr, uint32_t l_chunk_idx) {
            // The pointer doesn't matter for NCRISC, just need to reserve and push
            cb_reserve_back(cb_l, CTArgs::tiles_per_l_chunk);
            noc_semaphore_wait_min(sem_ptr, CTArgs::L_SEM_BASE_THRESHOLD + l_chunk_idx);
            cb_push_back(cb_l, CTArgs::tiles_per_l_chunk);
        }

        FORCE_INLINE void prepare_data_for_compute(
            uint32_t cb_l, uint32_t cb_ms, uint32_t sem_addr, uint32_t recv_buffer_addr) {
            volatile tt_l1_ptr uint32_t* sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
            prepare_ms_for_compute(cb_ms, sem_ptr, recv_buffer_addr);
            for (uint32_t i = 0; i < CTArgs::num_l_chunks; i++) {
                prepare_l_chunk_for_compute(cb_l, sem_ptr, i);
            }
            noc_semaphore_set(sem_ptr, 0);
        }

        void reader_impl(const ReaderArgs& args) {
            bool r2_neighbor_r1_valid = true;
            bool r1_neighbor_valid = true;

            if constexpr (CTArgs::position_enabled) {
                uint32_t position_id = args.global_pos;
                constexpr uint32_t chunk = CTArgs::per_device_chunk_size;
                r1_neighbor_valid = (position_id >= args.r1_neighbor_device_idx * chunk);
                r2_neighbor_r1_valid = (position_id >= args.r2_neighbor_device_idx * chunk) ||
                                       (position_id >= args.r2_neighbor_r1_neighbor_idx * chunk);
            }

            // Prepare R1 neighbor data for compute
            if (r1_neighbor_valid) {
                prepare_data_for_compute(
                    CTArgs::cb_neighbor_l, CTArgs::cb_neighbor_ms, args.r1_neighbor_sem_addr, args.r1_recv_buffer_addr);
            } else {
                cb_reserve_back(CTArgs::cb_neighbor_ms, 1);
                cb_push_back(CTArgs::cb_neighbor_ms, 1);
                cb_reserve_back(CTArgs::cb_neighbor_l, CTArgs::out_tiles);
                cb_push_back(CTArgs::cb_neighbor_l, CTArgs::out_tiles);
            }

            // Prepare R2 neighbor data for compute
            if (r2_neighbor_r1_valid) {
                prepare_data_for_compute(
                    CTArgs::cb_neighbor_l, CTArgs::cb_neighbor_ms, args.r2_neighbor_sem_addr, args.r2_recv_buffer_addr);
            } else {
                cb_reserve_back(CTArgs::cb_neighbor_ms, 1);
                cb_push_back(CTArgs::cb_neighbor_ms, 1);
                cb_reserve_back(CTArgs::cb_neighbor_l, CTArgs::out_tiles);
                cb_push_back(CTArgs::cb_neighbor_l, CTArgs::out_tiles);
            }

            // Clear the semaphores if the neighbor is not valid
            if (!r1_neighbor_valid) {
                volatile tt_l1_ptr uint32_t* sem_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.r1_neighbor_sem_addr);
                noc_semaphore_wait(sem_ptr, CTArgs::L_SEM_BASE_THRESHOLD + CTArgs::num_l_chunks - 1);
                noc_semaphore_set(sem_ptr, 0);
            }

            if (!r2_neighbor_r1_valid) {
                volatile tt_l1_ptr uint32_t* sem_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.r2_neighbor_sem_addr);
                noc_semaphore_wait(sem_ptr, CTArgs::L_SEM_BASE_THRESHOLD + CTArgs::num_l_chunks - 1);
                noc_semaphore_set(sem_ptr, 0);
            }
        }
#endif  // COMPILE_FOR_NCRISC

#if defined(COMPILE_FOR_BRISC)
        // ==================================================================
        // BRISC (Writer) - sends data to neighbors, scatters output
        // ==================================================================
        void writer_impl(const WriterArgs& args) {
            using Sender = SdpaChunkSender<
                CTArgs::l1_alignment,
                CTArgs::slot_size,
                CTArgs::ms_tile_size_bytes,
                CTArgs::l_chunk_size_bytes,
                CTArgs::num_l_chunks,
                CTArgs::tiles_per_l_chunk>;

            // Initialize sender with core coordinates
            Sender sender{args.current_core_x, args.current_core_y, args.fwd_core_x, args.fwd_core_y};
            sender.setup_forwarder_write_state();
            PacketHeaderPool::reset();
            auto* header = PacketHeaderPool::allocate_header(1);

            // ROUND 1: Send local input to R1 neighbor
            sender.setup_round(
                {CTArgs::cb_local_l,
                 CTArgs::cb_local_ms,
                 args.r1_dst_mesh_id,
                 args.r1_dst_chip_id,
                 args.r1_neighbor_dst_addr,
                 args.r1_neighbor_sem_addr,
                 args.r1_fwd_slot_addr,
                 args.r1_fwd_sem_addr,
                 args.r1_base_slot_idx,
                 reinterpret_cast<uint32_t>(header)});
            sender.send_all();

            // ROUND 2: Send R1 result to R2 neighbor (streaming)
            sender.setup_round(
                {CTArgs::cb_r1_result_l,
                 CTArgs::cb_r1_result_ms,
                 args.r2_dst_mesh_id,
                 args.r2_dst_chip_id,
                 args.r2_neighbor_dst_addr,
                 args.r2_neighbor_sem_addr,
                 args.r2_fwd_slot_addr,
                 args.r2_fwd_sem_addr,
                 args.r2_base_slot_idx,
                 reinterpret_cast<uint32_t>(header)});
            sender.send_streaming();

            noc_async_full_barrier();

            // SCATTER PHASE: Distribute output rows to destination cores
            if constexpr (CTArgs::scatter_num_rows > 0) {
                tt_l1_ptr uint32_t* scatter_dest_coords = (tt_l1_ptr uint32_t*)(args.scatter_dest_coords_addr);
                uint32_t scatter_dest_noc_x[CTArgs::scatter_num_rows];
                uint32_t scatter_dest_noc_y[CTArgs::scatter_num_rows];
                for (uint32_t i = 0; i < CTArgs::scatter_num_rows; i++) {
                    scatter_dest_noc_x[i] = scatter_dest_coords[i * 2];
                    scatter_dest_noc_y[i] = scatter_dest_coords[i * 2 + 1];
                }

                constexpr uint32_t scatter_payload_bytes = CTArgs::scatter_num_tiles * CTArgs::scatter_dst_tile_size;
                static_assert(scatter_payload_bytes <= NOC_MAX_BURST_SIZE);
                constexpr bool posted = CTArgs::scatter_arrival_enabled;
                uint32_t src_addr = get_read_ptr(CTArgs::cb_l_out);

                uint64_t dest_noc_addr =
                    get_noc_addr(scatter_dest_noc_x[0], scatter_dest_noc_y[0], args.scatter_dest_l1_addr);
                unified_kernels::unicast_write_set_state<posted, true, true, true, false, write_cmd_buf>(
                    src_addr, dest_noc_addr, scatter_payload_bytes);

                if constexpr (CTArgs::scatter_arrival_enabled) {
                    uint64_t sem_addr =
                        get_noc_addr(scatter_dest_noc_x[0], scatter_dest_noc_y[0], args.scatter_arrival_sem_addr);
                    unified_kernels::unicast_atomic_inc_set_state<false, true, true, false, write_at_cmd_buf>(
                        sem_addr, 1);
                }

                cb_wait_front(CTArgs::cb_l_out, CTArgs::scatter_num_tiles);
                unified_kernels::unicast_write_increment_counters<posted>(CTArgs::scatter_num_rows);
                unified_kernels::noc_async_write_issue_txn<posted, false>();
                if constexpr (CTArgs::scatter_arrival_enabled) {
                    unified_kernels::unicast_atomic_inc_increment_counters<false>(CTArgs::scatter_num_rows);
                    unified_kernels::noc_async_atomic_inc_issue_txn<false, false>();
                }
                src_addr += scatter_payload_bytes;

                for (uint32_t row = 1; row < CTArgs::scatter_num_rows; row++) {
                    uint64_t dest_noc_addr =
                        get_noc_addr(scatter_dest_noc_x[row], scatter_dest_noc_y[row], args.scatter_dest_l1_addr);
                    unified_kernels::unicast_write_set_state<posted, true, true, false, false, write_cmd_buf>(
                        src_addr, dest_noc_addr, scatter_payload_bytes);

                    unified_kernels::noc_async_write_issue_txn<posted, false>();

                    // Signal scatter arrival on destination core (used by fused ops
                    // to synchronize downstream stages like matmul1)
                    if constexpr (CTArgs::scatter_arrival_enabled) {
                        uint64_t sem_addr = get_noc_addr(
                            scatter_dest_noc_x[row], scatter_dest_noc_y[row], args.scatter_arrival_sem_addr);
                        unified_kernels::unicast_atomic_inc_set_state<false, true, false, false, write_at_cmd_buf>(
                            sem_addr, 1);
                        unified_kernels::noc_async_atomic_inc_issue_txn<false, false>();
                    }
                    src_addr += scatter_payload_bytes;
                }
                if constexpr (CTArgs::scatter_arrival_enabled) {
                    noc_async_atomic_barrier();
                } else {
                    noc_async_write_barrier();
                }
                cb_pop_front(CTArgs::cb_l_out, CTArgs::scatter_num_tiles);
            }
        }
#endif  // COMPILE_FOR_BRISC

#if defined(COMPILE_FOR_TRISC)
        // ==================================================================
        // TRISC (Compute) - streaming SDPA tail reduction
        // ==================================================================
        void compute_impl([[maybe_unused]] const ComputeArgs& args) {
            constexpr int vector_mode = VectorMode::RC_custom;

            reconfig_data_format<false, true>(CTArgs::cb_local_l, CTArgs::cb_local_l);
            pack_reconfig_data_format<true>(CTArgs::cb_l_out);
            exp_tile_init<EXP_APPROX_MODE>();

            bool local_valid = true;
            bool r1_neighbor_valid = true;
            bool r2_neighbor_valid = true;
            bool swap_r1_reduction_order = false;
            bool swap_r2_reduction_order = false;

            [[maybe_unused]] uint32_t device_idx = 0;
            [[maybe_unused]] uint32_t r1_neighbor_device_idx = 0;
            [[maybe_unused]] uint32_t r2_neighbor_device_idx = 0;
            [[maybe_unused]] uint32_t position_id = 0;

            if constexpr (CTArgs::position_enabled) {
                device_idx = args.device_idx;
                r1_neighbor_device_idx = args.r1_neighbor_device_idx;
                r2_neighbor_device_idx = args.r2_neighbor_device_idx;

                position_id = args.global_pos;

                constexpr uint32_t chunk = CTArgs::per_device_chunk_size;
                local_valid = (position_id >= device_idx * chunk);
                r1_neighbor_valid = (position_id >= r1_neighbor_device_idx * chunk);
                r2_neighbor_valid = (position_id >= r2_neighbor_device_idx * chunk) ||
                                    (position_id >= args.r2_neighbor_r1_neighbor_idx * chunk);
                swap_r1_reduction_order = args.swap_r1_reduction_order;
                swap_r2_reduction_order = args.swap_r2_reduction_order;
            }

            uint32_t neighbor_cb_base_rd_ptr = 0;
            uint32_t neighbor_cb_page_size = 0;
            UNPACK(({
                neighbor_cb_base_rd_ptr = unified_kernels::get_local_cb_rd_ptr(CTArgs::cb_neighbor_l);
                neighbor_cb_page_size = unified_kernels::get_local_cb_page_size(CTArgs::cb_neighbor_l);
                unified_kernels::update_local_cb_rd_ptr(
                    CTArgs::cb_neighbor_l, neighbor_cb_base_rd_ptr + 0 * neighbor_cb_page_size);
                unified_kernels::update_local_cb_rd_ptr(
                    CTArgs::cb_neighbor_ms, neighbor_cb_base_rd_ptr + CTArgs::total_l_tiles * neighbor_cb_page_size);
            }));

            // ROUND 1: reduce(local, r1_neighbor) -> r1_result (unnormalized)
            sdpa_tail_streaming_conditional<
                EXP_APPROX_MODE,
                false /* no normalize - R1 doesn't normalize */,
                false /* untilize - R1 doesn't untilize */,
                CTArgs::block_size,
                CTArgs::scale_fp32,
                CTArgs::num_l_blocks,
                vector_mode>(
                CTArgs::cb_neighbor_ms,
                CTArgs::cb_local_ms,
                CTArgs::cb_r1_result_ms,
                CTArgs::cb_neighbor_l,
                CTArgs::cb_local_l,
                CTArgs::cb_r1_result_l,
                r1_neighbor_valid,
                local_valid,
                swap_r1_reduction_order);

            // Pop R1 input MS CBs — TRISC-owned, no BRISC race.
            // cb_r1_neighbor_ms: incoming from neighbor, consumed only by TRISC
            // No cb_wait_front needed: NCRISC always pushes all MS CBs unconditionally
            // (even for invalid neighbors with dummy data), so tiles are guaranteed present.
            cb_pop_front(CTArgs::cb_neighbor_ms, 1);
            cb_pop_front(CTArgs::cb_neighbor_l, CTArgs::total_l_tiles);

            UNPACK(({
                unified_kernels::update_local_cb_rd_ptr(
                    CTArgs::cb_neighbor_l,
                    neighbor_cb_base_rd_ptr + (CTArgs::total_l_tiles + 1) * neighbor_cb_page_size);
                unified_kernels::update_local_cb_rd_ptr(
                    CTArgs::cb_neighbor_ms,
                    neighbor_cb_base_rd_ptr + (2 * CTArgs::total_l_tiles + 1) * neighbor_cb_page_size);
            }));

            // ROUND 2: reduce(r1_result, r2_neighbor) -> final output (normalized L)
            bool local_r1_valid = local_valid || r1_neighbor_valid;
            bool r2_neighbor_r1_valid = r2_neighbor_valid;

            // TODO: This is because only the reduction can currently perform untilize
            static_assert(CTArgs::final_reduction, "Final reduction must be enabled");
            sdpa_tail_streaming_conditional<
                EXP_APPROX_MODE,
                CTArgs::final_reduction,
                true /* untilize */,
                CTArgs::block_size,
                CTArgs::scale_fp32,
                CTArgs::num_l_blocks,
                vector_mode>(
                CTArgs::cb_neighbor_ms,
                CTArgs::cb_r1_result_ms,
                0,  // sdpa tail does not output MS when normalizing
                CTArgs::cb_neighbor_l,
                CTArgs::cb_r1_result_l,
                CTArgs::cb_l_out,
                r2_neighbor_r1_valid,
                local_r1_valid,
                swap_r2_reduction_order);

            // Pop R2 worker MS CB — incoming from neighbor, consumed only by TRISC.
            // Do NOT pop cb_r1_result_ms — BRISC owns that pop (writer_impl line 668)
            // after send_streaming() reads it for R2 forwarding.
            cb_pop_front(CTArgs::cb_neighbor_ms, 1);
            cb_pop_front(CTArgs::cb_neighbor_l, CTArgs::total_l_tiles);
            UNPACK(({
                unified_kernels::update_local_cb_rd_ptr(CTArgs::cb_neighbor_l, neighbor_cb_base_rd_ptr);
                unified_kernels::update_local_cb_rd_ptr(CTArgs::cb_neighbor_ms, neighbor_cb_base_rd_ptr);
            }));
        }
#endif  // COMPILE_FOR_TRISC
    };

};  // struct SdpaReduceWorker

}  // namespace deepseek_b1_ops
