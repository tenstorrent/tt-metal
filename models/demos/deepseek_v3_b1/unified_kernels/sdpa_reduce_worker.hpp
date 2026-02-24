// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"

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

/** Fabric destination addresses (where data lands on remote device) */
struct SdpaFabricDest {
    uint64_t dst_noc;  // Destination L1 address (varies per packet)
    uint64_t sem_noc;  // Semaphore address (constant per round)
};

/** Forwarder destination addresses (local forwarder slot) */
struct SdpaForwarderDest {
    uint64_t slot_noc;  // Forwarder slot NOC address
    uint64_t sem_noc;   // Forwarder semaphore NOC address (constant per round)
    uint32_t slot_idx;  // Slot index for bit-packed signaling
};

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
};

/**
 * Packet sender with cached core coordinates and round configuration.
 * Template parameters encode size constants for zero-overhead abstraction.
 */
template <
    uint32_t cb_packet_slot,
    uint32_t l1_alignment,
    uint32_t slot_size,
    uint32_t ms_tile_size_bytes,
    uint32_t l_chunk_size_bytes,
    uint32_t num_l_chunks,
    uint32_t tiles_per_l_chunk>
struct SdpaChunkSender {
    // Core coordinates (constant across rounds)
    uint32_t current_core_x;
    uint32_t current_core_y;
    uint32_t fwd_core_x;
    uint32_t fwd_core_y;

    // Round configuration (set by setup_round)
    SdpaRoundConfig cfg;

    // Precomputed NOC addresses (computed once per round in setup_round)
    uint64_t sem_noc;      // Fabric destination semaphore
    uint64_t fwd_sem_noc;  // Forwarder semaphore

    // Cached header address (set once per round, reused for all packets)
    uint32_t header_addr;

    // Derived constants
    static constexpr uint32_t total_l_bytes = num_l_chunks * l_chunk_size_bytes;
    static constexpr size_t packet_header_size_bytes = sizeof(PACKET_HEADER_TYPE);

    // Slot indices: MS = slot 0, L_chunk_i = slot (1 + i)
    static constexpr uint32_t MS_SLOT_OFFSET = 0;
    static constexpr uint32_t L_SLOT_OFFSET = 1;

    FORCE_INLINE void setup_round(const SdpaRoundConfig& new_cfg) {
        cfg = new_cfg;
        sem_noc = get_noc_addr(current_core_x, current_core_y, cfg.sem_addr);
        fwd_sem_noc = get_noc_addr(fwd_core_x, fwd_core_y, cfg.fwd_sem_addr);

        cb_reserve_back(cb_packet_slot, 1);
        header_addr = get_write_ptr(cb_packet_slot);

        auto* header = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(header_addr);
        (void)fabric_set_unicast_route(header, cfg.dst_chip_id, cfg.dst_mesh_id);
    }

    FORCE_INLINE void finish_round() {
        cb_push_back(cb_packet_slot, 1);
        cb_pop_front(cb_packet_slot, 1);
    }

    FORCE_INLINE SdpaFabricDest get_fabric_dest(uint32_t dst_addr) const {
        return {
            get_noc_addr(current_core_x, current_core_y, dst_addr),
            sem_noc  // precomputed
        };
    }

    template <bool is_ms>
    FORCE_INLINE SdpaForwarderDest get_forwarder_dest(uint32_t l_chunk_idx = 0) const {
        uint32_t slot_offset = is_ms ? MS_SLOT_OFFSET : (L_SLOT_OFFSET + l_chunk_idx);
        uint32_t slot_idx = cfg.base_slot_idx + slot_offset;
        uint32_t fwd_slot_addr = cfg.fwd_slot_addr + slot_offset * slot_size;
        return {
            get_noc_addr(fwd_core_x, fwd_core_y, fwd_slot_addr),
            fwd_sem_noc,  // precomputed
            slot_idx};
    }

    FORCE_INLINE void send_packet(
        const SdpaFabricDest& fabric_dest,
        const SdpaForwarderDest& fwd_dest,
        uint32_t src_addr,
        uint32_t payload_size) const {
        auto* header = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(header_addr);
        constexpr uint32_t ATOMIC_INC_VAL = 1;
        constexpr bool FLUSH_WRITES = false;
        header->to_noc_fused_unicast_write_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                fabric_dest.dst_noc, fabric_dest.sem_noc, ATOMIC_INC_VAL, FLUSH_WRITES},
            align(payload_size, l1_alignment));

        noc_async_write(header_addr, fwd_dest.slot_noc, packet_header_size_bytes);
        uint64_t fwd_payload_noc = fwd_dest.slot_noc + packet_header_size_bytes;
        noc_async_write(src_addr, fwd_payload_noc, payload_size);
        noc_async_writes_flushed();
        noc_semaphore_inc(fwd_dest.sem_noc, 1u << fwd_dest.slot_idx);
    }

    FORCE_INLINE void send_ms() const {
        uint32_t dst_addr = cfg.dst_base_addr + total_l_bytes;
        auto fabric_dest = get_fabric_dest(dst_addr);
        auto fwd_dest = get_forwarder_dest<true>();
        send_packet(fabric_dest, fwd_dest, get_read_ptr(cfg.cb_ms), ms_tile_size_bytes);
    }

    FORCE_INLINE void send_l_chunk(uint32_t l_chunk_idx) const {
        uint32_t dst_addr = cfg.dst_base_addr + l_chunk_idx * l_chunk_size_bytes;
        uint32_t src_addr = get_read_ptr(cfg.cb_l) + l_chunk_idx * l_chunk_size_bytes;
        auto fabric_dest = get_fabric_dest(dst_addr);
        auto fwd_dest = get_forwarder_dest<false>(l_chunk_idx);
        send_packet(fabric_dest, fwd_dest, src_addr, l_chunk_size_bytes);
    }

    FORCE_INLINE void send_all() const {
        send_ms();
        for (uint32_t i = 0; i < num_l_chunks; i++) {
            send_l_chunk(i);
        }
    }

    FORCE_INLINE void send_streaming() const {
        cb_wait_front(cfg.cb_ms, 1);
        send_ms();

        for (uint32_t i = 0; i < num_l_chunks; i++) {
            cb_wait_front(cfg.cb_l, (i + 1) * tiles_per_l_chunk);
            send_l_chunk(i);
        }
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
        pack_untilize_dest_init<total_size, total_size, false, TILE_C_DIM, dense>(cb_l_out, 8, dense ? 2 : 4);
        cb_wait_front(cb_l1, total_size);
        cb_wait_front(cb_l2, total_size);
        cb_reserve_back(cb_l_out, total_size);
        ckernel::sdpa_tail_l_block<total_size, 1, untilize, dense, false>(cb_l1, cb_l2, cb_l_out, 0, 0, false);
        cb_push_back(cb_l_out, total_size);
        pack_untilize_uninit(cb_l_out);
    } else {
        bool acquire_regs = !normalize;
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
    cb_wait_front(cb_prev_max_sum, 1);
    cb_reserve_back(cb_cur_max_sum, 1);

    tile_regs_acquire();
    copy_tile_init(cb_prev_max_sum);
    copy_tile(cb_prev_max_sum, 0, 0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(0, cb_cur_max_sum);
    tile_regs_release();

    cb_push_back(cb_cur_max_sum, 1);

    for (uint32_t chunk = 0; chunk < num_l_chunks; chunk++) {
        cb_wait_front(cb_l1, (chunk + 1) * block_size);
        cb_reserve_back(cb_l_out, block_size);

        uint32_t tile_index = chunk * block_size;
        for (uint32_t i = 0; i < block_size; i++) {
            tile_regs_acquire();
            copy_tile_init(cb_l1);
            copy_tile(cb_l1, tile_index + i, i);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(i, cb_l_out);
            tile_regs_release();
        }

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
    bool local_valid) {
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
        return;
    }

    // Just copy local data as fallback
    if (!neighbor_valid && !local_valid) {
        sdpa_forward_data(cb_prev_max_sum, cb_cur_max_sum, num_l_chunks, cb_l2, cb_l_out, block_size);
        return;
    }

    // Both valid - perform normal SDPA reduction
    sdpa_tail_streaming<SDPA_EXP_APPROX_MODE, normalize, untilize, block_size, scale_fp32, num_l_chunks, vector_mode>(
        cb_worker_max_sum, cb_prev_max_sum, cb_cur_max_sum, cb_l1, cb_l2, cb_l_out);
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
        uint32_t cbR1NeighborL,
        uint32_t cbR1NeighborMs,
        uint32_t cbR2NeighborL,
        uint32_t cbR2NeighborMs,
        uint32_t msTileSizeBytes,
        uint32_t lChunkSizeBytes,
        uint32_t numLChunks,
        uint32_t tilesPerLChunk,
        uint32_t positionEnabled,
        uint32_t perDeviceChunkSize>
    struct ReaderCTArgs {
        static constexpr uint32_t cb_local_l = cbLocalL;
        static constexpr uint32_t cb_local_ms = cbLocalMs;
        static constexpr uint32_t cb_r1_neighbor_l = cbR1NeighborL;
        static constexpr uint32_t cb_r1_neighbor_ms = cbR1NeighborMs;
        static constexpr uint32_t cb_r2_neighbor_l = cbR2NeighborL;
        static constexpr uint32_t cb_r2_neighbor_ms = cbR2NeighborMs;
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
        uint32_t cbPacketSlot,
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
        uint32_t scatterArrivalEnabled = 0,
        uint32_t scatterArrivalSemaphoreId = 0>
    struct WriterCTArgs {
        static constexpr uint32_t cb_local_l = cbLocalL;
        static constexpr uint32_t cb_local_ms = cbLocalMs;
        static constexpr uint32_t cb_r1_result_l = cbR1ResultL;
        static constexpr uint32_t cb_r1_result_ms = cbR1ResultMs;
        static constexpr uint32_t cb_packet_slot = cbPacketSlot;
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
        static constexpr uint32_t scatter_arrival_semaphore_id = scatterArrivalSemaphoreId;
    };

    // Compute CTArgs (TRISC)
    template <
        uint32_t cbLocalL,
        uint32_t cbLocalMs,
        uint32_t cbR1NeighborL,
        uint32_t cbR1NeighborMs,
        uint32_t cbR1ResultL,
        uint32_t cbR1ResultMs,
        uint32_t cbR2NeighborL,
        uint32_t cbR2NeighborMs,
        uint32_t cbLOut,
        uint32_t cbMsOut,
        uint32_t scaleFp32,
        uint32_t tilesPerLChunk,
        uint32_t numLChunks,
        uint32_t positionEnabled,
        uint32_t perDeviceChunkSize,
        uint32_t finalReduction>
    struct ComputeCTArgs {
        static constexpr uint32_t cb_local_l = cbLocalL;
        static constexpr uint32_t cb_local_ms = cbLocalMs;
        static constexpr uint32_t cb_r1_neighbor_l = cbR1NeighborL;
        static constexpr uint32_t cb_r1_neighbor_ms = cbR1NeighborMs;
        static constexpr uint32_t cb_r1_result_l = cbR1ResultL;
        static constexpr uint32_t cb_r1_result_ms = cbR1ResultMs;
        static constexpr uint32_t cb_r2_neighbor_l = cbR2NeighborL;
        static constexpr uint32_t cb_r2_neighbor_ms = cbR2NeighborMs;
        static constexpr uint32_t cb_l_out = cbLOut;
        static constexpr uint32_t cb_ms_out = cbMsOut;
        static constexpr uint32_t scale_fp32 = scaleFp32;
        static constexpr uint32_t tiles_per_l_chunk = tilesPerLChunk;
        static constexpr uint32_t num_l_chunks = numLChunks;
        static constexpr uint32_t position_enabled = positionEnabled;
        static constexpr uint32_t per_device_chunk_size = perDeviceChunkSize;
        static constexpr bool final_reduction = finalReduction;
        // SDPA uses "block_size" terminology
        static constexpr uint32_t block_size = tilesPerLChunk;
    };

    // ========================================================================
    // Op - unified worker operation
    //
    // ReaderCT: compile-time args for NCRISC reader
    // WriterCT: compile-time args for BRISC writer
    // ComputeCT: compile-time args for TRISC compute
    // ========================================================================
    template <typename ReaderCT, typename WriterCT, typename ComputeCT>
    class Op {
    public:
        void operator()() {
#if defined(COMPILE_FOR_NCRISC)
            reader_impl();
#elif defined(COMPILE_FOR_BRISC)
            writer_impl();
#elif defined(COMPILE_FOR_TRISC)
            compute_impl();
#endif
        }

    private:
#if defined(COMPILE_FOR_NCRISC)
        // ==================================================================
        // NCRISC (Reader) - prepares MS/L data for compute
        // ==================================================================
        FORCE_INLINE void prepare_ms_for_compute(
            uint32_t cb_ms, volatile tt_l1_ptr uint32_t* sem_ptr, uint32_t recv_buffer_addr) {
            cb_reserve_back(cb_ms, 1);
            noc_semaphore_wait_min(sem_ptr, ReaderCT::MS_SEM_THRESHOLD);
            tt_memmove<true, false, false, 0>(
                get_write_ptr(cb_ms), recv_buffer_addr + ReaderCT::total_l_bytes, ReaderCT::ms_tile_size_bytes);
            cb_push_back(cb_ms, 1);
        }

        FORCE_INLINE void prepare_l_chunk_for_compute(
            uint32_t cb_l, volatile tt_l1_ptr uint32_t* sem_ptr, uint32_t l_chunk_idx) {
            cb_reserve_back(cb_l, ReaderCT::tiles_per_l_chunk);
            noc_semaphore_wait_min(sem_ptr, ReaderCT::L_SEM_BASE_THRESHOLD + l_chunk_idx);
            cb_push_back(cb_l, ReaderCT::tiles_per_l_chunk);
        }

        FORCE_INLINE void prepare_data_for_compute(
            uint32_t cb_l, uint32_t cb_ms, uint32_t sem_addr, uint32_t recv_buffer_addr) {
            volatile tt_l1_ptr uint32_t* sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
            prepare_ms_for_compute(cb_ms, sem_ptr, recv_buffer_addr);
            for (uint32_t i = 0; i < ReaderCT::num_l_chunks; i++) {
                prepare_l_chunk_for_compute(cb_l, sem_ptr, i);
            }
            noc_semaphore_set(sem_ptr, 0);
        }

        void reader_impl() {
            size_t arg_idx = 0;
            const uint32_t r1_neighbor_sem_addr = get_arg_val<uint32_t>(arg_idx++);
            const uint32_t r2_neighbor_sem_addr = get_arg_val<uint32_t>(arg_idx++);
            const uint32_t r1_recv_buffer_addr = get_arg_val<uint32_t>(arg_idx++);
            const uint32_t r2_recv_buffer_addr = get_arg_val<uint32_t>(arg_idx++);

            // Push local input (aliased CBs, no copy needed)
            cb_reserve_back(ReaderCT::cb_local_l, ReaderCT::out_tiles);
            cb_push_back(ReaderCT::cb_local_l, ReaderCT::out_tiles);

            cb_reserve_back(ReaderCT::cb_local_ms, 1);
            cb_push_back(ReaderCT::cb_local_ms, 1);

            bool r2_neighbor_r1_valid = true;
            bool r1_neighbor_valid = true;

            if constexpr (ReaderCT::position_enabled) {
                uint32_t pos_addr = get_arg_val<uint32_t>(arg_idx++);
                uint32_t r1_neighbor_device_idx = get_arg_val<uint32_t>(arg_idx++);
                uint32_t r2_neighbor_device_idx = get_arg_val<uint32_t>(arg_idx++);
                uint32_t r2_neighbor_r1_neighbor_idx = get_arg_val<uint32_t>(arg_idx++);
                // Read position_id from HEIGHT_SHARDED L1 tensor
                volatile tt_l1_ptr uint32_t* pos_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pos_addr);
                uint32_t position_id = pos_ptr[0];
                constexpr uint32_t chunk = ReaderCT::per_device_chunk_size;
                r1_neighbor_valid = (position_id >= r1_neighbor_device_idx * chunk);
                r2_neighbor_r1_valid = (position_id >= r2_neighbor_device_idx * chunk) ||
                                       (position_id >= r2_neighbor_r1_neighbor_idx * chunk);
            }

            // Prepare R1 neighbor data for compute
            if (r1_neighbor_valid) {
                prepare_data_for_compute(
                    ReaderCT::cb_r1_neighbor_l, ReaderCT::cb_r1_neighbor_ms, r1_neighbor_sem_addr, r1_recv_buffer_addr);
            } else {
                cb_reserve_back(ReaderCT::cb_r1_neighbor_ms, 1);
                cb_push_back(ReaderCT::cb_r1_neighbor_ms, 1);
                cb_reserve_back(ReaderCT::cb_r1_neighbor_l, ReaderCT::out_tiles);
                cb_push_back(ReaderCT::cb_r1_neighbor_l, ReaderCT::out_tiles);
            }

            // Prepare R2 neighbor data for compute
            if (r2_neighbor_r1_valid) {
                prepare_data_for_compute(
                    ReaderCT::cb_r2_neighbor_l, ReaderCT::cb_r2_neighbor_ms, r2_neighbor_sem_addr, r2_recv_buffer_addr);
            } else {
                cb_reserve_back(ReaderCT::cb_r2_neighbor_ms, 1);
                cb_push_back(ReaderCT::cb_r2_neighbor_ms, 1);
                cb_reserve_back(ReaderCT::cb_r2_neighbor_l, ReaderCT::out_tiles);
                cb_push_back(ReaderCT::cb_r2_neighbor_l, ReaderCT::out_tiles);
            }
        }
#endif  // COMPILE_FOR_NCRISC

#if defined(COMPILE_FOR_BRISC)
        // ==================================================================
        // BRISC (Writer) - sends data to neighbors, scatters output
        // ==================================================================
        void writer_impl() {
            using Sender = SdpaChunkSender<
                WriterCT::cb_packet_slot,
                WriterCT::l1_alignment,
                WriterCT::slot_size,
                WriterCT::ms_tile_size_bytes,
                WriterCT::l_chunk_size_bytes,
                WriterCT::num_l_chunks,
                WriterCT::tiles_per_l_chunk>;

            size_t arg_idx = 0;

            // R1 destination
            const uint32_t r1_dst_mesh_id = get_arg_val<uint32_t>(arg_idx++);
            const uint32_t r1_dst_chip_id = get_arg_val<uint32_t>(arg_idx++);
            const uint32_t r1_neighbor_dst_addr = get_arg_val<uint32_t>(arg_idx++);
            const uint32_t r1_neighbor_sem_addr = get_arg_val<uint32_t>(arg_idx++);

            // R2 destination
            const uint32_t r2_dst_mesh_id = get_arg_val<uint32_t>(arg_idx++);
            const uint32_t r2_dst_chip_id = get_arg_val<uint32_t>(arg_idx++);
            const uint32_t r2_neighbor_dst_addr = get_arg_val<uint32_t>(arg_idx++);
            const uint32_t r2_neighbor_sem_addr = get_arg_val<uint32_t>(arg_idx++);

            // Core coordinates
            const uint32_t current_core_x = get_arg_val<uint32_t>(arg_idx++);
            const uint32_t current_core_y = get_arg_val<uint32_t>(arg_idx++);
            const uint32_t fwd_core_x = get_arg_val<uint32_t>(arg_idx++);
            const uint32_t fwd_core_y = get_arg_val<uint32_t>(arg_idx++);

            // R1 forwarder slot info
            const uint32_t r1_fwd_slot_addr = get_arg_val<uint32_t>(arg_idx++);
            const uint32_t r1_fwd_sem_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
            const uint32_t r1_base_slot_idx = get_arg_val<uint32_t>(arg_idx++);

            // R2 forwarder slot info
            const uint32_t r2_fwd_slot_addr = get_arg_val<uint32_t>(arg_idx++);
            const uint32_t r2_fwd_sem_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
            const uint32_t r2_base_slot_idx = get_arg_val<uint32_t>(arg_idx++);

            // Initialize sender with core coordinates
            Sender sender{current_core_x, current_core_y, fwd_core_x, fwd_core_y};

            // ROUND 1: Send local input to R1 neighbor
            sender.setup_round(
                {WriterCT::cb_local_l,
                 WriterCT::cb_local_ms,
                 r1_dst_mesh_id,
                 r1_dst_chip_id,
                 r1_neighbor_dst_addr,
                 r1_neighbor_sem_addr,
                 r1_fwd_slot_addr,
                 r1_fwd_sem_addr,
                 r1_base_slot_idx});
            sender.send_all();
            sender.finish_round();

            // ROUND 2: Send R1 result to R2 neighbor (streaming)
            sender.setup_round(
                {WriterCT::cb_r1_result_l,
                 WriterCT::cb_r1_result_ms,
                 r2_dst_mesh_id,
                 r2_dst_chip_id,
                 r2_neighbor_dst_addr,
                 r2_neighbor_sem_addr,
                 r2_fwd_slot_addr,
                 r2_fwd_sem_addr,
                 r2_base_slot_idx});
            sender.send_streaming();
            sender.finish_round();

            // Release the single MS tile from R1 result after R2 streaming send to
            // preserve BRISC/TRISC synchronization semantics from post_sdpa.
            cb_pop_front(WriterCT::cb_r1_result_ms, 1);
            noc_async_full_barrier();

            // SCATTER PHASE: Distribute output rows to destination cores
            if constexpr (WriterCT::scatter_num_rows > 0) {
                const uint32_t scatter_dest_l1_addr = get_arg_val<uint32_t>(arg_idx++);
                uint32_t scatter_dest_noc_x[WriterCT::scatter_num_rows];
                uint32_t scatter_dest_noc_y[WriterCT::scatter_num_rows];
                for (uint32_t i = 0; i < WriterCT::scatter_num_rows; i++) {
                    scatter_dest_noc_x[i] = get_arg_val<uint32_t>(arg_idx++);
                    scatter_dest_noc_y[i] = get_arg_val<uint32_t>(arg_idx++);
                }

                cb_wait_front(WriterCT::cb_l_out, WriterCT::scatter_num_tiles);
                uint32_t src_addr = get_read_ptr(WriterCT::cb_l_out);

                constexpr uint32_t scatter_payload_bytes =
                    WriterCT::scatter_num_tiles * WriterCT::scatter_dst_tile_size;

                for (uint32_t row = 0; row < WriterCT::scatter_num_rows; row++) {
                    uint64_t dest_noc_addr =
                        get_noc_addr(scatter_dest_noc_x[row], scatter_dest_noc_y[row], scatter_dest_l1_addr);
                    noc_async_write(src_addr, dest_noc_addr, scatter_payload_bytes);
                    src_addr += scatter_payload_bytes;

                    // Signal scatter arrival on destination core (used by fused ops
                    // to synchronize downstream stages like matmul1)
                    if constexpr (WriterCT::scatter_arrival_enabled) {
                        uint64_t sem_addr = get_noc_addr(
                            scatter_dest_noc_x[row],
                            scatter_dest_noc_y[row],
                            get_semaphore(WriterCT::scatter_arrival_semaphore_id));
                        noc_semaphore_inc(sem_addr, 1);
                    }
                }
                if constexpr (WriterCT::scatter_arrival_enabled) {
                    noc_async_atomic_barrier();
                } else {
                    noc_async_write_barrier();
                }
                cb_pop_front(WriterCT::cb_l_out, WriterCT::scatter_num_tiles);
            }
        }
#endif  // COMPILE_FOR_BRISC

#if defined(COMPILE_FOR_TRISC)
        // ==================================================================
        // TRISC (Compute) - streaming SDPA tail reduction
        // ==================================================================
        void compute_impl() {
            constexpr int vector_mode = VectorMode::RC_custom;

            binary_op_init_common(ComputeCT::cb_local_l, ComputeCT::cb_local_l, ComputeCT::cb_l_out);
            exp_tile_init<EXP_APPROX_MODE, false>();

            bool local_valid = true;
            bool r1_neighbor_valid = true;
            bool r2_neighbor_valid = true;

            [[maybe_unused]] uint32_t device_idx = 0;
            [[maybe_unused]] uint32_t r1_neighbor_device_idx = 0;
            [[maybe_unused]] uint32_t r2_neighbor_device_idx = 0;
            [[maybe_unused]] uint32_t position_id = 0;

            if constexpr (ComputeCT::position_enabled) {
                size_t arg_idx = 0;
                uint32_t pos_addr = get_arg_val<uint32_t>(arg_idx++);
                device_idx = get_arg_val<uint32_t>(arg_idx++);
                r1_neighbor_device_idx = get_arg_val<uint32_t>(arg_idx++);
                r2_neighbor_device_idx = get_arg_val<uint32_t>(arg_idx++);
                uint32_t r2_neighbor_r1_neighbor_idx = get_arg_val<uint32_t>(arg_idx++);

                // Read position_id from HEIGHT_SHARDED L1 tensor
                volatile tt_l1_ptr uint32_t* pos_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pos_addr);
                position_id = pos_ptr[0];

                constexpr uint32_t chunk = ComputeCT::per_device_chunk_size;
                local_valid = (position_id >= device_idx * chunk);
                r1_neighbor_valid = (position_id >= r1_neighbor_device_idx * chunk);
                r2_neighbor_valid = (position_id >= r2_neighbor_device_idx * chunk) ||
                                    (position_id >= r2_neighbor_r1_neighbor_idx * chunk);
            }

            // ROUND 1: reduce(local, r1_neighbor) -> r1_result (unnormalized)
            sdpa_tail_streaming_conditional<
                EXP_APPROX_MODE,
                false /* no normalize - R1 doesn't normalize */,
                false /* untilize - R1 doesn't untilize */,
                ComputeCT::block_size,
                ComputeCT::scale_fp32,
                ComputeCT::num_l_chunks,
                vector_mode>(
                ComputeCT::cb_r1_neighbor_ms,
                ComputeCT::cb_local_ms,
                ComputeCT::cb_r1_result_ms,
                ComputeCT::cb_r1_neighbor_l,
                ComputeCT::cb_local_l,
                ComputeCT::cb_r1_result_l,
                r1_neighbor_valid,
                local_valid);

            // Pop R1 input MS CBs — TRISC-owned, no BRISC race.
            // cb_r1_neighbor_ms: incoming from neighbor, consumed only by TRISC
            // cb_local_ms: BRISC reads via send_all() (L1 address, no cb_wait/pop)
            // No cb_wait_front needed: NCRISC always pushes all MS CBs unconditionally
            // (even for invalid neighbors with dummy data), so tiles are guaranteed present.
            cb_pop_front(ComputeCT::cb_r1_neighbor_ms, 1);
            cb_pop_front(ComputeCT::cb_local_ms, 1);

            // ROUND 2: reduce(r1_result, r2_neighbor) -> final output (normalized L)
            bool local_r1_valid = local_valid || r1_neighbor_valid;
            bool r2_neighbor_r1_valid = r2_neighbor_valid;

            // TODO: This is because only the reduction can currently perform untilize
            static_assert(ComputeCT::final_reduction, "Final reduction must be enabled");
            sdpa_tail_streaming_conditional<
                EXP_APPROX_MODE,
                ComputeCT::final_reduction,
                true /* untilize */,
                ComputeCT::block_size,
                ComputeCT::scale_fp32,
                ComputeCT::num_l_chunks,
                vector_mode>(
                ComputeCT::cb_r2_neighbor_ms,
                ComputeCT::cb_r1_result_ms,
                ComputeCT::cb_ms_out,
                ComputeCT::cb_r2_neighbor_l,
                ComputeCT::cb_r1_result_l,
                ComputeCT::cb_l_out,
                r2_neighbor_r1_valid,
                local_r1_valid);

            // Pop R2 worker MS CB — incoming from neighbor, consumed only by TRISC.
            // Do NOT pop cb_r1_result_ms — BRISC owns that pop (writer_impl line 668)
            // after send_streaming() reads it for R2 forwarding.
            cb_pop_front(ComputeCT::cb_r2_neighbor_ms, 1);
        }
#endif  // COMPILE_FOR_TRISC
    };

};  // struct SdpaReduceWorker

}  // namespace deepseek_b1_ops
