// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// SigLIP MLP FC2 matmul kernel — 2D K×N parallel + 3-way reduce-scatter.
//
// Shape: M=256, K=4320, N=1152. 27 cores in a 9×3 logical grid.
//   col=ng ∈ [0,9), row=kg ∈ [0,3).
//   N=1152 → 9 N-groups × 4 N-tiles each.
//   K=4320 → 3 K-groups × 45 K-tiles each, split 9-way (5 K-tiles per core).
//
// Dataflow per core (ng, kg):
//   BRISC Phase A: 8 parallel noc_async_read of peer activations within K-row
//     into a single-page act_gather_cb (720 KB, chunk-major: 9 × 40 tiles each).
//     One noc_async_read_barrier at the end.
//   TRISC Phase B: standard matmul_block on (M=8, K=45, N=4) → partial(M, 4).
//   BRISC Phase C: noc_async_write_multicast of partial to 2 N-col peers at
//     slot[my_kg] in their recv_cb + semaphore inc.
//   TRISC Phase D: 3-way reduce (own + 2 received) via binary_dest_reuse_tiles.

#include "../../../../demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#include "../../../../demos/deepseek_v3_b1/unified_kernels/kernel_utils.hpp"
#include "api/debug/dprint.h"

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#endif

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#endif

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    constexpr uint32_t act_local_cb = get_named_compile_time_arg_val("act_local_cb");
    constexpr uint32_t weight_cb = get_named_compile_time_arg_val("weight_cb");
    constexpr uint32_t out_cb = get_named_compile_time_arg_val("out_cb");
    constexpr uint32_t act_local_tiles = get_named_compile_time_arg_val("act_local_tiles");
    constexpr uint32_t weight_tiles = get_named_compile_time_arg_val("weight_tiles");
    constexpr uint32_t out_tiles = get_named_compile_time_arg_val("out_tiles");

    unified_kernels::setup_sharded_buffer(act_local_cb, act_local_tiles);
    unified_kernels::setup_sharded_buffer(weight_cb, weight_tiles);
    unified_kernels::setup_sharded_buffer(out_cb, out_tiles);
    DPRINT << "NC: setup done." << ENDL();

#elif defined(COMPILE_FOR_BRISC)
    constexpr uint32_t act_local_cb = get_named_compile_time_arg_val("act_local_cb");
    constexpr uint32_t act_gather_cb = get_named_compile_time_arg_val("act_gather_cb");
    constexpr uint32_t partial_cb = get_named_compile_time_arg_val("partial_cb");
    constexpr uint32_t recv_cb = get_named_compile_time_arg_val("recv_cb");
    constexpr uint32_t out_cb = get_named_compile_time_arg_val("out_cb");
    constexpr uint32_t act_chunk_bytes = get_named_compile_time_arg_val("act_chunk_bytes");
    constexpr uint32_t partial_bytes = get_named_compile_time_arg_val("partial_bytes");
    constexpr uint32_t act_local_tiles = get_named_compile_time_arg_val("act_local_tiles");
    constexpr uint32_t act_gather_tiles = get_named_compile_time_arg_val("act_gather_tiles");
    constexpr uint32_t partial_tiles = get_named_compile_time_arg_val("partial_tiles");

    // Runtime args
    uint32_t rt = 0;
    uint32_t my_col = get_arg_val<uint32_t>(rt++);
    uint32_t my_kg = get_arg_val<uint32_t>(rt++);
    uint32_t kg_peer_x[8], kg_peer_y[8], kg_peer_slot[8];
    for (uint32_t i = 0; i < 8; ++i) {
        kg_peer_x[i] = get_arg_val<uint32_t>(rt++);
        kg_peer_y[i] = get_arg_val<uint32_t>(rt++);
        kg_peer_slot[i] = get_arg_val<uint32_t>(rt++);
    }
    uint32_t nc_peer_x[2], nc_peer_y[2];
    for (uint32_t i = 0; i < 2; ++i) {
        nc_peer_x[i] = get_arg_val<uint32_t>(rt++);
        nc_peer_y[i] = get_arg_val<uint32_t>(rt++);
    }
    uint32_t recv_sem_addr = get_arg_val<uint32_t>(rt++);

    // ============================================================
    // Phase A: gather 8 peer acts + own slice into act_gather_cb.
    // Issue all reads in parallel (alternating NOC0/NOC1 for bandwidth),
    // single barrier at end.
    // ============================================================
    cb_wait_front(act_local_cb, act_local_tiles);
    cb_reserve_back(act_gather_cb, act_gather_tiles);

    uint32_t act_local_addr_local = get_read_ptr(act_local_cb);
    uint32_t gather_base = get_write_ptr(act_gather_cb);

    // Own slice (self-NoC, NOC0).
    {
        uint64_t self_noc = get_noc_addr(my_x[0], my_y[0], act_local_addr_local);
        noc_async_read(self_noc, gather_base + my_col * act_chunk_bytes, act_chunk_bytes);
    }
    // 8 peer reads, alternating NOC0/NOC1 for higher aggregate bandwidth.
    for (uint32_t i = 0; i < 8; ++i) {
        uint8_t noc_id = i & 1;  // 0,1,0,1,...
        uint64_t peer_noc = get_noc_addr(kg_peer_x[i], kg_peer_y[i], act_local_addr_local, noc_id);
        noc_async_read(peer_noc, gather_base + kg_peer_slot[i] * act_chunk_bytes, act_chunk_bytes, noc_id);
    }
    noc_async_read_barrier(0);
    noc_async_read_barrier(1);
    cb_push_back(act_gather_cb, act_gather_tiles);
    DPRINT << "BR: phase A done." << ENDL();

    // ============================================================
    // Phase C: multicast partial to 2 N-col peers, then wait for 2 receives.
    // Both peers receive the SAME partial, written to slot[my_kg] in their recv_cb.
    // Use noc_async_write_multicast to fan out in one op when both peers share
    // a NoC sub-row/col; otherwise fall back to 2 unicasts.
    // ============================================================
    cb_wait_front(partial_cb, partial_tiles);
    uint32_t partial_addr = get_read_ptr(partial_cb);

    uint32_t recv_base = get_write_ptr(recv_cb);
    uint32_t recv_slot = recv_base + my_kg * partial_bytes;

    // 2 unicast writes split across NOC0/NOC1 to use both NoC links.
    for (uint32_t i = 0; i < 2; ++i) {
        uint8_t noc_id = i & 1;
        uint64_t dst_data = get_noc_addr(nc_peer_x[i], nc_peer_y[i], recv_slot, noc_id);
        noc_async_write(partial_addr, dst_data, partial_bytes, noc_id);
    }
    noc_async_write_barrier(0);
    noc_async_write_barrier(1);

    for (uint32_t i = 0; i < 2; ++i) {
        uint64_t dst_sem = get_noc_addr(nc_peer_x[i], nc_peer_y[i], recv_sem_addr);
        noc_semaphore_inc(dst_sem, 1);
    }
    noc_async_atomic_barrier();

    volatile tt_l1_ptr uint32_t* recv_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(recv_sem_addr);
    noc_semaphore_wait_min(recv_sem_ptr, 2);
    noc_semaphore_set(recv_sem_ptr, 0);

    cb_reserve_back(recv_cb, 2 * partial_tiles);
    cb_push_back(recv_cb, 2 * partial_tiles);
    DPRINT << "BR: phase C done." << ENDL();

#elif defined(COMPILE_FOR_TRISC)
    constexpr uint32_t act_gather_cb = get_named_compile_time_arg_val("act_gather_cb");
    constexpr uint32_t weight_cb = get_named_compile_time_arg_val("weight_cb");
    constexpr uint32_t partial_cb = get_named_compile_time_arg_val("partial_cb");
    constexpr uint32_t recv_cb = get_named_compile_time_arg_val("recv_cb");
    constexpr uint32_t out_cb = get_named_compile_time_arg_val("out_cb");
    constexpr uint32_t M_TILES = get_named_compile_time_arg_val("m_tiles");
    constexpr uint32_t K_TILES = get_named_compile_time_arg_val("k_tiles");
    constexpr uint32_t K_PER_CHUNK = get_named_compile_time_arg_val("k_per_chunk");
    constexpr uint32_t N_PER_CORE = get_named_compile_time_arg_val("n_per_core");
    constexpr uint32_t ACT_GATHER_TILES = get_named_compile_time_arg_val("act_gather_tiles");
    constexpr uint32_t WEIGHT_TILES = get_named_compile_time_arg_val("weight_tiles");
    constexpr uint32_t PARTIAL_TILES = get_named_compile_time_arg_val("partial_tiles");
    constexpr uint32_t my_kg = get_named_compile_time_arg_val("my_kg");

    constexpr uint32_t SUBBLOCK_H = 1;
    constexpr uint32_t SUBBLOCK_W = N_PER_CORE;
    constexpr uint32_t CHUNK_STRIDE = M_TILES * K_PER_CHUNK;

    // ============================================================
    // Phase B: matmul on the gathered (M=8, K=45, N=4) activation.
    // ============================================================
    reconfig_data_format(weight_cb, act_gather_cb);
    pack_reconfig_data_format(partial_cb);
    mm_init(act_gather_cb, weight_cb, partial_cb);

    cb_wait_front(act_gather_cb, ACT_GATHER_TILES);
    cb_wait_front(weight_cb, WEIGHT_TILES);
    cb_reserve_back(partial_cb, PARTIAL_TILES);

    DPRINT << "TR: phase B start." << ENDL();

    for (uint32_t m_start = 0; m_start < M_TILES; m_start += SUBBLOCK_H) {
        mm_block_init_short(
            act_gather_cb,
            weight_cb,
            /*transpose=*/0,
            /*ct_dim=*/SUBBLOCK_W,
            /*rt_dim=*/SUBBLOCK_H,
            /*kt_dim=*/K_TILES);
        tile_regs_acquire();
        uint32_t in1_index = 0;
        for (uint32_t k = 0; k < K_TILES; ++k) {
            uint32_t chunk = k / K_PER_CHUNK;
            uint32_t k_local = k - chunk * K_PER_CHUNK;
            uint32_t in0_index = chunk * CHUNK_STRIDE + m_start * K_PER_CHUNK + k_local;
            matmul_block(
                act_gather_cb,
                weight_cb,
                in0_index,
                in1_index,
                /*idst=*/0,
                /*transpose=*/false,
                /*ct_dim=*/SUBBLOCK_W,
                /*rt_dim=*/SUBBLOCK_H,
                /*kt_dim=*/K_TILES);
            in1_index += N_PER_CORE;
        }
        tile_regs_commit();

        tile_regs_wait();
        for (uint32_t h = 0; h < SUBBLOCK_H; ++h) {
            for (uint32_t w = 0; w < SUBBLOCK_W; ++w) {
                uint32_t dst_idx = h * SUBBLOCK_W + w;
                uint32_t out_tile_id = (m_start + h) * N_PER_CORE + w;
                pack_tile<true>(dst_idx, partial_cb, out_tile_id);
            }
        }
        tile_regs_release();
    }
    cb_push_back(partial_cb, PARTIAL_TILES);
    cb_pop_front(act_gather_cb, ACT_GATHER_TILES);
    DPRINT << "TR: phase B done." << ENDL();

    // ============================================================
    // Phase D: 3-way reduce (own + 2 received) → out_cb.
    // ============================================================
    constexpr uint32_t slot_A = (my_kg == 0) ? 1 : 0;
    constexpr uint32_t slot_B = (my_kg == 2) ? 1 : 2;

    cb_wait_front(recv_cb, 2 * PARTIAL_TILES);
    cb_wait_front(partial_cb, PARTIAL_TILES);
    cb_reserve_back(out_cb, PARTIAL_TILES);

    binary_op_init_common(partial_cb, recv_cb, out_cb);
    reconfig_data_format(partial_cb, recv_cb);
    pack_reconfig_data_format(out_cb);

    constexpr uint32_t BATCH = 8;
    static_assert(PARTIAL_TILES % BATCH == 0);

    for (uint32_t b = 0; b < PARTIAL_TILES; b += BATCH) {
        copy_tile_to_dst_init_short(partial_cb);
        tile_regs_acquire();
        for (uint32_t i = 0; i < BATCH; ++i) {
            copy_tile(partial_cb, b + i, i);
        }
        binary_dest_reuse_tiles_init<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(recv_cb);
        for (uint32_t i = 0; i < BATCH; ++i) {
            binary_dest_reuse_tiles<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                recv_cb, slot_A * PARTIAL_TILES + b + i, i);
        }
        for (uint32_t i = 0; i < BATCH; ++i) {
            binary_dest_reuse_tiles<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                recv_cb, slot_B * PARTIAL_TILES + b + i, i);
        }
        tile_regs_commit();

        tile_regs_wait();
        for (uint32_t i = 0; i < BATCH; ++i) {
            pack_tile(i, out_cb, b + i);
        }
        tile_regs_release();
    }
    cb_push_back(out_cb, PARTIAL_TILES);
    cb_pop_front(partial_cb, PARTIAL_TILES);
    cb_pop_front(recv_cb, 2 * PARTIAL_TILES);
    DPRINT << "TR: phase D done." << ENDL();
#endif
}
