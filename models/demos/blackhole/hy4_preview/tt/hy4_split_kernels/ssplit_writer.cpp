// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// sparse_sdpa writer: per token, drain the untilized [H, V_DIM] row-major block from compute (Sqt*vDHt
// tile-sized pages, row-major bytes inside) and write each head-row to the ROW-MAJOR [1,H,S,V_DIM] output.
// Also builds the three persistent compute-input tiles once at startup (the reader is busier): the reduce
// identity scaler, the col-identity (row-sum finalize), and the all-(-inf) mask tile. Optional per-head
// attention sinks are gathered once here too, while the reader loads Q and indices.
// Uses the object-based NoC + circular-buffer APIs.

#include <stdint.h>
#include "api/tile_constants.h"  // tt::constants::TILE_WIDTH
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp"  // generate_bcast_col_scalar
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "ssplit_gather.hpp"  // dual-NoC trid-ring gather helper (shared with the reader)

constexpr uint32_t one_bf16_packed = 0x3F803F80u;  // bf16(1.0) double-packed; generate_bcast_col_scalar uses >>16
constexpr uint16_t neg_inf_bf16 = 0xFF80u;

void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(ssplit::writer_ct_arg::H);
    constexpr uint32_t S = get_compile_time_arg_val(ssplit::writer_ct_arg::S);
    constexpr uint32_t vDHt = get_compile_time_arg_val(ssplit::writer_ct_arg::V_DHT);
    // CB ids — passed by the factory (the SparseCB enum is the single source); order must match the
    // factory's writer CB-arg block (compile args 3..6).
    constexpr uint32_t cb_out_rm = get_compile_time_arg_val(ssplit::writer_ct_arg::CB_OUT_RM);
    constexpr uint32_t cb_scale = get_compile_time_arg_val(ssplit::writer_ct_arg::CB_SCALE);
    constexpr uint32_t cb_col_identity = get_compile_time_arg_val(ssplit::writer_ct_arg::CB_COL_IDENTITY);
    constexpr uint32_t cb_neginf = get_compile_time_arg_val(ssplit::writer_ct_arg::CB_NEGINF);
    constexpr uint32_t out_elem_bytes =
        get_compile_time_arg_val(ssplit::writer_ct_arg::OUT_ELEM_BYTES);  // output tensor dtype
    constexpr bool block_cyclic = get_compile_time_arg_val(ssplit::writer_ct_arg::BLOCK_CYCLIC) != 0;
    constexpr uint32_t bc_chunk_local = get_compile_time_arg_val(ssplit::writer_ct_arg::BC_CHUNK_LOCAL);
    constexpr uint32_t bc_sp = get_compile_time_arg_val(ssplit::writer_ct_arg::BC_SP);
    constexpr uint32_t bc_shard_stride_gap = get_compile_time_arg_val(ssplit::writer_ct_arg::BC_SHARD_STRIDE_GAP);
    constexpr uint32_t bc_slab_stride_gap = get_compile_time_arg_val(ssplit::writer_ct_arg::BC_SLAB_STRIDE_GAP);
    constexpr bool scaled_kv = get_compile_time_arg_val(ssplit::writer_ct_arg::SCALED_KV) != 0;
    constexpr uint32_t k_dim = get_compile_time_arg_val(ssplit::writer_ct_arg::K_DIM);
    constexpr uint32_t kv_elem_bytes = get_compile_time_arg_val(ssplit::writer_ct_arg::KV_ELEM_BYTES);
    constexpr uint32_t cb_idx = get_compile_time_arg_val(ssplit::writer_ct_arg::CB_IDX);
    constexpr uint32_t cb_kreq = get_compile_time_arg_val(ssplit::writer_ct_arg::CB_KREQ);
    constexpr uint32_t cb_kack = get_compile_time_arg_val(ssplit::writer_ct_arg::CB_KACK);
    constexpr uint32_t packed_row_bytes = get_compile_time_arg_val(ssplit::writer_ct_arg::PACKED_ROW_BYTES);
    constexpr bool use_attention_sink = get_compile_time_arg_val(ssplit::writer_ct_arg::USE_ATTENTION_SINK) != 0;
    constexpr uint32_t cb_attention_sink = get_compile_time_arg_val(ssplit::writer_ct_arg::CB_ATTENTION_SINK);
    constexpr uint32_t cb_sink_scratch = get_compile_time_arg_val(ssplit::writer_ct_arg::CB_SINK_SCRATCH);
    constexpr auto out_args = TensorAccessorArgs<ssplit::writer_ct_arg::END, ssplit::writer_common_arg::END>();
    // kv carries a RUNTIME tensor shape (T dim in common runtime args), so its accessor spans both arg streams.
    constexpr auto kv_args =
        TensorAccessorArgs<out_args.next_compile_time_args_offset(), out_args.next_common_runtime_args_offset()>();

    const uint32_t out_addr = get_common_arg_val<uint32_t>(ssplit::writer_common_arg::OUTPUT_ADDRESS);
    const uint32_t tok_start = get_arg_val<uint32_t>(0);
    const uint32_t tok_count = 1;
    const uint32_t leader_x = get_arg_val<uint32_t>(2);  // NoC coordinates of this row's leader core (split 0)
    const uint32_t leader_y = get_arg_val<uint32_t>(3);
    constexpr uint32_t NSPLIT = get_compile_time_arg_val(ssplit::writer_ct_arg::NSPLIT);
    constexpr uint32_t ring_depth = get_compile_time_arg_val(ssplit::writer_ct_arg::RING);
    constexpr uint32_t cb_q_rm = get_compile_time_arg_val(ssplit::writer_ct_arg::CB_Q_RM);
    constexpr uint32_t cb_recv_out = get_compile_time_arg_val(ssplit::writer_ct_arg::CB_RECV_OUT);
    constexpr uint32_t cb_recv_max = get_compile_time_arg_val(ssplit::writer_ct_arg::CB_RECV_MAX);
    constexpr uint32_t cb_recv_sum = get_compile_time_arg_val(ssplit::writer_ct_arg::CB_RECV_SUM);
    constexpr uint32_t cb_out_a = get_compile_time_arg_val(ssplit::writer_ct_arg::CB_OUT_A);
    constexpr uint32_t cb_out_b = get_compile_time_arg_val(ssplit::writer_ct_arg::CB_OUT_B);
    constexpr uint32_t cb_max_a = get_compile_time_arg_val(ssplit::writer_ct_arg::CB_MAX_A);
    constexpr uint32_t cb_max_b = get_compile_time_arg_val(ssplit::writer_ct_arg::CB_MAX_B);
    constexpr uint32_t cb_sum_a = get_compile_time_arg_val(ssplit::writer_ct_arg::CB_SUM_A);
    constexpr uint32_t cb_sum_b = get_compile_time_arg_val(ssplit::writer_ct_arg::CB_SUM_B);
    constexpr uint32_t sem_id = get_compile_time_arg_val(ssplit::writer_ct_arg::SEM_ID);
    const uint32_t kv_addr =
        get_common_arg_val<uint32_t>(ssplit::writer_common_arg::KV_ADDRESS);  // dual-NoC K-half gather
    // Indexed KV cache: page offset (cache_batch_idx * T) selecting the cache's batch slot; 0 if not indexed.
    const uint32_t kv_batch_page_offset = get_common_arg_val<uint32_t>(ssplit::writer_common_arg::KV_BATCH_PAGE_OFFSET);

    constexpr uint32_t row_bytes = vDHt * tt::constants::TILE_WIDTH * out_elem_bytes;  // V_DIM bytes per head-row
    constexpr uint32_t block_tiles = (H / tt::constants::TILE_HEIGHT) * vDHt;          // Sqt*vDHt: the [H,V_DIM] block

    Noc noc;
    experimental::CB out_cb(cb_out_rm);
    const auto out = TensorAccessor(out_args, out_addr);
    // Dual-NoC K gather: the writer co-gathers half of each K chunk on ITS NoC into the shared cb_k_rm L1
    // (the reader owns the reserve/push; we fill rows [0,half) at the same write ptr). This spreads the
    // K response-data load over both NoC links. Indices live in the reader's shared cb_idx
    // scratch; the kv accessor + row size mirror the reader.
    constexpr uint32_t k_row_bytes = k_dim * kv_elem_bytes;
    constexpr uint32_t latent_row_bytes = vDHt * tt::constants::TILE_WIDTH;
    constexpr uint32_t scale_blocks = ssplit::scale_block_count(latent_row_bytes);
    const auto kv = TensorAccessor(kv_args, kv_addr);
    constexpr auto sink_args_q =
        TensorAccessorArgs<kv_args.next_compile_time_args_offset(), kv_args.next_common_runtime_args_offset()>();
    constexpr auto q_args = TensorAccessorArgs<
        sink_args_q.next_compile_time_args_offset(),
        sink_args_q.next_common_runtime_args_offset()>();
    constexpr bool emit_stats = get_compile_time_arg_val(ssplit::writer_ct_arg::EMIT_STATS) != 0;
    // EMIT_STATS == 2: the stats ride in the output rows: head-row page = [V_DIM bf16 o | raw max | sum | 30 zeros]
    // (page V_DIM+32 elements, 64 B tail)
    // OWNER_S > 0 (emit_cols only): head-row pages are written in the cross-column owner layout [NC, 1, 8*S_users,
    // V_DIM+32]: query row tok = j*S_users + s is the OWNER column j's block of rows (h, s); only the first 8 heads are
    // written (the rest are zero pad). STAT_SCALE_BITS: float bits of the softmax scale; the tail max is stored
    // premultiplied (scaled-logit domain), col V_DIM+2 = 1.0 (so the merge's weighted sum also yields the denominator).
    constexpr uint32_t owner_s = get_compile_time_arg_val(ssplit::writer_ct_arg::OWNER_S);
    constexpr uint32_t stat_scale_bits = get_compile_time_arg_val(ssplit::writer_ct_arg::STAT_SCALE_BITS);
    // COMPACT (emit_stats 3 + owner_s only): the owner-layout TILE output [NC, TR = ceil(S_users / 4), 32, V_DIM + 64]
    // packs the 8 real head rows of up to 4 users into the 32 rows of one tile (user s -> row band 8*(s%4) of tile row
    // s/4): the cross-column all_to_all moves 4x fewer bytes. Every core writes ONLY its own 8-row band (two 256 B
    // face-row runs per column tile), the other bands belong to other cores.
    constexpr bool compact = get_compile_time_arg_val(ssplit::writer_ct_arg::COMPACT) != 0;
    constexpr uint32_t out_heads = owner_s ? 8 : H;
    constexpr bool emit_tile = get_compile_time_arg_val(ssplit::writer_ct_arg::EMIT_STATS) ==
                               3;  // owner-layout TILE output [NC, S_users, 8(32), V_DIM+32]: tile id = tok*(vDHt+3) +
                                   // ct, tail tiles [ms | l | 1.0] at ct = vDHt..vDHt+2
    constexpr bool emit_cols = get_compile_time_arg_val(ssplit::writer_ct_arg::EMIT_STATS) >= 2;
    constexpr uint32_t cb_m_out = get_compile_time_arg_val(ssplit::writer_ct_arg::CB_M_OUT);
    constexpr uint32_t cb_l_out = get_compile_time_arg_val(ssplit::writer_ct_arg::CB_L_OUT);
    constexpr uint32_t cb_stat_scr = get_compile_time_arg_val(ssplit::writer_ct_arg::CB_STAT_SCR);
    constexpr auto stats_args =
        TensorAccessorArgs<q_args.next_compile_time_args_offset(), q_args.next_common_runtime_args_offset()>();
    experimental::CB idx_cb(cb_idx), kreq_cb(cb_kreq), kack_cb(cb_kack);
    volatile tt_l1_ptr uint32_t* idx_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_cb.get_write_ptr());

    DeviceZoneScopedN("W_ALL");
    // --- persistent compute-input tiles
    // Reduce identity scaler (value 1.0; the softmax scale is applied in compute's exp).
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_scale,
        ckernel::PoolType::MAX,
        ckernel::ReduceDim::REDUCE_ROW,
        /*reduce_factor=*/1>();

    // Col-identity (column 0 = 1.0): compute matmul-reduces the partial row-sum against it to finalize the
    // within-tile reduction in normalize_row_streaming.
    generate_bcast_col_scalar(CircularBuffer(cb_col_identity), one_bf16_packed);

    // All-(-inf) tile (row 0; compute bcast-adds it to fully-masked key tiles). Only row 0 matters for the
    // row-broadcast add: the left and right half-rows live in face 0 and face 1 respectively.
    {
        experimental::CB neginf_cb(cb_neginf);
        neginf_cb.reserve_back(1);
        volatile tt_l1_ptr uint16_t* p = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(neginf_cb.get_write_ptr());
        for (uint32_t c = 0; c < tt::constants::FACE_WIDTH; ++c) {
            p[c] = neg_inf_bf16;
            p[tt::constants::FACE_HW + c] = neg_inf_bf16;
        }
        neginf_cb.push_back(1);
    }

    if constexpr (use_attention_sink) {
        if (tok_count > 0) {
            constexpr auto sink_args = TensorAccessorArgs<
                kv_args.next_compile_time_args_offset(),
                kv_args.next_common_runtime_args_offset()>();
            const auto sink = TensorAccessor(
                sink_args, get_common_arg_val<uint32_t>(ssplit::writer_common_arg::ATTENTION_SINK_ADDRESS));
            experimental::CB sink_cb(cb_attention_sink), scratch(cb_sink_scratch);
            constexpr uint32_t sink_bytes = H * sizeof(uint16_t);
            constexpr uint32_t tile_bytes = tt::constants::TILE_HW * sizeof(uint16_t);
            constexpr uint32_t faces_per_row = tt::constants::TILE_WIDTH / tt::constants::FACE_WIDTH;
            static_assert(H % tt::constants::TILE_HEIGHT == 0);
            static_assert(sink_bytes % NOC_DRAM_READ_ALIGNMENT_BYTES == 0);
            constexpr uint32_t sink_tiles = H / tt::constants::TILE_HEIGHT;
            sink_cb.reserve_back(sink_tiles);
            scratch.reserve_back(1);
            noc.async_write_zeros(sink_cb, sink_tiles * tile_bytes);
            // [1,1,1,H] BF16 stores all heads in one row. Read once, then pack the scalars
            // into the first column of the compute tiles (one head per row).
            noc.async_read(sink, scratch, sink_bytes, {.page_id = 0}, {.offset_bytes = 0});
            noc.async_read_barrier();
            noc.write_zeros_l1_barrier();
            auto* dst = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(sink_cb.get_write_ptr());
            auto* src = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch.get_write_ptr());
            for (uint32_t h = 0; h < H; ++h) {
                const uint32_t row = h % tt::constants::TILE_HEIGHT;
                const uint32_t offset = (h / tt::constants::TILE_HEIGHT) * tt::constants::TILE_HW +
                                        (row / tt::constants::FACE_HEIGHT) * faces_per_row * tt::constants::FACE_HW +
                                        (row % tt::constants::FACE_HEIGHT) * tt::constants::FACE_WIDTH;
                dst[offset] = src[h];
            }
            sink_cb.push_back(sink_tiles);  // reused by every token on this core
        }
    }

    auto out_page = [&](uint32_t h, uint32_t tok) -> uint32_t {
        if constexpr (owner_s != 0) {
            return (tok / owner_s) * (8 * owner_s) + h * owner_s + (tok % owner_s);
        } else {
            return h * S + tok;
        }
    };
    // fused partial exchange (fuse-sparse-merge A2b): rt args 4..: xflag (0 off, 1 remote owner: write into the comm cores' staging, 2 own owner: write into the own landing slot), TPC, NSPLIT, PKT, gsem addr, local sem id,
    // then per merge-core run g: (x, y, run base address)
    const uint32_t xflag = get_arg_val<uint32_t>(4);
    const uint32_t xtpc = get_arg_val<uint32_t>(5), xn = get_arg_val<uint32_t>(6), xpkt = get_arg_val<uint32_t>(7);
    const uint32_t xgsem = get_arg_val<uint32_t>(8), xsem = get_arg_val<uint32_t>(9);
    auto xaddr = [&](uint32_t g, uint32_t tile, bool stat) -> uint64_t {
        const uint32_t ex = get_arg_val<uint32_t>(10 + 3 * g), ey = get_arg_val<uint32_t>(11 + 3 * g), eb = get_arg_val<uint32_t>(12 + 3 * g);
        uint32_t a;
        if (xflag == 1) {
            a = stat ? eb + (xtpc / 2) * xpkt + tile * 2048 : eb + (tile / 2) * xpkt + (tile % 2) * 2048;
        } else {
            a = stat ? eb + (xtpc + tile) * 2048 : eb + tile * 2048;
        }
        return get_noc_addr(ex, ey, a);
    };
    auto xchg_write_o = [&](uint32_t src) {
        for (uint32_t ct = 0; ct < vDHt; ++ct) {
            noc_async_write(src + ct * 2048, xaddr(ct / xtpc, ct % xtpc, false), 2048);
        }
    };
    auto xchg_write_stats = [&](uint32_t src) {
        for (uint32_t g = 0; g < xn; ++g) {
            for (uint32_t t = 0; t < 2; ++t) {
                noc_async_write(src + t * 2048, xaddr(g, t, true), 2048);
            }
        }
    };
    for (uint32_t tok = tok_start; tok < tok_start + tok_count; ++tok) {
        // K-gather phase: process this token's active chunks (reader signals per chunk; is_last ends the token).
        // Gather rows [0,split) on the writer's NoC into the reader-reserved destination, then ack.
        // header page from the reader: my chunk count, partial states to merge (leader), my split index
        kreq_cb.wait_front(1);
        uint32_t my_cnt, n_remote, split_g, row_empty;
        {
            volatile tt_l1_ptr uint32_t* hdr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kreq_cb.get_read_ptr());
            my_cnt = hdr[ssplit::gather_request::BASE];
            n_remote = hdr[ssplit::gather_request::SPLIT];
            split_g = hdr[ssplit::gather_request::IS_LAST];
            row_empty = hdr[ssplit::gather_request::DST_L1];
        }
        kreq_cb.pop_front(1);
        DeviceZoneScopedN("W_TOKEN");
        const auto q = TensorAccessor(q_args, get_common_arg_val<uint32_t>(ssplit::writer_common_arg::Q_ADDRESS));
        experimental::CB q_cb(cb_q_rm);
        constexpr uint32_t q_rows = get_compile_time_arg_val(ssplit::writer_ct_arg::Q_ROWS);
        constexpr bool q_in_writer = get_compile_time_arg_val(ssplit::writer_ct_arg::QMODE_W) == 0;
        bool q_pushed = !q_in_writer;
        if constexpr (q_in_writer) {
            q_cb.reserve_back(H);
            for (uint32_t h = 0; h < q_rows; ++h) {
                noc.async_read(q, q_cb, 1024, {.page_id = h * S + tok}, {.offset_bytes = h * 1024});
            }
            if (q_rows < H) {
                noc.async_write_zeros(q_cb, (H - q_rows) * 1024, {.offset_bytes = q_rows * 1024});
            }
        }
        bool last = (my_cnt == 0);
        while (!last) {
            kreq_cb.wait_front(1);
            uint32_t base, split, dst_l1, scale_dst_l1 = 0;
            {
                volatile tt_l1_ptr uint32_t* request =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kreq_cb.get_read_ptr());
                base = request[ssplit::gather_request::BASE];
                split = request[ssplit::gather_request::SPLIT];
                last = request[ssplit::gather_request::IS_LAST] != 0;
                dst_l1 = request[ssplit::gather_request::DST_L1];
                if constexpr (scaled_kv) {
                    scale_dst_l1 = request[ssplit::gather_request::SCALE_DST_L1];
                }
            }
            kreq_cb.pop_front(1);
            if constexpr (scaled_kv) {
                ssplit::trid_ring_gather<
                    block_cyclic,
                    bc_chunk_local,
                    bc_sp,
                    bc_shard_stride_gap,
                    bc_slab_stride_gap,
                    ssplit::SCALED_K_TRID_RING>(
                    noc, kv, dst_l1, idx_ptr, base, 0, split, packed_row_bytes, kv_batch_page_offset);
                ssplit::scatter_packed_scales<scale_blocks>(
                    dst_l1, scale_dst_l1, 0, split, packed_row_bytes, latent_row_bytes);
            } else {
                ssplit::trid_ring_gather<
                    block_cyclic,
                    bc_chunk_local,
                    bc_sp,
                    bc_shard_stride_gap,
                    bc_slab_stride_gap,
                    ring_depth>(noc, kv, dst_l1, idx_ptr, base, 0, split, k_row_bytes, kv_batch_page_offset);
            }
            kack_cb.reserve_back(1);
            kack_cb.push_back(1);
            if (!q_pushed) {  // Q before the first chunk is consumed (compute tilizes Q first), not after the last
                              // request
                DeviceZoneScopedN("W_QWAIT");
                noc.async_read_barrier();
                noc.write_zeros_l1_barrier();
                q_cb.push_back(H);
                q_pushed = true;
            }
        }
        if (!q_pushed) {
            noc.async_read_barrier();
            noc.write_zeros_l1_barrier();
            q_cb.push_back(H);
        }
        constexpr uint32_t TILE_B = 2048;  // bf16 tile
        if (split_g != 0) {
            DeviceZoneScopedN("W_SEND");
            // sender: my partial state (out, max, sum) sits in the ping-pong CBs: chunk 1 lands in b, so an odd count
            // ends in b. The MAX tile is sent as soon as it exists (single chunk: right after the QK row max, ~4 us
            // before the PV output): the leader forms the global max and the corrections while the outputs are still in
            // flight.
            if (my_cnt > 0) {
                const bool in_b = (my_cnt & 1) != 0;
                experimental::CB s_out(in_b ? cb_out_b : cb_out_a), s_max(in_b ? cb_max_b : cb_max_a),
                    s_sum(in_b ? cb_sum_b : cb_sum_a);
                constexpr uint32_t ST = H / tt::constants::TILE_HEIGHT;  // state rows (tiles) per max / sum
                experimental::CB r_out(cb_recv_out), r_max(cb_recv_max), r_sum(cb_recv_sum);
                const uint32_t slot = split_g - 1;
                const uint32_t o_addr = r_out.get_write_ptr() + slot * block_tiles * TILE_B;
                const uint32_t m_addr = r_max.get_write_ptr() + slot * ST * TILE_B;
                const uint32_t l_addr = r_sum.get_write_ptr() + slot * ST * TILE_B;
                s_max.wait_front(ST);
                noc_async_write(s_max.get_read_ptr(), get_noc_addr(leader_x, leader_y, m_addr), ST * TILE_B);
                noc_async_write_barrier();
                noc_semaphore_inc(get_noc_addr(leader_x, leader_y, get_semaphore(sem_id + slot)), 1);
                s_out.wait_front(block_tiles);
                s_sum.wait_front(ST);
                noc_async_write(s_sum.get_read_ptr(), get_noc_addr(leader_x, leader_y, l_addr), ST * TILE_B);
                noc_async_write(s_out.get_read_ptr(), get_noc_addr(leader_x, leader_y, o_addr), block_tiles * TILE_B);
                noc_async_write_barrier();
                noc_semaphore_inc(get_noc_addr(leader_x, leader_y, get_semaphore(sem_id + (NSPLIT - 1) + slot)), 1);
                noc_async_atomic_barrier();
            }
            continue;
        }
        if (n_remote > 0) {
            DeviceZoneScopedN("W_RECV");
            constexpr uint32_t ST = H / tt::constants::TILE_HEIGHT;
            experimental::CB r_out(cb_recv_out), r_max(cb_recv_max), r_sum(cb_recv_sum);
            for (uint32_t q = 0; q < n_remote;
                 ++q) {  // the max tiles first (early), compute forms M and the corrections from them
                volatile tt_l1_ptr uint32_t* sem =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id + q));
                noc_semaphore_wait(sem, 1);
                noc_semaphore_set(sem, 0);
                r_max.reserve_back(ST);
                r_max.push_back(ST);
            }
            for (uint32_t q = 0; q < n_remote; ++q) {  // then each (out, sum) state as it lands
                volatile tt_l1_ptr uint32_t* sem =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id + (NSPLIT - 1) + q));
                noc_semaphore_wait(sem, 1);
                noc_semaphore_set(sem, 0);
                r_out.reserve_back(block_tiles);
                r_sum.reserve_back(ST);
                r_out.push_back(block_tiles);
                r_sum.push_back(ST);
            }
        }
        {
            DeviceZoneScopedN("W_OUTWAIT");
            out_cb.wait_front(block_tiles);
        }
        if constexpr (emit_stats) {
            // Per-head partial-softmax statistics for the cross-column merge, one 128 B page per query row:
            // bf16 [0,32) = running row max (RAW score units: multiply by the softmax scale on the host/merge, exact),
            // bf16 [32,64) = row sum of exp((s - max) * scale). Empty column (no selected key): max = -inf, sum = 0.
            experimental::CB m_cb(cb_m_out), l_cb(cb_l_out), scr(cb_stat_scr);
            m_cb.wait_front(1);
            l_cb.wait_front(1);
            scr.reserve_back(1);
            volatile tt_l1_ptr uint16_t* sp = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr.get_write_ptr());
            volatile tt_l1_ptr uint16_t* mt = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(m_cb.get_read_ptr());
            volatile tt_l1_ptr uint16_t* lt = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l_cb.get_read_ptr());
            for (uint32_t h = 0; h < H; ++h) {
                const uint32_t off = ssplit::tile_col0_offset(h);
                sp[h] = row_empty ? neg_inf_bf16 : mt[off];
                sp[H + h] = row_empty ? (uint16_t)0 : lt[off];
            }
            const auto stats =
                TensorAccessor(stats_args, get_common_arg_val<uint32_t>(ssplit::writer_common_arg::STATS_ADDRESS));
            if constexpr (emit_cols) {
                // scr is a 2048 B scratch page: rebuild it as H x 64 B tails [m, l, 0 ...]; one 64 B write per head row
                // (page h*S+tok, byte offset row_bytes)
                volatile tt_l1_ptr uint32_t* sp32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr.get_write_ptr());
                union FU {
                    uint32_t u;
                    float f;
                };
                FU scu;
                scu.u = stat_scale_bits;
                const float sc = scu.f;
                if constexpr (emit_tile) {
                    for (uint32_t k = 0; k < 2 * 512; ++k) {
                        sp32[k] = 0;
                    }
                }
                for (uint32_t h = 0; h < out_heads; ++h) {
                    uint16_t mv = row_empty ? neg_inf_bf16 : mt[ssplit::tile_col0_offset(h)];
                    const uint16_t lv = row_empty ? (uint16_t)0 : lt[ssplit::tile_col0_offset(h)];
                    if (!row_empty) {
                        FU x;
                        x.u = (uint32_t)mv << 16;
                        x.f = x.f * sc;
                        const uint32_t u = x.u;
                        mv = (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
                    }
                    if constexpr (emit_tile) {
                        // ms tile (tile 0 of the tail) col 0 / l tile (tile 1) col 0 of head row h (h < 8: face 0 row h
                        // = word h*8)
                        sp32[h * 8] = (uint32_t)mv;
                        sp32[512 + h * 8] = (uint32_t)lv;
                    } else {
                        sp32[h * 16] = (uint32_t)mv | ((uint32_t)lv << 16);
                        sp32[h * 16 + 1] = 0x3F80u;
                        for (uint32_t k = 2; k < 16; ++k) {
                            sp32[h * 16 + k] = 0;
                        }
                    }
                }
                if constexpr (emit_tile) {
                    if constexpr (compact) {
                        const uint32_t cj = tok / owner_s, cs = tok % owner_s;
                        constexpr uint32_t cTR = (owner_s + 3) / 4;
                        const uint32_t tile_base = (cj * cTR + (cs >> 2)) * (vDHt + 2);
                        const uint32_t band_off = ((cs & 3u) < 2 ? 0u : 1024u) + (cs & 1u) * 256u;
                        for (uint32_t t = 0; t < 2; ++t) {
                            noc.async_write(
                                scr,
                                stats,
                                256,
                                {.offset_bytes = t * 2048},
                                {.page_id = tile_base + vDHt + t, .offset_bytes = band_off});
                        }
                    } else if (xflag == 0) {
                        for (uint32_t t = 0; t < 2; ++t) {
                            noc.async_write(
                                scr, stats, 2048, {.offset_bytes = t * 2048}, {.page_id = tok * (vDHt + 2) + vDHt + t});
                        }
                    } else {
                        xchg_write_stats(scr.get_write_ptr());
                    }
                } else {
                    for (uint32_t h = 0; h < out_heads; ++h) {
                        noc.async_write(
                            scr,
                            stats,
                            64,
                            {.offset_bytes = h * 64},
                            {.page_id = out_page(h, tok), .offset_bytes = row_bytes});
                    }
                }
            } else {
                noc.async_write(scr, stats, 2 * H * sizeof(uint16_t), {.offset_bytes = 0}, {.page_id = tok});
                noc.async_write_barrier();
            }
            m_cb.pop_front(1);
            l_cb.pop_front(1);
        }
        if constexpr (emit_tile) {
            if constexpr (compact) {
                const uint32_t cj = tok / owner_s, cs = tok % owner_s;
                constexpr uint32_t cTR = (owner_s + 3) / 4;
                const uint32_t tile_base = (cj * cTR + (cs >> 2)) * (vDHt + 2);
                const uint32_t band_off = ((cs & 3u) < 2 ? 0u : 1024u) + (cs & 1u) * 256u;
                for (uint32_t ct = 0; ct < vDHt; ++ct) {
                    for (uint32_t f = 0; f < 2;
                         ++f) {  // face 0 / 1 rows 0..7 of the source tile -> this user's band of the shared tile
                        noc.async_write(
                            out_cb,
                            out,
                            256,
                            {.offset_bytes = ct * 2048 + f * 512},
                            {.page_id = tile_base + ct, .offset_bytes = band_off + f * 512});
                    }
                }
            } else if (xflag == 0) {
                for (uint32_t ct = 0; ct < vDHt; ++ct) {
                    noc.async_write(out_cb, out, 2048, {.offset_bytes = ct * 2048}, {.page_id = tok * (vDHt + 2) + ct});
                }
            } else {
                xchg_write_o(out_cb.get_read_ptr());
            }
        } else {
            for (uint32_t h = 0; h < out_heads; ++h) {
                // row h (head h) at CB read-ptr byte offset h*row_bytes; DRAM page = h*S + tok (owner layout:
                // out_page).
                noc.async_write(out_cb, out, row_bytes, {.offset_bytes = h * row_bytes}, {.page_id = out_page(h, tok)});
            }
        }
        {
            DeviceZoneScopedN("W_OUTWR");
            noc.async_write_barrier();
        }
        if (xflag == 3) {
            // B47a XPUSH (compact owner layout): this token's bands are in the DRAM output; tell the comm cores of
            // owner j = tok / owner_s (rt 5 = n comm cores, 6 = global semaphore address of owner j, 7.. = (x, y))
            for (uint32_t g = 0; g < xtpc; ++g) {
                noc_semaphore_inc(
                    get_noc_addr(get_arg_val<uint32_t>(7 + 2 * g), get_arg_val<uint32_t>(8 + 2 * g), xn), 1);
            }
            noc_async_atomic_barrier();
        } else if (xflag != 0) {  // fused exchange: the runs are in the comm cores' staging / the own landing slot;
                                  // signal them
            for (uint32_t g = 0; g < xn; ++g) {
                const uint32_t ex = get_arg_val<uint32_t>(10 + 3 * g), ey = get_arg_val<uint32_t>(11 + 3 * g);
                noc_semaphore_inc(get_noc_addr(ex, ey, xflag == 1 ? get_semaphore(xsem) : xgsem), 1);
            }
            noc_async_atomic_barrier();
        }
        out_cb.pop_front(block_tiles);
    }
}
