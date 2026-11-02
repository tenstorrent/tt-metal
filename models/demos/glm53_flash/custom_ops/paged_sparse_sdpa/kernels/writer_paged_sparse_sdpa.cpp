// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// paged_sparse_sdpa writer (model-local generic op, custom_ops/paged_sparse_sdpa): tt-metal's sparse_sdpa writer
// without the attention sink, with the dual-NoC K-half gather rewritten for the paged TILE cache: per chunk request
// the reader posts through cb_kreq (base, groups, the chunk's cb_k_in L1, the index row's L1) this writer gathers
// the ODD tiles t of every pool on ITS NoC (the reader takes the even ones) with the reader's exact page / face math,
// then acks through cb_kack. Builds the three persistent compute-input tiles once (the reduce identity scaler, the
// col-identity, the all-(-inf) tile), then per token drains the untilized [H, V_DIM] row-major block from compute
// into the ROW_MAJOR [1, H, S, V_DIM] output (head-row h of token tok = page h*S + tok). The gather requests of a
// token precede its output block, so the per-token loop serves requests until the one flagged last (every token
// has at least one chunk), then drains the block.
#include <stdint.h>
#include <tt-metalium/constants.hpp>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp"  // generate_bcast_col_scalar
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "api/core_local_mem.h"

constexpr uint32_t one_bf16_packed = 0x3F803F80u;  // bf16(1.0) double-packed; generate_bcast_col_scalar uses >>16
constexpr uint16_t neg_inf_bf16 = 0xFF80u;
constexpr uint32_t sentinel = 0xFFFFFFFFu;

namespace wct {
enum : uint32_t {
    H = 0,
    S,
    V_DHT,
    CB_OUT_RM,
    CB_SCALE,
    CB_COL_IDENTITY,
    CB_NEGINF,
    OUT_ELEM_BYTES,
    CB_KREQ,
    CB_KACK,
    DHT,
    SKT,
    BLOCKS_PER_SEQ,
    POOL,
    END
};
}
namespace kreq {  // the chunk request the reader hands this writer (one cb_kreq page)
enum : uint32_t { BASE = 0, N_GROUPS, DST_L1, IDX_L1, IS_LAST, WORDS };
}

void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(wct::H);
    constexpr uint32_t S = get_compile_time_arg_val(wct::S);
    constexpr uint32_t vDHt = get_compile_time_arg_val(wct::V_DHT);
    constexpr uint32_t cb_out_rm = get_compile_time_arg_val(wct::CB_OUT_RM);
    constexpr uint32_t cb_scale = get_compile_time_arg_val(wct::CB_SCALE);
    constexpr uint32_t cb_col_identity = get_compile_time_arg_val(wct::CB_COL_IDENTITY);
    constexpr uint32_t cb_neginf = get_compile_time_arg_val(wct::CB_NEGINF);
    constexpr uint32_t out_elem_bytes = get_compile_time_arg_val(wct::OUT_ELEM_BYTES);
    constexpr uint32_t cb_kreq = get_compile_time_arg_val(wct::CB_KREQ);
    constexpr uint32_t cb_kack = get_compile_time_arg_val(wct::CB_KACK);
    constexpr uint32_t DHt = get_compile_time_arg_val(wct::DHT);
    constexpr uint32_t Skt = get_compile_time_arg_val(wct::SKT);
    constexpr uint32_t blocks_per_seq = get_compile_time_arg_val(wct::BLOCKS_PER_SEQ);
    constexpr uint32_t P = get_compile_time_arg_val(wct::POOL);
    constexpr auto out_args = TensorAccessorArgs<wct::END>();
    constexpr auto cache_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

    const uint32_t tok_start = get_arg_val<uint32_t>(0);
    const uint32_t tok_count = get_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_arg_val<uint32_t>(2);
    const uint32_t cache_addr = get_arg_val<uint32_t>(3);
    const uint32_t slot_block0 = get_arg_val<uint32_t>(4);

    constexpr uint32_t row_bytes = vDHt * tt::constants::TILE_WIDTH * out_elem_bytes;  // V_DIM bytes per head-row
    constexpr uint32_t block_tiles = (H / tt::constants::TILE_HEIGHT) * vDHt;          // Sqt*vDHt: the [H,V_DIM] block
    constexpr uint32_t tile_bytes = 2048;
    constexpr uint32_t face_bytes = 512;
    constexpr uint32_t frow_bytes = 32;
    constexpr uint32_t run_bytes = P * frow_bytes;
    (void)Skt;

    Noc noc;
    experimental::CB out_cb(cb_out_rm), kreq_cb(cb_kreq), kack_cb(cb_kack);
    const auto out = TensorAccessor(out_args, out_addr);
    const auto cache = TensorAccessor(cache_args, cache_addr);
    auto face_off = [](uint32_t r) -> uint32_t {
        return (r / tt::constants::FACE_HEIGHT) * 2u * face_bytes + (r % tt::constants::FACE_HEIGHT) * frow_bytes;
    };

    // --- persistent compute-input tiles (built once) ---
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_scale,
        ckernel::PoolType::MAX,
        ckernel::ReduceDim::REDUCE_ROW,
        /*reduce_factor=*/1>();
    generate_bcast_col_scalar(CircularBuffer(cb_col_identity), one_bf16_packed);
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

    for (uint32_t tok = tok_start; tok < tok_start + tok_count; ++tok) {
        for (bool last = false; !last;) {  // the token's chunk requests, the last one flagged by the reader
            kreq_cb.wait_front(1);
            uint32_t base, n_groups, dst, idx_l1;
            {
                volatile tt_l1_ptr uint32_t* rq =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kreq_cb.get_read_ptr());
                base = rq[kreq::BASE];
                n_groups = rq[kreq::N_GROUPS];
                dst = rq[kreq::DST_L1];
                idx_l1 = rq[kreq::IDX_L1];
                last = rq[kreq::IS_LAST] != 0;
            }
            kreq_cb.pop_front(1);
            volatile tt_l1_ptr uint32_t* idx_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1);
            for (uint32_t g = 0; g < n_groups; ++g) {
                const uint32_t r = g * P;
                const uint32_t id0 = idx_ptr[base + r];
                const uint32_t block = id0 / tt::constants::TILE_HEIGHT;
                if (id0 == sentinel || block >= blocks_per_seq) {
                    continue;  // the reader zeroes it
                }
                const uint32_t soff = face_off(id0 % tt::constants::TILE_HEIGHT);
                const uint32_t rt = r / tt::constants::TILE_HEIGHT;
                const uint32_t doff = face_off(r % tt::constants::TILE_HEIGHT);
                const uint32_t pblock = slot_block0 + block;
                for (uint32_t t = 1; t < DHt; t += 2) {  // the odd tiles; the reader gathers the even ones
                    const uint32_t page = pblock * DHt + t;
                    const uint32_t dtile = dst + (rt * DHt + t) * tile_bytes;
                    for (uint32_t half = 0; half < 2; ++half) {
                        noc.async_read(
                            cache,
                            CoreLocalMem<uint32_t>(dtile + doff + half * face_bytes),
                            run_bytes,
                            {.page_id = page, .offset_bytes = soff + half * face_bytes},
                            {});
                    }
                }
            }
            noc.async_read_barrier();
            kack_cb.reserve_back(1);
            kack_cb.push_back(1);
        }
        out_cb.wait_front(block_tiles);  // one untilized [H, V_DIM] block
        for (uint32_t h = 0; h < H; ++h) {
            noc.async_write(out_cb, out, row_bytes, {.offset_bytes = h * row_bytes}, {.page_id = h * S + tok});
        }
        noc.async_write_barrier();
        out_cb.pop_front(block_tiles);
    }
}
