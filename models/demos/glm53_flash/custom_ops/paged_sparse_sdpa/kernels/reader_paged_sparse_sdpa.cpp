// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// paged_sparse_sdpa reader (model-local generic op, custom_ops/paged_sparse_sdpa): tt-metal's sparse_sdpa reader with
// the K gather rewritten for a PAGED bf16 TILE latent cache. Per token: the Q head-rows -> cb_q_rm, the token's index
// row -> L1, the valid-key count nv (first sentinel) -> cb_ctrl, then per active K chunk of k_chunk rows the selected
// rows are gathered straight out of the cache's tiles into cb_k_in in TILE layout (no ROW_MAJOR copy, no tilize):
// the index row is the indexer's expansion of 4-token pools (ids 4p..4p+3 at chunk rows 4g..4g+3, the tail pool's
// ids, then sentinels), so a pool is 4 consecutive rows inside one 16-row tile face of the source block AND of the
// destination tile: 32 reads of 128 contiguous bytes (16 tiles x 2 face columns), the alignment Blackhole's DRAM
// reads need. Rows at or past nv (the sentinel suffix, the over-read rows of a short tail pool) are zeroed in L1
// after the gather, exactly the stock reader's zero fill. The chunk's boundary mask tile (cols >= nv % 32 of the
// straddling key tile) is built as in the stock reader. Pages: cache tile (block, t) = page block*DHt + t; the
// user's pages are contiguous (identity page table): physical block = slot_block0 + id / 32.
#include <stdint.h>
#include <tt-metalium/constants.hpp>
#include "api/dataflow/dataflow_api.h"
#include "api/core_local_mem.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/sparse_sdpa_common.hpp"

constexpr uint32_t sentinel = 0xFFFFFFFFu;
constexpr uint16_t neg_inf_bf16 = 0xFF80u;

namespace rct {
enum : uint32_t { H = 0, S, TOPK, K_CHUNK, K_DIM, CB_Q_RM, CB_K_IN, CB_MASK_PART, CB_IDX, CB_CTRL, DHT, SKT, BLOCKS_PER_SEQ, POOL, END };
}

void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(rct::H);
    constexpr uint32_t S = get_compile_time_arg_val(rct::S);
    constexpr uint32_t topk = get_compile_time_arg_val(rct::TOPK);
    constexpr uint32_t k_chunk = get_compile_time_arg_val(rct::K_CHUNK);
    constexpr uint32_t k_dim = get_compile_time_arg_val(rct::K_DIM);
    constexpr uint32_t cb_q_rm = get_compile_time_arg_val(rct::CB_Q_RM);
    constexpr uint32_t cb_k_in = get_compile_time_arg_val(rct::CB_K_IN);
    constexpr uint32_t cb_mask_part = get_compile_time_arg_val(rct::CB_MASK_PART);
    constexpr uint32_t cb_idx = get_compile_time_arg_val(rct::CB_IDX);
    constexpr uint32_t cb_ctrl = get_compile_time_arg_val(rct::CB_CTRL);
    constexpr uint32_t DHt = get_compile_time_arg_val(rct::DHT);  // tiles per row (k_dim / 32)
    constexpr uint32_t Skt = get_compile_time_arg_val(rct::SKT);  // tile rows per chunk (k_chunk / 32)
    constexpr uint32_t blocks_per_seq = get_compile_time_arg_val(rct::BLOCKS_PER_SEQ);
    constexpr uint32_t P = get_compile_time_arg_val(rct::POOL);  // tokens per pool (4): the run length
    constexpr auto q_args = TensorAccessorArgs<rct::END>();
    constexpr auto cache_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto idx_args = TensorAccessorArgs<cache_args.next_compile_time_args_offset()>();

    const uint32_t tok_start = get_arg_val<uint32_t>(0);
    const uint32_t tok_count = get_arg_val<uint32_t>(1);
    const uint32_t q_addr = get_arg_val<uint32_t>(2);
    const uint32_t cache_addr = get_arg_val<uint32_t>(3);
    const uint32_t idx_addr = get_arg_val<uint32_t>(4);
    const uint32_t slot_block0 = get_arg_val<uint32_t>(5);  // the user's first physical block (identity table)

    constexpr uint32_t q_row_bytes = k_dim * 2;      // bf16 Q row
    constexpr uint32_t idx_row_bytes = topk * 4;     // one index row (uint32)
    constexpr uint32_t tile_bytes = 2048;            // bf16 32x32
    constexpr uint32_t face_bytes = 512;             // 16x16 bf16
    constexpr uint32_t frow_bytes = 32;              // 16 bf16 of one face row
    constexpr uint32_t run_bytes = P * frow_bytes;   // a pool's rows in one face column
    constexpr uint32_t chunk_tiles = Skt * DHt;
    static_assert(k_chunk % tt::constants::TILE_HEIGHT == 0, "k_chunk must be whole tile rows");
    static_assert(tt::constants::FACE_HEIGHT % P == 0, "a pool must not cross a tile face");

    Noc noc;
    experimental::CB q_cb(cb_q_rm), k_cb(cb_k_in), mask_part_cb(cb_mask_part), idx_cb(cb_idx), ctrl_cb(cb_ctrl);
    const auto q = TensorAccessor(q_args, q_addr);
    const auto cache = TensorAccessor(cache_args, cache_addr);
    const auto idx = TensorAccessor(idx_args, idx_addr);
    idx_cb.reserve_back(1);
    const uint32_t idx_l1 = idx_cb.get_write_ptr();
    volatile tt_l1_ptr uint32_t* idx_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1);

    // face-layout byte offset of row r (0..31) inside a tile, face column 0
    auto face_off = [](uint32_t r) -> uint32_t {
        return (r / tt::constants::FACE_HEIGHT) * 2u * face_bytes + (r % tt::constants::FACE_HEIGHT) * frow_bytes;
    };

    for (uint32_t tok = tok_start; tok < tok_start + tok_count; ++tok) {
        // Q: H head-rows -> cb_q_rm (row (h, tok) = page h*S + tok)
        q_cb.reserve_back(H);
        for (uint32_t h = 0; h < H; ++h) {
            noc.async_read(q, q_cb, q_row_bytes, {.page_id = h * S + tok}, {.offset_bytes = h * q_row_bytes});
        }
        noc.async_read_barrier();
        q_cb.push_back(H);

        // the token's index row -> L1 scratch
        noc.async_read(idx, idx_cb, idx_row_bytes, {.page_id = tok}, {.offset_bytes = 0});
        noc.async_read_barrier();

        // nv = the first sentinel (binary search; the sentinels are a contiguous tail)
        uint32_t nv = topk;
        {
            uint32_t lo = 0, hi = topk;
            while (lo < hi) {
                const uint32_t mid = (lo + hi) >> 1;
                if (idx_ptr[mid] == sentinel) {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
            }
            nv = lo == 0 ? 1 : lo;
        }
        const uint32_t n_active = (nv + k_chunk - 1) / k_chunk;

        ctrl_cb.reserve_back(1);
        {
            volatile tt_l1_ptr uint32_t* cp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctrl_cb.get_write_ptr());
            cp[sparse_sdpa::control_message::ACTIVE_CHUNKS] = n_active;
            cp[sparse_sdpa::control_message::VALID_KEYS] = nv;
        }
        ctrl_cb.push_back(1);

        for (uint32_t chunk = 0; chunk < n_active; ++chunk) {
            const uint32_t base = chunk * k_chunk;
            const uint32_t valid = (nv - base) < k_chunk ? (nv - base) : k_chunk;  // (0, k_chunk]

            k_cb.reserve_back(chunk_tiles);
            const uint32_t dst = k_cb.get_write_ptr();
            // the gather: one 4-row pool per group g (chunk rows P*g .. P*g+P-1); the group's rows past `valid` are
            // over-read (their ids may be sentinels) and zeroed below
            const uint32_t n_groups = (valid + P - 1) / P;
            for (uint32_t g = 0; g < n_groups; ++g) {
                const uint32_t r = g * P;  // chunk-local destination row
                const uint32_t id0 = idx_ptr[base + r];
                const uint32_t block = id0 / tt::constants::TILE_HEIGHT;
                if (id0 == sentinel || block >= blocks_per_seq) {
                    continue;  // zeroed below (a sentinel below nv or an id outside the user's pages)
                }
                const uint32_t srow = id0 % tt::constants::TILE_HEIGHT;
                const uint32_t soff = face_off(srow);
                const uint32_t rt = r / tt::constants::TILE_HEIGHT;
                const uint32_t doff = face_off(r % tt::constants::TILE_HEIGHT);
                const uint32_t pblock = slot_block0 + block;
                for (uint32_t t = 0; t < DHt; ++t) {
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
            // zero rows [valid, k_chunk) in L1 (the sentinel suffix; the masked keys must be finite), plus any group
            // skipped above: whole tiles for fully empty tile rows, face runs for the straddling one
            if (valid < k_chunk) {
                volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
                const uint32_t rt_first = valid / tt::constants::TILE_HEIGHT;
                for (uint32_t rt = rt_first; rt < Skt; ++rt) {
                    const uint32_t r0 = rt == rt_first ? valid % tt::constants::TILE_HEIGHT : 0;
                    for (uint32_t t = 0; t < DHt; ++t) {
                        const uint32_t tile_words = ((rt * DHt + t) * tile_bytes) / 4;
                        if (r0 == 0) {
                            for (uint32_t w = 0; w < tile_bytes / 4; ++w) {
                                z[tile_words + w] = 0;
                            }
                            continue;
                        }
                        for (uint32_t r = r0; r < tt::constants::TILE_HEIGHT; ++r) {
                            const uint32_t off = face_off(r) / 4;
                            for (uint32_t half = 0; half < 2; ++half) {
                                for (uint32_t w = 0; w < frow_bytes / 4; ++w) {
                                    z[tile_words + off + half * (face_bytes / 4) + w] = 0;
                                }
                            }
                        }
                    }
                }
            }
            for (uint32_t g = 0; g < n_groups; ++g) {  // a skipped group below valid: zero its rows
                const uint32_t r = g * P;
                const uint32_t id0 = idx_ptr[base + r];
                if (id0 != sentinel && id0 / tt::constants::TILE_HEIGHT < blocks_per_seq) {
                    continue;
                }
                volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
                const uint32_t rt = r / tt::constants::TILE_HEIGHT;
                const uint32_t off = face_off(r % tt::constants::TILE_HEIGHT) / 4;
                for (uint32_t t = 0; t < DHt; ++t) {
                    const uint32_t tile_words = ((rt * DHt + t) * tile_bytes) / 4;
                    for (uint32_t half = 0; half < 2; ++half) {
                        for (uint32_t w = 0; w < run_bytes / 4; ++w) {
                            z[tile_words + off + half * (face_bytes / 4) + w] = 0;
                        }
                    }
                }
            }
            k_cb.push_back(chunk_tiles);

            // the partial-boundary mask tile (row 0: -inf for cols >= valid % 32), last active chunk only
            if (chunk == n_active - 1 && (valid % tt::constants::TILE_WIDTH) != 0) {
                mask_part_cb.reserve_back(1);
                {
                    const uint32_t first_masked_col = valid % tt::constants::TILE_WIDTH;
                    volatile tt_l1_ptr uint16_t* mask_tile =
                        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(mask_part_cb.get_write_ptr());
                    for (uint32_t col = 0; col < tt::constants::TILE_WIDTH; ++col) {
                        const uint32_t elem_off = (col < tt::constants::FACE_WIDTH)
                                                      ? col
                                                      : (tt::constants::FACE_HW + col - tt::constants::FACE_WIDTH);
                        mask_tile[elem_off] = (col >= first_masked_col) ? neg_inf_bf16 : (uint16_t)0;
                    }
                }
                mask_part_cb.push_back(1);
            }
        }
    }
}
