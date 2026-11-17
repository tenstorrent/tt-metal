// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// B35 (HY4_B35_GFILT=1): cs_gather (ids = cand[pos]) + id_filter fused into ONE program, bit-identical to the pair.
//   pos   uint32 RM [S,1,1,K]  top-k positions into the W gathered candidates of the row (position >= W -> sentinel)
//   cand  uint32 RM [S,1,1,W]  all-gathered local ids (LAND=0)  OR  (LAND=2, "own": W = KL) THIS chip's local candidate
//   ids only: the candidate at gathered position p comes from column p / KL and its token lives on that column (layout
//   invariant),
//         so only positions of this column's own block survive the column filter and no id gather at all is needed  OR
//         the push landing tensor (LAND=1: page c*NP + s/RP, byte offset (s%RP)*PB, source column c, PB = bytes of one
//         row of one source, W = 8*PB/4)
//   colinfo (64 B page: element 0 = column, element 1 / 2+s = number of valid tokens)   out uint32 RM [1,1,OWN*S,K]
// LAND=1: the push program (hy4_qpush.Push) returns without waiting for the peers; this kernel waits for flag[c] of
// every source c (monotonic counters, per-core launch counter in a second global semaphore). One core per user row s.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t K = get_compile_time_arg_val(0);
    constexpr uint32_t S = get_compile_time_arg_val(1);
    constexpr uint32_t OWN = get_compile_time_arg_val(2);
    constexpr uint32_t cb_id = get_compile_time_arg_val(3);
    constexpr uint32_t PERROW = get_compile_time_arg_val(4);
    constexpr uint32_t W = get_compile_time_arg_val(5);
    constexpr uint32_t LAND = get_compile_time_arg_val(6);
    constexpr uint32_t PB = get_compile_time_arg_val(7);  // LAND: bytes per source row
    constexpr uint32_t RP = get_compile_time_arg_val(8);  // LAND: rows per landing page
    constexpr uint32_t NP = get_compile_time_arg_val(9);  // LAND: packets (landing pages) per source per launch
    constexpr auto p_args = TensorAccessorArgs<10>();
    constexpr auto g_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    constexpr uint32_t RB = K * 4;
    constexpr uint32_t CIDB = W * 4;
    constexpr uint32_t OWNM = (LAND == 2);
    const uint32_t s = get_arg_val<uint32_t>(0);
    const uint32_t a_p = get_common_arg_val<uint32_t>(0), a_g = get_common_arg_val<uint32_t>(1),
                   a_c = get_common_arg_val<uint32_t>(2), a_o = get_common_arg_val<uint32_t>(3);
    Noc noc;
    const auto p_acc = TensorAccessor(p_args, a_p, RB);
    const auto g_acc = TensorAccessor(g_args, a_g, (LAND == 1) ? RP * PB : CIDB);
    const auto c_acc = TensorAccessor(c_args, a_c, 64);
    const auto o_acc = TensorAccessor(o_args, a_o, RB);
    experimental::CB cb(cb_id);
    cb.reserve_back(1);
    const uint32_t base = cb.get_write_ptr();
    // L1: [pos RB][out RB][cid CIDB][colinfo 64][flag scratch 8 x 16 B]
    volatile tt_l1_ptr uint32_t* pos = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    const uint32_t cid_base = base + 2 * RB;
    volatile tt_l1_ptr uint32_t* cid = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cid_base);
    const uint32_t ci_base = cid_base + CIDB;
    volatile tt_l1_ptr uint32_t* ci = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ci_base);
    const uint32_t scr = ci_base + 64;
    {
        DeviceZoneScopedN("GF_READ");
        noc.async_read(p_acc, cb, RB, {.page_id = s, .offset_bytes = 0}, {.offset_bytes = 0});
        noc.async_read(c_acc, cb, 64, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = 2 * RB + CIDB});
        if constexpr (LAND == 1) {
            const uint32_t fx = get_arg_val<uint32_t>(1), fy = get_arg_val<uint32_t>(2);
            volatile tt_l1_ptr uint32_t* cnt =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(11));
            const uint32_t target = *cnt + NP;
            *cnt = target;
            noc.async_read_barrier();
            bool done;
            do {
                for (uint32_t c = 0; c < 8; ++c) {
                    noc_async_read(get_noc_addr(fx, fy, get_arg_val<uint32_t>(3 + c)), scr + 16 * c, 4);
                }
                noc_async_read_barrier();
                invalidate_l1_cache();
                done = true;
                for (uint32_t c = 0; c < 8; ++c) {
                    done &= ((int32_t)(*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + 16 * c) - target) >= 0);
                }
            } while (!done);
            for (uint32_t c = 0; c < 8; ++c) {
                noc.async_read(
                    g_acc,
                    cb,
                    PB,
                    {.page_id = c * NP + s / RP, .offset_bytes = (s % RP) * PB},
                    {.offset_bytes = 2 * RB + c * PB});
            }
        } else {
            noc.async_read(g_acc, cb, CIDB, {.page_id = s, .offset_bytes = 0}, {.offset_bytes = 2 * RB});
        }
        noc.async_read_barrier();
        invalidate_l1_cache();
    }
    const uint32_t c = ci[0], nvalid = PERROW ? ci[2 + s] : ci[1];
    uint32_t n = 0;
    // plain (non-volatile) L1 pointers: the reads above are complete (barrier + invalidate), unrolled so the loads of
    // several iterations overlap
    const uint32_t* __restrict pp = reinterpret_cast<const uint32_t*>(base);
    const uint32_t* __restrict cc = reinterpret_cast<const uint32_t*>(cid_base);
    uint32_t* __restrict op = reinterpret_cast<uint32_t*>(base + RB);
    {
        DeviceZoneScopedN("GF_LOOP");
        if constexpr (OWNM) {
            const uint32_t lo = c * W;  // W = KL: this column's block of the gathered candidates
#pragma GCC unroll 4
            for (uint32_t i = 0; i < K; ++i) {
                const uint32_t q = pp[i] - lo;
                if (q < W) {
                    const uint32_t t = cc[q];
                    if (t < nvalid && ((t >> 6) & 7u) == c) {
                        op[n++] = ((t >> 9) << 6) | (t & 63u);
                    }
                }
            }
        } else {
            for (uint32_t i = 0; i < K; ++i) {
                const uint32_t p = pp[i];
                const uint32_t t = (p < W) ? cc[p] : 0xFFFFFFFFu;
                if (t < nvalid && ((t >> 6) & 7u) == c) {
                    op[n++] = ((t >> 9) << 6) | (t & 63u);
                }
            }
        }
    }
    {
        DeviceZoneScopedN("GF_FILL");
#pragma GCC unroll 8
        for (uint32_t i = n; i < K; ++i) {
            op[i] = 0xFFFFFFFFu;
        }
    }
    {
        DeviceZoneScopedN("GF_WRITE");
        for (uint32_t j = 0; j < OWN; ++j) {
            noc.async_write(cb, o_acc, RB, {.offset_bytes = RB}, {.page_id = j * S + s, .offset_bytes = 0});
        }
        noc.async_write_barrier();
    }
}
