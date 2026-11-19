// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused mHC "pre" — stream kernel (BRISC, NoC0): tensor reads and output writes.
//
// K core: fn^T tiles + x tiles (flat page order s * 128 + c, or the stacked [T, 1, 4, 4096] residual re-laid out
//   row by row into flat tiles) + norm weight tiles in; optional residual copy out (stacked or flat order, for the
//   mix that follows); the two hn tiles out.
// Root: the scale / bias row tiles in.
//
// Compile-time args (K): X_MODE (0 flat, 1 stacked), XOUT_MODE (0 none, 1 stacked, 2 flat), then the
// TensorAccessorArgs of x, fn, w, xout, hn. Root: TensorAccessorArgs of sb.
// Runtime args (K): x, fn, w, xout, hn addresses, first global column tile c0, tokens T.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "mhc_pre_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/probe.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"

using namespace mhc;

namespace {

// copy one 16-element (64-byte) face row of an fp32 tile: element offsets src_off -> dst_off
FORCE_INLINE void copy_face_row(uint32_t src_l1, uint32_t src_off, uint32_t dst_l1, uint32_t dst_off) {
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_l1) + src_off;
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_l1) + dst_off;
    for (uint32_t i = 0; i < 16; ++i) {
        d[i] = s[i];
    }
}

// row `src_row` of an fp32 tile at src_l1 -> row `dst_row` of the fp32 tile at dst_l1 (both face halves)
FORCE_INLINE void copy_tile_row(uint32_t src_l1, uint32_t src_row, uint32_t dst_l1, uint32_t dst_row) {
    copy_face_row(src_l1, tile_off(src_row, 0), dst_l1, tile_off(dst_row, 0));
    copy_face_row(src_l1, tile_off(src_row, 16), dst_l1, tile_off(dst_row, 16));
}

FORCE_INLINE void zero_l1(uint32_t l1, uint32_t bytes) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    for (uint32_t i = 0; i < bytes / 4; ++i) {
        p[i] = 0;
    }
}

}  // namespace

void kernel_main() {
    DeviceZoneScopedN("MH-STREAM");  // device-profiler zone (no-op without the profiler)
#if defined(MHC_K)
    MK_MARK();  // 0: start
    constexpr uint32_t X_MODE = get_compile_time_arg_val(PH_CT0 + 0);
    constexpr uint32_t XOUT_MODE = get_compile_time_arg_val(PH_CT0 + 1);
    constexpr auto x_args = TensorAccessorArgs<PH_CT0 + 2>();
    constexpr auto fn_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<fn_args.next_compile_time_args_offset()>();
    constexpr auto xout_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto hn_args = TensorAccessorArgs<xout_args.next_compile_time_args_offset()>();

    const uint32_t x_addr = get_arg_val<uint32_t>(ph_rt0 + 0);
    const uint32_t fn_addr = get_arg_val<uint32_t>(ph_rt0 + 1);
    const uint32_t w_addr = get_arg_val<uint32_t>(ph_rt0 + 2);
    const uint32_t xout_addr = get_arg_val<uint32_t>(ph_rt0 + 3);
    const uint32_t hn_addr = get_arg_val<uint32_t>(ph_rt0 + 4);
    const uint32_t c0 = get_arg_val<uint32_t>(ph_rt0 + 5);  // first global column tile of this core
    const uint32_t T = get_arg_val<uint32_t>(ph_rt0 + 6);

    const auto x = TensorAccessor(x_args, x_addr);
    const auto fn = TensorAccessor(fn_args, fn_addr);
    const auto w = TensorAccessor(w_args, w_addr);
    const auto hn = TensorAccessor(hn_args, hn_addr);

    // fn^T rows of this core's K tiles: page = global K tile (s * 128 + c)
    cb_reserve_back(CB_FN, X_TILES);
    {
        const uint32_t l1 = get_write_ptr(CB_FN);
        const uint32_t fn_page = get_tile_size(CB_FN);  // bf16 or fp32 fn^T
        for (uint32_t s = 0; s < N_STREAMS; ++s) {
            for (uint32_t j = 0; j < COLS_PER_CORE; ++j) {
                noc_async_read_page(s * COL_TILES + c0 + j, fn, l1 + (s * COLS_PER_CORE + j) * fn_page);
            }
        }
    }
    // x tiles, index s * 2 + j
    cb_reserve_back(CB_X, X_TILES);
    cb_reserve_back(CB_X32, X_TILES);
    const uint32_t x_l1 = get_write_ptr(CB_X);
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
#if defined(MHC_ROWREAD)
    const bool row_x = X_MODE != 0 || T <= 8;  // decode-row reads into zeroed tiles (else whole flat tiles)
#else
    const bool row_x = X_MODE != 0;
#endif
    // the x tiles' zero fill (row reads land in zeroed tiles; completed before them by the caller's barrier)
    auto zero_x = [&]() {
        for (uint32_t off = 0; off < X_TILES * FP32_TILE_BYTES; off += MEM_ZEROS_SIZE) {
            noc_async_read(zeros, x_l1 + off, MEM_ZEROS_SIZE);
        }
    };
    auto read_x = [&]() {
        if (X_MODE == 0 && row_x) {
            // Stage 14: decode rows only — the T valid rows of each flat tile (64-byte face-row reads) into tiles
            // zeroed by a local DMA (the full 4 KB tile reads moved 32x the data at T = 1: 2 MB of DRAM per call)
            for (uint32_t s = 0; s < N_STREAMS; ++s) {
                for (uint32_t j = 0; j < COLS_PER_CORE; ++j) {
                    const uint32_t page = s * COL_TILES + c0 + j;
                    const uint32_t dst = x_l1 + (s * COLS_PER_CORE + j) * FP32_TILE_BYTES;
                    for (uint32_t t = 0; t < T; ++t) {
                        noc_async_read(x.get_noc_addr(page, tile_off(t, 0) * 4), dst + tile_off(t, 0) * 4, 64);
                        noc_async_read(x.get_noc_addr(page, tile_off(t, 16) * 4), dst + tile_off(t, 16) * 4, 64);
                    }
                }
            }
        } else if (X_MODE == 0) {
            for (uint32_t s = 0; s < N_STREAMS; ++s) {
                for (uint32_t j = 0; j < COLS_PER_CORE; ++j) {
                    noc_async_read_page(s * COL_TILES + c0 + j, x, x_l1 + (s * COLS_PER_CORE + j) * FP32_TILE_BYTES);
                }
            }
        } else {
            // stacked [T, 1, 4, 4096]: token t, column tile c at page t * 128 + c, stream s in row s. Flat tile (s, j)
            // row t = stacked (t, c0 + j) row s: 64-byte face-row reads straight from the stacked page; every other
            // row of the flat tiles is zero.
            for (uint32_t t = 0; t < T; ++t) {
                for (uint32_t j = 0; j < COLS_PER_CORE; ++j) {
                    const uint32_t page = t * COL_TILES + c0 + j;
                    for (uint32_t s = 0; s < N_STREAMS; ++s) {
                        const uint32_t dst = x_l1 + (s * COLS_PER_CORE + j) * FP32_TILE_BYTES;
                        noc_async_read(x.get_noc_addr(page, tile_off(s, 0) * 4), dst + tile_off(t, 0) * 4, 64);
                        noc_async_read(x.get_noc_addr(page, tile_off(s, 16) * 4), dst + tile_off(t, 16) * 4, 64);
                    }
                }
            }
        }
    };
    // norm weight tiles (row 0 = weight of global columns 32c .. 32c + 31)
    auto read_w = [&]() {
        cb_reserve_back(CB_W, COLS_PER_CORE);
        const uint32_t l1 = get_write_ptr(CB_W);
#if defined(MHC_ROWREAD)
        // Stage 14: row 0 only (the weight row; the tile's other rows are zero in the tensor): 2 face rows per tile
        for (uint32_t off = 0; off < COLS_PER_CORE * FP32_TILE_BYTES; off += MEM_ZEROS_SIZE) {
            noc_async_read(zeros, l1 + off, MEM_ZEROS_SIZE);
        }
        noc_async_read_barrier();
        for (uint32_t j = 0; j < COLS_PER_CORE; ++j) {
            noc_async_read(
                w.get_noc_addr(c0 + j, tile_off(0, 0) * 4), l1 + j * FP32_TILE_BYTES + tile_off(0, 0) * 4, 64);
            noc_async_read(
                w.get_noc_addr(c0 + j, tile_off(0, 16) * 4), l1 + j * FP32_TILE_BYTES + tile_off(0, 16) * 4, 64);
        }
#else
        for (uint32_t j = 0; j < COLS_PER_CORE; ++j) {
            noc_async_read_page(c0 + j, w, l1 + j * FP32_TILE_BYTES);
        }
#endif
    };
#if defined(MK_STREAM_XWAIT)
    // Stage 14 (MK_PRE_PREFETCH, kernels/mk/owner_dm0.cpp): everything that does not read the residual first — fn^T,
    // the norm weight, the x tiles' zero fill — then the residual's hand-off (MK_STREAM_XWAIT: P1 of a stack layer
    // l > 0 = layer l - 1's P10 writes; P7 = this layer's P6 s1 writes), then only the x row reads are left on it
    if (row_x) {
        zero_x();
    }
    read_w();
    noc_async_read_barrier();
    cb_push_back(CB_FN, X_TILES);
    cb_push_back(CB_W, COLS_PER_CORE);
    MK_STREAM_XWAIT();
    read_x();
#else
    if (row_x) {
        zero_x();
        noc_async_read_barrier();
    }
    read_x();
    read_w();
#endif
    noc_async_read_barrier();
    MK_MARK();  // 1: x / fn / w in L1
#if !defined(MK_STREAM_XWAIT)
    cb_push_back(CB_FN, X_TILES);
#endif
    cb_push_back(CB_X, X_TILES);
    cb_push_back(CB_X32, X_TILES);
#if !defined(MK_STREAM_XWAIT)
    cb_push_back(CB_W, COLS_PER_CORE);
#endif

    // residual copy for the mix (the x tiles stay valid in L1 until the compute pops them after the collapse)
    if constexpr (XOUT_MODE == 1) {
        // stacked [T, 1, 4, 4096]: page t * 128 + c, row s = flat (s, c) row t
        const auto xout = TensorAccessor(xout_args, xout_addr);
        cb_reserve_back(CB_XSTK, 1);
        const uint32_t stk = get_write_ptr(CB_XSTK);
        for (uint32_t t = 0; t < T; ++t) {
            for (uint32_t j = 0; j < COLS_PER_CORE; ++j) {
                zero_l1(stk, FP32_TILE_BYTES);
                for (uint32_t s = 0; s < N_STREAMS; ++s) {
                    copy_tile_row(x_l1 + (s * COLS_PER_CORE + j) * FP32_TILE_BYTES, t, stk, s);
                }
                noc_async_write_page(t * COL_TILES + c0 + j, xout, stk);
                noc_async_write_barrier();
            }
        }
    } else if constexpr (XOUT_MODE == 2) {
        const auto xout = TensorAccessor(xout_args, xout_addr);
        for (uint32_t s = 0; s < N_STREAMS; ++s) {
            for (uint32_t j = 0; j < COLS_PER_CORE; ++j) {
                noc_async_write_page(s * COL_TILES + c0 + j, xout, x_l1 + (s * COLS_PER_CORE + j) * FP32_TILE_BYTES);
            }
        }
        noc_async_write_barrier();
    }

    // hn tiles
    cb_wait_front(CB_OUT, COLS_PER_CORE);
#if defined(MHC_SYNC_MARKS)
    MK_MARK();  // (probe runs) hn packed: the compute's collapse + finish done, before the hn writes / consumer signals
#endif
    {
        const uint32_t l1 = get_read_ptr(CB_OUT);
        const uint32_t page_bytes = get_tile_size(CB_OUT);
        for (uint32_t j = 0; j < COLS_PER_CORE; ++j) {
            noc_async_write_page(c0 + j, hn, l1 + j * page_bytes);
        }
    }
    noc_async_write_barrier();
    cb_pop_front(CB_OUT, COLS_PER_CORE);
#if defined(MK_SIGNAL_SEM_ID)
    // megakernel hand-off: hn of this core's columns is in the hn tensor -> increment the consumer (reader) cores'
    // hand-off semaphore; runtime args 7.. = MK_SIGNAL_N (x, y) NoC coordinates
    for (uint32_t r = 0; r < MK_SIGNAL_N; ++r) {
        const uint32_t rx = get_arg_val<uint32_t>(ph_rt0 + 7 + 2 * r), ry = get_arg_val<uint32_t>(ph_rt0 + 8 + 2 * r);
        noc_semaphore_inc(get_noc_addr(rx, ry, get_semaphore(MK_SIGNAL_SEM_ID)), 1);
    }
    noc_async_atomic_barrier();
#endif
    MK_MARK();  // 2: hn written (+ consumers signalled)
#if defined(MK_PROBE_ARG)
    MK_FLUSH(get_write_ptr(MK_PROBE_CB));
#endif
#elif defined(MHC_ROOT)
    constexpr auto sb_args = TensorAccessorArgs<PH_CT0 + 0>();
    const uint32_t sb_addr = get_arg_val<uint32_t>(ph_rt0 + 0);
    const auto sb = TensorAccessor(sb_args, sb_addr);
    cb_reserve_back(CB_SB, 2);
    const uint32_t l1 = get_write_ptr(CB_SB);
    noc_async_read_page(0, sb, l1);
    noc_async_read_page(1, sb, l1 + FP32_TILE_BYTES);
    noc_async_read_barrier();
    cb_push_back(CB_SB, 2);
#endif
}
