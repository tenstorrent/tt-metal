// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PMK mHC phases — stream kernel (BRISC, NoC0; mhc_common.hpp). Derived from Stage 7a's kernels/mhc_pre_pf/stream.cpp
// (member fn^T / w / x reads, hn writes) and mhc_post_pf7a/{reader,writer}.cpp (y / x reads, output writes).
// PMK_RESIDENT = 1 (TP8): the group's one residual block is read from DRAM once (F_X_LOAD, the layer-boundary hand-off)
//   and written back once (F_X_STORE); between the sublayers it stays in CB_X. The post's 4 output tiles per column
//   are copied back IN PLACE (local NoC writes from CB_PSTAGE into the residual pages of that column); a TOKEN push
//   orders the copy-back before the following pre's matmuls.
// PMK_RESIDENT = 0 (TP2): per sublayer the group's NRB blocks stream through CB_X for the pre and again for the post;
//   the staged output tiles go to the DRAM residual pages in place.
// Parts (PMK_MHC_PART): P1 = [x load] fn^T/w(0) load, hn(0) out, [gated prefetch of fn^T/w(1)]; P10 = y(0) in,
// copy-back(0), TOKEN, [fn^T/w(1) load], hn(1) out; P16 = y(1) in, copy-back(1), TOKEN, [x store]; 0 = all three.
// Member compile-time (PH_CT0 +): Y_PAGE, then the TensorAccessorArgs of x, fn, w, hn, y (one accessor per tensor
// kind). Member runtime (ph_rt0 +): x address, first row block (= g), row-block stride (8), first column tile c0,
// flags, gate semaphore id, gate value (the family's "IN0 free after P7" counter: the prefetch waits >= value; id
// 0xFFFFFFFF = no gate), then per sublayer: fn, w, hn, y addresses (4 each).
// Root compile-time: TensorAccessorArgs of sb; runtime: sb(0) address, flags, sb(1) address (rev 6: per sublayer).
// Page orders (DRAM, interleaved TILE): residual tile (rb, s, c) = rb * 512 + s * 128 + c; hn / y tile (rb, c)
// = rb * 128 + c; fn^T tile (s, c) = s * 128 + c; w tile c.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "mhc_common.hpp"

using namespace pmk_mhc;

#if defined(PMK_MEMBER)
namespace {
constexpr uint32_t NRB = PMK_NRB;
constexpr uint32_t Y_PAGE = get_compile_time_arg_val(PH_CT0 + 0);
constexpr auto x_args = TensorAccessorArgs<PH_CT0 + 1>();
constexpr auto fn_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
constexpr auto w_args = TensorAccessorArgs<fn_args.next_compile_time_args_offset()>();
constexpr auto hn_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
constexpr auto y_args = TensorAccessorArgs<hn_args.next_compile_time_args_offset()>();
constexpr uint32_t SUB_ARGS = 7;

struct Member {
    uint32_t x_addr, rb0, step, c0, flags, gate_sem, gate_val;
    uint32_t sub_arg(uint32_t sub, uint32_t i) const { return get_arg_val<uint32_t>(ph_rt0 + SUB_ARGS + 4 * sub + i); }
    uint32_t x_l1() const { return get_read_ptr(CB_X); }  // resident: read_ptr == base (never popped)

    // the residual block rb -> CB_X (both aliases), 64 pages
    void read_x(uint32_t rb) const {
        const auto x = TensorAccessor(x_args, x_addr);
        cb_reserve_back(CB_X, X_TILES);
        cb_reserve_back(CB_X32, X_TILES);
        const uint32_t l1 = get_write_ptr(CB_X);
        const uint32_t base = rb * N_STREAMS * COL_TILES + c0;
        for (uint32_t s = 0; s < N_STREAMS; ++s) {
            for (uint32_t j = 0; j < CPC; ++j) {
                noc_async_read_page(base + s * COL_TILES + j, x, l1 + (s * CPC + j) * FP32_TILE_BYTES);
            }
        }
        noc_async_read_barrier();
        cb_push_back(CB_X, X_TILES);
        cb_push_back(CB_X32, X_TILES);
    }
    // fn^T slab + norm weight tiles of one sublayer (the compute pops CB_FN after its last phase 1 and CB_W after
    // its last finish)
    void load_fn_w(uint32_t sub) const {
        const auto fn = TensorAccessor(fn_args, sub_arg(sub, 0));
        const auto w = TensorAccessor(w_args, sub_arg(sub, 1));
        cb_reserve_back(CB_FN, X_TILES);
        cb_reserve_back(CB_W, CPC);
        const uint32_t l1 = get_write_ptr(CB_FN);
        const uint32_t wl1 = get_write_ptr(CB_W);
        for (uint32_t s = 0; s < N_STREAMS; ++s) {
            for (uint32_t j = 0; j < CPC; ++j) {
                noc_async_read_page(s * COL_TILES + c0 + j, fn, l1 + (s * CPC + j) * BF16_TILE_BYTES);
            }
        }
        for (uint32_t j = 0; j < CPC; ++j) {
            noc_async_read_page(c0 + j, w, wl1 + j * BF16_TILE_BYTES);
        }
        noc_async_read_barrier();
        cb_push_back(CB_FN, X_TILES);
        cb_push_back(CB_W, CPC);
    }
    // the hn tiles of one row block -> DRAM (F_XH_DRAM; the L1 push to the AG store is the sync RISC's job)
    void write_hn(uint32_t sub, uint32_t rb) const {
        const auto hn = TensorAccessor(hn_args, sub_arg(sub, 2));
        cb_wait_front(CB_OUT, CPC);
        const uint32_t l1 = get_read_ptr(CB_OUT);
        const uint32_t page = get_tile_size(CB_OUT);
        for (uint32_t j = 0; j < CPC; ++j) {
            noc_async_write_page(rb * COL_TILES + c0 + j, hn, l1 + j * page);
        }
        noc_async_write_barrier();
        cb_pop_front(CB_OUT, CPC);
    }
    // the y tiles of one row block -> CB_Y (F_Y_DRAM; lane-landed y is pushed by the sync RISC after FS_REDUCED)
    void read_y(uint32_t sub, uint32_t rb) const {
        const auto y = TensorAccessor(y_args, sub_arg(sub, 3));
        cb_reserve_back(CB_Y, CPC);
        const uint32_t l1 = get_write_ptr(CB_Y);
        for (uint32_t j = 0; j < CPC; ++j) {
            noc_async_read_page(rb * COL_TILES + c0 + j, y, l1 + j * Y_PAGE);
        }
        noc_async_read_barrier();
        cb_push_back(CB_Y, CPC);
    }
    // the 4 staged output tiles of each column: resident -> in place (local NoC writes into the residual pages);
    // streamed -> the DRAM residual pages of the block
    void copy_back(uint32_t rb) const {
        const auto x = TensorAccessor(x_args, x_addr);
        const uint32_t xl = x_l1();
        (void)x;  // (resident: local writes only)
        for (uint32_t j = 0; j < CPC; ++j) {
            cb_wait_front(CB_PSTAGE, N_STREAMS);
            const uint32_t st = get_read_ptr(CB_PSTAGE);
            for (uint32_t s = 0; s < N_STREAMS; ++s) {
#if PMK_RESIDENT
                noc_async_write(
                    st + s * FP32_TILE_BYTES, get_noc_addr(xl + (s * CPC + j) * FP32_TILE_BYTES), FP32_TILE_BYTES);
#else
                (void)xl;
                noc_async_write_page(rb * N_STREAMS * COL_TILES + s * COL_TILES + c0 + j, x, st + s * FP32_TILE_BYTES);
#endif
            }
            noc_async_write_barrier();
            cb_pop_front(CB_PSTAGE, N_STREAMS);
        }
    }

    // ---- the pre half of a sublayer: x blocks (streamed) and the hn writes (DRAM mode)
    void pre(uint32_t sub) const {
        for (uint32_t k = 0; k < NRB; ++k) {
            const uint32_t rb = rb0 + k * step;
#if !PMK_RESIDENT
            read_x(rb);
#endif
            if (flags & F_XH_DRAM) {
                write_hn(sub, rb);
            }
        }
    }
    // ---- the post half: y in (DRAM mode), copy-back per block, the TOKEN (resident)
    void post(uint32_t sub) const {
        for (uint32_t k = 0; k < NRB; ++k) {
            const uint32_t rb = rb0 + k * step;
#if !PMK_RESIDENT
            read_x(rb);
#endif
            if (flags & F_Y_DRAM) {
                read_y(sub, rb);
            }
            copy_back(rb);
        }
#if PMK_RESIDENT
        cb_reserve_back(CB_TOKEN, 1);  // "copy-back landed" for the following pre
        cb_push_back(CB_TOKEN, 1);
#endif
    }
    void prefetch_next() const {        // sublayer 1's fn^T / w, after the family's IN0-free gate (the FN / W region is
        if (gate_sem != 0xFFFFFFFFu) {  // the stationary MM cores' IN0 / XS during attention at TP8)
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(gate_sem)), gate_val);
        }
        load_fn_w(1);
    }

    void part_p1() const {
#if PMK_RESIDENT
        if (flags & F_X_LOAD) {
            read_x(rb0);
        }
#endif
        load_fn_w(0);
        pre(0);
        if (flags & F_FN_PREFETCH) {
            prefetch_next();
        }
    }
    void part_p10() const {
        post(0);
        if (!(flags & F_FN_PREFETCH)) {
            load_fn_w(1);
        }
        pre(1);
    }
    void part_p16() const {
        post(1);
#if PMK_RESIDENT
        if (flags & F_X_STORE) {  // the layer-boundary hand-off: the new residual block back to the DRAM residual
            const auto x = TensorAccessor(x_args, x_addr);
            const uint32_t xl = x_l1();
            const uint32_t base = rb0 * N_STREAMS * COL_TILES + c0;
            for (uint32_t s = 0; s < N_STREAMS; ++s) {
                for (uint32_t j = 0; j < CPC; ++j) {
                    noc_async_write_page(base + s * COL_TILES + j, x, xl + (s * CPC + j) * FP32_TILE_BYTES);
                }
            }
            noc_async_write_barrier();
        }
#endif
    }
};
}  // namespace
#endif  // PMK_MEMBER

void kernel_main() {
#if defined(PMK_MEMBER)
    const Member m{
        get_arg_val<uint32_t>(ph_rt0 + 0),
        get_arg_val<uint32_t>(ph_rt0 + 1),
        get_arg_val<uint32_t>(ph_rt0 + 2),
        get_arg_val<uint32_t>(ph_rt0 + 3),
        get_arg_val<uint32_t>(ph_rt0 + 4),
        get_arg_val<uint32_t>(ph_rt0 + 5),
        get_arg_val<uint32_t>(ph_rt0 + 6)};
#if PMK_MHC_PART == 1
    m.part_p1();
#elif PMK_MHC_PART == 10
    m.part_p10();
#elif PMK_MHC_PART == 16
    m.part_p16();
#else
    m.part_p1();
    m.part_p10();
    m.part_p16();
#endif
#elif defined(PMK_ROOT)
    // rev 6: CB_SB holds ONE sublayer's scale / bias / identity tiles (hc_attn's and hc_ffn's rows differ); the compute
    // pops them after the sublayer's rounds. P1 (F_INIT) loads sublayer 0's; P10 loads sublayer 1's unless the
    // integrator's prelude does (F_SB_PRELUDE); the all-in-one part loads both in order (the second reserve waits for
    // the compute's pop). Runtime: sb(0) address, flags, sb(1) address.
    constexpr auto sb_args = TensorAccessorArgs<PH_CT0 + 0>();
    const uint32_t flags = get_arg_val<uint32_t>(ph_rt0 + 1);
    auto load_sb = [&](uint32_t addr) {
        const auto sb = TensorAccessor(sb_args, addr);
        cb_reserve_back(CB_SB, 3);
        const uint32_t l1 = get_write_ptr(CB_SB);
        for (uint32_t i = 0; i < 3; ++i) {
            noc_async_read_page(i, sb, l1 + i * FP32_TILE_BYTES);
        }
        noc_async_read_barrier();
        cb_push_back(CB_SB, 3);
    };
    (void)load_sb;
#if PMK_MHC_PART == 1 || PMK_MHC_PART == 0
    if (flags & F_INIT) {
        load_sb(get_arg_val<uint32_t>(ph_rt0 + 0));
    }
#endif
#if PMK_MHC_PART == 10 || PMK_MHC_PART == 0
    if (!(flags & F_SB_PRELUDE)) {
        load_sb(get_arg_val<uint32_t>(ph_rt0 + 2));
    }
#endif
#endif
}
