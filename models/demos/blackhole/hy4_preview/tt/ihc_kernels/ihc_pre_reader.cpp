// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Hy4 iHC fused PRE kernel, reader (NCRISC).  32 cores (a physically contiguous 4x8 worker rectangle); core k owns the
// column tiles [k*GPC, (k+1)*GPC) of all four streams.  Token groups of 8 tokens are stacked into one x tile (row 4*tl
// + i = stream i of token tl).  Phases:
//   1. read the x chunk (two 256 B pieces per (token, tile)), the packed bf16 fn chunk (GPC tiles: row = hidden col c,
//   column 8i+m),
//      the base / scale-diagonal tiles (face 0 only) and the norm weights (rows 0..T-1 only)
//   2. exchange 1: the partial mixes + sum of squares tile of this core (face 0: row = token, columns 0..7 mixes, 8
//   mean of squares)
//      is sent (1 KB, flag word LAST) to slot k of EVERY core; every core sums all 32 slots (compute) and derives pre |
//      post
//   3. build the collapse matrices A_g from pre (RISC scatter); core 0 writes the pre | post tile to DRAM
//   4. exchange 2: partial sums of squares of the collapsed rows (same as the DSV4 mhc_cn kernel)

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

#define RW(r) ((((r) >> 4) << 9) + (((r) & 15) << 4))
#define RO(r) ((((r) >> 4) << 11) + (((r) & 15) << 6))

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_base = get_compile_time_arg_val(2);
    constexpr uint32_t cb_recv = get_compile_time_arg_val(3);
    constexpr uint32_t cb_part = get_compile_time_arg_val(4);
    constexpr uint32_t cb_pp = get_compile_time_arg_val(5);
    constexpr uint32_t cb_a = get_compile_time_arg_val(6);
    constexpr uint32_t cb_p2 = get_compile_time_arg_val(7);
    constexpr uint32_t cb_slot = get_compile_time_arg_val(8);
    constexpr uint32_t cb_tot = get_compile_time_arg_val(9);
    constexpr uint32_t cb_w4 = get_compile_time_arg_val(10);
    constexpr uint32_t T = get_compile_time_arg_val(11);
    constexpr uint32_t NT = get_compile_time_arg_val(12);
    constexpr uint32_t G = get_compile_time_arg_val(13);
    constexpr uint32_t GPC = get_compile_time_arg_val(14);
    constexpr uint32_t NC = get_compile_time_arg_val(15);
    constexpr uint32_t MX0 = get_compile_time_arg_val(16);
    constexpr uint32_t MY0 = get_compile_time_arg_val(17);
    constexpr uint32_t MW = get_compile_time_arg_val(18);
    constexpr uint32_t cb_sd = get_compile_time_arg_val(19);
    constexpr uint32_t cb_cst = get_compile_time_arg_val(20);
    constexpr auto x_args = TensorAccessorArgs<21>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto b_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto n_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    constexpr auto p_args = TensorAccessorArgs<n_args.next_compile_time_args_offset()>();
    constexpr auto d_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr uint32_t ABL = get_compile_time_arg_val(
        d_args.next_compile_time_args_offset());  // timing ablation bits (results wrong when set)
    constexpr uint32_t DBG =
        get_compile_time_arg_val(d_args.next_compile_time_args_offset() + 1);  // 1: per-phase wall-clock stamps
    constexpr uint32_t EXPAND = get_compile_time_arg_val(
        d_args.next_compile_time_args_offset() +
        2);  // 1: x_new = x + post_prev * y first (streams of the previous sub-block)
    constexpr uint32_t cb_xr = get_compile_time_arg_val(d_args.next_compile_time_args_offset() + 3);
    constexpr uint32_t cb_y = get_compile_time_arg_val(d_args.next_compile_time_args_offset() + 4);
    constexpr uint32_t cb_m = get_compile_time_arg_val(d_args.next_compile_time_args_offset() + 5);
    constexpr auto y_args = TensorAccessorArgs<d_args.next_compile_time_args_offset() + 6>();
    constexpr auto q_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    constexpr uint32_t DUAL = get_compile_time_arg_val(
        q_args.next_compile_time_args_offset());  // 1: the writer kernel sends the other half of the exchange writes
    constexpr uint32_t NCST = 10 + G;  // 8 constant tiles, G selection tiles, the all-ones tile, the identity
    constexpr uint32_t TILE = 4096;
    constexpr uint32_t SW = 36;              // exchange-2 slot words: T data words (<= 32), pad, flag
    constexpr uint32_t RR = T <= 2 ? 2 : (T <= 4 ? 4 : (T <= 8 ? 8 : 16));  // rows per exchange-1 slot
    constexpr uint32_t NR = RR / 2;                                         // recv tiles (32 slots of RR rows x 64 B)
    constexpr uint32_t SB = RR * 64;                                        // slot bytes
    constexpr uint32_t SWD = RR * 16;                                       // slot words
    constexpr uint32_t FLAG2 = 0xC0DE1234u;  // exchange 2 (slot memory only)
    constexpr uint32_t FLAGD = 0xD04E5AFEu;  // the writer kernel finished its half of the R tile
    constexpr uint32_t FLAG1 = 0x4B1C0DE1u;  // exchange 1: a FINITE fp32 (it is summed by matmuls), word 255 of a face

    if constexpr (ABL & 128) {
        return;  // launch-overhead probe
    }
    const uint32_t k = get_arg_val<uint32_t>(0);
    const uint32_t j0 = k * GPC;
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t b_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t n_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t p_addr = get_common_arg_val<uint32_t>(4);
    const uint32_t d_addr = get_common_arg_val<uint32_t>(5);
    const uint32_t y_addr = get_common_arg_val<uint32_t>(6);
    const uint32_t q_addr = get_common_arg_val<uint32_t>(7);

    Noc noc;
    const auto x_acc = TensorAccessor(x_args, x_addr, TILE);
    constexpr uint32_t WPAGE =
        get_compile_time_arg_val(q_args.next_compile_time_args_offset() + 1);  // 2048 (bf16) or 1088 (bfp8_b)
    const auto w_acc = TensorAccessor(w_args, w_addr, WPAGE);
    const auto b_acc = TensorAccessor(b_args, b_addr, TILE);
    const auto n_acc = TensorAccessor(n_args, n_addr, TILE);
    const auto p_acc = TensorAccessor(p_args, p_addr, TILE);
    const auto d_acc = TensorAccessor(d_args, d_addr, TILE);
    const auto y_acc = TensorAccessor(y_args, y_addr, 2048);
    const auto q_acc = TensorAccessor(q_args, q_addr, TILE);
    experimental::CB cbx(cb_x), cbw(cb_w), cbb(cb_base), cbsd(cb_sd), cbr(cb_recv), cbpart(cb_part), cbpp(cb_pp),
        cba(cb_a), cbp2(cb_p2), cbslot(cb_slot), cbtot(cb_tot), cbw4(cb_w4), cst(cb_cst), cbxr(cb_xr), cby(cb_y),
        cbm(cb_m);

    // memory other cores write into: reserved, NEVER touched before the flags arrive (flags are reset at the END of
    // every run)
    cbr.reserve_back(NR);
    const uint32_t recv_base = cbr.get_write_ptr();
    volatile tt_l1_ptr uint32_t* recv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(recv_base);
    cbslot.reserve_back(2);
    const uint32_t slot_base = cbslot.get_write_ptr();
    volatile tt_l1_ptr uint32_t* slots = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot_base);
    volatile tt_l1_ptr uint32_t* ts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot_base + 6144);
#define STAMP(i)                     \
    if constexpr (DBG) {             \
        ts[i] = get_timestamp_32b(); \
    }
    STAMP(0)

    cbb.reserve_back(1);
    cbsd.reserve_back(1);
    cbw.reserve_back(GPC);
    cbw4.reserve_back(GPC);
    if constexpr (EXPAND) {
        cbxr.reserve_back(G * GPC);
        cby.reserve_back(GPC);
        cbm.reserve_back(G);
    } else {
        cbx.reserve_back(G * GPC);
    }
    cba.reserve_back(G);
    cbtot.reserve_back(1);
    // 1. packed fn tiles (bf16, 2 KB), base / scale-diagonal (face 0 = rows 0..15), norm weights (rows 0..T-1 of faces
    // 0 and 1), x: issued FIRST
    if constexpr (!(ABL & 1)) {
        for (uint32_t n = 0; n < GPC; ++n) {
            noc.async_read(w_acc, cbw, WPAGE, {.page_id = k * GPC + n, .offset_bytes = 0}, {.offset_bytes = n * WPAGE});
        }
    }
    // stream tensor [1,1,4T,D] fp32 TILE: tile row g holds the rows (4*tl + i) of the tokens 8g .. 8g+7 -> one 4 KB
    // page per (g, j)
    // only the rows 4*tl + i of the T tokens exist (padding rows are never read: they are zeroed in L1); a full tile
    // when the group has 32 rows
    for (uint32_t g = 0; g < G; ++g) {
        const uint32_t rows_g = (4 * T - 32 * g) < 32 ? (4 * T - 32 * g) : 32;
        for (uint32_t j = 0; j < GPC; ++j) {
            if constexpr (ABL & 8) {
                break;
            }
            const uint32_t pg = g * NT + j0 + j;
            const uint32_t to = (g * GPC + j) * TILE;
            if (rows_g == 32) {
                if constexpr (EXPAND) {
                    noc.async_read(x_acc, cbxr, TILE, {.page_id = pg, .offset_bytes = 0}, {.offset_bytes = to});
                } else {
                    noc.async_read(x_acc, cbx, TILE, {.page_id = pg, .offset_bytes = 0}, {.offset_bytes = to});
                }
            } else {
                const uint32_t r0 = rows_g < 16 ? rows_g : 16;
                const uint32_t r1 = rows_g - r0;
                for (uint32_t f = 0; f < 4; ++f) {
                    const uint32_t rr = f < 2 ? r0 : r1;
                    if (rr == 0) {
                        continue;
                    }
                    if constexpr (EXPAND) {
                        noc.async_read(
                            x_acc,
                            cbxr,
                            rr * 64,
                            {.page_id = pg, .offset_bytes = f * 1024},
                            {.offset_bytes = to + f * 1024});
                    } else {
                        noc.async_read(
                            x_acc,
                            cbx,
                            rr * 64,
                            {.page_id = pg, .offset_bytes = f * 1024},
                            {.offset_bytes = to + f * 1024});
                    }
                }
                // zero the rest of the tile (complementary regions)
                for (uint32_t f = 0; f < 4; ++f) {
                    const uint32_t rr = f < 2 ? r0 : r1;
                    if (rr < 16) {
                        if constexpr (EXPAND) {
                            noc.async_write_zeros(cbxr, 1024 - rr * 64, {.offset_bytes = to + f * 1024 + rr * 64});
                        } else {
                            noc.async_write_zeros(cbx, 1024 - rr * 64, {.offset_bytes = to + f * 1024 + rr * 64});
                        }
                    }
                }
            }
        }
    }
    if constexpr (EXPAND) {
        // y [1,1,T,D] bf16 TILE: rows 0..T-1 of faces 0 and 1 of tile j (32 B per row); the previous pre|post tile
        // (face 0)
        for (uint32_t j = 0; j < GPC; ++j) {
            noc.async_read(y_acc, cby, T * 32, {.page_id = j0 + j, .offset_bytes = 0}, {.offset_bytes = j * 2048});
            noc.async_read(
                y_acc, cby, T * 32, {.page_id = j0 + j, .offset_bytes = 512}, {.offset_bytes = j * 2048 + 512});
        }
        noc.async_read(q_acc, cbslot, 1024, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = 4608});
        for (uint32_t j = 0; j < GPC; ++j) {  // absent rows of y read as 0 (never garbage: 0 * NaN)
            noc.async_write_zeros(cby, 512 - T * 32, {.offset_bytes = j * 2048 + T * 32});
            noc.async_write_zeros(cby, 512 - T * 32, {.offset_bytes = j * 2048 + 512 + T * 32});
            noc.async_write_zeros(cby, 1024, {.offset_bytes = j * 2048 + 1024});
        }
        noc.async_write_zeros(cbm, G * TILE, {.offset_bytes = 0});
    }
    // 2. zero what the reads do not write (complementary regions: no overlap with the reads in flight): rest of base /
    // scale-diagonal,
    //    norm-weight rows >= T, A, R, the unused rows of the last x group (absent tokens must read as 0, never garbage)
    noc.write_zeros_l1_barrier();
    // 3. constant tiles: a persistent L1 tensor backs the CB (built once by the host): just publish them
    cst.push_back(NCST);
    noc.async_read_barrier();
    STAMP(1)
    if constexpr (EXPAND) {
        // M_g[4*tl + i, t] = post_prev[t, i] (t = 8g + tl): x_new rows = x_prev rows + M_g @ Y
        volatile tt_l1_ptr uint32_t* qq = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot_base + 4608);
        volatile tt_l1_ptr uint32_t* M = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbm.get_write_ptr());
        for (uint32_t t = 0; t < T; ++t) {
            volatile tt_l1_ptr uint32_t* Mg = M + (t >> 3) * (TILE / 4);
            for (uint32_t i = 0; i < 4; ++i) {
                const uint32_t r = 4 * (t & 7) + i;
                Mg[((r >> 4) << 9) + ((t >> 4) << 8) + ((r & 15) << 4) + (t & 15)] = qq[t * 16 + 4 + i];
            }
        }
        cbxr.push_back(G * GPC);
        cby.push_back(GPC);
        cbm.push_back(G);
    } else {
        cbx.push_back(G * GPC);
    }
    cbw.push_back(GPC);
    // wave 2 (needed later: base / scale at the pre|post chain, norm weights at the end): issued now, completed before
    // exchange 1 ends
    noc.async_read(b_acc, cbb, 1024, {.page_id = 2 * k, .offset_bytes = 0}, {.offset_bytes = 0});
    noc.async_read(b_acc, cbsd, 1024, {.page_id = 2 * k + 1, .offset_bytes = 0}, {.offset_bytes = 0});
    for (uint32_t g = 0; g < GPC; ++g) {
        noc.async_read(n_acc, cbw4, T * 64, {.page_id = j0 + g, .offset_bytes = 0}, {.offset_bytes = g * TILE});
        noc.async_read(
            n_acc, cbw4, T * 64, {.page_id = j0 + g, .offset_bytes = 1024}, {.offset_bytes = g * TILE + 1024});
    }
    noc.async_write_zeros(cbb, 3072, {.offset_bytes = 1024});
    noc.async_write_zeros(cbsd, 3072, {.offset_bytes = 1024});
    for (uint32_t g = 0; g < GPC; ++g) {
        if constexpr (T < 16) {
            noc.async_write_zeros(cbw4, 1024 - T * 64, {.offset_bytes = g * TILE + T * 64});
            noc.async_write_zeros(cbw4, 1024 - T * 64, {.offset_bytes = g * TILE + 1024 + T * 64});
        }
        noc.async_write_zeros(cbw4, 2048, {.offset_bytes = g * TILE + 2048});
    }
    noc.async_write_zeros(cba, G * TILE, {.offset_bytes = 0});
    noc.async_write_zeros(cbtot, TILE, {.offset_bytes = 0});

    // ---- exchange 1: partial mixes + sum of squares ----
    cbpart.wait_front(1);
    STAMP(2)
    const uint32_t part_ptr = cbpart.get_read_ptr();
    volatile tt_l1_ptr uint32_t* pf = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(part_ptr);
    if constexpr (DBG) {
        noc_async_write(part_ptr, d_acc.get_noc_addr(k, 1024), 1024);
        noc_async_write_barrier();
    }
    pf[SWD - 1] = FLAG1;  // last word of the slot (row RR-1, column 15, unused): arrives last
    for (uint32_t d = 0; d < NC; d += (DUAL ? 2 : 1)) {
        if constexpr (ABL & 2) {
            break;
        }
        const uint32_t kk = (k + d) % NC;
        const uint64_t dst = get_noc_addr(MX0 + (kk % MW), MY0 + kk / MW, recv_base + k * SB);
        noc_async_write(part_ptr, dst, SB);
    }
    STAMP(9)
    for (uint32_t kk = 0; kk < NC; ++kk) {
        if constexpr (ABL & 2) {
            break;
        }
        // bounded spin: a lost/late flag must not wedge the device (the result is then wrong and the test catches it)
        for (uint32_t spin = 0; spin < 20000000u; ++spin) {
            invalidate_l1_cache();
            if (recv[kk * SWD + SWD - 1] == FLAG1) {
                break;
            }
        }
    }
    STAMP(3)
    noc.write_zeros_l1_barrier();
    noc.async_read_barrier();
    cbb.push_back(1);
    cbsd.push_back(1);
    cbw4.push_back(GPC);
    cbr.push_back(NR);
    noc_async_write_barrier();
    cbpart.pop_front(1);

    // ---- pre | post tile (from compute): collapse matrices, DRAM copy by core 0 ----
    cbpp.wait_front(1);
    STAMP(4) {
        volatile tt_l1_ptr uint32_t* pp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbpp.get_read_ptr());
        volatile tt_l1_ptr uint32_t* A = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cba.get_write_ptr());
        for (uint32_t t = 0; t < T; ++t) {
            volatile tt_l1_ptr uint32_t* At = A + (t >> 3) * (TILE / 4) + RW(t);
            const uint32_t c0 = 4 * (t & 7);
            for (uint32_t i = 0; i < 4; ++i) {
                const uint32_t c = c0 + i;
                At[((c >> 4) << 8) + (c & 15)] = pp[t * 16 + i];
            }
        }
    }
    cba.push_back(G);
    STAMP(5)
    if (k == 0) {
        noc.async_write(cbpp, p_acc, 1024, {.offset_bytes = 0}, {.page_id = 0, .offset_bytes = 0});
        noc.async_write_barrier();
    }
    cbpp.pop_front(1);

    // ---- exchange 2: partial sums of squares of the collapsed rows ----
    cbp2.wait_front(1);
    STAMP(6)
    volatile tt_l1_ptr uint32_t* pp2 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbp2.get_read_ptr());
    for (uint32_t t = 0; t < T; ++t) {
        slots[k * SW + t] = pp2[(((t >> 4) * 3) << 8) + ((t & 15) << 4) + (t & 15)];  // diag(HS HS^T): element (t, t)
    }
    slots[k * SW + SW - 1] = FLAG2;
    cbp2.pop_front(1);
    for (uint32_t d = 1; d < NC; d += (DUAL ? 2 : 1)) {
        if constexpr (ABL & 4) {
            break;
        }
        const uint32_t kk = (k + d) % NC;
        const uint64_t dst = get_noc_addr(MX0 + (kk % MW), MY0 + kk / MW, slot_base + k * SW * 4);
        noc_async_write(slot_base + k * SW * 4, dst, SW * 4);
    }
    STAMP(10)
    constexpr uint32_t NK = DUAL ? 16 : NC;  // with the writer kernel (DUAL) the reader waits for / transposes the
                                             // slots 0..15, the writer the slots 16..31
    for (uint32_t kk = 0; kk < NK; ++kk) {
        if constexpr (ABL & 4) {
            break;
        }
        for (uint32_t spin = 0; spin < 20000000u; ++spin) {
            invalidate_l1_cache();
            if (slots[kk * SW + SW - 1] == FLAG2) {
                break;
            }
        }
    }
    STAMP(8)
    volatile tt_l1_ptr uint32_t* R = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbtot.get_write_ptr());
    for (uint32_t t = 0; t < T; ++t) {
        volatile tt_l1_ptr uint32_t* Rt = R + RW(t);
        volatile tt_l1_ptr uint32_t* st = slots + t;
#pragma GCC unroll 16
        for (uint32_t kk = 0; kk < 16; ++kk) {
            Rt[kk] = st[kk * SW];
        }
        if constexpr (!DUAL) {
#pragma GCC unroll 16
            for (uint32_t kk = 16; kk < NC; ++kk) {
                Rt[256 + kk - 16] = st[kk * SW];
            }
        }
    }
    if constexpr (DUAL) {
        volatile tt_l1_ptr uint32_t* done =
            slots + 1792;  // slot_base + 7168 B: set by the writer once the second half of R is built
        for (uint32_t spin = 0; spin < 20000000u; ++spin) {
            invalidate_l1_cache();
            if (done[0] == FLAGD) {
                break;
            }
        }
        done[0] = 0;
    }
    cbtot.push_back(1);
    STAMP(7)
    for (uint32_t kk = 0; kk < NC; ++kk) {
        slots[kk * SW + SW - 1] = 0;
        recv[kk * SWD + SWD - 1] = 0;
    }
    noc_async_write_barrier();
    if constexpr (DBG) {
        noc_async_write(slot_base + 6144, d_acc.get_noc_addr(k, 0), 64);
        noc_async_write_barrier();
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();  // end every kernel with both NoCs drained (next kernel on these cores)
}
