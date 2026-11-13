// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Hy4 iHC fused PRE kernel, writer (BRISC): builds the constant tiles of the compute kernel (selection / fold /
// identity / scale tiles with a handful of words each) and writes the normalised bf16 tiles (and, optionally, the bf16
// row-major [T,1,1,D] copy).
//

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_rm = get_compile_time_arg_val(1);
    constexpr uint32_t EMIT_RM = get_compile_time_arg_val(2);
    constexpr uint32_t T = get_compile_time_arg_val(3);
    constexpr uint32_t ROWBYTES = get_compile_time_arg_val(4);
    [[maybe_unused]] constexpr uint32_t cb_cst = get_compile_time_arg_val(5);
    [[maybe_unused]] constexpr uint32_t cb_ones = get_compile_time_arg_val(6);
    constexpr uint32_t G = get_compile_time_arg_val(7);
    constexpr uint32_t GPC = get_compile_time_arg_val(8);
    [[maybe_unused]] constexpr uint32_t EPS_BITS = get_compile_time_arg_val(9);
    [[maybe_unused]] constexpr uint32_t INV_N_BITS = get_compile_time_arg_val(10);
    constexpr auto o_args = TensorAccessorArgs<11>();
    constexpr auto m_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    constexpr auto d_args = TensorAccessorArgs<m_args.next_compile_time_args_offset()>();
    constexpr uint32_t DBG = get_compile_time_arg_val(d_args.next_compile_time_args_offset());
    constexpr uint32_t cb_dbg = get_compile_time_arg_val(d_args.next_compile_time_args_offset() + 1);
    constexpr uint32_t EXPAND = get_compile_time_arg_val(d_args.next_compile_time_args_offset() + 2);
    constexpr uint32_t cb_x = get_compile_time_arg_val(d_args.next_compile_time_args_offset() + 3);
    constexpr uint32_t NT = get_compile_time_arg_val(d_args.next_compile_time_args_offset() + 4);
    constexpr auto xo_args = TensorAccessorArgs<d_args.next_compile_time_args_offset() + 5>();
    constexpr uint32_t XB = xo_args.next_compile_time_args_offset();
    constexpr uint32_t cb_part = get_compile_time_arg_val(XB);
    constexpr uint32_t cb_recv = get_compile_time_arg_val(XB + 1);
    constexpr uint32_t cb_slot = get_compile_time_arg_val(XB + 2);
    constexpr uint32_t NC = get_compile_time_arg_val(XB + 3);
    constexpr uint32_t MX0 = get_compile_time_arg_val(XB + 4);
    constexpr uint32_t MY0 = get_compile_time_arg_val(XB + 5);
    constexpr uint32_t MW = get_compile_time_arg_val(XB + 6);
    constexpr uint32_t DUAL = get_compile_time_arg_val(
        XB + 7);  // 1: this kernel sends half of the exchange writes (the reader the other half)
    constexpr uint32_t RR = T <= 2 ? 2 : (T <= 4 ? 4 : (T <= 8 ? 8 : 16));
    constexpr uint32_t SB = RR * 64;
    constexpr uint32_t SWD = RR * 16;
    constexpr uint32_t SW = 36;
    constexpr uint32_t FLAG1 = 0x4B1C0DE1u;
    constexpr uint32_t FLAG2 = 0xC0DE1234u;
    const uint32_t k = get_arg_val<uint32_t>(0);
    const uint32_t j0 = k * GPC;
    const uint32_t o_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t m_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t d_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t xo_addr = get_common_arg_val<uint32_t>(3);
    if constexpr (DBG == 128) {
        return;
    }
    Noc noc;
    const auto o_acc = TensorAccessor(o_args, o_addr, 2048);
    const auto m_acc = TensorAccessor(m_args, m_addr, ROWBYTES);
    const auto d_acc = TensorAccessor(d_args, d_addr, 4096);
    const auto xo_acc = TensorAccessor(xo_args, xo_addr, 4096);
    experimental::CB out(cb_out), rm(cb_rm), dbgb(cb_dbg), cbx(cb_x), cbpart(cb_part), cbr(cb_recv), cbslot(cb_slot);
    volatile tt_l1_ptr uint32_t* ts = nullptr;
    if constexpr (DBG) {
        dbgb.reserve_back(1);
        ts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dbgb.get_write_ptr());
    }
#define STAMP(i)                     \
    if constexpr (DBG) {             \
        ts[i] = get_timestamp_32b(); \
    }
    STAMP(0)

    STAMP(1)
    STAMP(2)

    const uint32_t ng = GPC;
    if constexpr (EXPAND) {
        // x_new tiles from compute (cb x is never popped: the compute kernel keeps using them)
        cbx.wait_front(G * GPC);
        for (uint32_t g = 0; g < G; ++g) {
            const uint32_t rows_g = (4 * T - 32 * g) < 32 ? (4 * T - 32 * g) : 32;
            for (uint32_t j = 0; j < GPC; ++j) {
                const uint32_t pg = g * NT + j0 + j;
                const uint32_t to = (g * GPC + j) * 4096;
                if (rows_g == 32) {
                    noc.async_write(cbx, xo_acc, 4096, {.offset_bytes = to}, {.page_id = pg, .offset_bytes = 0});
                } else {
                    const uint32_t r0 = rows_g < 16 ? rows_g : 16;
                    const uint32_t r1 = rows_g - r0;
                    for (uint32_t f = 0; f < 4; ++f) {
                        const uint32_t rr = f < 2 ? r0 : r1;
                        if (rr != 0) {
                            noc.async_write(
                                cbx,
                                xo_acc,
                                rr * 64,
                                {.offset_bytes = to + f * 1024},
                                {.page_id = pg, .offset_bytes = f * 1024});
                        }
                    }
                }
            }
        }
    }
    if constexpr (DUAL) {
        // exchange 1: odd destinations (the reader does the even ones, itself included); the reader sets the flag word
        // of the part tile first
        const uint32_t part_base = cbr.get_write_ptr();
        cbpart.wait_front(1);
        const uint32_t part_ptr = cbpart.get_read_ptr();
        volatile tt_l1_ptr uint32_t* pf = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(part_ptr);
        for (uint32_t spin = 0; spin < 20000000u; ++spin) {
            invalidate_l1_cache();
            if (pf[SWD - 1] == FLAG1) {
                break;
            }
        }
        for (uint32_t d = 1; d < NC; d += 2) {
            const uint32_t kk = (k + d) % NC;
            const uint64_t dst = get_noc_addr(MX0 + (kk % MW), MY0 + kk / MW, part_base + k * SB);
            noc_async_write(part_ptr, dst, SB);
        }
        // exchange 2: the even destinations d = 2, 4, ... once the reader has built the slot (data + flag word)
        const uint32_t slot_base = cbslot.get_write_ptr();
        volatile tt_l1_ptr uint32_t* slots = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot_base);
        for (uint32_t spin = 0; spin < 40000000u; ++spin) {
            invalidate_l1_cache();
            if (slots[k * SW + SW - 1] == FLAG2) {
                break;
            }
        }
        for (uint32_t d = 2; d < NC; d += 2) {
            const uint32_t kk = (k + d) % NC;
            const uint64_t dst = get_noc_addr(MX0 + (kk % MW), MY0 + kk / MW, slot_base + k * SW * 4);
            noc_async_write(slot_base + k * SW * 4, dst, SW * 4);
        }
    }
    out.wait_front(ng);
    STAMP(3)
    for (uint32_t g = 0; g < ng; ++g) {
        noc.async_write(out, o_acc, 2048, {.offset_bytes = g * 2048}, {.page_id = j0 + g, .offset_bytes = 0});
    }
    if (EMIT_RM) {
        if constexpr (T <= 8) {
            for (uint32_t g = 0; g < ng; ++g) {
                for (uint32_t t = 0; t < T; ++t) {
                    const uint32_t src = g * 2048 + t * 32;
                    noc.async_write(
                        out, m_acc, 32, {.offset_bytes = src}, {.page_id = t, .offset_bytes = (j0 + g) * 64});
                    noc.async_write(
                        out,
                        m_acc,
                        32,
                        {.offset_bytes = src + 512},
                        {.page_id = t, .offset_bytes = (j0 + g) * 64 + 32});
                }
            }
        } else {
            rm.reserve_back(1);
            volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out.get_read_ptr());
            volatile tt_l1_ptr uint32_t* dstp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rm.get_write_ptr());
            for (uint32_t g = 0; g < ng; ++g) {
                for (uint32_t t = 0; t < T; ++t) {
                    volatile tt_l1_ptr uint32_t* tile = src + g * 512;
                    volatile tt_l1_ptr uint32_t* d = dstp + (g * T + t) * 16;
                    for (uint32_t e = 0; e < 8; ++e) {
                        d[e] = tile[((t >> 4) << 8) + ((t & 15) << 3) + e];
                        d[8 + e] = tile[128 + ((t >> 4) << 8) + ((t & 15) << 3) + e];
                    }
                }
            }
            for (uint32_t g = 0; g < ng; ++g) {
                for (uint32_t t = 0; t < T; ++t) {
                    noc.async_write(
                        rm,
                        m_acc,
                        64,
                        {.offset_bytes = (g * T + t) * 64},
                        {.page_id = t, .offset_bytes = (j0 + g) * 64});
                }
            }
        }
    }
    noc.async_write_barrier();
    STAMP(4)
    if constexpr (DBG) {
        noc_async_write(dbgb.get_write_ptr(), d_acc.get_noc_addr(get_arg_val<uint32_t>(0), 64), 320);
        noc_async_write_barrier();
    }
    out.pop_front(ng);
}
