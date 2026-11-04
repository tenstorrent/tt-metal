// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DFlash2 rung-3 owner core, BRISC (NoC0): per phase the DRAM inputs (and the fabric lanes' packed all-reduce rows)
// into the compute's CBs, the outputs to DRAM, the readers' hand-off (df2_mk_common.hpp documents the phases, the
// argument layout and the semaphores).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/spec/df2_mk_common.hpp"

using namespace df2mk;

namespace {

constexpr uint32_t NPH = get_compile_time_arg_val(CT_NPH);
constexpr uint32_t NRD = get_compile_time_arg_val(CT_NRD);
constexpr uint32_t NQH = get_compile_time_arg_val(CT_NQH);
constexpr uint32_t NKVH = get_compile_time_arg_val(CT_NKVH);
constexpr uint32_t HT = get_compile_time_arg_val(CT_HT);
constexpr uint32_t GPT = get_compile_time_arg_val(CT_GPT);
constexpr uint32_t NG = get_compile_time_arg_val(CT_NG);
constexpr uint32_t IT = get_compile_time_arg_val(CT_IT);
constexpr uint32_t NTS = get_compile_time_arg_val(CT_NTS);
constexpr uint32_t FABV = get_compile_time_arg_val(CT_FABV);
constexpr uint32_t FABCH = get_compile_time_arg_val(CT_FABCH);
constexpr uint32_t RD0 = 10;             // the readers' NoC (x, y) pairs
constexpr uint32_t PB0 = RD0 + 2 * NRD;  // the phase blocks
constexpr uint32_t NH_ALL = NQH + 2 * NKVH;
constexpr uint32_t GQ = NQH / NKVH;
static_assert(NQH % NKVH == 0, "whole GQA groups per chip");
static_assert(NH_ALL <= N_OWNERS, "one head per owner");

FORCE_INLINE uint64_t dram(uint32_t addr, uint32_t page_size, uint32_t page) {
    const InterleavedAddrGen<true> g = {.bank_base_address = addr, .page_size = page_size};
    return g.get_noc_addr(page);
}

FORCE_INLINE void zero_l1(uint32_t l1, uint32_t bytes) {
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t off = 0; off < bytes; off += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, l1 + off, MEM_ZEROS_SIZE);
    }
}

// the all-reduced tile c's packed valid rows (face-0 rows 0..7, then face-1 rows: tt/fabric_ar.py) from its lane's
// result buffer into the (zeroed) bf16 tile at l1
FORCE_INLINE void ar_rows(uint32_t c, uint32_t l1) {
    const uint32_t lane = c / 64;
    const uint32_t lx = get_arg_val<uint32_t>(5 + 2 * lane), ly = get_arg_val<uint32_t>(6 + 2 * lane);
    const uint64_t src = get_noc_addr(lx, ly, get_arg_val<uint32_t>(9) + (c % 64) * FABV);
    noc_async_read(src, l1, FABCH);
    noc_async_read(src + FABCH, l1 + 512, FABCH);
}

FORCE_INLINE void wait_ar(uint32_t ordinal) { mk::wait_handoff((ordinal & 1) ? SEM_AR2 : SEM_AR, 2); }

// n bf16 tiles at l1 -> pages c0 .. c0 + n of the tap tensor `tap` (0 = none); the caller barriers
FORCE_INLINE void tap_tiles(uint32_t tap, uint32_t l1, uint32_t c0, uint32_t n) {
    if (tap == 0) {
        return;
    }
    for (uint32_t j = 0; j < n; ++j) {
        noc_async_write(l1 + j * TB, dram(tap, TB, c0 + j), TB);
    }
}

FORCE_INLINE void signal_readers() { mk::signal_handoff(SEM_R, RD0, NRD); }

// RMS (src = a DRAM bf16 [32, H] tensor) / RMS_AR (src = the fabric all-reduce result): x tiles c0, c0 + 1 and the
// norm weight row tiles in; the normed tiles out (+ the readers' hand-off unless flag bit 4 is clear)
FORCE_INLINE void ph_rms(uint32_t rb, uint32_t c0, bool from_ar, uint32_t flag) {
    const uint32_t src = get_arg_val<uint32_t>(rb), w = get_arg_val<uint32_t>(rb + 1),
                   out = get_arg_val<uint32_t>(rb + 2);
    cb_reserve_back(CB_X, 2);
    cb_reserve_back(CB_W, 2);
    const uint32_t xl = get_write_ptr(CB_X), wl = get_write_ptr(CB_W);
    for (uint32_t j = 0; j < 2; ++j) {
        noc_async_read(dram(w, TB32, c0 + j), wl + j * TB32, TB32);
    }
    if (from_ar) {
        zero_l1(xl, 2 * TB);
        noc_async_read_barrier();  // (the zero fill lands before the rows: NoC reads may complete out of order)
        wait_ar(flag & 0xF);
        for (uint32_t j = 0; j < 2; ++j) {
            ar_rows(c0 + j, xl + j * TB);
        }
    } else {
        for (uint32_t j = 0; j < 2; ++j) {
            noc_async_read(dram(src, TB, c0 + j), xl + j * TB, TB);
        }
    }
    noc_async_read_barrier();
    if (from_ar) {  // tap: the assembled all-reduce rows (the ctx fc output as the owners see it)
        tap_tiles(get_arg_val<uint32_t>(rb + RT_TAP2), xl, c0, 2);
        noc_async_write_barrier();
    }
    cb_push_back(CB_X, 2);
    cb_push_back(CB_W, 2);

    cb_wait_front(CB_OUT, 2);
    const uint32_t ol = get_read_ptr(CB_OUT);
    for (uint32_t j = 0; j < 2; ++j) {
        noc_async_write(ol + j * TB, dram(out, TB, c0 + j), TB);
    }
    tap_tiles(get_arg_val<uint32_t>(rb + RT_TAP), ol, c0, 2);
    noc_async_write_barrier();
    cb_pop_front(CB_OUT, 2);
    if (flag & 0x10) {
        signal_readers();
    }
}

// CONV (x = a DRAM bf16 tensor, side; the cf from the kp readers: SEM_CF) / FIN (x = the all-reduced partial's packed
// rows, side 1, + the residual tiles; out = the residual, in place): the rung-2 reader's reads per column tile
FORCE_INLINE void ph_conv(uint32_t rb, uint32_t c0, uint32_t side, bool fin, uint32_t ordinal) {
    const uint32_t x = get_arg_val<uint32_t>(rb), cf = get_arg_val<uint32_t>(rb + 1);
    const uint32_t out = get_arg_val<uint32_t>(rb + 2), b4 = get_arg_val<uint32_t>(rb + 3);
    const uint32_t e = get_arg_val<uint32_t>(2), jt = get_arg_val<uint32_t>(3), sh = get_arg_val<uint32_t>(4);
    // the constants of this call
    cb_reserve_back(CB_J, 2);
    cb_reserve_back(CB_SH, 1);
    cb_reserve_back(CB_ZERO, 1);
    noc_async_read(dram(jt, TB, 2 * side), get_write_ptr(CB_J), TB);
    noc_async_read(dram(jt, TB, 2 * side + 1), get_write_ptr(CB_J) + TB, TB);
    noc_async_read(dram(sh, TB, 0), get_write_ptr(CB_SH), TB);
    zero_l1(get_write_ptr(CB_ZERO), TB);
    noc_async_read_barrier();
    cb_push_back(CB_J, 2);
    cb_push_back(CB_SH, 1);
    cb_push_back(CB_ZERO, 1);
    if (!fin) {
        mk::wait_handoff(SEM_CF, NRD);  // the kp readers' cf landed
    }
    cb_reserve_back(CB_X, 2);
    cb_reserve_back(CB_CF, 4);
    cb_reserve_back(CB_E, 4);
    cb_reserve_back(CB_B, 2);
    const uint32_t xl = get_write_ptr(CB_X), cfl = get_write_ptr(CB_CF), el = get_write_ptr(CB_E);
    const uint32_t bl = get_write_ptr(CB_B);
    uint32_t rl = 0;
    if (fin) {
        cb_reserve_back(CB_RES, 2);
        rl = get_write_ptr(CB_RES);
        zero_l1(xl, 2 * TB);
        noc_async_read_barrier();
        wait_ar(ordinal);
    }
    for (uint32_t j = 0; j < 2; ++j) {
        const uint32_t c = c0 + j;
        if (fin) {
            ar_rows(c, xl + j * TB);
            noc_async_read(dram(x, TB, c), rl + j * TB, TB);  // the residual (x = the residual tensor)
        } else {
            noc_async_read(dram(x, TB, c), xl + j * TB, TB);
        }
        noc_async_read(dram(b4, TB, c), bl + j * TB, TB);
        for (uint32_t t = 0; t < 2; ++t) {
            const uint32_t q0 = side * 2 * NG + t * NG + c * GPT;
            noc_async_read(dram(cf, TB, q0 >> 5), cfl + (2 * j + t) * TB, TB);
            noc_async_read(dram(e, TB, q0 & 31u), el + (2 * j + t) * TB, TB);
        }
    }
    noc_async_read_barrier();
    if (fin) {  // tap: the assembled all-reduce rows (the o / down result as the owners see it)
        tap_tiles(get_arg_val<uint32_t>(rb + RT_TAP2), xl, c0, 2);
        noc_async_write_barrier();
    }
    cb_push_back(CB_X, 2);
    cb_push_back(CB_CF, 4);
    cb_push_back(CB_E, 4);
    cb_push_back(CB_B, 2);
    if (fin) {
        cb_push_back(CB_RES, 2);
    }

    cb_wait_front(CB_OUT, 2);
    const uint32_t ol = get_read_ptr(CB_OUT);
    for (uint32_t j = 0; j < 2; ++j) {
        noc_async_write(ol + j * TB, dram(out, TB, c0 + j), TB);
    }
    tap_tiles(get_arg_val<uint32_t>(rb + RT_TAP), ol, c0, 2);
    noc_async_write_barrier();
    cb_pop_front(CB_OUT, 2);
    if (!fin) {
        signal_readers();
    }
}

// HEADS: wait for the i-th q|k|v hand-off of the program (cumulative count; reset after the last), then head h = k:
// q / k heads -> the compute (4 tiles + the norm weight rows + cos / sin); a q head's RoPE'd tiles -> its kv group's
// SDPA input tensor; v heads -> CB_V (the NCRISC writes the cache rows)
FORCE_INLINE void ph_heads(uint32_t rb, uint32_t k, uint32_t flag) {
    const uint32_t i = flag & 0xFF, last = (flag >> 8) & 1u, noq = (flag >> 9) & 1u;
    {
        volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_QKV));
        noc_semaphore_wait_min(s, NRD * (i + 1));
        if (last) {
            noc_semaphore_set(s, 0);
        }
    }
    if (k >= NH_ALL) {
        return;
    }
    const uint32_t h = k;
    const uint32_t qkv = get_arg_val<uint32_t>(rb);
    const bool isq = h < NQH, isk = !isq && h < NQH + NKVH;
    if (isq && noq) {
        return;  // the ctx program: the q heads are not needed
    }
    if (isq || isk) {
        const uint32_t w = get_arg_val<uint32_t>(rb + (isq ? 1 : 2));
        const uint32_t cos = get_arg_val<uint32_t>(rb + 3), sin = get_arg_val<uint32_t>(rb + 4);
        cb_reserve_back(CB_QKV, HT);
        cb_reserve_back(CB_HW, HT);
        cb_reserve_back(CB_COS, HT);
        cb_reserve_back(CB_SIN, HT);
        const uint32_t ql = get_write_ptr(CB_QKV), wl = get_write_ptr(CB_HW);
        const uint32_t cl = get_write_ptr(CB_COS), sl = get_write_ptr(CB_SIN);
        for (uint32_t j = 0; j < HT; ++j) {
            noc_async_read(dram(qkv, TB, h * HT + j), ql + j * TB, TB);
            noc_async_read(dram(w, TB32, j), wl + j * TB32, TB32);
            noc_async_read(dram(cos, TB, j), cl + j * TB, TB);
            noc_async_read(dram(sin, TB, j), sl + j * TB, TB);
        }
        noc_async_read_barrier();
        cb_push_back(CB_QKV, HT);
        cb_push_back(CB_HW, HT);
        cb_push_back(CB_COS, HT);
        cb_push_back(CB_SIN, HT);
        if (isq) {  // the SDPA input: tile row h of ONE [1, NQH, 32 rows, hd] tensor (tt/dflash2_mk.py: batch = q head,
                    // every row's blocks through a per-q-head page table; the PB_Q fields all name that tensor)
            const uint32_t qo = get_arg_val<uint32_t>(rb + PB_Q);
            cb_wait_front(CB_OUT, HT);
            const uint32_t ol = get_read_ptr(CB_OUT);
            for (uint32_t j = 0; j < HT; ++j) {
                noc_async_write(ol + j * TB, dram(qo, TB, h * HT + j), TB);
            }
            {  // tap: the q head's tiles -> pages h HT + j of a [32, NQH hd] tensor
                const uint32_t tap = get_arg_val<uint32_t>(rb + RT_TAP);
                if (tap != 0) {
                    for (uint32_t j = 0; j < HT; ++j) {
                        noc_async_write(ol + j * TB, dram(tap, TB, h * HT + j), TB);
                    }
                }
            }
            noc_async_write_barrier();
            cb_pop_front(CB_OUT, HT);
        }
    } else {
        cb_reserve_back(CB_V, HT);
        const uint32_t vl = get_write_ptr(CB_V);
        for (uint32_t j = 0; j < HT; ++j) {
            noc_async_read(dram(qkv, TB, h * HT + j), vl + j * TB, TB);
        }
        noc_async_read_barrier();
        cb_push_back(CB_V, HT);
    }
}

// SWI: the up | gate tiles c0 .. c0 + NTS of this owner (up tile c, gate tile IT + c) -> m tiles c0 ..
FORCE_INLINE void ph_swi(uint32_t rb, uint32_t k) {
    mk::wait_handoff(SEM_UG, NRD);
    const uint32_t gu = get_arg_val<uint32_t>(rb), m = get_arg_val<uint32_t>(rb + 1);
    const uint32_t c0 = k * NTS;
    if (c0 < IT) {
        cb_reserve_back(CB_GU, 2 * NTS);
        const uint32_t gl = get_write_ptr(CB_GU);
        for (uint32_t j = 0; j < NTS; ++j) {
            noc_async_read(dram(gu, TB, c0 + j), gl + j * TB, TB);
            noc_async_read(dram(gu, TB, IT + c0 + j), gl + (NTS + j) * TB, TB);
        }
        noc_async_read_barrier();
        cb_push_back(CB_GU, 2 * NTS);
        cb_wait_front(CB_OUT, NTS);
        const uint32_t ol = get_read_ptr(CB_OUT);
        for (uint32_t j = 0; j < NTS; ++j) {
            noc_async_write(ol + j * TB, dram(m, TB, c0 + j), TB);
        }
        tap_tiles(get_arg_val<uint32_t>(rb + RT_TAP), ol, c0, NTS);
        noc_async_write_barrier();
        cb_pop_front(CB_OUT, NTS);
    }
    signal_readers();
}

template <uint32_t P>
FORCE_INLINE void run(uint32_t k, uint32_t c0) {
    if constexpr (P < NPH) {
        constexpr uint32_t TYPE = get_compile_time_arg_val(CT_PH0 + 2 * P);
        constexpr uint32_t FLAG = get_compile_time_arg_val(CT_PH0 + 2 * P + 1);
        constexpr uint32_t rb = PB0 + RT_W * P;
        if constexpr (TYPE == PH_RMS) {
            ph_rms(rb, c0, false, FLAG);
        } else if constexpr (TYPE == PH_RMS_AR) {
            ph_rms(rb, c0, true, FLAG);
        } else if constexpr (TYPE == PH_CONV) {
            ph_conv(rb, c0, FLAG, false, 0);
        } else if constexpr (TYPE == PH_FIN) {
            ph_conv(rb, c0, FLAG >> 4, true, FLAG & 0xF);
        } else if constexpr (TYPE == PH_HEADS) {
            ph_heads(rb, k, FLAG);
        } else {
            ph_swi(rb, k);
        }
        mk::cb_rewind_range(
            0, N_CBS);  // every chunk of this RISC's phase is complete: the next phase starts at the base
        run<P + 1>(k, c0);
    }
}

}  // namespace

void kernel_main() {
    const uint32_t k = get_arg_val<uint32_t>(0);
    const uint32_t c0 = get_arg_val<uint32_t>(1);
    run<0>(k, c0);
}
