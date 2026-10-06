// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Turns v27's output OUT [L][2 parts][NBX][OUT_LEN] (per strip, column by column, H per column) back into the
// natural row-major arrays even, odd = [L, N/2, N]. One job = one (level, part): read the 11 strips' blocks,
// reorder them in L1 into N/2 natural rows, write the rows. Jobs [j_first, j_end) with stride j_step; the reader
// and the writer RISC run this kernel on alternate jobs, each with its own scratch CB. Header R is prepended.
#include "api/dataflow/dataflow_api.h"

template <typename A>
inline void paged_read(const A& s, uint32_t off, uint32_t l1, uint32_t nbytes) {
    while (nbytes) {
        const uint32_t page = off / R::PAGE, in = off % R::PAGE;
        const uint32_t n = R::PAGE - in < nbytes ? R::PAGE - in : nbytes;
        noc_async_read(s.get_noc_addr(page) + in, l1, n);
        off += n;
        l1 += n;
        nbytes -= n;
    }
}

void kernel_main() {
    uint32_t a = 0;
    const uint32_t j_first = get_arg_val<uint32_t>(a++);
    const uint32_t j_end = get_arg_val<uint32_t>(a++);
    const uint32_t j_step = get_arg_val<uint32_t>(a++);
    const uint32_t cb = get_arg_val<uint32_t>(a++);
    const uint32_t ad_out = get_arg_val<uint32_t>(a++), ad_e = get_arg_val<uint32_t>(a++),
                   ad_o = get_arg_val<uint32_t>(a++);
    constexpr auto t0 = TensorAccessorArgs<0>();
    constexpr auto t1 = TensorAccessorArgs<t0.next_compile_time_args_offset()>();
    constexpr auto t2 = TensorAccessorArgs<t1.next_compile_time_args_offset()>();
    const auto OUT = TensorAccessor(t0, ad_out, R::PAGE);
    const auto EV = TensorAccessor(t1, ad_e, R::N * 4);
    const auto OD = TensorAccessor(t2, ad_o, R::N * 4);
    constexpr uint32_t HALF = R::N / 2;
    constexpr uint32_t NP =
        (R::N * 4 + 63) & ~63u;  // L1 row pitch: every row starts 64-byte aligned like its DRAM page

    const uint32_t base = (get_write_ptr(cb) + 63) & ~63u;
    const uint32_t inbuf = base;                              // NBX x OUT_LEN
    const uint32_t natbuf = inbuf + R::NBX * R::OUT_LEN * 4;  // HALF rows x NP bytes

    for (uint32_t j = j_first; j < j_end; j += j_step) {
        const uint32_t l = j / 2, part = j % 2;
        noc_async_writes_flushed();  // natbuf free again (data has left L1)
        for (uint32_t x = 0; x < R::NBX; ++x) {
            const uint32_t nb = (R::BANDW[x] * R::H * 4 + 63) & ~63u;
            paged_read(OUT, ((l * 2 + part) * R::NBX + x) * R::OUT_LEN * 4, inbuf + x * R::OUT_LEN * 4, nb);
        }
        noc_async_read_barrier();
        for (uint32_t x = 0; x < R::NBX; ++x) {
            const uint32_t* s = (const uint32_t*)(inbuf + x * R::OUT_LEN * 4);
            const uint32_t oc0 = R::BAND0[x], w = R::BANDW[x];
            // natural row r, columns oc0 + c .. + 3 <- strip columns c .. c + 3 at row r (4 neighbouring stores)
            uint32_t c = 0;
            for (; c + 4 <= w; c += 4) {
                const uint32_t *q0 = s + c * R::H, *q1 = q0 + R::H, *q2 = q1 + R::H, *q3 = q2 + R::H;
                uint32_t* dst = (uint32_t*)natbuf + oc0 + c;
                for (uint32_t r = 0; r < HALF; ++r) {
                    const uint32_t a0 = q0[r], a1 = q1[r], a2 = q2[r], a3 = q3[r];
                    dst[0] = a0;
                    dst[1] = a1;
                    dst[2] = a2;
                    dst[3] = a3;
                    dst += NP / 4;
                }
            }
            for (; c < w; ++c) {
                const uint32_t* q = s + c * R::H;
                uint32_t* dst = (uint32_t*)natbuf + oc0 + c;
                for (uint32_t r = 0; r < HALF; ++r) {
                    *dst = q[r];
                    dst += NP / 4;
                }
            }
        }
        for (uint32_t r = 0; r < HALF; ++r) {
            const uint64_t dst = part ? OD.get_noc_addr(l * HALF + r) : EV.get_noc_addr(l * HALF + r);
            noc_async_write(natbuf + r * NP, dst, R::N * 4);
        }
    }
    noc_async_write_barrier();
}
