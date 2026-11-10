// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Megakernel AR, split-K x delivery (oneshot_ar.py OneShotARSK, spec v0.6 §12): owner j sits in-row at logical
// (10 + j % 2, j / 2) (NCRISC/NOC1 by default, so the multicast runs west inside the owner's own physical row). As soon
// as chunk j of the new residual is in its output shard (its own one-shot reduction, or the all-gather via
// mk_os_receiver.cpp):
//   1. multicast it (CT tiles) into CB_X slot j % 2 on K-row k = j / 2 only (2 physical rectangles), then x flag
//      (slot j % 2) = SEQ + 1 on the same rectangles;
//   2. once compute has the chunk's row partial P_j[r] = sum_c x[r, c]^2 x 2^-12 (fp32, CB 25), multicast the 32
//      values to CB_RSTD + j x 128 B on the 10 N-column roots, then the tag word at CB_RSTD + 2048 + j x 16 = SEQ + 1.
// FLUSH = 1: data -> flag ordering relies on in-order delivery along the same NoC multicast route (writes_flushed
// instead of a full ack barrier between data and flag, as the v12 x-delivery bench).
// PDST = 1 (MK_SK_PDST=row): the partial goes in-row instead (same K-row rectangles as x, slot p = j % 2: CB_RSTD +
// p x 128, tag at 2048 + p x 16); each (k, n) forwards its 2 partials to the column root with its column-reduce payload.
// runtime args: [j, xdst, x_rect0 (5), x_rect1 (5), xflag, rbuf, r_rect0 (5), r_rect1 (5), pslot]
#include "api/debug/kernel_profiler.hpp"
#include "mk_ar_common.hpp"

constexpr uint32_t CB_PART = 25, CB_SCALER = 30, CB_STAGE = 23;

FORCE_INLINE void sk_rects(uint32_t& a, ArRect* r) {
    for (uint32_t i = 0; i < 2; ++i) {
        r[i].x0 = get_arg_val<uint32_t>(a++); r[i].y0 = get_arg_val<uint32_t>(a++);
        r[i].x1 = get_arg_val<uint32_t>(a++); r[i].y1 = get_arg_val<uint32_t>(a++); r[i].n = get_arg_val<uint32_t>(a++);
    }
}

void kernel_main() {
    constexpr uint32_t CB_OUT_OS = get_compile_time_arg_val(0);
    constexpr uint32_t CT = get_compile_time_arg_val(1);
    constexpr uint32_t XBYTES = get_compile_time_arg_val(2);
    constexpr uint32_t SEQ = get_compile_time_arg_val(3);
    constexpr uint32_t SCALER_BF16 = get_compile_time_arg_val(4);
    constexpr bool FLUSH = get_compile_time_arg_val(5) != 0;
    constexpr bool PART = get_compile_time_arg_val(6) != 0;  // 0 (MK_SK_PART=0): plain reduction, no partials
    constexpr bool PSEND = get_compile_time_arg_val(7) != 0;  // debug: 0 (MK_SK_PSEND=0) computes but does not send
    uint32_t a = 0;
    const uint32_t j = get_arg_val<uint32_t>(a++);
    const uint32_t xdst = get_arg_val<uint32_t>(a++);
    ArRect xr[2], rr[2];
    sk_rects(a, xr);
    const uint32_t xflag = get_arg_val<uint32_t>(a++);
    const uint32_t rbuf = get_arg_val<uint32_t>(a++);
    sk_rects(a, rr);
    const uint32_t pslot = get_arg_val<uint32_t>(a++);
    DeviceZoneScopedN("OS-XALL");
    if constexpr (PART) {
    cb_reserve_back(CB_SCALER, 1);
    {
        volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_SCALER));
        for (uint32_t i = 0; i < 512; ++i) w[i] = 0;
        volatile tt_l1_ptr uint16_t* h = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(CB_SCALER));
        for (uint32_t f = 0; f < 4; ++f)
            for (uint32_t c = 0; c < 16; ++c) h[(f << 8) + c] = (uint16_t)SCALER_BF16;
    }
    cb_push_back(CB_SCALER, 1);
    }
    {
        DeviceZoneScopedN("OS-XWAIT");
        cb_wait_front(CB_OUT_OS, CT);
    }
    volatile tt_l1_ptr uint32_t* f = ar_gsem(xflag);
    {
        DeviceZoneScopedN("OS-XCHUNK");
        ar_bcast(get_read_ptr(CB_OUT_OS), xdst, XBYTES, xr, 0xFF, noc_index);
        if constexpr (FLUSH) {
            noc_async_writes_flushed();
        } else {
            noc_async_write_barrier();
        }
        *f = SEQ + 1;
        ar_flag(xflag, xr, 0xFF, noc_index);
        noc_async_writes_flushed();
    }
    if constexpr (!PART) {
        noc_async_write_barrier();
        *f = 0;
        return;
    }
    cb_wait_front(CB_PART, 1);
    {
        DeviceZoneScopedN("OS-XPART");
        // column 0 of the REDUCE_ROW fp32 tile: rows 0-15 in face 0, rows 16-31 in face 2 (16 words per face row)
        volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(CB_PART));
        const uint32_t stage = get_write_ptr(CB_STAGE);
        volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage);
        for (uint32_t r = 0; r < 16; ++r) {
            dst[r] = src[r * 16];
            dst[16 + r] = src[512 + r * 16];
        }
        dst[32] = SEQ + 1;
        if constexpr (PSEND) {
        ar_bcast(stage, rbuf + pslot * 128, 128, rr, 0xFF, noc_index);
        if constexpr (FLUSH) {
            noc_async_writes_flushed();
        } else {
            noc_async_write_barrier();
        }
        ar_bcast(stage + 128, rbuf + 2048 + pslot * 16, 16, rr, 0xFF, noc_index);
        }
        noc_async_write_barrier();
    }
    cb_pop_front(CB_PART, 1);
    *f = 0;
    noc_async_atomic_barrier();
}
