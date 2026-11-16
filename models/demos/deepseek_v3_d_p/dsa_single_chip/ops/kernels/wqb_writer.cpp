// wq_b writer RISC (NOC1): its share of the weight stream (wqb_wstream.h, CB 5), the odd-block activation stream
// (wqb_in0.h), CB 0 pushes as activation blocks land, partial sends/receives and output writes, all driven by one
// polling loop so no duty waits behind another.
// Partials (wqb_plan.h): compute packs every non-finalized segment (all bottom segments, top HEAD) into CB 3 in
// processing order. A bottom core writes each to its finalizer's CB 4 slot ((fi * 3 + slot) * MT tiles; these travel
// -y, which NOC1 reaches without wrapping) followed by an atomic increment of the finalizer's semaphore NB + fi (same
// VC, so it lands after the data); the top HEAD goes out on the reader RISC. A finalizer pushes 3 * MT tiles of CB 4
// per finalized segment once all its partials arrived, and writes each finalized segment's MT output tiles.
// compile-time: [KH, NT, NCOLS, HROWS, GRID_Y, CH, MT, NB, KB, COL_SPLIT, WIN] + accessor(w) + accessor(in0)
//               + accessor(out)
// common runtime: as wqb_w_reader.cpp
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"
#include "wqb_plan.h"
#include "wqb_in0.h"
#include "wqb_wstream.h"

void kernel_main() {
    constexpr uint32_t KH = get_compile_time_arg_val(0), NT = get_compile_time_arg_val(1);
    constexpr uint32_t NCOLS = get_compile_time_arg_val(2), HROWS = get_compile_time_arg_val(3);
    constexpr uint32_t GRID_Y = get_compile_time_arg_val(4), CH = get_compile_time_arg_val(5);
    constexpr uint32_t MT = get_compile_time_arg_val(6), NB = get_compile_time_arg_val(7), KB = get_compile_time_arg_val(8);
    constexpr uint32_t COL_SPLIT = get_compile_time_arg_val(9), WIN = get_compile_time_arg_val(10);
    constexpr auto w_args = TensorAccessorArgs<11>();
    constexpr auto in_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    constexpr uint32_t CB_IN0 = 0, CB_OUT = 2, CB_PSEND = 3, CB_PRECV = 4, CB_W = 5, CB_TABLE = 7;
    constexpr uint32_t W_TILE = 1088, A_TILE = 2048, O_TILE = 2048, KT = 2 * KH;
    constexpr uint32_t ARG_RECT = 3, ARG_SND = ARG_RECT + 20, ARG_COORD = ARG_SND + 4 * NB;
    constexpr uint32_t ARG_PLAN = ARG_COORD + NCOLS * GRID_Y, SEM_PART = NB, SEM_CLAIM = NB + 3;
    static_assert(NB * KB == KH && SEM_CLAIM < 16);

    const auto plan = wqb::make_plan<KH, NT, NCOLS, HROWS, ARG_PLAN>();
    const uint32_t idx = static_cast<uint32_t>(my_logical_x_) * GRID_Y + my_logical_y_;
    const bool top = plan.half == 0;
    const auto w = TensorAccessor(w_args, get_common_arg_val<uint32_t>(0), W_TILE);

    cb_reserve_back(CB_IN0, MT * KH);
    cb_reserve_back(CB_PRECV, 9 * MT);
    const uint32_t precv_base = get_write_ptr(CB_PRECV);

    wqb::In0Sender<NB, KB, MT, KT, COL_SPLIT, ARG_RECT, ARG_SND, ARG_COORD> snd;
    snd.start(idx, plan.half, 1, TensorAccessor(in_args, get_common_arg_val<uint32_t>(1), A_TILE), get_write_ptr(CB_IN0));
    wqb::WStream<CH, WIN, CB_W, CB_TABLE, SEM_CLAIM, 1, KH, NT, decltype(w)> ws(w, plan);

    uint64_t send_dst[4], send_sem[4];
    uint32_t n_send = 0, fin_seg[4], n_fin = 0;
    for (uint32_t si = 0; si < plan.n; ++si) {
        const auto& sg = plan.s[si];
        if (top) {
            if (sg.kind != wqb::HEAD) {
                fin_seg[n_fin++] = si;
            }
            continue;
        }
        const uint32_t trank = sg.kind == wqb::HEAD ? plan.rank + 1 : plan.rank;
        const uint32_t fi = sg.kind == wqb::HEAD ? 0 : sg.fi;
        const uint32_t slot = sg.kind == wqb::HEAD ? 2 : 0;
        const uint32_t v = get_common_arg_val<uint32_t>(ARG_COORD + (trank / HROWS) * GRID_Y + trank % HROWS);
        const uint32_t px = v & 0xFFFF, py = v >> 16;
        send_dst[n_send] = get_noc_addr(px, py, precv_base + (fi * 3 + slot) * MT * O_TILE);
        send_sem[n_send] = get_noc_addr(px, py, get_semaphore(SEM_PART + fi));
        ++n_send;
    }
    const auto os = TensorAccessor(out_args, get_common_arg_val<uint32_t>(2), O_TILE);
    const uint32_t out_rd = get_read_ptr(CB_OUT), psend_rd = get_read_ptr(CB_PSEND);
    uint32_t in0_next = 0, sent = 0, recv = 0, outs = 0;
    while (snd.pending() || in0_next < NB || ws.pending() || sent < n_send || recv < n_fin || outs < n_fin) {
        snd.step();
        if (in0_next < NB && *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(in0_next)) != 0) {
            cb_push_back(CB_IN0, MT * KB);
            ++in0_next;
        }
        ws.step();
        if (sent < n_send && cb_pages_available_at_front(CB_PSEND, MT * (sent + 1))) {
            noc_async_write(psend_rd + sent * MT * O_TILE, send_dst[sent], MT * O_TILE);
            noc_async_writes_flushed();
            noc_semaphore_inc(send_sem[sent], 1);
            ++sent;
        }
        if (recv < n_fin) {
            const auto& sg = plan.s[fin_seg[recv]];
            if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_PART + sg.fi)) >=
                wqb::expected_partials(sg)) {
                cb_push_back(CB_PRECV, 3 * MT);
                ++recv;
            }
        }
        if (outs < n_fin && cb_pages_available_at_front(CB_OUT, MT * (outs + 1))) {
            const uint32_t col = plan.s[fin_seg[outs]].col;
            const uint32_t src = out_rd + outs * MT * O_TILE;
            for (uint32_t m = 0; m < MT; ++m) {
                noc_async_write(src + m * O_TILE, os.get_noc_addr(m * NT + col), O_TILE);
            }
            ++outs;
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
