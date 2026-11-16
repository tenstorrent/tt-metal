// wq_b reader RISC (NOC0): its share of the weight stream (wqb_wstream.h, CB 1), the even-block activation stream
// (wqb_in0.h) and, on a top core, the HEAD partial (it travels +y, which NOC0 reaches without wrapping): written to
// slot 1 of fi 0 on rank + 1, then the finalizer's semaphore NB is incremented (same VC, lands after the data).
// compile-time: [KH, NT, NCOLS, HROWS, GRID_Y, CH, MT, NB, KB, COL_SPLIT, WIN] + accessor(w) + accessor(in0)
// common runtime: [w_addr, in0_addr, out_addr, 4 x rect(5), 4 x NB sender idx, NCOLS * GRID_Y x (noc_x | noc_y << 16),
//                  NCOLS * HROWS + 1 plan offsets]
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
    constexpr uint32_t CB_IN0 = 0, CB_W = 1, CB_PSEND = 3, CB_PRECV = 4, CB_TABLE = 7;
    constexpr uint32_t W_TILE = 1088, A_TILE = 2048, O_TILE = 2048, KT = 2 * KH;
    constexpr uint32_t ARG_RECT = 3, ARG_SND = ARG_RECT + 20, ARG_COORD = ARG_SND + 4 * NB;
    constexpr uint32_t ARG_PLAN = ARG_COORD + NCOLS * GRID_Y, SEM_PART = NB, SEM_CLAIM = NB + 3;

    const auto plan = wqb::make_plan<KH, NT, NCOLS, HROWS, ARG_PLAN>();
    const uint32_t idx = static_cast<uint32_t>(my_logical_x_) * GRID_Y + my_logical_y_;
    const auto w = TensorAccessor(w_args, get_common_arg_val<uint32_t>(0), W_TILE);

    wqb::In0Sender<NB, KB, MT, KT, COL_SPLIT, ARG_RECT, ARG_SND, ARG_COORD> snd;
    snd.start(idx, plan.half, 0, TensorAccessor(in_args, get_common_arg_val<uint32_t>(1), A_TILE), get_write_ptr(CB_IN0));
    wqb::WStream<CH, WIN, CB_W, CB_TABLE, SEM_CLAIM, 0, KH, NT, decltype(w)> ws(w, plan);

    bool head = plan.half == 0 && plan.n > 0 && plan.s[0].kind == wqb::HEAD;
    uint64_t head_dst = 0, head_sem = 0;
    if (head) {
        const uint32_t trank = plan.rank + 1;
        const uint32_t v = get_common_arg_val<uint32_t>(ARG_COORD + (trank / HROWS) * GRID_Y + trank % HROWS);
        head_dst = get_noc_addr(v & 0xFFFF, v >> 16, get_write_ptr(CB_PRECV) + MT * O_TILE);
        head_sem = get_noc_addr(v & 0xFFFF, v >> 16, get_semaphore(SEM_PART));
    }
    while (ws.pending() || snd.pending() || head) {
        snd.step();
        ws.step();
        if (head && cb_pages_available_at_front(CB_PSEND, MT)) {
            noc_async_write(get_read_ptr(CB_PSEND), head_dst, MT * O_TILE);
            noc_async_writes_flushed();
            noc_semaphore_inc(head_sem, 1);
            head = false;
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
