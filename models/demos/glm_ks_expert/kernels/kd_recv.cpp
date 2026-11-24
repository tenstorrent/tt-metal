// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Stage 4 down core h receiver / chain forwarder / y writer / coordinator (BRISC, NOC1), the flat expert's
// down_core_writer:
//   * h(v) ([HK K tiles][MT rows] bfp8, buffer v % HBUF of h_all = HIN_CB) is complete on a chain head once every
//     gate/up core's slice has landed (GATH counts slices), elsewhere once its HP pieces came from the predecessor
//     (HARR counts pieces); each piece goes on to the successor (HDEPTH in flight, transaction ids) as soon as it is
//     here - the gate/up cores write h(v) only once every down core is done with h(v - HBUF), so the successor's
//     buffer is free;
//   * complete h(v) is pushed to compute; y(v) (MT x DC tiles, OUT_CB) is written to DRAM (the same pages every v);
//   * done with v (compute freed h(v), its pieces forwarded, y(v) written): the coordinator's DONE += 1; the
//     coordinator, once all ND down cores are done with v, bumps GO on every gate/up core.
// RT: 0 successor xy (0: tail), 1 h_all address, 2 head (1/0), 3 coordinator xy, 4 coordinator (1/0), 5 y address,
//     6 first y page, 7 gate/up cores (coordinator), 8.. their xy
// CT: TensorAccessorArgs(y)
#include "api/dataflow/dataflow_api.h"
#include "kd_common.hpp"

using namespace kd;

FORCE_INLINE volatile tt_l1_ptr uint32_t* sem(uint32_t id) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(id));
}
FORCE_INLINE uint64_t xy(uint32_t p, uint32_t a) { return get_noc_addr(p >> 16, p & 0xFFFF, a); }

void kernel_main() {
    constexpr auto ya = TensorAccessorArgs<0>();
    const uint32_t sxy = get_arg_val<uint32_t>(0), h_all = get_arg_val<uint32_t>(1);
    const bool head = get_arg_val<uint32_t>(2), coord = get_arg_val<uint32_t>(4);
    const uint64_t c_done = xy(get_arg_val<uint32_t>(3), get_semaphore(DONE));
    const auto ys = TensorAccessor(ya, get_arg_val<uint32_t>(5), Y_TILE);
    const uint32_t y_first = get_arg_val<uint32_t>(6);
    const uint32_t ngu = get_arg_val<uint32_t>(7);
    const uint64_t s_h = xy(sxy, h_all), s_harr = xy(sxy, get_semaphore(HARR));
    volatile tt_l1_ptr uint32_t* gath = sem(GATH);
    volatile tt_l1_ptr uint32_t* harr = sem(HARR);
    volatile tt_l1_ptr uint32_t* done_w = sem(DONE);
    constexpr uint32_t TOTAL_P = NSB * HP;
    uint32_t pub = 0, cons = 0, iss = 0, fwd = 0, ydone = 0, rep = 0, released = 0;
    auto complete = [&](uint32_t v) { return head ? *gath >= (v + 1) * NGU : *harr >= (v + 1) * HP; };
    while (rep < NSB || (coord && released < NSB)) {
        invalidate_l1_cache();
        // publish complete h buffers to compute
        while (pub < NSB && complete(pub)) {
            cb_push_back(HIN_CB, H_TILES);
            ++pub;
        }
        // forward pieces
        if (sxy) {
            while (iss < TOTAL_P && iss - fwd < HDEPTH && (head ? complete(iss / HP) : *harr > iss)) {
                const uint32_t off = ((iss / HP) % HBUF) * H_BYTES + (iss % HP) * H_PIECE;
                for (uint32_t o = 0; o < H_PIECE; o += NOC_MAX_BURST_SIZE) {
                    const uint32_t n = H_PIECE - o < NOC_MAX_BURST_SIZE ? H_PIECE - o : NOC_MAX_BURST_SIZE;
                    noc_async_write_one_packet_with_trid(h_all + off + o, s_h + off + o, n, 1 + iss % HDEPTH);
                }
                ++iss;
            }
            while (fwd < iss && !noc_cmd_buf_ready(noc_index, write_cmd_buf)) {
            }
            while (fwd < iss && ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc_index, 1 + fwd % HDEPTH)) {
                noc_semaphore_inc(s_harr, 1);
                ++fwd;
            }
        }
        // compute freed h buffers
        if (cons < pub && cb_pages_reservable_at_back(HIN_CB, (HBUF - (pub - cons) + 1) * H_TILES)) {
            ++cons;
        }
        // y
        if (ydone < NSB && cb_pages_available_at_front(OUT_CB, Y_TILES)) {
            uint32_t l1 = get_read_ptr(OUT_CB);
            for (uint32_t t = 0; t < Y_TILES; ++t) {
                noc_async_write_page(y_first + t, ys, l1);
                l1 += Y_TILE;
            }
            noc_async_write_barrier();
            cb_pop_front(OUT_CB, Y_TILES);
            ++ydone;
        }
        // done reports
        uint32_t d = cons < ydone ? cons : ydone;
        if (sxy && fwd / HP < d) {
            d = fwd / HP;
        }
        if (d > rep) {
            noc_semaphore_inc(c_done, d - rep);
            rep = d;
        }
        if (coord) {
            while (released < NSB && *done_w >= (released + 1) * ND) {
                for (uint32_t g = 0; g < ngu; ++g) {
                    noc_semaphore_inc(xy(get_arg_val<uint32_t>(8 + g), get_semaphore(GO)), 1);
                }
                ++released;
            }
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    noc_async_write_set_trid(0);
    noc_async_full_barrier();
}
