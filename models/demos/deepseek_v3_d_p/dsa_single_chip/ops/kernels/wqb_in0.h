// Activation distribution of the wq_b matmul. CB 0 holds the core's K half of in0 as [kk][m] tiles (KH x MT); the
// half is split in NB blocks of KB tiles. The grid is split in four rects: {left, right} NoC-contiguous column group
// x {top, bottom} half; rect index = (column group) * 2 + half. Semaphore b flags block b.
// Block b of a rect is carried by stream (rect, b % 2) on NOC0 (even, reader RISC) or NOC1 (odd, writer RISC), sent
// by the rect's corner where that NoC's multicast starts (NOC0: min x/y, NOC1: max x/y), so packets stay inside the
// rect instead of wrapping around the torus. Each block is read from the tensor once: a NOC0 stream's sender in the
// left group (a NOC1 stream's in the right group) reads it into its own CB 0 slot and forwards it, plus flag b, to
// the same stream's sender in the other group (+x on NOC0, -x on NOC1). A sender multicasts block b (linked) and
// flag b once the block is in its L1 and flag b - 2 (the stream's previous block) is set. Each flag is written once,
// so arrival order between flags does not matter. (Atomic-increment multicast would allow one counter per stream
// but does not support a source inside the rect.)
// common runtime args used: rects at ARG_RECT (4 x [x0, y0, x1, y1, ndests]), senders at ARG_SND (4 x NB core idx),
// core NoC coordinates at ARG_COORD (x | y << 16, indexed by logical x * GRID_Y + y).
#pragma once
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

namespace wqb {

// in0 reads of the i-th block a sender owns use transaction id IN0_TRID0 + i (weight chunks use 1..IN0_TRID0-1)
constexpr uint32_t IN0_TRID0 = 8, MAX_OWNED = 16 - IN0_TRID0;

template <uint32_t NB, uint32_t KB, uint32_t MT, uint32_t KT, uint32_t COL_SPLIT, uint32_t ARG_RECT, uint32_t ARG_SND,
          uint32_t ARG_COORD>
struct In0Sender {
    static constexpr uint32_t A_TILE = 2048, BLOCK_BYTES = MT * KB * A_TILE, KH = NB * KB;
    uint32_t owned[MAX_OWNED], partner[MAX_OWNED];
    uint32_t n_owned = 0, sent = 0, in0_base = 0;
    bool reads = false;
    uint32_t xs = 0, ys = 0, xe = 0, ye = 0, nd = 0;

    // risc 0 (NOC0) sends even blocks, risc 1 (NOC1) odd blocks
    template <typename Acc>
    void start(uint32_t idx, uint32_t half, uint32_t risc, const Acc& as, uint32_t base) {
        in0_base = base;
        const uint32_t rect = (my_logical_x_ < COL_SPLIT ? 0 : 2) + half;
        for (uint32_t b = risc; b < NB && n_owned < MAX_OWNED; b += 2) {
            if (get_common_arg_val<uint32_t>(ARG_SND + rect * NB + b) == idx) {
                owned[n_owned++] = b;
            }
        }
        if (n_owned == 0) {
            return;
        }
        const uint32_t r = ARG_RECT + 5 * rect;
        const uint32_t xa = get_common_arg_val<uint32_t>(r), ya = get_common_arg_val<uint32_t>(r + 1);
        const uint32_t xb = get_common_arg_val<uint32_t>(r + 2), yb = get_common_arg_val<uint32_t>(r + 3);
        nd = get_common_arg_val<uint32_t>(r + 4) - 1;
        xs = noc_index == 0 ? xa : xb;
        ys = noc_index == 0 ? ya : yb;
        xe = noc_index == 0 ? xb : xa;
        ye = noc_index == 0 ? yb : ya;
        reads = (rect < 2) == (risc == 0);
        if (!reads) {
            return;
        }
        for (uint32_t i = 0; i < n_owned; ++i) {
            partner[i] = get_common_arg_val<uint32_t>(
                ARG_COORD + get_common_arg_val<uint32_t>(ARG_SND + (rect ^ 2) * NB + owned[i]));
            noc_async_read_set_trid(IN0_TRID0 + i);
            for (uint32_t kk = owned[i] * KB; kk < (owned[i] + 1) * KB; ++kk) {
                for (uint32_t m = 0; m < MT; ++m) {
                    noc_async_read(as.get_noc_addr(m * KT + half * KH + kk), base + (kk * MT + m) * A_TILE, A_TILE);
                }
            }
        }
        noc_async_read_set_trid(0);
    }

    bool pending() const { return sent < n_owned; }

    void step() {
        if (sent >= n_owned) {
            return;
        }
        const uint32_t block = owned[sent];
        if (block >= 2 && *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(block - 2)) == 0) {
            return;
        }
        const uint32_t src = in0_base + block * BLOCK_BYTES;
        const uint32_t flag = get_semaphore(block);
        if (reads) {
            if (!ncrisc_noc_read_with_transaction_id_flushed(noc_index, IN0_TRID0 + sent)) {
                return;
            }
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flag) = 1;
            const uint32_t px = partner[sent] & 0xFFFF, py = partner[sent] >> 16;
            noc_async_write(src, get_noc_addr(px, py, src), BLOCK_BYTES);
            noc_semaphore_set_remote(flag, get_noc_addr(px, py, flag));
        } else if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flag) == 0) {
            return;
        }
        noc_async_write_multicast(src, get_noc_multicast_addr(xs, ys, xe, ye, src), BLOCK_BYTES, nd, true);
        noc_semaphore_set_multicast(flag, get_noc_multicast_addr(xs, ys, xe, ye, flag), nd);
        ++sent;
    }
};

}  // namespace wqb
