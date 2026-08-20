// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Raw hand-rolled multicast sender. Reads one tile from DRAM into a scratchpad, then multicasts it
// (loopback, so the sender's own `recv` entry is filled by the same transaction) into every core's
// `recv` entry, and bumps the `ready` semaphore across the receivers.
//
// Everything the model does NOT give you, in one kernel:
//   * The rectangle arrives as four plain uint32 runtime args in VIRTUAL NoC coords. Nothing types
//     them, nothing checks they cover the receiver WorkUnitSpec's nodes, and the corner ordering
//     that NOC_1 needs is the caller's problem.
//   * TWO hand-counted fan-outs for ONE rectangle: the data mcast is loopback so it counts the
//     sender, the semaphore mcast is not, so it does not. Miscount either and the sender hangs on
//     the ack.
//   * The mcast destination address is dfb_recv.get_write_ptr() -- MY entry address, reused as
//     everyone's, correct only because a DFB is allocated identically on every node.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "api/scratchpad.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t page = get_arg(args::page);
    const uint32_t dests_incl = get_arg(args::dests_incl);  // data mcast (loopback: counts me)
    const uint32_t dests_excl = get_arg(args::dests_excl);  // semaphore mcast (excludes me)
    const uint32_t x_start = get_arg(args::x_start);
    const uint32_t y_start = get_arg(args::y_start);
    const uint32_t x_end = get_arg(args::x_end);
    const uint32_t y_end = get_arg(args::y_end);

    Noc noc;
    DataflowBuffer dfb_recv(dfb::recv);
    Scratchpad<uint32_t> stage(scratch::stage);
    const auto acc_in = TensorAccessor(tensor::in);
    Semaphore ready(sem::ready);

    const uint32_t tile_bytes = dfb_recv.get_tile_size();

    noc.async_read(acc_in, stage, tile_bytes, {.page_id = page}, {.offset_bytes = 0});
    noc.async_read_barrier();

    dfb_recv.reserve_back(1);

    noc.async_write_multicast<NocOptions::MCAST_INCL_SRC>(
        stage,
        dfb_recv,
        tile_bytes,
        dests_incl,
        {.offset_bytes = 0},
        {.noc_x_start = x_start, .noc_y_start = y_start, .noc_x_end = x_end, .noc_y_end = y_end, .offset_bytes = 0});
    noc.async_write_barrier();

    if (dests_excl > 0) {
        ready.inc_multicast(noc, x_start, y_start, x_end, y_end, 1, dests_excl);
    }

    dfb_recv.push_back(1);
}
