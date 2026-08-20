// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Same topology as mcast_sender.cpp + mcast_receiver.cpp, but wired by ttnn.mcast_spec.McastFamily.
// One kernel for both roles, branching on an is_sender runtime arg. Every part of the wire -- the
// two semaphores, the rectangle, the fan-out count, the NoC corner ordering -- is decoded by
// MCAST_ARGS(bcast); this source spells no coordinate and no count.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe_spec.hpp"

void kernel_main() {
    const uint32_t page = get_arg(args::page);
    const uint32_t is_sender = get_arg(args::is_sender);

    Noc noc;
    DataflowBuffer dfb_recv(dfb::recv);
    constexpr auto mc = MCAST_ARGS(bcast);

    const uint32_t tile_bytes = dfb_recv.get_tile_size();

    dfb_recv.reserve_back(1);
    const uint32_t entry = dfb_recv.get_write_ptr();

    if (is_sender) {
        const auto acc_in = TensorAccessor(tensor::in);
        noc.async_read(acc_in, dfb_recv, tile_bytes, {.page_id = page}, {.offset_bytes = 0});
        noc.async_read_barrier();
        auto pipe = mc.sender(noc);
        if constexpr (mc.active) {
            pipe.send(entry, entry, tile_bytes);
        }
    } else {
        auto pipe = mc.receiver(noc);
        pipe.receive();
    }

    dfb_recv.push_back(1);
}
