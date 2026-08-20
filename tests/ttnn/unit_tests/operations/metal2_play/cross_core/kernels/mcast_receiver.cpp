// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Raw multicast receiver. Declared PRODUCER of `recv` and never writes it: reserve, wait for the
// sender's semaphore bump, push. This kernel is the fake producer the validator demands.
//
// It binds no tensor and no scratchpad -- its only job is to satisfy the DFB's
// exactly-one-producer-per-node rule on behalf of a remote writer.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    Noc noc;
    DataflowBuffer dfb_recv(dfb::recv);
    Semaphore ready(sem::ready);

    dfb_recv.reserve_back(1);
    // Nothing above tells the sender the entry is reservable; with num_entries == 1 and one tile
    // per program the address is the base either way. A real pipeline needs a second semaphore.
    ready.down(1);
    dfb_recv.push_back(1);
}
