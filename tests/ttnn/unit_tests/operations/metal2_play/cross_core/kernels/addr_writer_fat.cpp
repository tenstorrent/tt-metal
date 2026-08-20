// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t page = get_arg(args::page);
    Noc noc;
    DataflowBuffer dfb_pad(dfb::pad);
    DataflowBuffer dfb_recv(dfb::recv);
    const auto acc_out = TensorAccessor(tensor::out);

    dfb_pad.wait_front(1);
    dfb_pad.pop_front(1);

    dfb_recv.wait_front(1);
    noc.async_write(dfb_recv, acc_out, dfb_recv.get_entry_size(), {.offset_bytes = 0}, {.page_id = page});
    noc.async_write_barrier();
    dfb_recv.pop_front(1);
}
