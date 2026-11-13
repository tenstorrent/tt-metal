// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/local_tensor_accessor.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "internal/local_mem/address_access.h"

void kernel_main() {
    LocalTensorAccessor<uint32_t> report(tensor::report);
    TensorAccessor output(tensor::output);
    DataflowBuffer buf(dfb::fifo);
    Noc noc;
    for (uint32_t entry = 0; entry < 2; ++entry) {
        buf.wait_front(1);
        if (entry == 0) {
            auto lock = report.scoped_lock(2, 1);
            report[2] = buf.get_read_ptr();
        }
        noc.async_write(buf, output, buf.get_entry_size(), {}, {.page_id = get_arg(args::output_page_start) + entry});
        noc.async_write_barrier();
        buf.pop_front(1);
    }
}
