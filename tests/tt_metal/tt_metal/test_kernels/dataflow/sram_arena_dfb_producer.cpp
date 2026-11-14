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

    DataflowBuffer buf(dfb::fifo);
    const uint32_t words_per_entry = buf.get_entry_size() / sizeof(uint32_t);
    for (uint32_t entry = 0; entry < 2; ++entry) {
        buf.reserve_back(1);
        if (entry == 0) {
            auto lock = report.scoped_lock(1, 1);
            report[1] = buf.get_write_ptr();
        }
        {
            CoreLocalMem<uint32_t> words(buf.get_write_ptr());
            auto lock = words.scoped_lock(words_per_entry);
            for (uint32_t i = 0; i < words_per_entry; ++i) {
                words[i] = get_arg(args::dfb_seed) + entry * words_per_entry + i;
            }
        }
        buf.push_back(1);
    }
}
