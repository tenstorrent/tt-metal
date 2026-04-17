// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Reads the DFB entry_size and writes it to a specified L1 address so the host
// can verify the config was dispatched correctly.  No DFB sync operations.

#include "experimental/dataflow_buffer.h"

void kernel_main() {
    uint32_t output_l1_addr = get_arg_val<uint32_t>(0);
    uint32_t dfb_id = get_arg_val<uint32_t>(1);

    experimental::DataflowBuffer dfb(dfb_id);
    volatile uint32_t* output = reinterpret_cast<volatile uint32_t*>(output_l1_addr);
    output[0] = dfb.get_entry_size();
}
