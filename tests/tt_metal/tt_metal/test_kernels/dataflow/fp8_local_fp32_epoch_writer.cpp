// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"

void kernel_main() {
    const uint32_t fp32_addr = get_arg_val<uint32_t>(0);
    const uint32_t bf16_addr = get_arg_val<uint32_t>(1);

    Noc noc;
    AllocatorBank<AllocatorBankType::DRAM> dram;
    DataflowBuffer fp32(dfb::fp32_output);
    DataflowBuffer bf16(dfb::bf16_output);

    fp32.wait_front(1);
    noc.async_write(fp32, dram, fp32.get_entry_size(), {}, {.bank_id = 0, .addr = fp32_addr});
    noc.async_write_barrier();
    fp32.pop_front(1);

    bf16.wait_front(1);
    noc.async_write(bf16, dram, bf16.get_entry_size(), {}, {.bank_id = 0, .addr = bf16_addr});
    noc.async_write_barrier();
    bf16.pop_front(1);
}
