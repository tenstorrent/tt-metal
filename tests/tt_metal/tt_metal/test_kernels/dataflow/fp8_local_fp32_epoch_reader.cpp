// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"

void kernel_main() {
    const uint32_t fp8_addr = get_arg_val<uint32_t>(0);
    const uint32_t fp32_addr = get_arg_val<uint32_t>(1);
    const uint32_t bf16_addr = get_arg_val<uint32_t>(2);

    Noc noc;
    AllocatorBank<AllocatorBankType::DRAM> dram;
    DataflowBuffer fp8(dfb::fp8_input);
    DataflowBuffer fp32(dfb::fp32_scale);
    DataflowBuffer bf16(dfb::bf16_input);

    fp8.reserve_back(1);
    fp32.reserve_back(1);
    bf16.reserve_back(1);
    noc.async_read(dram, fp8, fp8.get_entry_size(), {.bank_id = 0, .addr = fp8_addr}, {});
    noc.async_read(dram, fp32, fp32.get_entry_size(), {.bank_id = 0, .addr = fp32_addr}, {});
    noc.async_read(dram, bf16, bf16.get_entry_size(), {.bank_id = 0, .addr = bf16_addr}, {});
    noc.async_read_barrier();
    fp8.push_back(1);
    fp32.push_back(1);
    bf16.push_back(1);
}
