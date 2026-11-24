// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 DM -> compute handoff over a semaphore: reads num_bytes from one DRAM bank into the resident L1 tensor
// in0, then signals the compute kernel through the shared semaphore `ready` (this kernel is its only writer).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/semaphore.h"
#include "api/tensor/local_tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    LocalTensorAccessor<std::uint32_t> dst(tensor::in0);
    AllocatorBank<AllocatorBankType::DRAM> src;
    Noc noc;
    noc.async_read(
        src,
        dst.local_mem(),
        get_arg(args::num_bytes),
        {.bank_id = get_arg(args::src_bank_id), .addr = get_arg(args::src_addr)},
        {});
    noc.async_read_barrier();  // the tensor is written before the credit
    Semaphore<> ready(sem::ready);
    ready.up(1);
}
