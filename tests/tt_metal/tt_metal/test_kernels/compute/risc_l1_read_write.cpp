// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/dprint.h"
#include "api/compute/common.h"
#include "api/compute/experimental/semaphore.h"
#include "dev_mem_map.h"
#include "ckernel.h"
#include "experimental/kernel_args.h"

void kernel_main() {
#ifdef TRISC_MATH
    const uint32_t neo_id = ckernel::csr_read<ckernel::CSR::NEO_ID>();

    const uint32_t base_src_l1_address = get_arg(args::base_src_l1_address);
    const uint32_t base_dst_l1_address = get_arg(args::base_dst_l1_address);

    ckernel::Semaphore semaphore(get_arg(args::sem_id));
    const uint32_t base_semaphore_value = get_arg(args::base_semaphore_value);
    semaphore.wait(base_semaphore_value + neo_id);

    const uint32_t l1_src_address = base_src_l1_address + neo_id * sizeof(uint32_t);
    const uint32_t l1_dst_address = base_dst_l1_address + neo_id * sizeof(uint32_t);

    DPRINT << "Reading from " << l1_src_address << " and writing to " << l1_dst_address << ENDL();
    DEVICE_PRINT("Reading from {} and writing to {}\n", l1_src_address, l1_dst_address);

    *((uint32_t*)(l1_dst_address + MEM_L1_UNCACHED_BASE)) = *((uint32_t*)(l1_src_address + MEM_L1_UNCACHED_BASE));

    semaphore.up(1);
#endif
}
