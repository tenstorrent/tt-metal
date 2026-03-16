// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/dprint.h"
#include "api/compute/common.h"
#include "api/compute/experimental/semaphore.h"
#include "dev_mem_map.h"
#include "ckernel.h"

void kernel_main() {
#ifdef TRISC_MATH
    const uint32_t neo_id = ckernel::csr_read<ckernel::CSR::NEO_ID>();

    const uint32_t base_src_l1_address = get_arg_val<uint32_t>(0);
    const uint32_t base_dst_l1_address = get_arg_val<uint32_t>(1);

    ckernel::Semaphore semaphore(get_compile_time_arg_val(0));
    const uint32_t base_semaphore_value = get_compile_time_arg_val(1);
    semaphore.wait(base_semaphore_value + neo_id);

    const uint32_t l1_src_address = base_src_l1_address + neo_id * sizeof(uint32_t);
    const uint32_t l1_dst_address = base_dst_l1_address + neo_id * sizeof(uint32_t);

    DPRINT << "Reading from " << l1_src_address << " and writing to " << l1_dst_address << ENDL();

    *((uint32_t*)(l1_dst_address + MEM_L1_UNCACHED_BASE)) = *((uint32_t*)(l1_src_address + MEM_L1_UNCACHED_BASE));

    semaphore.up(1);
#endif
}
