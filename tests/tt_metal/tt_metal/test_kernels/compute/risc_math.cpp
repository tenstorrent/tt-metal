// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/dprint.h"
#include "api/compute/common.h"
#include "dev_mem_map.h"
#include "ckernel.h"

void kernel_main() {
    const uint32_t neo_id = ckernel::csr_read<ckernel::CSR::NEO_ID>();
    const uint32_t trisc_id = ckernel::csr_read<ckernel::CSR::TRISC_ID>();
    [[maybe_unused]] const uint32_t thread_idx = NUM_TRISC_CORES * neo_id + trisc_id;
    const uint32_t l1_address = get_arg_val<uint32_t>(0);
#ifdef TRISC_PACK
    int32_t A = 1;
    int32_t B = 2;

    DPRINT << "TEST packer" << ENDL();
    const uint32_t value = A + B + thread_idx;
    DPRINT << value << ENDL();
    *((uint32_t*)(l1_address + MEM_L1_UNCACHED_BASE + thread_idx * sizeof(uint32_t))) = value;
#endif

#ifdef TRISC_UNPACK
    int32_t A = 2;
    int32_t B = 2;

    DPRINT << "TEST unpacker" << ENDL();
    const uint32_t value = A + B + thread_idx;
    DPRINT << value << ENDL();
    *((uint32_t*)(l1_address + MEM_L1_UNCACHED_BASE + thread_idx * sizeof(uint32_t))) = value;
#endif

#ifdef TRISC_MATH
    int32_t A = 3;
    int32_t B = 2;

    DPRINT << "TEST math" << ENDL();
    const uint32_t value = A + B + thread_idx;
    DPRINT << value << ENDL();
    *((uint32_t*)(l1_address + MEM_L1_UNCACHED_BASE + thread_idx * sizeof(uint32_t))) = value;
#endif
}
