// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "dev_mem_map.h"

// Consumer half of the QuasarDmToTriscMailbox test: UNPACK (T0), MATH (T1) and PACK (T2) each
// blocking-read their mailbox queue from writer slot IsolateSfpu/T3 -- the slot the DM writer
// kernel impersonates -- and record what arrived into their own slice of the L1 result buffer
// ([0]=UNPACK, [1]=MATH, [2]=PACK) for the host to check. ISOLATE_SFPU participates in neither
// half.
void kernel_main() {
    UNPACK({
        const std::uint32_t result_l1_addr = get_arg_val<std::uint32_t>(0);
        const std::uint32_t v = ckernel::mailbox_read(ckernel::ThreadId::IsolateSfpuThreadId);
        volatile tt_l1_ptr std::uint32_t* const result =
            reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(result_l1_addr);
        result[0] = v;
    })

    MATH({
        const std::uint32_t result_l1_addr = get_arg_val<std::uint32_t>(0);
        const std::uint32_t v = ckernel::mailbox_read(ckernel::ThreadId::IsolateSfpuThreadId);
        volatile tt_l1_ptr std::uint32_t* const result =
            reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(result_l1_addr);
        result[1] = v;
    })

    PACK({
        const std::uint32_t result_l1_addr = get_arg_val<std::uint32_t>(0);
        const std::uint32_t v = ckernel::mailbox_read(ckernel::ThreadId::IsolateSfpuThreadId);
        volatile tt_l1_ptr std::uint32_t* const result =
            reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(result_l1_addr);
        result[2] = v;
    })
}
