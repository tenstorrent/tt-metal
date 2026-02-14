// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hostdevcommon/dprint_common.h"
#include "hostdev/dev_msgs.h"

#include "hostdevcommon/dprint_common.h"
// Returns the buffer address for current thread+core. Differs for NC/BR/ER/TR0-2.
inline volatile tt_l1_ptr DebugPrintMemLayout* get_debug_print_buffer() {
#if defined(ARCH_QUASAR)
#ifdef COMPILE_FOR_TRISC
    uint32_t hartid;
    uint32_t neo_id = ckernel::csr_read<ckernel::CSR::NEO_ID>();
    uint32_t trisc_id = ckernel::csr_read<ckernel::CSR::TRISC_ID>();
    hartid = 8 + 4 * neo_id + trisc_id;  // after 8 DM cores
#else
    std::uint64_t hartid;
    asm volatile("csrr %0, mhartid" : "=r"(hartid));
#endif
    return GET_MAILBOX_ADDRESS_DEV(dprint_buf.data[hartid]);
#else
    return GET_MAILBOX_ADDRESS_DEV(dprint_buf.data[PROCESSOR_INDEX]);
#endif
}

inline volatile tt_l1_ptr DevicePrintMemoryLayout* get_device_print_buffer() {
    return GET_MAILBOX_ADDRESS_DEV(dprint_buf.shared_data);
}
