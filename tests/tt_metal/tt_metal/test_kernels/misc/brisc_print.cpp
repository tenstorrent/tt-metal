// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "internal/debug/dprint_test_common.h"
#include "ckernel.h"
#include "ckernel_defs.h"

/*
 * Test printing from a kernel running on BRISC.
 */

void kernel_main() {
    // Write some data to the CB that will be used to test TSLICE.
    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    cb_reserve_back(cb_id_in0, 1);
    auto ptr = reinterpret_cast<BF16*>(get_write_ptr(cb_id_in0));
    uint16_t bfloat16_base = 0x3dfb;
    for (uint16_t idx = 0; idx < 32 * 32; idx++) {
        ptr[idx] = BF16(idx + bfloat16_base);
    }

    // Let the other threads (ncrisc) know we're done writing to CB
    cb_push_back(cb_id_in0, 1);

    // Let the trisc threads know that they can proceed
    mailbox_write(ckernel::ThreadId::MathThreadId, 1);
    mailbox_write(ckernel::ThreadId::PackThreadId, 1);
    mailbox_write(ckernel::ThreadId::UnpackThreadId, 1);

    DPRINT_DATA0(DPRINT << "Test Debug Print: Data0" << ENDL(); print_test_data(););
}
