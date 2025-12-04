// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint_test_common.h"

/*
 * Test printing from a kernel running on NCRISC.
 */

void kernel_main() {
    // Wait for BRISC to finish writing to CB, then call DPRINT
    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    cb_wait_front(cb_id_in0, 1);
    DPRINT_DATA1(DPRINT << "Test Debug Print: Data1" << ENDL(); print_test_data(););
}
