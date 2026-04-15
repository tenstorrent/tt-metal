// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "internal/debug/dprint_test_common.h"

/*
 * Test printing from a kernel running on a DRAM programmable core (DRISC).
 */

void kernel_main() {
    DPRINT << "Test Debug Print: DRISC" << ENDL();
    print_test_data();
}
