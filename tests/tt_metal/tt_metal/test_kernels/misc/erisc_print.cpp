// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint_test_common.h"

/*
 * Test printing from a kernel running on BRISC.
 */

void kernel_main() {
    DPRINT << "Test Debug Print: ERISC" << ENDL();
    print_test_data();
}
