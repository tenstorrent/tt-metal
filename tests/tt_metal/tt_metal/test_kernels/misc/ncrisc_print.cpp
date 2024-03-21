// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint_test_common.h"

/*
 * Test printing from a kernel running on NCRISC.
*/

void kernel_main() {
    DPRINT_DATA1(
        // Wait for previous core (UNPACK) to finish printing.
        DPRINT << WAIT{4};
        DPRINT << "Test Debug Print: Data1" << ENDL();
        print_test_data();
    );
}
