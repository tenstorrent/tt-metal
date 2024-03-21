// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/common.h"
#include "debug/dprint_test_common.h"

/*
 * Test printing from a kernel running on TRISC.
*/
namespace NAMESPACE {
void MAIN {
    DPRINT_UNPACK(
        // Wait for previous core (DATA0) to finish printing.
        DPRINT << WAIT{1};
        DPRINT << "Test Debug Print: Unpack" << ENDL();
        print_test_data();
        // Let the next core (MATH) know to start printing.
        DPRINT << RAISE{2};
    );
    DPRINT_MATH(
        // Wait for previous core (DATA0) to finish printing.
        DPRINT << WAIT{2};
        DPRINT << "Test Debug Print: Math" << ENDL();
        print_test_data();
        // Let the next core (PACK) know to start printing.
        DPRINT << RAISE{3};
    );
    DPRINT_PACK(
        // Wait for previous core (DATA0) to finish printing.
        DPRINT << WAIT{3};
        DPRINT << "Test Debug Print: Pack" << ENDL();
        print_test_data();
        // Let the next core (DATA0) know to start printing.
        DPRINT << RAISE{4};
    );
}
}
