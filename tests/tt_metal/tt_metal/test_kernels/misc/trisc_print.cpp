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
    DPRINT_UNPACK(DPRINT << "Test Debug Print: Unpack" << ENDL(); print_test_data(););
    DPRINT_MATH(DPRINT << "Test Debug Print: Math" << ENDL(); print_test_data(););
    DPRINT_PACK(DPRINT << "Test Debug Print: Pack" << ENDL(); print_test_data(););
}
}  // namespace NAMESPACE
