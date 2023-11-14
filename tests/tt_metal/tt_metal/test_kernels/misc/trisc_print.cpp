// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug_print.h"
#include "compute_kernel_api/common.h"

/*
 * Test printing from a kernel running on TRISC.
*/
namespace NAMESPACE {
void MAIN {
    DPRINT_UNPACK(DPRINT << "Test Debug Print: Unpack" << ENDL();)
    DPRINT_MATH(DPRINT << "Test Debug Print: Math" << ENDL();)
    DPRINT_PACK(DPRINT << "Test Debug Print: Pack" << ENDL();)
}
}
