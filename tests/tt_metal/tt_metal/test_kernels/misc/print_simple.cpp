// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/dprint.h"

#if defined(COMPILE_FOR_TRISC)
#include "compute_kernel_api/common.h"
namespace NAMESPACE {
void MAIN {
#else
void kernel_main() {
#endif
    DPRINT << "Printing on a RISC." << ENDL();
    DPRINT_UNPACK(DPRINT << "Printing on TR0." << ENDL(););
    DPRINT_MATH(DPRINT << "Printing on TR1." << ENDL(););
    DPRINT_PACK(DPRINT << "Printing on TR2." << ENDL(););
    DPRINT_DATA0(DPRINT << "Printing on BR." << ENDL(););
    DPRINT_DATA1(DPRINT << "Printing on NC." << ENDL(););
}
#if defined(COMPILE_FOR_TRISC)
}
#endif
