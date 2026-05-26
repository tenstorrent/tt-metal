// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/device_print.h"

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/common.h"
#endif
void kernel_main() {
    DEVICE_PRINT("Printing on a RISC.\n");
    DEVICE_PRINT_UNPACK("Printing on TR0.\n");
    DEVICE_PRINT_MATH("Printing on TR1.\n");
    DEVICE_PRINT_PACK("Printing on TR2.\n");
    DEVICE_PRINT_DATA0("Printing on BR.\n");
    DEVICE_PRINT_DATA1("Printing on NC.\n");
}
