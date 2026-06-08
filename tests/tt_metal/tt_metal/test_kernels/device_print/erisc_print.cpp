// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/device_print.h"

/*
 * Test printing from a kernel running on ERISC.
 */

void kernel_main() { DEVICE_PRINT("Test Debug Print: ERISC\n"); }
