// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/dprint.h"
#include "debug/dprint_test_common.h"

/*
 * Test kernel that wait for a signal that never raises.
*/

void kernel_main() {
    DPRINT << WAIT{1};
}
