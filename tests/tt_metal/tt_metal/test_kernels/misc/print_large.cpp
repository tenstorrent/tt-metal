// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "debug/assert.h"

/*
 * Test kernel that DPRINTs a message that is larger than what the device buffer can hold.
*/

void kernel_main() {
    uint32_t core_x = get_arg_val<uint32_t>(0);
    uint32_t core_y = get_arg_val<uint32_t>(1);

    const char* msg1 = "This is a large DPRINT message that should not be interleaved with other DPRINT messages.";
    const char* msg2 = "Adding the alphabet to extend the size of this message: ABCDEFGHIJKLMNOPQRSTUVWXYZ.";
    const char* msg3 = "Now, in reverse, to make it even longer: ZYXWVUTSRQPONMLKJIHGFEDCBA.";

    // const char* large_msg = "AAAAAA";
    ASSERT(msg1.size() + msg2.size() + msg3.size() > DPRINT_BUFFER_SIZE);
    DPRINT << "(" << core_x << "," << core_y << "): " << msg1 << " (" << core_x << "," << core_y << "): " << msg2
           << " (" << core_x << "," << core_y << "): " << msg3 << ENDL();
}
