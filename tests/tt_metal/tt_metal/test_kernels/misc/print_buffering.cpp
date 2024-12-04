// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "debug/assert.h"

void kernel_main() {
    uint32_t core_x = get_arg_val<uint32_t>(0);
    uint32_t core_y = get_arg_val<uint32_t>(1);

    const char* msg1 = "This is a large DPRINT message that should not be interleaved with other DPRINT messages.";
    const char* msg2 = "Adding the alphabet to extend the size of this message: ABCDEFGHIJKLMNOPQRSTUVWXYZ.";
    const char* msg3 = "Now, in reverse, to make it even longer: ZYXWVUTSRQPONMLKJIHGFEDCBA.";

    ASSERT(msg1.size() + msg2.size() + msg3.size() > DPRINT_BUFFER_SIZE);
    DPRINT << "(" << core_x << "," << core_y << "): " << msg1 << " (" << core_x << "," << core_y << "): " << msg2
           << " (" << core_x << "," << core_y << "): " << msg3 << ENDL();

    const char* large_msg =
        "Once upon a time, in a small village, there was a little mouse named Tim. Tim wasn't like other mice. He was "
        "brave and curious, always venturing into places others wouldn't dare. One day, while exploring the forest, he "
        "found a big cheese trapped in a cage. Tim knew he had to help. Using his sharp teeth, he gnawed through the "
        "bars and set the cheese free. To his surprise, a kind old owl had been watching and offered him a gift - the "
        "ability to talk to all creatures. From that day on, Tim helped others, becoming a hero in the animal kingdom. "
        "And so, the little mouse learned that bravery and kindness can change the world.";

    ASSERT(large_msg > DPRINT_BUFFER_SIZE);
    DPRINT << "(" << core_x << "," << core_y << "): " << large_msg << ENDL();

    const char* msg_with_newlines =
        "This DPRINT message\n"
        "contains several newline characters\n"
        "and should be displayed over multiple lines.\n";

    DPRINT << "(" << core_x << "," << core_y << "): " << msg_with_newlines;
}
