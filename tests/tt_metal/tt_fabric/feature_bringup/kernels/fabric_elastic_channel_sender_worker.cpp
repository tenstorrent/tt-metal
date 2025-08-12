// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

// What do I need to get the next chunk:
// address? (could be an ID instead, in which case it could be packed in a single word)

// What happens if a worker is granted a chunk but it doesn't use it and then it needs to be relinquished to the pool?


// protocol is as follows:

// 1 bit to indicate new value
// value to indicate rest of it

struct SenderChannelView {
    static constexpr uint32_t NEXT_CHUNK_VALID = 1 << 31;
    static constexpr uint32_t NEXT_CHUNK_VALUE_MASK = NEXT_CHUNK_VALID - 1;
    volatile uint32_t *next_chunk_ptr;

    void wait_for_new_chunk() {
        while (!*next_chunk_ptr) {
        }
    }

    uint32_t get_next_chunk() {
        uint32_t value = *next_chunk_ptr;
        return value & NEXT_CHUNK_VALUE_MASK;
    }
};



struct FabricWriterAdapter {
    SenderChannelView sender_channel_view;
    OnePassIterator<
    
};


void kernel_main() {

    size_t n_pkts = get_arg_val<size_t>(0);
    

    for (size_t i = 0; i < n_pkts; i++) {



    }


}
