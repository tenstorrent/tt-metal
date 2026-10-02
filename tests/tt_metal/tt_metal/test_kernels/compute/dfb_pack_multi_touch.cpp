// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Pack-only probe for the compact Pack DFB table and shared TC-slot pool.

#include "api/dataflow/dataflow_buffer.h"

#ifndef TEST_NUM_DFBS
#error "TEST_NUM_DFBS must be defined by the host"
#endif

#if defined(UCK_CHLKC_PACK)
#define TOUCH_DFB(I)                      \
    do {                                  \
        DataflowBuffer dfb(dfb::dfb_##I); \
        (void)dfb.get_entry_size();       \
    } while (0)
#else
#define TOUCH_DFB(I)
#endif

void kernel_main() {
#if TEST_NUM_DFBS > 0
    TOUCH_DFB(0);
#endif
#if TEST_NUM_DFBS > 1
    TOUCH_DFB(1);
#endif
#if TEST_NUM_DFBS > 2
    TOUCH_DFB(2);
#endif
#if TEST_NUM_DFBS > 3
    TOUCH_DFB(3);
#endif
#if TEST_NUM_DFBS > 4
    TOUCH_DFB(4);
#endif
#if TEST_NUM_DFBS > 5
    TOUCH_DFB(5);
#endif
#if TEST_NUM_DFBS > 6
    TOUCH_DFB(6);
#endif
#if TEST_NUM_DFBS > 7
    TOUCH_DFB(7);
#endif
#if TEST_NUM_DFBS > 8
    TOUCH_DFB(8);
#endif
#if TEST_NUM_DFBS > 9
    TOUCH_DFB(9);
#endif
#if TEST_NUM_DFBS > 10
    TOUCH_DFB(10);
#endif
#if TEST_NUM_DFBS > 11
    TOUCH_DFB(11);
#endif
#if TEST_NUM_DFBS > 12
    TOUCH_DFB(12);
#endif
#if TEST_NUM_DFBS > 13
    TOUCH_DFB(13);
#endif
#if TEST_NUM_DFBS > 14
    TOUCH_DFB(14);
#endif
#if TEST_NUM_DFBS > 15
    TOUCH_DFB(15);
#endif
#if TEST_NUM_DFBS > 16
    TOUCH_DFB(16);
#endif
#if TEST_NUM_DFBS > 17
    TOUCH_DFB(17);
#endif
#if TEST_NUM_DFBS > 18
    TOUCH_DFB(18);
#endif
#if TEST_NUM_DFBS > 19
    TOUCH_DFB(19);
#endif
#if TEST_NUM_DFBS > 20
    TOUCH_DFB(20);
#endif
#if TEST_NUM_DFBS > 21
    TOUCH_DFB(21);
#endif
#if TEST_NUM_DFBS > 22
    TOUCH_DFB(22);
#endif
#if TEST_NUM_DFBS > 23
    TOUCH_DFB(23);
#endif
}
