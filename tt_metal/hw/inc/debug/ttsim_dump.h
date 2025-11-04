// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// #include <string.h>
#ifdef TTSIM_ENABLED
#include <cstring>
#include "debug/dprint.h"
#include "debug/dprint_buffer.h"

#define TTSIM_DUMP_DST true
#define TTSIM_TENSIX_DUMP(title, dump_dst) UNPACK(ttsim_tensix_dump(title, dump_dst))

void ttsim_tensix_dump(const char* title, bool dump_dst) {
    if (title != nullptr && strlen(title) > 0) {
        // Emit title via DPRINT and wait until host consumes it
#if defined(DEBUG_PRINT_ENABLED) && !defined(FORCE_DPRINT_OFF)
        volatile tt_l1_ptr DebugPrintMemLayout* dprint_buffer = get_debug_print_buffer();
        if (dprint_buffer->aux.wpos != DEBUG_PRINT_SERVER_DISABLED_MAGIC) {
            DPRINT << title << ENDL();
            uint32_t target_wpos = dprint_buffer->aux.wpos;
            while (dprint_buffer->aux.rpos < target_wpos) {
                invalidate_l1_cache();
#if defined(COMPILE_FOR_ERISC)
                internal_::risc_context_switch();
#endif
                if (dprint_buffer->aux.wpos == DEBUG_PRINT_SERVER_DISABLED_MAGIC) {
                    break;
                }
            }
        }
#endif
    }
    if (dump_dst) {
        __asm__ volatile(".word 0x800000FF");
    } else {
        __asm__ volatile(".word 0x000000FF");
    }
}
#else
#define TTSIM_DUMP_DST
#define TTSIM_TENSIX_DUMP(title, dump_dst)
#endif
