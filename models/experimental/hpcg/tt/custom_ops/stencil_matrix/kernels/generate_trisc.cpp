// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Matrix generation, the compute RISCs' part (generate.cpp has the rest): the three TRISCs of a core write most of
// the offset slots of each stencil chunk, so the data-movement RISCs, whose stores bound the generation, share the
// work five ways. Each TRISC serves both data-movement RISCs: it waits for a message (fill.h's work mailbox), writes
// its slots (DMSLOTS + t, + 3, ...) of every described run, marks the message done, and stops once both RISCs have
// signalled the end. Common argument [0]: the launch's nonce (fill.h).
#include <cstdint>
#include "api/compute/common.h"
#include "fill.h"

void kernel_main() {
#if defined(TRISC_UNPACK)
    constexpr uint32_t t = 0;
#elif defined(TRISC_MATH)
    constexpr uint32_t t = 1;
#else
    constexpr uint32_t t = 2;
#endif
    const uint32_t launch = get_common_arg_val<uint32_t>(0);
    // CB_W's L1 address from the launch's CB config table (the math thread has no CB interfaces)
    tt_l1_ptr mailboxes_t* const mbx = (tt_l1_ptr mailboxes_t*)(MEM_MAILBOX_BASE);
    launch_msg_t* lm = &(mbx->launch[mbx->launch_msg_rd_ptr]);
    const uint32_t cfg = lm->kernel_config.kernel_config_base[ProgrammableCoreType::TENSIX];
    const volatile uint32_t* cbt = (const volatile uint32_t*)(cfg + lm->kernel_config.local_cb_offset);
    volatile uint32_t* mb = (volatile uint32_t*)cbt[CB_W * UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG];
    uint32_t last[4] = {0, 0, 0, 0};
    while (true) {
        const bool ended = mb[MB_END] == launch && mb[MB_END + 16] == launch;
        for (uint32_t i = 0; i < 4; ++i) {
            const uint32_t seq = mb[16 * i];
            if (seq == 0 || seq == last[i]) {
                continue;
            }
            asm volatile("fence" ::: "memory");
            if (mb[16 * i + 4] != launch) {
                continue;  // a message slot an earlier launch left
            }
            const uint32_t n = mb[16 * i + 1];
            uint32_t* O = (uint32_t*)mb[16 * i + 2];
            const volatile uint32_t* runs = (const volatile uint32_t*)mb[16 * i + 3];
            for (uint32_t r = 0; r < n; ++r) {
                const volatile uint32_t* d = runs + r * RUN_WORDS;
                const uint32_t w0 = d[0], self0 = d[3];
                const int32_t i0 = (int32_t)d[1], len = (int32_t)d[2];
                for (uint32_t j = DMSLOTS + t; j < 27; j += 3) {
                    slot_offsets(O + j * 1024 + w0, d[4 + 3 * j], (int32_t)d[5 + 3 * j], (int32_t)d[6 + 3 * j], len,
                                 i0, self0);
                }
            }
            asm volatile("fence" ::: "memory");  // the stores are in L1 before the owner sees the message done
            mb[MB_DONE + 16 * (4 * t + i) + 1] = launch;
            asm volatile("fence" ::: "memory");  // this launch's mark before the number
            mb[MB_DONE + 16 * (4 * t + i)] = seq;
            last[i] = seq;
        }
        if (ended) {
            break;  // both RISCs waited for their last message before signalling the end
        }
    }
}
