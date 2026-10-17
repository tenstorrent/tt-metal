// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
//
// SPDX-License-Identifier: Apache-2.0

// Test responder: for every new req_seq r of the channel, write reply[i] = r * 64 + i (16 words), fence, publish
// done_seq = r (data before the sequence number), drain the MSI catcher FIFO. Polls req_seq (the doorbell is not
// needed to make progress; the FIFO is drained so it never stays full). Stops when the stop word is non-zero.
#include <stdint.h>

#include "../kernels/l2cpu_link.h"

#define R32(a) (*(volatile uint32_t*)(uintptr_t)(a))
#define R64(a) (*(volatile uint64_t*)(uintptr_t)(a))

void responder_main(uint64_t image) {
    uint64_t b = image - L2CPU_LINK_IMAGE_OFFSET;
    uint32_t done = R32(b + L2CPU_LINK_OFF_DONE_SEQ);
    uint64_t served = 0, polls = 0;
    R64(b + L2CPU_LINK_OFF_RESP_STATUS + 8) = 0;
    __asm__ volatile("fence rw, rw" ::: "memory");
    R32(b + L2CPU_LINK_OFF_RESP_STATUS) = L2CPU_LINK_RESP_MAGIC;
    for (;;) {
        uint32_t req = R32(b + L2CPU_LINK_OFF_REQ_SEQ);
        if (req == done) {
            if ((++polls & 0x3ff) == 0 && R32(b + L2CPU_LINK_OFF_STOP)) {
                break;
            }
            continue;
        }
        __asm__ volatile("fence r, r" ::: "memory");
        for (uint32_t i = 0; i < L2CPU_LINK_REPLY_WORDS; i++) {
            R32(b + L2CPU_LINK_OFF_REPLY + 4 * i) = req * 64 + i;
        }
        __asm__ volatile("fence w, w" ::: "memory");
        R32(b + L2CPU_LINK_OFF_DONE_SEQ) = req;
        done = req;
        R64(b + L2CPU_LINK_OFF_RESP_STATUS + 8) = ++served;
        __asm__ volatile("fence rw, rw" ::: "memory");
        while (R32(L2CPU_BH_MSI_STATUS) & L2CPU_BH_MSI_STATUS_NONEMPTY) {
            (void)R32(L2CPU_BH_MSI_FIFO);
        }
    }
    R32(b + L2CPU_LINK_OFF_RESP_STATUS) = 0;
}
