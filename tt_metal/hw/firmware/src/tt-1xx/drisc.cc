// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "risc_common.h"
#include "noc.h"
#include "noc_nonblocking_api.h"
#include "internal/firmware_common.h"
#include "hostdev/dev_msgs.h"
#include "internal/risc_attribs.h"

uint8_t noc_index;

uint32_t noc_reads_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_acked[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_atomics_acked[NUM_NOCS] __attribute__((used));
uint32_t noc_posted_writes_num_issued[NUM_NOCS] __attribute__((used));

uint8_t my_x[NUM_NOCS] __attribute__((used));
uint8_t my_y[NUM_NOCS] __attribute__((used));
uint8_t my_logical_x_ __attribute__((used));
uint8_t my_logical_y_ __attribute__((used));

uint16_t dram_bank_to_noc_xy[NUM_NOCS][NUM_DRAM_BANKS] __attribute__((used));
uint16_t l1_bank_to_noc_xy[NUM_NOCS][NUM_L1_BANKS] __attribute__((used));
int32_t bank_to_dram_offset[NUM_DRAM_BANKS] __attribute__((used));
int32_t bank_to_l1_offset[NUM_L1_BANKS] __attribute__((used));

tt_l1_ptr mailboxes_t* const mailboxes = (tt_l1_ptr mailboxes_t*)(MEM_DRISC_MAILBOX_BASE);

int main() {
    configure_csr();
    do_crt1((uint32_t*)MEM_DRISC_INIT_LOCAL_L1_BASE_SCRATCH);

    noc_bank_table_init(MEM_DRISC_BANK_TO_NOC_SCRATCH);

    my_logical_x_ = mailboxes->core_info.absolute_logical_x;
    my_logical_y_ = mailboxes->core_info.absolute_logical_y;

    risc_init();

    noc_init(MEM_NOC_ATOMIC_RET_VAL_ADDR);
    for (uint32_t n = 0; n < NUM_NOCS; n++) {
        noc_local_state_init(n);
    }

    mailboxes->go_messages[0].signal = RUN_MSG_DONE;
    mailboxes->launch_msg_rd_ptr = 0;

    while (1) {
        while (mailboxes->go_messages[0].signal != RUN_MSG_GO) {
            invalidate_l1_cache();
        }

        mailboxes->go_messages[0].signal = RUN_MSG_DONE;
    }

    return 0;
}
