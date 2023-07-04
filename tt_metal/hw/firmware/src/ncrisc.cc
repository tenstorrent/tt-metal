// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "risc_common.h"
#include "noc_overlay_parameters.h"
#include "noc_nonblocking_api.h"
#include "dev_msgs.h"
#include "stream_io_map.h"
#include "firmware_common.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "risc_attribs.h"
#include "generated_bank_to_noc_coord_mapping.h"
#include "circular_buffer.h"

#include "debug/status.h"

uint32_t halt_stack_ptr_save;

tt_l1_ptr mailboxes_t * const mailboxes = (tt_l1_ptr mailboxes_t *)(MEM_MAILBOX_BASE);

uint8_t my_x[NUM_NOCS] __attribute__((used));
uint8_t my_y[NUM_NOCS] __attribute__((used));

uint32_t noc_reads_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_acked[NUM_NOCS] __attribute__((used));

CBInterface cb_interface[NUM_CIRCULAR_BUFFERS] __attribute__((used));

namespace kernel_profiler {
    uint32_t wIndex __attribute__((used));
    uint32_t stackSize __attribute__((used));
    uint32_t sums[SUM_COUNT] __attribute__((used));
    uint32_t sumIDs[SUM_COUNT] __attribute__((used));
}

extern "C" void ncrisc_resume(void);
extern "C" void notify_brisc_and_halt(uint32_t status);

int main(int argc, char *argv[]) {

  DEBUG_STATUS('I');

  int32_t num_words = ((uint)__ldm_data_end - (uint)__ldm_data_start) >> 2;
  l1_to_local_mem_copy((uint*)__ldm_data_start, (uint tt_l1_ptr *)MEM_NCRISC_INIT_LOCAL_L1_BASE, num_words);

  risc_init();

  mailboxes->ncrisc_halt.resume_addr = (uint32_t)ncrisc_resume;

  // Cleanup profiler buffer incase we never get the go message
  while (1) {
      DEBUG_STATUS('W');
      notify_brisc_and_halt(RUN_SYNC_MSG_DONE);
      DeviceZoneScopedMainN("NCRISC-FW");


      setup_cb_read_write_interfaces(0, mailboxes->launch.max_cb_index, true, true);

      DEBUG_STATUS('R');
      kernel_init();
      DEBUG_STATUS('D');
  }

  return 0;
}
