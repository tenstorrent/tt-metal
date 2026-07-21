// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 watcher test kernel (producer) for the device-zero "zero -> barrier -> reuse" contract.
// An L1 zero borrows the overlay write command buffer on Quasar until write_zeros_l1_barrier()
// restores it; issuing any NoC write in between silently corrupts data. NOC_ASSERT_NOT_ZERO_MODE()
// catches that in watcher builds.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "api/kernel_thread_globals.h"
#include "internal/firmware_common.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t should_trip = get_arg(args::should_trip);
    const uint32_t zero_bytes = get_arg(args::zero_bytes);

    DataflowBuffer dfb(dfb::scratch);
    dfb.reserve_back(1);

    Noc noc;
    noc.async_write_zeros(dfb, zero_bytes);  // marks zero-mode active (sets the watcher flag)

    if (should_trip != 0) {
        volatile tt_l1_ptr go_msg_t* go_message_in = GET_MAILBOX_ADDRESS_DEV(go_messages[0]);
#if defined(COMPILE_FOR_DM)
        go_message_in->signal = RUN_MSG_DONE;
#else
        uint64_t dispatch_addr = calculate_dispatch_addr(go_message_in);
        notify_dispatch_core_done(dispatch_addr, noc_index);
#endif
    } else {
        noc.write_zeros_l1_barrier();  // safe: clears zero-mode before any NoC write
    }

    // Loopback write to this core's own L1 (the DFB entry). On the trip path the guard fires here
    // before the write executes; on the safe path the barrier above already cleared the flag, so
    // this is a normal, sanitizer-clean write (the data is irrelevant to the test).
    UnicastEndpoint self;
    noc.async_write(
        dfb,
        self,
        zero_bytes,
        {.offset_bytes = 0},
        {.noc_x = my_x[noc.get_noc_id()], .noc_y = my_y[noc.get_noc_id()], .addr = dfb.get_write_ptr()});
    noc.async_write_barrier();

    if (should_trip == 0) {
        dfb.push_back(1);  // hand the zeroed entry to the consumer (safe path only)
    }
}
