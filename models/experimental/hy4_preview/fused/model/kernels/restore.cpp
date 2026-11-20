// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The fused step's L1 contents at rest, written again after something that is not the step wrote over them (a
// prefill pass by trace replay: fused/model/PLAN.md section 26). One instance on every core that holds any of it;
// what it writes is the host-write registry's plan (fused/model/hostwrites.py HostWrites.plan, restore.py):
//   ranges that are zero at rest           -> zeroed
//   words with a value (global semaphores) -> stored
//   buffers with contents (constants the host wrote: the MoE cores' sync words, the dense FFN's slot words, ...)
//                                          -> read from this chip's copy of them in DRAM (the restore image)
// Data movement only: writes to this core's own L1 and NoC reads from DRAM. No circular buffer, no semaphore wait,
// no fabric, no loop without a bound: it cannot stall.
//
// Compile-time args: the image's TensorAccessorArgs.
// Runtime args (per core):
//   0  the image's address
//   1  nz, then nz x (L1 address, bytes: a multiple of 4)
//   .. nw, then nw x (L1 address, value)
//   .. ns, then ns x (L1 address, bytes: a multiple of 16, first page of the image)
// Defines: RESTORE_PAGE (bytes of an image page).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto image_args = TensorAccessorArgs<0>();
    uint32_t i = 0;
    const uint32_t image_address = get_arg_val<uint32_t>(i++);

    const uint32_t nz = get_arg_val<uint32_t>(i++);
    for (uint32_t k = 0; k < nz; ++k) {
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(i++));
        const uint32_t words = get_arg_val<uint32_t>(i++) >> 2;
        for (uint32_t w = 0; w < words; ++w) {
            p[w] = 0;
        }
    }

    const uint32_t nw = get_arg_val<uint32_t>(i++);
    for (uint32_t k = 0; k < nw; ++k) {
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(i++));
        *p = get_arg_val<uint32_t>(i++);
    }

    const uint32_t ns = get_arg_val<uint32_t>(i++);
    if (ns != 0) {
        const auto image = TensorAccessor(image_args, image_address, RESTORE_PAGE);
        for (uint32_t k = 0; k < ns; ++k) {
            const uint32_t address = get_arg_val<uint32_t>(i++);
            const uint32_t bytes = get_arg_val<uint32_t>(i++);
            const uint32_t page = get_arg_val<uint32_t>(i++);
            for (uint32_t done = 0; done < bytes; done += RESTORE_PAGE) {
                const uint32_t n = bytes - done < RESTORE_PAGE ? bytes - done : RESTORE_PAGE;
                noc_async_read(image.get_noc_addr(page + done / RESTORE_PAGE), address + done, n);
            }
        }
        noc_async_read_barrier();
    }
}
