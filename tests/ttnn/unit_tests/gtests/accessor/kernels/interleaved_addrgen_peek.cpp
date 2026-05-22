// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Phase 1: interleaved TensorAccessor + HW addrgen peek test kernel.
//
// Compile-time args:
//   [0] is_read       — 1 = READ binding (pop src), 0 = WRITE binding (pop dest)
//   [1] l1_scratch_base — byte offset in core L1 for page staging
// Runtime vararg [0]: num_pages
//
// Metal 2.0 binding: ta::tensor (addrgen_mode READ or WRITE set on host)

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "api/debug/device_print.h"
#include "api/core_local_mem.h"
#include "internal/tt-2xx/quasar/overlay/addrgen_api.hpp"

void kernel_main() {
    constexpr uint32_t is_read = get_arg(args::is_read);
    constexpr uint32_t l1_scratch_base = get_arg(args::l1_scratch_base);
    const uint32_t num_pages = get_vararg(0);

    if constexpr (is_read) {
        auto tensor = TensorAccessor(ta::tensor);
        const uint32_t page_size = tensor.get_aligned_page_size();

        for (uint32_t i = 0; i < num_pages; ++i) {
            const uint64_t tensor_addr = overlay::peek_src_addrgen_0();
            overlay::pop_src_addrgen_0();
            CoreLocalMem<uint32_t> local(l1_scratch_base + static_cast<uint64_t>(i) * page_size);
            DPRINT << "READ i=" << i << " tensor_addr=" << tensor_addr << " local_l1=" << local.get_address() << ENDL();
            // Future: noc.async_read(tensor, local, page_size, {}, {});
        }
    } else {
        auto tensor = TensorAccessor(ta::tensor);
        const uint32_t page_size = tensor.get_aligned_page_size();

        for (uint32_t i = 0; i < num_pages; ++i) {
            CoreLocalMem<uint32_t> local(l1_scratch_base + static_cast<uint64_t>(i) * page_size);
            const uint64_t tensor_addr = overlay::peek_dest_addrgen_0();
            overlay::pop_dest_addrgen_0();
            DPRINT << "WRITE i=" << i << " local_l1=" << local.get_address() << " tensor_addr=" << tensor_addr << ENDL();
            // Future: noc.async_write(local, tensor, page_size, {}, {});
        }
    }
}
