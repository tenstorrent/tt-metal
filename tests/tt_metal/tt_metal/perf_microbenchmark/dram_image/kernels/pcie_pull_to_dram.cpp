// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Pull pages of a DRAM image from NOC-mapped pinned host memory and write them into an interleaved DRAM buffer.
// Runtime arg 6 is unused. Core c handles pages first_page, first_page + stride, ...; page j lives at pinned offset j *
// page_size and goes to page j of the destination buffer. The image is split into pinned chunks; page_off is this
// chunk's first page. Two L1 slots: the read of page k overlaps the DRAM write of page k - 1.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t pcie_xy_enc = get_compile_time_arg_val(0);
    constexpr uint32_t page_size = get_compile_time_arg_val(1);
    constexpr uint32_t burst = get_compile_time_arg_val(2);  // <= NOC_MAX_BURST_SIZE, divides page_size
    constexpr auto dst_args = TensorAccessorArgs<3>();

    const uint32_t src_lo = get_arg_val<uint32_t>(0);
    const uint32_t src_hi = get_arg_val<uint32_t>(1);
    const uint32_t dst_addr = get_arg_val<uint32_t>(2);
    const uint32_t first_page = get_arg_val<uint32_t>(3);
    const uint32_t stride = get_arg_val<uint32_t>(4);
    const uint32_t num_pages = get_arg_val<uint32_t>(5);
    // two page slots in CB 0, which the host sizes to 2 * page_size (the allocator keeps it clear of live L1 buffers)
    const uint32_t l1_base = get_write_ptr(0);
    const uint32_t page_off = get_arg_val<uint32_t>(7);  // first page of this pinned chunk

    const uint64_t src_base = (static_cast<uint64_t>(src_hi) << 32) | src_lo;
    const auto dst = TensorAccessor(dst_args, dst_addr);

    for (uint32_t k = 0; k < num_pages; k++) {
        const uint32_t page = first_page + k * stride;
        const uint32_t slot = l1_base + (k & 1) * page_size;
        if (k >= 2) {
            noc_async_writes_flushed();  // slot's previous page (k - 2) has left L1
        }
        uint64_t src = src_base + static_cast<uint64_t>(page - page_off) * page_size;
        for (uint32_t off = 0; off < page_size; off += burst) {
            noc_read_with_state<noc_mode, read_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT>(
                noc_index, pcie_xy_enc, src + off, slot + off, burst);
        }
        noc_async_read_barrier();
        noc_async_write(slot, dst.get_noc_addr(page), page_size);
    }
    noc_async_write_barrier();
}
