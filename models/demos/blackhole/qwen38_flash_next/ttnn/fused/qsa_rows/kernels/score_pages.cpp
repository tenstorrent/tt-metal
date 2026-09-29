// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// qsa_rows program 4 (qsa_score_pages): the indexer's local score rows [1, 1, rows, W >= blocks] bf16 ROW_MAJOR (one
// page per row of W elements) repaged as the all-gather's 2 KB pages [1, 1, rows * chunks, 1024] (row r's chunk c at
// page r * chunks + c), in place of the chain's slice to the resident blocks and the row-major reshape.  One
// data-movement kernel per core over a contiguous range of destination pages; pure copies, so the bytes are the
// chain's.
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

namespace {
constexpr uint32_t CB_PAGES = 0;
constexpr uint32_t CHUNKS = get_compile_time_arg_val(0);  // pages per row = blocks / 1024
constexpr uint32_t BATCH = get_compile_time_arg_val(1);   // pages per read / write round (the CB's depth)
constexpr uint32_t PAGE_BYTES = 2048;                     // 1024 bf16 block scores
}  // namespace

void kernel_main() {
    constexpr auto src_args = TensorAccessorArgs<2>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t first = get_arg_val<uint32_t>(2);      // first destination page of this core
    const uint32_t count = get_arg_val<uint32_t>(3);      // destination pages of this core
    const auto src = TensorAccessor(src_args, src_addr);  // page = one score row of W elements
    const auto dst = TensorAccessor(dst_args, dst_addr);  // page = 1024 bf16
    cb_reserve_back(CB_PAGES, BATCH);
    const uint32_t l1 = get_write_ptr(CB_PAGES);
    for (uint32_t done = 0; done < count; done += BATCH) {
        const uint32_t n = count - done < BATCH ? count - done : BATCH;
        {
            FUSED_ZONE("fz_qr_sp_read");
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t page = first + done + i;
                const uint32_t row = page / CHUNKS, chunk = page % CHUNKS;
                noc_async_read(src.get_noc_addr(row, chunk * PAGE_BYTES), l1 + i * PAGE_BYTES, PAGE_BYTES);
            }
            noc_async_read_barrier();
        }
        {
            FUSED_ZONE("fz_qr_sp_write");
            for (uint32_t i = 0; i < n; ++i) {
                noc_async_write(l1 + i * PAGE_BYTES, dst.get_noc_addr(first + done + i, 0), PAGE_BYTES);
            }
            noc_async_write_barrier();
        }
    }
}
