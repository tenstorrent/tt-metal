// SPDX-License-Identifier: Apache-2.0
// Pure read-only BW probe: read `nblocks` blocks of `read_bytes` from a DRAM bank at
// base_off + i*stride_bytes, double-buffered with `depth` blocks per batch (=> up to 2*depth reads in
// flight). Contiguous mode: stride_bytes==read_bytes. Strided (N-sub) mode: stride_bytes==N_band*tb,
// read_bytes==ns*tb. No compute / no cb consumer — measures raw read.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t read_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t stride_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t nblocks = get_compile_time_arg_val(2);
    constexpr uint32_t depth = get_compile_time_arg_val(3);

    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t base_off = get_arg_val<uint32_t>(1);
    const uint32_t bank_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb = 0;  // scratch ring: 2*depth * read_bytes
    const uint32_t base_l1 = get_write_ptr(cb);

    uint32_t blk = 0, region = 0;
    bool prev = false;
    while (blk < nblocks) {
        uint32_t n = (nblocks - blk < depth) ? (nblocks - blk) : depth;
        uint32_t l1 = base_l1 + region * depth * read_bytes;
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t off = base_off + (blk + i) * stride_bytes;
            uint64_t src = get_noc_addr_from_bank_id<true>(bank_id, in_addr + off);
            noc_async_read(src, l1, read_bytes);
            l1 += read_bytes;
        }
        if (prev) {
            noc_async_read_barrier();
        }
        prev = true;
        region ^= 1;
        blk += n;
    }
    noc_async_read_barrier();
}
