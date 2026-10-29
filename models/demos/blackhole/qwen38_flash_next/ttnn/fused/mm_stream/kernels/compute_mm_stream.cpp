// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// x52 streamed decode matmul, compute (one per core): the stock DRAM-sharded decode matmul's arithmetic for `ncols`
// output tiles of one row tile: ONE fp32 DEST accumulation over all K tiles in natural order (matmul_block, ct = ncols,
// the stock's kt_dim = in0_block_w), no spill / reload (the stock's spill is a lossless fp32 round trip), packed to
// bf16. The fidelity and fp32 DEST are the compute config's (the stock's: HiFi2 for BF8 weights, HiFi4 for BF16).
#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"
#include "../../kernels/zones.h"

template <uint32_t CT>
FORCE_INLINE void run(uint32_t cb_x, uint32_t cb_w, uint32_t cb_out, uint32_t kt, uint32_t blk) {
    CircularBuffer x(cb_x);
    CircularBuffer w(cb_w);
    CircularBuffer out(cb_out);
    FUSED_ZONE("fz_mm_c");
    matmul_block_init(cb_x, cb_w, false, CT, 1, blk);
    tile_regs_acquire();
    for (uint32_t b = 0; b < kt / blk; ++b) {
        x.wait_front(blk);
        w.wait_front(blk * CT);
        for (uint32_t r = 0; r < blk; ++r) {
            matmul_block(cb_x, cb_w, r, r * CT, 0, false, CT, 1, blk);
        }
        x.pop_front(blk);
        w.pop_front(blk * CT);
    }
    tile_regs_commit();
    out.reserve_back(CT);
    tile_regs_wait();
    pack_reconfig_data_format(cb_out);
    pack_block(0, cb_out, CT);
    tile_regs_release();
    out.push_back(CT);
}

void kernel_main() {
    constexpr uint32_t cb_x = get_named_compile_time_arg_val("cb_x");
    constexpr uint32_t cb_w = get_named_compile_time_arg_val("cb_w");
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t kt = get_named_compile_time_arg_val("kt");
    constexpr uint32_t blk = get_named_compile_time_arg_val("blk");
    const uint32_t ncols = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_out);
    if (ncols == 1) {
        run<1>(cb_x, cb_w, cb_out, kt, blk);
    } else if (ncols == 2) {
        run<2>(cb_x, cb_w, cb_out, kt, blk);
    } else if (ncols == 3) {
        run<3>(cb_x, cb_w, cb_out, kt, blk);
    } else {
        run<4>(cb_x, cb_w, cb_out, kt, blk);
    }
}
