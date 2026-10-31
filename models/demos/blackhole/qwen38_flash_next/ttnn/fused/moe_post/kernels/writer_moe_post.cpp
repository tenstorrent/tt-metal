// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// MoE post program, writer: the local sum's column tile into the reduce-scatter's input.  The fold build (MP_FOLD) also
// runs the zero seeding of the activation tiles and the sigmoid tile read on this core's second data-movement
// processor, which is idle until the sum is ready: the reader would wait for the 80 cores' reads of one DRAM page and
// for 40 KB of zero writes in its setup, now both overlap the reader's routing work.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

#include "../../kernels/zones.h"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t tile_col = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t tile_bytes = 2048;
    constexpr auto out_args = TensorAccessorArgs<0, 0>();
    const auto out = TensorAccessor(out_args, out_addr);

    DataflowBuffer out_tile(cb_out);
#ifdef MP_FOLD
    {
        FUSED_ZONE("fz_mp_w_seed");
        constexpr uint32_t cb_act = get_named_compile_time_arg_val("cb_act");
        constexpr uint32_t cb_sig = get_named_compile_time_arg_val("cb_sig");
        constexpr uint32_t cb_zdone = get_named_compile_time_arg_val("cb_zdone");
        constexpr uint32_t top_k = get_named_compile_time_arg_val("top_k");
        constexpr uint32_t has_sig = get_named_compile_time_arg_val("has_sig");
        constexpr auto sig_args =
            TensorAccessorArgs<out_args.next_compile_time_args_offset(), out_args.next_common_runtime_args_offset()>();
        const uint32_t sig_addr = get_arg_val<uint32_t>(2);
        const auto sig = TensorAccessor(sig_args, sig_addr);
        Noc noc;
        DataflowBuffer act(cb_act);
        DataflowBuffer sig_tile(cb_sig);
        DataflowBuffer zdone(cb_zdone);
        // the reader owns the pages (it reserves them); the write pointers of a CB nobody pushed yet are its base
        if constexpr (has_sig) {
            sig_tile.reserve_back(1);
            noc.async_read(sig, CoreLocalMem<uint32_t>(sig_tile.get_write_ptr()), tile_bytes, {.page_id = 0}, {});
        }
        noc.async_write_zeros(CoreLocalMem<uint32_t>(act.get_write_ptr()), top_k * tile_bytes, {});
        noc.async_read_barrier();
        zdone.reserve_back(1);
        zdone.push_back(1);
        if constexpr (has_sig) {
            sig_tile.push_back(1);
        }
    }
#endif
    {
        FUSED_ZONE("fz_mp_w_write");
        out_tile.wait_front(1);
        noc_async_write(out_tile.get_read_ptr(), out.get_noc_addr(tile_col), tile_bytes);
        noc_async_write_barrier();
    }
    out_tile.pop_front(1);
}
