// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ISOLATED BENCH WRITER (perf idea I4). Stores this core's one stat tile so the
// bench has a correctness gate (the reduce output IS the scaler-scaled row sum).
//
// VARIANT 5 relocates the reduce-scaler preparation to THIS kernel: BRISC is idle
// from kernel start until the stat tile exists, so the whole cost lands in dead
// time and NCRISC's pre-read path keeps nothing at all. Same precedent as
// rms_norm's cb_zero_tile, which moved reader -> writer for exactly this reason.
// Producer of cb_scaler is still exactly ONE kernel, just a different one.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

namespace {
constexpr uint32_t cb_scaler = 2;
constexpr uint32_t cb_wmask = 3;
constexpr uint32_t cb_stat_partial = 7;
}  // namespace

void kernel_main() {
    constexpr uint32_t VARIANT = get_compile_time_arg_val(0);
    constexpr uint32_t WITH_MASK = get_compile_time_arg_val(1);
    constexpr auto dst_args = TensorAccessorArgs<2>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_tile = get_arg_val<uint32_t>(1);
    const uint32_t inv_w_bits = get_arg_val<uint32_t>(2);

    if constexpr (VARIANT == 5) {
        {
            // MASK FIRST: its consumer (the masked tail chain inside sumsq) is the
            // earlier of the two deadlines; the scaler is not touched until the
            // reduce that follows sumsq. Both are microseconds of slack from here.
            MaybeDeviceZoneScope("wr_prep_mask");
            if constexpr (WITH_MASK) {
                dataflow_kernel_lib::prepare_reduce_mask<cb_wmask, ckernel::ReduceDim::REDUCE_ROW>(32);
            }
        }
        MaybeDeviceZoneScope("wr_prep_scaler");
        float inv_w;
        __builtin_memcpy(&inv_w, &inv_w_bits, sizeof(inv_w));
        dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
            inv_w);
    }

    const uint32_t tile_bytes = get_tile_size(cb_stat_partial);
    const auto out_acc = TensorAccessor(dst_args, dst_addr, tile_bytes);
    {
        MaybeDeviceZoneScope("wr_wait_stat");
        cb_wait_front(cb_stat_partial, 1);
    }
    {
        MaybeDeviceZoneScope("wr_issue");
        noc_async_write_tile(out_tile, out_acc, get_read_ptr(cb_stat_partial));
    }
    {
        MaybeDeviceZoneScope("wr_barrier");
        noc_async_write_barrier();
    }
    cb_pop_front(cb_stat_partial, 1);
}
