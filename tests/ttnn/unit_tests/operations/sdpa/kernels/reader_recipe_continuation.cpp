// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t chunks = get_compile_time_arg_val(0);
    constexpr uint32_t split = get_compile_time_arg_val(1);
    constexpr bool reload_q = get_compile_time_arg_val(2);
    constexpr bool stage = get_compile_time_arg_val(3);
    constexpr auto qa = TensorAccessorArgs<4>();
    constexpr auto ka = TensorAccessorArgs<qa.next_compile_time_args_offset()>();
    constexpr auto va = TensorAccessorArgs<ka.next_compile_time_args_offset()>();
    auto q = TensorAccessor(qa, get_arg_val<uint32_t>(0));
    auto k = TensorAccessor(ka, get_arg_val<uint32_t>(1));
    auto v = TensorAccessor(va, get_arg_val<uint32_t>(2));
    Noc noc;
    DataflowBuffer qcb(0), kcb(1), vcb(2);
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        3,
        ckernel::PoolType::MAX,
        ckernel::ReduceDim::REDUCE_ROW,
        dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>();
    generate_bcast_col_scalar(CircularBuffer(4), 0x3f803f80);
    for (uint32_t iteration = 0; iteration < (stage ? 2 * chunks : chunks); ++iteration) {
        const uint32_t chunk = !stage || iteration < split  ? iteration
                               : iteration < split + chunks ? iteration - split
                                                            : iteration - chunks;
        if (iteration == 0 || (stage && (iteration == split || iteration == split + chunks)) ||
            (!stage && reload_q && chunk == split)) {
            qcb.reserve_back(32);
            for (uint32_t p = 0; p < 32; ++p) {
                noc.async_read(q, qcb, 2048, {.page_id = p}, {.offset_bytes = p * 2048});
            }
            noc.async_read_barrier();
            qcb.push_back(32);
        }
        kcb.reserve_back(64);
        for (uint32_t p = 0; p < 64; ++p) {
            const uint32_t destination = (p % 4) * 16 + p / 4;
            noc.async_read(
                k,
                kcb,
                get_tile_size(1),
                {.page_id = chunk * 64 + p},
                {.offset_bytes = destination * get_tile_size(1)});
        }
        noc.async_read_barrier();
        kcb.push_back(64);
        vcb.reserve_back(64);
        for (uint32_t p = 0; p < 64; ++p) {
            noc.async_read(
                v, vcb, get_tile_size(2), {.page_id = chunk * 64 + p}, {.offset_bytes = p * get_tile_size(2)});
        }
        noc.async_read_barrier();
        vcb.push_back(64);
    }
}
