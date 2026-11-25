// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Quasar 2.0 reduce: in0 (data) + in1 (one scaler tile) -> reduce -> Dest -> pack -> out0, using only the id-free 2.0
// compute API. Compile-time args: pool (0 MAX, 1 SUM, 2 AVG), dim (0 ROW, 1 COL, 2 SCALAR), num_out, tiles_per_out,
// block. block = 0: each output accumulates tiles_per_out data tiles into DST 0 (reduce_tile at a fixed idst).
// block = 1: tiles_per_out data tiles reduce each into its own DST slot (reduce_block), tiles_per_out outputs per step.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/reg_api.h"
#include "api/compute/experimental/2_0/hw_startup.h"
#include "api/compute/experimental/2_0/pack.h"
#include "api/compute/experimental/2_0/reduce.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/llk_operand_from_tokens.h"
#include "experimental/kernel_args.h"

namespace ex = ckernel::experimental;

void kernel_main() {
    using Data = LLKOperandFrom<dfb::in0>;
    using Scaler = LLKOperandFrom<dfb::in1>;
    using Out = LLKOperandFrom<dfb::out0>;
    constexpr std::uint32_t pool = get_arg(args::pool);
    constexpr std::uint32_t dim = get_arg(args::dim);
    constexpr PoolType P = pool == 0 ? PoolType::MAX : pool == 1 ? PoolType::SUM : PoolType::AVG;
    constexpr ReduceDim D = dim == 0   ? ReduceDim::REDUCE_ROW
                            : dim == 1 ? ReduceDim::REDUCE_COL
                                       : ReduceDim::REDUCE_SCALAR;
    constexpr std::uint32_t num_out = get_arg(args::num_out);
    constexpr std::uint32_t n = get_arg(args::tiles_per_out);
    constexpr bool block = get_arg(args::block) != 0;
    static_assert(!block || num_out % n == 0, "no tails: the block divides num_out");
    constexpr std::uint32_t steps = block ? num_out / n : num_out;
    constexpr std::uint32_t packed = block ? n : 1;

    DataflowBuffer data(dfb::in0);
    DataflowBuffer scaler(dfb::in1);
    DataflowBuffer out(dfb::out0);
    ckernel::compute_kernel_hw_startup(
        data.template front<Data>(), scaler.template front<Scaler>(), out.template back<Out>());

    scaler.wait_front(1);
    ex::reduce_init<P, D>(data.template front<Data>(), scaler.template front<Scaler>(), out.template back<Out>());
    for (std::uint32_t s = 0; s < steps; ++s) {
        data.wait_front(n);
        out.reserve_back(packed);
        ckernel::tile_regs_acquire();
        if constexpr (block) {
            ex::reduce_block<P, D, n>(data.template front<Data>(), scaler.template front<Scaler>(), 0, 0, 0);
        } else {
            for (std::uint32_t i = 0; i < n; ++i) {
                ex::reduce_tile<P, D>(data.template front<Data>(), scaler.template front<Scaler>(), i, 0, 0);
            }
        }
        ckernel::tile_regs_commit();
        ckernel::tile_regs_wait();
        ex::pack_block<packed, 1>(out.template back<Out>(), 0);
        ckernel::tile_regs_release();
        out.push_back(packed);
        data.pop_front(n);
    }
    ex::reduce_uninit();
    scaler.pop_front(1);
}
