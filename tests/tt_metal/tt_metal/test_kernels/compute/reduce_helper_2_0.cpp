// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

namespace {

constexpr uint32_t rows = get_arg(args::rows);
constexpr uint32_t cols = get_arg(args::cols);
constexpr uint32_t batches = get_arg(args::batches);
constexpr uint32_t row_stride = get_arg(args::row_stride);
constexpr uint32_t batch_stride = get_arg(args::batch_stride);
constexpr uint32_t input_tiles = get_arg(args::input_tiles);
constexpr uint32_t auxiliary_tiles = get_arg(args::auxiliary_tiles);
constexpr uint32_t auxiliary_offset = get_arg(args::auxiliary_offset);
constexpr uint32_t output_group = get_arg(args::output_group);
constexpr uint32_t scale_bits = get_arg(args::scale_bits);
constexpr uint32_t post_mul_bits = get_arg(args::post_mul_bits);

constexpr auto input_policy = compute_kernel_lib::ReduceInputPolicy::REDUCE_INPUT_POLICY;
constexpr bool indexed_input = input_policy == compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop ||
                               input_policy == compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop;

ALWI auto make_post_op() {
#ifdef REDUCE_POST_MUL
    return [](uint32_t dst) {
        binop_with_scalar_tile_init();
        mul_unary_tile(dst, post_mul_bits);
    };
#else
    return compute_kernel_lib::NoOp{};
#endif
}

template <uint32_t output_dfb_id, typename InputDFB, typename AuxDFB, typename OutputDFB, typename AccumulateT>
ALWI void run_reduce(InputDFB& in, AuxDFB& aux, OutputDFB& out, AccumulateT accumulate) {
    if constexpr (input_policy == compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop) {
        in.wait_front(input_tiles);
    }
    compute_kernel_lib::reduce<
        REDUCE_OP,
        REDUCE_DIM,
        dfb::in,
        dfb::aux,
        output_dfb_id,
        input_policy,
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
        ReduceFp32Mode::Fast,
        REDUCE_ALGORITHM,
        compute_kernel_lib::ReduceWithinTile::REDUCE_WITHIN_TILE,
        scale_bits>(
        in,
        aux,
        out,
        compute_kernel_lib::ReduceInputBlockShape::of(rows, cols, batches),
        compute_kernel_lib::ReduceInputMemoryLayout::with_strides(row_stride, batch_stride),
        accumulate,
        make_post_op(),
        REDUCE_PARTIAL_MODE,
        output_group,
        auxiliary_offset);
    if constexpr (indexed_input) {
        in.pop_front(input_tiles);
    }
}

}  // namespace

void kernel_main() {
    DataflowBuffer in(dfb::in);
    DataflowBuffer aux(dfb::aux);
    DataflowBuffer out(dfb::out);

#ifdef REDUCE_ACCUMULATE
    constexpr auto reload = compute_kernel_lib::AccumulateReloadMode::REDUCE_RELOAD_MODE;
    DataflowBuffer acc(dfb::acc);
    compute_kernel_hw_startup(dfb::in, dfb::aux, dfb::acc);
    run_reduce<dfb::acc>(in, aux, acc, compute_kernel_lib::Accumulate::at(acc, 0).with_reload(reload));
    run_reduce<dfb::out>(in, aux, out, compute_kernel_lib::Accumulate::at_last(acc, 1).with_reload(reload));
#else
    compute_kernel_hw_startup(dfb::in, dfb::aux, dfb::out);
    run_reduce<dfb::out>(in, aux, out, compute_kernel_lib::NoAccumulation{});
#endif

    aux.pop_front(auxiliary_tiles);
}
