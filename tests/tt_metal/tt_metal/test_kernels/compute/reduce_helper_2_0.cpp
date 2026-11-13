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
constexpr uint32_t auxiliary_tiles = get_arg(args::auxiliary_tiles);
constexpr uint32_t output_group = get_arg(args::output_group);
constexpr uint32_t scale_bits = get_arg(args::scale_bits);

template <uint32_t output_dfb_id, typename InputDFB, typename AuxDFB, typename OutputDFB, typename AccumulateT>
ALWI void run_reduce(InputDFB& in, AuxDFB& aux, OutputDFB& out, AccumulateT accumulate) {
    compute_kernel_lib::reduce<
        REDUCE_OP,
        REDUCE_DIM,
        dfb::in,
        dfb::aux,
        output_dfb_id,
        compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop,
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
        ReduceFp32Mode::Fast,
        REDUCE_ALGORITHM,
        compute_kernel_lib::ReduceWithinTile::Collapse,
        scale_bits>(
        in,
        aux,
        out,
        compute_kernel_lib::ReduceInputBlockShape::of(rows, cols, batches),
        compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
        accumulate,
        compute_kernel_lib::NoOp{},
        REDUCE_PARTIAL_MODE,
        output_group);
}

}  // namespace

void kernel_main() {
    DataflowBuffer in(dfb::in);
    DataflowBuffer aux(dfb::aux);
    DataflowBuffer out(dfb::out);

#ifdef REDUCE_ACCUMULATE
    DataflowBuffer acc(dfb::acc);
    compute_kernel_hw_startup(dfb::in, dfb::aux, dfb::acc);
    run_reduce<dfb::acc>(
        in,
        aux,
        acc,
        compute_kernel_lib::Accumulate::at(acc, 0).with_reload(
            compute_kernel_lib::AccumulateReloadMode::CopySeedPairs));
    run_reduce<dfb::out>(
        in,
        aux,
        out,
        compute_kernel_lib::Accumulate::at_last(acc, 1).with_reload(
            compute_kernel_lib::AccumulateReloadMode::CopySeedPairs));
#else
    compute_kernel_hw_startup(dfb::in, dfb::aux, dfb::out);
    run_reduce<dfb::out>(in, aux, out, compute_kernel_lib::NoAccumulation{});
#endif

    aux.pop_front(auxiliary_tiles);
}
