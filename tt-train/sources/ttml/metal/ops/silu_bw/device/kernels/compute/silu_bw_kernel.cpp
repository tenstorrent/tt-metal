// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_activations.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu_basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_scalar.hpp"

constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);
constexpr uint32_t block_size = get_compile_time_arg_val(1);
constexpr uint32_t Wt = get_compile_time_arg_val(2);

// CBs with input data
constexpr uint32_t cb_input_idx = tt::CBIndex::c_0;
constexpr uint32_t cb_dL_out_idx = tt::CBIndex::c_1;
// CBs with output data
constexpr uint32_t cb_dL_da_idx = tt::CBIndex::c_2;
void kernel_main() {
    namespace ckl = compute_kernel_lib;
    constexpr uint32_t one = 0x3F800000;
    constexpr uint32_t padded_Wt = ((Wt + block_size - 1) / block_size) * block_size;

    compute_kernel_hw_startup(cb_input_idx, cb_dL_da_idx);

    ckl::eltwise_chain(
        ckl::EltwiseShape::grid(num_rows_per_core, padded_Wt, block_size),
        ckl::CopyTile<
            ckl::input(
                cb_input_idx, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::OperandKind::Block),
            ckl::Dst::D0>{},
        ckl::CopyDest<ckl::Dst::D0, ckl::Dst::D1>{},
        ckl::Sigmoid<ckl::Dst::D1>{},
        ckl::CopyDest<ckl::Dst::D1, ckl::Dst::D2>{},
        ckl::RsubUnary<ckl::Dst::D2>{one},
        ckl::MulBinary<ckl::Dst::D0, ckl::Dst::D2, ckl::Dst::D2>{},
        ckl::AddUnary<ckl::Dst::D2>{one},
        ckl::MulBinary<ckl::Dst::D1, ckl::Dst::D2, ckl::Dst::D1>{},
        ckl::CopyTile<
            ckl::input(
                cb_dL_out_idx, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::OperandKind::Block),
            ckl::Dst::D2>{},
        ckl::MulBinary<ckl::Dst::D1, ckl::Dst::D2, ckl::Dst::D0>{},
        ckl::PackTile<ckl::output(
            cb_dL_da_idx,
            ckl::ReservePolicy::PerBlockSize,
            ckl::PushPolicy::PerBlockSize,
            ckl::DataFormatReconfig::Disabled)>{});
}
