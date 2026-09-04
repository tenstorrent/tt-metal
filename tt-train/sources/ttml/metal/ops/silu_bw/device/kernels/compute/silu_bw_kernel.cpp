// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/activations.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/scalar.hpp"

constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);
constexpr uint32_t block_size = get_compile_time_arg_val(1);
constexpr uint32_t Wt = get_compile_time_arg_val(2);

// DFBs with input data
constexpr uint32_t dfb_input_idx_id = tt::CBIndex::c_0;
constexpr uint32_t dfb_dL_out_idx_id = tt::CBIndex::c_1;
// DFBs with output data
constexpr uint32_t dfb_dL_da_idx_id = tt::CBIndex::c_2;
void kernel_main() {
    namespace ckl = compute_kernel_lib;
    constexpr uint32_t one = 0x3F800000;  // FP32 encoding of 1.0
    constexpr uint32_t padded_Wt = ((Wt + block_size - 1) / block_size) * block_size;

    compute_kernel_hw_startup(dfb_input_idx_id, dfb_dL_da_idx_id);

    // dL/dx = dL/dout * sigmoid(x) * (1 + x*(1-sigmoid(x))).
    ckl::eltwise_chain(
        ckl::IterationShape::grid(num_rows_per_core, padded_Wt).block_size(block_size),
        ckl::CopyTile<
            ckl::input(
                dfb_input_idx_id,
                ckl::WaitPolicy::PerBlockSize,
                ckl::PopPolicy::PerBlockSize,
                ckl::InputTileMapping::Block,
                ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D0>{},
        ckl::CopyDest<ckl::Dst::D0, ckl::Dst::D1>{},
        // Compute: sigmoid(x) = 1 / (1 + exp(-x))
        ckl::Sigmoid<ckl::Dst::D1>{},
        ckl::CopyDest<ckl::Dst::D1, ckl::Dst::D2>{},
        // Compute: 1 - sigmoid(x)
        ckl::RsubUnary<ckl::Dst::D2>{one},  // 1.0F is the constant to subtract from.
        // Compute: (1 - sigmoid(x)) * input + 1
        ckl::MulBinary<ckl::Dst::D0, ckl::Dst::D2, ckl::Dst::D2>{},
        ckl::AddUnary<ckl::Dst::D2>{one},  // Add 1.0F to the result.
        // Compute: ((1 - sigmoid(x)) * input + 1) * sigmoid(x)
        ckl::MulBinary<ckl::Dst::D1, ckl::Dst::D2, ckl::Dst::D1>{},
        ckl::CopyTile<
            ckl::input(
                dfb_dL_out_idx_id,
                ckl::WaitPolicy::PerBlockSize,
                ckl::PopPolicy::PerBlockSize,
                ckl::InputTileMapping::Block,
                ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D2>{},
        // Compute: ((1 - sigmoid(x)) * input + 1) * sigmoid(x) * dL_dout
        // The result is stored in dfb_dL_da_idx_id.
        ckl::MulBinary<ckl::Dst::D1, ckl::Dst::D2, ckl::Dst::D0>{},
        ckl::PackTile<ckl::output(
            dfb_dL_da_idx_id,
            ckl::ReservePolicy::PerBlockSize,
            ckl::PushPolicy::PerBlockSize,
            ckl::DataFormatReconfig::Disabled)>{});
}
