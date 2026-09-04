// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Fused SwiGLU elemwise backward kernel.

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

constexpr uint32_t dfb_linear1_id = tt::CBIndex::c_0;
constexpr uint32_t dfb_gate_id = tt::CBIndex::c_1;
constexpr uint32_t dfb_dL_dprod_id = tt::CBIndex::c_2;
constexpr uint32_t dfb_dL_dlinear1_id = tt::CBIndex::c_3;
constexpr uint32_t dfb_dL_dgate_id = tt::CBIndex::c_4;

void kernel_main() {
    namespace ckl = compute_kernel_lib;
    constexpr uint32_t one = 0x3F800000;
    constexpr uint32_t padded_Wt = ((Wt + block_size - 1) / block_size) * block_size;

    compute_kernel_hw_startup(dfb_linear1_id, dfb_dL_dlinear1_id);

    // Input tiles are consumed in blocks:
    //   linear1(U), gate, dL/dprod
    // and produce:
    //   dL/dlinear1, dL/dgate
    //
    // dL/dgate = dL/dprod * U * sigmoid(U).
    // dL/dlinear1 = dL/dprod * gate * sigmoid(U) * (1 + U*(1-sigmoid(U))).
    ckl::eltwise_chain(
        ckl::IterationShape::grid(num_rows_per_core, padded_Wt).block_size(block_size),
        // D0 = U, D1 = sigmoid(U), D2 = dL/dprod.
        ckl::CopyTile<
            ckl::input(
                dfb_linear1_id,
                ckl::WaitPolicy::PerBlockSize,
                ckl::PopPolicy::PerBlockSize,
                ckl::InputTileMapping::Block,
                ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D0>{},
        ckl::CopyDest<ckl::Dst::D0, ckl::Dst::D1>{},
        // Computes sigmoid(U) tile-wise and stores it for reuse in both output gradients.
        ckl::Sigmoid<ckl::Dst::D1>{},
        ckl::CopyTile<
            ckl::input(
                dfb_dL_dprod_id,
                ckl::WaitPolicy::PerBlockSize,
                ckl::PopPolicy::PerBlockSize,
                ckl::InputTileMapping::Block,
                ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D2>{},
        // D3 = dL/dgate = dL/dprod * U * sigmoid(U).
        ckl::MulBinary<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D3>{},
        ckl::MulBinary<ckl::Dst::D3, ckl::Dst::D2, ckl::Dst::D3>{},
        // PackTile runs in the chain's final pack phase, so D3 must
        // keep dL/dgate live until all computation is complete.
        ckl::PackTile<
            ckl::output(
                dfb_dL_dgate_id,
                ckl::ReservePolicy::PerBlockSize,
                ckl::PushPolicy::PerBlockSize,
                ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D3>{},
        // Avoid a fifth destination slot by rewriting
        // dL/dlinear1 = gate * (dL/dprod * sigmoid(U) +
        //                        dL/dgate * (1 - sigmoid(U))).
        ckl::MulBinary<ckl::Dst::D2, ckl::Dst::D1, ckl::Dst::D0>{},
        ckl::RsubUnary<ckl::Dst::D1>{one},
        ckl::MulBinary<ckl::Dst::D3, ckl::Dst::D1, ckl::Dst::D1>{},
        ckl::AddBinary<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0>{},
        ckl::CopyTile<
            ckl::input(
                dfb_gate_id,
                ckl::WaitPolicy::PerBlockSize,
                ckl::PopPolicy::PerBlockSize,
                ckl::InputTileMapping::Block,
                ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D2>{},
        ckl::MulBinary<ckl::Dst::D0, ckl::Dst::D2, ckl::Dst::D0>{},
        ckl::PackTile<
            ckl::output(
                dfb_dL_dlinear1_id,
                ckl::ReservePolicy::PerBlockSize,
                ckl::PushPolicy::PerBlockSize,
                ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D0>{});
}
