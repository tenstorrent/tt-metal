// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Fused SwiGLU elemwise backward kernel.

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

constexpr uint32_t cb_linear1 = tt::CBIndex::c_0;
constexpr uint32_t cb_gate = tt::CBIndex::c_1;
constexpr uint32_t cb_dL_dprod = tt::CBIndex::c_2;
constexpr uint32_t cb_dL_dlinear1 = tt::CBIndex::c_3;
constexpr uint32_t cb_dL_dgate = tt::CBIndex::c_4;

void kernel_main() {
    namespace ckl = compute_kernel_lib;
    constexpr uint32_t one = 0x3F800000;
    constexpr uint32_t padded_Wt = ((Wt + block_size - 1) / block_size) * block_size;

    compute_kernel_hw_startup(cb_linear1, cb_dL_dlinear1);

    ckl::eltwise_chain(
        ckl::EltwiseShape::grid(num_rows_per_core, padded_Wt, block_size),
        // D0 = U, D1 = sigmoid(U), D2 = dL/dprod.
        ckl::CopyTile<
            ckl::input(
                cb_linear1, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::OperandKind::Block),
            ckl::Dst::D0>{},
        ckl::CopyDest<ckl::Dst::D0, ckl::Dst::D1>{},
        ckl::Sigmoid<ckl::Dst::D1>{},
        ckl::CopyTile<
            ckl::input(
                cb_dL_dprod, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::OperandKind::Block),
            ckl::Dst::D2>{},
        // D3 = dL/dgate = dL/dprod * U * sigmoid(U).
        ckl::MulBinary<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D3>{},
        ckl::MulBinary<ckl::Dst::D3, ckl::Dst::D2, ckl::Dst::D3>{},
        // PackTile runs in the chain's final pack phase, so D3 must
        // keep dL/dgate live until all computation is complete.
        ckl::PackTile<
            ckl::output(
                cb_dL_dgate,
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
            ckl::input(cb_gate, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::OperandKind::Block),
            ckl::Dst::D2>{},
        ckl::MulBinary<ckl::Dst::D0, ckl::Dst::D2, ckl::Dst::D0>{},
        ckl::PackTile<
            ckl::output(
                cb_dL_dlinear1,
                ckl::ReservePolicy::PerBlockSize,
                ckl::PushPolicy::PerBlockSize,
                ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D0>{});
}
