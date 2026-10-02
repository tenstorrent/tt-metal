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

    // Input tiles are consumed in blocks:
    //   linear1(U), gate, dL/dprod
    // and produce:
    //   dL/dlinear1, dL/dgate
    //
    // dL/dgate = dL/dprod * silu(U), where silu(U) = U * sigmoid(U).
    // dL/dlinear1 = dL/dprod * gate * silu'(U),
    //   where silu'(U) = sigmoid(U) * (1 + U * (1 - sigmoid(U))).
    ckl::eltwise_chain(
        ckl::IterationShape::grid(num_rows_per_core, padded_Wt).block_size(block_size),
        // D0 = U, D1 = sigmoid(U).
        ckl::CopyTile<
            ckl::input(
                cb_linear1,
                ckl::WaitPolicy::PerBlockSize,
                ckl::PopPolicy::PerBlockSize,
                ckl::InputTileMapping::Block,
                ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D0>{},
        ckl::CopyDest<ckl::Dst::D0, ckl::Dst::D1, DataFormat::Float32>{},
        // Computes sigmoid(U) tile-wise and keeps it for reuse in both output gradients.
        ckl::Sigmoid<ckl::Dst::D1>{},
        // D2 = silu'(U) = sigmoid(U) * (U * (1 - sigmoid(U)) + 1).
        ckl::CopyDest<ckl::Dst::D1, ckl::Dst::D2, DataFormat::Float32>{},
        ckl::RsubUnary<ckl::Dst::D2>{one},
        ckl::MulBinary<ckl::Dst::D0, ckl::Dst::D2, ckl::Dst::D2>{},
        ckl::AddUnary<ckl::Dst::D2>{one},
        ckl::MulBinary<ckl::Dst::D1, ckl::Dst::D2, ckl::Dst::D2>{},
        // D3 = dL/dprod.
        ckl::CopyTile<
            ckl::input(
                cb_dL_dprod,
                ckl::WaitPolicy::PerBlockSize,
                ckl::PopPolicy::PerBlockSize,
                ckl::InputTileMapping::Block,
                ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D3>{},
        // D0 = dL/dgate = (U * sigmoid(U)) * dL/dprod.
        ckl::MulBinary<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0>{},
        ckl::MulBinary<ckl::Dst::D0, ckl::Dst::D3, ckl::Dst::D0>{},
        // PackTile runs in the chain's final pack phase, so D0 must
        // keep dL/dgate live until all computation is complete.
        ckl::PackTile<
            ckl::output(
                cb_dL_dgate,
                ckl::ReservePolicy::PerBlockSize,
                ckl::PushPolicy::PerBlockSize,
                ckl::DataFormatReconfig::Enabled),
            ckl::Dst::D0>{},
        // D1 = dL/dlinear1 = (gate * dL/dprod) * silu'(U); sigmoid(U) is no longer needed.
        ckl::CopyTile<
            ckl::input(
                cb_gate,
                ckl::WaitPolicy::PerBlockSize,
                ckl::PopPolicy::PerBlockSize,
                ckl::InputTileMapping::Block,
                ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D1>{},
        ckl::MulBinary<ckl::Dst::D1, ckl::Dst::D3, ckl::Dst::D1>{},
        ckl::MulBinary<ckl::Dst::D1, ckl::Dst::D2, ckl::Dst::D1>{},
        ckl::PackTile<
            ckl::output(
                cb_dL_dlinear1,
                ckl::ReservePolicy::PerBlockSize,
                ckl::PushPolicy::PerBlockSize,
                ckl::DataFormatReconfig::Enabled),
            ckl::Dst::D1>{});
}
