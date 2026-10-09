// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Shared compute for the SwiGLU-gating backward, used by both `swiglu_elemwise_bw` (two separate
// [.,I] tensors) and `swiglu_packed_bw` (one packed [.,2I] tensor).
//
// Given the gate branch (the one that is silu'd), the up branch, and the upstream grad dL/dh:
//   dL/d(up)   = dL/dh * silu(gate),                       silu(gate) = gate * sigmoid(gate)
//   dL/d(gate) = dL/dh * up * silu'(gate),
//                 silu'(gate) = sigmoid(gate) * (1 + gate * (1 - sigmoid(gate)))
//
// KERNEL-SIDE ONLY: include this from a compute kernel .cpp; it pulls in the LLK compute API.

#pragma once

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/activations.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/scalar.hpp"

// Compute one block of the SwiGLU-gating backward. The caller must have already
// waited `block_size` tiles on cb_gate / cb_up / cb_dh; this function keeps the intermediates
// (sigmoid, silu') in DEST and leaves cb_grad_gate / cb_grad_up pushed. Input CBs are popped
// by the caller.
template <
    uint32_t cb_gate,       // gate branch (silu'd)
    uint32_t cb_up,         // up branch
    uint32_t cb_dh,         // upstream grad dL/dh
    uint32_t cb_grad_gate,  // out: grad wrt gate branch
    uint32_t cb_grad_up,    // out: grad wrt up branch
    uint32_t block_size>
inline void swiglu_gate_bw_block() {
    namespace ckl = compute_kernel_lib;
    constexpr uint32_t one = 0x3F800000;  // 1.0f bits

    ckl::eltwise_chain(
        ckl::IterationShape::tiles(block_size).block_size(block_size),
        // D0 = gate, D1 = sigmoid(gate), kept for reuse in both gradients.
        ckl::CopyTile<
            ckl::input(
                cb_gate,
                ckl::WaitPolicy::None,
                ckl::PopPolicy::None,
                ckl::InputTileMapping::Block,
                ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D0>{},
        ckl::CopyDest<ckl::Dst::D0, ckl::Dst::D1, DataFormat::Float32>{},
        ckl::Sigmoid<ckl::Dst::D1>{},
        // D2 = silu'(gate) = sigmoid(gate) * (1 + gate * (1 - sigmoid(gate))).
        ckl::CopyDest<ckl::Dst::D1, ckl::Dst::D2, DataFormat::Float32>{},
        ckl::RsubUnary<ckl::Dst::D2>{one},
        ckl::MulBinary<ckl::Dst::D0, ckl::Dst::D2, ckl::Dst::D2>{},
        ckl::AddUnary<ckl::Dst::D2>{one},
        ckl::MulBinary<ckl::Dst::D1, ckl::Dst::D2, ckl::Dst::D2>{},
        // D3 = dL/dh.
        ckl::CopyTile<
            ckl::input(
                cb_dh,
                ckl::WaitPolicy::None,
                ckl::PopPolicy::None,
                ckl::InputTileMapping::Block,
                ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D3>{},
        // D0 = dL/d(up) = dL/dh * silu(gate), silu(gate) = gate * sigmoid(gate).
        ckl::MulBinary<ckl::Dst::D0, ckl::Dst::D1, ckl::Dst::D0>{},
        ckl::MulBinary<ckl::Dst::D0, ckl::Dst::D3, ckl::Dst::D0>{},
        // PackTile runs in the chain's final pack phase, so D0 must
        // keep dL/d(up) live until all computation is complete.
        ckl::PackTile<
            ckl::output(
                cb_grad_up,
                ckl::ReservePolicy::PerBlockSize,
                ckl::PushPolicy::PerBlockSize,
                ckl::DataFormatReconfig::Enabled),
            ckl::Dst::D0>{},
        // D1 = dL/d(gate) = dL/dh * up * silu'(gate); sigmoid(gate) is no longer needed.
        ckl::CopyTile<
            ckl::input(
                cb_up,
                ckl::WaitPolicy::None,
                ckl::PopPolicy::None,
                ckl::InputTileMapping::Block,
                ckl::DataFormatReconfig::Disabled),
            ckl::Dst::D1>{},
        ckl::MulBinary<ckl::Dst::D1, ckl::Dst::D3, ckl::Dst::D1>{},
        ckl::MulBinary<ckl::Dst::D1, ckl::Dst::D2, ckl::Dst::D1>{},
        ckl::PackTile<
            ckl::output(
                cb_grad_gate,
                ckl::ReservePolicy::PerBlockSize,
                ckl::PushPolicy::PerBlockSize,
                ckl::DataFormatReconfig::Enabled),
            ckl::Dst::D1>{});
}
