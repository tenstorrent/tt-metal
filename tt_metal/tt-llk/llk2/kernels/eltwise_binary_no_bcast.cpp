// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ttnn's eltwise_binary_no_bcast compute kernel (eltwise/binary_ng/device/kernels/compute/) on LLK 2.0, for add, sub
// and mul of two tensors of the same shape without activations: Tensors, the Compute API and owners for every sync.
// One body for all three threads; no LLK calls, no init / uninit / reconfig, no tile_regs_* and no per-thread code.
//
// Syncs, and where each one is (compute_api/sync.h):
//   lhs, rhs, out   the kernel's dataflow buffers, through Tensors: wait_front / reserve_back return the Tensor,
//                   release() pops or pushes its pages (compute_api/dataflow.h)
//   Dest            a ring of two sections (compute_api/dest.h): math writes one block while pack reads the other
//   math -> pack    inside pack
//
// Flow, per block of n tiles: tiles_per_cycle of them (the 1.0 kernel's process_tiles), the last block what is left:
//   a, b = wait_front(lhs, n), wait_front(rhs, n)
//   y = a op b                                   into a free Dest section
//   pack(y, reserve_back(out, n))                PACK_RELU: with ReLU
//   release everything
//
// The block's tile count is a run-time value (tensor/tensor.h, "run-time sizes"): the specs hold the largest block
// (tiles_per_cycle tiles: what the buffers and a Dest section hold) and mark their tile columns run time, and each
// Tensor holds its own count, which the layers' loops, the buffers' page counts and the packer's MOP count follow.
// So there is one block() and no tile-by-tile tail.
//
// What this file assumes from tt-metal that does not exist today: each CB's data format is the named compile-time
// arg <cb>_format on all three TRISCs, the op is the named arg binary_op (0 add, 1 sub, 2 mul), the Dest mode and the
// math fidelity come as the defines LLK2_DEST_FP32 and LLK2_MATH_FIDELITY, every CB has pages of one tile, and
// llk2/inc is on the include path.
//
// What the 1.0 kernel does that this one does not (yet): activations on either input or on the result other than a
// ReLU on pack, the other binary ops, and formats LLK 2.0's add / sub / mul do not take.

#include <cstdint>

#include "api/compute/common.h" // get_arg_val only (tt-metal has run-time args nowhere else): no 1.0 compute API call may appear in this file
#include "api/dataflow/dataflow_buffer.h"
#include "compute_api/dataflow.h"
#include "compute_api/dest.h"
#include "compute_api/eltwise_binary.h"
#include "compute_api/pack.h"
#include "kernel_args.h"

namespace
{

using namespace llk2;

constexpr std::uint32_t tiles_per_cycle = get_compile_time_arg_val(0); // compile-time arg 0 of the 1.0 kernel
constexpr std::uint32_t binary_op       = get_named_compile_time_arg_val("binary_op");
static_assert(binary_op <= 2, "LLK 2.0: add (0), sub (1) and mul (2) only");

#ifdef PACK_RELU
constexpr bool pack_relu = true;
#else
constexpr bool pack_relu = false;
#endif

// A block of up to tiles_per_cycle tiles of one tile row; how many is each Tensor's (run time).
inline constexpr const auto& lhs_spec = runtime_cols<l1_spec<default_tile_dim, tiles_per_cycle * default_tile_dim, LLK2_CB_FORMAT("cb_lhs")>>;
inline constexpr const auto& rhs_spec = runtime_cols<l1_spec<default_tile_dim, tiles_per_cycle * default_tile_dim, LLK2_CB_FORMAT("cb_rhs")>>;
inline constexpr const auto& out_spec = runtime_cols<l1_spec<default_tile_dim, tiles_per_cycle * default_tile_dim, LLK2_CB_FORMAT("cb_out")>>;
/// Two Dest sections of the largest block's size: math writes one block while pack reads the other.
using AccRing = DestRing<dst_spec<default_tile_dim, tiles_per_cycle * default_tile_dim, kernel_dest_format>>;

/// out = lhs op rhs for the next n tiles of each buffer (run time: n).
inline __attribute__((always_inline)) void block(DataflowBuffer& lhs, DataflowBuffer& rhs, DataflowBuffer& out, std::uint32_t n)
{
    const auto a   = wait_front<lhs_spec>(lhs, n);
    const auto b   = wait_front<rhs_spec>(rhs, n);
    const auto acc = reserve_back<AccRing>();
    if constexpr (binary_op == 0)
    {
        add(a, b, acc);
    }
    else if constexpr (binary_op == 1)
    {
        sub(a, b, acc);
    }
    else
    {
        mul(a, b, acc);
    }
    release(a);
    release(b);
    release(pack<pack_modifiers {.relu = pack_relu}>(acc, reserve_back<out_spec>(out, n)));
    release(acc);
}

} // namespace

void kernel_main()
{
    DataflowBuffer lhs(get_named_compile_time_arg_val("cb_lhs"));
    DataflowBuffer rhs(get_named_compile_time_arg_val("cb_rhs"));
    DataflowBuffer out(get_named_compile_time_arg_val("cb_out"));

    std::uint32_t left = get_arg_val<std::uint32_t>(0); // runtime arg 0 of the 1.0 kernel: this core's tiles
    while (left > 0)
    {
        const std::uint32_t n = left < tiles_per_cycle ? left : tiles_per_cycle; // the last block takes what is left
        block(lhs, rhs, out, n);
        left -= n;
    }
}
