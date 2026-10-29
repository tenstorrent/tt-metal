// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ttnn's bmm_large_block_zm_fused_bias_activation compute kernel (matmul/device/kernels/compute/) on LLK 2.0:
// Tensors, the Compute API and owners for every sync. One body for all three threads; no LLK calls, no init /
// uninit / reconfig, no tile_regs_* and no per-thread code.
//
// Syncs, and where each one is (compute_api/sync.h):
//   in0, in1, partials, bias, out   the kernel's dataflow buffers, through Tensors: wait_front / reserve_back return
//                                   the Tensor, release() pops or pushes its pages (compute_api/dataflow.h)
//   Dest                            a ring of two sections (compute_api/dest.h): reserve_back<AccRing>() waits for a
//                                   free one on math; release() frees it once the packer has read it. Math works on
//                                   the next block while pack reads the last one
//   math -> pack                    inside pack (or inside the activation, when it runs on the pack thread): each op
//                                   returns what it wrote, owned by its unit, and the next op hands it over
//
// Flow, per output block (the 1.0 kernel's bh / bw loop; one batch):
//   for each K block:
//     x, w = wait_front(in0), wait_front(in1)                  [RT, KB] and [KB, CT] tiles at the read pointers
//     acc = reserve_back<AccRing>()                            a zeroed Dest section
//     reload: copy(wait_front(partials), acc)                  the previous blocks' sum back into Dest
//     y = matmul(x, w, acc)                                    y = acc + x * w
//     last block: pack(activation(y), reserve_back(out))       the activation on the pack thread; PACK_RELU
//                 (FUSE_BIAS: pack into the partials instead)
//     else:       pack(y, reserve_back(partials))              spill, or with L1 acc add into the partials
//     release everything
//   FUSE_BIAS:
//     y = add<Row>(wait_front(partials), peek_front(bias), reserve_back<AccRing>())
//     pack<relu>(activation(y), reserve_back(out))
//   the bias stays for every output block and is popped once at the end, as 1.0.
//
// Accumulating over K blocks, two ways (as the 1.0 kernel):
//   spill (default)  each block but the last packs its sum to the partials buffer, and the next block copies it back
//                    into Dest before its matmul adds to it
//   L1 acc           the packer adds into the partials buffer in L1 (pack<{.l1_acc = Toggle::Runtime}>, on from the
//                    second block). The partials buffer must hold exactly one output block, so popping it and
//                    reserving again gives the same pages back. Without FUSE_BIAS the last block still reloads the sum
//                    (one copy) and packs to the output; with FUSE_BIAS every block packs into the partials buffer
//                    and the bias stage reads it
//
// What this file assumes from tt-metal that does not exist today:
//   - Each CB's data format is a named compile-time arg (<cb>_format), the same on all three TRISCs: the Tensors'
//     specs (and through them the Resolver) need it at compile time on every thread. Today each TRISC sees only its
//     own generated table (unpack_src_format[] on unpack, pack_dst_format[] on pack).
//   - Every CB has pages of one tile (the Tensors are counted in tiles, as 1.0 counts them).
//   - The partials CB (cb_intermed0) holds exactly one output block, so L1 acc reserves the same pages again (true
//     of the 1.0 host setup for this kernel; nothing here can check it at compile time).
//   - llk2/inc on the include path.
// And as today: the firmware sets up each TRISC's CB pointers and zeroes the stream counters before the kernel
// starts, and the Dest mode and math fidelity come from DST_ACCUM_MODE and MATH_FIDELITY.
//
// The tt-llk harness runs this same file (tests/sources/llk2_bmm_fused_compute_api_test.cpp includes it), with its
// own stand-ins for the tt-metal pieces above (tests/helpers/tt_metal_kernel.h).
//
// What the 1.0 kernel does that this one does not (yet): subblocks smaller than the per-core block, in0 transpose,
// in1 transpose, untilize_out, batch from the reader (sparsity), BIAS_FULL_BLOCK / element-wise bias,
// MATMUL_DRAM_SHARDED padding, SKIP_COMPUTE, and the activations other than silu, relu6 and hardtanh.

#include <cstdint>

#define LLK2_DEST_FP32     DST_ACCUM_MODE
#define LLK2_MATH_FIDELITY static_cast<std::uint32_t>(MATH_FIDELITY)
#include "api/dataflow/dataflow_buffer.h"
#include "compute_api/copy.h"
#include "compute_api/dataflow.h"
#include "compute_api/dest.h"
#include "compute_api/eltwise_binary.h"
#include "compute_api/matmul.h"
#include "compute_api/pack.h"
#include "compute_api/sync.h"
#include "compute_api/vec.h"
#ifdef SFPU_ACTIVATION
#include "ttnn/operations/matmul/shared_with_host/activation_type.hpp"
#endif

namespace
{

using namespace llk2;

/// CB name's data format, from the host's named compile-time arg <name>_format.
#define LLK2_CB_FORMAT(name) static_cast<DataFormat>(get_named_compile_time_arg_val(name "_format"))

// The 1.0 kernel's positional compile-time args this one reads; the others it ignores.
constexpr std::uint32_t in0_block_w       = get_compile_time_arg_val(0);  // compile-time arg 0 of the 1.0 kernel: K tiles per block
constexpr std::uint32_t in0_num_subblocks = get_compile_time_arg_val(1);  // compile-time arg 1 of the 1.0 kernel
constexpr std::uint32_t in1_num_subblocks = get_compile_time_arg_val(4);  // compile-time arg 4 of the 1.0 kernel
constexpr std::uint32_t k_blocks          = get_compile_time_arg_val(7);  // compile-time arg 7 of the 1.0 kernel: num_blocks_inner_dim
constexpr std::uint32_t num_blocks_w_dim  = get_compile_time_arg_val(8);  // compile-time arg 8 of the 1.0 kernel
constexpr std::uint32_t num_blocks_h_dim  = get_compile_time_arg_val(9);  // compile-time arg 9 of the 1.0 kernel
constexpr std::uint32_t out_subblock_h    = get_compile_time_arg_val(10); // compile-time arg 10 of the 1.0 kernel
constexpr std::uint32_t out_subblock_w    = get_compile_time_arg_val(11); // compile-time arg 11 of the 1.0 kernel
constexpr std::uint32_t batch             = get_compile_time_arg_val(13); // compile-time arg 13 of the 1.0 kernel
constexpr bool untilize_out               = get_compile_time_arg_val(15); // compile-time arg 15 of the 1.0 kernel

static_assert(in0_num_subblocks == 1 && in1_num_subblocks == 1, "LLK 2.0: one output subblock per block (out_subblock = the per-core block)");
static_assert(!untilize_out, "LLK 2.0: untilize_out is not ported (pack has no untilize yet); the output is tiled");

constexpr std::uint32_t rt_dim     = out_subblock_h * in0_num_subblocks; // output block height in tiles
constexpr std::uint32_t ct_dim     = out_subblock_w * in1_num_subblocks; // output block width in tiles
constexpr std::uint32_t out_blocks = batch * num_blocks_h_dim * num_blocks_w_dim;

#ifdef SFPU_ACTIVATION
using ttnn::operations::matmul::KernelActivation;
constexpr KernelActivation kernel_activation = static_cast<KernelActivation>(get_named_compile_time_arg_val("activation_type"));
static_assert(
    kernel_activation == KernelActivation::SILU || kernel_activation == KernelActivation::RELU6 || kernel_activation == KernelActivation::HARDTANH,
    "LLK 2.0 kernel: only silu, relu6 and hardtanh are wired up (activation below)");
constexpr VecFunction activation_function = kernel_activation == KernelActivation::SILU    ? VecFunction::Silu
                                            : kernel_activation == KernelActivation::RELU6 ? VecFunction::ReluMax
                                                                                           : VecFunction::Hardtanh;
constexpr std::uint32_t relu6_max     = 0x40c00000u; // 6.0, the 1.0 kernel's RELU6 default when param0 is 0
constexpr std::uint32_t param0        = get_named_compile_time_arg_val("activation_param0");
constexpr std::uint32_t activation_p0 = kernel_activation == KernelActivation::RELU6 && param0 == 0 ? relu6_max : param0;
constexpr std::uint32_t activation_p1 = get_named_compile_time_arg_val("activation_param1");
#endif

#ifndef SFPU_ACTIVATION
constexpr VecFunction activation_function = VecFunction::None;
constexpr std::uint32_t activation_p0     = 0;
constexpr std::uint32_t activation_p1     = 0;
#endif

#ifdef PACKER_L1_ACC
constexpr bool l1_acc = true;
#else
constexpr bool l1_acc = false;
#endif
#ifdef FUSE_BIAS
constexpr bool fuse_bias         = true;
constexpr std::uint32_t bias_cb  = get_named_compile_time_arg_val("cb_bias");
constexpr DataFormat bias_format = LLK2_CB_FORMAT("cb_bias");
static_assert(num_blocks_w_dim == 1, "LLK 2.0: the bias is waited for once, kept and popped at the end, which needs num_blocks_w_dim == 1");
#else
constexpr bool fuse_bias         = false;
constexpr std::uint32_t bias_cb  = 0;                     // no bias CB: the body never touches it
constexpr DataFormat bias_format = DataFormat::Float16_b; // unused
#endif
#ifdef PACK_RELU
constexpr bool pack_relu = true;
#else
constexpr bool pack_relu = false;
#endif

constexpr std::uint32_t tile_rows = rt_dim * default_tile_dim, tile_cols = ct_dim * default_tile_dim, k_cols = in0_block_w * default_tile_dim;
constexpr const auto& x_spec        = l1_spec<tile_rows, k_cols, LLK2_CB_FORMAT("cb_in0")>;           // [RT, KB] tiles
constexpr const auto& w_spec        = l1_spec<k_cols, tile_cols, LLK2_CB_FORMAT("cb_in1")>;           // [KB, CT] tiles
constexpr const auto& bias_spec     = l1_spec<default_tile_dim, tile_cols, bias_format>;              // one tile row
constexpr const auto& partials_spec = l1_spec<tile_rows, tile_cols, LLK2_CB_FORMAT("cb_intermed0")>; // [RT, CT] tiles
constexpr const auto& out_spec      = l1_spec<tile_rows, tile_cols, LLK2_CB_FORMAT("cb_out")>;        // [RT, CT] tiles
constexpr const auto& acc_spec      = dst_spec<tile_rows, tile_cols, kernel_dest_format>;             // [RT, CT] tiles in Dest
// The last K block packs to the output, or with FUSE_BIAS to the partials the bias stage reads (1.0's mm_out_dfb).
constexpr const auto& mm_out_spec = fuse_bias ? partials_spec : out_spec;
constexpr bool relu_on_matmul_out = pack_relu && !fuse_bias;

/// Two Dest sections: math writes one block while pack reads the other.
using AccRing = DestRing<acc_spec>;

/// The activation on y, driven from the pack thread, as the 1.0 kernel's apply_activation_from_pack: math moves on to
/// the next block while pack runs it. Without one, y. The thread is per call: .thread = Thread::Math here would run it
/// on math instead, and a kernel can switch between calls with migrate (compute_api/sync.h).
template <const TensorSpec<tensor_rank>& D, class R>
inline __attribute__((always_inline)) auto activation(const Tensor<D, R>& y)
{
    constexpr vec_modifiers mods {.thread = Thread::Pack};
    if constexpr (activation_function == VecFunction::Silu)
    {
        return vec_silu<mods>(y);
    }
    else if constexpr (activation_function == VecFunction::ReluMax)
    {
        return vec_relu_max<mods>(y, activation_p0);
    }
    else if constexpr (activation_function == VecFunction::Hardtanh)
    {
        return vec_hardtanh<mods>(y, activation_p0, activation_p1);
    }
    else
    {
        static_assert(activation_function == VecFunction::None, "only silu, relu_max and hardtanh are wired up here; add the other vec_ calls the same way");
        return y;
    }
}

} // namespace

void kernel_main()
{
    DataflowBuffer in0(get_named_compile_time_arg_val("cb_in0"));
    DataflowBuffer in1(get_named_compile_time_arg_val("cb_in1"));
    DataflowBuffer bias(bias_cb);
    DataflowBuffer partials(get_named_compile_time_arg_val("cb_intermed0"));
    DataflowBuffer out(get_named_compile_time_arg_val("cb_out"));
    DataflowBuffer& mm_out = fuse_bias ? partials : out;

    for (std::uint32_t block_out = 0; block_out < out_blocks; ++block_out)
    {
        bool reload = false;
        for (std::uint32_t block = 0; block < k_blocks; ++block)
        {
            const bool last_k_block = block == k_blocks - 1;
            const auto x            = wait_front<x_spec>(in0);
            const auto w            = wait_front<w_spec>(in1);
            const auto acc          = reserve_back<AccRing>();
            if (reload)
            {
                const auto partial = wait_front<partials_spec>(partials);
                copy(partial, acc); // acc = the sum of the earlier K blocks
                release(partial);
            }
            const auto y = matmul(x, w, acc); // y = acc + x * w
            release(x);
            release(w);

            if (last_k_block)
            {
                const auto slot = reserve_back<mm_out_spec>(mm_out);
                if constexpr (fuse_bias)
                {
                    // kernel_main is not a template, so this branch is compiled even without FUSE_BIAS: its modifiers
                    // must be valid for the output buffer too (l1_acc only into the partials' format).
                    release(pack<pack_modifiers {.l1_acc = l1_acc && fuse_bias ? Toggle::Runtime : Toggle::Off}>(y, slot, block > 0));
                }
                else
                {
                    release(pack<pack_modifiers {.relu = relu_on_matmul_out}>(activation(y), slot));
                }
            }
            else
            {
                const auto slot = reserve_back<partials_spec>(partials);
                release(pack<pack_modifiers {.l1_acc = l1_acc ? Toggle::Runtime : Toggle::Off}>(y, slot, block > 0)); // add from the second K block on
            }
            release(y); // the Dest section back to the ring

            if constexpr (l1_acc)
            {
                const std::uint32_t next_block = block + 1;
                const bool next_to_last        = next_block == k_blocks - 1; // the next block is the last
                const bool before_next_to_last = next_block < k_blocks - 1;  // the next block is not the last either
                // Pop the partials so the next block reserves the same pages and accumulates into them. Without
                // FUSE_BIAS the one before last keeps them for the last block's reload.
                const bool pop = fuse_bias ? !last_k_block : before_next_to_last;
                if (pop)
                {
                    release(wait_front<partials_spec>(partials));
                }
                reload = !fuse_bias && next_to_last;
            }
            else
            {
                reload = k_blocks > 1;
            }
        }

        if constexpr (fuse_bias)
        {
            // The reader pushes the bias once; the kernel keeps it for every output block and pops it at the end.
            const auto bias_row = peek_front<bias_spec>(bias);
            const auto partial  = wait_front<partials_spec>(partials);
            const auto y        = add<add_modifiers {.bcast = Broadcast::Row}>(partial, bias_row, reserve_back<AccRing>()); // y = partial + bias's row 0
            release(partial);
            release(pack<pack_modifiers {.relu = pack_relu}>(activation(y), reserve_back<out_spec>(out)));
            release(y);
        }
    }
    if constexpr (fuse_bias)
    {
        pop_front<bias_spec>(bias); // balance the one push of the bias (1.0, num_blocks_w_dim == 1)
    }
}
