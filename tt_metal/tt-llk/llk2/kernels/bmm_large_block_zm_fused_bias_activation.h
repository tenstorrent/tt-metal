// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The body of ttnn's bmm_large_block_zm_fused_bias_activation.cpp (matmul/device/kernels/compute/) on LLK 2.0:
// only Tensors, the Compute API (matmul, copy, add, vec_*, pack), the Dest sync API and the dataflow buffer API.
// One body for all three threads; no LLK calls, no init / uninit / reconfig, no per-thread code.
//
// The harness test (tests/sources/llk2_bmm_fused_compute_api_test.cpp) and the tt-metal kernel
// (bmm_large_block_zm_fused_bias_activation.cpp next to this file) both call this body, so they cannot drift apart.
//
// Flow, per output block (the 1.0 kernel's bh / bw loop; one batch):
//   for each K block:
//     x = wait_front(in0), w = wait_front(in1)                   in0 [RT, KB] and in1 [KB, CT] tiles
//     acc = dest_acquire()                                       a zeroed Dest section
//     reload: p = wait_front(partials); copy(p, acc); release(p) the previous blocks' sum back into Dest
//     matmul(x, w, acc)                                          acc += x * w
//     dest_commit(acc)
//     last block: pack into the output buffer (with the activation from the pack thread and PACK_RELU), or, with
//                 FUSE_BIAS, into the partials buffer
//     else:       pack into the partials buffer (spill)
//     dest_release(acc); release(x), release(w)
//   FUSE_BIAS:
//     p = wait_front(partials), bias = wait_front(bias)
//     acc = dest_acquire(); add<Row>(p, bias, acc); dest_commit(acc); release(p)
//     activation from the pack thread; pack<relu>(acc, reserve_back(out)); dest_release(acc)
//
// Accumulating over K blocks, two ways (as the 1.0 kernel):
//   spill (default)  each block but the last packs its sum to the partials buffer, and the next block copies it back
//                    into Dest before its matmul adds to it
//   L1 acc           the packer adds into the partials buffer in L1 (pack<{.l1_acc = Toggle::Runtime}>, on from the second block).
//                    The partials buffer holds one block, so popping it and reserving again gives the same pages
//                    back. Without FUSE_BIAS the last block still reloads the sum (one copy) and packs to the output;
//                    with FUSE_BIAS every block packs into the partials buffer and the bias stage reads it
//
// Subblocks: one output subblock per block (out_subblock = [RT, CT], the whole per-core block). Smaller subblocks
// would need views of the in0 / in1 Tensors (a strided slice), which matmul's Resolver does not take yet.

#pragma once

#include <cstdint>

#include "compute_api/add.h"
#include "compute_api/copy.h"
#include "compute_api/dest.h"
#include "compute_api/matmul.h"
#include "compute_api/pack.h"
#include "compute_api/vec.h"
#include "tensor/dataflow_buffer.h"

namespace llk2
{

/// What the 1.0 kernel takes as compile-time args and defines, as one struct.
struct bmm_fused_config
{
    std::uint32_t rt_dim;        ///< output block height in tiles (out_subblock_h * in0_num_subblocks)
    std::uint32_t ct_dim;        ///< output block width in tiles (out_subblock_w * in1_num_subblocks)
    std::uint32_t kb_dim;        ///< K tiles per block (in0_block_w)
    std::uint32_t k_blocks;      ///< num_blocks_inner_dim
    std::uint32_t out_blocks;    ///< output blocks (num_blocks_h_dim * num_blocks_w_dim * batch, one loop here)
    bool l1_acc;                 ///< PACKER_L1_ACC
    bool fuse_bias;              ///< FUSE_BIAS, row broadcast
    bool pack_relu;              ///< PACK_RELU
    VecFunction activation;      ///< SFPU_ACTIVATION: the activation, or None
    std::uint32_t activation_p0; ///< activation_param0 (FP32 bit pattern): relu_max's max, hardtanh's min
    std::uint32_t activation_p1; ///< activation_param1 (FP32 bit pattern): hardtanh's max
};

/// The activation on acc, driven from the pack thread between dest_commit and pack (the 1.0 kernel's
/// apply_activation_from_pack).
template <const bmm_fused_config& C, const TensorSpec<tensor_rank>& D, class R>
inline __attribute__((always_inline)) void bmm_fused_activation(const Tensor<D, R>& acc)
{
    constexpr vec_modifiers mods {.thread = Thread::Pack};
    if constexpr (C.activation == VecFunction::Silu)
    {
        vec_silu<mods>(acc);
    }
    else if constexpr (C.activation == VecFunction::ReluMax)
    {
        vec_relu_max<mods>(acc, C.activation_p0);
    }
    else if constexpr (C.activation == VecFunction::Hardtanh)
    {
        vec_hardtanh<mods>(acc, C.activation_p0, C.activation_p1);
    }
    else
    {
        static_assert(C.activation == VecFunction::None, "only silu, relu_max and hardtanh are wired up here; add the other vec_ calls the same way");
    }
}

/// The kernel body. DestFormat is the kernel's Dest format; the buffers are the 1.0 kernel's in0, in1, bias,
/// mm_partials (intermed0) and out CBs. Bias is unused without fuse_bias.
template <
    const bmm_fused_config& C,
    DataFormat DestFormat,
    const DataflowBufferSpec& In0,
    const DataflowBufferSpec& In1,
    const DataflowBufferSpec& Bias,
    const DataflowBufferSpec& Partials,
    const DataflowBufferSpec& Out>
inline __attribute__((always_inline)) void bmm_large_block_zm_fused_bias_activation()
{
    static_assert(
        Partials.pages * Partials.page_bytes == C.rt_dim * C.ct_dim * buffer_tile_bytes<Partials>,
        "the partials buffer holds exactly one output block: L1 acc relies on reserving the same pages again");
    static constexpr const TensorSpec<tensor_rank>& acc_spec = dst_spec<C.rt_dim * default_tile_dim, C.ct_dim * default_tile_dim, DestFormat>;
    // The last K block packs to the output, or with FUSE_BIAS to the partials the bias stage reads (1.0's mm_out_dfb).
    static constexpr const DataflowBufferSpec& MmOut = C.fuse_bias ? Partials : Out;
    constexpr bool relu_on_matmul_out                = C.pack_relu && !C.fuse_bias;

    for (std::uint32_t block_out = 0; block_out < C.out_blocks; ++block_out)
    {
        bool reload = false;
        for (std::uint32_t block = 0; block < C.k_blocks; ++block)
        {
            const bool last_k_block = block == C.k_blocks - 1;

            const auto x   = wait_front<In0, C.rt_dim, C.kb_dim>();
            const auto w   = wait_front<In1, C.kb_dim, C.ct_dim>();
            const auto acc = dest_acquire<acc_spec>();
            if (reload)
            {
                const auto partial = wait_front<Partials, C.rt_dim, C.ct_dim>();
                copy(partial, acc); // acc = the sum of the earlier K blocks
                release(partial);
            }
            matmul(x, w, acc); // acc += x * w
            dest_commit(acc);

            if (last_k_block)
            {
                const auto out = reserve_back<MmOut, C.rt_dim, C.ct_dim>();
                if constexpr (!C.fuse_bias)
                {
                    bmm_fused_activation<C>(acc);
                }
                pack<pack_modifiers {.relu = relu_on_matmul_out, .l1_acc = C.l1_acc && C.fuse_bias ? Toggle::Runtime : Toggle::Off}>(acc, out, block > 0);
                release(out);
            }
            else
            {
                const auto partial = reserve_back<Partials, C.rt_dim, C.ct_dim>();
                pack<pack_modifiers {.l1_acc = C.l1_acc ? Toggle::Runtime : Toggle::Off}>(acc, partial, block > 0); // add from the second K block on
                release(partial);
            }
            dest_release(acc);

            if constexpr (C.l1_acc)
            {
                const std::uint32_t next_block = block + 1;
                const bool next_to_last        = next_block == C.k_blocks - 1; // the next block is the last
                const bool before_next_to_last = next_block < C.k_blocks - 1;  // the next block is not the last either
                // Pop the partials so the next block reserves the same pages and accumulates into them. Without
                // FUSE_BIAS the one before last keeps them for the last block's reload.
                const bool pop = C.fuse_bias ? !last_k_block : before_next_to_last;
                if (pop)
                {
                    release(wait_front<Partials, C.rt_dim, C.ct_dim>());
                }
                reload = !C.fuse_bias && next_to_last;
            }
            else
            {
                reload = C.k_blocks > 1;
            }
            release(x);
            release(w);
        }

        if constexpr (C.fuse_bias)
        {
            // The bias is pushed once and never popped (1.0, num_blocks_w_dim == 1): waiting again returns the same pages.
            const auto bias    = wait_front<Bias, 1 /* one tile row: the bias */, C.ct_dim>();
            const auto partial = wait_front<Partials, C.rt_dim, C.ct_dim>();
            const auto acc     = dest_acquire<acc_spec>();
            add<add_modifiers {.bcast = Broadcast::Row}>(partial, bias, acc); // acc = partial + bias's row 0
            dest_commit(acc);
            release(partial);

            const auto out = reserve_back<Out, C.rt_dim, C.ct_dim>();
            bmm_fused_activation<C>(acc);
            pack<pack_modifiers {.relu = C.pack_relu}>(acc, out);
            release(out);
            dest_release(acc);
        }
    }
}

} // namespace llk2
