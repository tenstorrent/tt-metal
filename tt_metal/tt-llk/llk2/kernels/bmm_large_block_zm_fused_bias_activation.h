// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The body of ttnn's bmm_large_block_zm_fused_bias_activation.cpp (matmul/device/kernels/compute/) on LLK 2.0:
// Tensors, the Compute API (matmul, copy, add, vec_*, pack) and the Dest sync API. One body for all three threads;
// no LLK calls, no init / uninit / reconfig.
//
// The dataflow buffers are the kernel's, as in 1.0: the body calls wait_front / pop_front / reserve_back / push_back
// on them and makes Tensors at their pointers (dataflow_tensor.h). Buffer is tt-metal's DataflowBuffer
// (bmm_large_block_zm_fused_bias_activation.cpp next to this file) or the harness's stand-in
// (tests/helpers/test_dataflow_buffer.h, used by tests/sources/llk2_bmm_fused_compute_api_test.cpp). A buffer's pointer is
// valid only on the thread that owns that side (read: unpack, write: pack), and only that thread waits or signals;
// llk2 reads an input Tensor's address only on unpack and an output's only on pack, so that is enough.
//
// Flow, per output block (the 1.0 kernel's bh / bw loop; one batch):
//   for each K block:
//     in0.wait_front, in1.wait_front; x, w at their read pointers  in0 [RT, KB] and in1 [KB, CT] tiles
//     acc = dest_acquire()                                         a zeroed Dest section
//     reload: partials.wait_front; copy(p, acc); partials.pop_front  the previous blocks' sum back into Dest
//     matmul(x, w, acc)                                            acc += x * w
//     dest_commit(acc)
//     last block: pack into the output buffer (with the activation from the pack thread and PACK_RELU), or, with
//                 FUSE_BIAS, into the partials buffer
//     else:       pack into the partials buffer (spill)
//     dest_release(acc); in0.pop_front, in1.pop_front
//   FUSE_BIAS:
//     bias.wait_front (first output block only: the bias stays, popped once at the end of the kernel, as 1.0)
//     partials.wait_front; acc = dest_acquire(); add<Row>(p, bias, acc); dest_commit(acc); partials.pop_front
//     out.reserve_back; activation from the pack thread; pack<relu>(acc, o); out.push_back; dest_release(acc)
//
// Accumulating over K blocks, two ways (as the 1.0 kernel):
//   spill (default)  each block but the last packs its sum to the partials buffer, and the next block copies it back
//                    into Dest before its matmul adds to it
//   L1 acc           the packer adds into the partials buffer in L1 (pack<{.l1_acc = Toggle::Runtime}>, on from the second block).
//                    The partials buffer must hold exactly one output block, so popping it and reserving again gives
//                    the same pages back. Without FUSE_BIAS the last block still reloads the sum (one copy) and packs
//                    to the output; with FUSE_BIAS every block packs into the partials buffer and the bias stage reads it
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
#include "dataflow_tensor.h"

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
    DataFormat in0_format;       ///< in0's tiles (cb_in0's format)
    DataFormat in1_format;       ///< in1's tiles (cb_in1's format)
    DataFormat bias_format;      ///< the bias tiles (cb_bias's format)
    DataFormat partials_format;  ///< mm_partials' tiles (cb_intermed0's format)
    DataFormat out_format;       ///< the output tiles (cb_out's format)
};

/// The activation on acc, driven from the pack thread between dest_commit and pack (the 1.0 kernel's
/// apply_activation_from_pack).
template <const bmm_fused_config& C, const TensorSpec<tensor_rank>& D>
inline __attribute__((always_inline)) void bmm_fused_activation(const Tensor<D>& acc)
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
/// mm_partials (intermed0) and out CBs, each with pages of one tile. bias is unused without fuse_bias.
template <const bmm_fused_config& C, DataFormat DestFormat, class Buffer>
inline __attribute__((always_inline)) void bmm_large_block_zm_fused_bias_activation(
    Buffer& in0_buf, Buffer& in1_buf, Buffer& bias_buf, Buffer& partials_buf, Buffer& out_buf)
{
    constexpr std::uint32_t tile_rows = C.rt_dim * default_tile_dim, tile_cols = C.ct_dim * default_tile_dim, k_cols = C.kb_dim * default_tile_dim;
    static constexpr const TensorSpec<tensor_rank>& x_spec        = l1_spec<tile_rows, k_cols, C.in0_format>;            // [RT, KB] tiles
    static constexpr const TensorSpec<tensor_rank>& w_spec        = l1_spec<k_cols, tile_cols, C.in1_format>;            // [KB, CT] tiles
    static constexpr const TensorSpec<tensor_rank>& bias_spec     = l1_spec<default_tile_dim, tile_cols, C.bias_format>; // one tile row
    static constexpr const TensorSpec<tensor_rank>& partials_spec = l1_spec<tile_rows, tile_cols, C.partials_format>;    // [RT, CT] tiles
    static constexpr const TensorSpec<tensor_rank>& out_spec      = l1_spec<tile_rows, tile_cols, C.out_format>;         // [RT, CT] tiles
    static constexpr const TensorSpec<tensor_rank>& acc_spec      = dst_spec<tile_rows, tile_cols, DestFormat>;
    constexpr std::uint16_t x_tiles                               = C.rt_dim * C.kb_dim;
    constexpr std::uint16_t w_tiles                               = C.kb_dim * C.ct_dim;
    constexpr std::uint16_t out_tiles                             = C.rt_dim * C.ct_dim;
    constexpr std::uint16_t bias_tiles                            = C.ct_dim;
    // The last K block packs to the output, or with FUSE_BIAS to the partials the bias stage reads (1.0's mm_out_dfb).
    Buffer& mm_out_buf                                          = C.fuse_bias ? partials_buf : out_buf;
    static constexpr const TensorSpec<tensor_rank>& mm_out_spec = C.fuse_bias ? partials_spec : out_spec;
    constexpr bool relu_on_matmul_out                           = C.pack_relu && !C.fuse_bias;

    for (std::uint32_t block_out = 0; block_out < C.out_blocks; ++block_out)
    {
        bool reload = false;
        for (std::uint32_t block = 0; block < C.k_blocks; ++block)
        {
            const bool last_k_block = block == C.k_blocks - 1;

            in0_buf.wait_front(x_tiles);
            in1_buf.wait_front(w_tiles);
            const auto x   = front<x_spec>(in0_buf);
            const auto w   = front<w_spec>(in1_buf);
            const auto acc = dest_acquire<acc_spec>();
            if (reload)
            {
                partials_buf.wait_front(out_tiles);
                copy(front<partials_spec>(partials_buf), acc); // acc = the sum of the earlier K blocks
                partials_buf.pop_front(out_tiles);
            }
            matmul(x, w, acc); // acc += x * w
            dest_commit(acc);

            if (last_k_block)
            {
                mm_out_buf.reserve_back(out_tiles);
                const auto out = back<mm_out_spec>(mm_out_buf);
                if constexpr (!C.fuse_bias)
                {
                    bmm_fused_activation<C>(acc);
                }
                pack<pack_modifiers {.relu = relu_on_matmul_out, .l1_acc = C.l1_acc && C.fuse_bias ? Toggle::Runtime : Toggle::Off}>(acc, out, block > 0);
                mm_out_buf.push_back(out_tiles);
            }
            else
            {
                partials_buf.reserve_back(out_tiles);
                const auto partial = back<partials_spec>(partials_buf);
                pack<pack_modifiers {.l1_acc = C.l1_acc ? Toggle::Runtime : Toggle::Off}>(acc, partial, block > 0); // add from the second K block on
                partials_buf.push_back(out_tiles);
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
                    partials_buf.wait_front(out_tiles);
                    partials_buf.pop_front(out_tiles);
                }
                reload = !C.fuse_bias && next_to_last;
            }
            else
            {
                reload = C.k_blocks > 1;
            }
            in0_buf.pop_front(x_tiles);
            in1_buf.pop_front(w_tiles);
        }

        if constexpr (C.fuse_bias)
        {
            // The reader pushes the bias once; the kernel keeps it for every output block and pops it at the end.
            if (block_out == 0)
            {
                bias_buf.wait_front(bias_tiles);
            }
            partials_buf.wait_front(out_tiles);
            const auto acc = dest_acquire<acc_spec>();
            add<add_modifiers {.bcast = Broadcast::Row}>(front<partials_spec>(partials_buf), front<bias_spec>(bias_buf), acc); // acc = partial + bias's row 0
            dest_commit(acc);
            partials_buf.pop_front(out_tiles);

            out_buf.reserve_back(out_tiles);
            const auto out = back<out_spec>(out_buf);
            bmm_fused_activation<C>(acc);
            pack<pack_modifiers {.relu = C.pack_relu}>(acc, out);
            out_buf.push_back(out_tiles);
            dest_release(acc);
        }
    }
    if constexpr (C.fuse_bias)
    {
        bias_buf.pop_front(bias_tiles); // balance the one push of the bias (1.0, num_blocks_w_dim == 1)
    }
}

} // namespace llk2
