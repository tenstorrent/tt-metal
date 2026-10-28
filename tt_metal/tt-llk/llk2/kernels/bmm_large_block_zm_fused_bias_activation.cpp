// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ttnn's bmm_large_block_zm_fused_bias_activation compute kernel on LLK 2.0, as tt-metal would build it. NOT compiled
// by the tt-llk harness: tests/sources/llk2_bmm_fused_compute_api_test.cpp runs the same body
// (bmm_large_block_zm_fused_bias_activation.h) there.
//
// The dataflow buffers are tt-metal's own DataflowBuffer (api/dataflow/dataflow_buffer.h), used as 1.0 uses them:
// wait_front / pop_front run on unpack, reserve_back / push_back on pack, and each pointer is valid only there (0 on
// math). The body makes Tensors at those pointers (dataflow_tensor.h); llk2 itself knows nothing of buffers.
//
// What this file still assumes from tt-metal that does not exist today:
//   - Each CB's data format is a named compile-time arg (<cb>_format), the same on all three TRISCs: the Tensors'
//     specs (and through them the Resolver) need it at compile time on every thread. Today each TRISC sees only its
//     own generated table (unpack_src_format[] on unpack, pack_dst_format[] on pack).
//   - Every CB has pages of one tile (the Tensors are counted in tiles, as 1.0 counts them).
//   - The partials CB (cb_intermed0) holds exactly one output block, so L1 acc reserves the same pages again (true
//     of the 1.0 host setup for this kernel; nothing here can check it at compile time).
// And as today: the firmware sets up each TRISC's CB pointers and zeroes the stream counters before the kernel
// starts, and the Dest mode and math fidelity come from DST_ACCUM_MODE and MATH_FIDELITY.
//
// What the 1.0 kernel does that this one does not (yet): subblocks smaller than the per-core block, in0 transpose,
// in1 transpose, untilize_out, batch from the reader (sparsity), BIAS_FULL_BLOCK / element-wise bias,
// MATMUL_DRAM_SHARDED padding, SKIP_COMPUTE, and the activations other than silu, relu_max and hardtanh.

#include <cstdint>

#define LLK2_DEST_FP32     DST_ACCUM_MODE
#define LLK2_MATH_FIDELITY MATH_FIDELITY
#include "api/dataflow/dataflow_buffer.h"
#include "bmm_large_block_zm_fused_bias_activation.h"
#ifdef SFPU_ACTIVATION
#include "ttnn/operations/matmul/shared_with_host/activation_type.hpp"
#endif

namespace
{

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

static_assert(in0_num_subblocks == 1 && in1_num_subblocks == 1, "LLK 2.0: one output subblock per block (out_subblock = the per-core block)");

#ifdef SFPU_ACTIVATION
using ttnn::operations::matmul::KernelActivation;
constexpr KernelActivation kernel_activation = static_cast<KernelActivation>(get_named_compile_time_arg_val("activation_type"));
static_assert(
    kernel_activation == KernelActivation::SILU || kernel_activation == KernelActivation::RELU6 || kernel_activation == KernelActivation::HARDTANH,
    "LLK 2.0 kernel: only silu, relu6 and hardtanh are wired up (bmm_fused_activation)");
constexpr llk2::VecFunction activation = kernel_activation == KernelActivation::SILU    ? llk2::VecFunction::Silu
                                         : kernel_activation == KernelActivation::RELU6 ? llk2::VecFunction::ReluMax
                                                                                        : llk2::VecFunction::Hardtanh;
constexpr std::uint32_t relu6_max      = 0x40c00000u; // 6.0, the 1.0 kernel's RELU6 default when param0 is 0
constexpr std::uint32_t param0         = get_named_compile_time_arg_val("activation_param0");
constexpr std::uint32_t activation_p0  = kernel_activation == KernelActivation::RELU6 && param0 == 0 ? relu6_max : param0;
constexpr std::uint32_t activation_p1  = get_named_compile_time_arg_val("activation_param1");
#else
constexpr llk2::VecFunction activation = llk2::VecFunction::None;
constexpr std::uint32_t activation_p0  = 0;
constexpr std::uint32_t activation_p1  = 0;
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

inline constexpr llk2::bmm_fused_config config {
    .rt_dim          = out_subblock_h * in0_num_subblocks,
    .ct_dim          = out_subblock_w * in1_num_subblocks,
    .kb_dim          = in0_block_w,
    .k_blocks        = k_blocks,
    .out_blocks      = batch * num_blocks_h_dim * num_blocks_w_dim,
    .l1_acc          = l1_acc,
    .fuse_bias       = fuse_bias,
    .pack_relu       = pack_relu,
    .activation      = activation,
    .activation_p0   = activation_p0,
    .activation_p1   = activation_p1,
    .in0_format      = LLK2_CB_FORMAT("cb_in0"),
    .in1_format      = LLK2_CB_FORMAT("cb_in1"),
    .bias_format     = bias_format,
    .partials_format = LLK2_CB_FORMAT("cb_intermed0"),
    .out_format      = LLK2_CB_FORMAT("cb_out"),
};

} // namespace

void kernel_main()
{
    DataflowBuffer in0(get_named_compile_time_arg_val("cb_in0"));
    DataflowBuffer in1(get_named_compile_time_arg_val("cb_in1"));
    DataflowBuffer bias(bias_cb);
    DataflowBuffer partials(get_named_compile_time_arg_val("cb_intermed0"));
    DataflowBuffer out(get_named_compile_time_arg_val("cb_out"));
    llk2::bmm_large_block_zm_fused_bias_activation<config, llk2::kernel_dest_format>(in0, in1, bias, partials, out);
}
