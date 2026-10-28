// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ttnn's bmm_large_block_zm_fused_bias_activation compute kernel on LLK 2.0, as tt-metal would build it. NOT compiled
// by the tt-llk harness: tests/sources/llk2_bmm_fused_compute_api_test.cpp runs the same body
// (bmm_large_block_zm_fused_bias_activation.h) there.
//
// What this file assumes from tt-metal, none of which exists today:
//   - Every TRISC gets both pointers of every CB. Today trisc.cc sets up only the read pointer on TRISC0 and only
//     the write pointer on TRISC2, and the math TRISC has no cb_interface at all; the dataflow buffer API keeps
//     both on all three (tensor/dataflow_buffer.h), so cb_init_read and cb_init_write are true on every TRISC.
//     dataflow_buffer_state<id> (llk_lib/dataflow_buffer.h) becomes get_local_cb_interface(id), and the firmware's
//     setup replaces setup_dataflow_buffer.
//   - The firmware zeroes every CB's stream counters before the kernel starts (init_sync_registers, as today).
//     Nothing here zeroes them and no barrier is needed.
//   - A CB's format, page size and page count are compile time. Today the host passes them at run time (the CB
//     config in L1, unpack_src_format[] / pack_dst_format[] tables); here they come in as named compile-time args
//     (<cb>_format, <cb>_page_size, <cb>_num_pages), which the host would have to emit.
//   - The kernel's Dest mode and math fidelity come from DST_ACCUM_MODE and MATH_FIDELITY, as today.
//
// What the 1.0 kernel does that this one does not (yet): subblocks smaller than the per-core block, in0 transpose,
// in1 transpose, untilize_out, batch from the reader (sparsity), BIAS_FULL_BLOCK / element-wise bias,
// MATMUL_DRAM_SHARDED padding, SKIP_COMPUTE, and the activations other than silu, relu_max and hardtanh.

#include <cstdint>

#define LLK2_DEST_FP32     DST_ACCUM_MODE
#define LLK2_MATH_FIDELITY MATH_FIDELITY
#include "bmm_large_block_zm_fused_bias_activation.h"
#ifdef SFPU_ACTIVATION
#include "ttnn/operations/matmul/shared_with_host/activation_type.hpp"
#endif

namespace
{

/// A CB as the dataflow buffer API takes it, from the host's named compile-time args for CB Name.
#define LLK2_CB(name)                                                                                                                  \
    llk2::DataflowBufferSpec                                                                                                           \
    {                                                                                                                                  \
        .id = get_named_compile_time_arg_val(name), .format = static_cast<DataFormat>(get_named_compile_time_arg_val(name "_format")), \
        .page_bytes = get_named_compile_time_arg_val(name "_page_size"), .pages = get_named_compile_time_arg_val(name "_num_pages")    \
    }

inline constexpr llk2::DataflowBufferSpec in0_buf      = LLK2_CB("cb_in0");
inline constexpr llk2::DataflowBufferSpec in1_buf      = LLK2_CB("cb_in1");
inline constexpr llk2::DataflowBufferSpec bias_buf     = LLK2_CB("cb_bias");
inline constexpr llk2::DataflowBufferSpec partials_buf = LLK2_CB("cb_intermed0");
inline constexpr llk2::DataflowBufferSpec out_buf      = LLK2_CB("cb_out");

constexpr std::uint32_t in0_block_w       = get_compile_time_arg_val(0); // K tiles per block
constexpr std::uint32_t in0_num_subblocks = get_compile_time_arg_val(1);
constexpr std::uint32_t in1_num_subblocks = get_compile_time_arg_val(4);
constexpr std::uint32_t k_blocks          = get_compile_time_arg_val(7); // num_blocks_inner_dim
constexpr std::uint32_t num_blocks_w_dim  = get_compile_time_arg_val(8);
constexpr std::uint32_t num_blocks_h_dim  = get_compile_time_arg_val(9);
constexpr std::uint32_t out_subblock_h    = get_compile_time_arg_val(10);
constexpr std::uint32_t out_subblock_w    = get_compile_time_arg_val(11);
constexpr std::uint32_t batch             = get_compile_time_arg_val(13);

static_assert(in0_num_subblocks == 1 && in1_num_subblocks == 1, "LLK 2.0: one output subblock per block (out_subblock = the per-core block)");
static_assert(num_blocks_w_dim == 1, "LLK 2.0: the bias is waited for once and kept, which needs num_blocks_w_dim == 1");

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
constexpr bool fuse_bias = true;
#else
constexpr bool fuse_bias = false;
#endif
#ifdef PACK_RELU
constexpr bool pack_relu = true;
#else
constexpr bool pack_relu = false;
#endif

inline constexpr llk2::bmm_fused_config config {
    .rt_dim        = out_subblock_h * in0_num_subblocks,
    .ct_dim        = out_subblock_w * in1_num_subblocks,
    .kb_dim        = in0_block_w,
    .k_blocks      = k_blocks,
    .out_blocks    = batch * num_blocks_h_dim * num_blocks_w_dim,
    .l1_acc        = l1_acc,
    .fuse_bias     = fuse_bias,
    .pack_relu     = pack_relu,
    .activation    = activation,
    .activation_p0 = activation_p0,
    .activation_p1 = activation_p1,
};

constexpr DataFormat dest_format = llk2::kernel_settings.dest_fp32 ? DataFormat::Float32 : DataFormat::Float16_b;

} // namespace

void kernel_main()
{
    llk2::bmm_large_block_zm_fused_bias_activation<config, dest_format, in0_buf, in1_buf, bias_buf, partials_buf, out_buf>();
}
