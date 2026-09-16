// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"  // legacy compute_kernel_hw_startup + pack_tile (id-free block ops isolated)
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
#include "api/compute/experimental/2_0/matmul.h"
#include "tests/tt_metal/tt_metal/test_kernels/compute/cb_operand_helpers.h"

// Id-free (2.0) block matmul kernel, classic circular buffers. Mirrors matmul_block_legacy.cpp exactly (same
// compile args {ct_dim, rt_dim, kt_dim}, same single-block structure) but drives the block matmul with the
// id-free ops: one LLKOperand per input (in0 -> SrcB, in1 -> SrcA), format+geometry as compile-time NTTPs,
// with the L1 base address + the runtime block dims as the only runtime state. hw_startup + pack are legacy
// CB-id in BOTH kernels; only the matmul_block_init / matmul_block calls differ, isolating them. Output must be
// bit-identical to matmul_block_legacy.cpp.
void kernel_main() {
    constexpr std::uint32_t ct_dim = get_compile_time_arg_val(0);
    constexpr std::uint32_t rt_dim = get_compile_time_arg_val(1);
    constexpr std::uint32_t kt_dim = get_compile_time_arg_val(2);
    constexpr std::uint32_t in0_block_tile_cnt = rt_dim * kt_dim;  // A block
    constexpr std::uint32_t in1_block_tile_cnt = kt_dim * ct_dim;  // B block
    constexpr std::uint32_t out_block_tile_cnt = rt_dim * ct_dim;  // C block

    CircularBuffer cb0(tt::CBIndex::c_0);
    CircularBuffer cb1(tt::CBIndex::c_1);
    CircularBuffer cb16(tt::CBIndex::c_16);

    constexpr auto in0_cb = experimental::Cb<tt::CBIndex::c_0>{};
    constexpr auto in1_cb = experimental::Cb<tt::CBIndex::c_1>{};
    constexpr auto d0 = experimental::to_llk_mem_descriptor(in0_cb);
    constexpr auto d1 = experimental::to_llk_mem_descriptor(in1_cb);
    using In0Op = experimental::LLKOperand<static_cast<DataFormat>(d0.format), d0.shape>;
    using In1Op = experimental::LLKOperand<static_cast<DataFormat>(d1.format), d1.shape>;

    // hw_startup + pack stay legacy CB-id in both kernels; only matmul_block(_init) is id-free here.
    compute_kernel_hw_startup<SrcOrder::Reverse>(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16);
    experimental::matmul_block_init(
        In0Op(in0_cb.read_address()), In1Op(in1_cb.read_address()), false, ct_dim, rt_dim, kt_dim);

    tile_regs_acquire();
    cb0.wait_front(in0_block_tile_cnt);
    cb1.wait_front(in1_block_tile_cnt);
    experimental::matmul_block(
        In0Op(in0_cb.read_address()), In1Op(in1_cb.read_address()), 0, 0, 0, ct_dim, rt_dim, kt_dim);
    cb0.pop_front(in0_block_tile_cnt);
    cb1.pop_front(in1_block_tile_cnt);
    tile_regs_commit();
    tile_regs_wait();

    // Pack out (legacy CB-id in both kernels).
    cb16.reserve_back(out_block_tile_cnt);
    for (std::uint32_t i = 0; i < out_block_tile_cnt; ++i) {
        pack_tile(i, tt::CBIndex::c_16);
    }
    cb16.push_back(out_block_tile_cnt);

    tile_regs_release();
}
