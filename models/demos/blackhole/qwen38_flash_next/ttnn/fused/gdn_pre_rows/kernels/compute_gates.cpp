// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Group B compute: the gates and the z sigmoid in a 32-bit DEST with APPROX = false, the chain's configuration for
// every op here. Every fp32 CB read with copy_tile is in the program's unpack_to_dest_fp32, so fp32 operands arrive
// exactly instead of through the FPU's 19-bit source path.
// Unit kind 0 (the a/b tile): beta = sigmoid(fp32(b)); g = neg_exp_A * softplus(fp32(a) + dt_bias, beta = 1,
// threshold = 20). The chain packs fp32 between these ops, which is lossless, so fusing them in DEST is bitwise.
// Unit kind 1..12 (one z value head): bf16(sigmoid(fp32(z))) with the explicit RNE of typecast_tile.
//
// CBs: ab 20, z 21, const 22 (dt_bias tile 0, neg_exp_A tile 1), beta 23, g 24, sig 25.
// Compile-time args: none (the page maps live in the reader and the writer).  Runtime args: 0 units, then
// (tile row, kind) pairs.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "../../kernels/source_softplus.h"
#include "api/compute/eltwise_unary/typecast.h"

using namespace ckernel;

namespace {
constexpr uint32_t cb_ab = 20, cb_z = 21, cb_const = 22, cb_beta = 23, cb_g = 24, cb_sig = 25;
constexpr uint32_t head_tiles = 4, dt_bias_index = 0, neg_exp_a_index = 1;
constexpr uint32_t a_index = 0, b_index = 1;
constexpr uint32_t one_bits = 0x3F800000u, twenty_bits = 0x41A00000u;  // beta = 1, 1 / beta, threshold = 20
constexpr uint32_t fp32_format = (uint32_t)DataFormat::Float32, bf16_format = (uint32_t)DataFormat::Float16_b;
constexpr uint32_t dst0 = 0, dst1 = 1;

ALWI void gates() {
    CircularBuffer(cb_ab).wait_front(2);
    CircularBuffer(cb_const).wait_front(2);

    // beta = sigmoid(fp32(b))
    {
        CircularBuffer(cb_beta).reserve_back(1);
        pack_reconfig_data_format(cb_beta);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_ab);
        copy_init(cb_ab);
        copy_tile(cb_ab, b_index, dst0);
        sigmoid_tile_init<false>();
        sigmoid_tile<VectorMode::RC, false>(dst0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, cb_beta);
        tile_regs_release();
        CircularBuffer(cb_beta).push_back(1);
    }

    // g = neg_exp_A * softplus(fp32(a) + dt_bias)
    {
        CircularBuffer(cb_g).reserve_back(1);
        pack_reconfig_data_format(cb_g);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_ab);
        copy_init(cb_ab);
        copy_tile(cb_ab, a_index, dst0);
        reconfig_data_format_srca(cb_const);
        copy_init(cb_const);
        copy_tile(cb_const, dt_bias_index, dst1);
        add_binary_tile_init();
        add_binary_tile<ckernel::DstRoundingMode::NearestEven>(dst0, dst1, dst0);
        softplus_tile_init();
        qwen38_softplus_tile(dst0, one_bits, one_bits, twenty_bits);
        copy_tile(cb_const, neg_exp_a_index, dst1);
        mul_binary_tile_init();
        mul_binary_tile(dst0, dst1, dst0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, cb_g);
        tile_regs_release();
        CircularBuffer(cb_g).push_back(1);
    }
    CircularBuffer(cb_ab).pop_front(2);
}

ALWI void z_sigmoid() {
    CircularBuffer(cb_z).wait_front(head_tiles);
    CircularBuffer(cb_sig).reserve_back(head_tiles);
    pack_reconfig_data_format(cb_sig);
    for (uint32_t d = 0; d < head_tiles; ++d) {
        tile_regs_acquire();
        reconfig_data_format_srca(cb_z);
        copy_init(cb_z);
        copy_tile(cb_z, d, dst0);
        sigmoid_tile_init<false>();
        sigmoid_tile<VectorMode::RC, false>(dst0);
        typecast_tile_init<fp32_format, bf16_format>();
        typecast_tile<fp32_format, bf16_format>(dst0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, cb_sig);
        tile_regs_release();
    }
    CircularBuffer(cb_sig).push_back(head_tiles);
    CircularBuffer(cb_z).pop_front(head_tiles);
}
}  // namespace

void kernel_main() {
    const uint32_t units = get_arg_val<uint32_t>(0);
    constexpr uint32_t first_pair_arg = 1;

    compute_kernel_hw_startup(cb_ab, cb_const, cb_beta);
    CircularBuffer(cb_const).wait_front(2);

    for (uint32_t unit = 0; unit < units; ++unit) {
        const uint32_t kind = get_arg_val<uint32_t>(first_pair_arg + 2 * unit + 1);
        if (kind == 0) {
            gates();
        } else {
            z_sigmoid();
        }
    }
}
