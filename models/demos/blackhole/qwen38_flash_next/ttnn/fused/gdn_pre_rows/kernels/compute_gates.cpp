// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Group B compute: the gates and the z sigmoid, in a 32-BIT DEST (fp32_dest_acc_en = true) with APPROX = false --
// the chain's configuration for every op here.  Every fp32 CB this kernel reads with copy_tile is listed in the
// program's unpack_to_dest_fp32, so an fp32 operand arrives exactly instead of through the FPU's 19-bit source path.
//
// Unit kind 0 (the a/b tile of the tile row):
//   beta = ttnn.sigmoid(ttnn.typecast(b, fp32)): the typecast is a copy_tile (the packer does bf16 -> fp32), then
//     sigmoid_tile<VectorMode::RC, false> -- in a 32-bit DEST that is the accurate exp and the two-iteration
//     reciprocal, chosen by the DEST width and not by a flag;
//   g = ttnn.multiply(neg_exp_A, ttnn.softplus(ttnn.add(a_fp32, dt_bias), beta = 1, threshold = 20)): the SFPU add
//     against the full dt_bias tile, softplus with the op's own packed parameters, then the SFPU multiply by the
//     full neg_exp_A tile.  The chain packs fp32 between these ops, which is lossless, so fusing them in the
//     32-bit DEST is bitwise; the fp32 SFPU multiply has neither the bf16 RNE nor the 0 * x clamp.
// Unit kind 1..12 (one z value head): bf16(sigmoid(fp32(z))) -- copy_tile (the exact bf16 -> fp32 typecast),
//   the fp32 sigmoid, then the explicit RNE of typecast_tile<Float32, Float16_b>, packed bf16.
//
// CBs: ab 20, z 21, const 22 (dt_bias tile 0, neg_exp_A tile 1), beta 23, g 24, sig 25.
// Compile-time args: none (the page maps live in the reader and the writer).  Runtime args: 0 units, then
// (tile row, kind) pairs.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/softplus.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "../../kernels/zones.h"

using namespace ckernel;

namespace {
constexpr uint32_t CB_AB = 20, CB_Z = 21, CB_CONST = 22, CB_BETA = 23, CB_G = 24, CB_SIG = 25;
constexpr uint32_t HEAD_TILES = 4, CONST_DT_BIAS = 0, CONST_NEG_EXP_A = 1;
constexpr uint32_t A_INDEX = 0, B_INDEX = 1;
constexpr uint32_t F_ONE = 0x3F800000u, F_TWENTY = 0x41A00000u;  // beta = 1, 1 / beta, threshold = 20
constexpr uint32_t DF_FP32 = (uint32_t)DataFormat::Float32, DF_BF16 = (uint32_t)DataFormat::Float16_b;
constexpr uint32_t dst0 = 0, dst1 = 1;

ALWI void gates() {
    cb_wait_front(CB_AB, 2);
    cb_wait_front(CB_CONST, 2);

    // beta = sigmoid(fp32(b))
    {
        FUSED_ZONE("fz_gpr_cg_beta");
        cb_reserve_back(CB_BETA, 1);
        pack_reconfig_data_format(CB_BETA);
        tile_regs_acquire();
        reconfig_data_format_srca(CB_AB);
        copy_init(CB_AB);
        copy_tile(CB_AB, B_INDEX, dst0);
        sigmoid_tile_init<false>();
        sigmoid_tile<VectorMode::RC, false>(dst0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, CB_BETA);
        tile_regs_release();
        cb_push_back(CB_BETA, 1);
    }

    // g = neg_exp_A * softplus(fp32(a) + dt_bias)
    {
        FUSED_ZONE("fz_gpr_cg_g");
        cb_reserve_back(CB_G, 1);
        pack_reconfig_data_format(CB_G);
        tile_regs_acquire();
        reconfig_data_format_srca(CB_AB);
        copy_init(CB_AB);
        copy_tile(CB_AB, A_INDEX, dst0);
        reconfig_data_format_srca(CB_CONST);
        copy_init(CB_CONST);
        copy_tile(CB_CONST, CONST_DT_BIAS, dst1);
        add_binary_tile_init();
        add_binary_tile<ckernel::DstRoundingMode::NearestEven>(dst0, dst1, dst0);
        softplus_tile_init();
        softplus_tile(dst0, F_ONE, F_ONE, F_TWENTY);
        copy_tile(CB_CONST, CONST_NEG_EXP_A, dst1);
        mul_binary_tile_init();
        mul_binary_tile(dst0, dst1, dst0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, CB_G);
        tile_regs_release();
        cb_push_back(CB_G, 1);
    }
    cb_pop_front(CB_AB, 2);
}

ALWI void z_sigmoid() {
    FUSED_ZONE("fz_gpr_cg_sig");
    cb_wait_front(CB_Z, HEAD_TILES);
    cb_reserve_back(CB_SIG, HEAD_TILES);
    pack_reconfig_data_format(CB_SIG);
    for (uint32_t d = 0; d < HEAD_TILES; ++d) {
        tile_regs_acquire();
        reconfig_data_format_srca(CB_Z);
        copy_init(CB_Z);
        copy_tile(CB_Z, d, dst0);
        sigmoid_tile_init<false>();
        sigmoid_tile<VectorMode::RC, false>(dst0);
        typecast_tile_init<DF_FP32, DF_BF16>();
        typecast_tile<DF_FP32, DF_BF16>(dst0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, CB_SIG);
        tile_regs_release();
    }
    cb_push_back(CB_SIG, HEAD_TILES);
    cb_pop_front(CB_Z, HEAD_TILES);
}
}  // namespace

void kernel_main() {
    const uint32_t units = get_arg_val<uint32_t>(0);
    constexpr uint32_t PAIRS = 1;

    compute_kernel_hw_startup(CB_AB, CB_CONST, CB_BETA);
    cb_wait_front(CB_CONST, 2);

    for (uint32_t unit = 0; unit < units; ++unit) {
        const uint32_t kind = get_arg_val<uint32_t>(PAIRS + 2 * unit + 1);
        if (kind == 0) {
            gates();
        } else {
            z_sigmoid();
        }
    }
}
