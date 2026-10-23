// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// ``post_cast`` compute: ``ttnn.typecast(o fp32 -> bf16)`` call for call.  The op's own kernel
// (operations/copy/typecast/device/kernels/compute/eltwise_typecast.cpp) is one tile per DST acquire:
// ``copy_tile`` of the fp32 input (its CB unpacked straight to DST, ``preserve_fp32_precision`` on an fp32 input:
// typecast.cpp:33-38), then the format-selected LLK pair, then one pack.  fp32 -> bf16 is
// ``calculate_typecast_fp32_to_fp16b`` (the explicit round-to-nearest-even of typecast.h), so the kernel runs with
// ``fp32_dest_acc_en`` (the op sets it from ``preserve_fp32_precision``) and APPROX = false (the op's compute config
// is Precise: typecast_program_factory.cpp:73).  Both are the Python side's compute-kernel settings.
// CBs: CB_IN (0, fp32), CB_OUT (16, bf16).  Runtime args: 0 groups on this core.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "../../kernels/zones.h"

using namespace ckernel;

namespace {
constexpr uint32_t CB_IN = 0, CB_OUT = 16;
constexpr uint32_t GROUP = 4;  // 128 / 32 column tiles of one head
constexpr uint32_t DF_FP32 = static_cast<uint32_t>(DataFormat::Float32);
constexpr uint32_t DF_BF16 = static_cast<uint32_t>(DataFormat::Float16_b);
}  // namespace

void kernel_main() {
    const uint32_t groups = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(CB_IN, CB_OUT);
    copy_init(CB_IN);

    for (uint32_t g = 0; g < groups; ++g) {
        FUSED_ZONE("fz_gpo_cc_group");
        cb_reserve_back(CB_OUT, GROUP);
        for (uint32_t d = 0; d < GROUP; ++d) {
            tile_regs_acquire();
            cb_wait_front(CB_IN, 1);
            copy_tile(CB_IN, 0, 0);
            typecast_tile_init<DF_FP32, DF_BF16>();
            typecast_tile<DF_FP32, DF_BF16>(0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, CB_OUT);
            cb_pop_front(CB_IN, 1);
            tile_regs_release();
        }
        cb_push_back(CB_OUT, GROUP);
    }
}
