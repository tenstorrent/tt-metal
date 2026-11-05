// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// ``post_cast`` compute: ``ttnn.typecast(o fp32 -> bf16)`` call for call, as the op's eltwise_typecast.cpp does it:
// one tile per DST acquire, ``copy_tile`` of the fp32 input (its CB unpacked straight to DST), the explicit RNE of
// ``calculate_typecast_fp32_to_fp16b``, one pack.  Must run with ``fp32_dest_acc_en`` and APPROX = false, as the op's
// compute config does (set on the Python side).
// CBs: cb_in (0, fp32), cb_out (16, bf16).  Runtime args: 0 groups on this core.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/typecast.h"

using namespace ckernel;

namespace {
constexpr uint32_t cb_in = 0, cb_out = 16;
constexpr uint32_t group_tiles = 4;  // 128 / 32 column tiles of one head
constexpr uint32_t fp32_format = static_cast<uint32_t>(DataFormat::Float32);
constexpr uint32_t bf16_format = static_cast<uint32_t>(DataFormat::Float16_b);
}  // namespace

void kernel_main() {
    const uint32_t groups = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(cb_in, cb_out);
    copy_init(cb_in);

    for (uint32_t g = 0; g < groups; ++g) {
        CircularBuffer(cb_out).reserve_back(group_tiles);
        for (uint32_t d = 0; d < group_tiles; ++d) {
            tile_regs_acquire();
            CircularBuffer(cb_in).wait_front(1);
            copy_tile(cb_in, 0, 0);
            typecast_tile_init<fp32_format, bf16_format>();
            typecast_tile<fp32_format, bf16_format>(0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out);
            CircularBuffer(cb_in).pop_front(1);
            tile_regs_release();
        }
        CircularBuffer(cb_out).push_back(group_tiles);
    }
}
