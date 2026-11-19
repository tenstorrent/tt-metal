// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Router top-k, compute side: ONE exact fp32 tile add, choice = scores + bias, on the row's score tile
// (the 288 scores of this core's token row packed as 9 x 32 into one fp32 tile by the writer; the bias
// tile holds the 288 biases at the same positions). Both operands are unpacked straight into the fp32
// dest (UnpackToDestFp32) and added on the SFPU, which is the same fp32 path ttnn.add takes for fp32
// tensors, so the biased scores match the composite's bit for bit (measured 0 / 9216 mismatches vs torch).
// The writer ranks the result (dataflow side); this kernel only produces the choice tile.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

namespace {
constexpr uint32_t CB_S = get_compile_time_arg_val(0);   // scores tile (fp32)
constexpr uint32_t CB_B = get_compile_time_arg_val(1);   // bias tile (fp32)
constexpr uint32_t CB_CH = get_compile_time_arg_val(2);  // choice tile (fp32) -> writer
}  // namespace

void kernel_main() {
    compute_kernel_hw_startup(CB_S, CB_CH);
    CircularBuffer cbs(CB_S), cbb(CB_B), cbch(CB_CH);

    cbs.wait_front(1);
    cbb.wait_front(1);
    tile_regs_acquire();
    copy_init(CB_S);
    copy_tile(CB_S, 0, 0);
    copy_init(CB_B);
    copy_tile(CB_B, 0, 1);
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);  // dest0 = scores + bias (fp32, exact)
    tile_regs_commit();
    tile_regs_wait();
    cbch.reserve_back(1);
    pack_tile(0, CB_CH);
    cbch.push_back(1);
    tile_regs_release();
    cbs.pop_front(1);
    cbb.pop_front(1);
}
