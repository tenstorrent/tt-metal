// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/experimental/softmax_k.h"
#include "api/dataflow/circular_buffer.h"

/** Run the public softmax_k API once per input tile.
 * softmax_k reads row 0 of face 0, takes its first element as the row maximum, and writes the softmax over the
 * first k columns back to row 0 (columns >= k become 0.0). The host checks row 0 of each packed tile.
 * Compile arguments: tile count, k.
 */
void kernel_main() {
    constexpr uint32_t tiles = get_compile_time_arg_val(0);
    constexpr int k = get_compile_time_arg_val(1);
    constexpr uint32_t input = tt::CBIndex::c_0;
    constexpr uint32_t output = tt::CBIndex::c_16;
    CircularBuffer in(input);
    CircularBuffer out(output);
    compute_kernel_hw_startup(input, output);
    softmax_k_init(input);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        in.wait_front(1);
        out.reserve_back(1);
        tile_regs_acquire();
        softmax_k<k>(input);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, output);
        tile_regs_release();
        in.pop_front(1);
        out.push_back(1);
    }
}
