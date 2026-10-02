// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/experimental/rmsnorm.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr bool explicit_fidelity = get_compile_time_arg_val(0) != 0;
    constexpr auto fidelity = static_cast<MathFidelity>(get_compile_time_arg_val(1));
    constexpr bool reuse_dest = get_compile_time_arg_val(2) != 0;
    constexpr auto dest_to_srca = EltwiseBinaryReuseDestType::DEST_TO_SRCA;
    CircularBuffer input_a(tt::CBIndex::c_0);
    CircularBuffer input_b(tt::CBIndex::c_1);
    CircularBuffer output(tt::CBIndex::c_16);

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16);
    input_a.wait_front(1);
    input_b.wait_front(1);
    output.reserve_back(1);

    if constexpr (reuse_dest) {
        copy_init(tt::CBIndex::c_0);
        tile_regs_acquire();
        copy_tile(tt::CBIndex::c_0, 0, 0);
        if constexpr (explicit_fidelity) {
            mul_reuse_dest_init_fidelity<dest_to_srca, fidelity>(tt::CBIndex::c_1);
            mul_reuse_dest_tiles_fidelity<dest_to_srca, fidelity>(tt::CBIndex::c_1, 0, 0);
        } else {
            mul_reuse_dest_init<dest_to_srca>(tt::CBIndex::c_1);
            mul_reuse_dest_tiles<dest_to_srca>(tt::CBIndex::c_1, 0, 0);
        }
    } else {
        if constexpr (explicit_fidelity) {
            mul_bcast_scalar_init_fidelity<fidelity>(tt::CBIndex::c_0, tt::CBIndex::c_1);
            tile_regs_acquire();
            mul_tiles_bcast_scalar_fidelity<fidelity>(tt::CBIndex::c_0, tt::CBIndex::c_1, 0, 0, 0);
        } else {
            mul_bcast_scalar_init(tt::CBIndex::c_0, tt::CBIndex::c_1);
            tile_regs_acquire();
            mul_tiles_bcast_scalar(tt::CBIndex::c_0, tt::CBIndex::c_1, 0, 0, 0);
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, tt::CBIndex::c_16);
    tile_regs_release();

    input_a.pop_front(1);
    input_b.pop_front(1);
    output.push_back(1);
}
