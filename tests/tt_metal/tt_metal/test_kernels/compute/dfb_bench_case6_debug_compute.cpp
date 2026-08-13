// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute consumer kernel for BenchmarkCaseSixDebug.
//
// Single 1Sx1A DFB (logical id 0): Neo0 drains via wait_front → copy_tile → pop_front.
// Host uses entry_size=2048 (default 32×32 Float16_b unpack).

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    uint32_t neo_id = ckernel::csr_read<ckernel::CSR::NEO_ID>();
    if (neo_id != 0) {
        return;
    }

    unary_op_init_common(0, 0);

    DataflowBuffer dfb(0);
    tile_regs_acquire();
    tile_regs_wait();
    dfb.wait_front(1);
    copy_tile(0, 0, 0);
    dfb.pop_front(1);
    tile_regs_commit();
    tile_regs_release();
}
