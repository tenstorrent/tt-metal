// SPDX-License-Identifier: Apache-2.0
// PROBE I2: the RAW compute matmul API driven from dfb:: tokens (the token's implicit
// operator uint32_t() in plain function-argument position, alongside pack/unpack state config).
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    DataflowBuffer in0(dfb::in0), in1(dfb::in1), out(dfb::out_tiles);

    compute_kernel_hw_startup(dfb::in0, dfb::in1, dfb::out_tiles);
    matmul_block_init(dfb::in0, dfb::in1, 0, 1, 1, 1);

    in0.wait_front(1);
    in1.wait_front(1);
    out.reserve_back(1);

    tile_regs_acquire();
    matmul_block(dfb::in0, dfb::in1, 0, 0, 0, false, 1, 1, 1);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(0, dfb::out_tiles, 0);
    tile_regs_release();

    in0.pop_front(1);
    in1.pop_front(1);
    out.push_back(1);
}
