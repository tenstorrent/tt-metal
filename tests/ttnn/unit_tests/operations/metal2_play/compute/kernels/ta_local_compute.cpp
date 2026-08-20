// SPDX-License-Identifier: Apache-2.0
// PROBE F2: a COMPUTE kernel reaching resident L1 tensor memory via LocalTensorAccessor.
// The scale factor is a *device-resident* fp32 value the compute kernel needs as an SFPU scalar --
// exactly the case where a compute kernel must read memory that is not a CB.
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/local_tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);

    DataflowBuffer in(dfb::in_tiles);
    DataflowBuffer out(dfb::out_tiles);

    // No NoC anywhere in this type -> compiles on TRISC.
    const LocalTensorAccessor<uint32_t> scale(tensor::scale);
    const uint32_t scale_bits = scale[0];

    compute_kernel_hw_startup(dfb::in_tiles, dfb::out_tiles);
    copy_tile_init(dfb::in_tiles);

    for (uint32_t i = 0; i < num_tiles; ++i) {
        in.wait_front(1);
        out.reserve_back(1);

        tile_regs_acquire();
        copy_tile(dfb::in_tiles, 0, 0);
        mul_unary_tile(0, scale_bits);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, dfb::out_tiles, 0);
        tile_regs_release();

        in.pop_front(1);
        out.push_back(1);
    }
}
